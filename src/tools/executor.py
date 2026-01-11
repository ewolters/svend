"""
Tool Executor - Manages tool execution lifecycle.

Handles:
- Parallel tool execution
- Retry logic
- Result caching
- Execution logging
"""

import asyncio
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

from .registry import ToolRegistry, Tool, ToolResult, ToolStatus


@dataclass
class ExecutionLog:
    """Log entry for a tool execution."""
    tool_name: str
    args: Dict[str, Any]
    result: ToolResult
    timestamp: datetime = field(default_factory=datetime.now)
    attempt: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "args": self.args,
            "result": self.result.to_dict(),
            "timestamp": self.timestamp.isoformat(),
            "attempt": self.attempt,
        }


class ExecutionCache:
    """
    Cache for tool execution results.

    Caches deterministic tool outputs to avoid redundant computation.
    """

    def __init__(self, max_size: int = 1000):
        self.cache: Dict[str, ToolResult] = {}
        self.max_size = max_size
        self.hits = 0
        self.misses = 0

    def _make_key(self, tool_name: str, args: Dict[str, Any]) -> str:
        """Create cache key from tool name and args."""
        args_json = json.dumps(args, sort_keys=True)
        content = f"{tool_name}:{args_json}"
        return hashlib.sha256(content.encode()).hexdigest()

    def get(self, tool_name: str, args: Dict[str, Any]) -> Optional[ToolResult]:
        """Get cached result if available."""
        key = self._make_key(tool_name, args)
        result = self.cache.get(key)
        if result:
            self.hits += 1
        else:
            self.misses += 1
        return result

    def set(self, tool_name: str, args: Dict[str, Any], result: ToolResult) -> None:
        """Cache a result."""
        if len(self.cache) >= self.max_size:
            # Simple eviction: remove oldest entries
            keys = list(self.cache.keys())
            for key in keys[:len(keys) // 4]:
                del self.cache[key]

        key = self._make_key(tool_name, args)
        self.cache[key] = result

    def clear(self) -> None:
        """Clear the cache."""
        self.cache.clear()

    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self.hits + self.misses
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hits / total if total > 0 else 0,
        }


class ToolExecutor:
    """
    Manages tool execution with caching, retries, and logging.

    Features:
    - Synchronous and async execution
    - Parallel execution of multiple tools
    - Result caching for deterministic tools
    - Automatic retries on failure
    - Execution logging for debugging
    """

    def __init__(
        self,
        registry: ToolRegistry,
        use_cache: bool = True,
        max_workers: int = 4,
        default_timeout_ms: int = 30000,
    ):
        self.registry = registry
        self.use_cache = use_cache
        self.max_workers = max_workers
        self.default_timeout_ms = default_timeout_ms

        self.cache = ExecutionCache() if use_cache else None
        self.execution_log: List[ExecutionLog] = []
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    def execute(
        self,
        tool_name: str,
        args: Dict[str, Any],
        use_cache: Optional[bool] = None,
        timeout_ms: Optional[int] = None,
    ) -> ToolResult:
        """
        Execute a single tool.

        Args:
            tool_name: Name of the tool to execute
            args: Arguments for the tool
            use_cache: Override cache setting for this call
            timeout_ms: Override timeout for this call

        Returns:
            ToolResult with execution outcome
        """
        # Check cache
        should_cache = use_cache if use_cache is not None else self.use_cache
        if should_cache and self.cache:
            cached = self.cache.get(tool_name, args)
            if cached:
                return cached

        # Get tool
        tool = self.registry.get(tool_name)
        if tool is None:
            return ToolResult(
                status=ToolStatus.ERROR,
                output=None,
                error=f"Unknown tool: {tool_name}",
            )

        # Execute with timeout
        timeout = (timeout_ms or tool.timeout_ms or self.default_timeout_ms) / 1000

        try:
            future = self.executor.submit(tool.execute, **args)
            result = future.result(timeout=timeout)
        except FuturesTimeoutError:
            result = ToolResult(
                status=ToolStatus.TIMEOUT,
                output=None,
                error=f"Tool execution timed out after {timeout}s",
            )
        except Exception as e:
            result = ToolResult(
                status=ToolStatus.ERROR,
                output=None,
                error=str(e),
            )

        # Log execution
        self.execution_log.append(ExecutionLog(
            tool_name=tool_name,
            args=args,
            result=result,
        ))

        # Cache successful results
        if should_cache and self.cache and result.status == ToolStatus.SUCCESS:
            self.cache.set(tool_name, args, result)

        return result

    def execute_with_retry(
        self,
        tool_name: str,
        args: Dict[str, Any],
        max_retries: int = 3,
        retry_delay_ms: int = 1000,
    ) -> ToolResult:
        """Execute a tool with automatic retries on failure."""
        import time

        tool = self.registry.get(tool_name)
        retries = max_retries if tool is None else min(max_retries, tool.max_retries)

        last_result = None
        for attempt in range(retries + 1):
            result = self.execute(tool_name, args, use_cache=False)

            if result.status == ToolStatus.SUCCESS:
                return result

            last_result = result

            # Log retry
            self.execution_log.append(ExecutionLog(
                tool_name=tool_name,
                args=args,
                result=result,
                attempt=attempt + 1,
            ))

            # Don't retry on invalid input
            if result.status == ToolStatus.INVALID_INPUT:
                break

            # Wait before retry
            if attempt < retries:
                time.sleep(retry_delay_ms / 1000)

        return last_result

    def execute_parallel(
        self,
        calls: List[Tuple[str, Dict[str, Any]]],
        timeout_ms: Optional[int] = None,
    ) -> List[ToolResult]:
        """
        Execute multiple tool calls in parallel.

        Args:
            calls: List of (tool_name, args) tuples
            timeout_ms: Timeout for all calls

        Returns:
            List of ToolResults in same order as input
        """
        timeout = (timeout_ms or self.default_timeout_ms) / 1000

        futures = []
        for tool_name, args in calls:
            tool = self.registry.get(tool_name)
            if tool is None:
                futures.append(None)
            else:
                future = self.executor.submit(tool.execute, **args)
                futures.append((tool_name, args, future))

        results = []
        for item in futures:
            if item is None:
                results.append(ToolResult(
                    status=ToolStatus.ERROR,
                    output=None,
                    error="Unknown tool",
                ))
            else:
                tool_name, args, future = item
                try:
                    result = future.result(timeout=timeout)
                    results.append(result)
                except FuturesTimeoutError:
                    results.append(ToolResult(
                        status=ToolStatus.TIMEOUT,
                        output=None,
                        error="Execution timed out",
                    ))
                except Exception as e:
                    results.append(ToolResult(
                        status=ToolStatus.ERROR,
                        output=None,
                        error=str(e),
                    ))

                # Log
                self.execution_log.append(ExecutionLog(
                    tool_name=tool_name,
                    args=args,
                    result=results[-1],
                ))

        return results

    async def execute_async(
        self,
        tool_name: str,
        args: Dict[str, Any],
    ) -> ToolResult:
        """Execute a tool asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            lambda: self.execute(tool_name, args),
        )

    async def execute_parallel_async(
        self,
        calls: List[Tuple[str, Dict[str, Any]]],
    ) -> List[ToolResult]:
        """Execute multiple tools asynchronously in parallel."""
        tasks = [
            self.execute_async(tool_name, args)
            for tool_name, args in calls
        ]
        return await asyncio.gather(*tasks)

    def get_execution_history(
        self,
        tool_name: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get execution history, optionally filtered by tool."""
        logs = self.execution_log[-limit:]
        if tool_name:
            logs = [log for log in logs if log.tool_name == tool_name]
        return [log.to_dict() for log in logs]

    def clear_history(self) -> None:
        """Clear execution history."""
        self.execution_log.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get executor statistics."""
        stats = {
            "total_executions": len(self.execution_log),
            "tools_available": len(self.registry.list_tools()),
        }

        if self.cache:
            stats["cache"] = self.cache.stats()

        # Count by status
        by_status = {}
        for log in self.execution_log:
            status = log.result.status.value
            by_status[status] = by_status.get(status, 0) + 1
        stats["by_status"] = by_status

        # Count by tool
        by_tool = {}
        for log in self.execution_log:
            by_tool[log.tool_name] = by_tool.get(log.tool_name, 0) + 1
        stats["by_tool"] = by_tool

        return stats

    def shutdown(self) -> None:
        """Shutdown the executor."""
        self.executor.shutdown(wait=True)
