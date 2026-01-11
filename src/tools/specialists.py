"""
Specialist Tool System

Extensible architecture for domain-specific tools.

Design principles:
1. Base interface for all tools
2. Easy plugin system for new domains
3. Safety checks integrated
4. Consistent error handling
5. Metrics and logging

Adding a new specialist:
1. Create a class inheriting from SpecialistTool
2. Implement required methods (execute, validate, etc.)
3. Register with the ToolOrchestrator

Usage:
    from src.tools.specialists import ToolOrchestrator, ChemistryTool

    orchestrator = ToolOrchestrator()
    orchestrator.register(ChemistryTool())

    result = orchestrator.execute("chemistry", "molecular_weight", {"formula": "H2O"})
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Type, Callable
from enum import Enum
import time
import json
from pathlib import Path


class ToolCategory(Enum):
    """Categories of specialist tools."""
    MATH = "math"
    CODE = "code"
    LOGIC = "logic"
    CHEMISTRY = "chemistry"
    PHYSICS = "physics"
    DATA = "data"
    SEARCH = "search"
    GENERAL = "general"


@dataclass
class ToolOperation:
    """Definition of a single tool operation."""

    name: str
    description: str
    parameters: Dict[str, Dict[str, Any]]  # name -> {type, description, required}
    returns: str  # Description of return value
    examples: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "returns": self.returns,
            "examples": self.examples,
        }


@dataclass
class ToolResult:
    """Result from a tool operation."""

    success: bool
    data: Any = None
    error: Optional[str] = None
    execution_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "execution_time_ms": self.execution_time_ms,
            "metadata": self.metadata,
        }

    def to_model_string(self) -> str:
        """Format for inclusion in model context."""
        if self.success:
            if isinstance(self.data, str):
                return self.data
            return json.dumps(self.data, indent=2, default=str)
        else:
            return f"Error: {self.error}"


class SpecialistTool(ABC):
    """
    Base class for specialist tools.

    Subclasses implement domain-specific operations.
    """

    name: str = "base"
    category: ToolCategory = ToolCategory.GENERAL
    description: str = "Base specialist tool"
    version: str = "1.0.0"

    def __init__(self):
        self._operations: Dict[str, ToolOperation] = {}
        self._register_operations()

        # Metrics
        self.call_count = 0
        self.error_count = 0
        self.total_time_ms = 0.0

    @abstractmethod
    def _register_operations(self):
        """Register available operations. Called during __init__."""
        pass

    def register_operation(self, operation: ToolOperation):
        """Register an operation."""
        self._operations[operation.name] = operation

    def get_operations(self) -> List[ToolOperation]:
        """Get all available operations."""
        return list(self._operations.values())

    def get_operation(self, name: str) -> Optional[ToolOperation]:
        """Get operation by name."""
        return self._operations.get(name)

    def execute(self, operation: str, **kwargs) -> ToolResult:
        """
        Execute an operation.

        Args:
            operation: Name of operation to execute
            **kwargs: Operation arguments

        Returns:
            ToolResult with success/failure and data
        """
        start_time = time.perf_counter()
        self.call_count += 1

        # Check operation exists
        op = self.get_operation(operation)
        if op is None:
            self.error_count += 1
            return ToolResult(
                success=False,
                error=f"Unknown operation: {operation}. Available: {list(self._operations.keys())}",
            )

        # Validate arguments
        validation_error = self._validate_args(op, kwargs)
        if validation_error:
            self.error_count += 1
            return ToolResult(
                success=False,
                error=validation_error,
            )

        # Execute
        try:
            result = self._execute_operation(operation, **kwargs)
            elapsed = (time.perf_counter() - start_time) * 1000
            self.total_time_ms += elapsed

            if isinstance(result, ToolResult):
                result.execution_time_ms = elapsed
                return result
            else:
                return ToolResult(
                    success=True,
                    data=result,
                    execution_time_ms=elapsed,
                )

        except Exception as e:
            self.error_count += 1
            elapsed = (time.perf_counter() - start_time) * 1000
            self.total_time_ms += elapsed

            return ToolResult(
                success=False,
                error=f"{type(e).__name__}: {str(e)}",
                execution_time_ms=elapsed,
            )

    @abstractmethod
    def _execute_operation(self, operation: str, **kwargs) -> Any:
        """Execute a specific operation. Implemented by subclasses."""
        pass

    def _validate_args(self, op: ToolOperation, kwargs: Dict[str, Any]) -> Optional[str]:
        """Validate arguments against operation schema."""
        for param_name, param_spec in op.parameters.items():
            if param_spec.get("required", True) and param_name not in kwargs:
                return f"Missing required parameter: {param_name}"

            if param_name in kwargs:
                expected_type = param_spec.get("type", "any")
                value = kwargs[param_name]

                if expected_type == "string" and not isinstance(value, str):
                    return f"Parameter {param_name} must be a string"
                elif expected_type == "number" and not isinstance(value, (int, float)):
                    return f"Parameter {param_name} must be a number"
                elif expected_type == "boolean" and not isinstance(value, bool):
                    return f"Parameter {param_name} must be a boolean"
                elif expected_type == "array" and not isinstance(value, list):
                    return f"Parameter {param_name} must be an array"

        return None

    def get_schema(self) -> Dict[str, Any]:
        """Get full schema for the tool."""
        return {
            "name": self.name,
            "category": self.category.value,
            "description": self.description,
            "version": self.version,
            "operations": {
                name: op.to_dict()
                for name, op in self._operations.items()
            },
        }

    def get_model_description(self) -> str:
        """Get description formatted for model context."""
        lines = [
            f"Tool: {self.name}",
            f"Category: {self.category.value}",
            f"Description: {self.description}",
            "",
            "Operations:",
        ]

        for op in self._operations.values():
            lines.append(f"  {op.name}: {op.description}")
            for param_name, param_spec in op.parameters.items():
                req = "(required)" if param_spec.get("required", True) else "(optional)"
                lines.append(f"    - {param_name} ({param_spec.get('type', 'any')}) {req}: {param_spec.get('description', '')}")

        return "\n".join(lines)

    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics."""
        return {
            "name": self.name,
            "call_count": self.call_count,
            "error_count": self.error_count,
            "error_rate": self.error_count / max(self.call_count, 1),
            "avg_time_ms": self.total_time_ms / max(self.call_count, 1),
        }


class ToolOrchestrator:
    """
    Orchestrates multiple specialist tools.

    Handles:
    - Tool registration
    - Routing requests to appropriate tools
    - Safety checks before execution
    - Logging and metrics
    """

    def __init__(self):
        self.tools: Dict[str, SpecialistTool] = {}
        self.execution_log: List[Dict[str, Any]] = []

    def register(self, tool: SpecialistTool):
        """Register a specialist tool."""
        self.tools[tool.name] = tool
        print(f"Registered tool: {tool.name} ({len(tool.get_operations())} operations)")

    def unregister(self, name: str):
        """Unregister a tool."""
        if name in self.tools:
            del self.tools[name]

    def get_tool(self, name: str) -> Optional[SpecialistTool]:
        """Get a tool by name."""
        return self.tools.get(name)

    def list_tools(self) -> List[str]:
        """List all registered tool names."""
        return list(self.tools.keys())

    def execute(
        self,
        tool_name: str,
        operation: str,
        args: Optional[Dict[str, Any]] = None,
        safety_check: bool = True,
    ) -> ToolResult:
        """
        Execute a tool operation.

        Args:
            tool_name: Name of the tool
            operation: Operation to execute
            args: Arguments for the operation
            safety_check: Whether to run safety checks

        Returns:
            ToolResult with outcome
        """
        args = args or {}

        # Get tool
        tool = self.get_tool(tool_name)
        if tool is None:
            return ToolResult(
                success=False,
                error=f"Unknown tool: {tool_name}. Available: {self.list_tools()}",
            )

        # Safety check (optional)
        if safety_check:
            safety_result = self._safety_check(tool_name, operation, args)
            if not safety_result["safe"]:
                return ToolResult(
                    success=False,
                    error=f"Safety check failed: {safety_result['reason']}",
                )

        # Execute
        result = tool.execute(operation, **args)

        # Log
        self._log_execution(tool_name, operation, args, result)

        return result

    def _safety_check(
        self,
        tool_name: str,
        operation: str,
        args: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run safety checks on tool execution."""
        # Basic safety checks
        # In production, integrate with SafetyGate

        # Check for obvious dangerous patterns
        args_str = json.dumps(args, default=str).lower()

        dangerous_patterns = [
            "rm -rf", "format c:", "drop table",
            "exec(", "eval(", "__import__",
        ]

        for pattern in dangerous_patterns:
            if pattern in args_str:
                return {
                    "safe": False,
                    "reason": f"Dangerous pattern detected: {pattern}",
                }

        return {"safe": True, "reason": None}

    def _log_execution(
        self,
        tool_name: str,
        operation: str,
        args: Dict[str, Any],
        result: ToolResult,
    ):
        """Log an execution for auditing."""
        self.execution_log.append({
            "timestamp": time.time(),
            "tool": tool_name,
            "operation": operation,
            "args": args,
            "success": result.success,
            "execution_time_ms": result.execution_time_ms,
            "error": result.error,
        })

        # Keep log bounded
        if len(self.execution_log) > 10000:
            self.execution_log = self.execution_log[-5000:]

    def get_all_schemas(self) -> Dict[str, Any]:
        """Get schemas for all registered tools."""
        return {
            name: tool.get_schema()
            for name, tool in self.tools.items()
        }

    def get_model_context(self) -> str:
        """Get tool descriptions for model context."""
        descriptions = []
        for tool in self.tools.values():
            descriptions.append(tool.get_model_description())
        return "\n\n".join(descriptions)

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics for all tools."""
        return {
            name: tool.get_stats()
            for name, tool in self.tools.items()
        }

    def save_log(self, path: str):
        """Save execution log to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w') as f:
            json.dump(self.execution_log, f, indent=2)


def create_orchestrator(include_defaults: bool = True) -> ToolOrchestrator:
    """
    Create a tool orchestrator with optional default tools.

    Args:
        include_defaults: Whether to register default tools

    Returns:
        Configured ToolOrchestrator
    """
    orchestrator = ToolOrchestrator()

    if include_defaults:
        # Import and register default tools
        from .math_engine import create_symbolic_math_tool, create_logic_solver_tool
        from .code_sandbox import create_code_tool

        # Wrap legacy tools in specialist interface
        # (For new tools, use SpecialistTool directly)

        # We'll add the new chemistry/physics tools as proper specialists
        pass

    return orchestrator
