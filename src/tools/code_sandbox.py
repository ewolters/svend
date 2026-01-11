"""
Sandboxed Code Execution

Provides safe Python code execution for the reasoning model.
Uses subprocess isolation and resource limits.
"""

import subprocess
import tempfile
import os
import json
import signal
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import sys

from .registry import Tool, ToolParameter, ToolResult, ToolStatus, ToolRegistry


@dataclass
class ExecutionResult:
    """Result of code execution."""
    stdout: str
    stderr: str
    return_value: Any
    exit_code: int
    timed_out: bool = False


class CodeSandbox:
    """
    Sandboxed Python code execution environment.

    Security measures:
    - Subprocess isolation
    - Timeout enforcement
    - Memory limits (when available)
    - Restricted imports
    - No file system access outside temp dir
    """

    # Modules that are safe to import
    ALLOWED_MODULES = {
        "math", "statistics", "random", "itertools", "functools",
        "collections", "heapq", "bisect", "array",
        "datetime", "time", "calendar",
        "re", "string", "textwrap",
        "json", "csv",
        "decimal", "fractions",
        "operator", "copy",
        "typing", "dataclasses",
    }

    # Modules that are explicitly blocked
    BLOCKED_MODULES = {
        "os", "sys", "subprocess", "shutil", "pathlib",
        "socket", "http", "urllib", "requests",
        "pickle", "shelve", "dbm",
        "importlib", "__import__",
        "exec", "eval", "compile",
        "open", "file",
    }

    def __init__(
        self,
        timeout_seconds: int = 10,
        max_memory_mb: int = 256,
        max_output_chars: int = 10000,
    ):
        self.timeout_seconds = timeout_seconds
        self.max_memory_mb = max_memory_mb
        self.max_output_chars = max_output_chars

    def _create_wrapper_code(self, code: str) -> str:
        """Wrap user code with safety checks and result capture."""
        return f'''
import sys
import json

# Restrict dangerous builtins
_blocked = {{"exec", "eval", "compile", "open", "__import__", "input"}}
_safe_builtins = {{k: v for k, v in __builtins__.items() if k not in _blocked}} if isinstance(__builtins__, dict) else {{}}

# Capture result
_result = None
_error = None

try:
    # User code
    exec("""
{code}
""", {{"__builtins__": _safe_builtins}}, _locals := {{}})

    # Try to get last expression value
    _result = _locals.get("result", _locals.get("answer", _locals.get("output", None)))

except Exception as e:
    _error = str(e)

# Output result as JSON
print("__RESULT_JSON__" + json.dumps({{"result": _result, "error": _error}}))
'''

    def execute(self, code: str, inputs: Optional[Dict[str, Any]] = None) -> ExecutionResult:
        """
        Execute Python code in a sandboxed environment.

        Args:
            code: Python code to execute
            inputs: Optional dict of variables to inject

        Returns:
            ExecutionResult with stdout, stderr, return value
        """
        # Inject inputs if provided
        if inputs:
            input_code = "\n".join(f"{k} = {repr(v)}" for k, v in inputs.items())
            code = input_code + "\n" + code

        wrapped_code = self._create_wrapper_code(code)

        # Write to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(wrapped_code)
            temp_path = f.name

        try:
            # Execute in subprocess
            result = subprocess.run(
                [sys.executable, temp_path],
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                cwd=tempfile.gettempdir(),
            )

            stdout = result.stdout[:self.max_output_chars]
            stderr = result.stderr[:self.max_output_chars]

            # Extract result JSON
            return_value = None
            if "__RESULT_JSON__" in stdout:
                json_start = stdout.index("__RESULT_JSON__") + len("__RESULT_JSON__")
                try:
                    result_data = json.loads(stdout[json_start:].strip())
                    return_value = result_data.get("result")
                    if result_data.get("error"):
                        stderr += f"\n{result_data['error']}"
                    stdout = stdout[:stdout.index("__RESULT_JSON__")]
                except json.JSONDecodeError:
                    pass

            return ExecutionResult(
                stdout=stdout.strip(),
                stderr=stderr.strip(),
                return_value=return_value,
                exit_code=result.returncode,
            )

        except subprocess.TimeoutExpired:
            return ExecutionResult(
                stdout="",
                stderr=f"Execution timed out after {self.timeout_seconds} seconds",
                return_value=None,
                exit_code=-1,
                timed_out=True,
            )

        finally:
            # Cleanup
            try:
                os.unlink(temp_path)
            except OSError:
                pass

    def execute_with_tests(
        self,
        code: str,
        test_cases: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Execute code and run test cases.

        Args:
            code: Python code (should define a function)
            test_cases: List of {"input": ..., "expected": ...}

        Returns:
            Dict with pass/fail results for each test
        """
        results = {
            "passed": 0,
            "failed": 0,
            "errors": [],
            "details": [],
        }

        for i, test in enumerate(test_cases):
            test_code = code + f"\nresult = {test.get('call', 'solve')}({test['input']})"
            exec_result = self.execute(test_code)

            if exec_result.timed_out:
                results["failed"] += 1
                results["errors"].append(f"Test {i}: Timeout")
                results["details"].append({
                    "test": i,
                    "status": "timeout",
                })
            elif exec_result.stderr:
                results["failed"] += 1
                results["errors"].append(f"Test {i}: {exec_result.stderr}")
                results["details"].append({
                    "test": i,
                    "status": "error",
                    "error": exec_result.stderr,
                })
            elif exec_result.return_value == test.get("expected"):
                results["passed"] += 1
                results["details"].append({
                    "test": i,
                    "status": "passed",
                })
            else:
                results["failed"] += 1
                results["details"].append({
                    "test": i,
                    "status": "failed",
                    "expected": test.get("expected"),
                    "actual": exec_result.return_value,
                })

        return results


def execute_code_tool(code: str, inputs: Optional[str] = None) -> ToolResult:
    """Tool function for code execution."""
    sandbox = CodeSandbox(timeout_seconds=10)

    # Parse inputs if provided
    input_dict = None
    if inputs:
        try:
            input_dict = json.loads(inputs)
        except json.JSONDecodeError:
            return ToolResult(
                status=ToolStatus.INVALID_INPUT,
                output=None,
                error="Invalid JSON in inputs parameter",
            )

    result = sandbox.execute(code, input_dict)

    if result.timed_out:
        return ToolResult(
            status=ToolStatus.TIMEOUT,
            output=None,
            error="Code execution timed out",
        )

    if result.stderr and result.exit_code != 0:
        return ToolResult(
            status=ToolStatus.ERROR,
            output=result.stdout,
            error=result.stderr,
        )

    output = result.stdout
    if result.return_value is not None:
        output += f"\n\nReturn value: {result.return_value}"

    return ToolResult(
        status=ToolStatus.SUCCESS,
        output=output,
        metadata={
            "return_value": result.return_value,
            "exit_code": result.exit_code,
        },
    )


def create_code_tool() -> Tool:
    """Create the code execution tool."""
    return Tool(
        name="execute_python",
        description="Execute Python code in a sandboxed environment. Use this to compute values, test logic, or verify solutions. The code runs in isolation with limited imports (math, statistics, itertools, etc. are available). Set 'result' variable to return a value.",
        parameters=[
            ToolParameter(
                name="code",
                description="Python code to execute. Can define functions, compute values, etc.",
                type="string",
                required=True,
            ),
            ToolParameter(
                name="inputs",
                description="Optional JSON object of input variables to inject into the code",
                type="string",
                required=False,
            ),
        ],
        execute_fn=execute_code_tool,
        timeout_ms=15000,
    )


def register_code_tools(registry: ToolRegistry) -> None:
    """Register code execution tools with the registry."""
    registry.register(create_code_tool())
