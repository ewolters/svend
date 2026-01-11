"""
Tool Registry - Manages available tools and their schemas.

Each tool has:
- Name and description (for model to understand)
- Input/output schema (for validation)
- Execution function
- Token representation (special tokens for model)
"""

from typing import Optional, Dict, Any, Callable, List, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import re


class ToolStatus(Enum):
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"
    INVALID_INPUT = "invalid_input"


@dataclass
class ToolResult:
    """Result from a tool execution."""
    status: ToolStatus
    output: Any
    error: Optional[str] = None
    execution_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "output": self.output,
            "error": self.error,
            "execution_time_ms": self.execution_time_ms,
            "metadata": self.metadata,
        }

    def to_model_string(self) -> str:
        """Format result for inclusion in model context."""
        if self.status == ToolStatus.SUCCESS:
            if isinstance(self.output, str):
                return self.output
            return json.dumps(self.output, indent=2)
        else:
            return f"Error: {self.error}"


@dataclass
class ToolParameter:
    """Single parameter for a tool."""
    name: str
    description: str
    type: str  # "string", "number", "boolean", "array", "object"
    required: bool = True
    default: Any = None
    enum: Optional[List[Any]] = None


@dataclass
class Tool:
    """
    Definition of an external tool.

    The model learns to call tools by generating special tokens
    followed by structured arguments.
    """
    name: str
    description: str
    parameters: List[ToolParameter]
    execute_fn: Callable[..., ToolResult]

    # Token configuration
    token_id: Optional[int] = None  # Assigned by registry

    # Execution settings
    timeout_ms: int = 30000
    max_retries: int = 1
    requires_confirmation: bool = False  # For dangerous operations

    def get_schema(self) -> Dict[str, Any]:
        """Get JSON schema for tool parameters."""
        properties = {}
        required = []

        for param in self.parameters:
            prop = {
                "type": param.type,
                "description": param.description,
            }
            if param.enum:
                prop["enum"] = param.enum
            if param.default is not None:
                prop["default"] = param.default

            properties[param.name] = prop

            if param.required:
                required.append(param.name)

        return {
            "type": "object",
            "properties": properties,
            "required": required,
        }

    def get_model_description(self) -> str:
        """Format tool for model understanding."""
        params_desc = []
        for p in self.parameters:
            req = "(required)" if p.required else "(optional)"
            params_desc.append(f"  - {p.name} ({p.type}) {req}: {p.description}")

        return f"""Tool: {self.name}
Description: {self.description}
Parameters:
{chr(10).join(params_desc)}"""

    def validate_args(self, args: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """Validate arguments against schema."""
        for param in self.parameters:
            if param.required and param.name not in args:
                return False, f"Missing required parameter: {param.name}"

            if param.name in args:
                value = args[param.name]
                # Basic type checking
                if param.type == "string" and not isinstance(value, str):
                    return False, f"Parameter {param.name} must be string"
                elif param.type == "number" and not isinstance(value, (int, float)):
                    return False, f"Parameter {param.name} must be number"
                elif param.type == "boolean" and not isinstance(value, bool):
                    return False, f"Parameter {param.name} must be boolean"
                elif param.type == "array" and not isinstance(value, list):
                    return False, f"Parameter {param.name} must be array"

                if param.enum and value not in param.enum:
                    return False, f"Parameter {param.name} must be one of {param.enum}"

        return True, None

    def execute(self, **kwargs) -> ToolResult:
        """Execute the tool with given arguments."""
        import time

        # Validate
        valid, error = self.validate_args(kwargs)
        if not valid:
            return ToolResult(
                status=ToolStatus.INVALID_INPUT,
                output=None,
                error=error,
            )

        # Execute with timing
        start = time.perf_counter()
        try:
            result = self.execute_fn(**kwargs)
            elapsed = (time.perf_counter() - start) * 1000

            if isinstance(result, ToolResult):
                result.execution_time_ms = elapsed
                return result
            else:
                return ToolResult(
                    status=ToolStatus.SUCCESS,
                    output=result,
                    execution_time_ms=elapsed,
                )

        except TimeoutError:
            return ToolResult(
                status=ToolStatus.TIMEOUT,
                output=None,
                error=f"Tool execution timed out after {self.timeout_ms}ms",
            )
        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            return ToolResult(
                status=ToolStatus.ERROR,
                output=None,
                error=str(e),
                execution_time_ms=elapsed,
            )


class ToolRegistry:
    """
    Registry of all available tools.

    Manages tool registration, token assignment, and lookup.
    """

    # Special tokens for tool calling
    TOOL_CALL_START = "<|tool_call|>"
    TOOL_CALL_END = "<|/tool_call|>"
    TOOL_RESULT_START = "<|tool_result|>"
    TOOL_RESULT_END = "<|/tool_result|>"
    TOOL_NAME_SEP = "<|tool_name|>"
    TOOL_ARGS_SEP = "<|tool_args|>"

    # Special tokens for meta-cognitive behaviors
    CLARIFICATION_START = "<|clarification|>"
    CLARIFICATION_END = "<|/clarification|>"
    CANNOT_SOLVE_START = "<|cannot_solve|>"
    CANNOT_SOLVE_END = "<|/cannot_solve|>"

    def __init__(self):
        self.tools: Dict[str, Tool] = {}
        self._next_token_id = 0

    def register(self, tool: Tool) -> None:
        """Register a tool."""
        if tool.name in self.tools:
            raise ValueError(f"Tool already registered: {tool.name}")

        tool.token_id = self._next_token_id
        self._next_token_id += 1
        self.tools[tool.name] = tool

    def get(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self.tools.get(name)

    def list_tools(self) -> List[str]:
        """List all registered tool names."""
        return list(self.tools.keys())

    def get_all_descriptions(self) -> str:
        """Get descriptions of all tools for model context."""
        descriptions = []
        for tool in self.tools.values():
            descriptions.append(tool.get_model_description())
        return "\n\n".join(descriptions)

    def get_special_tokens(self) -> List[str]:
        """Get all special tokens for tool calling and meta-cognitive behaviors."""
        tokens = [
            # Tool calling tokens
            self.TOOL_CALL_START,
            self.TOOL_CALL_END,
            self.TOOL_RESULT_START,
            self.TOOL_RESULT_END,
            self.TOOL_NAME_SEP,
            self.TOOL_ARGS_SEP,
            # Meta-cognitive tokens
            self.CLARIFICATION_START,
            self.CLARIFICATION_END,
            self.CANNOT_SOLVE_START,
            self.CANNOT_SOLVE_END,
        ]
        # Add per-tool tokens
        for tool in self.tools.values():
            tokens.append(f"<|tool:{tool.name}|>")
        return tokens

    def format_tool_call(self, tool_name: str, args: Dict[str, Any]) -> str:
        """Format a tool call for model output."""
        args_json = json.dumps(args)
        return f"{self.TOOL_CALL_START}{self.TOOL_NAME_SEP}{tool_name}{self.TOOL_ARGS_SEP}{args_json}{self.TOOL_CALL_END}"

    def format_tool_result(self, result: ToolResult) -> str:
        """Format a tool result for model input."""
        return f"{self.TOOL_RESULT_START}{result.to_model_string()}{self.TOOL_RESULT_END}"

    def parse_tool_call(self, text: str) -> Optional[tuple[str, Dict[str, Any]]]:
        """Parse a tool call from model output."""
        pattern = rf"{re.escape(self.TOOL_CALL_START)}{re.escape(self.TOOL_NAME_SEP)}(\w+){re.escape(self.TOOL_ARGS_SEP)}(.+?){re.escape(self.TOOL_CALL_END)}"

        match = re.search(pattern, text, re.DOTALL)
        if not match:
            return None

        tool_name = match.group(1)
        try:
            args = json.loads(match.group(2))
        except json.JSONDecodeError:
            return None

        return tool_name, args

    def parse_all_tool_calls(self, text: str) -> List[tuple[str, Dict[str, Any]]]:
        """Parse all tool calls from model output."""
        calls = []
        pattern = rf"{re.escape(self.TOOL_CALL_START)}{re.escape(self.TOOL_NAME_SEP)}(\w+){re.escape(self.TOOL_ARGS_SEP)}(.+?){re.escape(self.TOOL_CALL_END)}"

        for match in re.finditer(pattern, text, re.DOTALL):
            tool_name = match.group(1)
            try:
                args = json.loads(match.group(2))
                calls.append((tool_name, args))
            except json.JSONDecodeError:
                continue

        return calls

    def execute(self, tool_name: str, args: Dict[str, Any]) -> ToolResult:
        """Execute a tool by name."""
        tool = self.get(tool_name)
        if tool is None:
            return ToolResult(
                status=ToolStatus.ERROR,
                output=None,
                error=f"Unknown tool: {tool_name}",
            )
        return tool.execute(**args)


def create_default_registry() -> ToolRegistry:
    """Create a registry with default tools."""
    registry = ToolRegistry()

    # These will be populated by the individual tool modules
    # when they're imported

    return registry
