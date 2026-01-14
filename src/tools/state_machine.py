"""
State Machine Tool - Discrete State Transitions and Simulations

Kills hallucinations in:
- Multi-step simulations ("what happens after X, then Y, then Z")
- Inventory/resource flows
- Queues, cooldowns, state-dependent logic
- Any "over time" reasoning

The model should NEVER improvise state transitions - it should simulate.
"""

from typing import Optional, Dict, Any, List, Tuple, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
from copy import deepcopy

from .registry import Tool, ToolParameter, ToolResult, ToolStatus, ToolRegistry


class SimulationStatus(Enum):
    """Status of a simulation run."""
    COMPLETE = "complete"
    HALTED = "halted"  # Reached terminal state
    MAX_STEPS = "max_steps"  # Hit step limit
    INVALID_ACTION = "invalid_action"
    CYCLE_DETECTED = "cycle_detected"
    DEADLOCK = "deadlock"  # No valid actions but not terminal


@dataclass
class State:
    """A discrete state with named properties."""
    name: str
    properties: Dict[str, Any] = field(default_factory=dict)

    def __hash__(self):
        # Hash based on name and frozen properties for cycle detection
        return hash((self.name, tuple(sorted(self.properties.items()))))

    def __eq__(self, other):
        if not isinstance(other, State):
            return False
        return self.name == other.name and self.properties == other.properties

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "properties": self.properties}


@dataclass
class Transition:
    """A state transition with optional guard and effects."""
    name: str
    from_state: str  # State name pattern ("*" for any)
    to_state: str
    guard: Optional[Dict[str, Any]] = None  # Property conditions
    effects: Optional[Dict[str, Any]] = None  # Property changes

    def matches(self, state: State) -> bool:
        """Check if transition can fire from given state."""
        # Check state name
        if self.from_state != "*" and self.from_state != state.name:
            return False

        # Check guard conditions
        if self.guard:
            for prop, condition in self.guard.items():
                value = state.properties.get(prop)
                if isinstance(condition, dict):
                    # Comparison operators
                    if "eq" in condition and value != condition["eq"]:
                        return False
                    if "ne" in condition and value == condition["ne"]:
                        return False
                    if "gt" in condition and not (value is not None and value > condition["gt"]):
                        return False
                    if "gte" in condition and not (value is not None and value >= condition["gte"]):
                        return False
                    if "lt" in condition and not (value is not None and value < condition["lt"]):
                        return False
                    if "lte" in condition and not (value is not None and value <= condition["lte"]):
                        return False
                    if "in" in condition and value not in condition["in"]:
                        return False
                else:
                    # Direct equality
                    if value != condition:
                        return False

        return True

    def apply(self, state: State) -> State:
        """Apply transition to produce new state."""
        new_props = deepcopy(state.properties)

        if self.effects:
            for prop, effect in self.effects.items():
                if isinstance(effect, dict):
                    # Operations
                    current = new_props.get(prop, 0)
                    if "add" in effect:
                        new_props[prop] = current + effect["add"]
                    elif "sub" in effect:
                        new_props[prop] = current - effect["sub"]
                    elif "mul" in effect:
                        new_props[prop] = current * effect["mul"]
                    elif "set" in effect:
                        new_props[prop] = effect["set"]
                    elif "append" in effect:
                        if prop not in new_props:
                            new_props[prop] = []
                        new_props[prop] = new_props[prop] + [effect["append"]]
                    elif "pop" in effect:
                        if prop in new_props and new_props[prop]:
                            new_props[prop] = new_props[prop][:-1]
                else:
                    # Direct assignment
                    new_props[prop] = effect

        return State(name=self.to_state, properties=new_props)


class StateMachine:
    """
    Discrete state machine simulator.

    Supports:
    - Named states with properties
    - Guarded transitions
    - Property effects (add, subtract, set, etc.)
    - Cycle detection
    - Step-by-step or run-to-completion
    """

    def __init__(
        self,
        initial_state: State,
        transitions: List[Transition],
        terminal_states: Optional[Set[str]] = None,
        max_steps: int = 1000
    ):
        self.initial_state = initial_state
        self.transitions = transitions
        self.terminal_states = terminal_states or set()
        self.max_steps = max_steps

        # Simulation state
        self.current_state = deepcopy(initial_state)
        self.history: List[Tuple[State, str, State]] = []  # (from, action, to)
        self.visited: Set[int] = {hash(initial_state)}

    def reset(self):
        """Reset to initial state."""
        self.current_state = deepcopy(self.initial_state)
        self.history = []
        self.visited = {hash(self.initial_state)}

    def available_actions(self) -> List[str]:
        """Get actions available from current state."""
        return [t.name for t in self.transitions if t.matches(self.current_state)]

    def is_terminal(self) -> bool:
        """Check if current state is terminal."""
        return self.current_state.name in self.terminal_states

    def step(self, action: str) -> Tuple[State, SimulationStatus]:
        """Execute one action."""
        if self.is_terminal():
            return self.current_state, SimulationStatus.HALTED

        # Find matching transition
        for t in self.transitions:
            if t.name == action and t.matches(self.current_state):
                old_state = self.current_state
                new_state = t.apply(old_state)

                # Check for cycle
                state_hash = hash(new_state)
                if state_hash in self.visited:
                    self.history.append((old_state, action, new_state))
                    self.current_state = new_state
                    return new_state, SimulationStatus.CYCLE_DETECTED

                self.visited.add(state_hash)
                self.history.append((old_state, action, new_state))
                self.current_state = new_state

                if self.is_terminal():
                    return new_state, SimulationStatus.HALTED

                return new_state, SimulationStatus.COMPLETE

        return self.current_state, SimulationStatus.INVALID_ACTION

    def run(self, actions: List[str]) -> Tuple[State, SimulationStatus, int]:
        """Run a sequence of actions."""
        for i, action in enumerate(actions):
            if len(self.history) >= self.max_steps:
                return self.current_state, SimulationStatus.MAX_STEPS, i

            state, status = self.step(action)

            if status != SimulationStatus.COMPLETE:
                return state, status, i

        return self.current_state, SimulationStatus.COMPLETE, len(actions)

    def run_until_terminal(self, action_selector: Optional[Callable] = None) -> Tuple[State, SimulationStatus]:
        """
        Run until terminal state or max steps.

        action_selector: function(available_actions) -> action
                        Defaults to first available action.
        """
        if action_selector is None:
            action_selector = lambda actions: actions[0] if actions else None

        while len(self.history) < self.max_steps:
            if self.is_terminal():
                return self.current_state, SimulationStatus.HALTED

            available = self.available_actions()
            if not available:
                return self.current_state, SimulationStatus.DEADLOCK

            action = action_selector(available)
            if action is None:
                return self.current_state, SimulationStatus.DEADLOCK

            state, status = self.step(action)

            if status == SimulationStatus.CYCLE_DETECTED:
                return state, status
            if status == SimulationStatus.INVALID_ACTION:
                return state, status

        return self.current_state, SimulationStatus.MAX_STEPS

    def get_trace(self) -> List[Dict[str, Any]]:
        """Get execution trace."""
        trace = [{"step": 0, "state": self.initial_state.to_dict(), "action": None}]
        for i, (from_state, action, to_state) in enumerate(self.history):
            trace.append({
                "step": i + 1,
                "state": to_state.to_dict(),
                "action": action
            })
        return trace


class ResourceSimulator:
    """
    Simplified resource flow simulator.

    For inventory, queues, production chains.
    """

    def __init__(self, initial_resources: Dict[str, float]):
        self.resources = dict(initial_resources)
        self.history: List[Dict[str, Any]] = [{"step": 0, "resources": dict(self.resources), "action": "init"}]

    def add(self, resource: str, amount: float) -> bool:
        """Add to a resource."""
        self.resources[resource] = self.resources.get(resource, 0) + amount
        self.history.append({
            "step": len(self.history),
            "resources": dict(self.resources),
            "action": f"add {amount} {resource}"
        })
        return True

    def remove(self, resource: str, amount: float) -> bool:
        """Remove from a resource (fails if insufficient)."""
        current = self.resources.get(resource, 0)
        if current < amount:
            return False
        self.resources[resource] = current - amount
        self.history.append({
            "step": len(self.history),
            "resources": dict(self.resources),
            "action": f"remove {amount} {resource}"
        })
        return True

    def transfer(self, from_res: str, to_res: str, amount: float, ratio: float = 1.0) -> bool:
        """Transfer between resources (with optional conversion ratio)."""
        if not self.remove(from_res, amount):
            return False
        self.add(to_res, amount * ratio)
        # Combine last two history entries
        self.history[-2:] = [{
            "step": len(self.history) - 1,
            "resources": dict(self.resources),
            "action": f"transfer {amount} {from_res} -> {amount * ratio} {to_res}"
        }]
        return True

    def can_afford(self, costs: Dict[str, float]) -> bool:
        """Check if we have enough of all resources."""
        return all(self.resources.get(res, 0) >= amt for res, amt in costs.items())

    def apply_recipe(self, inputs: Dict[str, float], outputs: Dict[str, float]) -> bool:
        """Apply a production recipe (consume inputs, produce outputs)."""
        if not self.can_afford(inputs):
            return False

        for res, amt in inputs.items():
            self.resources[res] = self.resources.get(res, 0) - amt

        for res, amt in outputs.items():
            self.resources[res] = self.resources.get(res, 0) + amt

        self.history.append({
            "step": len(self.history),
            "resources": dict(self.resources),
            "action": f"recipe: {inputs} -> {outputs}"
        })
        return True


# Tool implementation

def state_machine_tool(
    operation: str,
    initial_state: Optional[Dict[str, Any]] = None,
    transitions: Optional[List[Dict[str, Any]]] = None,
    actions: Optional[List[str]] = None,
    terminal_states: Optional[List[str]] = None,
    max_steps: int = 100,
    # For resource simulator
    initial_resources: Optional[Dict[str, float]] = None,
    resource_actions: Optional[List[Dict[str, Any]]] = None,
) -> ToolResult:
    """Execute state machine simulation."""
    try:
        if operation == "simulate":
            # Full state machine simulation
            if not initial_state or not transitions:
                return ToolResult(
                    status=ToolStatus.INVALID_INPUT,
                    output=None,
                    error="Need initial_state and transitions"
                )

            # Build state
            state = State(
                name=initial_state.get("name", "start"),
                properties=initial_state.get("properties", {})
            )

            # Build transitions
            trans_list = []
            for t in transitions:
                trans_list.append(Transition(
                    name=t["name"],
                    from_state=t.get("from", "*"),
                    to_state=t.get("to", t.get("from", "*")),
                    guard=t.get("guard"),
                    effects=t.get("effects")
                ))

            # Create machine
            machine = StateMachine(
                initial_state=state,
                transitions=trans_list,
                terminal_states=set(terminal_states or []),
                max_steps=max_steps
            )

            # Run actions if provided
            if actions:
                final_state, status, steps_completed = machine.run(actions)
            else:
                # Run until terminal
                final_state, status = machine.run_until_terminal()
                steps_completed = len(machine.history)

            trace = machine.get_trace()

            output_lines = [
                f"Simulation: {status.value}",
                f"Steps: {steps_completed}",
                f"Final state: {final_state.name}",
                f"Properties: {final_state.properties}"
            ]

            if len(trace) <= 20:
                output_lines.append(f"Trace: {trace}")
            else:
                output_lines.append(f"Trace: [{trace[0]}, ..., {trace[-1]}] ({len(trace)} steps)")

            return ToolResult(
                status=ToolStatus.SUCCESS,
                output="\n".join(output_lines),
                metadata={
                    "status": status.value,
                    "steps": steps_completed,
                    "final_state": final_state.to_dict(),
                    "trace": trace if len(trace) <= 100 else trace[:50] + trace[-50:],
                    "available_actions": machine.available_actions()
                }
            )

        elif operation == "available_actions":
            # Query what actions are possible from a state
            if not initial_state or not transitions:
                return ToolResult(
                    status=ToolStatus.INVALID_INPUT,
                    output=None,
                    error="Need initial_state and transitions"
                )

            state = State(
                name=initial_state.get("name", "start"),
                properties=initial_state.get("properties", {})
            )

            trans_list = []
            for t in transitions:
                trans_list.append(Transition(
                    name=t["name"],
                    from_state=t.get("from", "*"),
                    to_state=t.get("to", t.get("from", "*")),
                    guard=t.get("guard"),
                    effects=t.get("effects")
                ))

            available = [t.name for t in trans_list if t.matches(state)]

            return ToolResult(
                status=ToolStatus.SUCCESS,
                output=f"Available actions from {state.name}: {available}",
                metadata={"available": available, "state": state.to_dict()}
            )

        elif operation == "resource_sim":
            # Resource flow simulation
            if not initial_resources or not resource_actions:
                return ToolResult(
                    status=ToolStatus.INVALID_INPUT,
                    output=None,
                    error="Need initial_resources and resource_actions"
                )

            sim = ResourceSimulator(initial_resources)
            results = []

            for action in resource_actions:
                action_type = action.get("type")
                success = False

                if action_type == "add":
                    success = sim.add(action["resource"], action["amount"])
                elif action_type == "remove":
                    success = sim.remove(action["resource"], action["amount"])
                elif action_type == "transfer":
                    success = sim.transfer(
                        action["from"],
                        action["to"],
                        action["amount"],
                        action.get("ratio", 1.0)
                    )
                elif action_type == "recipe":
                    success = sim.apply_recipe(action["inputs"], action["outputs"])
                elif action_type == "check":
                    success = sim.can_afford(action.get("costs", {}))
                    results.append({"action": action, "success": success, "resources": dict(sim.resources)})
                    continue

                results.append({"action": action, "success": success, "resources": dict(sim.resources)})

                if not success:
                    return ToolResult(
                        status=ToolStatus.SUCCESS,
                        output=f"FAILED at action: {action}\nFinal resources: {sim.resources}",
                        metadata={
                            "success": False,
                            "failed_at": action,
                            "final_resources": sim.resources,
                            "history": sim.history
                        }
                    )

            return ToolResult(
                status=ToolStatus.SUCCESS,
                output=f"All actions succeeded. Final resources: {sim.resources}",
                metadata={
                    "success": True,
                    "final_resources": sim.resources,
                    "history": sim.history,
                    "action_results": results
                }
            )

        elif operation == "what_if":
            # Counterfactual: what happens if we do X from state Y?
            if not initial_state or not transitions or not actions:
                return ToolResult(
                    status=ToolStatus.INVALID_INPUT,
                    output=None,
                    error="Need initial_state, transitions, and actions"
                )

            state = State(
                name=initial_state.get("name", "start"),
                properties=initial_state.get("properties", {})
            )

            trans_list = []
            for t in transitions:
                trans_list.append(Transition(
                    name=t["name"],
                    from_state=t.get("from", "*"),
                    to_state=t.get("to", t.get("from", "*")),
                    guard=t.get("guard"),
                    effects=t.get("effects")
                ))

            machine = StateMachine(
                initial_state=state,
                transitions=trans_list,
                terminal_states=set(terminal_states or []),
                max_steps=max_steps
            )

            # Simulate action sequence
            final_state, status, steps = machine.run(actions)

            return ToolResult(
                status=ToolStatus.SUCCESS,
                output=f"After {actions}: state={final_state.name}, properties={final_state.properties}, status={status.value}",
                metadata={
                    "question": f"What if {actions} from {initial_state}?",
                    "answer": {
                        "final_state": final_state.to_dict(),
                        "status": status.value,
                        "steps_completed": steps
                    },
                    "trace": machine.get_trace()
                }
            )

        else:
            return ToolResult(
                status=ToolStatus.INVALID_INPUT,
                output=None,
                error=f"Unknown operation: {operation}. Valid: simulate, available_actions, resource_sim, what_if"
            )

    except Exception as e:
        return ToolResult(status=ToolStatus.ERROR, output=None, error=str(e))


def create_state_machine_tool() -> Tool:
    """Create state machine simulation tool."""
    return Tool(
        name="state_machine",
        description="Simulate discrete state transitions. Use for multi-step processes, inventory flows, 'what happens next' questions. NEVER improvise state over time - simulate it.",
        parameters=[
            ToolParameter(
                name="operation",
                description="simulate (run actions), available_actions (query), resource_sim (inventory), what_if (counterfactual)",
                type="string",
                required=True,
                enum=["simulate", "available_actions", "resource_sim", "what_if"]
            ),
            ToolParameter(
                name="initial_state",
                description="Starting state: {name: str, properties: {key: value, ...}}",
                type="object",
                required=False,
            ),
            ToolParameter(
                name="transitions",
                description="List of transitions: [{name, from, to, guard, effects}, ...]",
                type="array",
                required=False,
            ),
            ToolParameter(
                name="actions",
                description="Sequence of action names to execute",
                type="array",
                required=False,
            ),
            ToolParameter(
                name="terminal_states",
                description="State names that end simulation",
                type="array",
                required=False,
            ),
            ToolParameter(
                name="max_steps",
                description="Maximum simulation steps (default: 100)",
                type="number",
                required=False,
            ),
            ToolParameter(
                name="initial_resources",
                description="For resource_sim: {resource: amount, ...}",
                type="object",
                required=False,
            ),
            ToolParameter(
                name="resource_actions",
                description="For resource_sim: [{type: add/remove/transfer/recipe, ...}, ...]",
                type="array",
                required=False,
            ),
        ],
        execute_fn=state_machine_tool,
        timeout_ms=30000,
    )


def register_state_machine_tools(registry: ToolRegistry) -> None:
    """Register state machine tools."""
    registry.register(create_state_machine_tool())
