"""
Svend Safety Module

Separate safety classifier for input/output filtering.

Architecture:
- SafetyClassifier: Lightweight transformer-based classifier
- InputFilter: Classifies user inputs as safe/review/refuse
- OutputFilter: Classifies model outputs as safe/edit/block
- RuleEngine: Fast rule-based pre-filtering

Design Principles:
1. Safety is a separate, maintainable component
2. Fail closed - when uncertain, be conservative
3. Speed matters - runs on every request
4. Auditable - all decisions are logged

Usage:
    from src.safety import SafetyGate

    gate = SafetyGate.from_pretrained("checkpoints/safety")

    # Check input
    input_result = gate.check_input("user message")
    if input_result.action == "refuse":
        return refusal_message(input_result.reason)

    # Generate response...

    # Check output
    output_result = gate.check_output(response)
    if output_result.action == "block":
        return safe_fallback()
"""

from .classifier import (
    SafetyClassifier,
    SafetyClassifierConfig,
    SafetyLabel,
)

from .filters import (
    InputFilter,
    OutputFilter,
    FilterResult,
)

from .rules import (
    RuleEngine,
    SafetyRule,
)

from .gate import (
    SafetyGate,
    SafetyDecision,
)

from .training import (
    SafetyTrainer,
    SafetyDataset,
)

__all__ = [
    # Classifier
    "SafetyClassifier",
    "SafetyClassifierConfig",
    "SafetyLabel",
    # Filters
    "InputFilter",
    "OutputFilter",
    "FilterResult",
    # Rules
    "RuleEngine",
    "SafetyRule",
    # Gate
    "SafetyGate",
    "SafetyDecision",
    # Training
    "SafetyTrainer",
    "SafetyDataset",
]
