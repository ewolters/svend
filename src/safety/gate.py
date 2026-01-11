"""
Safety Gate

High-level interface that combines all safety components.

This is the main entry point for safety checking in production.

Usage:
    from src.safety import SafetyGate

    gate = SafetyGate.from_pretrained("checkpoints/safety")

    # Check input
    decision = gate.check_input("user message here")
    if decision.should_block():
        return gate.get_refusal_message(decision)

    # Process request...
    response = model.generate(...)

    # Check output
    decision = gate.check_output(response)
    if decision.should_block():
        return gate.get_safe_fallback()

    return decision.get_content()  # Possibly modified
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from pathlib import Path
import json
import time

from .classifier import SafetyClassifier, SafetyClassifierConfig
from .filters import InputFilter, OutputFilter, FilterResult, FilterAction
from .rules import RuleEngine


@dataclass
class SafetyDecision:
    """
    High-level safety decision.

    Provides a simple interface for application code.
    """

    # Core decision
    allowed: bool
    action: FilterAction
    reason: str

    # Content handling
    original_content: str
    final_content: str  # May be modified

    # Details for logging
    input_filter_result: Optional[FilterResult] = None
    output_filter_result: Optional[FilterResult] = None

    # Timing
    processing_time_ms: float = 0.0

    def should_block(self) -> bool:
        """Check if content should be blocked."""
        return self.action == FilterAction.BLOCK

    def should_review(self) -> bool:
        """Check if content needs human review."""
        return self.action == FilterAction.REVIEW

    def was_modified(self) -> bool:
        """Check if content was modified."""
        return self.original_content != self.final_content

    def get_content(self) -> str:
        """Get the final (possibly modified) content."""
        return self.final_content

    def to_dict(self) -> Dict[str, Any]:
        return {
            "allowed": self.allowed,
            "action": self.action.value,
            "reason": self.reason,
            "was_modified": self.was_modified(),
            "processing_time_ms": self.processing_time_ms,
        }


class SafetyGate:
    """
    Main safety gateway for the application.

    Combines:
    - Rule engine (fast, interpretable)
    - ML classifier (nuanced, learned)
    - Input filtering
    - Output filtering

    Design principles:
    - Fail closed: when uncertain, be conservative
    - Speed: rules run first for fast path
    - Auditability: all decisions are logged
    - Flexibility: thresholds can be tuned
    """

    # Default refusal messages by category
    REFUSAL_MESSAGES = {
        "violence": "I can't help with requests involving violence or weapons.",
        "illegal_activity": "I can't assist with illegal activities.",
        "self_harm": "I'm concerned about your wellbeing. If you're struggling, please reach out to a crisis helpline.",
        "sexual_content": "I can't generate sexual content.",
        "harassment": "I can't help with harassment or targeting individuals.",
        "hate_speech": "I can't generate hate speech or discriminatory content.",
        "dangerous_instructions": "I can't provide instructions for dangerous activities.",
        "malware": "I can't help create malicious software.",
        "fraud": "I can't assist with fraud or scams.",
        "default": "I'm not able to help with that request.",
    }

    # Safe fallback response
    SAFE_FALLBACK = "I apologize, but I can't provide that response. Is there something else I can help you with?"

    def __init__(
        self,
        classifier: Optional[SafetyClassifier] = None,
        rule_engine: Optional[RuleEngine] = None,
        enable_rules: bool = True,
        enable_classifier: bool = True,
        enable_redaction: bool = True,
        log_decisions: bool = True,
    ):
        """
        Initialize safety gate.

        Args:
            classifier: ML safety classifier
            rule_engine: Rule-based filter
            enable_rules: Whether to use rules
            enable_classifier: Whether to use ML classifier
            enable_redaction: Whether to redact sensitive data from outputs
            log_decisions: Whether to log all decisions
        """
        self.classifier = classifier
        self.rule_engine = rule_engine or RuleEngine()
        self.enable_rules = enable_rules
        self.enable_classifier = enable_classifier
        self.enable_redaction = enable_redaction
        self.log_decisions = log_decisions

        # Create filters
        self.input_filter = InputFilter(
            classifier=classifier if enable_classifier else None,
            rule_engine=rule_engine if enable_rules else None,
            use_rules=enable_rules,
            use_classifier=enable_classifier,
        )

        self.output_filter = OutputFilter(
            classifier=classifier if enable_classifier else None,
            rule_engine=rule_engine if enable_rules else None,
        )

        # Decision log
        self.decision_log: List[Dict[str, Any]] = []

    def check_input(self, text: str) -> SafetyDecision:
        """
        Check if user input is safe to process.

        Args:
            text: User input text

        Returns:
            SafetyDecision indicating whether to proceed
        """
        start_time = time.perf_counter()

        result = self.input_filter.check(text)

        decision = SafetyDecision(
            allowed=(result.action == FilterAction.PASS),
            action=result.action,
            reason=result.reason,
            original_content=text,
            final_content=text,  # Input is not modified
            input_filter_result=result,
            processing_time_ms=(time.perf_counter() - start_time) * 1000,
        )

        if self.log_decisions:
            self._log_decision("input", decision)

        return decision

    def check_output(self, text: str) -> SafetyDecision:
        """
        Check if model output is safe to return.

        Args:
            text: Model output text

        Returns:
            SafetyDecision with possibly modified content
        """
        start_time = time.perf_counter()

        result = self.output_filter.check(text, redact=self.enable_redaction)

        # Determine final content
        if result.action == FilterAction.BLOCK:
            final_content = self.SAFE_FALLBACK
        elif result.modified_content:
            final_content = result.modified_content
        else:
            final_content = text

        decision = SafetyDecision(
            allowed=(result.action != FilterAction.BLOCK),
            action=result.action,
            reason=result.reason,
            original_content=text,
            final_content=final_content,
            output_filter_result=result,
            processing_time_ms=(time.perf_counter() - start_time) * 1000,
        )

        if self.log_decisions:
            self._log_decision("output", decision)

        return decision

    def get_refusal_message(self, decision: SafetyDecision) -> str:
        """
        Get appropriate refusal message for a blocked input.

        Args:
            decision: The safety decision

        Returns:
            Human-friendly refusal message
        """
        # Try to get category-specific message
        if decision.input_filter_result:
            if decision.input_filter_result.classifier_prediction:
                category = decision.input_filter_result.classifier_prediction.category.value
                if category in self.REFUSAL_MESSAGES:
                    return self.REFUSAL_MESSAGES[category]

        return self.REFUSAL_MESSAGES["default"]

    def get_safe_fallback(self) -> str:
        """Get safe fallback response."""
        return self.SAFE_FALLBACK

    def _log_decision(self, check_type: str, decision: SafetyDecision):
        """Log a safety decision."""
        log_entry = {
            "type": check_type,
            "timestamp": time.time(),
            "allowed": decision.allowed,
            "action": decision.action.value,
            "reason": decision.reason,
            "processing_time_ms": decision.processing_time_ms,
        }
        self.decision_log.append(log_entry)

        # Keep log bounded
        if len(self.decision_log) > 10000:
            self.decision_log = self.decision_log[-5000:]

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics on safety decisions."""
        if not self.decision_log:
            return {"total": 0}

        total = len(self.decision_log)
        blocked = sum(1 for d in self.decision_log if not d["allowed"])
        reviewed = sum(1 for d in self.decision_log if d["action"] == "review")
        avg_time = sum(d["processing_time_ms"] for d in self.decision_log) / total

        by_type = {}
        for d in self.decision_log:
            t = d["type"]
            if t not in by_type:
                by_type[t] = {"total": 0, "blocked": 0}
            by_type[t]["total"] += 1
            if not d["allowed"]:
                by_type[t]["blocked"] += 1

        return {
            "total": total,
            "blocked": blocked,
            "blocked_rate": blocked / total,
            "reviewed": reviewed,
            "avg_processing_time_ms": avg_time,
            "by_type": by_type,
        }

    def save_log(self, path: str):
        """Save decision log to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w') as f:
            json.dump(self.decision_log, f, indent=2)

    def save(self, path: str):
        """Save gate configuration and rules."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save config
        config = {
            "enable_rules": self.enable_rules,
            "enable_classifier": self.enable_classifier,
            "enable_redaction": self.enable_redaction,
        }
        with open(path / "gate_config.json", 'w') as f:
            json.dump(config, f, indent=2)

        # Save rules
        self.rule_engine.save(path / "rules.json")

        # Save classifier if exists
        if self.classifier is not None:
            self.classifier.save_pretrained(path / "classifier")

    @classmethod
    def from_pretrained(
        cls,
        path: str,
        device: Optional[str] = None,
    ) -> "SafetyGate":
        """Load gate from saved checkpoint."""
        path = Path(path)

        # Load config
        config_path = path / "gate_config.json"
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
        else:
            config = {}

        # Load rules
        rules_path = path / "rules.json"
        rule_engine = RuleEngine()
        if rules_path.exists():
            rule_engine.load(rules_path)

        # Load classifier
        classifier_path = path / "classifier"
        classifier = None
        if classifier_path.exists():
            classifier = SafetyClassifier.from_pretrained(classifier_path, device=device)

        return cls(
            classifier=classifier,
            rule_engine=rule_engine,
            **config,
        )

    @classmethod
    def create_default(cls, with_classifier: bool = False) -> "SafetyGate":
        """Create a gate with default settings."""
        classifier = None
        if with_classifier:
            classifier = SafetyClassifier(SafetyClassifierConfig())

        return cls(
            classifier=classifier,
            rule_engine=RuleEngine(),
            enable_rules=True,
            enable_classifier=with_classifier,
        )


def quick_safety_check(text: str, is_input: bool = True) -> SafetyDecision:
    """
    Quick safety check using only rules.

    Use this for fast checks when ML classifier isn't loaded.
    """
    gate = SafetyGate.create_default(with_classifier=False)

    if is_input:
        return gate.check_input(text)
    else:
        return gate.check_output(text)
