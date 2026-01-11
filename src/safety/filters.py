"""
Input and Output Filters

High-level filtering interfaces that combine rules and ML classification.

InputFilter: Checks user inputs before processing
OutputFilter: Checks model outputs before returning to user
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from enum import Enum
import time

from .rules import RuleEngine, RuleAction, RuleMatch
from .classifier import SafetyClassifier, SafetyPrediction, SafetyLabel


class FilterAction(Enum):
    """Actions a filter can take."""
    PASS = "pass"        # Content is safe, proceed
    MODIFY = "modify"    # Content needs modification
    BLOCK = "block"      # Content should be blocked
    REVIEW = "review"    # Content needs human review


@dataclass
class FilterResult:
    """Result from a filter check."""

    action: FilterAction
    reason: str

    # Source of decision
    decided_by: str = "unknown"  # "rule" or "classifier"

    # Details
    rule_match: Optional[RuleMatch] = None
    classifier_prediction: Optional[SafetyPrediction] = None

    # Timing
    processing_time_ms: float = 0.0

    # For modifications
    modified_content: Optional[str] = None
    modifications_made: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "action": self.action.value,
            "reason": self.reason,
            "decided_by": self.decided_by,
            "processing_time_ms": self.processing_time_ms,
        }

        if self.rule_match:
            result["rule_match"] = self.rule_match.to_dict()

        if self.classifier_prediction:
            result["classifier_prediction"] = self.classifier_prediction.to_dict()

        if self.modifications_made:
            result["modifications_made"] = self.modifications_made

        return result


class InputFilter:
    """
    Filters user inputs before processing.

    Flow:
    1. Check rules (fast path)
    2. If no rule match, run classifier
    3. Return decision with reason
    """

    def __init__(
        self,
        classifier: Optional[SafetyClassifier] = None,
        rule_engine: Optional[RuleEngine] = None,
        use_rules: bool = True,
        use_classifier: bool = True,
    ):
        """
        Initialize input filter.

        Args:
            classifier: ML classifier (None = rules only)
            rule_engine: Rule engine (None = classifier only)
            use_rules: Whether to use rule-based filtering
            use_classifier: Whether to use ML classifier
        """
        self.classifier = classifier
        self.rule_engine = rule_engine or RuleEngine()
        self.use_rules = use_rules
        self.use_classifier = use_classifier

    def check(self, text: str) -> FilterResult:
        """
        Check if input is safe.

        Args:
            text: User input text

        Returns:
            FilterResult with decision
        """
        start_time = time.perf_counter()

        # Step 1: Rule-based check
        if self.use_rules:
            rule_result = self.rule_engine.check(text)

            if rule_result.matched:
                action = self._rule_action_to_filter_action(rule_result.action)

                return FilterResult(
                    action=action,
                    reason=rule_result.reason,
                    decided_by="rule",
                    rule_match=rule_result,
                    processing_time_ms=(time.perf_counter() - start_time) * 1000,
                )

        # Step 2: Classifier check
        if self.use_classifier and self.classifier is not None:
            prediction = self.classifier.predict(text)

            action = self._safety_label_to_filter_action(prediction.action)

            return FilterResult(
                action=action,
                reason=f"Classified as {prediction.action.value} ({prediction.category.value})",
                decided_by="classifier",
                classifier_prediction=prediction,
                processing_time_ms=(time.perf_counter() - start_time) * 1000,
            )

        # No filtering applied
        return FilterResult(
            action=FilterAction.PASS,
            reason="No filter applied",
            decided_by="none",
            processing_time_ms=(time.perf_counter() - start_time) * 1000,
        )

    def _rule_action_to_filter_action(self, rule_action: RuleAction) -> FilterAction:
        """Convert rule action to filter action."""
        mapping = {
            RuleAction.ALLOW: FilterAction.PASS,
            RuleAction.BLOCK: FilterAction.BLOCK,
            RuleAction.FLAG: FilterAction.REVIEW,
            RuleAction.DEFER: FilterAction.PASS,
        }
        return mapping.get(rule_action, FilterAction.PASS)

    def _safety_label_to_filter_action(self, label: SafetyLabel) -> FilterAction:
        """Convert safety label to filter action."""
        mapping = {
            SafetyLabel.SAFE: FilterAction.PASS,
            SafetyLabel.REVIEW: FilterAction.REVIEW,
            SafetyLabel.REFUSE: FilterAction.BLOCK,
            SafetyLabel.EDIT: FilterAction.MODIFY,
            SafetyLabel.BLOCK: FilterAction.BLOCK,
        }
        return mapping.get(label, FilterAction.PASS)


class OutputFilter:
    """
    Filters model outputs before returning to user.

    Can:
    - Block harmful outputs
    - Redact sensitive information
    - Add safety warnings
    - Modify problematic content
    """

    def __init__(
        self,
        classifier: Optional[SafetyClassifier] = None,
        rule_engine: Optional[RuleEngine] = None,
        redact_patterns: Optional[List[str]] = None,
    ):
        """
        Initialize output filter.

        Args:
            classifier: ML classifier for output checking
            rule_engine: Rule engine for output checking
            redact_patterns: Regex patterns to redact from outputs
        """
        self.classifier = classifier
        self.rule_engine = rule_engine or RuleEngine()
        self.redact_patterns = redact_patterns or self._default_redact_patterns()

        import re
        self._compiled_redact = [
            (re.compile(p, re.IGNORECASE), name)
            for name, p in self.redact_patterns
        ]

    def _default_redact_patterns(self) -> List[tuple]:
        """Default patterns to redact from outputs."""
        return [
            ("email", r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
            ("phone", r"\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b"),
            ("ssn", r"\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b"),
            ("credit_card", r"\b(?:\d{4}[-.\s]?){3}\d{4}\b"),
            ("ip_address", r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
        ]

    def check(self, text: str, redact: bool = True) -> FilterResult:
        """
        Check and optionally modify output.

        Args:
            text: Model output text
            redact: Whether to redact sensitive patterns

        Returns:
            FilterResult with decision and possibly modified content
        """
        start_time = time.perf_counter()
        modifications = []

        # Step 1: Redact sensitive patterns
        modified_text = text
        if redact:
            for pattern, name in self._compiled_redact:
                if pattern.search(modified_text):
                    modified_text = pattern.sub(f"[REDACTED_{name.upper()}]", modified_text)
                    modifications.append(f"Redacted {name}")

        # Step 2: Rule check
        rule_result = self.rule_engine.check(modified_text)

        if rule_result.matched and rule_result.action == RuleAction.BLOCK:
            return FilterResult(
                action=FilterAction.BLOCK,
                reason=rule_result.reason,
                decided_by="rule",
                rule_match=rule_result,
                processing_time_ms=(time.perf_counter() - start_time) * 1000,
            )

        # Step 3: Classifier check
        if self.classifier is not None:
            prediction = self.classifier.predict(modified_text)

            if prediction.action == SafetyLabel.REFUSE:
                return FilterResult(
                    action=FilterAction.BLOCK,
                    reason=f"Output classified as harmful: {prediction.category.value}",
                    decided_by="classifier",
                    classifier_prediction=prediction,
                    processing_time_ms=(time.perf_counter() - start_time) * 1000,
                )

        # Return result (with modifications if any)
        if modifications:
            return FilterResult(
                action=FilterAction.MODIFY,
                reason="Output modified for safety",
                decided_by="redaction",
                modified_content=modified_text,
                modifications_made=modifications,
                processing_time_ms=(time.perf_counter() - start_time) * 1000,
            )

        return FilterResult(
            action=FilterAction.PASS,
            reason="Output is safe",
            decided_by="none",
            processing_time_ms=(time.perf_counter() - start_time) * 1000,
        )

    def add_safety_warning(
        self,
        text: str,
        warning: str = "This response was generated by an AI and may contain errors.",
    ) -> str:
        """Add a safety warning to output."""
        return f"{warning}\n\n{text}"


def create_input_filter(
    classifier: Optional[SafetyClassifier] = None,
    include_default_rules: bool = True,
) -> InputFilter:
    """Factory function to create input filter."""
    rule_engine = RuleEngine() if include_default_rules else None

    return InputFilter(
        classifier=classifier,
        rule_engine=rule_engine,
    )


def create_output_filter(
    classifier: Optional[SafetyClassifier] = None,
    include_default_rules: bool = True,
) -> OutputFilter:
    """Factory function to create output filter."""
    rule_engine = RuleEngine() if include_default_rules else None

    return OutputFilter(
        classifier=classifier,
        rule_engine=rule_engine,
    )
