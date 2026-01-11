"""
Rule-Based Safety Filtering

Fast, interpretable rules that run before the ML classifier.

Purpose:
1. Catch obvious cases without model inference (speed)
2. Provide explainable decisions (auditing)
3. Handle known patterns (blocklists, allowlists)
4. Defense in depth (rules + ML)

Rules are checked in order:
1. Allowlist rules (fast path for known-safe patterns)
2. Blocklist rules (immediate rejection)
3. Pattern rules (regex-based detection)
4. If no rule matches, defer to ML classifier
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable, Pattern
from enum import Enum
import re
from pathlib import Path
import json


class RuleAction(Enum):
    """Action to take when rule matches."""
    ALLOW = "allow"    # Definitely safe, skip ML
    BLOCK = "block"    # Definitely unsafe, reject
    FLAG = "flag"      # Borderline, flag for review
    DEFER = "defer"    # No match, continue to ML


@dataclass
class RuleMatch:
    """Result of a rule check."""
    matched: bool
    action: RuleAction
    rule_name: str
    reason: str
    matched_pattern: Optional[str] = None
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "matched": self.matched,
            "action": self.action.value,
            "rule_name": self.rule_name,
            "reason": self.reason,
            "matched_pattern": self.matched_pattern,
            "confidence": self.confidence,
        }


@dataclass
class SafetyRule:
    """
    A single safety rule.

    Rules can be:
    - Keyword-based (fast string matching)
    - Pattern-based (regex)
    - Function-based (custom logic)
    """

    name: str
    description: str
    action: RuleAction
    priority: int = 100  # Lower = higher priority

    # Matching criteria (use one)
    keywords: List[str] = field(default_factory=list)
    patterns: List[str] = field(default_factory=list)  # Regex patterns
    check_fn: Optional[Callable[[str], bool]] = None

    # Compiled patterns (internal)
    _compiled_patterns: List[Pattern] = field(default_factory=list, repr=False)

    def __post_init__(self):
        # Compile regex patterns
        self._compiled_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.patterns
        ]

    def check(self, text: str) -> RuleMatch:
        """Check if rule matches the text."""
        text_lower = text.lower()

        # Keyword matching
        for keyword in self.keywords:
            if keyword.lower() in text_lower:
                return RuleMatch(
                    matched=True,
                    action=self.action,
                    rule_name=self.name,
                    reason=self.description,
                    matched_pattern=keyword,
                )

        # Pattern matching
        for pattern in self._compiled_patterns:
            match = pattern.search(text)
            if match:
                return RuleMatch(
                    matched=True,
                    action=self.action,
                    rule_name=self.name,
                    reason=self.description,
                    matched_pattern=match.group(),
                )

        # Custom function
        if self.check_fn is not None:
            if self.check_fn(text):
                return RuleMatch(
                    matched=True,
                    action=self.action,
                    rule_name=self.name,
                    reason=self.description,
                )

        return RuleMatch(
            matched=False,
            action=RuleAction.DEFER,
            rule_name=self.name,
            reason="No match",
        )


class RuleEngine:
    """
    Rule engine for safety filtering.

    Checks rules in priority order and returns first match.
    """

    def __init__(self):
        self.rules: List[SafetyRule] = []
        self._load_default_rules()

    def _load_default_rules(self):
        """Load default safety rules."""

        # === BLOCK RULES (high priority) ===

        # Explicit violence instructions
        self.add_rule(SafetyRule(
            name="violence_instructions",
            description="Request for instructions on violence or weapons",
            action=RuleAction.BLOCK,
            priority=10,
            patterns=[
                r"how\s+(?:to|do\s+i|can\s+i)\s+(?:make|build|create)\s+(?:a\s+)?(?:bomb|explosive|weapon)",
                r"instructions?\s+(?:for|to)\s+(?:make|build)\s+(?:a\s+)?(?:bomb|explosive|gun)",
                r"synthesize\s+(?:explosives?|chemical\s+weapons?)",
            ],
        ))

        # Malware requests
        self.add_rule(SafetyRule(
            name="malware_requests",
            description="Request for malicious software",
            action=RuleAction.BLOCK,
            priority=10,
            patterns=[
                r"(?:write|create|make)\s+(?:a\s+)?(?:virus|malware|trojan|ransomware|keylogger)",
                r"(?:code|script)\s+(?:to|that)\s+(?:steal|hack|infect|spread)",
                r"(?:bypass|evade)\s+(?:antivirus|security|detection)",
            ],
        ))

        # Illegal drug synthesis
        self.add_rule(SafetyRule(
            name="drug_synthesis",
            description="Request for illegal drug synthesis",
            action=RuleAction.BLOCK,
            priority=10,
            patterns=[
                r"(?:how\s+to|instructions?\s+for)\s+(?:make|synthesize|cook|produce)\s+(?:meth|methamphetamine|cocaine|heroin|fentanyl|lsd)",
                r"(?:recipe|synthesis|production)\s+(?:of|for)\s+(?:meth|methamphetamine|mdma|cocaine)",
            ],
        ))

        # Self-harm
        self.add_rule(SafetyRule(
            name="self_harm",
            description="Content related to self-harm",
            action=RuleAction.BLOCK,
            priority=10,
            patterns=[
                r"(?:how\s+to|best\s+way\s+to)\s+(?:kill|hurt)\s+(?:myself|yourself)",
                r"(?:methods?|ways?)\s+(?:of|to)\s+(?:suicide|self.?harm)",
            ],
        ))

        # CSAM
        self.add_rule(SafetyRule(
            name="csam",
            description="Child sexual abuse material",
            action=RuleAction.BLOCK,
            priority=1,  # Highest priority
            keywords=["child porn", "cp links", "underage sex"],
            patterns=[
                r"(?:sexual|nude|naked)\s+(?:images?|photos?|videos?)\s+(?:of\s+)?(?:children|minors?|kids?)",
            ],
        ))

        # === FLAG RULES (medium priority) ===

        # Hacking (context dependent)
        self.add_rule(SafetyRule(
            name="hacking_requests",
            description="Potentially malicious hacking request",
            action=RuleAction.FLAG,
            priority=50,
            patterns=[
                r"(?:how\s+to|help\s+me)\s+hack\s+(?:into|someone)",
                r"(?:steal|get)\s+(?:someone's|their)\s+(?:password|credentials|data)",
                r"(?:break\s+into|access)\s+(?:someone's|their)\s+(?:account|email|computer)",
            ],
        ))

        # Phishing
        self.add_rule(SafetyRule(
            name="phishing",
            description="Phishing or social engineering",
            action=RuleAction.FLAG,
            priority=50,
            patterns=[
                r"(?:write|create)\s+(?:a\s+)?(?:phishing|scam)\s+(?:email|message)",
                r"(?:trick|deceive)\s+(?:someone|people)\s+(?:into|to)\s+(?:giving|sending)",
            ],
        ))

        # Chemistry (context dependent - legitimate vs harmful)
        self.add_rule(SafetyRule(
            name="dangerous_chemistry",
            description="Potentially dangerous chemistry request",
            action=RuleAction.FLAG,
            priority=60,
            patterns=[
                r"(?:make|synthesize|produce)\s+(?:at\s+home\s+)?(?:poison|toxin|nerve\s+agent)",
                r"(?:chlorine|mustard)\s+gas\s+(?:synthesis|production|recipe)",
            ],
        ))

        # === ALLOW RULES (fast path for safe content) ===

        # Educational chemistry/physics
        self.add_rule(SafetyRule(
            name="educational_science",
            description="Educational science questions",
            action=RuleAction.ALLOW,
            priority=100,
            patterns=[
                r"what\s+is\s+the\s+(?:formula|equation|chemical\s+structure)",
                r"explain\s+(?:how|why)\s+(?:\w+\s+){0,3}(?:works?|happens?|occurs?)",
                r"(?:history|discovery)\s+of\s+(?:\w+\s+)?(?:chemistry|physics|element)",
            ],
        ))

        # Programming help
        self.add_rule(SafetyRule(
            name="programming_help",
            description="General programming assistance",
            action=RuleAction.ALLOW,
            priority=100,
            patterns=[
                r"(?:how\s+to|help\s+me)\s+(?:write|code|implement|debug)\s+(?:a\s+)?(?:function|class|program|script)",
                r"(?:what\s+is|explain)\s+(?:a\s+)?(?:\w+\s+)?(?:algorithm|data\s+structure|pattern)",
                r"(?:fix|debug)\s+(?:this|my)\s+(?:code|error|bug)",
            ],
        ))

        # Math questions
        self.add_rule(SafetyRule(
            name="math_questions",
            description="Mathematical questions",
            action=RuleAction.ALLOW,
            priority=100,
            patterns=[
                r"(?:solve|calculate|compute|find)\s+(?:the\s+)?(?:\w+\s+)?(?:equation|integral|derivative|sum)",
                r"(?:what\s+is|prove\s+that)\s+(?:\d+|\w+)\s*[\+\-\*\/\=]",
            ],
        ))

    def add_rule(self, rule: SafetyRule):
        """Add a rule to the engine."""
        self.rules.append(rule)
        # Keep rules sorted by priority
        self.rules.sort(key=lambda r: r.priority)

    def remove_rule(self, name: str):
        """Remove a rule by name."""
        self.rules = [r for r in self.rules if r.name != name]

    def check(self, text: str) -> RuleMatch:
        """
        Check text against all rules.

        Returns first matching rule result.
        """
        for rule in self.rules:
            result = rule.check(text)
            if result.matched:
                return result

        # No rule matched
        return RuleMatch(
            matched=False,
            action=RuleAction.DEFER,
            rule_name="none",
            reason="No rule matched, defer to ML classifier",
        )

    def check_all(self, text: str) -> List[RuleMatch]:
        """Check text against all rules and return all matches."""
        matches = []
        for rule in self.rules:
            result = rule.check(text)
            if result.matched:
                matches.append(result)
        return matches

    def save(self, path: str):
        """Save rules to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        rules_data = []
        for rule in self.rules:
            rules_data.append({
                "name": rule.name,
                "description": rule.description,
                "action": rule.action.value,
                "priority": rule.priority,
                "keywords": rule.keywords,
                "patterns": rule.patterns,
            })

        with open(path, 'w') as f:
            json.dump(rules_data, f, indent=2)

    def load(self, path: str):
        """Load rules from JSON file."""
        with open(path) as f:
            rules_data = json.load(f)

        for data in rules_data:
            rule = SafetyRule(
                name=data["name"],
                description=data["description"],
                action=RuleAction(data["action"]),
                priority=data.get("priority", 100),
                keywords=data.get("keywords", []),
                patterns=data.get("patterns", []),
            )
            self.add_rule(rule)

    def summary(self) -> str:
        """Get summary of loaded rules."""
        lines = [
            "Safety Rule Engine",
            "=" * 40,
            f"Total rules: {len(self.rules)}",
            "",
        ]

        by_action = {}
        for rule in self.rules:
            action = rule.action.value
            if action not in by_action:
                by_action[action] = []
            by_action[action].append(rule.name)

        for action, names in by_action.items():
            lines.append(f"{action.upper()} rules ({len(names)}):")
            for name in names:
                lines.append(f"  - {name}")
            lines.append("")

        return "\n".join(lines)


def create_rule_engine(include_defaults: bool = True) -> RuleEngine:
    """Factory function to create rule engine."""
    engine = RuleEngine()

    if not include_defaults:
        engine.rules = []

    return engine
