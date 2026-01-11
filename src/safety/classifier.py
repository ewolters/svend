"""
Safety Classifier Model

Lightweight transformer-based classifier for safety decisions.

Architecture choices for production:
- DistilBERT base (fast, small, good enough)
- Multi-head output (action + category + confidence)
- Calibrated probabilities for threshold tuning

The classifier predicts:
1. Action: safe / review / refuse (for inputs) or safe / edit / block (for outputs)
2. Category: violence, illegal, self-harm, sexual, harassment, etc.
3. Confidence: Calibrated probability score
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple
from enum import Enum
from pathlib import Path
import json

import torch
import torch.nn as nn
import torch.nn.functional as F


class SafetyLabel(Enum):
    """Safety classification labels."""

    # Input actions
    SAFE = "safe"
    REVIEW = "review"  # Borderline, may need human review
    REFUSE = "refuse"

    # Output actions (mapped from same model)
    # SAFE = "safe"  # Same
    EDIT = "edit"  # Output needs modification
    BLOCK = "block"  # Output should not be shown


class HarmCategory(Enum):
    """Categories of potential harm."""

    NONE = "none"
    VIOLENCE = "violence"
    ILLEGAL_ACTIVITY = "illegal_activity"
    SELF_HARM = "self_harm"
    SEXUAL_CONTENT = "sexual_content"
    HARASSMENT = "harassment"
    HATE_SPEECH = "hate_speech"
    DANGEROUS_INSTRUCTIONS = "dangerous_instructions"
    MALWARE = "malware"
    FRAUD = "fraud"
    PRIVACY_VIOLATION = "privacy_violation"
    MISINFORMATION = "misinformation"


@dataclass
class SafetyClassifierConfig:
    """Configuration for safety classifier."""

    # Model architecture
    model_name: str = "distilbert-base-uncased"  # Base encoder
    hidden_size: int = 768
    num_action_labels: int = 3  # safe, review/edit, refuse/block
    num_category_labels: int = 12  # HarmCategory enum size
    dropout: float = 0.1

    # Classification heads
    use_category_head: bool = True
    use_confidence_head: bool = True

    # Thresholds (tunable after calibration)
    refuse_threshold: float = 0.8  # Confidence to refuse
    review_threshold: float = 0.5  # Confidence to flag for review

    # Training
    max_length: int = 512
    freeze_encoder_layers: int = 0  # Freeze first N layers

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "hidden_size": self.hidden_size,
            "num_action_labels": self.num_action_labels,
            "num_category_labels": self.num_category_labels,
            "dropout": self.dropout,
            "use_category_head": self.use_category_head,
            "use_confidence_head": self.use_confidence_head,
            "refuse_threshold": self.refuse_threshold,
            "review_threshold": self.review_threshold,
            "max_length": self.max_length,
        }

    def save(self, path: str):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "SafetyClassifierConfig":
        with open(path) as f:
            return cls(**json.load(f))


@dataclass
class SafetyPrediction:
    """Output from safety classifier."""

    # Primary prediction
    action: SafetyLabel
    action_confidence: float

    # Category prediction
    category: HarmCategory
    category_confidence: float

    # Raw probabilities for analysis
    action_probs: Dict[str, float] = field(default_factory=dict)
    category_probs: Dict[str, float] = field(default_factory=dict)

    # For logging/debugging
    input_text: str = ""
    processing_time_ms: float = 0.0

    def is_safe(self) -> bool:
        return self.action == SafetyLabel.SAFE

    def needs_review(self) -> bool:
        return self.action == SafetyLabel.REVIEW

    def should_refuse(self) -> bool:
        return self.action == SafetyLabel.REFUSE

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action.value,
            "action_confidence": self.action_confidence,
            "category": self.category.value,
            "category_confidence": self.category_confidence,
            "action_probs": self.action_probs,
            "category_probs": self.category_probs,
            "processing_time_ms": self.processing_time_ms,
        }


class SafetyClassifier(nn.Module):
    """
    Safety classification model.

    Architecture:
    - Transformer encoder (DistilBERT)
    - Action classification head
    - Category classification head (optional)
    - Confidence calibration layer (optional)
    """

    def __init__(self, config: SafetyClassifierConfig):
        super().__init__()
        self.config = config

        # We'll initialize the encoder lazily to avoid HuggingFace import at module load
        self._encoder = None
        self._tokenizer = None

        # Classification heads
        self.action_head = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size // 2, config.num_action_labels),
        )

        if config.use_category_head:
            self.category_head = nn.Sequential(
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_size, config.hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_size // 2, config.num_category_labels),
            )
        else:
            self.category_head = None

        # Temperature for calibration
        self.temperature = nn.Parameter(torch.ones(1))

    @property
    def encoder(self):
        """Lazy load encoder."""
        if self._encoder is None:
            from transformers import AutoModel
            self._encoder = AutoModel.from_pretrained(self.config.model_name)

            # Freeze layers if configured
            if self.config.freeze_encoder_layers > 0:
                for i, layer in enumerate(self._encoder.transformer.layer):
                    if i < self.config.freeze_encoder_layers:
                        for param in layer.parameters():
                            param.requires_grad = False

        return self._encoder

    @property
    def tokenizer(self):
        """Lazy load tokenizer."""
        if self._tokenizer is None:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        return self._tokenizer

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            input_ids: Token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]

        Returns:
            Dict with action_logits, category_logits
        """
        # Encode
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # Use [CLS] token representation
        cls_output = outputs.last_hidden_state[:, 0, :]

        # Action classification
        action_logits = self.action_head(cls_output)

        result = {
            "action_logits": action_logits,
            "cls_embedding": cls_output,
        }

        # Category classification
        if self.category_head is not None:
            category_logits = self.category_head(cls_output)
            result["category_logits"] = category_logits

        return result

    def predict(
        self,
        text: str,
        device: Optional[torch.device] = None,
    ) -> SafetyPrediction:
        """
        Make a safety prediction for a single text.

        Args:
            text: Input text to classify
            device: Device to run on

        Returns:
            SafetyPrediction with action and category
        """
        import time
        start_time = time.perf_counter()

        device = device or next(self.parameters()).device

        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_length,
            padding=True,
        ).to(device)

        # Forward pass
        self.eval()
        with torch.no_grad():
            outputs = self.forward(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
            )

        # Get action prediction with temperature scaling
        action_logits = outputs["action_logits"] / self.temperature
        action_probs = F.softmax(action_logits, dim=-1)[0]

        action_labels = [SafetyLabel.SAFE, SafetyLabel.REVIEW, SafetyLabel.REFUSE]
        action_idx = action_probs.argmax().item()
        action = action_labels[action_idx]
        action_confidence = action_probs[action_idx].item()

        # Get category prediction
        if "category_logits" in outputs:
            category_logits = outputs["category_logits"] / self.temperature
            category_probs = F.softmax(category_logits, dim=-1)[0]

            categories = list(HarmCategory)
            category_idx = category_probs.argmax().item()
            category = categories[category_idx]
            category_confidence = category_probs[category_idx].item()

            category_probs_dict = {
                c.value: category_probs[i].item()
                for i, c in enumerate(categories)
            }
        else:
            category = HarmCategory.NONE
            category_confidence = 1.0
            category_probs_dict = {}

        # Apply thresholds to adjust action
        if action == SafetyLabel.REFUSE and action_confidence < self.config.refuse_threshold:
            action = SafetyLabel.REVIEW
        elif action == SafetyLabel.REVIEW and action_confidence < self.config.review_threshold:
            action = SafetyLabel.SAFE

        processing_time = (time.perf_counter() - start_time) * 1000

        return SafetyPrediction(
            action=action,
            action_confidence=action_confidence,
            category=category,
            category_confidence=category_confidence,
            action_probs={
                l.value: action_probs[i].item()
                for i, l in enumerate(action_labels)
            },
            category_probs=category_probs_dict,
            input_text=text[:100],  # Truncate for logging
            processing_time_ms=processing_time,
        )

    def predict_batch(
        self,
        texts: List[str],
        device: Optional[torch.device] = None,
        batch_size: int = 32,
    ) -> List[SafetyPrediction]:
        """Batch prediction for efficiency."""
        predictions = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            device = device or next(self.parameters()).device

            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.max_length,
                padding=True,
            ).to(device)

            self.eval()
            with torch.no_grad():
                outputs = self.forward(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                )

            action_probs = F.softmax(outputs["action_logits"] / self.temperature, dim=-1)

            if "category_logits" in outputs:
                category_probs = F.softmax(outputs["category_logits"] / self.temperature, dim=-1)
            else:
                category_probs = None

            action_labels = [SafetyLabel.SAFE, SafetyLabel.REVIEW, SafetyLabel.REFUSE]
            categories = list(HarmCategory)

            for j in range(len(batch_texts)):
                action_idx = action_probs[j].argmax().item()
                action = action_labels[action_idx]
                action_conf = action_probs[j][action_idx].item()

                if category_probs is not None:
                    cat_idx = category_probs[j].argmax().item()
                    category = categories[cat_idx]
                    cat_conf = category_probs[j][cat_idx].item()
                else:
                    category = HarmCategory.NONE
                    cat_conf = 1.0

                predictions.append(SafetyPrediction(
                    action=action,
                    action_confidence=action_conf,
                    category=category,
                    category_confidence=cat_conf,
                    input_text=batch_texts[j][:100],
                ))

        return predictions

    def save_pretrained(self, path: str):
        """Save model and config."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save config
        self.config.save(path / "config.json")

        # Save model weights (excluding encoder if using pretrained)
        state_dict = {
            k: v for k, v in self.state_dict().items()
            if not k.startswith("_encoder")
        }
        torch.save(state_dict, path / "classifier.pt")

        # Save full model for easy loading
        torch.save(self.state_dict(), path / "full_model.pt")

    @classmethod
    def from_pretrained(cls, path: str, device: Optional[torch.device] = None) -> "SafetyClassifier":
        """Load model from path."""
        path = Path(path)

        config = SafetyClassifierConfig.load(path / "config.json")
        model = cls(config)

        # Load weights
        if (path / "full_model.pt").exists():
            state_dict = torch.load(path / "full_model.pt", map_location=device or "cpu")
            model.load_state_dict(state_dict)
        elif (path / "classifier.pt").exists():
            state_dict = torch.load(path / "classifier.pt", map_location=device or "cpu")
            model.load_state_dict(state_dict, strict=False)

        if device:
            model.to(device)

        return model


def create_safety_classifier(
    pretrained: bool = True,
    config: Optional[SafetyClassifierConfig] = None,
) -> SafetyClassifier:
    """
    Factory function to create a safety classifier.

    Args:
        pretrained: Whether to use pretrained encoder
        config: Optional custom config

    Returns:
        SafetyClassifier instance
    """
    if config is None:
        config = SafetyClassifierConfig()

    return SafetyClassifier(config)
