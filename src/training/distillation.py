"""
Knowledge Distillation framework for reasoning models.

Implements multiple distillation strategies:
1. Soft label distillation (KL divergence on logits)
2. Hidden state matching (intermediate layer alignment)
3. Attention transfer (attention pattern matching)
4. Reasoning trace distillation (structured reasoning transfer)

The goal: Transfer reasoning capabilities from 7B teacher to 500M student.
"""

import math
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


@dataclass
class DistillationConfig:
    """Configuration for knowledge distillation."""

    # Temperature for softening logits
    temperature: float = 2.0

    # Loss weights
    alpha_ce: float = 0.5  # Cross-entropy with ground truth
    alpha_kl: float = 0.5  # KL divergence with teacher
    alpha_hidden: float = 0.1  # Hidden state matching
    alpha_attention: float = 0.05  # Attention transfer

    # Hidden state matching
    hidden_layers_to_match: List[int] = None  # Which layers to align
    hidden_projection: bool = True  # Project student hidden to teacher dim

    # Attention matching
    attention_layers_to_match: List[int] = None

    # Training
    freeze_teacher: bool = True
    use_teacher_forcing: bool = True

    def __post_init__(self):
        # Default: match every 4th layer for hidden states
        if self.hidden_layers_to_match is None:
            self.hidden_layers_to_match = [4, 8, 12, 16, 20, 24]

        # Default: match attention at same intervals
        if self.attention_layers_to_match is None:
            self.attention_layers_to_match = [4, 12, 20]


class HiddenStateProjector(nn.Module):
    """
    Projects student hidden states to teacher dimension.

    Allows distillation even when hidden sizes differ.
    """

    def __init__(
        self,
        student_dim: int,
        teacher_dim: int,
        num_layers: int = 1,
    ):
        super().__init__()

        if num_layers == 1:
            self.projector = nn.Linear(student_dim, teacher_dim)
        else:
            layers = []
            hidden_dim = (student_dim + teacher_dim) // 2
            for i in range(num_layers):
                in_dim = student_dim if i == 0 else hidden_dim
                out_dim = teacher_dim if i == num_layers - 1 else hidden_dim
                layers.extend([
                    nn.Linear(in_dim, out_dim),
                    nn.GELU() if i < num_layers - 1 else nn.Identity(),
                ])
            self.projector = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projector(x)


class AttentionTransfer(nn.Module):
    """
    Transfers attention patterns from teacher to student.

    Maps student attention heads to teacher attention heads
    and minimizes the difference in attention distributions.
    """

    def __init__(
        self,
        student_heads: int,
        teacher_heads: int,
    ):
        super().__init__()
        self.student_heads = student_heads
        self.teacher_heads = teacher_heads

        # Learnable mapping from student heads to teacher heads
        # Each student head learns to match a weighted combination of teacher heads
        self.head_mapping = nn.Parameter(
            torch.zeros(student_heads, teacher_heads)
        )
        nn.init.eye_(self.head_mapping[:min(student_heads, teacher_heads), :min(student_heads, teacher_heads)])

    def forward(
        self,
        student_attn: torch.Tensor,  # [batch, student_heads, seq, seq]
        teacher_attn: torch.Tensor,  # [batch, teacher_heads, seq, seq]
    ) -> torch.Tensor:
        """Compute attention transfer loss."""
        # Normalize mapping weights
        weights = F.softmax(self.head_mapping, dim=1)  # [student_heads, teacher_heads]

        # Compute target attention for each student head
        # target[h] = sum_t(weights[h,t] * teacher_attn[t])
        target_attn = torch.einsum("st,btsq->bhsq", weights, teacher_attn)

        # MSE loss between student and target
        loss = F.mse_loss(student_attn, target_attn)

        return loss


class DistillationLoss(nn.Module):
    """
    Combined distillation loss function.

    Computes:
    - Cross-entropy loss with ground truth labels
    - KL divergence loss with teacher soft labels
    - Hidden state matching loss (optional)
    - Attention transfer loss (optional)
    """

    def __init__(
        self,
        config: DistillationConfig,
        student_config,
        teacher_config,
    ):
        super().__init__()
        self.config = config
        self.temperature = config.temperature

        # Hidden state projectors
        if config.alpha_hidden > 0 and config.hidden_projection:
            self.hidden_projectors = nn.ModuleDict({
                str(layer_idx): HiddenStateProjector(
                    student_config.hidden_size,
                    teacher_config.hidden_size,
                )
                for layer_idx in config.hidden_layers_to_match
            })
        else:
            self.hidden_projectors = None

        # Attention transfer modules
        if config.alpha_attention > 0:
            self.attention_transfers = nn.ModuleDict({
                str(layer_idx): AttentionTransfer(
                    student_config.num_attention_heads,
                    teacher_config.num_attention_heads,
                )
                for layer_idx in config.attention_layers_to_match
            })
        else:
            self.attention_transfers = None

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
        student_hidden_states: Optional[List[torch.Tensor]] = None,
        teacher_hidden_states: Optional[List[torch.Tensor]] = None,
        student_attentions: Optional[List[torch.Tensor]] = None,
        teacher_attentions: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined distillation loss.

        Returns:
            (total_loss, loss_components_dict)
        """
        losses = {}
        total_loss = torch.tensor(0.0, device=student_logits.device)

        # 1. Cross-entropy with ground truth
        if self.config.alpha_ce > 0:
            # Shift for next-token prediction
            shift_logits = student_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            ce_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            losses["ce_loss"] = ce_loss.item()
            total_loss = total_loss + self.config.alpha_ce * ce_loss

        # 2. KL divergence with teacher
        if self.config.alpha_kl > 0:
            # Soften logits with temperature
            student_soft = F.log_softmax(student_logits / self.temperature, dim=-1)
            teacher_soft = F.softmax(teacher_logits / self.temperature, dim=-1)

            # KL divergence (scaled by T^2 as per Hinton et al.)
            kl_loss = F.kl_div(
                student_soft,
                teacher_soft,
                reduction="batchmean",
            ) * (self.temperature ** 2)

            losses["kl_loss"] = kl_loss.item()
            total_loss = total_loss + self.config.alpha_kl * kl_loss

        # 3. Hidden state matching
        if (
            self.config.alpha_hidden > 0 and
            student_hidden_states is not None and
            teacher_hidden_states is not None
        ):
            hidden_loss = torch.tensor(0.0, device=student_logits.device)

            for layer_idx in self.config.hidden_layers_to_match:
                if layer_idx < len(student_hidden_states) and layer_idx < len(teacher_hidden_states):
                    student_hidden = student_hidden_states[layer_idx]
                    teacher_hidden = teacher_hidden_states[layer_idx]

                    # Project student hidden states if needed
                    if self.hidden_projectors is not None:
                        student_hidden = self.hidden_projectors[str(layer_idx)](student_hidden)

                    # Normalize and compute MSE
                    student_hidden = F.normalize(student_hidden, dim=-1)
                    teacher_hidden = F.normalize(teacher_hidden, dim=-1)

                    hidden_loss = hidden_loss + F.mse_loss(student_hidden, teacher_hidden)

            hidden_loss = hidden_loss / len(self.config.hidden_layers_to_match)
            losses["hidden_loss"] = hidden_loss.item()
            total_loss = total_loss + self.config.alpha_hidden * hidden_loss

        # 4. Attention transfer
        if (
            self.config.alpha_attention > 0 and
            student_attentions is not None and
            teacher_attentions is not None
        ):
            attn_loss = torch.tensor(0.0, device=student_logits.device)

            for layer_idx in self.config.attention_layers_to_match:
                if layer_idx < len(student_attentions) and layer_idx < len(teacher_attentions):
                    attn_loss = attn_loss + self.attention_transfers[str(layer_idx)](
                        student_attentions[layer_idx],
                        teacher_attentions[layer_idx],
                    )

            attn_loss = attn_loss / len(self.config.attention_layers_to_match)
            losses["attention_loss"] = attn_loss.item()
            total_loss = total_loss + self.config.alpha_attention * attn_loss

        losses["total_loss"] = total_loss.item()
        return total_loss, losses


class DistillationTrainer:
    """
    Trainer for knowledge distillation.

    Manages teacher-student training loop with proper
    handling of teacher inference and student updates.
    """

    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        config: DistillationConfig,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: Optional[torch.device] = None,
    ):
        self.teacher = teacher
        self.student = student
        self.config = config
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Move models to device
        self.teacher.to(self.device)
        self.student.to(self.device)

        # Freeze teacher
        if config.freeze_teacher:
            self.teacher.eval()
            for param in self.teacher.parameters():
                param.requires_grad = False

        # Setup loss function
        self.loss_fn = DistillationLoss(
            config,
            student.config,
            teacher.config,
        ).to(self.device)

        # Setup optimizer
        if optimizer is None:
            # Optimize student and loss function parameters
            params = list(self.student.parameters())
            if self.loss_fn.hidden_projectors is not None:
                params.extend(self.loss_fn.hidden_projectors.parameters())
            if self.loss_fn.attention_transfers is not None:
                params.extend(self.loss_fn.attention_transfers.parameters())

            self.optimizer = torch.optim.AdamW(
                params,
                lr=1e-4,
                weight_decay=0.1,
            )
        else:
            self.optimizer = optimizer

        # Training state
        self.global_step = 0

    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """Execute a single training step."""
        self.student.train()

        # Move batch to device
        batch = {k: v.to(self.device) for k, v in batch.items()}
        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask")
        labels = batch["labels"]

        # Teacher forward (no grad)
        with torch.no_grad():
            teacher_outputs = self.teacher(
                input_ids,
                attention_mask=attention_mask,
                return_hidden_states=self.config.alpha_hidden > 0,
            )
            teacher_logits = teacher_outputs["logits"]
            teacher_hidden = teacher_outputs.get("hidden_states")

        # Student forward
        student_outputs = self.student(
            input_ids,
            attention_mask=attention_mask,
            return_hidden_states=self.config.alpha_hidden > 0,
        )
        student_logits = student_outputs["logits"]
        student_hidden = student_outputs.get("hidden_states")

        # Compute loss
        loss, loss_components = self.loss_fn(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            labels=labels,
            student_hidden_states=student_hidden,
            teacher_hidden_states=teacher_hidden,
        )

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.student.parameters(), 1.0)
        self.optimizer.step()

        self.global_step += 1
        return loss_components

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        total_losses = {}
        num_batches = 0

        for batch in self.train_dataloader:
            losses = self.train_step(batch)

            for key, value in losses.items():
                total_losses[key] = total_losses.get(key, 0) + value
            num_batches += 1

            if self.global_step % 100 == 0:
                avg_loss = total_losses["total_loss"] / num_batches
                print(f"Step {self.global_step} | Loss: {avg_loss:.4f}")

        return {k: v / num_batches for k, v in total_losses.items()}

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Evaluate the student model."""
        if self.eval_dataloader is None:
            return {}

        self.student.eval()
        total_loss = 0
        num_batches = 0

        for batch in self.eval_dataloader:
            batch = {k: v.to(self.device) for k, v in batch.items()}

            outputs = self.student(
                batch["input_ids"],
                attention_mask=batch.get("attention_mask"),
                labels=batch["labels"],
            )

            total_loss += outputs["loss"].item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        return {
            "eval_loss": avg_loss,
            "eval_perplexity": math.exp(avg_loss),
        }


def distill_reasoning_model(
    teacher,
    student,
    train_data,
    eval_data=None,
    config: Optional[DistillationConfig] = None,
    num_epochs: int = 3,
    batch_size: int = 4,
):
    """
    High-level function to distill a reasoning model.

    Args:
        teacher: Pre-trained teacher model (e.g., 7B)
        student: Smaller student model (e.g., 500M)
        train_data: Training dataset
        eval_data: Evaluation dataset
        config: Distillation configuration
        num_epochs: Number of training epochs
        batch_size: Batch size

    Returns:
        Trained student model
    """
    if config is None:
        config = DistillationConfig()

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
    )

    eval_loader = None
    if eval_data is not None:
        eval_loader = DataLoader(
            eval_data,
            batch_size=batch_size,
            shuffle=False,
        )

    trainer = DistillationTrainer(
        teacher=teacher,
        student=student,
        config=config,
        train_dataloader=train_loader,
        eval_dataloader=eval_loader,
    )

    print(f"\nStarting distillation")
    print(f"  Teacher parameters: {sum(p.numel() for p in teacher.parameters()) / 1e9:.2f}B")
    print(f"  Student parameters: {sum(p.numel() for p in student.parameters()) / 1e9:.2f}B")
    print(f"  Compression ratio: {sum(p.numel() for p in teacher.parameters()) / sum(p.numel() for p in student.parameters()):.1f}x")

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        train_losses = trainer.train_epoch()

        print(f"  Train losses: {train_losses}")

        if eval_loader is not None:
            eval_metrics = trainer.evaluate()
            print(f"  Eval metrics: {eval_metrics}")

    return student
