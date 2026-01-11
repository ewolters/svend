"""
Training infrastructure for reasoning models.

Supports:
- Full precision and mixed precision training
- Gradient accumulation for effective large batches
- Gradient checkpointing for memory efficiency
- Checkpoint saving and resumption
- Learning rate scheduling
- Logging to wandb/tensorboard
"""

import os
import math
import time
from typing import Optional, Dict, Any, Callable, List
from dataclasses import dataclass, field
from pathlib import Path
import json

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


@dataclass
class TrainingConfig:
    """Configuration for training loop."""

    # Basic training
    num_epochs: int = 3
    max_steps: Optional[int] = None  # If set, overrides num_epochs
    learning_rate: float = 1e-4
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0

    # Batch settings
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    effective_batch_size: int = field(init=False)

    # Learning rate schedule
    warmup_steps: int = 500
    warmup_ratio: float = 0.0  # Alternative: fraction of total steps
    lr_scheduler: str = "cosine"  # "cosine", "linear", "constant"
    min_lr_ratio: float = 0.1  # Final LR = initial * min_lr_ratio

    # Precision
    mixed_precision: bool = True
    bf16: bool = True  # Use bf16 if available, else fp16

    # Checkpointing
    output_dir: str = "checkpoints"
    save_steps: int = 1000
    save_total_limit: int = 3
    resume_from: Optional[str] = None

    # Logging
    logging_steps: int = 10
    eval_steps: int = 500
    use_wandb: bool = True
    wandb_project: str = "reasoning-model"
    wandb_run_name: Optional[str] = None

    # Optimization
    use_fused_adam: bool = True
    gradient_checkpointing: bool = True

    def __post_init__(self):
        self.effective_batch_size = self.batch_size * self.gradient_accumulation_steps

        # Validate precision settings
        if self.bf16 and not torch.cuda.is_bf16_supported():
            print("Warning: bf16 not supported, falling back to fp16")
            self.bf16 = False


class Trainer:
    """
    Training loop for custom transformer models.

    Handles:
    - Mixed precision training
    - Gradient accumulation
    - Checkpointing
    - Learning rate scheduling
    - Logging
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        compute_metrics: Optional[Callable] = None,
    ):
        self.model = model
        self.config = config
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.compute_metrics = compute_metrics

        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Setup directories
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize training state
        self.global_step = 0
        self.epoch = 0
        self.best_eval_loss = float("inf")

        # Setup optimizer
        self._setup_optimizer()

        # Setup scheduler
        self._setup_scheduler()

        # Setup mixed precision
        self._setup_amp()

        # Setup logging
        self._setup_logging()

        # Enable gradient checkpointing if requested
        if config.gradient_checkpointing and hasattr(model, "enable_gradient_checkpointing"):
            model.enable_gradient_checkpointing()

        # Resume from checkpoint if specified
        if config.resume_from:
            self.load_checkpoint(config.resume_from)

    def _setup_optimizer(self):
        """Setup AdamW optimizer with proper weight decay."""
        # Separate parameters that should/shouldn't have weight decay
        decay_params = []
        no_decay_params = []

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if "bias" in name or "norm" in name or "layernorm" in name:
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)

        optimizer_groups = [
            {"params": decay_params, "weight_decay": self.config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        # Try to use fused AdamW for better performance
        use_fused = (
            self.config.use_fused_adam and
            torch.cuda.is_available() and
            "fused" in torch.optim.AdamW.__init__.__code__.co_varnames
        )

        self.optimizer = AdamW(
            optimizer_groups,
            lr=self.config.learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8,
            fused=use_fused if use_fused else None,
        )

    def _setup_scheduler(self):
        """Setup learning rate scheduler with warmup."""
        total_steps = self._get_total_steps()

        # Calculate warmup steps
        if self.config.warmup_ratio > 0:
            warmup_steps = int(total_steps * self.config.warmup_ratio)
        else:
            warmup_steps = self.config.warmup_steps

        # Warmup scheduler
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=warmup_steps,
        )

        # Main scheduler
        if self.config.lr_scheduler == "cosine":
            main_scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=total_steps - warmup_steps,
                eta_min=self.config.learning_rate * self.config.min_lr_ratio,
            )
        elif self.config.lr_scheduler == "linear":
            main_scheduler = LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=self.config.min_lr_ratio,
                total_iters=total_steps - warmup_steps,
            )
        else:  # constant
            main_scheduler = LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=1.0,
                total_iters=total_steps - warmup_steps,
            )

        # Combine schedulers
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_steps],
        )

    def _setup_amp(self):
        """Setup automatic mixed precision."""
        if self.config.mixed_precision:
            self.scaler = GradScaler()
            self.autocast_dtype = torch.bfloat16 if self.config.bf16 else torch.float16
        else:
            self.scaler = None
            self.autocast_dtype = None

    def _setup_logging(self):
        """Setup wandb logging."""
        if self.config.use_wandb and WANDB_AVAILABLE:
            wandb.init(
                project=self.config.wandb_project,
                name=self.config.wandb_run_name,
                config={
                    **self.config.__dict__,
                    "model_params": sum(p.numel() for p in self.model.parameters()),
                },
            )
            self.use_wandb = True
        else:
            self.use_wandb = False

    def _get_total_steps(self) -> int:
        """Calculate total training steps."""
        if self.config.max_steps:
            return self.config.max_steps

        steps_per_epoch = len(self.train_dataloader) // self.config.gradient_accumulation_steps
        return steps_per_epoch * self.config.num_epochs

    def train(self) -> Dict[str, Any]:
        """Run the full training loop."""
        total_steps = self._get_total_steps()
        print(f"\n{'='*60}")
        print(f"Starting training")
        print(f"  Total steps: {total_steps}")
        print(f"  Epochs: {self.config.num_epochs}")
        print(f"  Batch size: {self.config.batch_size}")
        print(f"  Gradient accumulation: {self.config.gradient_accumulation_steps}")
        print(f"  Effective batch size: {self.config.effective_batch_size}")
        print(f"  Learning rate: {self.config.learning_rate}")
        print(f"  Device: {self.device}")
        print(f"  Mixed precision: {self.config.mixed_precision} ({'bf16' if self.config.bf16 else 'fp16'})")
        print(f"{'='*60}\n")

        self.model.train()
        training_start = time.time()

        for epoch in range(self.epoch, self.config.num_epochs):
            self.epoch = epoch
            epoch_loss = self._train_epoch()

            print(f"\nEpoch {epoch + 1}/{self.config.num_epochs} completed")
            print(f"  Average loss: {epoch_loss:.4f}")

            if self.config.max_steps and self.global_step >= self.config.max_steps:
                break

        # Final save
        self.save_checkpoint("final")

        training_time = time.time() - training_start
        print(f"\nTraining completed in {training_time/3600:.2f} hours")

        return {
            "total_steps": self.global_step,
            "final_loss": epoch_loss,
            "training_time": training_time,
        }

    def _train_epoch(self) -> float:
        """Train for one epoch."""
        total_loss = 0.0
        num_batches = 0
        accumulated_loss = 0.0

        self.optimizer.zero_grad()

        for step, batch in enumerate(self.train_dataloader):
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}

            # Forward pass with optional mixed precision
            if self.config.mixed_precision:
                with autocast(device_type='cuda', dtype=self.autocast_dtype):
                    outputs = self.model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch.get("attention_mask"),
                        labels=batch["labels"],
                    )
                    loss = outputs["loss"] / self.config.gradient_accumulation_steps
            else:
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch.get("attention_mask"),
                    labels=batch["labels"],
                )
                loss = outputs["loss"] / self.config.gradient_accumulation_steps

            # Backward pass
            if self.config.mixed_precision:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            accumulated_loss += loss.item()

            # Gradient accumulation step
            if (step + 1) % self.config.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.config.mixed_precision:
                    self.scaler.unscale_(self.optimizer)

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm,
                )

                # Optimizer step
                if self.config.mixed_precision:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.scheduler.step()
                self.optimizer.zero_grad()

                self.global_step += 1
                total_loss += accumulated_loss
                num_batches += 1

                # Logging
                if self.global_step % self.config.logging_steps == 0:
                    self._log_step(accumulated_loss)

                # Evaluation
                if (
                    self.eval_dataloader is not None and
                    self.global_step % self.config.eval_steps == 0
                ):
                    eval_metrics = self.evaluate()
                    self._log_eval(eval_metrics)
                    self.model.train()

                # Checkpointing
                if self.global_step % self.config.save_steps == 0:
                    self.save_checkpoint(f"step_{self.global_step}")

                accumulated_loss = 0.0

                # Check if we've reached max steps
                if self.config.max_steps and self.global_step >= self.config.max_steps:
                    break

        return total_loss / max(num_batches, 1)

    def _log_step(self, loss: float):
        """Log training step metrics."""
        lr = self.scheduler.get_last_lr()[0]
        log_data = {
            "train/loss": loss,
            "train/learning_rate": lr,
            "train/epoch": self.epoch,
            "train/step": self.global_step,
        }

        if self.use_wandb:
            wandb.log(log_data, step=self.global_step)

        print(
            f"Step {self.global_step} | Loss: {loss:.4f} | LR: {lr:.2e}"
        )

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Run evaluation on the eval dataset."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in self.eval_dataloader:
            batch = {k: v.to(self.device) for k, v in batch.items()}

            if self.config.mixed_precision:
                with autocast(device_type='cuda', dtype=self.autocast_dtype):
                    outputs = self.model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch.get("attention_mask"),
                        labels=batch["labels"],
                    )
            else:
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch.get("attention_mask"),
                    labels=batch["labels"],
                )

            total_loss += outputs["loss"].item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        perplexity = math.exp(avg_loss)

        metrics = {
            "eval/loss": avg_loss,
            "eval/perplexity": perplexity,
        }

        # Compute additional metrics if provided
        if self.compute_metrics:
            additional = self.compute_metrics(self.model, self.eval_dataloader)
            metrics.update(additional)

        return metrics

    def _log_eval(self, metrics: Dict[str, float]):
        """Log evaluation metrics."""
        if self.use_wandb:
            wandb.log(metrics, step=self.global_step)

        print(f"\nEvaluation at step {self.global_step}:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")
        print()

        # Track best model
        eval_loss = metrics.get("eval/loss", float("inf"))
        if eval_loss < self.best_eval_loss:
            self.best_eval_loss = eval_loss
            self.save_checkpoint("best")

    def save_checkpoint(self, name: str):
        """Save a training checkpoint."""
        checkpoint_dir = self.output_dir / name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        torch.save(
            self.model.state_dict(),
            checkpoint_dir / "model.pt",
        )

        # Save optimizer and scheduler
        torch.save(
            {
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "scaler": self.scaler.state_dict() if self.scaler else None,
            },
            checkpoint_dir / "optimizer.pt",
        )

        # Save training state
        state = {
            "global_step": self.global_step,
            "epoch": self.epoch,
            "best_eval_loss": self.best_eval_loss,
            "config": self.config.__dict__,
        }
        with open(checkpoint_dir / "training_state.json", "w") as f:
            json.dump(state, f, indent=2)

        print(f"Saved checkpoint: {checkpoint_dir}")

        # Cleanup old checkpoints
        self._cleanup_checkpoints()

    def load_checkpoint(self, path: str):
        """Load a training checkpoint."""
        checkpoint_dir = Path(path)

        # Load model
        model_path = checkpoint_dir / "model.pt"
        if model_path.exists():
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))

        # Load optimizer and scheduler
        optimizer_path = checkpoint_dir / "optimizer.pt"
        if optimizer_path.exists():
            state = torch.load(optimizer_path, map_location=self.device)
            self.optimizer.load_state_dict(state["optimizer"])
            self.scheduler.load_state_dict(state["scheduler"])
            if self.scaler and state.get("scaler"):
                self.scaler.load_state_dict(state["scaler"])

        # Load training state
        state_path = checkpoint_dir / "training_state.json"
        if state_path.exists():
            with open(state_path) as f:
                state = json.load(f)
            self.global_step = state["global_step"]
            self.epoch = state["epoch"]
            self.best_eval_loss = state.get("best_eval_loss", float("inf"))

        print(f"Resumed from checkpoint: {checkpoint_dir}")
        print(f"  Step: {self.global_step}, Epoch: {self.epoch}")

    def _cleanup_checkpoints(self):
        """Remove old checkpoints, keeping only the most recent ones."""
        if self.config.save_total_limit <= 0:
            return

        checkpoints = []
        for path in self.output_dir.iterdir():
            if path.is_dir() and path.name.startswith("step_"):
                try:
                    step = int(path.name.split("_")[1])
                    checkpoints.append((step, path))
                except ValueError:
                    continue

        # Sort by step and remove old ones
        checkpoints.sort(key=lambda x: x[0], reverse=True)
        for _, path in checkpoints[self.config.save_total_limit:]:
            import shutil
            shutil.rmtree(path)
