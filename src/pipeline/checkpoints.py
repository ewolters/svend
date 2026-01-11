"""
Checkpoint Management System

Handles saving, loading, and managing model checkpoints.
Designed for Colab training with Drive persistence.

Features:
- Automatic checkpointing with configurable frequency
- Best model tracking
- Checkpoint cleanup (keep N most recent)
- Resume from any checkpoint
- Metadata tracking for reproducibility

Usage:
    manager = CheckpointManager(output_dir="checkpoints/my-run")

    # During training
    manager.save(model, optimizer, scheduler, step=1000, metrics={"loss": 0.5})

    # Resume training
    state = manager.load_latest(model, optimizer, scheduler)

    # Load best model for inference
    manager.load_best(model)
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
import json
import shutil
import time
from datetime import datetime
import hashlib

import torch
import torch.nn as nn


@dataclass
class CheckpointMetadata:
    """Metadata for a checkpoint."""

    # Identity
    checkpoint_id: str
    run_id: str
    step: int
    epoch: int

    # Timing
    timestamp: float
    training_time_seconds: float

    # Metrics
    metrics: Dict[str, float] = field(default_factory=dict)

    # Model info
    model_config: Dict[str, Any] = field(default_factory=dict)
    num_parameters: int = 0

    # Validation
    validation_passed: bool = True
    validation_result: Optional[Dict[str, Any]] = None

    # Files
    files: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CheckpointMetadata":
        return cls(**data)

    def save(self, path: Path):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "CheckpointMetadata":
        with open(path) as f:
            return cls.from_dict(json.load(f))


class CheckpointManager:
    """
    Manages model checkpoints during training.

    Handles:
    - Regular checkpoint saving
    - Best model tracking (by any metric)
    - Checkpoint cleanup
    - Resume from checkpoint
    - Google Drive persistence for Colab
    """

    def __init__(
        self,
        output_dir: str,
        run_id: Optional[str] = None,
        save_total_limit: int = 3,
        best_metric: str = "loss",
        best_metric_mode: str = "min",  # "min" or "max"
    ):
        """
        Initialize checkpoint manager.

        Args:
            output_dir: Base directory for checkpoints
            run_id: Unique identifier for this run
            save_total_limit: Maximum checkpoints to keep (0 = unlimited)
            best_metric: Metric to track for best model
            best_metric_mode: "min" to minimize, "max" to maximize
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.run_id = run_id or self._generate_run_id()
        self.save_total_limit = save_total_limit
        self.best_metric = best_metric
        self.best_metric_mode = best_metric_mode

        # State
        self.checkpoints: List[CheckpointMetadata] = []
        self.best_value: Optional[float] = None
        self.best_checkpoint_id: Optional[str] = None
        self.training_start_time: float = time.time()

        # Load existing checkpoints if resuming
        self._load_checkpoint_registry()

    def _generate_run_id(self) -> str:
        """Generate a unique run ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        hash_suffix = hashlib.md5(str(time.time()).encode()).hexdigest()[:6]
        return f"run_{timestamp}_{hash_suffix}"

    def _load_checkpoint_registry(self):
        """Load registry of existing checkpoints."""
        registry_path = self.output_dir / "checkpoint_registry.json"

        if registry_path.exists():
            with open(registry_path) as f:
                data = json.load(f)

            self.checkpoints = [
                CheckpointMetadata.from_dict(c) for c in data.get("checkpoints", [])
            ]
            self.best_value = data.get("best_value")
            self.best_checkpoint_id = data.get("best_checkpoint_id")

            print(f"Loaded {len(self.checkpoints)} existing checkpoints")
            if self.best_checkpoint_id:
                print(f"Best checkpoint: {self.best_checkpoint_id} ({self.best_metric}={self.best_value})")

    def _save_checkpoint_registry(self):
        """Save registry of checkpoints."""
        registry_path = self.output_dir / "checkpoint_registry.json"

        data = {
            "run_id": self.run_id,
            "checkpoints": [c.to_dict() for c in self.checkpoints],
            "best_value": self.best_value,
            "best_checkpoint_id": self.best_checkpoint_id,
            "best_metric": self.best_metric,
            "best_metric_mode": self.best_metric_mode,
        }

        with open(registry_path, 'w') as f:
            json.dump(data, f, indent=2)

    def save(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        scaler: Optional[Any] = None,
        step: int = 0,
        epoch: int = 0,
        metrics: Optional[Dict[str, float]] = None,
        model_config: Optional[Dict[str, Any]] = None,
        validation_result: Optional[Any] = None,
        extra_state: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Save a checkpoint.

        Args:
            model: The model to save
            optimizer: Optional optimizer state
            scheduler: Optional LR scheduler state
            scaler: Optional gradient scaler state (for AMP)
            step: Current training step
            epoch: Current epoch
            metrics: Current metrics (loss, accuracy, etc.)
            model_config: Model configuration for reproducibility
            validation_result: Validation result if available
            extra_state: Any additional state to save

        Returns:
            Checkpoint ID
        """
        metrics = metrics or {}
        model_config = model_config or {}
        extra_state = extra_state or {}

        # Generate checkpoint ID
        checkpoint_id = f"step_{step:08d}"
        checkpoint_dir = self.output_dir / checkpoint_id
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        files = []

        # Save model
        model_path = checkpoint_dir / "model.pt"
        torch.save(model.state_dict(), model_path)
        files.append("model.pt")

        # Save optimizer
        if optimizer is not None:
            optimizer_path = checkpoint_dir / "optimizer.pt"
            torch.save(optimizer.state_dict(), optimizer_path)
            files.append("optimizer.pt")

        # Save scheduler
        if scheduler is not None:
            scheduler_path = checkpoint_dir / "scheduler.pt"
            torch.save(scheduler.state_dict(), scheduler_path)
            files.append("scheduler.pt")

        # Save scaler
        if scaler is not None:
            scaler_path = checkpoint_dir / "scaler.pt"
            torch.save(scaler.state_dict(), scaler_path)
            files.append("scaler.pt")

        # Save extra state
        if extra_state:
            extra_path = checkpoint_dir / "extra_state.pt"
            torch.save(extra_state, extra_path)
            files.append("extra_state.pt")

        # Count parameters
        num_params = sum(p.numel() for p in model.parameters())

        # Create metadata
        metadata = CheckpointMetadata(
            checkpoint_id=checkpoint_id,
            run_id=self.run_id,
            step=step,
            epoch=epoch,
            timestamp=time.time(),
            training_time_seconds=time.time() - self.training_start_time,
            metrics=metrics,
            model_config=model_config,
            num_parameters=num_params,
            validation_passed=validation_result.passed if validation_result else True,
            validation_result=validation_result.to_dict() if validation_result else None,
            files=files,
        )

        # Save metadata
        metadata.save(checkpoint_dir / "metadata.json")

        # Update registry
        self.checkpoints.append(metadata)

        # Check if this is the best model
        if self.best_metric in metrics:
            metric_value = metrics[self.best_metric]
            is_better = False

            if self.best_value is None:
                is_better = True
            elif self.best_metric_mode == "min":
                is_better = metric_value < self.best_value
            else:
                is_better = metric_value > self.best_value

            if is_better:
                self.best_value = metric_value
                self.best_checkpoint_id = checkpoint_id

                # Create/update symlink to best
                best_link = self.output_dir / "best"
                if best_link.exists():
                    if best_link.is_symlink():
                        best_link.unlink()
                    else:
                        shutil.rmtree(best_link)

                # Copy best checkpoint (symlinks can be fragile on some systems)
                shutil.copytree(checkpoint_dir, best_link)
                print(f"New best model! {self.best_metric}={metric_value:.4f}")

        # Save registry
        self._save_checkpoint_registry()

        # Cleanup old checkpoints
        self._cleanup_old_checkpoints()

        print(f"Saved checkpoint: {checkpoint_id}")
        return checkpoint_id

    def load(
        self,
        checkpoint_id: str,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        scaler: Optional[Any] = None,
        device: Optional[torch.device] = None,
    ) -> Tuple[int, int, Dict[str, Any]]:
        """
        Load a specific checkpoint.

        Args:
            checkpoint_id: ID of checkpoint to load
            model: Model to load weights into
            optimizer: Optional optimizer to load state into
            scheduler: Optional scheduler to load state into
            scaler: Optional scaler to load state into
            device: Device to load to

        Returns:
            Tuple of (step, epoch, extra_state)
        """
        checkpoint_dir = self.output_dir / checkpoint_id

        if not checkpoint_dir.exists():
            raise ValueError(f"Checkpoint not found: {checkpoint_id}")

        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model
        model_path = checkpoint_dir / "model.pt"
        model.load_state_dict(torch.load(model_path, map_location=device))

        # Load optimizer
        if optimizer is not None and (checkpoint_dir / "optimizer.pt").exists():
            optimizer.load_state_dict(
                torch.load(checkpoint_dir / "optimizer.pt", map_location=device)
            )

        # Load scheduler
        if scheduler is not None and (checkpoint_dir / "scheduler.pt").exists():
            scheduler.load_state_dict(
                torch.load(checkpoint_dir / "scheduler.pt", map_location=device)
            )

        # Load scaler
        if scaler is not None and (checkpoint_dir / "scaler.pt").exists():
            scaler.load_state_dict(
                torch.load(checkpoint_dir / "scaler.pt", map_location=device)
            )

        # Load metadata
        metadata = CheckpointMetadata.load(checkpoint_dir / "metadata.json")

        # Load extra state
        extra_state = {}
        if (checkpoint_dir / "extra_state.pt").exists():
            extra_state = torch.load(checkpoint_dir / "extra_state.pt", map_location=device)

        print(f"Loaded checkpoint: {checkpoint_id} (step={metadata.step}, epoch={metadata.epoch})")

        return metadata.step, metadata.epoch, extra_state

    def load_latest(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        scaler: Optional[Any] = None,
        device: Optional[torch.device] = None,
    ) -> Tuple[int, int, Dict[str, Any]]:
        """Load the most recent checkpoint."""
        if not self.checkpoints:
            raise ValueError("No checkpoints available")

        latest = max(self.checkpoints, key=lambda c: c.step)
        return self.load(latest.checkpoint_id, model, optimizer, scheduler, scaler, device)

    def load_best(
        self,
        model: nn.Module,
        device: Optional[torch.device] = None,
    ) -> Tuple[int, int, Dict[str, Any]]:
        """Load the best checkpoint (by tracked metric)."""
        if self.best_checkpoint_id is None:
            raise ValueError("No best checkpoint available")

        return self.load(self.best_checkpoint_id, model, device=device)

    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints, keeping only the most recent ones."""
        if self.save_total_limit <= 0:
            return

        # Sort by step, newest first
        sorted_checkpoints = sorted(self.checkpoints, key=lambda c: c.step, reverse=True)

        # Keep the N most recent, plus best
        to_keep = set()

        for checkpoint in sorted_checkpoints[:self.save_total_limit]:
            to_keep.add(checkpoint.checkpoint_id)

        if self.best_checkpoint_id:
            to_keep.add(self.best_checkpoint_id)

        # Remove old checkpoints
        for checkpoint in self.checkpoints:
            if checkpoint.checkpoint_id not in to_keep:
                checkpoint_dir = self.output_dir / checkpoint.checkpoint_id
                if checkpoint_dir.exists():
                    shutil.rmtree(checkpoint_dir)
                    print(f"Removed old checkpoint: {checkpoint.checkpoint_id}")

        # Update registry
        self.checkpoints = [c for c in self.checkpoints if c.checkpoint_id in to_keep]
        self._save_checkpoint_registry()

    def list_checkpoints(self) -> List[CheckpointMetadata]:
        """List all available checkpoints."""
        return sorted(self.checkpoints, key=lambda c: c.step)

    def get_checkpoint_info(self, checkpoint_id: str) -> Optional[CheckpointMetadata]:
        """Get metadata for a specific checkpoint."""
        for c in self.checkpoints:
            if c.checkpoint_id == checkpoint_id:
                return c
        return None

    def print_summary(self):
        """Print summary of checkpoints."""
        print("\n" + "=" * 60)
        print("Checkpoint Summary")
        print("=" * 60)
        print(f"Output directory: {self.output_dir}")
        print(f"Run ID: {self.run_id}")
        print(f"Total checkpoints: {len(self.checkpoints)}")
        print(f"Best metric: {self.best_metric} ({self.best_metric_mode})")
        print(f"Best value: {self.best_value}")
        print(f"Best checkpoint: {self.best_checkpoint_id}")
        print()

        if self.checkpoints:
            print("Available checkpoints:")
            for c in sorted(self.checkpoints, key=lambda x: x.step):
                is_best = " (BEST)" if c.checkpoint_id == self.best_checkpoint_id else ""
                metrics_str = ", ".join(f"{k}={v:.4f}" for k, v in c.metrics.items())
                print(f"  {c.checkpoint_id}: step={c.step}, {metrics_str}{is_best}")

        print("=" * 60)


class DriveCheckpointManager(CheckpointManager):
    """
    Checkpoint manager optimized for Google Colab with Drive.

    Automatically mounts Drive and saves checkpoints there
    for persistence across session restarts.
    """

    def __init__(
        self,
        drive_path: str = "/content/drive/MyDrive/svend-checkpoints",
        local_cache: str = "/content/checkpoints",
        **kwargs,
    ):
        """
        Initialize Drive-backed checkpoint manager.

        Args:
            drive_path: Path on Google Drive for checkpoints
            local_cache: Local cache directory for faster access
            **kwargs: Additional arguments for CheckpointManager
        """
        self.drive_path = Path(drive_path)
        self.local_cache = Path(local_cache)

        # Mount drive if in Colab
        self._mount_drive()

        # Use drive path as output dir
        super().__init__(output_dir=str(self.drive_path), **kwargs)

    def _mount_drive(self):
        """Mount Google Drive if in Colab environment."""
        try:
            from google.colab import drive
            drive.mount('/content/drive')
            print("Google Drive mounted successfully")

            # Create checkpoint directory on Drive
            self.drive_path.mkdir(parents=True, exist_ok=True)
        except ImportError:
            print("Not in Colab environment, using local storage")
        except Exception as e:
            print(f"Warning: Could not mount Drive: {e}")
            print("Using local storage instead")

    def save(self, *args, **kwargs) -> str:
        """Save checkpoint to Drive with local caching."""
        # Save to Drive (via parent class)
        checkpoint_id = super().save(*args, **kwargs)

        # Also keep a local copy for faster access
        if self.local_cache != self.drive_path:
            src = self.output_dir / checkpoint_id
            dst = self.local_cache / checkpoint_id
            if src.exists():
                dst.mkdir(parents=True, exist_ok=True)
                for file in src.iterdir():
                    shutil.copy2(file, dst / file.name)

        return checkpoint_id
