"""
Pipeline Runner

Orchestrates the full training pipeline with validation gates.
Manages the train -> validate -> scale cycle.

Usage:
    from src.pipeline import PipelineRunner, PipelineConfig

    config = PipelineConfig(...)
    runner = PipelineRunner(config)

    # Run full pipeline
    results = runner.run()

    # Or run individual stages
    runner.validate_infrastructure()
    runner.train_model()
    runner.evaluate()
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Callable
from pathlib import Path
import json
import time
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import (
    PipelineConfig,
    TrainingConfig,
    ModelScale,
    get_model_config,
    SCALE_CONFIGS,
)
from .validation import PipelineValidator, ValidationLevel, ValidationResult
from .checkpoints import CheckpointManager


@dataclass
class PipelineResult:
    """Result of a pipeline run."""

    success: bool
    scale: ModelScale
    run_id: str

    # Training results
    final_step: int = 0
    final_loss: float = 0.0
    training_time_seconds: float = 0.0

    # Validation results
    validation_results: List[ValidationResult] = None

    # Metrics
    metrics: Dict[str, float] = None

    # Errors
    error: Optional[str] = None

    def __post_init__(self):
        if self.validation_results is None:
            self.validation_results = []
        if self.metrics is None:
            self.metrics = {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "scale": self.scale.value,
            "run_id": self.run_id,
            "final_step": self.final_step,
            "final_loss": self.final_loss,
            "training_time_seconds": self.training_time_seconds,
            "metrics": self.metrics,
            "error": self.error,
            "validation_results": [v.to_dict() for v in self.validation_results],
        }


class PipelineRunner:
    """
    Orchestrates the training pipeline.

    Handles:
    - Infrastructure validation
    - Model creation at specified scale
    - Training with validation gates
    - Checkpoint management
    - Evaluation and benchmarking
    - Scale-up progression
    """

    def __init__(
        self,
        config: PipelineConfig,
        model_factory: Optional[Callable[[ModelScale], nn.Module]] = None,
        data_factory: Optional[Callable[[TrainingConfig], tuple]] = None,
    ):
        """
        Initialize pipeline runner.

        Args:
            config: Pipeline configuration
            model_factory: Function to create model for a given scale
            data_factory: Function to create train/eval dataloaders
        """
        self.config = config
        self.model_factory = model_factory or self._default_model_factory
        self.data_factory = data_factory or self._default_data_factory

        # Components
        self.model: Optional[nn.Module] = None
        self.train_dataloader: Optional[DataLoader] = None
        self.eval_dataloader: Optional[DataLoader] = None

        # Managers
        self.checkpoint_manager: Optional[CheckpointManager] = None
        self.validator: Optional[PipelineValidator] = None

        # State
        self.current_step = 0
        self.current_epoch = 0
        self.run_results: List[PipelineResult] = []

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Setup directories
        self._setup_directories()

    def _setup_directories(self):
        """Create necessary directories."""
        Path(self.config.base_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.data_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.eval_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.log_dir).mkdir(parents=True, exist_ok=True)

    def _default_model_factory(self, scale: ModelScale) -> nn.Module:
        """Default model factory using built-in configs."""
        from ..models.transformer import ReasoningTransformer
        from ..models.config import TransformerConfig

        scale_config = get_model_config(scale)

        # Convert to transformer config
        model_config = TransformerConfig(
            name=scale_config.name,
            vocab_size=scale_config.vocab_size,
            hidden_size=scale_config.hidden_size,
            intermediate_size=scale_config.intermediate_size,
            num_hidden_layers=scale_config.num_hidden_layers,
            num_attention_heads=scale_config.num_attention_heads,
            num_key_value_heads=scale_config.num_key_value_heads,
            max_position_embeddings=scale_config.max_position_embeddings,
            tool_calling=scale_config.supports_tools,
            gradient_checkpointing=self.config.training.gradient_checkpointing,
        )

        return ReasoningTransformer(model_config)

    def _default_data_factory(
        self,
        training_config: TrainingConfig,
    ) -> tuple:
        """Default data factory using built-in datasets."""
        from ..data.datasets import create_reasoning_dataloader

        train_dataloader = create_reasoning_dataloader(
            sources=training_config.data.sources,
            batch_size=training_config.batch_size,
            max_length=training_config.data.max_seq_length,
            num_workers=training_config.data.num_workers,
            split="train",
        )

        eval_dataloader = create_reasoning_dataloader(
            sources=training_config.data.sources,
            batch_size=training_config.batch_size,
            max_length=training_config.data.max_seq_length,
            num_workers=training_config.data.num_workers,
            split="validation",
        )

        return train_dataloader, eval_dataloader

    def validate_infrastructure(self) -> bool:
        """
        Validate that all infrastructure is working.

        Checks:
        - GPU availability and memory
        - Data loading
        - Model creation
        - Forward/backward pass
        - Checkpoint saving/loading
        """
        print("\n" + "=" * 60)
        print("Infrastructure Validation")
        print("=" * 60)

        checks_passed = 0
        checks_total = 0

        # Check 1: Device
        checks_total += 1
        print(f"\n[1] Device check...")
        print(f"    Device: {self.device}")
        if torch.cuda.is_available():
            print(f"    GPU: {torch.cuda.get_device_name()}")
            print(f"    Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            checks_passed += 1
            print("    PASSED")
        else:
            print("    WARNING: No GPU available, using CPU")
            checks_passed += 1  # Still pass, just slower
            print("    PASSED (CPU mode)")

        # Check 2: Model creation
        checks_total += 1
        print(f"\n[2] Model creation check...")
        try:
            test_scale = ModelScale.TINY
            model = self.model_factory(test_scale)
            num_params = sum(p.numel() for p in model.parameters())
            print(f"    Created {test_scale.value} model: {num_params/1e6:.1f}M parameters")
            model.to(self.device)
            checks_passed += 1
            print("    PASSED")
        except Exception as e:
            print(f"    FAILED: {e}")
            return False

        # Check 3: Forward pass
        checks_total += 1
        print(f"\n[3] Forward pass check...")
        try:
            batch_size = 2
            seq_len = 128
            dummy_input = torch.randint(0, 1000, (batch_size, seq_len), device=self.device)

            with torch.no_grad():
                outputs = model(dummy_input)

            if isinstance(outputs, dict):
                print(f"    Output keys: {list(outputs.keys())}")
            else:
                print(f"    Output shape: {outputs.shape}")

            checks_passed += 1
            print("    PASSED")
        except Exception as e:
            print(f"    FAILED: {e}")
            return False

        # Check 4: Backward pass
        checks_total += 1
        print(f"\n[4] Backward pass check...")
        try:
            dummy_input = torch.randint(0, 1000, (batch_size, seq_len), device=self.device)
            dummy_labels = torch.randint(0, 1000, (batch_size, seq_len), device=self.device)

            outputs = model(dummy_input, labels=dummy_labels)
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs.mean()
            loss.backward()

            # Check gradients exist
            has_grads = any(p.grad is not None for p in model.parameters())
            print(f"    Loss: {loss.item():.4f}")
            print(f"    Gradients computed: {has_grads}")

            checks_passed += 1
            print("    PASSED")
        except Exception as e:
            print(f"    FAILED: {e}")
            return False

        # Check 5: Checkpoint save/load
        checks_total += 1
        print(f"\n[5] Checkpoint save/load check...")
        try:
            import tempfile
            with tempfile.TemporaryDirectory() as tmpdir:
                manager = CheckpointManager(tmpdir, save_total_limit=2)

                # Save
                checkpoint_id = manager.save(
                    model=model,
                    step=100,
                    metrics={"loss": loss.item()},
                )

                # Load
                step, epoch, _ = manager.load(checkpoint_id, model)
                print(f"    Saved and loaded checkpoint: step={step}")

                checks_passed += 1
                print("    PASSED")
        except Exception as e:
            print(f"    FAILED: {e}")
            return False

        # Summary
        print("\n" + "=" * 60)
        print(f"Infrastructure Validation: {checks_passed}/{checks_total} checks passed")
        print("=" * 60)

        if checks_passed == checks_total:
            print("All checks PASSED - ready for training")
            return True
        else:
            print("Some checks FAILED - please fix issues before training")
            return False

    def run(self) -> PipelineResult:
        """
        Run the full training pipeline.

        Returns:
            PipelineResult with training outcomes
        """
        training_config = self.config.training
        scale = training_config.model_scale

        print("\n" + "=" * 60)
        print(f"Starting Pipeline Run: {training_config.run_id}")
        print(f"Scale: {scale.value}")
        print("=" * 60)

        start_time = time.time()

        try:
            # Step 1: Validate infrastructure
            print("\n--- Step 1: Infrastructure Validation ---")
            if not self.validate_infrastructure():
                return PipelineResult(
                    success=False,
                    scale=scale,
                    run_id=training_config.run_id,
                    error="Infrastructure validation failed",
                )

            # Step 2: Create model
            print("\n--- Step 2: Creating Model ---")
            self.model = self.model_factory(scale)
            self.model.to(self.device)

            num_params = sum(p.numel() for p in self.model.parameters())
            print(f"Model created: {num_params/1e6:.1f}M parameters")

            # Step 3: Create data loaders
            print("\n--- Step 3: Loading Data ---")
            self.train_dataloader, self.eval_dataloader = self.data_factory(training_config)
            print(f"Train batches: {len(self.train_dataloader)}")
            print(f"Eval batches: {len(self.eval_dataloader)}")

            # Step 4: Setup managers
            print("\n--- Step 4: Setting Up Managers ---")
            checkpoint_dir = Path(self.config.checkpoint_dir) / training_config.run_id
            self.checkpoint_manager = CheckpointManager(
                output_dir=str(checkpoint_dir),
                run_id=training_config.run_id,
                save_total_limit=training_config.save_total_limit,
            )

            self.validator = PipelineValidator(
                config=self.config,
                strict=self.config.strict_validation,
            )

            # Step 5: Train
            print("\n--- Step 5: Training ---")
            train_result = self._train()

            # Step 6: Final validation
            print("\n--- Step 6: Final Validation ---")
            final_validation = self.validator.validate_full(
                self.model,
                self.eval_dataloader,
            )

            # Build result
            training_time = time.time() - start_time

            result = PipelineResult(
                success=final_validation.passed,
                scale=scale,
                run_id=training_config.run_id,
                final_step=self.current_step,
                final_loss=train_result.get("final_loss", 0.0),
                training_time_seconds=training_time,
                validation_results=self.validator.validation_history,
                metrics=final_validation.metrics,
            )

            # Save result
            self._save_result(result)

            print("\n" + "=" * 60)
            print(f"Pipeline Run Complete: {'SUCCESS' if result.success else 'FAILED'}")
            print(f"Final step: {result.final_step}")
            print(f"Final loss: {result.final_loss:.4f}")
            print(f"Training time: {training_time/3600:.2f} hours")
            print("=" * 60)

            return result

        except Exception as e:
            import traceback
            error_msg = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
            print(f"\nPipeline failed with error:\n{error_msg}")

            return PipelineResult(
                success=False,
                scale=scale,
                run_id=training_config.run_id,
                training_time_seconds=time.time() - start_time,
                error=error_msg,
            )

    def _train(self) -> Dict[str, Any]:
        """Run the training loop."""
        from ..training.trainer import Trainer, TrainingConfig as LegacyTrainingConfig

        # Convert to trainer config
        trainer_config = LegacyTrainingConfig(
            num_epochs=self.config.training.num_epochs,
            max_steps=self.config.training.max_steps,
            learning_rate=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
            max_grad_norm=self.config.training.max_grad_norm,
            batch_size=self.config.training.batch_size,
            gradient_accumulation_steps=self.config.training.gradient_accumulation_steps,
            warmup_ratio=self.config.training.warmup_ratio,
            lr_scheduler=self.config.training.lr_scheduler,
            mixed_precision=self.config.training.mixed_precision,
            bf16=self.config.training.bf16,
            output_dir=str(Path(self.config.checkpoint_dir) / self.config.training.run_id),
            save_steps=self.config.training.save_steps,
            save_total_limit=self.config.training.save_total_limit,
            logging_steps=self.config.training.logging_steps,
            eval_steps=self.config.training.eval_steps,
            use_wandb=self.config.training.use_wandb,
            wandb_project=self.config.training.wandb_project,
            wandb_run_name=self.config.training.run_id,
            gradient_checkpointing=self.config.training.gradient_checkpointing,
        )

        # Create trainer with validation hooks
        trainer = Trainer(
            model=self.model,
            config=trainer_config,
            train_dataloader=self.train_dataloader,
            eval_dataloader=self.eval_dataloader,
        )

        # Add validation callback
        original_log_step = trainer._log_step

        def log_step_with_validation(loss):
            original_log_step(loss)

            # Quick validation every N steps
            if trainer.global_step % 100 == 0:
                self.validator.validate_quick(
                    self.model,
                    loss=loss,
                    step=trainer.global_step,
                )

        trainer._log_step = log_step_with_validation

        # Run training
        result = trainer.train()

        self.current_step = trainer.global_step
        self.current_epoch = trainer.epoch

        return result

    def _save_result(self, result: PipelineResult):
        """Save pipeline result to file."""
        result_path = Path(self.config.log_dir) / f"{result.run_id}_result.json"

        with open(result_path, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)

        print(f"Saved result to {result_path}")

    def run_progressive(
        self,
        start_scale: ModelScale = ModelScale.TINY,
        end_scale: ModelScale = ModelScale.FLAGSHIP,
        validation_between_scales: bool = True,
    ) -> List[PipelineResult]:
        """
        Run progressive training from small to large scales.

        Trains at each scale, validates, and only proceeds
        to the next scale if validation passes.

        Args:
            start_scale: Scale to start at
            end_scale: Maximum scale to train
            validation_between_scales: Whether to validate before scaling up

        Returns:
            List of results for each scale
        """
        scales = list(ModelScale)
        start_idx = scales.index(start_scale)
        end_idx = scales.index(end_scale)

        results = []

        for scale in scales[start_idx:end_idx + 1]:
            print("\n" + "#" * 60)
            print(f"# Progressive Training: {scale.value}")
            print("#" * 60)

            # Update config for this scale
            self.config.training.model_scale = scale
            self.config.training.apply_scale_defaults()
            self.config.training.run_id = f"{self.config.training.experiment_name}_{scale.value}"

            # Run training
            result = self.run()
            results.append(result)

            # Check if we should continue
            if not result.success:
                print(f"\nTraining at {scale.value} failed, stopping progression")
                break

            if validation_between_scales and scale != end_scale:
                print(f"\n{scale.value} training complete, validating before scale-up...")

                # Full validation
                validation = self.validator.validate_full(
                    self.model,
                    self.eval_dataloader,
                )

                if not validation.passed:
                    print(f"Validation failed at {scale.value}, stopping progression")
                    break

                print(f"Validation passed, proceeding to next scale")

        return results


def quick_test() -> bool:
    """
    Run a quick infrastructure test.

    Use this to verify everything is working before starting training.
    """
    from .config import create_quick_test_config

    config = create_quick_test_config()
    runner = PipelineRunner(config)

    return runner.validate_infrastructure()


def test_scale(scale: ModelScale) -> PipelineResult:
    """
    Test training at a specific scale.

    Runs a short training run to validate the scale works.
    """
    from .config import create_scale_test_config

    config = create_scale_test_config(scale)
    runner = PipelineRunner(config)

    return runner.run()
