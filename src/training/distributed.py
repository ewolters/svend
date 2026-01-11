"""
Distributed training support for multi-GPU and multi-node training.

Supports:
- PyTorch DDP (Distributed Data Parallel)
- FSDP (Fully Sharded Data Parallel) for large models
- DeepSpeed integration for ZeRO optimization
- Future: Raspberry Pi cluster for inference distribution
"""

import os
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

try:
    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
        MixedPrecision,
        BackwardPrefetch,
        ShardingStrategy,
        CPUOffload,
    )
    from torch.distributed.fsdp.wrap import (
        size_based_auto_wrap_policy,
        transformer_auto_wrap_policy,
    )
    FSDP_AVAILABLE = True
except ImportError:
    FSDP_AVAILABLE = False

try:
    import deepspeed
    DEEPSPEED_AVAILABLE = True
except ImportError:
    DEEPSPEED_AVAILABLE = False


@dataclass
class DistributedConfig:
    """Configuration for distributed training."""

    # Strategy
    strategy: str = "ddp"  # "ddp", "fsdp", "deepspeed"

    # Basic distributed
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0
    backend: str = "nccl"  # "nccl" for GPU, "gloo" for CPU

    # FSDP settings
    fsdp_sharding_strategy: str = "full_shard"  # "full_shard", "shard_grad_op", "no_shard"
    fsdp_cpu_offload: bool = False
    fsdp_backward_prefetch: str = "backward_pre"
    fsdp_min_params: int = 100_000_000  # Min params to shard a layer

    # DeepSpeed settings
    deepspeed_stage: int = 2  # ZeRO stage: 1, 2, or 3
    deepspeed_offload: bool = False
    deepspeed_config_path: Optional[str] = None

    # Mixed precision
    mixed_precision: bool = True
    bf16: bool = True

    # Gradient settings
    gradient_accumulation_steps: int = 1
    gradient_clipping: float = 1.0

    def __post_init__(self):
        # Auto-detect from environment
        if "WORLD_SIZE" in os.environ:
            self.world_size = int(os.environ["WORLD_SIZE"])
        if "RANK" in os.environ:
            self.rank = int(os.environ["RANK"])
        if "LOCAL_RANK" in os.environ:
            self.local_rank = int(os.environ["LOCAL_RANK"])


def setup_distributed(config: DistributedConfig) -> bool:
    """
    Initialize distributed training environment.

    Returns True if distributed training is active.
    """
    if config.world_size <= 1:
        return False

    # Initialize process group
    if not dist.is_initialized():
        dist.init_process_group(
            backend=config.backend,
            rank=config.rank,
            world_size=config.world_size,
        )

    # Set device for this rank
    if torch.cuda.is_available():
        torch.cuda.set_device(config.local_rank)

    return True


def cleanup_distributed():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def wrap_model_ddp(
    model: nn.Module,
    config: DistributedConfig,
) -> nn.Module:
    """Wrap model with DistributedDataParallel."""
    device = torch.device(f"cuda:{config.local_rank}" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    return DDP(
        model,
        device_ids=[config.local_rank] if torch.cuda.is_available() else None,
        output_device=config.local_rank if torch.cuda.is_available() else None,
        find_unused_parameters=False,
    )


def wrap_model_fsdp(
    model: nn.Module,
    config: DistributedConfig,
    transformer_layer_cls: Optional[type] = None,
) -> nn.Module:
    """Wrap model with Fully Sharded Data Parallel."""
    if not FSDP_AVAILABLE:
        raise ImportError("FSDP requires PyTorch >= 1.12")

    # Sharding strategy
    sharding_strategies = {
        "full_shard": ShardingStrategy.FULL_SHARD,
        "shard_grad_op": ShardingStrategy.SHARD_GRAD_OP,
        "no_shard": ShardingStrategy.NO_SHARD,
    }
    sharding_strategy = sharding_strategies.get(
        config.fsdp_sharding_strategy,
        ShardingStrategy.FULL_SHARD,
    )

    # Mixed precision policy
    if config.mixed_precision:
        mp_policy = MixedPrecision(
            param_dtype=torch.bfloat16 if config.bf16 else torch.float16,
            reduce_dtype=torch.bfloat16 if config.bf16 else torch.float16,
            buffer_dtype=torch.bfloat16 if config.bf16 else torch.float16,
        )
    else:
        mp_policy = None

    # CPU offload
    cpu_offload = CPUOffload(offload_params=True) if config.fsdp_cpu_offload else None

    # Backward prefetch
    prefetch_policies = {
        "backward_pre": BackwardPrefetch.BACKWARD_PRE,
        "backward_post": BackwardPrefetch.BACKWARD_POST,
    }
    backward_prefetch = prefetch_policies.get(
        config.fsdp_backward_prefetch,
        BackwardPrefetch.BACKWARD_PRE,
    )

    # Auto-wrap policy
    if transformer_layer_cls is not None:
        # Wrap at transformer layer boundaries
        from functools import partial
        auto_wrap_policy = partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={transformer_layer_cls},
        )
    else:
        # Wrap based on parameter count
        from functools import partial
        auto_wrap_policy = partial(
            size_based_auto_wrap_policy,
            min_num_params=config.fsdp_min_params,
        )

    model = FSDP(
        model,
        sharding_strategy=sharding_strategy,
        mixed_precision=mp_policy,
        cpu_offload=cpu_offload,
        backward_prefetch=backward_prefetch,
        auto_wrap_policy=auto_wrap_policy,
        device_id=config.local_rank,
    )

    return model


def create_deepspeed_config(config: DistributedConfig) -> Dict[str, Any]:
    """Create DeepSpeed configuration."""
    ds_config = {
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "gradient_clipping": config.gradient_clipping,

        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": "auto",
                "betas": [0.9, 0.95],
                "eps": 1e-8,
                "weight_decay": 0.1,
            }
        },

        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": "auto",
                "warmup_num_steps": "auto",
                "total_num_steps": "auto",
            }
        },

        "fp16": {
            "enabled": config.mixed_precision and not config.bf16,
            "loss_scale": 0,
            "loss_scale_window": 1000,
            "initial_scale_power": 16,
            "hysteresis": 2,
            "min_loss_scale": 1,
        },

        "bf16": {
            "enabled": config.mixed_precision and config.bf16,
        },

        "zero_optimization": {
            "stage": config.deepspeed_stage,
            "offload_optimizer": {
                "device": "cpu" if config.deepspeed_offload else "none",
                "pin_memory": True,
            },
            "offload_param": {
                "device": "cpu" if config.deepspeed_offload and config.deepspeed_stage == 3 else "none",
                "pin_memory": True,
            },
            "overlap_comm": True,
            "contiguous_gradients": True,
            "sub_group_size": 1e9,
            "reduce_bucket_size": "auto",
            "stage3_prefetch_bucket_size": "auto",
            "stage3_param_persistence_threshold": "auto",
            "stage3_max_live_parameters": 1e9,
            "stage3_max_reuse_distance": 1e9,
            "stage3_gather_16bit_weights_on_model_save": True,
        },

        "activation_checkpointing": {
            "partition_activations": True,
            "cpu_checkpointing": False,
            "contiguous_memory_optimization": True,
            "number_checkpoints": None,
            "synchronize_checkpoint_boundary": False,
            "profile": False,
        },
    }

    return ds_config


def wrap_model_deepspeed(
    model: nn.Module,
    config: DistributedConfig,
    optimizer: Optional[torch.optim.Optimizer] = None,
    lr_scheduler=None,
):
    """Wrap model with DeepSpeed."""
    if not DEEPSPEED_AVAILABLE:
        raise ImportError("DeepSpeed is not installed")

    # Load or create config
    if config.deepspeed_config_path and Path(config.deepspeed_config_path).exists():
        import json
        with open(config.deepspeed_config_path) as f:
            ds_config = json.load(f)
    else:
        ds_config = create_deepspeed_config(config)

    # Initialize DeepSpeed
    model, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        config=ds_config,
    )

    return model, optimizer, lr_scheduler


def create_distributed_dataloader(
    dataset,
    batch_size: int,
    config: DistributedConfig,
    shuffle: bool = True,
    **kwargs,
) -> DataLoader:
    """Create a DataLoader with distributed sampling."""
    if config.world_size > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=config.world_size,
            rank=config.rank,
            shuffle=shuffle,
        )
        shuffle = False  # Sampler handles shuffling
    else:
        sampler = None

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        **kwargs,
    )


def all_reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    """Average tensor across all processes."""
    if not dist.is_initialized():
        return tensor

    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= dist.get_world_size()
    return tensor


def all_gather_object(obj: Any) -> list:
    """Gather objects from all processes."""
    if not dist.is_initialized():
        return [obj]

    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, obj)
    return gathered


def is_main_process(config: DistributedConfig) -> bool:
    """Check if this is the main process (rank 0)."""
    return config.rank == 0


def print_rank_0(message: str, config: DistributedConfig):
    """Print only from rank 0."""
    if is_main_process(config):
        print(message)


class DistributedTrainer:
    """
    Trainer with distributed training support.

    Automatically selects the appropriate parallelism strategy
    based on configuration and available resources.
    """

    def __init__(
        self,
        model: nn.Module,
        config: DistributedConfig,
        train_dataloader: DataLoader,
        optimizer: Optional[torch.optim.Optimizer] = None,
        transformer_layer_cls: Optional[type] = None,
    ):
        self.config = config
        self.is_distributed = setup_distributed(config)

        # Wrap model based on strategy
        if self.is_distributed:
            if config.strategy == "fsdp":
                self.model = wrap_model_fsdp(model, config, transformer_layer_cls)
            elif config.strategy == "deepspeed":
                self.model, optimizer, _ = wrap_model_deepspeed(model, config, optimizer)
            else:  # ddp
                self.model = wrap_model_ddp(model, config)
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = model.to(device)

        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.global_step = 0

    def train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """Execute a single training step."""
        device = next(self.model.parameters()).device
        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
            labels=batch["labels"],
        )
        loss = outputs["loss"]

        if self.config.strategy == "deepspeed":
            self.model.backward(loss)
            self.model.step()
        else:
            loss.backward()
            if self.optimizer:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clipping,
                )
                self.optimizer.step()
                self.optimizer.zero_grad()

        self.global_step += 1

        # Reduce loss across processes
        if self.is_distributed:
            loss = all_reduce_mean(loss.detach())

        return loss.item()

    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = 0

        for batch in self.train_dataloader:
            loss = self.train_step(batch)
            total_loss += loss
            num_batches += 1

            if self.global_step % 100 == 0:
                print_rank_0(
                    f"Step {self.global_step} | Loss: {loss:.4f}",
                    self.config,
                )

        return total_loss / max(num_batches, 1)

    def save_checkpoint(self, path: str):
        """Save checkpoint (handles distributed state)."""
        if not is_main_process(self.config):
            return

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        if self.config.strategy == "fsdp":
            # FSDP requires special handling
            from torch.distributed.fsdp import FullStateDictConfig, StateDictType
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

            save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT, save_policy):
                state_dict = self.model.state_dict()
        elif self.config.strategy == "deepspeed":
            # DeepSpeed handles saving internally
            self.model.save_checkpoint(str(path))
            return
        else:
            state_dict = self.model.module.state_dict() if hasattr(self.model, "module") else self.model.state_dict()

        torch.save(state_dict, path / "model.pt")
        print_rank_0(f"Saved checkpoint to {path}", self.config)

    def cleanup(self):
        """Clean up distributed resources."""
        cleanup_distributed()
