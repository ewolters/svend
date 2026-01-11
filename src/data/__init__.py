"""
Data processing and generation for reasoning model training.
"""

from .tokenizer import (
    create_tokenizer,
    SPECIAL_TOKENS,
    ChatTemplate,
    create_training_sample,
)

from .datasets import (
    DatasetConfig,
    create_combined_dataset,
    ReasoningDataset,
    StreamingReasoningDataset,
    create_dataloaders,
)

from .synthetic import (
    SyntheticConfig,
    SyntheticDataGenerator,
    generate_seed_problems,
    generate_full_dataset,
    load_synthetic_data,
)

__all__ = [
    # Tokenizer
    "create_tokenizer",
    "SPECIAL_TOKENS",
    "ChatTemplate",
    "create_training_sample",
    # Datasets
    "DatasetConfig",
    "create_combined_dataset",
    "ReasoningDataset",
    "StreamingReasoningDataset",
    "create_dataloaders",
    # Synthetic
    "SyntheticConfig",
    "SyntheticDataGenerator",
    "generate_seed_problems",
    "generate_full_dataset",
    "load_synthetic_data",
]
