"""
Reasoning Lab - Custom transformer models for reasoning tasks.

A complete framework for:
- Training custom transformer models from scratch
- Knowledge distillation from large to small models
- Reasoning evaluation and benchmarking

Import submodules directly to avoid circular imports:
    from src.models import create_model
    from src.data import create_tokenizer
"""

__version__ = "0.1.0"

# Don't auto-import submodules to avoid circular imports
# Users should import directly: from src.models import ...
