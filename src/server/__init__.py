"""
Svend Server - Production API for reasoning models.

Provides:
- FastAPI REST endpoint
- WebSocket for streaming
- vLLM integration for efficient inference
- Rate limiting and authentication
"""

from .api import create_app
from .inference import InferenceEngine

__all__ = ["create_app", "InferenceEngine"]
