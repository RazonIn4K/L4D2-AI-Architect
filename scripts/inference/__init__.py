# Inference package for L4D2 Copilot
"""
L4D2 Copilot Inference Package

This package provides tools for code generation, completion, and inference
using fine-tuned models for Left 4 Dead 2 modding.

Main Components:
    - L4D2Copilot: Main SDK class for code generation
    - CopilotServer: FastAPI server for inference
    - CopilotClient: HTTP client for server communication
    - OllamaClient: Client for local Ollama inference
"""

from scripts.inference.l4d2_sdk import (
    # Main SDK
    L4D2Copilot,
    create_copilot,
    # Backends
    Backend,
    OllamaBackend,
    OpenAIBackend,
    ServerBackend,
    # Enums
    Language,
    TemplateType,
    # Data Classes
    GenerationConfig,
    GenerationResult,
    ValidationResult,
    BatchResult,
    # Utilities
    CodeValidator,
    TemplateManager,
    # Exceptions
    L4D2Error,
    BackendError,
    ValidationError,
    ConfigurationError,
)

__all__ = [
    # Main SDK
    "L4D2Copilot",
    "create_copilot",
    # Backends
    "Backend",
    "OllamaBackend",
    "OpenAIBackend",
    "ServerBackend",
    # Enums
    "Language",
    "TemplateType",
    # Data Classes
    "GenerationConfig",
    "GenerationResult",
    "ValidationResult",
    "BatchResult",
    # Utilities
    "CodeValidator",
    "TemplateManager",
    # Exceptions
    "L4D2Error",
    "BackendError",
    "ValidationError",
    "ConfigurationError",
]

__version__ = "1.0.0"
