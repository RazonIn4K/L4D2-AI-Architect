#!/usr/bin/env python3
"""
L4D2 Copilot SDK

A comprehensive Python SDK for the L4D2 Copilot API supporting multiple backends,
streaming, code validation, template expansion, batch generation, and async operations.

Usage:
    from l4d2_sdk import L4D2Copilot

    copilot = L4D2Copilot(backend="ollama")
    code = copilot.generate("Create a tank spawn plugin")
    validated = copilot.validate(code)

Backends:
    - ollama: Local inference via Ollama (recommended for CPU)
    - openai: OpenAI API for cloud inference
    - server: FastAPI server for GPU inference

Installation:
    pip install l4d2-copilot-sdk
    # or install from source
    pip install -e .
"""

from __future__ import annotations

import asyncio
import functools
import json
import logging
import os
import re
import shutil
import subprocess
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
)
from urllib.parse import urlparse

try:
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
except ImportError:
    requests = None  # type: ignore
    HTTPAdapter = None  # type: ignore
    Retry = None  # type: ignore

try:
    import httpx
except ImportError:
    httpx = None  # type: ignore

try:
    import openai
except ImportError:
    openai = None  # type: ignore

# Type variable for generic result handling
T = TypeVar("T")

# Module-level logger
logger = logging.getLogger(__name__)


# ==============================================================================
# Exceptions
# ==============================================================================


class L4D2Error(Exception):
    """Base exception for L4D2 SDK errors."""

    pass


class BackendError(L4D2Error):
    """Error communicating with backend."""

    def __init__(self, message: str, backend: str, original_error: Optional[Exception] = None):
        self.backend = backend
        self.original_error = original_error
        super().__init__(f"[{backend}] {message}")


class ValidationError(L4D2Error):
    """Code validation error."""

    def __init__(self, message: str, errors: Optional[List[str]] = None):
        self.errors = errors or []
        super().__init__(message)


class ConfigurationError(L4D2Error):
    """SDK configuration error."""

    pass


class TimeoutError(L4D2Error):
    """Request timeout error."""

    pass


class RetryExhaustedError(L4D2Error):
    """All retry attempts exhausted."""

    pass


# ==============================================================================
# Enums and Data Classes
# ==============================================================================


class Backend(Enum):
    """Supported inference backends."""

    OLLAMA = "ollama"
    OPENAI = "openai"
    SERVER = "server"


class Language(Enum):
    """Supported programming languages for L4D2 modding."""

    SOURCEPAWN = "sourcepawn"
    VSCRIPT = "vscript"
    AUTO = "auto"


class TemplateType(Enum):
    """Available code templates."""

    PLUGIN = "plugin"
    COMMAND = "command"
    VSCRIPT = "vscript"
    ENTITY = "entity"
    HOOK = "hook"
    MENU = "menu"
    TIMER = "timer"
    ZOMBIE = "zombie"


@dataclass
class GenerationConfig:
    """Configuration for code generation."""

    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    stop_sequences: List[str] = field(default_factory=list)
    language: Language = Language.SOURCEPAWN
    stream: bool = False


@dataclass
class ValidationResult:
    """Result of code validation."""

    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    language: Optional[Language] = None


@dataclass
class GenerationResult:
    """Result of code generation."""

    code: str
    tokens_generated: int = 0
    generation_time: float = 0.0
    model: str = ""
    backend: str = ""
    prompt: str = ""
    validation: Optional[ValidationResult] = None


@dataclass
class BatchResult:
    """Result of batch generation."""

    results: List[GenerationResult]
    total_time: float = 0.0
    success_count: int = 0
    failure_count: int = 0
    errors: List[str] = field(default_factory=list)


# ==============================================================================
# Retry Decorator
# ==============================================================================


def with_retry(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    retryable_exceptions: Tuple[Type[Exception], ...] = (ConnectionError, TimeoutError),
) -> Callable:
    """
    Decorator for automatic retry with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay between retries in seconds
        backoff_factor: Multiplier for delay after each retry
        retryable_exceptions: Tuple of exception types to retry on
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            delay = initial_delay
            last_exception: Optional[Exception] = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay:.1f}s...")
                        time.sleep(delay)
                        delay *= backoff_factor
                    else:
                        logger.error(f"All {max_retries + 1} attempts failed")

            raise RetryExhaustedError(f"Failed after {max_retries + 1} attempts: {last_exception}")

        return wrapper

    return decorator


def with_retry_async(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    retryable_exceptions: Tuple[Type[Exception], ...] = (ConnectionError, TimeoutError),
) -> Callable:
    """Async version of retry decorator."""

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            delay = initial_delay
            last_exception: Optional[Exception] = None

            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay:.1f}s...")
                        await asyncio.sleep(delay)
                        delay *= backoff_factor
                    else:
                        logger.error(f"All {max_retries + 1} attempts failed")

            raise RetryExhaustedError(f"Failed after {max_retries + 1} attempts: {last_exception}")

        return wrapper

    return decorator


# ==============================================================================
# Backend Implementations
# ==============================================================================


class BaseBackend(ABC):
    """Abstract base class for inference backends."""

    name: str = "base"

    @abstractmethod
    def generate(self, prompt: str, config: GenerationConfig) -> GenerationResult:
        """Generate code from prompt."""
        pass

    @abstractmethod
    def generate_stream(self, prompt: str, config: GenerationConfig) -> Iterator[str]:
        """Generate code with streaming output."""
        pass

    @abstractmethod
    async def generate_async(self, prompt: str, config: GenerationConfig) -> GenerationResult:
        """Async code generation."""
        pass

    @abstractmethod
    async def generate_stream_async(self, prompt: str, config: GenerationConfig) -> AsyncIterator[str]:
        """Async streaming generation."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if backend is available."""
        pass

    def get_system_prompt(self, language: Language) -> str:
        """Get system prompt for the given language."""
        prompts = {
            Language.SOURCEPAWN: (
                "You are an expert SourcePawn plugin developer for Left 4 Dead 2. "
                "Write clean, efficient, and well-commented code that follows best practices. "
                "Include proper includes, pragmas, and plugin info blocks."
            ),
            Language.VSCRIPT: (
                "You are an expert VScript developer for Left 4 Dead 2. "
                "Write functional Squirrel scripts that integrate properly with the game engine. "
                "Use proper event hooks and DirectorOptions."
            ),
            Language.AUTO: (
                "You are an expert programmer for Left 4 Dead 2 modding. "
                "Analyze the request and write code in the appropriate language "
                "(SourcePawn for plugins, VScript for director scripts)."
            ),
        }
        return prompts.get(language, prompts[Language.AUTO])


class OllamaBackend(BaseBackend):
    """Backend for local Ollama inference."""

    name = "ollama"

    def __init__(
        self,
        model: str = "l4d2-code-v10plus",
        host: str = "http://localhost:11434",
        timeout: int = 120,
    ):
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout

        if not self._check_ollama_installed():
            raise ConfigurationError("Ollama is not installed. Install from https://ollama.ai")

    def _check_ollama_installed(self) -> bool:
        """Check if Ollama CLI is installed."""
        return shutil.which("ollama") is not None

    def is_available(self) -> bool:
        """Check if Ollama server is running and model is available."""
        try:
            if requests is None:
                # Fall back to subprocess
                result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=5)
                return self.model in result.stdout
            else:
                response = requests.get(f"{self.host}/api/tags", timeout=5)
                if response.status_code == 200:
                    models = response.json().get("models", [])
                    return any(m.get("name", "").startswith(self.model) for m in models)
        except Exception:
            pass
        return False

    @with_retry(max_retries=3, initial_delay=1.0)
    def generate(self, prompt: str, config: GenerationConfig) -> GenerationResult:
        """Generate code using Ollama."""
        start_time = time.time()
        system_prompt = self.get_system_prompt(config.language)
        full_prompt = f"{system_prompt}\n\n{prompt}"

        if requests is not None:
            # Use API for better control
            response = requests.post(
                f"{self.host}/api/generate",
                json={
                    "model": self.model,
                    "prompt": full_prompt,
                    "stream": False,
                    "options": {
                        "temperature": config.temperature,
                        "top_p": config.top_p,
                        "top_k": config.top_k,
                        "repeat_penalty": config.repetition_penalty,
                        "num_predict": config.max_tokens,
                    },
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
            code = data.get("response", "")
            tokens = data.get("eval_count", 0)
        else:
            # Fall back to CLI
            result = subprocess.run(
                ["ollama", "run", self.model, full_prompt],
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            code = result.stdout.strip()
            tokens = len(code.split())  # Approximate

        return GenerationResult(
            code=code,
            tokens_generated=tokens,
            generation_time=time.time() - start_time,
            model=self.model,
            backend=self.name,
            prompt=prompt,
        )

    def generate_stream(self, prompt: str, config: GenerationConfig) -> Iterator[str]:
        """Stream generation using Ollama API."""
        if requests is None:
            raise ConfigurationError("requests library required for streaming. Install with: pip install requests")

        system_prompt = self.get_system_prompt(config.language)
        full_prompt = f"{system_prompt}\n\n{prompt}"

        response = requests.post(
            f"{self.host}/api/generate",
            json={
                "model": self.model,
                "prompt": full_prompt,
                "stream": True,
                "options": {
                    "temperature": config.temperature,
                    "top_p": config.top_p,
                    "top_k": config.top_k,
                    "repeat_penalty": config.repetition_penalty,
                    "num_predict": config.max_tokens,
                },
            },
            stream=True,
            timeout=self.timeout,
        )
        response.raise_for_status()

        for line in response.iter_lines():
            if line:
                try:
                    data = json.loads(line)
                    if "response" in data:
                        yield data["response"]
                    if data.get("done", False):
                        break
                except json.JSONDecodeError:
                    continue

    async def generate_async(self, prompt: str, config: GenerationConfig) -> GenerationResult:
        """Async generation using Ollama."""
        if httpx is None:
            # Fall back to sync in thread pool
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self.generate, prompt, config)

        start_time = time.time()
        system_prompt = self.get_system_prompt(config.language)
        full_prompt = f"{system_prompt}\n\n{prompt}"

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.host}/api/generate",
                json={
                    "model": self.model,
                    "prompt": full_prompt,
                    "stream": False,
                    "options": {
                        "temperature": config.temperature,
                        "top_p": config.top_p,
                        "top_k": config.top_k,
                        "repeat_penalty": config.repetition_penalty,
                        "num_predict": config.max_tokens,
                    },
                },
            )
            response.raise_for_status()
            data = response.json()

        return GenerationResult(
            code=data.get("response", ""),
            tokens_generated=data.get("eval_count", 0),
            generation_time=time.time() - start_time,
            model=self.model,
            backend=self.name,
            prompt=prompt,
        )

    async def generate_stream_async(self, prompt: str, config: GenerationConfig) -> AsyncIterator[str]:
        """Async streaming generation."""
        if httpx is None:
            raise ConfigurationError("httpx library required for async streaming. Install with: pip install httpx")

        system_prompt = self.get_system_prompt(config.language)
        full_prompt = f"{system_prompt}\n\n{prompt}"

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream(
                "POST",
                f"{self.host}/api/generate",
                json={
                    "model": self.model,
                    "prompt": full_prompt,
                    "stream": True,
                    "options": {
                        "temperature": config.temperature,
                        "top_p": config.top_p,
                        "top_k": config.top_k,
                        "repeat_penalty": config.repetition_penalty,
                        "num_predict": config.max_tokens,
                    },
                },
            ) as response:
                async for line in response.aiter_lines():
                    if line:
                        try:
                            data = json.loads(line)
                            if "response" in data:
                                yield data["response"]
                            if data.get("done", False):
                                break
                        except json.JSONDecodeError:
                            continue


class OpenAIBackend(BaseBackend):
    """Backend for OpenAI API inference."""

    name = "openai"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4",
        base_url: Optional[str] = None,
        timeout: int = 60,
    ):
        if openai is None:
            raise ConfigurationError("openai library required. Install with: pip install openai")

        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ConfigurationError("OpenAI API key required. Set OPENAI_API_KEY environment variable.")

        self.model = model
        self.timeout = timeout
        self.client = openai.OpenAI(api_key=self.api_key, base_url=base_url, timeout=timeout)
        self.async_client = openai.AsyncOpenAI(api_key=self.api_key, base_url=base_url, timeout=timeout)

    def is_available(self) -> bool:
        """Check if OpenAI API is accessible."""
        try:
            self.client.models.list()
            return True
        except Exception:
            return False

    @with_retry(max_retries=3, initial_delay=1.0)
    def generate(self, prompt: str, config: GenerationConfig) -> GenerationResult:
        """Generate code using OpenAI API."""
        start_time = time.time()
        system_prompt = self.get_system_prompt(config.language)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            stop=config.stop_sequences or None,
        )

        code = response.choices[0].message.content or ""
        tokens = response.usage.completion_tokens if response.usage else 0

        return GenerationResult(
            code=code,
            tokens_generated=tokens,
            generation_time=time.time() - start_time,
            model=self.model,
            backend=self.name,
            prompt=prompt,
        )

    def generate_stream(self, prompt: str, config: GenerationConfig) -> Iterator[str]:
        """Stream generation using OpenAI API."""
        system_prompt = self.get_system_prompt(config.language)

        stream = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            stop=config.stop_sequences or None,
            stream=True,
        )

        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    @with_retry_async(max_retries=3, initial_delay=1.0)
    async def generate_async(self, prompt: str, config: GenerationConfig) -> GenerationResult:
        """Async generation using OpenAI API."""
        start_time = time.time()
        system_prompt = self.get_system_prompt(config.language)

        response = await self.async_client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            stop=config.stop_sequences or None,
        )

        code = response.choices[0].message.content or ""
        tokens = response.usage.completion_tokens if response.usage else 0

        return GenerationResult(
            code=code,
            tokens_generated=tokens,
            generation_time=time.time() - start_time,
            model=self.model,
            backend=self.name,
            prompt=prompt,
        )

    async def generate_stream_async(self, prompt: str, config: GenerationConfig) -> AsyncIterator[str]:
        """Async streaming generation."""
        system_prompt = self.get_system_prompt(config.language)

        stream = await self.async_client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            stop=config.stop_sequences or None,
            stream=True,
        )

        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


class ServerBackend(BaseBackend):
    """Backend for FastAPI Copilot server."""

    name = "server"

    # Allowed hosts for SSRF prevention
    ALLOWED_HOSTS: Dict[str, str] = {
        "localhost": "localhost",
        "127.0.0.1": "127.0.0.1",
        "0.0.0.0": "0.0.0.0",
    }

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        timeout: int = 30,
    ):
        if requests is None:
            raise ConfigurationError("requests library required. Install with: pip install requests")

        self.base_url = self._validate_url(base_url)
        self.timeout = timeout

        # Setup session with retry
        self.session = requests.Session()
        if Retry is not None and HTTPAdapter is not None:
            retry_strategy = Retry(
                total=3,
                backoff_factor=1,
                status_forcelist=[429, 500, 502, 503, 504],
            )
            adapter = HTTPAdapter(max_retries=retry_strategy)
            self.session.mount("http://", adapter)
            self.session.mount("https://", adapter)

    def _validate_url(self, url: str) -> str:
        """Validate and sanitize server URL to prevent SSRF."""
        parsed = urlparse(url)

        # Validate scheme
        if parsed.scheme not in ("http", "https"):
            raise ConfigurationError(f"Invalid URL scheme: {parsed.scheme}")

        # Validate hostname
        hostname = parsed.hostname or ""
        if hostname not in self.ALLOWED_HOSTS:
            raise ConfigurationError(f"Host '{hostname}' not allowed. Use localhost or 127.0.0.1")

        # Reconstruct safe URL
        safe_host = self.ALLOWED_HOSTS[hostname]
        port = f":{parsed.port}" if parsed.port else ""
        return f"{parsed.scheme}://{safe_host}{port}"

    def is_available(self) -> bool:
        """Check if server is healthy."""
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except Exception:
            return False

    @with_retry(max_retries=3, initial_delay=1.0)
    def generate(self, prompt: str, config: GenerationConfig) -> GenerationResult:
        """Generate code using Copilot server."""
        start_time = time.time()

        response = self.session.post(
            f"{self.base_url}/v1/complete",
            json={
                "prompt": prompt,
                "max_tokens": config.max_tokens,
                "temperature": config.temperature,
                "top_p": config.top_p,
                "stop_sequences": config.stop_sequences,
                "language": config.language.value,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()

        return GenerationResult(
            code=data.get("completion", ""),
            tokens_generated=data.get("tokens_generated", 0),
            generation_time=time.time() - start_time,
            model=data.get("model", "unknown"),
            backend=self.name,
            prompt=prompt,
        )

    def generate_stream(self, prompt: str, config: GenerationConfig) -> Iterator[str]:
        """Stream generation from server (if supported)."""
        # Server doesn't support streaming, fall back to non-streaming
        result = self.generate(prompt, config)
        yield result.code

    async def generate_async(self, prompt: str, config: GenerationConfig) -> GenerationResult:
        """Async generation using server."""
        if httpx is None:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self.generate, prompt, config)

        start_time = time.time()

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/v1/complete",
                json={
                    "prompt": prompt,
                    "max_tokens": config.max_tokens,
                    "temperature": config.temperature,
                    "top_p": config.top_p,
                    "stop_sequences": config.stop_sequences,
                    "language": config.language.value,
                },
            )
            response.raise_for_status()
            data = response.json()

        return GenerationResult(
            code=data.get("completion", ""),
            tokens_generated=data.get("tokens_generated", 0),
            generation_time=time.time() - start_time,
            model=data.get("model", "unknown"),
            backend=self.name,
            prompt=prompt,
        )

    async def generate_stream_async(self, prompt: str, config: GenerationConfig) -> AsyncIterator[str]:
        """Async streaming (falls back to non-streaming)."""
        result = await self.generate_async(prompt, config)
        yield result.code


# ==============================================================================
# Code Validator
# ==============================================================================


class CodeValidator:
    """Validates generated L4D2 mod code."""

    # SourcePawn patterns
    SP_PATTERNS = {
        "include": r'#include\s*<[^>]+>',
        "pragma": r'#pragma\s+\w+',
        "plugin_info": r'public\s+Plugin\s+myinfo\s*=',
        "function": r'(public|static|stock)\s+\w+\s+\w+\s*\([^)]*\)',
        "hook": r'Hook(?:Event|EntityOutput|SingleEntityOutput)',
        "command": r'Reg(?:Console|Admin)Cmd',
    }

    # VScript patterns
    VS_PATTERNS = {
        "function": r'function\s+\w+\s*\([^)]*\)',
        "director": r'DirectorOptions\s*\.',
        "event": r'OnGameEvent_\w+',
        "variable": r'(?:local|::)\s+\w+\s*<-',
    }

    # Common issues to check
    ISSUE_PATTERNS = {
        "hardcoded_paths": r'[A-Za-z]:\\|/home/',
        "sql_injection": r'SQL_Query\([^)]*\+[^)]*\)',
        "buffer_overflow": r'Format(?:Ex)?\([^)]*,\s*\d+\s*,',
        "missing_null_check": r'GetClient(?:Of)?UserId\([^)]*\)(?!\s*[;]?\s*(?:if|==|!=|>|<))',
    }

    @classmethod
    def detect_language(cls, code: str) -> Language:
        """Detect the programming language from code content."""
        sp_score = sum(1 for pattern in cls.SP_PATTERNS.values() if re.search(pattern, code))
        vs_score = sum(1 for pattern in cls.VS_PATTERNS.values() if re.search(pattern, code))

        if sp_score > vs_score:
            return Language.SOURCEPAWN
        elif vs_score > sp_score:
            return Language.VSCRIPT
        return Language.AUTO

    @classmethod
    def validate(cls, code: str, language: Optional[Language] = None) -> ValidationResult:
        """
        Validate code for common issues and best practices.

        Args:
            code: The code to validate
            language: Expected language (auto-detected if None)

        Returns:
            ValidationResult with errors, warnings, and suggestions
        """
        if not code.strip():
            return ValidationResult(is_valid=False, errors=["Empty code provided"])

        detected_lang = language or cls.detect_language(code)
        errors: List[str] = []
        warnings: List[str] = []
        suggestions: List[str] = []

        # Language-specific validation
        if detected_lang == Language.SOURCEPAWN:
            cls._validate_sourcepawn(code, errors, warnings, suggestions)
        elif detected_lang == Language.VSCRIPT:
            cls._validate_vscript(code, errors, warnings, suggestions)

        # Common validations
        cls._validate_common(code, errors, warnings, suggestions)

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            suggestions=suggestions,
            language=detected_lang,
        )

    @classmethod
    def _validate_sourcepawn(
        cls,
        code: str,
        errors: List[str],
        warnings: List[str],
        suggestions: List[str],
    ) -> None:
        """Validate SourcePawn-specific patterns."""
        # Check for required includes
        if not re.search(cls.SP_PATTERNS["include"], code):
            warnings.append("No #include directives found")

        # Check for new decls pragma
        if "#pragma newdecls required" not in code:
            suggestions.append("Consider adding '#pragma newdecls required' for modern syntax")

        # Check for semicolon pragma
        if "#pragma semicolon 1" not in code:
            suggestions.append("Consider adding '#pragma semicolon 1'")

        # Check for plugin info
        if not re.search(cls.SP_PATTERNS["plugin_info"], code):
            warnings.append("Missing plugin info block (public Plugin myinfo)")

        # Check for OnPluginStart
        if "OnPluginStart" not in code and "void OnPluginStart" not in code:
            warnings.append("No OnPluginStart function found")

        # Check for common L4D2 includes
        l4d2_includes = ["<sdktools>", "<left4dhooks>", "<l4d2_direct>"]
        has_l4d2_include = any(inc in code for inc in l4d2_includes)
        if not has_l4d2_include and "l4d" in code.lower():
            suggestions.append("Consider including L4D2-specific headers like <left4dhooks>")

    @classmethod
    def _validate_vscript(
        cls,
        code: str,
        errors: List[str],
        warnings: List[str],
        suggestions: List[str],
    ) -> None:
        """Validate VScript-specific patterns."""
        # Check for function definitions
        if not re.search(cls.VS_PATTERNS["function"], code):
            warnings.append("No function definitions found")

        # Check for DirectorOptions usage
        if "DirectorOptions" in code and not re.search(r"DirectorOptions\s*<-\s*\{", code):
            suggestions.append("Consider initializing DirectorOptions with DirectorOptions <- {}")

        # Check for printl usage
        if "print(" in code and "printl(" not in code:
            suggestions.append("Use printl() instead of print() for better logging")

    @classmethod
    def _validate_common(
        cls,
        code: str,
        errors: List[str],
        warnings: List[str],
        suggestions: List[str],
    ) -> None:
        """Validate common code quality issues."""
        # Check for potential security issues
        for name, pattern in cls.ISSUE_PATTERNS.items():
            if re.search(pattern, code):
                if name == "hardcoded_paths":
                    warnings.append("Hardcoded file paths detected - use relative paths")
                elif name == "sql_injection":
                    errors.append("Potential SQL injection vulnerability detected")
                elif name == "buffer_overflow":
                    warnings.append("Consider using dynamic string buffers for Format()")

        # Check for commented-out code
        comment_lines = len(re.findall(r'^\s*//.*$', code, re.MULTILINE))
        total_lines = len(code.strip().split('\n'))
        if total_lines > 0 and comment_lines / total_lines > 0.5:
            warnings.append("High ratio of commented code - consider cleanup")

        # Check for TODO/FIXME
        if re.search(r'(?:TODO|FIXME|XXX|HACK):', code, re.IGNORECASE):
            suggestions.append("Found TODO/FIXME comments - ensure they are addressed")


# ==============================================================================
# Template Manager
# ==============================================================================


class TemplateManager:
    """Manages L4D2 code templates."""

    TEMPLATES: Dict[TemplateType, str] = {
        TemplateType.PLUGIN: '''#include <sourcemod>
#include <sdktools>

#pragma newdecls required
#pragma semicolon 1

#define PLUGIN_VERSION "1.0.0"

public Plugin myinfo = {{
    name = "{name}",
    author = "{author}",
    description = "{description}",
    version = PLUGIN_VERSION,
    url = "{url}"
}};

public void OnPluginStart() {{
    // Plugin initialization
    PrintToServer("[{name}] Plugin loaded v%s", PLUGIN_VERSION);
}}

public void OnMapStart() {{
    // Map-specific initialization
}}

public void OnClientPutInServer(int client) {{
    // Player joined
}}
''',
        TemplateType.COMMAND: '''#include <sourcemod>

#pragma newdecls required
#pragma semicolon 1

public Plugin myinfo = {{
    name = "{name}",
    author = "{author}",
    description = "Custom commands plugin",
    version = "1.0.0"
}};

public void OnPluginStart() {{
    RegAdminCmd("sm_{command}", Command_{command_cap}, ADMFLAG_GENERIC, "{description}");
}}

public Action Command_{command_cap}(int client, int args) {{
    if (args < 1) {{
        ReplyToCommand(client, "Usage: sm_{command} <parameter>");
        return Plugin_Handled;
    }}

    char param[256];
    GetCmdArg(1, param, sizeof(param));

    // Command logic here
    ReplyToCommand(client, "Command executed with: %s", param);
    return Plugin_Handled;
}}
''',
        TemplateType.VSCRIPT: '''// {name} - VScript for Left 4 Dead 2
// {description}

DirectorOptions <-
{{
    cm_CommonLimit = 20
    cm_MaxSpecials = 8
    cm_DominatorLimit = 4
}}

function OnGameEvent_round_start(params)
{{
    printl("[{name}] Round started!")
    // Initialize custom variables
}}

function OnGameEvent_player_death(params)
{{
    local victim = params["userid"]
    local attacker = params["attacker"]

    if (victim != null && attacker != null)
    {{
        printl("[{name}] Player killed")
    }}
}}

function Update()
{{
    // Called every frame - add custom logic
}}
''',
        TemplateType.ENTITY: '''#include <sourcemod>
#include <sdktools>

#pragma newdecls required
#pragma semicolon 1

public Plugin myinfo = {{
    name = "{name}",
    author = "{author}",
    description = "Entity management plugin",
    version = "1.0.0"
}};

public void OnPluginStart() {{
    HookEvent("player_spawn", Event_PlayerSpawn);
    HookEvent("infected_death", Event_InfectedDeath);
}}

public void Event_PlayerSpawn(Event event, const char[] name, bool dontBroadcast) {{
    int client = GetClientOfUserId(event.GetInt("userid"));

    if (IsClientInGame(client) && GetClientTeam(client) == 2) {{
        // Survivor spawned
        GivePlayerItem(client, "pistol");
    }}
}}

public void Event_InfectedDeath(Event event, const char[] name, bool dontBroadcast) {{
    int killer = GetClientOfUserId(event.GetInt("attacker"));

    if (killer > 0 && IsClientInGame(killer)) {{
        // Handle infected death
    }}
}}

stock int SpawnEntity(const char[] classname, float pos[3]) {{
    int entity = CreateEntityByName(classname);
    if (entity != -1) {{
        TeleportEntity(entity, pos, NULL_VECTOR, NULL_VECTOR);
        DispatchSpawn(entity);
        ActivateEntity(entity);
    }}
    return entity;
}}
''',
        TemplateType.HOOK: '''#include <sourcemod>
#include <sdkhooks>
#include <sdktools>

#pragma newdecls required
#pragma semicolon 1

public Plugin myinfo = {{
    name = "{name}",
    author = "{author}",
    description = "Hook-based plugin",
    version = "1.0.0"
}};

public void OnPluginStart() {{
    HookEvent("{event_name}", Event_Handler);
}}

public void OnClientPutInServer(int client) {{
    SDKHook(client, SDKHook_OnTakeDamage, OnTakeDamage);
}}

public void OnClientDisconnect(int client) {{
    SDKUnhook(client, SDKHook_OnTakeDamage, OnTakeDamage);
}}

public Action OnTakeDamage(int victim, int &attacker, int &inflictor,
                           float &damage, int &damagetype) {{
    // Modify damage here
    return Plugin_Continue;
}}

public void Event_Handler(Event event, const char[] name, bool dontBroadcast) {{
    // Handle event
}}
''',
        TemplateType.MENU: '''#include <sourcemod>

#pragma newdecls required
#pragma semicolon 1

public Plugin myinfo = {{
    name = "{name}",
    author = "{author}",
    description = "Menu plugin",
    version = "1.0.0"
}};

public void OnPluginStart() {{
    RegConsoleCmd("sm_{command}", Command_ShowMenu, "Show the menu");
}}

public Action Command_ShowMenu(int client, int args) {{
    if (client == 0) {{
        ReplyToCommand(client, "This command is for in-game use only");
        return Plugin_Handled;
    }}

    ShowMainMenu(client);
    return Plugin_Handled;
}}

void ShowMainMenu(int client) {{
    Menu menu = new Menu(MenuHandler_Main);
    menu.SetTitle("{name}");

    menu.AddItem("option1", "Option 1");
    menu.AddItem("option2", "Option 2");
    menu.AddItem("option3", "Option 3");

    menu.ExitButton = true;
    menu.Display(client, MENU_TIME_FOREVER);
}}

public int MenuHandler_Main(Menu menu, MenuAction action, int param1, int param2) {{
    switch (action) {{
        case MenuAction_Select: {{
            char info[32];
            menu.GetItem(param2, info, sizeof(info));

            if (StrEqual(info, "option1")) {{
                PrintToChat(param1, "You selected Option 1");
            }}
        }}
        case MenuAction_End: {{
            delete menu;
        }}
    }}
    return 0;
}}
''',
        TemplateType.TIMER: '''#include <sourcemod>
#include <sdktools>

#pragma newdecls required
#pragma semicolon 1

Handle g_hTimer = null;
float g_fInterval = 1.0;

public Plugin myinfo = {{
    name = "{name}",
    author = "{author}",
    description = "Timer-based plugin",
    version = "1.0.0"
}};

public void OnPluginStart() {{
    RegAdminCmd("sm_starttimer", Command_StartTimer, ADMFLAG_GENERIC);
    RegAdminCmd("sm_stoptimer", Command_StopTimer, ADMFLAG_GENERIC);
}}

public void OnMapStart() {{
    g_hTimer = CreateTimer(g_fInterval, Timer_Callback, _, TIMER_REPEAT);
}}

public void OnMapEnd() {{
    if (g_hTimer != null) {{
        delete g_hTimer;
        g_hTimer = null;
    }}
}}

public Action Timer_Callback(Handle timer) {{
    // Timer logic here
    return Plugin_Continue;
}}

public Action Command_StartTimer(int client, int args) {{
    if (g_hTimer == null) {{
        g_hTimer = CreateTimer(g_fInterval, Timer_Callback, _, TIMER_REPEAT);
        ReplyToCommand(client, "Timer started");
    }}
    return Plugin_Handled;
}}

public Action Command_StopTimer(int client, int args) {{
    if (g_hTimer != null) {{
        delete g_hTimer;
        g_hTimer = null;
        ReplyToCommand(client, "Timer stopped");
    }}
    return Plugin_Handled;
}}
''',
        TemplateType.ZOMBIE: '''#include <sourcemod>
#include <sdktools>
#include <left4dhooks>

#pragma newdecls required
#pragma semicolon 1

public Plugin myinfo = {{
    name = "{name}",
    author = "{author}",
    description = "Special infected control plugin",
    version = "1.0.0"
}};

public void OnPluginStart() {{
    HookEvent("player_spawn", Event_PlayerSpawn);
    HookEvent("player_death", Event_PlayerDeath);
}}

public void Event_PlayerSpawn(Event event, const char[] name, bool dontBroadcast) {{
    int client = GetClientOfUserId(event.GetInt("userid"));

    if (IsClientInGame(client) && GetClientTeam(client) == 3) {{
        // Infected player spawned
        int class = GetEntProp(client, Prop_Send, "m_zombieClass");
        HandleInfectedSpawn(client, class);
    }}
}}

void HandleInfectedSpawn(int client, int zombieClass) {{
    switch (zombieClass) {{
        case 1: PrintToChat(client, "You are a Smoker");
        case 2: PrintToChat(client, "You are a Boomer");
        case 3: PrintToChat(client, "You are a Hunter");
        case 4: PrintToChat(client, "You are a Spitter");
        case 5: PrintToChat(client, "You are a Jockey");
        case 6: PrintToChat(client, "You are a Charger");
        case 8: PrintToChat(client, "You are a Tank!");
    }}
}}

public void Event_PlayerDeath(Event event, const char[] name, bool dontBroadcast) {{
    int victim = GetClientOfUserId(event.GetInt("userid"));

    if (IsClientInGame(victim) && GetClientTeam(victim) == 3) {{
        // Infected died
    }}
}}
''',
    }

    @classmethod
    def get_template(
        cls,
        template_type: TemplateType,
        **kwargs: Any,
    ) -> str:
        """
        Get a code template with variable substitution.

        Args:
            template_type: Type of template to generate
            **kwargs: Variables to substitute in template

        Returns:
            Formatted template string
        """
        if template_type not in cls.TEMPLATES:
            raise ValueError(f"Unknown template type: {template_type}")

        template = cls.TEMPLATES[template_type]

        # Default values
        defaults = {
            "name": "L4D2 Plugin",
            "author": "Developer",
            "description": "A Left 4 Dead 2 plugin",
            "url": "https://github.com/yourusername",
            "command": "custom",
            "command_cap": "Custom",
            "event_name": "player_spawn",
        }

        # Merge defaults with provided kwargs
        variables = {**defaults, **kwargs}

        # Format template
        try:
            return template.format(**variables)
        except KeyError as e:
            raise ValueError(f"Missing template variable: {e}")

    @classmethod
    def list_templates(cls) -> List[str]:
        """Get list of available template names."""
        return [t.value for t in TemplateType]


# ==============================================================================
# Main SDK Class
# ==============================================================================


class L4D2Copilot:
    """
    L4D2 Copilot SDK - AI-powered code generation for Left 4 Dead 2 modding.

    Supports multiple backends (Ollama, OpenAI, Server), streaming generation,
    code validation, template expansion, batch generation, and async operations.

    Usage:
        # Basic usage
        copilot = L4D2Copilot(backend="ollama")
        code = copilot.generate("Create a tank spawn plugin")

        # With validation
        result = copilot.generate_with_validation("Heal all survivors")

        # Streaming
        for chunk in copilot.generate_stream("Create a menu plugin"):
            print(chunk, end="")

        # Async
        code = await copilot.generate_async("Create a timer plugin")

        # Batch
        prompts = ["Plugin 1", "Plugin 2", "Plugin 3"]
        results = copilot.generate_batch(prompts)

        # Template
        code = copilot.expand_template("plugin", name="My Plugin")
    """

    def __init__(
        self,
        backend: Union[str, Backend] = Backend.OLLAMA,
        *,
        # Ollama options
        ollama_model: str = "l4d2-code-v10plus",
        ollama_host: str = "http://localhost:11434",
        # OpenAI options
        openai_api_key: Optional[str] = None,
        openai_model: str = "gpt-4",
        openai_base_url: Optional[str] = None,
        # Server options
        server_url: str = "http://localhost:8000",
        # Common options
        timeout: int = 120,
        default_language: Language = Language.SOURCEPAWN,
        auto_validate: bool = False,
        log_level: int = logging.INFO,
    ):
        """
        Initialize L4D2 Copilot SDK.

        Args:
            backend: Backend to use ("ollama", "openai", or "server")
            ollama_model: Ollama model name (default: l4d2-code-v10plus)
            ollama_host: Ollama API host
            openai_api_key: OpenAI API key (or set OPENAI_API_KEY env var)
            openai_model: OpenAI model name
            openai_base_url: Custom OpenAI API base URL
            server_url: Copilot server URL
            timeout: Request timeout in seconds
            default_language: Default programming language
            auto_validate: Automatically validate generated code
            log_level: Logging level
        """
        # Setup logging
        logging.basicConfig(level=log_level)
        self.logger = logging.getLogger(__name__)

        # Store configuration
        self.default_language = default_language
        self.auto_validate = auto_validate
        self.timeout = timeout

        # Initialize backend
        backend_enum = Backend(backend) if isinstance(backend, str) else backend

        if backend_enum == Backend.OLLAMA:
            self._backend: BaseBackend = OllamaBackend(
                model=ollama_model,
                host=ollama_host,
                timeout=timeout,
            )
        elif backend_enum == Backend.OPENAI:
            self._backend = OpenAIBackend(
                api_key=openai_api_key,
                model=openai_model,
                base_url=openai_base_url,
                timeout=timeout,
            )
        elif backend_enum == Backend.SERVER:
            self._backend = ServerBackend(
                base_url=server_url,
                timeout=timeout,
            )
        else:
            raise ConfigurationError(f"Unknown backend: {backend}")

        # Validators and templates
        self.validator = CodeValidator()
        self.templates = TemplateManager()

        # Thread pool for batch operations
        self._executor = ThreadPoolExecutor(max_workers=4)

    @property
    def backend_name(self) -> str:
        """Get the name of the current backend."""
        return self._backend.name

    def is_available(self) -> bool:
        """Check if the backend is available."""
        return self._backend.is_available()

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
        language: Optional[Language] = None,
        validate: Optional[bool] = None,
    ) -> str:
        """
        Generate code from a prompt.

        Args:
            prompt: The code generation prompt
            max_tokens: Maximum tokens to generate
            temperature: Generation temperature (0.0-1.0)
            language: Target language (uses default if None)
            validate: Validate generated code (uses auto_validate if None)

        Returns:
            Generated code as string
        """
        config = GenerationConfig(
            max_tokens=max_tokens,
            temperature=temperature,
            language=language or self.default_language,
        )

        result = self._backend.generate(prompt, config)

        if validate if validate is not None else self.auto_validate:
            validation = self.validate(result.code)
            if not validation.is_valid:
                self.logger.warning(f"Generated code validation failed: {validation.errors}")

        return result.code

    def generate_result(
        self,
        prompt: str,
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
        language: Optional[Language] = None,
    ) -> GenerationResult:
        """
        Generate code and return full result with metadata.

        Args:
            prompt: The code generation prompt
            max_tokens: Maximum tokens to generate
            temperature: Generation temperature
            language: Target language

        Returns:
            GenerationResult with code and metadata
        """
        config = GenerationConfig(
            max_tokens=max_tokens,
            temperature=temperature,
            language=language or self.default_language,
        )
        return self._backend.generate(prompt, config)

    def generate_with_validation(
        self,
        prompt: str,
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
        language: Optional[Language] = None,
    ) -> GenerationResult:
        """
        Generate code with automatic validation.

        Args:
            prompt: The code generation prompt
            max_tokens: Maximum tokens to generate
            temperature: Generation temperature
            language: Target language

        Returns:
            GenerationResult with validation attached
        """
        result = self.generate_result(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            language=language,
        )
        result.validation = self.validate(result.code)
        return result

    def generate_stream(
        self,
        prompt: str,
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
        language: Optional[Language] = None,
    ) -> Iterator[str]:
        """
        Generate code with streaming output.

        Args:
            prompt: The code generation prompt
            max_tokens: Maximum tokens to generate
            temperature: Generation temperature
            language: Target language

        Yields:
            Code chunks as they are generated
        """
        config = GenerationConfig(
            max_tokens=max_tokens,
            temperature=temperature,
            language=language or self.default_language,
            stream=True,
        )
        yield from self._backend.generate_stream(prompt, config)

    async def generate_async(
        self,
        prompt: str,
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
        language: Optional[Language] = None,
    ) -> str:
        """
        Asynchronously generate code.

        Args:
            prompt: The code generation prompt
            max_tokens: Maximum tokens to generate
            temperature: Generation temperature
            language: Target language

        Returns:
            Generated code as string
        """
        config = GenerationConfig(
            max_tokens=max_tokens,
            temperature=temperature,
            language=language or self.default_language,
        )
        result = await self._backend.generate_async(prompt, config)
        return result.code

    async def generate_stream_async(
        self,
        prompt: str,
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
        language: Optional[Language] = None,
    ) -> AsyncIterator[str]:
        """
        Asynchronously generate code with streaming.

        Args:
            prompt: The code generation prompt
            max_tokens: Maximum tokens to generate
            temperature: Generation temperature
            language: Target language

        Yields:
            Code chunks as they are generated
        """
        config = GenerationConfig(
            max_tokens=max_tokens,
            temperature=temperature,
            language=language or self.default_language,
            stream=True,
        )
        async for chunk in self._backend.generate_stream_async(prompt, config):
            yield chunk

    def generate_batch(
        self,
        prompts: List[str],
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
        language: Optional[Language] = None,
        parallel: bool = True,
    ) -> BatchResult:
        """
        Generate code for multiple prompts.

        Args:
            prompts: List of prompts to generate code for
            max_tokens: Maximum tokens per generation
            temperature: Generation temperature
            language: Target language
            parallel: Run generations in parallel

        Returns:
            BatchResult with all generation results
        """
        start_time = time.time()
        results: List[GenerationResult] = []
        errors: List[str] = []

        config = GenerationConfig(
            max_tokens=max_tokens,
            temperature=temperature,
            language=language or self.default_language,
        )

        if parallel:
            futures = [
                self._executor.submit(self._backend.generate, prompt, config)
                for prompt in prompts
            ]

            for i, future in enumerate(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"Batch generation failed for prompt {i}: {e}")
                    errors.append(f"Prompt {i}: {str(e)}")
                    results.append(
                        GenerationResult(
                            code="",
                            prompt=prompts[i],
                            backend=self._backend.name,
                        )
                    )
        else:
            for i, prompt in enumerate(prompts):
                try:
                    result = self._backend.generate(prompt, config)
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"Batch generation failed for prompt {i}: {e}")
                    errors.append(f"Prompt {i}: {str(e)}")
                    results.append(
                        GenerationResult(
                            code="",
                            prompt=prompt,
                            backend=self._backend.name,
                        )
                    )

        return BatchResult(
            results=results,
            total_time=time.time() - start_time,
            success_count=len([r for r in results if r.code]),
            failure_count=len([r for r in results if not r.code]),
            errors=errors,
        )

    async def generate_batch_async(
        self,
        prompts: List[str],
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
        language: Optional[Language] = None,
    ) -> BatchResult:
        """
        Asynchronously generate code for multiple prompts.

        Args:
            prompts: List of prompts
            max_tokens: Maximum tokens per generation
            temperature: Generation temperature
            language: Target language

        Returns:
            BatchResult with all generation results
        """
        start_time = time.time()
        config = GenerationConfig(
            max_tokens=max_tokens,
            temperature=temperature,
            language=language or self.default_language,
        )

        tasks = [self._backend.generate_async(prompt, config) for prompt in prompts]
        results_or_errors = await asyncio.gather(*tasks, return_exceptions=True)

        results: List[GenerationResult] = []
        errors: List[str] = []

        for i, result in enumerate(results_or_errors):
            if isinstance(result, Exception):
                self.logger.error(f"Async batch generation failed for prompt {i}: {result}")
                errors.append(f"Prompt {i}: {str(result)}")
                results.append(
                    GenerationResult(
                        code="",
                        prompt=prompts[i],
                        backend=self._backend.name,
                    )
                )
            else:
                results.append(result)

        return BatchResult(
            results=results,
            total_time=time.time() - start_time,
            success_count=len([r for r in results if r.code]),
            failure_count=len([r for r in results if not r.code]),
            errors=errors,
        )

    def validate(self, code: str, language: Optional[Language] = None) -> ValidationResult:
        """
        Validate generated code.

        Args:
            code: Code to validate
            language: Expected language (auto-detected if None)

        Returns:
            ValidationResult with errors, warnings, and suggestions
        """
        return self.validator.validate(code, language)

    def expand_template(
        self,
        template: Union[str, TemplateType],
        **kwargs: Any,
    ) -> str:
        """
        Expand a code template with variables.

        Args:
            template: Template name or TemplateType
            **kwargs: Variables to substitute

        Returns:
            Expanded template code
        """
        template_type = TemplateType(template) if isinstance(template, str) else template
        return self.templates.get_template(template_type, **kwargs)

    def list_templates(self) -> List[str]:
        """Get list of available template names."""
        return self.templates.list_templates()

    def chat(
        self,
        messages: List[Dict[str, str]],
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        """
        Chat-style interaction with the model.

        Args:
            messages: List of message dicts with 'role' and 'content'
            max_tokens: Maximum response tokens
            temperature: Generation temperature

        Returns:
            Assistant response
        """
        # Build conversation context
        context_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                context_parts.append(f"System: {content}")
            elif role == "user":
                context_parts.append(f"User: {content}")
            elif role == "assistant":
                context_parts.append(f"Assistant: {content}")

        prompt = "\n\n".join(context_parts) + "\n\nAssistant:"
        return self.generate(prompt, max_tokens=max_tokens, temperature=temperature)

    def __repr__(self) -> str:
        return f"L4D2Copilot(backend={self._backend.name!r}, language={self.default_language.value!r})"


# ==============================================================================
# Convenience Functions
# ==============================================================================


def create_copilot(
    backend: str = "ollama",
    **kwargs: Any,
) -> L4D2Copilot:
    """
    Factory function to create an L4D2Copilot instance.

    Args:
        backend: Backend name ("ollama", "openai", "server")
        **kwargs: Additional configuration options

    Returns:
        Configured L4D2Copilot instance
    """
    return L4D2Copilot(backend=backend, **kwargs)


# ==============================================================================
# Module Exports
# ==============================================================================

__all__ = [
    # Main SDK
    "L4D2Copilot",
    "create_copilot",
    # Backends
    "Backend",
    "BaseBackend",
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
    "TimeoutError",
    "RetryExhaustedError",
]

__version__ = "1.0.0"
