#!/usr/bin/env python3
"""
L4D2 Production Model Server

A production-ready model serving infrastructure that supports:
- Multiple model backends (Ollama, OpenAI, vLLM, local transformers)
- Load balancing with round-robin and fallback
- Response caching (in-memory or Redis)
- Per-client rate limiting
- Comprehensive metrics and health checks

Usage:
    python model_server.py --backends ollama,openai --port 8000
    python model_server.py --backends vllm,ollama --redis-url redis://localhost:6379
"""

import asyncio
import hashlib
import json
import logging
import os
import sys
import time
import argparse
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from functools import wraps
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)
import threading
import uuid
from contextlib import asynccontextmanager

# Add parent to path for security utils
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.security import safe_read_yaml

try:
    from fastapi import FastAPI, HTTPException, Request, Response, Depends
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field
    import uvicorn
except ImportError:
    print("FastAPI not installed. Run: pip install fastapi uvicorn")
    sys.exit(1)

try:
    import httpx
except ImportError:
    httpx = None
    print("Warning: httpx not installed, some backends may not work. Run: pip install httpx")

try:
    import redis.asyncio as aioredis
except ImportError:
    aioredis = None

# Optional imports for backends
try:
    import openai
except ImportError:
    openai = None

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

PROJECT_ROOT = Path(__file__).parent.parent.parent

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
# Request/Response Models
# =============================================================================

class Message(BaseModel):
    """Chat message model"""
    role: str = Field(..., description="Role: system, user, or assistant")
    content: str = Field(..., description="Message content")


class CompletionRequest(BaseModel):
    """Standard completion request"""
    prompt: str = Field(..., description="Input prompt")
    model: Optional[str] = Field(None, description="Model to use")
    max_tokens: int = Field(256, ge=1, le=4096, description="Maximum tokens to generate")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float = Field(0.9, ge=0.0, le=1.0, description="Nucleus sampling parameter")
    stop: Optional[List[str]] = Field(None, description="Stop sequences")
    stream: bool = Field(False, description="Enable streaming (not yet implemented)")


class ChatCompletionRequest(BaseModel):
    """Chat completion request"""
    messages: List[Message] = Field(..., description="Chat messages")
    model: Optional[str] = Field(None, description="Model to use")
    max_tokens: int = Field(512, ge=1, le=4096, description="Maximum tokens to generate")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float = Field(0.9, ge=0.0, le=1.0, description="Nucleus sampling parameter")
    stop: Optional[List[str]] = Field(None, description="Stop sequences")
    stream: bool = Field(False, description="Enable streaming (not yet implemented)")


class EmbeddingRequest(BaseModel):
    """Embedding request"""
    input: Union[str, List[str]] = Field(..., description="Text to embed")
    model: Optional[str] = Field(None, description="Embedding model to use")


class CompletionChoice(BaseModel):
    """Completion choice"""
    index: int
    text: str
    finish_reason: str = "stop"


class CompletionResponse(BaseModel):
    """Completion response"""
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: List[CompletionChoice]
    usage: Dict[str, int]


class ChatChoice(BaseModel):
    """Chat choice"""
    index: int
    message: Message
    finish_reason: str = "stop"


class ChatCompletionResponse(BaseModel):
    """Chat completion response"""
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatChoice]
    usage: Dict[str, int]


class EmbeddingData(BaseModel):
    """Embedding data"""
    index: int
    embedding: List[float]
    object: str = "embedding"


class EmbeddingResponse(BaseModel):
    """Embedding response"""
    object: str = "list"
    data: List[EmbeddingData]
    model: str
    usage: Dict[str, int]


class ModelInfo(BaseModel):
    """Model information"""
    id: str
    object: str = "model"
    created: int
    owned_by: str
    backend: str


class ModelListResponse(BaseModel):
    """Model list response"""
    object: str = "list"
    data: List[ModelInfo]


class BackendHealth(BaseModel):
    """Backend health status"""
    name: str
    healthy: bool
    latency_ms: Optional[float] = None
    error: Optional[str] = None
    last_check: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    backends: List[BackendHealth]
    cache_enabled: bool
    cache_type: str
    uptime_seconds: float


class MetricsResponse(BaseModel):
    """Metrics response"""
    requests_total: int
    requests_by_backend: Dict[str, int]
    cache_hits: int
    cache_misses: int
    cache_hit_rate: float
    avg_latency_ms: float
    errors_total: int
    errors_by_backend: Dict[str, int]
    rate_limit_rejections: int
    uptime_seconds: float


# =============================================================================
# Cache Layer
# =============================================================================

class CacheBackend(ABC):
    """Abstract cache backend"""

    @abstractmethod
    async def get(self, key: str) -> Optional[str]:
        """Get value from cache"""
        pass

    @abstractmethod
    async def set(self, key: str, value: str, ttl: int = 3600) -> bool:
        """Set value in cache"""
        pass

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete value from cache"""
        pass

    @abstractmethod
    async def clear(self) -> bool:
        """Clear all cache"""
        pass

    @abstractmethod
    async def size(self) -> int:
        """Get cache size"""
        pass


class InMemoryCache(CacheBackend):
    """In-memory LRU cache with TTL"""

    def __init__(self, max_size: int = 1000):
        self._cache: Dict[str, Tuple[str, float]] = {}
        self._max_size = max_size
        self._access_order: List[str] = []
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Optional[str]:
        async with self._lock:
            if key in self._cache:
                value, expires_at = self._cache[key]
                if time.time() < expires_at:
                    # Update access order
                    self._access_order.remove(key)
                    self._access_order.append(key)
                    return value
                else:
                    # Expired
                    del self._cache[key]
                    self._access_order.remove(key)
            return None

    async def set(self, key: str, value: str, ttl: int = 3600) -> bool:
        async with self._lock:
            # Evict oldest if at capacity
            while len(self._cache) >= self._max_size and self._access_order:
                oldest = self._access_order.pop(0)
                if oldest in self._cache:
                    del self._cache[oldest]

            expires_at = time.time() + ttl
            self._cache[key] = (value, expires_at)

            if key in self._access_order:
                self._access_order.remove(key)
            self._access_order.append(key)
            return True

    async def delete(self, key: str) -> bool:
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                self._access_order.remove(key)
                return True
            return False

    async def clear(self) -> bool:
        async with self._lock:
            self._cache.clear()
            self._access_order.clear()
            return True

    async def size(self) -> int:
        return len(self._cache)


class RedisCache(CacheBackend):
    """Redis cache backend"""

    def __init__(self, url: str, prefix: str = "l4d2_model_server:"):
        if aioredis is None:
            raise ImportError("redis package not installed. Run: pip install redis")
        self._url = url
        self._prefix = prefix
        self._client: Optional[Any] = None

    async def _get_client(self):
        if self._client is None:
            self._client = await aioredis.from_url(self._url)
        return self._client

    async def get(self, key: str) -> Optional[str]:
        try:
            client = await self._get_client()
            value = await client.get(f"{self._prefix}{key}")
            return value.decode() if value else None
        except Exception as e:
            logger.error(f"Redis get error: {e}")
            return None

    async def set(self, key: str, value: str, ttl: int = 3600) -> bool:
        try:
            client = await self._get_client()
            await client.setex(f"{self._prefix}{key}", ttl, value)
            return True
        except Exception as e:
            logger.error(f"Redis set error: {e}")
            return False

    async def delete(self, key: str) -> bool:
        try:
            client = await self._get_client()
            await client.delete(f"{self._prefix}{key}")
            return True
        except Exception as e:
            logger.error(f"Redis delete error: {e}")
            return False

    async def clear(self) -> bool:
        try:
            client = await self._get_client()
            keys = []
            async for key in client.scan_iter(f"{self._prefix}*"):
                keys.append(key)
            if keys:
                await client.delete(*keys)
            return True
        except Exception as e:
            logger.error(f"Redis clear error: {e}")
            return False

    async def size(self) -> int:
        try:
            client = await self._get_client()
            count = 0
            async for _ in client.scan_iter(f"{self._prefix}*"):
                count += 1
            return count
        except Exception as e:
            logger.error(f"Redis size error: {e}")
            return 0


# =============================================================================
# Rate Limiter
# =============================================================================

@dataclass
class RateLimitConfig:
    """Rate limit configuration"""
    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    burst_size: int = 10


class RateLimiter:
    """Token bucket rate limiter with per-client tracking"""

    def __init__(self, config: RateLimitConfig):
        self.config = config
        self._client_buckets: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()

    def _get_client_id(self, request: Request) -> str:
        """Get client identifier from request"""
        # Check for API key first
        api_key = request.headers.get("X-API-Key") or request.headers.get("Authorization")
        if api_key:
            return hashlib.sha256(api_key.encode()).hexdigest()[:16]

        # Fall back to IP
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    async def check_rate_limit(self, request: Request) -> Tuple[bool, Dict[str, Any]]:
        """Check if request is within rate limits"""
        client_id = self._get_client_id(request)
        now = time.time()

        async with self._lock:
            if client_id not in self._client_buckets:
                self._client_buckets[client_id] = {
                    "minute_tokens": self.config.requests_per_minute,
                    "hour_tokens": self.config.requests_per_hour,
                    "minute_last_update": now,
                    "hour_last_update": now,
                    "burst_tokens": self.config.burst_size,
                }

            bucket = self._client_buckets[client_id]

            # Refill minute tokens
            minute_elapsed = now - bucket["minute_last_update"]
            minute_refill = (minute_elapsed / 60.0) * self.config.requests_per_minute
            bucket["minute_tokens"] = min(
                self.config.requests_per_minute,
                bucket["minute_tokens"] + minute_refill
            )
            bucket["minute_last_update"] = now

            # Refill hour tokens
            hour_elapsed = now - bucket["hour_last_update"]
            hour_refill = (hour_elapsed / 3600.0) * self.config.requests_per_hour
            bucket["hour_tokens"] = min(
                self.config.requests_per_hour,
                bucket["hour_tokens"] + hour_refill
            )
            bucket["hour_last_update"] = now

            # Check limits
            if bucket["minute_tokens"] < 1 or bucket["hour_tokens"] < 1:
                return False, {
                    "retry_after": 60 if bucket["minute_tokens"] < 1 else 3600,
                    "limit_type": "minute" if bucket["minute_tokens"] < 1 else "hour",
                }

            # Consume token
            bucket["minute_tokens"] -= 1
            bucket["hour_tokens"] -= 1

            return True, {
                "remaining_minute": int(bucket["minute_tokens"]),
                "remaining_hour": int(bucket["hour_tokens"]),
            }


# =============================================================================
# Metrics Collector
# =============================================================================

@dataclass
class Metrics:
    """Server metrics"""
    start_time: float = field(default_factory=time.time)
    requests_total: int = 0
    requests_by_backend: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    cache_hits: int = 0
    cache_misses: int = 0
    latencies: List[float] = field(default_factory=list)
    errors_total: int = 0
    errors_by_backend: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    rate_limit_rejections: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def record_request(self, backend: str, latency_ms: float, error: bool = False):
        """Record a request"""
        with self._lock:
            self.requests_total += 1
            self.requests_by_backend[backend] += 1
            self.latencies.append(latency_ms)
            # Keep only last 1000 latencies
            if len(self.latencies) > 1000:
                self.latencies = self.latencies[-1000:]
            if error:
                self.errors_total += 1
                self.errors_by_backend[backend] += 1

    def record_cache_hit(self):
        with self._lock:
            self.cache_hits += 1

    def record_cache_miss(self):
        with self._lock:
            self.cache_misses += 1

    def record_rate_limit(self):
        with self._lock:
            self.rate_limit_rejections += 1

    def get_metrics(self) -> Dict[str, Any]:
        with self._lock:
            total_cache = self.cache_hits + self.cache_misses
            cache_hit_rate = self.cache_hits / total_cache if total_cache > 0 else 0.0
            avg_latency = sum(self.latencies) / len(self.latencies) if self.latencies else 0.0

            return {
                "requests_total": self.requests_total,
                "requests_by_backend": dict(self.requests_by_backend),
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
                "cache_hit_rate": round(cache_hit_rate, 4),
                "avg_latency_ms": round(avg_latency, 2),
                "errors_total": self.errors_total,
                "errors_by_backend": dict(self.errors_by_backend),
                "rate_limit_rejections": self.rate_limit_rejections,
                "uptime_seconds": round(time.time() - self.start_time, 2),
            }


# =============================================================================
# Model Backends
# =============================================================================

class BackendStatus(Enum):
    """Backend status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class BackendInfo:
    """Backend information"""
    name: str
    status: BackendStatus = BackendStatus.UNKNOWN
    last_check: Optional[datetime] = None
    last_error: Optional[str] = None
    latency_ms: Optional[float] = None
    models: List[str] = field(default_factory=list)


class ModelBackend(ABC):
    """Abstract model backend"""

    def __init__(self, name: str, config: Dict[str, Any] = None):
        self.name = name
        self.config = config or {}
        self.info = BackendInfo(name=name)

    @abstractmethod
    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        """Generate completion"""
        pass

    @abstractmethod
    async def chat_complete(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """Generate chat completion"""
        pass

    @abstractmethod
    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """Generate embeddings"""
        pass

    @abstractmethod
    async def list_models(self) -> List[str]:
        """List available models"""
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """Check backend health"""
        pass


class OllamaBackend(ModelBackend):
    """Ollama local backend"""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("ollama", config)
        self.base_url = config.get("base_url", "http://localhost:11434") if config else "http://localhost:11434"
        self.default_model = config.get("model", "l4d2-code-v10plus") if config else "l4d2-code-v10plus"
        self._client: Optional[Any] = None

    async def _get_client(self):
        if httpx is None:
            raise ImportError("httpx not installed")
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=120.0)
        return self._client

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        client = await self._get_client()
        model = request.model or self.default_model

        response = await client.post(
            f"{self.base_url}/api/generate",
            json={
                "model": model,
                "prompt": request.prompt,
                "options": {
                    "temperature": request.temperature,
                    "top_p": request.top_p,
                    "num_predict": request.max_tokens,
                    "stop": request.stop or [],
                },
                "stream": False,
            }
        )
        response.raise_for_status()
        data = response.json()

        return CompletionResponse(
            id=f"cmpl-{uuid.uuid4().hex[:8]}",
            created=int(time.time()),
            model=model,
            choices=[CompletionChoice(index=0, text=data.get("response", ""))],
            usage={
                "prompt_tokens": data.get("prompt_eval_count", 0),
                "completion_tokens": data.get("eval_count", 0),
                "total_tokens": data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
            }
        )

    async def chat_complete(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        client = await self._get_client()
        model = request.model or self.default_model

        messages = [{"role": m.role, "content": m.content} for m in request.messages]

        response = await client.post(
            f"{self.base_url}/api/chat",
            json={
                "model": model,
                "messages": messages,
                "options": {
                    "temperature": request.temperature,
                    "top_p": request.top_p,
                    "num_predict": request.max_tokens,
                    "stop": request.stop or [],
                },
                "stream": False,
            }
        )
        response.raise_for_status()
        data = response.json()

        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
            created=int(time.time()),
            model=model,
            choices=[ChatChoice(
                index=0,
                message=Message(role="assistant", content=data.get("message", {}).get("content", "")),
            )],
            usage={
                "prompt_tokens": data.get("prompt_eval_count", 0),
                "completion_tokens": data.get("eval_count", 0),
                "total_tokens": data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
            }
        )

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        client = await self._get_client()
        model = request.model or "nomic-embed-text"

        inputs = [request.input] if isinstance(request.input, str) else request.input
        embeddings = []

        for i, text in enumerate(inputs):
            response = await client.post(
                f"{self.base_url}/api/embeddings",
                json={"model": model, "prompt": text}
            )
            response.raise_for_status()
            data = response.json()
            embeddings.append(EmbeddingData(index=i, embedding=data.get("embedding", [])))

        return EmbeddingResponse(
            data=embeddings,
            model=model,
            usage={"prompt_tokens": sum(len(t.split()) for t in inputs), "total_tokens": 0}
        )

    async def list_models(self) -> List[str]:
        try:
            client = await self._get_client()
            response = await client.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            data = response.json()
            return [m["name"] for m in data.get("models", [])]
        except Exception:
            return []

    async def health_check(self) -> bool:
        try:
            start = time.time()
            client = await self._get_client()
            response = await client.get(f"{self.base_url}/api/version")
            latency = (time.time() - start) * 1000

            if response.status_code == 200:
                self.info.status = BackendStatus.HEALTHY
                self.info.latency_ms = latency
                self.info.last_error = None
                self.info.models = await self.list_models()
                return True
        except Exception as e:
            self.info.last_error = str(e)

        self.info.status = BackendStatus.UNHEALTHY
        self.info.last_check = datetime.now()
        return False


class OpenAIBackend(ModelBackend):
    """OpenAI API backend"""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("openai", config)
        if openai is None:
            raise ImportError("openai package not installed")

        api_key = (config or {}).get("api_key") or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not provided")

        self.client = openai.AsyncOpenAI(api_key=api_key)
        self.default_model = (config or {}).get("model", "gpt-3.5-turbo")
        self.embedding_model = (config or {}).get("embedding_model", "text-embedding-3-small")

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        model = request.model or self.default_model

        # OpenAI deprecated completions, use chat instead
        response = await self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": request.prompt}],
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            stop=request.stop,
        )

        return CompletionResponse(
            id=response.id,
            created=response.created,
            model=response.model,
            choices=[CompletionChoice(
                index=0,
                text=response.choices[0].message.content or "",
                finish_reason=response.choices[0].finish_reason or "stop",
            )],
            usage={
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                "total_tokens": response.usage.total_tokens if response.usage else 0,
            }
        )

    async def chat_complete(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        model = request.model or self.default_model

        messages = [{"role": m.role, "content": m.content} for m in request.messages]

        response = await self.client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            stop=request.stop,
        )

        return ChatCompletionResponse(
            id=response.id,
            created=response.created,
            model=response.model,
            choices=[ChatChoice(
                index=0,
                message=Message(
                    role="assistant",
                    content=response.choices[0].message.content or ""
                ),
                finish_reason=response.choices[0].finish_reason or "stop",
            )],
            usage={
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                "total_tokens": response.usage.total_tokens if response.usage else 0,
            }
        )

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        model = request.model or self.embedding_model
        inputs = [request.input] if isinstance(request.input, str) else request.input

        response = await self.client.embeddings.create(
            model=model,
            input=inputs,
        )

        return EmbeddingResponse(
            data=[EmbeddingData(index=e.index, embedding=e.embedding) for e in response.data],
            model=response.model,
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "total_tokens": response.usage.total_tokens,
            }
        )

    async def list_models(self) -> List[str]:
        try:
            response = await self.client.models.list()
            return [m.id for m in response.data]
        except Exception:
            return [self.default_model]

    async def health_check(self) -> bool:
        try:
            start = time.time()
            await self.client.models.list()
            latency = (time.time() - start) * 1000

            self.info.status = BackendStatus.HEALTHY
            self.info.latency_ms = latency
            self.info.last_error = None
            self.info.models = await self.list_models()
            return True
        except Exception as e:
            self.info.status = BackendStatus.UNHEALTHY
            self.info.last_error = str(e)

        self.info.last_check = datetime.now()
        return False


class VLLMBackend(ModelBackend):
    """vLLM high-throughput backend"""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("vllm", config)
        self.base_url = (config or {}).get("base_url", "http://localhost:8001")
        self.default_model = (config or {}).get("model", "mistralai/Mistral-7B-Instruct-v0.3")
        self._client: Optional[Any] = None

    async def _get_client(self):
        if httpx is None:
            raise ImportError("httpx not installed")
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=120.0)
        return self._client

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        client = await self._get_client()
        model = request.model or self.default_model

        response = await client.post(
            f"{self.base_url}/v1/completions",
            json={
                "model": model,
                "prompt": request.prompt,
                "max_tokens": request.max_tokens,
                "temperature": request.temperature,
                "top_p": request.top_p,
                "stop": request.stop,
            }
        )
        response.raise_for_status()
        data = response.json()

        return CompletionResponse(
            id=data.get("id", f"cmpl-{uuid.uuid4().hex[:8]}"),
            created=data.get("created", int(time.time())),
            model=data.get("model", model),
            choices=[CompletionChoice(
                index=c["index"],
                text=c["text"],
                finish_reason=c.get("finish_reason", "stop"),
            ) for c in data.get("choices", [])],
            usage=data.get("usage", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
        )

    async def chat_complete(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        client = await self._get_client()
        model = request.model or self.default_model

        messages = [{"role": m.role, "content": m.content} for m in request.messages]

        response = await client.post(
            f"{self.base_url}/v1/chat/completions",
            json={
                "model": model,
                "messages": messages,
                "max_tokens": request.max_tokens,
                "temperature": request.temperature,
                "top_p": request.top_p,
                "stop": request.stop,
            }
        )
        response.raise_for_status()
        data = response.json()

        return ChatCompletionResponse(
            id=data.get("id", f"chatcmpl-{uuid.uuid4().hex[:8]}"),
            created=data.get("created", int(time.time())),
            model=data.get("model", model),
            choices=[ChatChoice(
                index=c["index"],
                message=Message(role="assistant", content=c["message"]["content"]),
                finish_reason=c.get("finish_reason", "stop"),
            ) for c in data.get("choices", [])],
            usage=data.get("usage", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
        )

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        raise HTTPException(status_code=501, detail="Embeddings not supported by vLLM backend")

    async def list_models(self) -> List[str]:
        try:
            client = await self._get_client()
            response = await client.get(f"{self.base_url}/v1/models")
            response.raise_for_status()
            data = response.json()
            return [m["id"] for m in data.get("data", [])]
        except Exception:
            return [self.default_model]

    async def health_check(self) -> bool:
        try:
            start = time.time()
            client = await self._get_client()
            response = await client.get(f"{self.base_url}/health")
            latency = (time.time() - start) * 1000

            if response.status_code == 200:
                self.info.status = BackendStatus.HEALTHY
                self.info.latency_ms = latency
                self.info.last_error = None
                self.info.models = await self.list_models()
                return True
        except Exception as e:
            self.info.last_error = str(e)

        self.info.status = BackendStatus.UNHEALTHY
        self.info.last_check = datetime.now()
        return False


class LocalTransformersBackend(ModelBackend):
    """Local transformers fallback backend"""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("transformers", config)
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers/torch not installed")

        self.model_path = (config or {}).get("model_path", "model_adapters/l4d2-mistral-v10plus-lora/final")
        self.base_model = (config or {}).get("base_model", "unsloth/mistral-7b-instruct-v0.3-bnb-4bit")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None
        self._load_lock = asyncio.Lock()

    async def _ensure_loaded(self):
        """Lazy load model"""
        if self.model is not None:
            return

        async with self._load_lock:
            if self.model is not None:
                return

            logger.info(f"Loading transformers model from {self.model_path}")

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.base_model,
                trust_remote_code=True
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Load model
            model_path = Path(self.model_path)
            if model_path.exists() and (model_path / "adapter_config.json").exists():
                base = AutoModelForCausalLM.from_pretrained(
                    self.base_model,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    device_map="auto" if self.device == "cuda" else None,
                    trust_remote_code=True,
                )
                self.model = PeftModel.from_pretrained(base, str(model_path))
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.base_model,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    device_map="auto" if self.device == "cuda" else None,
                    trust_remote_code=True,
                )

            logger.info(f"Model loaded on {self.device}")

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        await self._ensure_loaded()

        inputs = self.tokenizer(request.prompt, return_tensors="pt", truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature if request.temperature > 0 else 1.0,
                top_p=request.top_p,
                do_sample=request.temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated = outputs[0][inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(generated, skip_special_tokens=True)

        return CompletionResponse(
            id=f"cmpl-{uuid.uuid4().hex[:8]}",
            created=int(time.time()),
            model=self.model_path,
            choices=[CompletionChoice(index=0, text=text)],
            usage={
                "prompt_tokens": inputs["input_ids"].shape[1],
                "completion_tokens": len(generated),
                "total_tokens": inputs["input_ids"].shape[1] + len(generated),
            }
        )

    async def chat_complete(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        # Format as prompt
        prompt = ""
        for msg in request.messages:
            if msg.role == "system":
                prompt += f"<s>[INST] {msg.content} [/INST]"
            elif msg.role == "user":
                prompt += f"[INST] {msg.content} [/INST]"
            elif msg.role == "assistant":
                prompt += f" {msg.content}</s>"

        completion_req = CompletionRequest(
            prompt=prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
        )

        result = await self.complete(completion_req)

        return ChatCompletionResponse(
            id=result.id.replace("cmpl-", "chatcmpl-"),
            created=result.created,
            model=result.model,
            choices=[ChatChoice(
                index=0,
                message=Message(role="assistant", content=result.choices[0].text),
            )],
            usage=result.usage,
        )

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        raise HTTPException(status_code=501, detail="Embeddings not supported by transformers backend")

    async def list_models(self) -> List[str]:
        return [self.model_path]

    async def health_check(self) -> bool:
        try:
            if self.model is not None:
                self.info.status = BackendStatus.HEALTHY
                self.info.models = [self.model_path]
                return True
            else:
                # Not loaded yet, but that's ok
                self.info.status = BackendStatus.HEALTHY
                self.info.last_error = "Model not yet loaded (lazy loading)"
                return True
        except Exception as e:
            self.info.status = BackendStatus.UNHEALTHY
            self.info.last_error = str(e)
            return False


# =============================================================================
# Load Balancer
# =============================================================================

class LoadBalancer:
    """Round-robin load balancer with fallback"""

    def __init__(self, backends: List[ModelBackend]):
        self.backends = backends
        self._current_index = 0
        self._lock = asyncio.Lock()
        self._health_status: Dict[str, bool] = {}

    async def get_backend(self) -> Optional[ModelBackend]:
        """Get next healthy backend using round-robin"""
        async with self._lock:
            healthy = [b for b in self.backends if self._health_status.get(b.name, True)]

            if not healthy:
                # Try all backends as fallback
                return self.backends[0] if self.backends else None

            backend = healthy[self._current_index % len(healthy)]
            self._current_index = (self._current_index + 1) % len(healthy)
            return backend

    async def execute_with_fallback(
        self,
        operation: Callable[[ModelBackend], Any],
        max_retries: int = 2
    ) -> Tuple[Any, str]:
        """Execute operation with automatic fallback on failure"""
        last_error = None
        tried_backends: Set[str] = set()

        for attempt in range(max_retries + 1):
            backend = await self.get_backend()

            if backend is None:
                raise HTTPException(status_code=503, detail="No backends available")

            if backend.name in tried_backends:
                # Already tried this backend
                continue

            tried_backends.add(backend.name)

            try:
                result = await operation(backend)
                return result, backend.name
            except Exception as e:
                last_error = e
                logger.warning(f"Backend {backend.name} failed: {e}")
                self._health_status[backend.name] = False

        raise HTTPException(
            status_code=503,
            detail=f"All backends failed. Last error: {last_error}"
        )

    async def update_health(self):
        """Update health status of all backends"""
        for backend in self.backends:
            try:
                healthy = await backend.health_check()
                self._health_status[backend.name] = healthy
            except Exception as e:
                logger.error(f"Health check failed for {backend.name}: {e}")
                self._health_status[backend.name] = False

    def get_health_status(self) -> Dict[str, BackendInfo]:
        """Get health status of all backends"""
        return {b.name: b.info for b in self.backends}


# =============================================================================
# Model Server
# =============================================================================

class ModelServer:
    """Production model server with load balancing, caching, and metrics"""

    def __init__(
        self,
        backends: List[str],
        backend_configs: Dict[str, Dict[str, Any]] = None,
        cache_config: Dict[str, Any] = None,
        rate_limit_config: RateLimitConfig = None,
    ):
        self.start_time = time.time()
        self.backend_configs = backend_configs or {}

        # Initialize backends
        self.backends: List[ModelBackend] = []
        for name in backends:
            try:
                backend = self._create_backend(name)
                if backend:
                    self.backends.append(backend)
                    logger.info(f"Initialized backend: {name}")
            except Exception as e:
                logger.warning(f"Failed to initialize backend {name}: {e}")

        if not self.backends:
            raise ValueError("No backends could be initialized")

        # Initialize load balancer
        self.load_balancer = LoadBalancer(self.backends)

        # Initialize cache
        cache_config = cache_config or {}
        if cache_config.get("redis_url"):
            self.cache: CacheBackend = RedisCache(
                cache_config["redis_url"],
                cache_config.get("prefix", "l4d2_model_server:")
            )
            self.cache_type = "redis"
        else:
            self.cache = InMemoryCache(cache_config.get("max_size", 1000))
            self.cache_type = "memory"

        self.cache_ttl = cache_config.get("ttl", 3600)
        self.cache_enabled = cache_config.get("enabled", True)

        # Initialize rate limiter
        self.rate_limiter = RateLimiter(rate_limit_config or RateLimitConfig())

        # Initialize metrics
        self.metrics = Metrics()

        # Create FastAPI app
        self.app = self._create_app()

    def _create_backend(self, name: str) -> Optional[ModelBackend]:
        """Create backend by name"""
        config = self.backend_configs.get(name, {})

        if name == "ollama":
            return OllamaBackend(config)
        elif name == "openai":
            return OpenAIBackend(config)
        elif name == "vllm":
            return VLLMBackend(config)
        elif name == "transformers":
            return LocalTransformersBackend(config)
        else:
            logger.warning(f"Unknown backend: {name}")
            return None

    def _cache_key(self, prefix: str, data: dict) -> str:
        """Generate cache key from request data"""
        content = json.dumps(data, sort_keys=True)
        return f"{prefix}:{hashlib.sha256(content.encode()).hexdigest()[:32]}"

    def _create_app(self) -> FastAPI:
        """Create FastAPI application"""

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            # Startup: Run initial health checks
            await self.load_balancer.update_health()

            # Start background health check task
            async def health_check_loop():
                while True:
                    await asyncio.sleep(30)
                    await self.load_balancer.update_health()

            task = asyncio.create_task(health_check_loop())

            yield

            # Shutdown
            task.cancel()

        app = FastAPI(
            title="L4D2 Model Server",
            description="Production-ready model serving with load balancing and caching",
            version="1.0.0",
            lifespan=lifespan,
        )

        # Add CORS - restricted to localhost for security
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["http://localhost:8000", "http://localhost:3000", "http://127.0.0.1:8000", "http://127.0.0.1:3000"],
            allow_credentials=False,
            allow_methods=["GET", "POST", "DELETE"],
            allow_headers=["Content-Type", "Authorization", "X-API-Key"],
        )

        # Rate limiting middleware
        @app.middleware("http")
        async def rate_limit_middleware(request: Request, call_next):
            # Skip rate limiting for health/metrics endpoints
            if request.url.path in ["/health", "/metrics", "/v1/models"]:
                return await call_next(request)

            allowed, info = await self.rate_limiter.check_rate_limit(request)
            if not allowed:
                self.metrics.record_rate_limit()
                return JSONResponse(
                    status_code=429,
                    content={
                        "error": "Rate limit exceeded",
                        "retry_after": info["retry_after"],
                        "limit_type": info["limit_type"],
                    },
                    headers={"Retry-After": str(info["retry_after"])}
                )

            response = await call_next(request)

            # Add rate limit headers
            if "remaining_minute" in info:
                response.headers["X-RateLimit-Remaining-Minute"] = str(info["remaining_minute"])
                response.headers["X-RateLimit-Remaining-Hour"] = str(info["remaining_hour"])

            return response

        # Routes
        @app.get("/")
        async def root():
            return {
                "name": "L4D2 Model Server",
                "version": "1.0.0",
                "backends": [b.name for b in self.backends],
            }

        @app.get("/health", response_model=HealthResponse)
        async def health():
            await self.load_balancer.update_health()
            status_map = self.load_balancer.get_health_status()

            backends = []
            for name, info in status_map.items():
                backends.append(BackendHealth(
                    name=name,
                    healthy=info.status == BackendStatus.HEALTHY,
                    latency_ms=info.latency_ms,
                    error=info.last_error,
                    last_check=info.last_check.isoformat() if info.last_check else None,
                ))

            all_healthy = all(b.healthy for b in backends)

            return HealthResponse(
                status="healthy" if all_healthy else "degraded",
                backends=backends,
                cache_enabled=self.cache_enabled,
                cache_type=self.cache_type,
                uptime_seconds=round(time.time() - self.start_time, 2),
            )

        @app.get("/metrics", response_model=MetricsResponse)
        async def metrics():
            return MetricsResponse(**self.metrics.get_metrics())

        @app.get("/v1/models", response_model=ModelListResponse)
        async def list_models():
            models = []
            seen = set()

            for backend in self.backends:
                for model in backend.info.models:
                    if model not in seen:
                        seen.add(model)
                        models.append(ModelInfo(
                            id=model,
                            created=int(self.start_time),
                            owned_by=backend.name,
                            backend=backend.name,
                        ))

            return ModelListResponse(data=models)

        @app.post("/v1/completions", response_model=CompletionResponse)
        async def completions(request: CompletionRequest):
            start = time.time()

            # Check cache
            cache_key = self._cache_key("completion", request.model_dump())
            if self.cache_enabled:
                cached = await self.cache.get(cache_key)
                if cached:
                    self.metrics.record_cache_hit()
                    return CompletionResponse(**json.loads(cached))
                self.metrics.record_cache_miss()

            # Execute with load balancing
            async def do_complete(backend: ModelBackend):
                return await backend.complete(request)

            try:
                result, backend_name = await self.load_balancer.execute_with_fallback(do_complete)
                latency = (time.time() - start) * 1000
                self.metrics.record_request(backend_name, latency)

                # Cache result
                if self.cache_enabled:
                    await self.cache.set(cache_key, result.model_dump_json(), self.cache_ttl)

                return result
            except Exception as e:
                latency = (time.time() - start) * 1000
                self.metrics.record_request("unknown", latency, error=True)
                raise

        @app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
        async def chat_completions(request: ChatCompletionRequest):
            start = time.time()

            # Check cache
            cache_key = self._cache_key("chat", request.model_dump())
            if self.cache_enabled:
                cached = await self.cache.get(cache_key)
                if cached:
                    self.metrics.record_cache_hit()
                    return ChatCompletionResponse(**json.loads(cached))
                self.metrics.record_cache_miss()

            # Execute with load balancing
            async def do_chat(backend: ModelBackend):
                return await backend.chat_complete(request)

            try:
                result, backend_name = await self.load_balancer.execute_with_fallback(do_chat)
                latency = (time.time() - start) * 1000
                self.metrics.record_request(backend_name, latency)

                # Cache result
                if self.cache_enabled:
                    await self.cache.set(cache_key, result.model_dump_json(), self.cache_ttl)

                return result
            except Exception as e:
                latency = (time.time() - start) * 1000
                self.metrics.record_request("unknown", latency, error=True)
                raise

        @app.post("/v1/embeddings", response_model=EmbeddingResponse)
        async def embeddings(request: EmbeddingRequest):
            start = time.time()

            # Check cache
            cache_key = self._cache_key("embed", {
                "input": request.input if isinstance(request.input, str) else list(request.input),
                "model": request.model,
            })
            if self.cache_enabled:
                cached = await self.cache.get(cache_key)
                if cached:
                    self.metrics.record_cache_hit()
                    return EmbeddingResponse(**json.loads(cached))
                self.metrics.record_cache_miss()

            # Execute with load balancing
            async def do_embed(backend: ModelBackend):
                return await backend.embed(request)

            try:
                result, backend_name = await self.load_balancer.execute_with_fallback(do_embed)
                latency = (time.time() - start) * 1000
                self.metrics.record_request(backend_name, latency)

                # Cache result
                if self.cache_enabled:
                    await self.cache.set(cache_key, result.model_dump_json(), self.cache_ttl)

                return result
            except Exception as e:
                latency = (time.time() - start) * 1000
                self.metrics.record_request("unknown", latency, error=True)
                raise

        @app.delete("/cache")
        async def clear_cache():
            await self.cache.clear()
            return {"status": "ok", "message": "Cache cleared"}

        return app

    def run(self, host: str = "0.0.0.0", port: int = 8000, **kwargs):
        """Run the server"""
        logger.info(f"Starting Model Server on {host}:{port}")
        logger.info(f"Backends: {[b.name for b in self.backends]}")
        logger.info(f"Cache: {self.cache_type} (enabled={self.cache_enabled})")
        uvicorn.run(self.app, host=host, port=port, **kwargs)


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="L4D2 Production Model Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python model_server.py --backends ollama,openai --port 8000
  python model_server.py --backends vllm --vllm-url http://localhost:8001
  python model_server.py --backends ollama --redis-url redis://localhost:6379
  python model_server.py --config configs/model_server.yaml
        """
    )

    parser.add_argument(
        "--backends", "-b",
        type=str,
        default="ollama",
        help="Comma-separated list of backends (ollama,openai,vllm,transformers)"
    )
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    parser.add_argument("--port", "-p", type=int, default=8000, help="Server port")

    # Backend-specific options
    parser.add_argument("--ollama-url", type=str, default="http://localhost:11434",
                        help="Ollama API URL")
    parser.add_argument("--ollama-model", type=str, default="l4d2-code-v10plus",
                        help="Default Ollama model")
    parser.add_argument("--vllm-url", type=str, default="http://localhost:8001",
                        help="vLLM API URL")
    parser.add_argument("--vllm-model", type=str, help="Default vLLM model")
    parser.add_argument("--openai-model", type=str, default="gpt-3.5-turbo",
                        help="Default OpenAI model")
    parser.add_argument("--transformers-model", type=str,
                        default="model_adapters/l4d2-mistral-v10plus-lora/final",
                        help="Path to transformers model")

    # Cache options
    parser.add_argument("--redis-url", type=str, help="Redis URL for caching")
    parser.add_argument("--cache-ttl", type=int, default=3600, help="Cache TTL in seconds")
    parser.add_argument("--cache-size", type=int, default=1000, help="In-memory cache size")
    parser.add_argument("--no-cache", action="store_true", help="Disable caching")

    # Rate limiting
    parser.add_argument("--rate-limit-minute", type=int, default=60,
                        help="Requests per minute limit")
    parser.add_argument("--rate-limit-hour", type=int, default=1000,
                        help="Requests per hour limit")

    # Config file
    parser.add_argument("--config", "-c", type=str, help="Path to YAML config file")

    # Logging
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        default="INFO", help="Log level")

    args = parser.parse_args()

    # Set log level
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    # Load config file if specified
    config = {}
    if args.config:
        try:
            config = safe_read_yaml(args.config, PROJECT_ROOT)
            logger.info(f"Loaded config from {args.config}")
        except Exception as e:
            logger.warning(f"Failed to load config: {e}")

    # Parse backends
    backends = config.get("backends", args.backends.split(","))

    # Build backend configs
    backend_configs = config.get("backend_configs", {})

    if "ollama" not in backend_configs:
        backend_configs["ollama"] = {
            "base_url": args.ollama_url,
            "model": args.ollama_model,
        }

    if "vllm" not in backend_configs:
        backend_configs["vllm"] = {
            "base_url": args.vllm_url,
            "model": args.vllm_model,
        }

    if "openai" not in backend_configs:
        backend_configs["openai"] = {
            "model": args.openai_model,
        }

    if "transformers" not in backend_configs:
        backend_configs["transformers"] = {
            "model_path": args.transformers_model,
        }

    # Build cache config
    cache_config = config.get("cache", {
        "enabled": not args.no_cache,
        "redis_url": args.redis_url,
        "ttl": args.cache_ttl,
        "max_size": args.cache_size,
    })

    # Build rate limit config
    rate_limit_config = RateLimitConfig(
        requests_per_minute=config.get("rate_limit", {}).get("per_minute", args.rate_limit_minute),
        requests_per_hour=config.get("rate_limit", {}).get("per_hour", args.rate_limit_hour),
    )

    # Create and run server
    try:
        server = ModelServer(
            backends=backends,
            backend_configs=backend_configs,
            cache_config=cache_config,
            rate_limit_config=rate_limit_config,
        )
        server.run(host=args.host, port=args.port)
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
