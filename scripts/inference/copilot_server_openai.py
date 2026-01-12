#!/usr/bin/env python3
"""
L4D2 Copilot Inference Server - OpenAI Version

Uses the V7 fine-tuned OpenAI model for production code generation.
Provides API endpoints compatible with VSCode extensions and IDE integrations.

Usage:
    # With Doppler (recommended)
    doppler run --project local-mac-work --config dev_personal -- python scripts/inference/copilot_server_openai.py

    # With environment variable
    OPENAI_API_KEY=sk-... python scripts/inference/copilot_server_openai.py

    # Custom port
    python scripts/inference/copilot_server_openai.py --port 8080
"""

import os
import sys
import time
import logging
import argparse
from typing import List
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

PROJECT_ROOT = Path(__file__).parent.parent.parent

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, field_validator
    import uvicorn
except ImportError:
    print("FastAPI not installed. Run: pip install fastapi uvicorn")
    sys.exit(1)

try:
    from openai import OpenAI
except ImportError:
    print("OpenAI not installed. Run: pip install openai")
    sys.exit(1)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# V7 Fine-tuned Model Configuration
MODEL_ID = "ft:gpt-4o-mini-2024-07-18:highencodelearning:l4d2-sourcemod-v7:CvTBCVPi"

SYSTEM_PROMPT = """You are an expert SourcePawn and VScript developer for Left 4 Dead 2 SourceMod plugins.
Write clean, well-documented code with proper error handling. Use correct L4D2 APIs and events.

CRITICAL L4D2 API RULES:
- Use GetRandomFloat() and GetRandomInt(), NOT RandomFloat() or RandomInt()
- Use lunge_pounce event for Hunter pounces, NOT pounce
- Use tongue_grab for Smoker, NOT smoker_tongue_grab
- Use player_now_it for bile, NOT boomer_vomit
- Use charger_carry_start for Charger, NOT charger_grab
- Use m_flLaggedMovementValue for speed, NOT m_flSpeed or m_flMaxSpeed"""


class CompletionRequest(BaseModel):
    """Request model for code completion"""
    prompt: str
    max_tokens: int = 1024
    temperature: float = 0.1
    language: str = "sourcepawn"
    stop_sequences: List[str] = []


class CompletionResponse(BaseModel):
    """Response model for code completion"""
    completion: str
    tokens_used: int
    model: str
    cost: float
    timestamp: float


class ChatMessage(BaseModel):
    """Chat message model"""
    role: str
    content: str


class ChatRequest(BaseModel):
    """Request model for chat completion"""
    messages: List[ChatMessage]
    max_tokens: int = 2048
    temperature: float = 0.1


class PluginGenerationRequest(BaseModel):
    """Request model for plugin generation"""
    description: str
    max_tokens: int = 2048
    temperature: float = 0.1

    @field_validator('description')
    @classmethod
    def description_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError('Description cannot be empty')
        return v.strip()


class CopilotServerOpenAI:
    """OpenAI-powered copilot server using V7 fine-tuned model"""

    def __init__(self, model_id: str = MODEL_ID):
        self.model_id = model_id
        self.client = None
        self._init_client()

        # Pricing (fine-tuned GPT-4o-mini)
        self.pricing = {
            "input_per_1m": 0.30,
            "output_per_1m": 1.20,
        }

        # Create FastAPI app
        self.app = FastAPI(
            title="L4D2 Copilot API (OpenAI V7)",
            description="AI-powered L4D2 SourcePawn code completion using fine-tuned GPT-4o-mini",
            version="2.0.0"
        )

        # Add CORS - restricted to localhost for security
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["http://localhost:8000", "http://localhost:3000", "http://127.0.0.1:8000", "http://127.0.0.1:3000"],
            allow_credentials=False,
            allow_methods=["GET", "POST"],
            allow_headers=["Content-Type"],
        )

        self._setup_routes()

    def _init_client(self):
        """Initialize OpenAI client"""
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            logger.error("OPENAI_API_KEY environment variable not set")
            logger.info("Use: doppler run --project local-mac-work --config dev_personal -- python ...")
            raise ValueError("OPENAI_API_KEY required")

        self.client = OpenAI(api_key=api_key)
        logger.info(f"OpenAI client initialized, using model: {self.model_id}")

    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate API cost"""
        return (input_tokens * self.pricing["input_per_1m"] / 1_000_000 +
                output_tokens * self.pricing["output_per_1m"] / 1_000_000)

    def _setup_routes(self):
        """Setup API routes"""

        @self.app.get("/")
        async def root():
            return {
                "message": "L4D2 Copilot API (OpenAI V7) is running",
                "model": self.model_id,
                "version": "2.0.0"
            }

        @self.app.get("/health")
        async def health():
            return {
                "status": "healthy",
                "model": self.model_id,
                "backend": "openai"
            }

        @self.app.get("/v1/models")
        async def list_models():
            return {
                "models": [{
                    "id": self.model_id,
                    "object": "model",
                    "owned_by": "highencodelearning",
                    "permission": []
                }]
            }

        @self.app.post("/v1/complete", response_model=CompletionResponse)
        async def complete_code(request: CompletionRequest):
            """Generate code completion"""
            try:
                # Build language-specific system prompt
                system_content = SYSTEM_PROMPT
                if request.language.lower() == "vscript":
                    system_content += "\nFocus on VScript/Squirrel syntax for L4D2 Director scripts."

                # Call OpenAI API
                start_time = time.time()
                response = self.client.chat.completions.create(
                    model=self.model_id,
                    messages=[
                        {"role": "system", "content": system_content},
                        {"role": "user", "content": request.prompt}
                    ],
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    stop=request.stop_sequences if request.stop_sequences else None
                )
                elapsed = time.time() - start_time

                # Extract response
                completion = response.choices[0].message.content
                if completion is None:
                    raise HTTPException(status_code=502, detail="Model returned empty response")

                usage = response.usage
                cost = self._calculate_cost(usage.prompt_tokens, usage.completion_tokens)

                logger.info(f"Completion generated in {elapsed:.2f}s, cost: ${cost:.4f}")

                return CompletionResponse(
                    completion=completion,
                    tokens_used=usage.total_tokens,
                    model=self.model_id,
                    cost=cost,
                    timestamp=time.time()
                )

            except HTTPException:
                raise  # Preserve HTTP errors (502, etc.)
            except Exception as e:
                logger.error(f"Completion error: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail="Code completion failed. Check server logs for details.")

        @self.app.post("/v1/chat/completions")
        async def chat_completions(request: ChatRequest):
            """OpenAI-compatible chat completions endpoint"""
            try:
                # Build messages with system prompt
                messages = [{"role": "system", "content": SYSTEM_PROMPT}]
                messages.extend([{"role": m.role, "content": m.content} for m in request.messages])

                # Call OpenAI API
                response = self.client.chat.completions.create(
                    model=self.model_id,
                    messages=messages,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature
                )

                # Extract and validate response content
                content = response.choices[0].message.content
                if content is None:
                    raise HTTPException(status_code=502, detail="Model returned empty response")

                # Return OpenAI-compatible response
                return {
                    "id": response.id,
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": self.model_id,
                    "choices": [{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": content
                        },
                        "finish_reason": response.choices[0].finish_reason
                    }],
                    "usage": {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens
                    }
                }

            except HTTPException:
                raise  # Preserve HTTP errors (502, etc.)
            except Exception as e:
                logger.error(f"Chat error: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail="Chat completion failed. Check server logs for details.")

        @self.app.post("/v1/generate-plugin")
        async def generate_plugin(request: PluginGenerationRequest):
            """Generate a complete plugin from a description"""
            try:
                prompt = f"Write a complete L4D2 SourcePawn plugin that: {request.description}"

                response = self.client.chat.completions.create(
                    model=self.model_id,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=request.max_tokens,
                    temperature=request.temperature
                )

                content = response.choices[0].message.content
                if content is None:
                    raise HTTPException(status_code=502, detail="Model returned empty response")

                return {
                    "plugin_code": content,
                    "model": self.model_id,
                    "tokens_used": response.usage.total_tokens
                }

            except HTTPException:
                raise  # Preserve HTTP errors (400, 502, etc.)
            except Exception as e:
                logger.error(f"Plugin generation error: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail="Plugin generation failed. Check server logs for details.")

    def run(self, host: str = "127.0.0.1", port: int = 8000):
        """Run the server (defaults to localhost for security)"""
        logger.info(f"Starting L4D2 Copilot Server on {host}:{port}")
        logger.info(f"Model: {self.model_id}")
        logger.info("Endpoints:")
        logger.info("  POST /v1/complete - Code completion")
        logger.info("  POST /v1/chat/completions - Chat (OpenAI-compatible)")
        logger.info("  POST /v1/generate-plugin - Full plugin generation")
        uvicorn.run(self.app, host=host, port=port)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="L4D2 Copilot Server (OpenAI V7)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Server host (use 0.0.0.0 for network access)")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--model", type=str, default=MODEL_ID, help="Model ID to use")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       default="INFO", help="Log level")

    args = parser.parse_args()

    # Set log level
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    try:
        server = CopilotServerOpenAI(model_id=args.model)
        server.run(host=args.host, port=args.port)
    except ValueError as e:
        logger.error(f"Failed to start server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
