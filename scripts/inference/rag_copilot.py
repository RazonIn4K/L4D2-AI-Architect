#!/usr/bin/env python3
"""
L4D2 RAG-Enhanced Copilot

Combines retrieval-augmented generation (RAG) with fine-tuned models
for improved code completion accuracy. Retrieves relevant examples from
the training data to provide context for generation.

Usage:
    # Interactive chat with RAG
    python rag_copilot.py chat --backend ollama

    # Single completion
    python rag_copilot.py complete "How do I detect when a tank spawns?"

    # Start RAG server
    python rag_copilot.py server --port 8080
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

# Add scripts to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.security import safe_read_json, safe_path

PROJECT_ROOT = Path(__file__).parent.parent.parent

# Lazy imports for optional dependencies
faiss = None
np = None
SentenceTransformer = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class RAGConfig:
    """Configuration for RAG copilot"""
    # Paths
    index_path: str = "data/embeddings/faiss_index.bin"
    metadata_path: str = "data/embeddings/metadata.json"

    # Retrieval settings
    top_k: int = 3
    similarity_threshold: float = 0.3

    # Model settings
    embedding_model: str = "all-MiniLM-L6-v2"
    backend: str = "ollama"  # ollama or openai
    llm_model: str = "l4d2-code-v10plus"  # For Ollama
    openai_model: str = "gpt-4o-mini"  # For OpenAI
    temperature: float = 0.7
    max_tokens: int = 1024

    # Server settings
    host: str = "0.0.0.0"
    port: int = 8080


# ============================================================================
# FAISS Index Manager
# ============================================================================

class FAISSIndexManager:
    """Manages FAISS index loading and similarity search"""

    def __init__(self, index_path: Path, metadata_path: Path):
        """Initialize the FAISS index manager.

        Args:
            index_path: Path to the FAISS index file
            metadata_path: Path to the metadata JSON file
        """
        global faiss, np

        # Lazy import faiss and numpy
        try:
            import faiss as _faiss
            import numpy as _np
            faiss = _faiss
            np = _np
        except ImportError:
            raise ImportError(
                "FAISS and NumPy are required. Install with:\n"
                "  pip install faiss-cpu numpy\n"
                "Or for GPU support:\n"
                "  pip install faiss-gpu numpy"
            )

        self.index_path = index_path
        self.metadata_path = metadata_path
        self.index = None
        self.metadata = None
        self.examples = []

        self._load_index()
        self._load_metadata()

    def _load_index(self):
        """Load the FAISS index from disk"""
        if not self.index_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found at {self.index_path}\n"
                "Generate embeddings first with: python scripts/training/generate_embeddings.py"
            )

        logger.info(f"Loading FAISS index from {self.index_path}")
        self.index = faiss.read_index(str(self.index_path))
        logger.info(f"Loaded index with {self.index.ntotal} vectors")

    def _load_metadata(self):
        """Load metadata from disk"""
        if not self.metadata_path.exists():
            raise FileNotFoundError(
                f"Metadata not found at {self.metadata_path}\n"
                "Generate embeddings first with: python scripts/training/generate_embeddings.py"
            )

        logger.info(f"Loading metadata from {self.metadata_path}")
        self.metadata = safe_read_json(str(self.metadata_path), PROJECT_ROOT)
        self.examples = self.metadata.get("examples", [])
        logger.info(f"Loaded metadata for {len(self.examples)} examples")

    def search(self, query_embedding: "np.ndarray", top_k: int = 3) -> List[Dict[str, Any]]:
        """Search for similar examples.

        Args:
            query_embedding: Query embedding vector (1D or 2D array)
            top_k: Number of results to return

        Returns:
            List of retrieved examples with scores
        """
        # Ensure query is 2D
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)

        # Search the index
        distances, indices = self.index.search(query_embedding.astype(np.float32), top_k)

        # Build results
        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx >= 0 and idx < len(self.examples):
                example = self.examples[idx].copy()
                # Convert L2 distance to similarity score (higher is better)
                example["similarity_score"] = float(1.0 / (1.0 + dist))
                example["rank"] = i + 1
                results.append(example)

        return results


# ============================================================================
# Embedding Manager
# ============================================================================

class EmbeddingManager:
    """Manages sentence embeddings using sentence-transformers"""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize the embedding manager.

        Args:
            model_name: Name of the sentence-transformers model to use
        """
        global SentenceTransformer

        try:
            from sentence_transformers import SentenceTransformer as _ST
            SentenceTransformer = _ST
        except ImportError:
            raise ImportError(
                "sentence-transformers is required. Install with:\n"
                "  pip install sentence-transformers"
            )

        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        logger.info("Embedding model loaded")

    def encode(self, text: str) -> "np.ndarray":
        """Encode text to embedding vector.

        Args:
            text: Text to encode

        Returns:
            Embedding vector as numpy array
        """
        return self.model.encode(text, convert_to_numpy=True)

    def encode_batch(self, texts: List[str]) -> "np.ndarray":
        """Encode multiple texts to embedding vectors.

        Args:
            texts: List of texts to encode

        Returns:
            Embedding matrix as numpy array
        """
        return self.model.encode(texts, convert_to_numpy=True)


# ============================================================================
# LLM Backends
# ============================================================================

class OllamaBackend:
    """Ollama backend for local LLM inference"""

    def __init__(self, model: str = "l4d2-code-v10plus"):
        """Initialize Ollama backend.

        Args:
            model: Ollama model name
        """
        self.model = model
        self._check_ollama()

    def _check_ollama(self):
        """Check if Ollama is available"""
        if not shutil.which("ollama"):
            raise RuntimeError(
                "Ollama not found. Install from https://ollama.ai"
            )

    def is_model_available(self) -> bool:
        """Check if model is available in Ollama"""
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True
            )
            return self.model in result.stdout
        except Exception:
            return False

    def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 1024) -> str:
        """Generate completion using Ollama.

        Args:
            prompt: The prompt to complete
            temperature: Generation temperature
            max_tokens: Maximum tokens to generate

        Returns:
            Generated text
        """
        try:
            result = subprocess.run(
                ["ollama", "run", self.model, prompt],
                capture_output=True,
                text=True,
                timeout=120
            )
            return result.stdout.strip()
        except subprocess.TimeoutExpired:
            return "Error: Generation timed out"
        except Exception as e:
            return f"Error: {e}"


class OpenAIBackend:
    """OpenAI backend for API-based LLM inference"""

    def __init__(self, model: str = "gpt-4o-mini"):
        """Initialize OpenAI backend.

        Args:
            model: OpenAI model name
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "OpenAI package is required. Install with:\n"
                "  pip install openai"
            )

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable not set.\n"
                "Set it with: export OPENAI_API_KEY='your-api-key'"
            )

        self.client = OpenAI(api_key=api_key)
        self.model = model

    def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 1024) -> str:
        """Generate completion using OpenAI API.

        Args:
            prompt: The prompt to complete
            temperature: Generation temperature
            max_tokens: Maximum tokens to generate

        Returns:
            Generated text
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert SourcePawn and VScript developer "
                            "specializing in Left 4 Dead 2 modding. Write clean, "
                            "efficient, and well-documented code."
                        )
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error: {e}"


# ============================================================================
# RAG Copilot
# ============================================================================

class RAGCopilot:
    """RAG-enhanced copilot for L4D2 code generation"""

    def __init__(self, config: RAGConfig):
        """Initialize the RAG copilot.

        Args:
            config: RAG configuration
        """
        self.config = config

        # Initialize components
        logger.info("Initializing RAG Copilot...")

        # Load FAISS index
        index_path = safe_path(config.index_path, PROJECT_ROOT)
        metadata_path = safe_path(config.metadata_path, PROJECT_ROOT)
        self.index_manager = FAISSIndexManager(index_path, metadata_path)

        # Load embedding model
        self.embedding_manager = EmbeddingManager(config.embedding_model)

        # Initialize LLM backend
        if config.backend == "ollama":
            self.llm = OllamaBackend(config.llm_model)
        elif config.backend == "openai":
            self.llm = OpenAIBackend(config.openai_model)
        else:
            raise ValueError(f"Unknown backend: {config.backend}")

        logger.info("RAG Copilot initialized successfully")

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """Retrieve relevant examples for a query.

        Args:
            query: The user's query
            top_k: Number of examples to retrieve (uses config default if None)

        Returns:
            List of retrieved examples
        """
        if top_k is None:
            top_k = self.config.top_k

        # Encode query
        query_embedding = self.embedding_manager.encode(query)

        # Search for similar examples
        results = self.index_manager.search(query_embedding, top_k)

        # Filter by similarity threshold
        filtered = [
            r for r in results
            if r.get("similarity_score", 0) >= self.config.similarity_threshold
        ]

        return filtered

    def build_prompt(self, query: str, retrieved_examples: List[Dict[str, Any]]) -> str:
        """Build an enhanced prompt with retrieved context.

        Args:
            query: The user's query
            retrieved_examples: List of retrieved examples

        Returns:
            Enhanced prompt string
        """
        # Build context from retrieved examples
        context_parts = []

        if retrieved_examples:
            context_parts.append("Here are some relevant examples from the codebase:\n")

            for i, example in enumerate(retrieved_examples, 1):
                prompt = example.get("prompt", "")
                response = example.get("response_preview", "")
                score = example.get("similarity_score", 0)

                context_parts.append(f"--- Example {i} (similarity: {score:.2f}) ---")
                context_parts.append(f"Task: {prompt}")
                context_parts.append(f"Code:\n{response}")
                context_parts.append("")

        # Build the full prompt
        prompt_parts = [
            "You are an expert SourcePawn and VScript developer for Left 4 Dead 2.",
            "Generate clean, well-documented code based on the user's request.",
            ""
        ]

        if context_parts:
            prompt_parts.extend(context_parts)
            prompt_parts.append("---\n")

        prompt_parts.extend([
            f"User Request: {query}",
            "",
            "Please provide a complete implementation:"
        ])

        return "\n".join(prompt_parts)

    def complete(
        self,
        query: str,
        top_k: Optional[int] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        show_retrieved: bool = False
    ) -> Dict[str, Any]:
        """Generate a RAG-enhanced completion.

        Args:
            query: The user's query
            top_k: Number of examples to retrieve
            temperature: Generation temperature
            max_tokens: Maximum tokens to generate
            show_retrieved: Whether to include retrieved examples in response

        Returns:
            Dictionary with completion and metadata
        """
        start_time = time.time()

        # Retrieve relevant examples
        retrieved = self.retrieve(query, top_k)
        retrieval_time = time.time() - start_time

        # Build enhanced prompt
        prompt = self.build_prompt(query, retrieved)

        # Generate completion
        generation_start = time.time()
        completion = self.llm.generate(
            prompt,
            temperature=temperature or self.config.temperature,
            max_tokens=max_tokens or self.config.max_tokens
        )
        generation_time = time.time() - generation_start

        total_time = time.time() - start_time

        # Build response
        response = {
            "completion": completion,
            "query": query,
            "num_retrieved": len(retrieved),
            "retrieval_time_ms": round(retrieval_time * 1000, 2),
            "generation_time_ms": round(generation_time * 1000, 2),
            "total_time_ms": round(total_time * 1000, 2),
            "backend": self.config.backend,
            "model": (
                self.config.llm_model if self.config.backend == "ollama"
                else self.config.openai_model
            )
        }

        if show_retrieved:
            response["retrieved_examples"] = retrieved

        return response

    def chat_interactive(self):
        """Start an interactive chat session with RAG support"""
        print(f"\nL4D2 RAG Copilot ({self.config.backend})")
        print(f"Using {self.config.top_k} retrieved examples per query")
        print("Commands: /quit, /top_k N, /temp N, /examples")
        print("-" * 50)

        history = []
        show_examples = False

        while True:
            try:
                user_input = input("\nYou: ").strip()

                if not user_input:
                    continue

                # Handle commands
                if user_input.startswith("/"):
                    parts = user_input.split()
                    cmd = parts[0].lower()

                    if cmd in ["/quit", "/exit", "/q"]:
                        break
                    elif cmd == "/top_k" and len(parts) > 1:
                        try:
                            self.config.top_k = int(parts[1])
                            print(f"Set top_k to {self.config.top_k}")
                        except ValueError:
                            print("Invalid number")
                    elif cmd == "/temp" and len(parts) > 1:
                        try:
                            self.config.temperature = float(parts[1])
                            print(f"Set temperature to {self.config.temperature}")
                        except ValueError:
                            print("Invalid number")
                    elif cmd == "/examples":
                        show_examples = not show_examples
                        print(f"Show examples: {show_examples}")
                    elif cmd == "/help":
                        print("Commands:")
                        print("  /quit - Exit the chat")
                        print("  /top_k N - Set number of retrieved examples")
                        print("  /temp N - Set generation temperature")
                        print("  /examples - Toggle showing retrieved examples")
                    else:
                        print(f"Unknown command: {cmd}")
                    continue

                # Generate RAG-enhanced response
                print("\nSearching for relevant examples...")
                result = self.complete(user_input, show_retrieved=show_examples)

                if show_examples and result.get("retrieved_examples"):
                    print(f"\n[Retrieved {result['num_retrieved']} examples]")
                    for ex in result["retrieved_examples"]:
                        print(f"  - {ex.get('prompt', '')[:60]}... (score: {ex.get('similarity_score', 0):.2f})")

                print(f"\nAssistant: {result['completion']}")
                print(f"\n[{result['total_time_ms']:.0f}ms total, {result['num_retrieved']} examples used]")

                # Store in history
                history.append({"query": user_input, "response": result})

            except KeyboardInterrupt:
                break
            except EOFError:
                break

        print("\nGoodbye!")


# ============================================================================
# FastAPI Server
# ============================================================================

def create_server(config: RAGConfig):
    """Create FastAPI server for RAG completions"""
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.middleware.cors import CORSMiddleware
        from pydantic import BaseModel, Field
    except ImportError:
        raise ImportError(
            "FastAPI is required for server mode. Install with:\n"
            "  pip install fastapi uvicorn"
        )

    # Initialize RAG copilot
    copilot = RAGCopilot(config)

    # Create FastAPI app
    app = FastAPI(
        title="L4D2 RAG Copilot API",
        description="RAG-enhanced code completion for Left 4 Dead 2 modding",
        version="1.0.0"
    )

    # Add CORS - restricted to localhost for security
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:8080", "http://localhost:3000", "http://127.0.0.1:8080", "http://127.0.0.1:3000"],
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    # Request/Response models
    class CompletionRequest(BaseModel):
        query: str = Field(..., description="The coding question or task")
        top_k: int = Field(default=3, ge=1, le=10, description="Number of examples to retrieve")
        temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Generation temperature")
        max_tokens: int = Field(default=1024, ge=1, le=4096, description="Maximum tokens to generate")
        include_examples: bool = Field(default=False, description="Include retrieved examples in response")

    class RetrievalRequest(BaseModel):
        query: str = Field(..., description="The query to search for")
        top_k: int = Field(default=5, ge=1, le=20, description="Number of examples to retrieve")

    class CompletionResponse(BaseModel):
        completion: str
        query: str
        num_retrieved: int
        retrieval_time_ms: float
        generation_time_ms: float
        total_time_ms: float
        backend: str
        model: str
        retrieved_examples: Optional[List[Dict[str, Any]]] = None

    # Routes
    @app.get("/")
    async def root():
        return {
            "message": "L4D2 RAG Copilot API",
            "backend": config.backend,
            "model": config.llm_model if config.backend == "ollama" else config.openai_model,
            "top_k": config.top_k
        }

    @app.get("/health")
    async def health():
        return {
            "status": "healthy",
            "index_loaded": copilot.index_manager.index is not None,
            "num_examples": len(copilot.index_manager.examples),
            "embedding_model": config.embedding_model,
            "backend": config.backend
        }

    @app.post("/v1/complete", response_model=CompletionResponse)
    async def complete(request: CompletionRequest):
        """Generate a RAG-enhanced code completion"""
        try:
            result = copilot.complete(
                query=request.query,
                top_k=request.top_k,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                show_retrieved=request.include_examples
            )
            return CompletionResponse(**result)
        except Exception as e:
            logger.error(f"Completion error: {e}")
            raise HTTPException(status_code=500, detail="Completion failed. Check server logs for details.")

    @app.post("/v1/retrieve")
    async def retrieve(request: RetrievalRequest):
        """Retrieve similar examples without generation"""
        try:
            examples = copilot.retrieve(request.query, request.top_k)
            return {
                "query": request.query,
                "num_results": len(examples),
                "examples": examples
            }
        except Exception as e:
            logger.error(f"Retrieval error: {e}")
            raise HTTPException(status_code=500, detail="Retrieval failed. Check server logs for details.")

    @app.get("/v1/stats")
    async def stats():
        """Get index statistics"""
        return {
            "num_examples": len(copilot.index_manager.examples),
            "embedding_model": copilot.index_manager.metadata.get("model_name"),
            "embedding_dim": copilot.index_manager.metadata.get("embedding_dim"),
            "created_at": copilot.index_manager.metadata.get("created_at"),
            "index_vectors": copilot.index_manager.index.ntotal
        }

    return app


# ============================================================================
# CLI Commands
# ============================================================================

def cmd_complete(args, config: RAGConfig):
    """Handle complete command"""
    config.top_k = args.top_k
    config.temperature = args.temperature
    config.backend = args.backend

    if args.backend == "openai" and args.model:
        config.openai_model = args.model
    elif args.backend == "ollama" and args.model:
        config.llm_model = args.model

    copilot = RAGCopilot(config)

    result = copilot.complete(
        args.query,
        show_retrieved=args.show_examples
    )

    if args.show_examples and result.get("retrieved_examples"):
        print("\n--- Retrieved Examples ---")
        for ex in result["retrieved_examples"]:
            print(f"\nPrompt: {ex.get('prompt', '')[:100]}...")
            print(f"Score: {ex.get('similarity_score', 0):.3f}")
        print("\n--- Completion ---")

    print(result["completion"])

    if args.verbose:
        print(f"\n[{result['num_retrieved']} examples, {result['total_time_ms']:.0f}ms]")


def cmd_chat(args, config: RAGConfig):
    """Handle chat command"""
    config.top_k = args.top_k
    config.temperature = args.temperature
    config.backend = args.backend

    if args.backend == "openai" and args.model:
        config.openai_model = args.model
    elif args.backend == "ollama" and args.model:
        config.llm_model = args.model

    copilot = RAGCopilot(config)
    copilot.chat_interactive()


def cmd_server(args, config: RAGConfig):
    """Handle server command"""
    try:
        import uvicorn
    except ImportError:
        print("Uvicorn is required for server mode. Install with:")
        print("  pip install uvicorn")
        sys.exit(1)

    config.host = args.host
    config.port = args.port
    config.backend = args.backend
    config.top_k = args.top_k

    if args.backend == "openai" and args.model:
        config.openai_model = args.model
    elif args.backend == "ollama" and args.model:
        config.llm_model = args.model

    app = create_server(config)

    logger.info(f"Starting RAG Copilot server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


def cmd_retrieve(args, config: RAGConfig):
    """Handle retrieve command (search only, no generation)"""
    config.top_k = args.top_k

    # We only need the retrieval components, not the LLM
    index_path = safe_path(config.index_path, PROJECT_ROOT)
    metadata_path = safe_path(config.metadata_path, PROJECT_ROOT)

    index_manager = FAISSIndexManager(index_path, metadata_path)
    embedding_manager = EmbeddingManager(config.embedding_model)

    # Encode and search
    query_embedding = embedding_manager.encode(args.query)
    results = index_manager.search(query_embedding, args.top_k)

    print(f"\nTop {len(results)} results for: {args.query}\n")
    print("-" * 60)

    for i, result in enumerate(results, 1):
        print(f"\n{i}. [Score: {result.get('similarity_score', 0):.3f}]")
        print(f"   Prompt: {result.get('prompt', '')[:80]}...")
        if args.verbose:
            print(f"   Response: {result.get('response_preview', '')[:100]}...")


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="L4D2 RAG-Enhanced Copilot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive chat with RAG using Ollama
  python rag_copilot.py chat --backend ollama

  # Single completion with OpenAI
  python rag_copilot.py complete "How do I detect tank spawns?" --backend openai

  # Start RAG server
  python rag_copilot.py server --port 8080

  # Search for similar examples only
  python rag_copilot.py retrieve "heal all survivors" --top-k 5
        """
    )

    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        default="INFO", help="Log level")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Complete command
    complete_parser = subparsers.add_parser("complete", help="Generate a single completion")
    complete_parser.add_argument("query", help="The coding question or task")
    complete_parser.add_argument("--backend", choices=["ollama", "openai"], default="ollama",
                                  help="LLM backend to use")
    complete_parser.add_argument("--model", help="Model name (backend-specific)")
    complete_parser.add_argument("--top-k", type=int, default=3,
                                  help="Number of examples to retrieve")
    complete_parser.add_argument("--temperature", type=float, default=0.7,
                                  help="Generation temperature")
    complete_parser.add_argument("--show-examples", action="store_true",
                                  help="Show retrieved examples")
    complete_parser.add_argument("--verbose", "-v", action="store_true",
                                  help="Show timing information")

    # Chat command
    chat_parser = subparsers.add_parser("chat", help="Interactive chat with RAG")
    chat_parser.add_argument("--backend", choices=["ollama", "openai"], default="ollama",
                              help="LLM backend to use")
    chat_parser.add_argument("--model", help="Model name (backend-specific)")
    chat_parser.add_argument("--top-k", type=int, default=3,
                              help="Number of examples to retrieve")
    chat_parser.add_argument("--temperature", type=float, default=0.7,
                              help="Generation temperature")

    # Server command
    server_parser = subparsers.add_parser("server", help="Start RAG API server")
    server_parser.add_argument("--host", default="0.0.0.0", help="Server host")
    server_parser.add_argument("--port", type=int, default=8080, help="Server port")
    server_parser.add_argument("--backend", choices=["ollama", "openai"], default="ollama",
                                help="LLM backend to use")
    server_parser.add_argument("--model", help="Model name (backend-specific)")
    server_parser.add_argument("--top-k", type=int, default=3,
                                help="Default number of examples to retrieve")

    # Retrieve command (search only)
    retrieve_parser = subparsers.add_parser("retrieve", help="Search for similar examples")
    retrieve_parser.add_argument("query", help="The query to search for")
    retrieve_parser.add_argument("--top-k", type=int, default=5,
                                  help="Number of examples to retrieve")
    retrieve_parser.add_argument("--verbose", "-v", action="store_true",
                                  help="Show response previews")

    args = parser.parse_args()

    # Set log level
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    # Create default config
    config = RAGConfig()

    # Execute command
    if args.command == "complete":
        cmd_complete(args, config)
    elif args.command == "chat":
        cmd_chat(args, config)
    elif args.command == "server":
        cmd_server(args, config)
    elif args.command == "retrieve":
        cmd_retrieve(args, config)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
