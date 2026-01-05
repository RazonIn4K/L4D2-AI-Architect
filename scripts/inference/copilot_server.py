#!/usr/bin/env python3
"""
L4D2 Copilot Inference Server

Provides API endpoints for code completion and generation
using fine-tuned models for SourcePawn and VScript.
"""

import os
import time
import logging
import torch
from typing import Dict, List, Optional
from pathlib import Path
import argparse
from dataclasses import dataclass

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    import uvicorn
except ImportError:
    print("FastAPI not installed. Run: pip install fastapi uvicorn")
    exit(1)

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
except ImportError:
    print("Transformers/PEFT not installed. Run: pip install transformers peft")
    exit(1)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for loaded model"""
    model_path: str
    base_model: str
    max_length: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    do_sample: bool = True


class CompletionRequest(BaseModel):
    """Request model for code completion"""
    prompt: str
    max_tokens: int = 256
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    stop_sequences: List[str] = []
    language: str = "sourcepawn"  # sourcepawn, vscript, or auto


class CompletionResponse(BaseModel):
    """Response model for code completion"""
    completion: str
    tokens_generated: int
    model: str
    timestamp: float


class CopilotServer:
    """Main copilot inference server"""
    
    def __init__(self, model_path: str, base_model: str = "unsloth/mistral-7b-instruct-v0.3-bnb-4bit"):
        self.model_path = Path(model_path)
        self.base_model = base_model
        self.config = ModelConfig(model_path=model_path, base_model=base_model)
        
        # Model components
        self.tokenizer = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load model
        self._load_model()
        
        # Create FastAPI app
        self.app = FastAPI(
            title="L4D2 Copilot API",
            description="AI-powered code completion for SourcePawn and VScript",
            version="1.0.0"
        )
        
        # Add CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Setup routes
        self._setup_routes()
    
    def _load_model(self):
        """Load the fine-tuned model"""
        logger.info(f"Loading model from {self.model_path}")
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.base_model,
                trust_remote_code=True
            )
            
            # Add special tokens if needed
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load base model
            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True,
                load_in_4bit=True if self.device == "cuda" else False
            )
            
            # Load LoRA adapter if exists
            if self.model_path.exists() and (self.model_path / "adapter_config.json").exists():
                logger.info("Loading LoRA adapter")
                self.model = PeftModel.from_pretrained(base_model, str(self.model_path))
            else:
                logger.warning("No LoRA adapter found, using base model")
                self.model = base_model
            
            # Move to device if not already
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            
            logger.info(f"Model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _setup_routes(self):
        """Setup API routes"""
        
        @self.app.get("/")
        async def root():
            return {"message": "L4D2 Copilot API is running", "model": str(self.model_path)}
        
        @self.app.get("/health")
        async def health():
            return {
                "status": "healthy",
                "device": self.device,
                "model_loaded": self.model is not None
            }
        
        @self.app.post("/v1/complete", response_model=CompletionResponse)
        async def complete_code(request: CompletionRequest):
            """Generate code completion"""
            try:
                # Format prompt based on language
                formatted_prompt = self._format_prompt(request.prompt, request.language)
                
                # Generate completion
                completion, tokens = self._generate(
                    formatted_prompt,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature or self.config.temperature,
                    top_p=request.top_p or self.config.top_p,
                    stop_sequences=request.stop_sequences
                )
                
                # Clean up completion
                completion = self._clean_completion(completion, formatted_prompt)
                
                return CompletionResponse(
                    completion=completion,
                    tokens_generated=tokens,
                    model=str(self.model_path),
                    timestamp=time.time()
                )
                
            except Exception as e:
                logger.error(f"Generation error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/v1/chat")
        async def chat_completion(request: dict):
            """Chat-style completion for more interactive help"""
            try:
                messages = request.get("messages", [])
                if not messages:
                    raise HTTPException(status_code=400, detail="No messages provided")
                
                # Format as chat
                prompt = self._format_chat(messages)
                
                # Generate response
                completion, tokens = self._generate(
                    prompt,
                    max_tokens=request.get("max_tokens", 512),
                    temperature=request.get("temperature", 0.7)
                )
                
                return {
                    "choices": [{
                        "message": {
                            "role": "assistant",
                            "content": completion
                        },
                        "finish_reason": "stop"
                    }],
                    "usage": {
                        "prompt_tokens": len(self.tokenizer.encode(prompt)),
                        "completion_tokens": tokens,
                        "total_tokens": len(self.tokenizer.encode(prompt)) + tokens
                    }
                }
                
            except Exception as e:
                logger.error(f"Chat error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
    
    def _format_prompt(self, prompt: str, language: str) -> str:
        """Format prompt based on programming language"""
        
        # Add language-specific context
        if language.lower() == "sourcepawn":
            context = """You are an expert SourcePawn plugin developer for Left 4 Dead 2.
Write clean, efficient, and well-commented code that follows best practices."""
        elif language.lower() == "vscript":
            context = """You are an expert VScript developer for Left 4 Dead 2.
Write functional scripts that integrate properly with the game engine."""
        else:
            context = """You are an expert programmer for Left 4 Dead 2 modding.
Write clean, efficient code for either SourcePawn or VScript."""
        
        # Create formatted prompt
        formatted = f"""<s>[INST] {context}

Complete the following code:

```
{prompt}
``` [/INST]"""
        
        return formatted
    
    def _format_chat(self, messages: List[Dict[str, str]]) -> str:
        """Format messages for chat completion"""
        formatted = "<s>"
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system":
                formatted += f"[INST] {content} [/INST]"
            elif role == "user":
                formatted += f"[INST] {content} [/INST]"
            elif role == "assistant":
                formatted += f" {content}</s>"
        
        return formatted
    
    def _generate(self, 
                  prompt: str, 
                  max_tokens: int = 256,
                  temperature: float = 0.7,
                  top_p: float = 0.9,
                  stop_sequences: List[str] = None) -> tuple[str, int]:
        """Generate text from prompt"""
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=self.config.top_k,
                repetition_penalty=self.config.repetition_penalty,
                do_sample=self.config.do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True
            )
        
        # Decode only the new tokens
        generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        completion = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        # Handle stop sequences
        if stop_sequences:
            for stop in stop_sequences:
                if stop in completion:
                    completion = completion.split(stop)[0]
        
        return completion, len(generated_tokens)
    
    def _clean_completion(self, completion: str, prompt: str) -> str:
        """Clean up the generated completion"""
        
        # Remove any remaining prompt text
        if prompt in completion:
            completion = completion.replace(prompt, "")
        
        # Remove common artifacts
        completion = completion.strip()
        
        # Remove code block markers if present
        if completion.startswith("```"):
            lines = completion.split("\n")
            if len(lines) > 1:
                completion = "\n".join(lines[1:])
        
        if completion.endswith("```"):
            completion = completion[:-3].strip()
        
        return completion
    
    def run(self, host: str = "0.0.0.0", port: int = 8000):
        """Run the server"""
        logger.info(f"Starting server on {host}:{port}")
        uvicorn.run(self.app, host=host, port=port)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="L4D2 Copilot Inference Server")
    parser.add_argument("--model-path", type=str, 
                       default="./model_adapters/l4d2-code-lora",
                       help="Path to fine-tuned model")
    parser.add_argument("--base-model", type=str,
                       default="unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
                       help="Base model name")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                       help="Server host")
    parser.add_argument("--port", type=int, default=8000,
                       help="Server port")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       default="INFO", help="Log level")
    
    args = parser.parse_args()
    
    # Set log level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Create and run server
    server = CopilotServer(
        model_path=args.model_path,
        base_model=args.base_model
    )
    
    server.run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
