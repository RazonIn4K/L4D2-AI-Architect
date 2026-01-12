#!/usr/bin/env python3
"""
Embedding Generation Pipeline for L4D2 Copilot

This script generates embeddings for the L4D2 training dataset to enable
RAG-style (Retrieval Augmented Generation) semantic search. It creates
embeddings for prompts, responses, and combined pairs using sentence-transformers,
then builds a FAISS index for fast similarity search.

Usage:
    python scripts/training/generate_embeddings.py
    python scripts/training/generate_embeddings.py --demo "How do I spawn a tank?"
    python scripts/training/generate_embeddings.py --model all-mpnet-base-v2

Output Files:
    data/embeddings/prompts.npy       - User prompt embeddings
    data/embeddings/responses.npy     - Assistant response embeddings
    data/embeddings/combined.npy      - Combined prompt+response embeddings
    data/embeddings/metadata.json     - Index-to-example mapping
    data/embeddings/faiss_index.bin   - FAISS index for similarity search
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Add parent to path for security utils
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.security import safe_path, safe_write_json

# Project root for security validation
PROJECT_ROOT = Path(__file__).parent.parent.parent


def load_training_data(data_path: Path) -> List[Dict[str, Any]]:
    """
    Load training data from JSONL file.

    Args:
        data_path: Path to the JSONL training file

    Returns:
        List of training examples with messages
    """
    examples = []

    # Validate path is within project
    validated_path = safe_path(str(data_path), PROJECT_ROOT)

    with open(validated_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                example = json.loads(line)
                examples.append(example)
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping invalid JSON on line {line_num}: {e}")
                continue

    return examples


def extract_texts(examples: List[Dict[str, Any]]) -> Tuple[List[str], List[str], List[str]]:
    """
    Extract prompts, responses, and combined texts from training examples.

    Args:
        examples: List of training examples in ChatML format

    Returns:
        Tuple of (prompts, responses, combined_texts)
    """
    prompts = []
    responses = []
    combined = []

    for example in examples:
        messages = example.get("messages", [])

        # Extract user prompt (skip system message)
        user_prompt = ""
        assistant_response = ""

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")

            if role == "user":
                user_prompt = content
            elif role == "assistant":
                assistant_response = content

        prompts.append(user_prompt)
        responses.append(assistant_response)
        combined.append(f"Query: {user_prompt}\n\nCode:\n{assistant_response}")

    return prompts, responses, combined


def generate_embeddings(
    texts: List[str],
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 32,
    show_progress: bool = True
) -> np.ndarray:
    """
    Generate embeddings for a list of texts using sentence-transformers.

    Args:
        texts: List of text strings to embed
        model_name: Name of the sentence-transformers model
        batch_size: Batch size for encoding
        show_progress: Whether to show progress bar

    Returns:
        NumPy array of embeddings (n_texts, embedding_dim)
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("Error: sentence-transformers not installed.")
        print("Install with: pip install sentence-transformers")
        sys.exit(1)

    print(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name)

    print(f"Generating embeddings for {len(texts)} texts...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=True  # Normalize for cosine similarity
    )

    return embeddings


def create_faiss_index(embeddings: np.ndarray) -> Any:
    """
    Create a FAISS index for fast similarity search.

    Args:
        embeddings: NumPy array of embeddings

    Returns:
        FAISS index object
    """
    try:
        import faiss
    except ImportError:
        print("Error: faiss not installed.")
        print("Install with: pip install faiss-cpu (or faiss-gpu for CUDA)")
        sys.exit(1)

    # Get embedding dimension
    d = embeddings.shape[1]

    # Use IndexFlatIP for inner product (cosine similarity with normalized vectors)
    # For larger datasets, consider IndexIVFFlat or IndexHNSW
    index = faiss.IndexFlatIP(d)

    # Add embeddings to index
    # Ensure float32 for FAISS
    embeddings_f32 = embeddings.astype(np.float32)
    index.add(embeddings_f32)

    print(f"FAISS index created with {index.ntotal} vectors of dimension {d}")

    return index


def save_embeddings(
    output_dir: Path,
    prompts_emb: np.ndarray,
    responses_emb: np.ndarray,
    combined_emb: np.ndarray,
    metadata: Dict[str, Any],
    faiss_index: Any
) -> None:
    """
    Save embeddings, metadata, and FAISS index to disk.

    Args:
        output_dir: Directory to save outputs
        prompts_emb: Prompt embeddings
        responses_emb: Response embeddings
        combined_emb: Combined embeddings
        metadata: Metadata dictionary
        faiss_index: FAISS index object
    """
    try:
        import faiss
    except ImportError:
        print("Error: faiss not installed.")
        sys.exit(1)

    # Ensure output directory exists
    output_dir = safe_path(str(output_dir), PROJECT_ROOT, create_parents=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save embeddings as .npy files
    prompts_path = output_dir / "prompts.npy"
    responses_path = output_dir / "responses.npy"
    combined_path = output_dir / "combined.npy"

    np.save(prompts_path, prompts_emb)
    np.save(responses_path, responses_emb)
    np.save(combined_path, combined_emb)

    print(f"Saved prompt embeddings: {prompts_path}")
    print(f"Saved response embeddings: {responses_path}")
    print(f"Saved combined embeddings: {combined_path}")

    # Save metadata
    metadata_path = output_dir / "metadata.json"
    safe_write_json(str(metadata_path), metadata, PROJECT_ROOT, indent=2)
    print(f"Saved metadata: {metadata_path}")

    # Save FAISS index
    index_path = output_dir / "faiss_index.bin"
    faiss.write_index(faiss_index, str(index_path))
    print(f"Saved FAISS index: {index_path}")


def build_metadata(
    examples: List[Dict[str, Any]],
    prompts: List[str],
    responses: List[str],
    model_name: str,
    embedding_dim: int
) -> Dict[str, Any]:
    """
    Build metadata dictionary mapping indices to original examples.

    Args:
        examples: Original training examples
        prompts: Extracted prompts
        responses: Extracted responses
        model_name: Name of embedding model used
        embedding_dim: Dimension of embeddings

    Returns:
        Metadata dictionary
    """
    metadata = {
        "model_name": model_name,
        "embedding_dim": embedding_dim,
        "num_examples": len(examples),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "examples": []
    }

    for idx, (example, prompt, response) in enumerate(zip(examples, prompts, responses)):
        # Extract system prompt if present
        system_prompt = ""
        for msg in example.get("messages", []):
            if msg.get("role") == "system":
                system_prompt = msg.get("content", "")
                break

        metadata["examples"].append({
            "index": idx,
            "prompt": prompt,
            "response_preview": response[:200] + "..." if len(response) > 200 else response,
            "response_length": len(response),
            "system_prompt_preview": system_prompt[:100] + "..." if len(system_prompt) > 100 else system_prompt
        })

    return metadata


def load_index_and_metadata(embeddings_dir: Path) -> Tuple[Any, Dict[str, Any], np.ndarray]:
    """
    Load FAISS index, metadata, and combined embeddings from disk.

    Args:
        embeddings_dir: Directory containing embeddings

    Returns:
        Tuple of (faiss_index, metadata, combined_embeddings)
    """
    try:
        import faiss
    except ImportError:
        print("Error: faiss not installed.")
        sys.exit(1)

    # Validate paths
    embeddings_dir = safe_path(str(embeddings_dir), PROJECT_ROOT)

    # Load FAISS index
    index_path = embeddings_dir / "faiss_index.bin"
    if not index_path.exists():
        raise FileNotFoundError(f"FAISS index not found: {index_path}")
    faiss_index = faiss.read_index(str(index_path))

    # Load metadata
    metadata_path = embeddings_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    # Load combined embeddings
    combined_path = embeddings_dir / "combined.npy"
    if not combined_path.exists():
        raise FileNotFoundError(f"Combined embeddings not found: {combined_path}")
    combined_emb = np.load(combined_path)

    return faiss_index, metadata, combined_emb


def search_similar(
    query: str,
    faiss_index: Any,
    metadata: Dict[str, Any],
    model_name: str = "all-MiniLM-L6-v2",
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """
    Search for similar training examples given a query.

    Args:
        query: Query string
        faiss_index: FAISS index
        metadata: Metadata dictionary
        model_name: Embedding model name (must match index)
        top_k: Number of results to return

    Returns:
        List of similar examples with scores
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("Error: sentence-transformers not installed.")
        sys.exit(1)

    # Load model and encode query
    model = SentenceTransformer(model_name)
    query_emb = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype(np.float32)

    # Search
    scores, indices = faiss_index.search(query_emb, top_k)

    # Build results
    results = []
    for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
        if idx < 0 or idx >= len(metadata["examples"]):
            continue

        example_meta = metadata["examples"][idx]
        results.append({
            "rank": i + 1,
            "score": float(score),
            "index": int(idx),
            "prompt": example_meta["prompt"],
            "response_preview": example_meta["response_preview"],
            "response_length": example_meta["response_length"]
        })

    return results


def demo_search(
    query: str,
    embeddings_dir: Path,
    model_name: str = "all-MiniLM-L6-v2",
    top_k: int = 5
) -> None:
    """
    Demonstrate similarity search with a query.

    Args:
        query: Query string
        embeddings_dir: Directory containing embeddings
        model_name: Embedding model name
        top_k: Number of results to show
    """
    print(f"\nSearching for: \"{query}\"")
    print("=" * 60)

    # Load index and metadata
    faiss_index, metadata, _ = load_index_and_metadata(embeddings_dir)

    # Search
    start_time = time.time()
    results = search_similar(query, faiss_index, metadata, model_name, top_k)
    search_time = time.time() - start_time

    print(f"\nTop {len(results)} results (search time: {search_time:.3f}s):\n")

    for result in results:
        print(f"Rank {result['rank']} (score: {result['score']:.4f})")
        print(f"  Prompt: {result['prompt'][:80]}...")
        print(f"  Response preview: {result['response_preview'][:100]}...")
        print(f"  Response length: {result['response_length']} chars")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Generate embeddings for L4D2 training data"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/processed/l4d2_train_v12.jsonl",
        help="Path to training data JSONL file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/embeddings",
        help="Output directory for embeddings"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="Sentence-transformers model name"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for encoding"
    )
    parser.add_argument(
        "--demo",
        type=str,
        default=None,
        help="Run demo search with this query (skips generation)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of results for demo search"
    )

    args = parser.parse_args()

    # Handle paths
    data_path = PROJECT_ROOT / args.data
    output_dir = PROJECT_ROOT / args.output

    # Demo mode - just search
    if args.demo:
        if not output_dir.exists():
            print(f"Error: Embeddings not found at {output_dir}")
            print("Run without --demo first to generate embeddings.")
            sys.exit(1)

        demo_search(args.demo, output_dir, args.model, args.top_k)
        return

    # Check if data exists
    if not data_path.exists():
        print(f"Error: Training data not found at {data_path}")
        sys.exit(1)

    print("=" * 60)
    print("L4D2 Embedding Generation Pipeline")
    print("=" * 60)

    total_start = time.time()

    # Step 1: Load training data
    print("\n[1/5] Loading training data...")
    examples = load_training_data(data_path)
    print(f"  Loaded {len(examples)} training examples")

    # Step 2: Extract texts
    print("\n[2/5] Extracting prompts and responses...")
    prompts, responses, combined = extract_texts(examples)
    print(f"  Extracted {len(prompts)} prompts")
    print(f"  Extracted {len(responses)} responses")
    print(f"  Created {len(combined)} combined texts")

    # Step 3: Generate embeddings
    print(f"\n[3/5] Generating embeddings with {args.model}...")

    print("\n  Encoding prompts...")
    prompts_emb = generate_embeddings(prompts, args.model, args.batch_size)

    print("\n  Encoding responses...")
    responses_emb = generate_embeddings(responses, args.model, args.batch_size)

    print("\n  Encoding combined texts...")
    combined_emb = generate_embeddings(combined, args.model, args.batch_size)

    embedding_dim = prompts_emb.shape[1]
    print(f"\n  Embedding dimension: {embedding_dim}")

    # Step 4: Create FAISS index (on combined embeddings for full-context search)
    print("\n[4/5] Creating FAISS index...")
    faiss_index = create_faiss_index(combined_emb)

    # Step 5: Build metadata and save
    print("\n[5/5] Saving embeddings and index...")
    metadata = build_metadata(examples, prompts, responses, args.model, embedding_dim)
    save_embeddings(
        output_dir,
        prompts_emb,
        responses_emb,
        combined_emb,
        metadata,
        faiss_index
    )

    total_time = time.time() - total_start

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Model: {args.model}")
    print(f"  Training examples: {len(examples)}")
    print(f"  Embedding dimension: {embedding_dim}")
    print(f"  Total time: {total_time:.2f}s")
    print(f"\nOutput files:")
    print(f"  {output_dir / 'prompts.npy'}")
    print(f"  {output_dir / 'responses.npy'}")
    print(f"  {output_dir / 'combined.npy'}")
    print(f"  {output_dir / 'metadata.json'}")
    print(f"  {output_dir / 'faiss_index.bin'}")

    # Run a quick demo
    print("\n" + "=" * 60)
    print("DEMO: Searching for 'How do I spawn a tank?'")
    print("=" * 60)
    demo_search("How do I spawn a tank?", output_dir, args.model, args.top_k)


if __name__ == "__main__":
    main()
