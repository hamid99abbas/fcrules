"""
Generate embeddings locally and save to embeddings.npy
Run this ONCE locally before deploying to Vercel

This script:
1. Loads your chunks from chunks.jsonl
2. Generates embeddings using SentenceTransformer (locally)
3. Saves embeddings to embeddings.npy
4. You upload both chunks.jsonl and embeddings.npy to Vercel
"""

import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Paths
CHUNKS_PATH = Path("rag_chunks/chunks.jsonl")  # Adjust path as needed
EMBEDDINGS_OUTPUT = Path("rag_chunks/embeddings.npy")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_chunks(path: Path):
    """Load chunks from JSONL file"""
    if not path.exists():
        raise FileNotFoundError(f"chunks.jsonl not found at: {path.resolve()}")

    chunks = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))

    return chunks


def generate_embeddings(chunks, model_name=MODEL_NAME):
    """Generate embeddings for all chunks"""
    print(f"\n📦 Loading model: {model_name}")
    model = SentenceTransformer(model_name)

    print(f"\n🔢 Generating embeddings for {len(chunks)} chunks...")
    texts = [chunk.get("text", "") for chunk in chunks]

    # Generate embeddings with progress bar
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=32
    )

    return embeddings.astype('float32')


def main():
    print("=" * 60)
    print("🚀 EMBEDDING GENERATION SCRIPT")
    print("=" * 60)

    # Load chunks
    print(f"\n📂 Loading chunks from: {CHUNKS_PATH.resolve()}")
    chunks = load_chunks(CHUNKS_PATH)
    print(f"✓ Loaded {len(chunks)} chunks")

    # Generate embeddings
    embeddings = generate_embeddings(chunks)
    print(f"✓ Generated embeddings with shape: {embeddings.shape}")

    # Save embeddings
    print(f"\n💾 Saving embeddings to: {EMBEDDINGS_OUTPUT.resolve()}")
    EMBEDDINGS_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    np.save(EMBEDDINGS_OUTPUT, embeddings)

    # Verify file size
    file_size_mb = EMBEDDINGS_OUTPUT.stat().st_size / (1024 * 1024)
    print(f"✓ Saved embeddings ({file_size_mb:.2f} MB)")

    # Verification
    print("\n🔍 Verifying embeddings...")
    loaded = np.load(EMBEDDINGS_OUTPUT)
    assert loaded.shape == embeddings.shape, "Shape mismatch!"
    assert np.allclose(loaded, embeddings), "Data mismatch!"
    print("✓ Verification passed")

    print("\n" + "=" * 60)
    print("✅ SUCCESS!")
    print("=" * 60)
    print("\n📋 Next steps:")
    print("1. Upload these files to Vercel:")
    print(f"   - {CHUNKS_PATH}")
    print(f"   - {EMBEDDINGS_OUTPUT}")
    print("\n2. Set environment variables in Vercel:")
    print("   - GEMINI_API_KEY=your_gemini_key")
    print("   - HF_TOKEN=your_huggingface_token")
    print("\n3. Deploy!")
    print("\n💡 Get your free HF token from:")
    print("   https://huggingface.co/settings/tokens")
    print("=" * 60)


if __name__ == "__main__":
    main()