"""
Module 7 Project — Semantic Search Engine
==========================================
ingest.py — document loading, chunking, and ChromaDB storage

Run with:
    python ingest.py
    python ingest.py --chunk-size 200 --overlap 50
"""

import argparse
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

# ── Configuration ─────────────────────────────────────────────────────────────

DOCS_DIR = Path("docs")
CHROMA_PATH = Path("chroma_data")
COLLECTION_NAME = "semantic_search"
MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_CHUNK_SIZE = 500
DEFAULT_OVERLAP = 100


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Split text into fixed-size chunks with overlap.

    Args:
        text: Full document text.
        chunk_size: Maximum characters per chunk.
        overlap: Characters of overlap between consecutive chunks.

    Returns:
        List of non-empty chunk strings.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0.")

    if overlap < 0:
        raise ValueError("overlap cannot be negative.")

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size.")

    text = text.strip()

    if not text:
        return []

    chunks = []
    start = 0
    step = chunk_size - overlap

    while start < len(text):
        chunk = text[start:start + chunk_size].strip()

        if chunk:
            chunks.append(chunk)

        start += step

    return chunks


def load_documents(docs_dir: Path) -> list[dict]:
    """
    Read all .txt and .md files from docs_dir.

    Returns:
        List of dicts: {"filename": str, "text": str}
    """
    if not docs_dir.exists():
        raise FileNotFoundError(
            f"Documents directory not found: {docs_dir.resolve()}"
        )

    documents = []

    for path in sorted(docs_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in {".txt", ".md"}:
            text = path.read_text(encoding="utf-8").strip()

            if text:
                documents.append(
                    {
                        "filename": path.name,
                        "text": text,
                    }
                )

    return documents


def get_collection(chroma_path: Path, collection_name: str):
    """Create (or retrieve) a persistent ChromaDB collection."""
    chroma_path.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(chroma_path))

    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={
            "description": "Semantic search document collection"
        },
    )

    return collection


def ingest(
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
):
    """
    Full ingestion pipeline: load → chunk → embed → upsert.

    Each chunk is stored with metadata: source filename, chunk index,
    and the chunk size used — so experiments with different sizes can
    be compared without ambiguity.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0.")

    if overlap < 0:
        raise ValueError("overlap cannot be negative.")

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size.")

    print("Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    print(f"Loading documents from: {DOCS_DIR.resolve()}")

    documents = load_documents(DOCS_DIR)

    if not documents:
        raise ValueError(
            f"No .txt or .md documents found in {DOCS_DIR.resolve()}."
        )

    collection = get_collection(
        CHROMA_PATH,
        COLLECTION_NAME,
    )

    # Remove existing records so that re-indexing produces a clean
    # collection when experimenting with different chunk sizes.
    existing = collection.get()

    if existing["ids"]:
        collection.delete(ids=existing["ids"])

    ids = []
    texts = []
    metadatas = []

    for document in documents:
        filename = document["filename"]
        text = document["text"]

        chunks = chunk_text(
            text=text,
            chunk_size=chunk_size,
            overlap=overlap,
        )

        print(f"  {filename}: {len(chunks)} chunks")

        for chunk_index, chunk in enumerate(chunks):
            chunk_id = f"{filename}__chunk_{chunk_index}"

            ids.append(chunk_id)
            texts.append(chunk)

            metadatas.append(
                {
                    "source": filename,
                    "chunk_index": chunk_index,
                    "chunk_size": chunk_size,
                }
            )

    if not texts:
        raise ValueError(
            "No text chunks were created from the documents."
        )

    print(f"Creating embeddings for {len(texts)} chunks...")

    embeddings = model.encode(
        texts,
        show_progress_bar=True,
    ).tolist()

    collection.upsert(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    print()
    print("Ingestion complete.")
    print(f"Documents: {len(documents)}")
    print(f"Chunks: {len(texts)}")
    print(f"Chunk size: {chunk_size}")
    print(f"Overlap: {overlap}")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Chroma path: {CHROMA_PATH.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Index docs/ into ChromaDB"
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
    )

    parser.add_argument(
        "--overlap",
        type=int,
        default=DEFAULT_OVERLAP,
    )

    args = parser.parse_args()

    ingest(
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )