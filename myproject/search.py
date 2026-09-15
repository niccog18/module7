"""
Module 7 Project — Semantic Search Engine
==========================================
search.py — query ChromaDB and return ranked results

Import this module into app.py:
    from search import search, get_collection_stats
"""

from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


# ── Configuration (must match ingest.py) ─────────────────────────────────────
CHROMA_PATH = Path("chroma_data")
COLLECTION_NAME = "semantic_search"
MODEL_NAME = "all-MiniLM-L6-v2"


def get_collection():
    """Return the persistent ChromaDB collection."""
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))

    return client.get_or_create_collection(
        name=COLLECTION_NAME
    )


def search(
    query: str,
    n_results: int = 5,
    sources: list[str] = None,
    distance_threshold: float = None,
) -> list[dict]:
    """
    Search the ChromaDB collection and return ranked results.

    Args:
        query: Natural language search query.
        n_results: Maximum number of results to return.
        sources: If provided, only return chunks from these filenames.
        distance_threshold: If provided, exclude results with distance above this
                            value (lower = more similar).

    Returns:
        List of result dicts sorted by distance ascending (best first):
            {
                "text": str,
                "source": str,
                "chunk_index": int,
                "distance": float,
                "score": float,
            }

        Returns [] for empty queries or if the collection has no documents.
    """
    if not query or not query.strip():
        return []

    if n_results <= 0:
        return []

    if distance_threshold is not None and distance_threshold < 0:
        raise ValueError("distance_threshold cannot be negative.")

    collection = get_collection()

    if collection.count() == 0:
        return []

    # Load the embedding model used during ingestion.
    model = SentenceTransformer(MODEL_NAME)

    # Create an embedding for the user's query.
    query_embedding = model.encode(
        [query.strip()]
    ).tolist()

    # ChromaDB's where filter allows us to restrict results by source.
    where_filter = None

    if sources:
        where_filter = {
            "source": {
                "$in": sources
            }
        }

    # Request enough results to allow the threshold to be applied afterward.
    # ChromaDB can return fewer results when a source filter is active.
    query_kwargs = {
        "query_embeddings": query_embedding,
        "n_results": min(n_results, collection.count()),
        "include": [
            "documents",
            "metadatas",
            "distances",
        ],
    }

    if where_filter:
        query_kwargs["where"] = where_filter

    results = collection.query(**query_kwargs)

    if not results["documents"] or not results["documents"][0]:
        return []

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    ranked_results = []

    for text, metadata, distance in zip(
        documents,
        metadatas,
        distances,
    ):
        distance = float(distance)

        if (
            distance_threshold is not None
            and distance > distance_threshold
        ):
            continue

        ranked_results.append(
            {
                "text": text,
                "source": metadata.get("source", "Unknown"),
                "chunk_index": int(
                    metadata.get("chunk_index", 0)
                ),
                "distance": distance,
                "score": 1.0 - distance,
            }
        )

    ranked_results.sort(
        key=lambda result: result["distance"]
    )

    return ranked_results[:n_results]


def get_collection_stats() -> dict:
    """
    Return basic stats about the indexed collection.

    Returns:
        {
            "total_chunks": int,
            "unique_sources": int,
            "source_names": list[str],
        }
    """
    collection = get_collection()

    total_chunks = collection.count()

    if total_chunks == 0:
        return {
            "total_chunks": 0,
            "unique_sources": 0,
            "source_names": [],
        }

    data = collection.get(
        include=["metadatas"]
    )

    source_names = set()

    for metadata in data["metadatas"]:
        if metadata and metadata.get("source"):
            source_names.add(metadata["source"])

    sorted_sources = sorted(source_names)

    return {
        "total_chunks": total_chunks,
        "unique_sources": len(sorted_sources),
        "source_names": sorted_sources,
    }