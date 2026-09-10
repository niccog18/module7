from sentence_transformers import SentenceTransformer, util
import json

model = SentenceTransformer('all-MiniLM-L6-v2')

# --- Our "knowledge base" with IDs ---
documents = {
    "doc_jwt": "JWT tokens provide stateless authentication for REST APIs. The server creates a signed token containing user info.",
    "doc_pydantic": "Pydantic models define data schemas for FastAPI. They automatically validate incoming request data.",
    "doc_session": "Streamlit session state persists data across re-runs. Initialize with: if key not in st.session_state.",
    "doc_cors": "CORS middleware in FastAPI allows cross-origin requests from frontend applications running on different ports.",
    "doc_embed": "Embeddings convert text into numerical vectors capturing semantic meaning. Similar texts get similar vectors.",
    "doc_cosine": "Cosine similarity measures the angle between two vectors. A score of 1.0 means identical direction.",
    "doc_chunk": "Chunking splits documents into smaller pieces for embedding. Chunk size affects search precision and recall.",
    "doc_chroma": "ChromaDB is a vector database for storing and querying embeddings. It supports metadata filtering.",
    "doc_flex": "CSS Flexbox arranges elements in rows or columns. Use display:flex on the container.",
    "doc_dom": "The DOM is the browser's tree representation of HTML. JavaScript uses it to modify page content.",
}

# Embed all documents
doc_ids = list(documents.keys())
doc_texts = list(documents.values())
doc_embeddings = model.encode(doc_texts)

# --- Evaluation Set (Gold Standard) ---
eval_set = [
    {
        "query": "How do I authenticate API requests?",
        "relevant": ["doc_jwt", "doc_cors"]  # JWT and CORS are both auth-related
    },
    {
        "query": "How does Streamlit remember data between interactions?",
        "relevant": ["doc_session"]
    },
    {
        "query": "What is semantic similarity and how is it measured?",
        "relevant": ["doc_embed", "doc_cosine"]
    },
    {
        "query": "How should I prepare documents for a search system?",
        "relevant": ["doc_chunk", "doc_chroma"]
    },
    {
        "query": "How do I make a webpage interactive?",
        "relevant": ["doc_dom", "doc_flex"]
    },
]

def search(query, top_k=3, threshold=0.3):
    """Search documents and return IDs above threshold."""
    query_emb = model.encode(query)
    scores = util.cos_sim(query_emb, doc_embeddings)[0]

    results = []
    for i, score in enumerate(scores):
        if score.item() >= threshold:
            results.append((doc_ids[i], score.item()))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]

def evaluate(eval_set, top_k=3, threshold=0.3):
    """Run evaluation and compute precision/recall."""
    total_precision = 0
    total_recall = 0

    print(f"\nEvaluation (top_k={top_k}, threshold={threshold})")
    print("=" * 60)

    for item in eval_set:
        query = item["query"]
        relevant = set(item["relevant"])
        results = search(query, top_k, threshold)
        returned = set(doc_id for doc_id, _ in results)

        # Precision: of returned, how many are relevant?
        true_positives = returned & relevant
        precision = len(true_positives) / len(returned) if returned else 0

        # Recall: of relevant, how many were returned?
        recall = len(true_positives) / len(relevant) if relevant else 0

        total_precision += precision
        total_recall += recall

        status = "✅" if precision >= 0.5 and recall >= 0.5 else "⚠️"

        print(f"\n{status} Query: '{query}'")
        print(f"   Expected: {sorted(relevant)}")
        print(f"   Got:      {[(d, f'{s:.3f}') for d, s in results]}")
        print(f"   Precision: {precision:.1%} | Recall: {recall:.1%}")

    avg_precision = total_precision / len(eval_set)
    avg_recall = total_recall / len(eval_set)

    print(f"\n{'=' * 60}")
    print(f"AVERAGE Precision: {avg_precision:.1%}")
    print(f"AVERAGE Recall:    {avg_recall:.1%}")

    return avg_precision, avg_recall

# --- Run evaluation at different thresholds ---
print("\n" + "═" * 60)
print("EXPERIMENT: How does threshold affect precision vs recall?")
print("═" * 60)

for threshold in [0.2, 0.35, 0.5]:
    p, r = evaluate(eval_set, top_k=5, threshold=threshold)
    print(f"\n\u27a1 Threshold {threshold}: Precision={p:.1%}, Recall={r:.1%}")
