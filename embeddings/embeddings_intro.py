from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')

documents = [
    "FastAPI automatically generates interactive API documentation using Swagger UI.",
    "Use pip install to add Python packages to your virtual environment.",
    "SQLAlchemy is an ORM that maps Python classes to database tables.",
    "Streamlit re-runs the entire script every time a user interacts with a widget.",
    "JWT tokens are used for stateless authentication in REST APIs.",
    "The requests library lets you make HTTP calls from Python.",
    "CSS Flexbox arranges elements in a row or column with flexible sizing.",
    "ChromaDB stores document embeddings for fast similarity search.",
]

doc_embeddings = model.encode(documents)

query = "How do I authenticate users in my API?"
query_embedding = model.encode(query)

scores = util.cos_sim(query_embedding, doc_embeddings)[0]
ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

print(f"Query: '{query}'\n")
print("Top 3 results:")
for rank, (idx, score) in enumerate(ranked[:3], 1):
    print(f"  {rank}. [{score:.4f}] {documents[idx]}")