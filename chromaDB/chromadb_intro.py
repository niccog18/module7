import chromadb

# Create a persistent client (data survives script restarts)
client = chromadb.PersistentClient(path="./chroma_demo")

# Create (or get) a collection
# If it already exists, get_or_create_collection returns the existing one
collection = client.get_or_create_collection(
    name="course_notes",
    metadata={"description": "AI Engineering Foundations course content"}
)

print(f"Collection '{collection.name}' ready.")
print(f"Current document count: {collection.count()}")


# Course content chunks with metadata
documents = [
    "FastAPI is a modern Python web framework that automatically validates request data using Pydantic models and generates interactive API documentation.",
    "SQLAlchemy is an ORM that maps Python classes to database tables. It supports relationships like one-to-many and many-to-many.",
    "Streamlit re-runs the entire Python script every time a user interacts with a widget. Use st.session_state to persist data across re-runs.",
    "JWT tokens provide stateless authentication for REST APIs. The token contains encoded user information and is verified on each request.",
    "CSS Flexbox arranges child elements in a row or column. Use display:flex on the container and gap for spacing between items.",
    "The DOM is the browser's tree representation of an HTML page. JavaScript uses querySelector and addEventListener to interact with it.",
    "Embeddings convert text into numerical vectors that capture semantic meaning. Similar texts produce vectors that are close together.",
    "Cosine similarity measures the angle between two vectors. A score of 1.0 means identical direction (same meaning), 0.0 means unrelated.",
]

# Metadata for each document (source module, topic category)
metadatas = [
    {"module": "5", "topic": "api"},
    {"module": "3", "topic": "database"},
    {"module": "6", "topic": "frontend"},
    {"module": "5", "topic": "security"},
    {"module": "6", "topic": "frontend"},
    {"module": "6", "topic": "frontend"},
    {"module": "7", "topic": "ai"},
    {"module": "7", "topic": "ai"},
]

# Unique IDs for each document
ids = [f"doc_{i}" for i in range(len(documents))]

# Add to ChromaDB — it embeds them automatically!
collection.add(
    documents=documents,
    metadatas=metadatas,
    ids=ids
)

print(f"Added {len(documents)} documents.")
print(f"Collection now has {collection.count()} documents.")


# Search for documents similar to a query
results = collection.query(
    query_texts=["How do I make my API verify user identity?"],  # Your search query
    n_results=3  # Return top 3 matches
)

print("\n=== Search Results ===")
print(f"Query: 'How do I make my API verify user identity?'\n")

for i in range(len(results['documents'][0])):
    doc = results['documents'][0][i]
    distance = results['distances'][0][i]  # Lower distance = more similar
    metadata = results['metadatas'][0][i]
    doc_id = results['ids'][0][i]

    print(f"{i+1}. [distance: {distance:.4f}] (Module {metadata['module']}, {metadata['topic']})")
    print(f"   {doc[:100]}...")
    print()


# Search only within Module 6 content
results_filtered = collection.query(
    query_texts=["How do I make my web page interactive?"],
    n_results=3,
    where={"module": "6"}  # Only search Module 6 documents
)

print("\n=== Filtered Results (Module 6 only) ===")
for i in range(len(results_filtered['documents'][0])):
    doc = results_filtered['documents'][0][i]
    metadata = results_filtered['metadatas'][0][i]
    print(f"{i+1}. [{metadata['topic']}] {doc[:100]}...")