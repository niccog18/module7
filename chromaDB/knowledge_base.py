import chromadb


# ============================================================
# 1. CREATE A PERSISTENT CHROMADB CLIENT
# ============================================================

client = chromadb.PersistentClient(path="./chroma_data")


# ============================================================
# 2. CREATE OR LOAD THE KNOWLEDGE COLLECTION
# ============================================================

collection = client.get_or_create_collection(
    name="my_knowledge"
)


# ============================================================
# 3. COURSE KNOWLEDGE
#    15+ documents from Modules 4, 5, and 6
# ============================================================

documents = [
    # -------------------------
    # MODULE 4 — REST APIs
    # -------------------------
    {
        "id": "m4_001",
        "text": "REST APIs allow different applications to communicate using HTTP requests and responses.",
        "module": "4",
        "topic": "api",
    },
    {
        "id": "m4_002",
        "text": "GET requests are commonly used to retrieve resources from an API.",
        "module": "4",
        "topic": "api",
    },
    {
        "id": "m4_003",
        "text": "POST requests are commonly used to create new resources through an API.",
        "module": "4",
        "topic": "api",
    },
    {
        "id": "m4_004",
        "text": "JSON is a common data format used to send structured information between clients and APIs.",
        "module": "4",
        "topic": "api",
    },
    {
        "id": "m4_005",
        "text": "API endpoints use URLs to identify specific resources and operations.",
        "module": "4",
        "topic": "api",
    },

    # -------------------------
    # MODULE 5 — FASTAPI
    # -------------------------
    {
        "id": "m5_001",
        "text": "FastAPI is a modern Python framework for building web APIs with automatic documentation.",
        "module": "5",
        "topic": "api",
    },
    {
        "id": "m5_002",
        "text": "Pydantic models validate incoming API data and define the expected structure of requests and responses.",
        "module": "5",
        "topic": "validation",
    },
    {
        "id": "m5_003",
        "text": "SQLAlchemy provides tools for connecting Python applications to relational databases and working with database models.",
        "module": "5",
        "topic": "database",
    },
    {
        "id": "m5_004",
        "text": "JWT authentication allows an API to securely identify users through signed access tokens.",
        "module": "5",
        "topic": "authentication",
    },
    {
        "id": "m5_005",
        "text": "CRUD operations represent creating, reading, updating, and deleting records in an application.",
        "module": "5",
        "topic": "database",
    },

    # -------------------------
    # MODULE 6 — STREAMLIT
    # -------------------------
    {
        "id": "m6_001",
        "text": "Streamlit makes it possible to build interactive Python web applications without writing traditional frontend JavaScript.",
        "module": "6",
        "topic": "frontend",
    },
    {
        "id": "m6_002",
        "text": "Streamlit widgets such as buttons, sliders, checkboxes, and text inputs allow users to interact with an application.",
        "module": "6",
        "topic": "frontend",
    },
    {
        "id": "m6_003",
        "text": "Streamlit session state allows application data to persist between script reruns during a user's session.",
        "module": "6",
        "topic": "state",
    },
    {
        "id": "m6_004",
        "text": "The Streamlit cache decorator can improve performance by avoiding unnecessary repeated data loading.",
        "module": "6",
        "topic": "performance",
    },
    {
        "id": "m6_005",
        "text": "An AI chat interface can use chat messages, user input, session history, and a system prompt to maintain conversation context.",
        "module": "6",
        "topic": "ai",
    },
    {
        "id": "m6_006",
        "text": "A Streamlit frontend can communicate with a FastAPI backend by sending HTTP requests to API endpoints.",
        "module": "6",
        "topic": "api",
    },
]


# ============================================================
# 4. ADD DOCUMENTS USING UPSERT
# ============================================================

collection.upsert(
    ids=[doc["id"] for doc in documents],
    documents=[doc["text"] for doc in documents],
    metadatas=[
        {
            "module": doc["module"],
            "topic": doc["topic"],
        }
        for doc in documents
    ],
)


print(f"Knowledge base contains {collection.count()} documents.")


# ============================================================
# 5. SEARCH FUNCTION
# ============================================================

def search_knowledge(query, module=None):
    """
    Search the knowledge base.

    Args:
        query (str): The question or search phrase.
        module (str, optional): Module number to filter by.

    Returns:
        ChromaDB search results.
    """

    search_parameters = {
        "query_texts": [query],
        "n_results": 5,
    }

    # Add module filtering only when requested
    if module is not None:
        search_parameters["where"] = {
            "module": module
        }

    results = collection.query(**search_parameters)

    return results


# ============================================================
# 6. DISPLAY SEARCH RESULTS
# ============================================================

def display_results(query, module=None):
    print("\n" + "=" * 70)
    print(f"QUERY: {query}")

    if module:
        print(f"MODULE FILTER: {module}")

    print("=" * 70)

    results = search_knowledge(query, module)

    ids = results["ids"][0]
    documents_found = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for i in range(len(ids)):
        print(f"\nResult {i + 1}")
        print(f"ID:       {ids[i]}")
        print(f"Distance: {distances[i]:.4f}")
        print(f"Module:   {metadatas[i]['module']}")
        print(f"Topic:    {metadatas[i]['topic']}")
        print(f"Content:  {documents_found[i]}")


# ============================================================
# 7. TEST QUERIES
# ============================================================

if __name__ == "__main__":

    # Test 1 — Broad search with no module filter
    display_results(
        "How do applications communicate with web services?"
    )

    # Test 2 — Search filtered specifically to Module 5
    display_results(
        "How does authentication and database access work?",
        module="5"
    )

    # Test 3 — Completely different wording
    # The stored documents talk about "Streamlit widgets",
    # "buttons", "sliders", etc., while the query uses
    # different terminology.
    display_results(
        "What tools let someone interact with a Python website?"
    )