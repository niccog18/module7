from sentence_transformers import SentenceTransformer, util


# ============================================================
# CHUNK AND COMPARE
# ============================================================

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


# ============================================================
# SOURCE DOCUMENT
# ============================================================

document = """
# Course Concepts: FastAPI, Streamlit, Authentication, and AI

FastAPI is a modern Python web framework used for building APIs. It is designed
to make API development fast and straightforward while providing features such
as automatic documentation, type validation, and asynchronous support. FastAPI
uses Python type hints extensively, which allows developers to define the data
that an endpoint expects and returns. When a FastAPI application is running,
developers can use Swagger UI to interactively test endpoints. ReDoc provides
another automatically generated documentation interface.

A REST API is organized around resources and HTTP methods. Common HTTP methods
include GET for retrieving information, POST for creating new resources, PUT
or PATCH for updating resources, and DELETE for removing resources. In a task
management API, for example, a GET request might retrieve a list of tasks,
while a POST request creates a new task. A DELETE request can remove a task
from the database. Good API design uses meaningful URL paths and appropriate
HTTP status codes to communicate whether an operation succeeded or failed.

Pydantic is an important part of FastAPI because it provides data validation
and serialization. Developers can create Pydantic models, called schemas, to
describe the structure of incoming and outgoing data. If a client sends data
that does not match the expected schema, FastAPI can automatically return a
validation error. This reduces the amount of manual validation code that a
developer needs to write and makes APIs more predictable.

SQLAlchemy can be used to connect a FastAPI application to a relational
database. SQLAlchemy provides an object-relational mapping system that allows
Python classes to represent database tables. A User model might contain fields
such as an ID, name, email, and password hash. A Task model might contain an
ID, title, description, priority, completion status, and a user ID that links
the task to its owner. Foreign keys are useful because they establish
relationships between database tables.

Authentication is another important part of an API. A common approach is to
use JSON Web Tokens, also called JWTs. When a user successfully logs in, the
server can generate a token containing information about the authenticated
user. The client then sends that token with future requests. The server
verifies the token before allowing access to protected endpoints. Passwords
should never be stored as plain text. Instead, applications should hash
passwords using an appropriate password hashing algorithm and verify the
provided password against the stored hash during login.

User-scoped data is especially important in a task management application.
After a user is authenticated, the API should use the identity in the token
to determine which tasks belong to that user. A user should be able to create,
view, update, and delete their own tasks without being able to access another
user's private tasks. This is an example of authorization. Authentication
answers the question of who a user is, while authorization determines what
that user is allowed to do.

Testing is an important part of developing reliable APIs. FastAPI provides
TestClient, which can be used with pytest to send requests to application
endpoints without manually starting a server. Tests can verify registration,
login, task creation, retrieving tasks, updating tasks, and deleting tasks.
A test database can be used so that tests do not interfere with production
data. Dependency overrides can also allow an application to use a temporary
database during testing.

Streamlit provides a different approach to building applications in Python.
Instead of creating a traditional HTML and JavaScript frontend, developers
can use Streamlit commands to create interactive web interfaces. Widgets such
as buttons, text inputs, sliders, select boxes, checkboxes, and file uploaders
allow users to interact with an application. Streamlit automatically reruns
the Python script when a user interacts with a widget, which makes it possible
to build dashboards and prototypes quickly.

Streamlit session state is useful when an application needs to remember
information between reruns. For example, a login token can be stored in
st.session_state after authentication. A chat application can also store
conversation history in session state. Without session state, values created
during one script run would not necessarily be available during the next
rerun. A Clear Chat button can remove the stored conversation history and
reset the interface.

Caching can improve the performance of Streamlit applications. The
st.cache_data decorator is useful when an application repeatedly retrieves
data that does not change frequently. For example, an application that
downloads a list of users from an API can cache the result instead of making
the same request every time the page reruns. This can reduce unnecessary API
calls and make the application feel faster.

AI applications can use embeddings to compare the meaning of text. An
embedding model converts text into numerical vectors. Texts with similar
meanings tend to have vectors that are closer together in embedding space.
Sentence Transformers provides models that can create these embeddings.
all-MiniLM-L6-v2 is a relatively small model that is useful for semantic
search demonstrations. Cosine similarity can then be used to compare a query
embedding with document chunk embeddings.

Chunking is an important step in a retrieval system. A long document is
usually divided into smaller pieces before the pieces are embedded. One
strategy is fixed-size chunking, where text is divided according to a
specified number of characters. Overlap can be added so that information near
the boundary of one chunk is also present in the next chunk. This can help
prevent important context from being lost at chunk boundaries.

Another strategy is paragraph-based chunking. Instead of cutting text at a
fixed character count, paragraph-based chunking keeps each paragraph together.
This can preserve the natural meaning and context of a section. However,
paragraphs can vary greatly in size, and a very long paragraph may produce a
chunk that contains several unrelated ideas.

The best chunking strategy depends on the document and the questions users
will ask. Fixed-size chunks provide predictable chunk sizes and can work well
when documents contain long sections without clear boundaries. Paragraph
chunking can work particularly well when the document has clearly organized
paragraphs, because each chunk represents a more complete idea.

A retrieval system can compare a user's query against all embedded chunks and
return the chunks with the highest similarity scores. This process is often
called semantic search. The quality of retrieval depends on several factors,
including the embedding model, the chunk size, the amount of overlap, and the
structure of the source document. Comparing different chunking strategies
using realistic queries is therefore a useful way to decide which approach
works best for a particular document collection.
"""


# ============================================================
# STRATEGY 1: FIXED-SIZE CHUNKING
# ============================================================

def fixed_size_chunks(text, chunk_size=300, overlap=50):
    """
    Split text into fixed-size chunks with overlap.
    """
    chunks = []

    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


# ============================================================
# STRATEGY 2: PARAGRAPH-BASED CHUNKING
# ============================================================

def paragraph_chunks(text):
    """
    Split the document into chunks based on double newlines.
    """
    chunks = [
        paragraph.strip()
        for paragraph in text.split("\n\n")
        if paragraph.strip()
    ]

    return chunks


# Create chunks
fixed_chunks = fixed_size_chunks(document)
paragraph_chunks_list = paragraph_chunks(document)


# ============================================================
# EMBEDDINGS
# ============================================================

fixed_embeddings = model.encode(
    fixed_chunks,
    convert_to_tensor=True
)

paragraph_embeddings = model.encode(
    paragraph_chunks_list,
    convert_to_tensor=True
)


# ============================================================
# SEARCH FUNCTION
# ============================================================

def search_chunks(query, chunks, embeddings, top_k=2):
    """
    Embed the query and return the top matching chunks.
    """
    query_embedding = model.encode(
        query,
        convert_to_tensor=True
    )

    scores = util.cos_sim(query_embedding, embeddings)[0]

    top_results = scores.topk(k=min(top_k, len(chunks)))

    results = []

    for score, index in zip(top_results.values, top_results.indices):
        results.append(
            (
                float(score),
                chunks[int(index)]
            )
        )

    return results


# ============================================================
# QUERIES
# ============================================================

queries = [
    "How does JWT authentication protect API endpoints?",
    "How does Streamlit remember information between reruns?",
    "Why is chunking important for semantic search?"
]


# ============================================================
# RUN COMPARISON
# ============================================================

print("=" * 80)
print("CHUNK AND COMPARE")
print("=" * 80)

print(f"\nFixed-size chunks: {len(fixed_chunks)}")
print(f"Paragraph chunks:  {len(paragraph_chunks_list)}")


for query_number, query in enumerate(queries, start=1):

    print("\n" + "=" * 80)
    print(f"QUERY {query_number}: {query}")
    print("=" * 80)

    # --------------------------------------------------------
    # Fixed-size results
    # --------------------------------------------------------

    print("\n--- FIXED-SIZE CHUNKING ---")

    fixed_results = search_chunks(
        query,
        fixed_chunks,
        fixed_embeddings,
        top_k=2
    )

    for rank, (score, chunk) in enumerate(fixed_results, start=1):
        print(f"\nResult {rank} | Score: {score:.4f}")
        print("-" * 60)
        print(chunk)

    # --------------------------------------------------------
    # Paragraph results
    # --------------------------------------------------------

    print("\n--- PARAGRAPH-BASED CHUNKING ---")

    paragraph_results = search_chunks(
        query,
        paragraph_chunks_list,
        paragraph_embeddings,
        top_k=2
    )

    for rank, (score, chunk) in enumerate(paragraph_results, start=1):
        print(f"\nResult {rank} | Score: {score:.4f}")
        print("-" * 60)
        print(chunk)


# ============================================================
# WRITTEN COMPARISON
# ============================================================

print("\n" + "=" * 80)
print("COMPARISON")
print("=" * 80)

print("""
For this document, paragraph-based chunking generally performs better for
semantic search because the source material is already organized into
paragraphs, and each paragraph focuses on a relatively specific course
concept. Keeping paragraphs together preserves the context of ideas such as
JWT authentication, Streamlit session state, and semantic search.

Fixed-size chunking has the advantage of producing predictable chunk sizes,
and the 50-character overlap helps preserve information that falls near a
chunk boundary. However, fixed-size chunks can split a sentence or concept
in the middle, which may reduce the amount of useful context available to the
embedding model.

Therefore, paragraph-based chunking is the better strategy for this particular
document because its paragraphs have clear semantic boundaries. Fixed-size
chunking could be preferable for documents with very long paragraphs or
poorly structured text where natural paragraph boundaries are not reliable.
The best strategy ultimately depends on the structure of the documents and
the types of queries the retrieval system needs to answer.
""")
