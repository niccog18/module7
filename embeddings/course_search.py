from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim


# Sentences summarizing things learned throughout the course
course_content = [
    "FastAPI can be used to build REST APIs with Python.",
    "Pydantic models validate and organize request and response data.",
    "SQLAlchemy allows Python applications to interact with databases.",
    "JWT tokens can be used to authenticate users and protect API endpoints.",
    "CRUD operations allow applications to create, read, update, and delete data.",
    "HTTP status codes communicate whether an API request was successful or failed.",
    "Pytest can be used to automatically test Python applications and API endpoints.",
    "Streamlit makes it possible to build interactive web applications with Python.",
    "Session state in Streamlit stores information while a user interacts with an app.",
    "The requests library can send HTTP requests from Python applications to APIs.",
    "CORS allows a frontend application to communicate with a backend on another origin.",
    "Git and GitHub help developers track changes and collaborate on projects.",
]

# Load the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Embed all course sentences once
sentence_embeddings = model.encode(
    course_content,
    convert_to_tensor=True
)

print("Course Content Semantic Search")
print("Type a question to search what you've learned.")
print("Type 'quit' to exit.\n")

while True:
    query = input("Search (or 'quit'): ")

    # Exit the search loop
    if query.lower() == "quit":
        print("Goodbye!")
        break

    # Embed the user's query
    query_embedding = model.encode(
        query,
        convert_to_tensor=True
    )

    # Calculate similarity scores
    similarity_scores = cos_sim(
        query_embedding,
        sentence_embeddings
    )[0]

    # Get the top 3 results
    top_results = similarity_scores.argsort(
        descending=True
    )[:3]

    print("\nTop 3 results:")

    for rank, index in enumerate(top_results, start=1):
        score = similarity_scores[index].item()
        sentence = course_content[index.item()]

        print(f"  {rank}. [{score:.4f}] {sentence}")

    print()