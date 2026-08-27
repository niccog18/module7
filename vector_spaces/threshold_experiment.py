from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim


# Knowledge base covering 4 different topics
knowledge_base = [
    # Python / Development
    "Python virtual environments keep project dependencies isolated.",
    "FastAPI is a Python framework for building web APIs.",
    "Uvicorn runs FastAPI applications as an ASGI server.",
    "Pytest is used to write and run automated Python tests.",
    "Git helps developers track changes and manage source code.",
    "Docker packages applications and their dependencies into containers.",

    # Cooking
    "Pasta should be cooked in boiling salted water.",
    "Fresh herbs can add flavor and color to many dishes.",
    "Baking bread requires yeast, flour, water, and careful temperature control.",
    "A sharp chef's knife makes chopping vegetables easier and safer.",

    # Space
    "Mars is the fourth planet from the Sun.",
    "The James Webb Space Telescope observes distant galaxies and stars.",
    "Black holes have extremely strong gravitational fields.",
    "Astronauts experience microgravity while orbiting Earth.",

    # Music
    "A guitar produces sound when its strings vibrate.",
    "Musical scales are sequences of notes arranged by pitch.",
    "Drums provide rhythm and timing in many styles of music.",
    "A microphone converts sound into an electrical signal.",
]


# Five test queries
queries = [
    "How do I deploy my Python application?",
    "How can I test my Python code?",
    "What can I learn about planets and space?",
    "How can I prepare a good meal?",
    "How do I make something work better?",
]


# Similarity thresholds to test
thresholds = [0.3, 0.5, 0.7]


# Load the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


# Embed all knowledge base sentences once
document_embeddings = model.encode(
    knowledge_base,
    convert_to_tensor=True
)


for query in queries:

    print("=" * 70)
    print(f'Query: "{query}"')
    print("=" * 70)

    # Embed the query
    query_embedding = model.encode(
        query,
        convert_to_tensor=True
    )

    # Calculate similarity against every document
    similarity_scores = cos_sim(
        query_embedding,
        document_embeddings
    )[0]

    # Create scored results and sort from highest to lowest
    results = []

    for index, score in enumerate(similarity_scores):
        results.append(
            (score.item(), knowledge_base[index])
        )

    results.sort(reverse=True)

    # Store results for each threshold
    threshold_results = {}

    # Show results at each threshold
    for threshold in thresholds:

        matching_results = [
            (score, sentence)
            for score, sentence in results
            if score >= threshold
        ]

        threshold_results[threshold] = matching_results

        print(
            f"\n  Threshold {threshold}: "
            f"{len(matching_results)} results"
        )

        if matching_results:
            for score, sentence in matching_results:
                print(f"    [{score:.4f}] {sentence}")
        else:
            print("    No results passed this threshold.")

    # Show results that were missed as the threshold became stricter
    print("\n  Missed at stricter thresholds:")

    for i in range(1, len(thresholds)):
        lower_threshold = thresholds[i - 1]
        higher_threshold = thresholds[i]

        lower_results = threshold_results[lower_threshold]
        higher_results = threshold_results[higher_threshold]

        higher_sentences = {
            sentence for score, sentence in higher_results
        }

        missed_results = [
            (score, sentence)
            for score, sentence in lower_results
            if sentence not in higher_sentences
        ]

        print(
            f"\n    From {lower_threshold} to {higher_threshold}: "
            f"{len(missed_results)} result(s) missed"
        )

        if missed_results:
            for score, sentence in missed_results:
                print(f"      [{score:.4f}] {sentence}")
        else:
            print("      No results were missed.")

    print()