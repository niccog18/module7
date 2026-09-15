"""
Module 7 Project — Semantic Search Engine
==========================================
evaluate.py — precision and recall for your search system

Run with:
    python evaluate.py
    python evaluate.py --n-results 5 --threshold 0.4

Define your evaluation set in EVAL_SET, then run this script against
different index configurations (different chunk sizes) to compare results.
"""

import argparse

from search import search


# Define the evaluation set here.
# Each entry contains a query and the source filenames expected to be relevant.
EVAL_SET = [
    {
        "query": "How does FastAPI work?",
        "relevant_sources": ["fastapi.txt"],
    },
    {
        "query": "What are the fundamentals of Python programming?",
        "relevant_sources": ["python-fundamentals.txt"],
    },
    {
        "query": "How do SQL databases store and retrieve data?",
        "relevant_sources": ["sql-databases.txt"],
    },
    {
        "query": "How does Streamlit build interactive web applications?",
        "relevant_sources": ["streamlit.txt"],
    },
    {
        "query": "What are embeddings and vector databases used for?",
        "relevant_sources": ["embeddings-and-vectors.txt"],
    },
]


def precision_recall(
    retrieved_sources: list[str],
    relevant_sources: list[str],
) -> tuple[float, float]:
    """
    Compute source-level precision and recall.

    Returns:
        Tuple containing precision and recall as floats from 0 to 1.
    """
    retrieved = set(retrieved_sources)
    relevant = set(relevant_sources)

    if not retrieved:
        precision = 0.0
    else:
        precision = len(retrieved & relevant) / len(retrieved)

    if not relevant:
        recall = 0.0
    else:
        recall = len(retrieved & relevant) / len(relevant)

    return precision, recall


def evaluate(
    n_results: int = 5,
    distance_threshold: float = None,
):
    """
    Run every query in EVAL_SET and print detailed evaluation results.

    For each query, display the top three results, their distances,
    scores, and whether each result came from an expected relevant source.
    Also calculate average precision and recall across all queries.
    """
    if not EVAL_SET:
        print("EVAL_SET is empty. Add evaluation queries first.")
        return

    if n_results <= 0:
        raise ValueError("n_results must be greater than 0.")

    if (
        distance_threshold is not None
        and distance_threshold < 0
    ):
        raise ValueError(
            "distance_threshold cannot be negative."
        )

    print(
        f"=== Evaluation at threshold={distance_threshold}, "
        f"n_results={n_results} ==="
    )

    precision_scores = []
    recall_scores = []

    for index, evaluation in enumerate(EVAL_SET, start=1):
        query = evaluation["query"]
        relevant_sources = evaluation["relevant_sources"]

        results = search(
            query=query,
            n_results=n_results,
            distance_threshold=distance_threshold,
        )

        retrieved_sources = [
            result["source"]
            for result in results
        ]

        precision, recall = precision_recall(
            retrieved_sources=retrieved_sources,
            relevant_sources=relevant_sources,
        )

        precision_scores.append(precision)
        recall_scores.append(recall)

        print()
        print(f"--- Query {index} ---")
        print(f"Query: {query}")
        print(f"Expected sources: {relevant_sources}")

        if results:
            print()
            print("Top 3 results:")

            for result_number, result in enumerate(
                results[:3],
                start=1,
            ):
                source = result["source"]
                distance = result["distance"]
                score = result["score"]

                is_relevant = source in relevant_sources

                if is_relevant:
                    relevance = "RELEVANT"
                else:
                    relevance = "NOT RELEVANT"

                print(
                    f"  {result_number}. "
                    f"{source} | "
                    f"chunk={result['chunk_index']} | "
                    f"distance={distance:.4f} | "
                    f"score={score:.4f} | "
                    f"{relevance}"
                )
        else:
            print()
            print("No results returned.")

        print()
        print(
            f"Retrieved sources: {retrieved_sources}"
        )
        print(
            f"Precision: {precision * 100:.1f}%"
        )
        print(
            f"Recall: {recall * 100:.1f}%"
        )

    average_precision = (
        sum(precision_scores) / len(precision_scores)
    )

    average_recall = (
        sum(recall_scores) / len(recall_scores)
    )

    print()
    print("=== Average Results ===")
    print(
        f"Average Precision: "
        f"{average_precision * 100:.1f}%"
    )
    print(
        f"Average Recall: "
        f"{average_recall * 100:.1f}%"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate search quality"
    )

    parser.add_argument(
        "--n-results",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
    )

    args = parser.parse_args()

    evaluate(
        n_results=args.n_results,
        distance_threshold=args.threshold,
    )