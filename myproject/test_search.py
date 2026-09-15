from search import search, get_collection_stats


print("=== COLLECTION STATS ===")
stats = get_collection_stats()
print(stats)


print("\n=== SEARCH TEST ===")
results = search(
    "How does FastAPI work?"
    ,
    n_results=3,
)

for result in results:
    print("\nSource:", result["source"])
    print("Chunk:", result["chunk_index"])
    print("Distance:", result["distance"])
    print("Score:", result["score"])
    print("Text:", result["text"][:200])


print("\n=== EMPTY QUERY TEST ===")
print(search(""))


print("\n=== SOURCE FILTER TEST ===")
results = search(
    "API framework",
    n_results=3,
    sources=["fastapi.txt"],
)

for result in results:
    print(result["source"], result["score"])