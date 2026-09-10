import chromadb
 
# ---------------------------------------------------------------------------
# 1) SET UP THE COLLECTION — 12 documents across 4 topics
# ---------------------------------------------------------------------------
 
client = chromadb.Client()  # in-memory, fresh each run
collection = client.create_collection(
    name="eval_demo",
    metadata={"hnsw:space": "cosine"},  # so distance is a cosine distance in [0, 2]
)
 
documents = {
    # --- Python programming ---
    "py1": "Python functions are defined using the def keyword and can accept parameters with default values.",
    "py2": "Python lists and dictionaries are core data structures for storing collections of data.",
    "py3": "List comprehensions in Python provide a concise way to create lists using a single line of code.",
 
    # --- Space & astronomy ---
    "sp1": "A black hole is a region of spacetime where gravity is so strong that nothing, not even light, can escape.",
    "sp2": "The solar system consists of the sun and the celestial objects bound to it by gravity, including eight planets.",
    "sp3": "Stars are born in nebulae, giant clouds of gas and dust that collapse under their own gravity.",
 
    # --- Cooking & baking ---
    "ck1": "To bake bread at home, mix flour, water, yeast, and salt, then let the dough rise before baking.",
    "ck2": "Sauteing vegetables in olive oil over medium heat brings out their natural sweetness.",
    "ck3": "A classic vinaigrette is made by whisking olive oil, vinegar, mustard, salt, and pepper together.",
 
    # --- Fitness & health ---
    "ft1": "Strength training with weights builds muscle mass and increases bone density over time.",
    "ft2": "Cardiovascular exercise like running or cycling improves heart health and endurance.",
    "ft3": "Proper protein intake after a workout supports muscle recovery and growth.",
}
 
collection.add(
    ids=list(documents.keys()),
    documents=list(documents.values()),
)
 
# ---------------------------------------------------------------------------
# 2) EVALUATION SET — 8 test queries with expected relevant doc IDs
# ---------------------------------------------------------------------------
 
test_queries = [
    {"query": "How do I define a function in Python?", "relevant_ids": ["py1"]},
    {"query": "What Python data structures store collections of items?", "relevant_ids": ["py2", "py3"]},
    {"query": "What is a black hole and how does gravity work in space?", "relevant_ids": ["sp1"]},
    {"query": "Tell me about the planets in our solar system", "relevant_ids": ["sp2"]},
    {"query": "How do I bake bread at home?", "relevant_ids": ["ck1"]},
    {"query": "What's a good recipe for salad dressing?", "relevant_ids": ["ck3"]},
    {"query": "How can I build muscle through exercise?", "relevant_ids": ["ft1", "ft3"]},
    {"query": "Best cardio workouts for heart health", "relevant_ids": ["ft2"]},
]
 
# ---------------------------------------------------------------------------
# 3) evaluate() — runs every query, computes precision/recall, prints results
# ---------------------------------------------------------------------------
 
def evaluate(collection, test_queries, threshold, n_results, verbose=True):
    """
    For each test query:
      - retrieve n_results candidates from the collection
      - keep only candidates whose similarity (1 - cosine distance) >= threshold
      - precision = (relevant docs retrieved) / (docs retrieved)
      - recall    = (relevant docs retrieved) / (total relevant docs)
    Returns (avg_precision, avg_recall) and prints a per-query + summary report.
    """
    precisions, recalls = [], []
 
    if verbose:
        print(f"\n=== Evaluation at threshold={threshold}, n_results={n_results} ===")
 
    for i, tq in enumerate(test_queries, 1):
        result = collection.query(query_texts=[tq["query"]], n_results=n_results)
        ids = result["ids"][0]
        distances = result["distances"][0]
 
        # convert distance -> similarity and apply threshold
        retrieved = [doc_id for doc_id, dist in zip(ids, distances) if (1 - dist) >= threshold]
 
        relevant = set(tq["relevant_ids"])
        retrieved_set = set(retrieved)
        true_positives = len(retrieved_set & relevant)
 
        precision = true_positives / len(retrieved_set) if retrieved_set else 0.0
        recall = true_positives / len(relevant) if relevant else 0.0
 
        precisions.append(precision)
        recalls.append(recall)
 
        if verbose:
            print(f"Query {i}: P={precision*100:.1f}% R={recall*100:.1f}%  "
                  f"| retrieved={retrieved} expected={sorted(relevant)}")
 
    avg_precision = sum(precisions) / len(precisions)
    avg_recall = sum(recalls) / len(recalls)
 
    if verbose:
        print(f"AVERAGE: P={avg_precision*100:.1f}% R={avg_recall*100:.1f}%")
 
    return avg_precision, avg_recall
 
 
# ---------------------------------------------------------------------------
# 4) RUN AT 3 DIFFERENT SETTINGS
# ---------------------------------------------------------------------------
 
settings = [
    {"threshold": 0.3, "n_results": 3},
    {"threshold": 0.5, "n_results": 3},
    {"threshold": 0.5, "n_results": 5},
]
 
results_summary = []
for s in settings:
    avg_p, avg_r = evaluate(collection, test_queries, **s)
    results_summary.append((s, avg_p, avg_r))
 
# ---------------------------------------------------------------------------
# 5) ANALYSIS
# ---------------------------------------------------------------------------
print("\n=== SETTINGS COMPARISON ===")
for s, avg_p, avg_r in results_summary:
    print(f"threshold={s['threshold']}, n_results={s['n_results']}  ->  "
          f"P={avg_p*100:.1f}%  R={avg_r*100:.1f}%")
 
print("""
ANALYSIS:
 
- Low threshold (0.3) + n_results=3 produced the highest recall at 100.0%,
meaning every expected relevant document was retrieved across the test
queries. However, precision was lower at 66.7% because the search also
returned some irrelevant documents. For example, Query 1 returned py1,
py2, and py3 even though only py1 was expected to be relevant. This shows
that a lower threshold can improve recall while allowing more false
positives.

- Raising the threshold to 0.5 with n_results=3 increased average precision
from 66.7% to 87.5%. This means the search results became more focused and
contained fewer irrelevant documents. However, average recall decreased
from 100.0% to 81.2% because some relevant documents were filtered out.
Query 6 is a good example: the relevant cooking document ck3 was retrieved
at the lower threshold, but no document was returned at the 0.5 threshold.

- Increasing n_results from 3 to 5 while keeping the threshold at 0.5 did
not change the overall results. Both settings produced 87.5% precision
and 81.2% recall. This suggests that the additional candidates returned
when n_results was increased were not similar enough to pass the 0.5
threshold.


Queries that worked well:

- Query 2, about Python data structures, achieved 100.0% precision and
100.0% recall at all settings. Both expected documents, py2 and py3,
were retrieved correctly.

- Query 5, about baking bread, achieved 100.0% precision and 100.0%
recall at all settings.

- Query 8, about cardio workouts and heart health, also achieved 100.0%
precision and 100.0% recall at all settings.

- The space-related queries also performed very well at the 0.5 threshold,
with Query 3 and Query 4 both achieving 100.0% precision and recall.


Queries that struggled:

- Query 6, about salad dressing, was the most difficult query. At the
0.3 threshold it retrieved the correct document ck3 but also retrieved
ck2, resulting in 50.0% precision. At the 0.5 threshold it retrieved
nothing, resulting in 0.0% precision and 0.0% recall.

- Query 7, about building muscle through exercise, also showed a tradeoff.
At the 0.3 threshold it retrieved both relevant documents, ft1 and ft3,
but also retrieved the irrelevant cardio document ft2, resulting in
66.7% precision and 100.0% recall. At the 0.5 threshold, only ft3 was
retrieved, giving 100.0% precision but only 50.0% recall.

- Query 1 also demonstrated the precision problem at the lower threshold.
It retrieved all three Python documents even though only py1 was expected,
resulting in 33.3% precision.


What I'd change to improve scores:

- Rewrite documents that are expected to be relevant to a query so they
contain more vocabulary related to the query. For example, the fitness
documents could mention both "building muscle" and "strength training"
so that both ft1 and ft3 are easier to retrieve for muscle-building
questions.

- Adjust the similarity threshold based on testing rather than assuming
that a higher threshold is always better. The results show that 0.5
improved precision but caused some relevant documents to be missed.

- Add more evaluation queries for each topic. More test queries would
provide a better measurement of whether the search system works well
across different ways of asking the same question.

- Experiment with different n_results values and thresholds together.
The results show that increasing n_results from 3 to 5 did not help at
a 0.5 threshold, so additional experiments could determine whether a
different threshold or result count provides a better balance between
precision and recall.


Overall, the 0.5 threshold with n_results=3 produced the best balance for
this dataset because it increased precision from 66.7% to 87.5%, although
it reduced recall from 100.0% to 81.2%. The experiment demonstrates the
tradeoff between retrieving more relevant documents and avoiding irrelevant
results.
""")