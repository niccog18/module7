# Module 7 Project — Semantic Search Engine

## Overview

This project is a semantic search tool built with Python, Sentence Transformers, ChromaDB, and Streamlit.

The application loads a collection of technical documents, splits them into overlapping chunks, converts the chunks into vector embeddings, and stores them in a persistent ChromaDB collection. Users can then enter natural-language questions through a Streamlit interface and receive ranked search results with similarity information and source documents.

The project also includes an evaluation framework and a chunking experiment comparing two different chunk sizes.

## Setup

### 1. Install dependencies

Create and activate a virtual environment, then install the project dependencies:

```bash
pip install -r requirements.txt
```

### 2. Index the document corpus

Run the ingestion script:

```bash
python ingest.py
```

The default configuration uses:

* Chunk size: 500 characters
* Overlap: 100 characters
* Embedding model: `all-MiniLM-L6-v2`

The documents are stored as embeddings in the persistent `chroma_data/` directory.

### 3. Run the Streamlit application

Start the search interface:

```bash
streamlit run app.py
```

The application provides:

* Semantic search
* Configurable result count
* Distance threshold
* Source-file filtering
* Indexed document and chunk metrics
* Document re-indexing
* Ranked search results with source, chunk, distance, score, and relevance information

### 4. Run the evaluation

The search evaluation can be run from the terminal:

```bash
python evaluate.py
```

Optional arguments can be used to change the number of results or distance threshold:

```bash
python evaluate.py --n-results 5 --threshold 0.4
```

## Project Files

| File               | Purpose                                                                                |
| ------------------ | --------------------------------------------------------------------------------------- |
| `app.py`           | Streamlit semantic search interface                                                     |
| `ingest.py`        | Loads documents, chunks text, creates embeddings, and stores them in ChromaDB           |
| `search.py`        | Queries ChromaDB and returns ranked semantic search results                             |
| `evaluate.py`      | Calculates source-level precision and recall and displays detailed evaluation results   |
| `docs/`            | Collection of technical documents used for semantic search                              |
| `chroma_data/`     | Persistent ChromaDB storage created during ingestion                                    |
| `requirements.txt` | Python package dependencies                                                             |
| `README.md`        | Project documentation and experiment results                                            |

## Document Collection

The project uses 8 technical documents covering topics related to Python, APIs, databases, AI, embeddings, and Streamlit.

The documents are:

1. `embeddings-and-vectors.txt`
2. `fastapi.txt`
3. `llms-and-ai.txt`
4. `python-advanced.txt`
5. `python-fundamentals.txt`
6. `rest-apis.txt`
7. `sql-databases.txt`
8. `streamlit.txt`

The default 500-character configuration produced 56 indexed chunks across the 8 documents.

## Architecture

The project is divided into separate components:

```text
Documents
    ↓
ingest.py
    ↓
Text Chunking
    ↓
Sentence Transformer Embeddings
    ↓
Persistent ChromaDB Collection
    ↓
search.py
    ↓
Ranked Search Results
    ↓
app.py
    ↓
Streamlit Interface
```

### Ingestion

`ingest.py` loads `.txt` and `.md` files from the `docs/` directory. Each document is split into configurable fixed-size chunks with configurable overlap.

Each chunk is stored with metadata including:

* Source filename
* Chunk index
* Chunk size

The `all-MiniLM-L6-v2` Sentence Transformer model is used to generate embeddings.

### Search

`search.py` converts the user's query into an embedding and searches the persistent ChromaDB collection for the most similar chunks.

Search results include:

* Matched text
* Source filename
* Chunk index
* Distance
* Similarity score

The search function also supports:

* Configurable `n_results`
* Source-file filtering
* Distance thresholds
* Empty-query handling

### Streamlit Interface

`app.py` provides the user interface for the search engine.

The sidebar includes:

* Indexed chunk count
* Document count
* Result-count slider
* Distance-threshold slider
* Source-file multiselect
* Re-index button

Search results display the matched text, source file, chunk number, distance, score, and a relevance indicator.

## Chunking Experiment

The chunking experiment compared two different chunk configurations using the same five test queries.

### Chunk Sizes Tested

**Configuration 1**

* Chunk size: 500 characters
* Overlap: 100 characters

**Configuration 2**

* Chunk size: 300 characters
* Overlap: 50 characters

The same five queries were evaluated against both configurations using the same search settings.

### Test Queries

1. How does FastAPI work?
2. What are the fundamentals of Python programming?
3. How do SQL databases store and retrieve data?
4. How does Streamlit build interactive web applications?
5. What are embeddings and vector databases used for?

### Top-3 Results Comparison

The evaluation script displayed the top three results for every query, including the source file, chunk number, distance, similarity score, and whether the result came from an expected relevant source.

The comparison showed the following overall patterns:

| Query                            | 500-character results                                                                   | 300-character results                                                                   | Better        |
| -------------------------------- | ----------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- | ------------- |
| FastAPI                          | Relevant FastAPI results dominated the top results                                        | Relevant FastAPI results dominated the top results                                        | Tie           |
| Python fundamentals              | Python fundamentals results dominated, with some overlap into Python advanced content     | Python fundamentals results dominated, with some overlap into Python advanced content     | Tie           |
| SQL databases                    | Top results were SQL database content                                                     | An unrelated embeddings result appeared among the top results                             | **500 chars** |
| Streamlit                        | Top results were Streamlit content                                                        | An unrelated FastAPI result appeared among the top results                                | **500 chars** |
| Embeddings and vector databases  | Embeddings results dominated, with some SQL overlap                                       | Embeddings results dominated, with some SQL overlap                                       | Tie           |

The evaluation output also classified individual top-three results as `RELEVANT` or `NOT RELEVANT`. This made it possible to compare not only whether the correct document was retrieved, but also whether unrelated documents appeared near the top of the ranking.

### Precision and Recall Results

| Query                            | 500-char Precision | 300-char Precision | 500-char Recall | 300-char Recall |
| --------------------------------- | ------------------- | ------------------- | ---------------- | ---------------- |
| FastAPI                           | 100%                | 100%                | 100%              | 100%              |
| Python fundamentals                | 50%                 | 50%                 | 100%              | 100%              |
| SQL databases                      | 100%                | 50%                 | 100%              | 100%              |
| Streamlit                          | 100%                | 50%                 | 100%              | 100%              |
| Embeddings and vector databases    | 50%                 | 50%                 | 100%              | 100%              |
| **Average**                        | **80%**             | **60%**             | **100%**          | **100%**          |

### Findings

The **500-character chunks performed better overall** for this document collection and evaluation set.

The 500-character configuration achieved an average precision of **80%**, compared with **60%** for the 300-character configuration. Both configurations achieved **100% average recall**.

The largest differences appeared in the SQL database and Streamlit queries. Both achieved 100% precision with 500-character chunks but only 50% precision with 300-character chunks.

The top-three results showed why this difference occurred. With the smaller chunks, unrelated documents appeared among the highest-ranked results for some queries. The larger chunks generally preserved more surrounding context and produced more focused results for those queries.

The results suggest that the larger chunks provided more surrounding context for the embedding model. The smaller chunks were more focused, but in several cases they removed enough surrounding context that semantically related but incorrect documents ranked higher.

This is a hypothesis based on the observed results rather than a definitive explanation. The experiment demonstrates that chunk size can have a measurable effect on semantic search quality.

For this project, the **500-character chunk size with 100-character overlap** was therefore selected as the better-performing configuration.

## Evaluation

The evaluation framework uses source-level precision and recall.

For each query:

* **Precision** measures how many unique retrieved sources were relevant.
* **Recall** measures whether the expected relevant source was retrieved.

The evaluation also displays the top three results for each query, including:

* Source filename
* Chunk index
* Distance
* Similarity score
* Relevant/not relevant classification

This evaluation was used to compare the two chunking configurations.

## Challenges

One challenge was determining how chunk size affects search quality. Smaller chunks can provide more focused pieces of information, but they can also remove contextual information that helps the embedding model identify the correct topic.

Another challenge was evaluating semantic search results consistently. The evaluation framework helped make the comparison more objective by using the same five queries and measuring precision and recall for each configuration.

## Possible Improvements

With additional development time, the project could be improved by:

* Testing additional chunk sizes and overlap values
* Adding more evaluation queries
* Using a larger evaluation dataset
* Improving the relevance scoring method
* Caching the Sentence Transformer model to reduce repeated model loading
* Adding document upload functionality
* Adding search-result highlighting
* Comparing different embedding models
* Adding more advanced ranking or reranking methods