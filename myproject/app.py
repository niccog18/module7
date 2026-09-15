"""
Module 7 Project — Semantic Search Engine
==========================================
app.py — Streamlit search interface

Run with:
    streamlit run app.py

Make sure you've indexed documents first:
    python ingest.py
"""

import streamlit as st

from search import search, get_collection_stats
from ingest import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_OVERLAP,
    ingest,
)


# ── Page Configuration ────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Semantic Search Engine",
    page_icon="🔎",
    layout="wide",
)


# ── Header ────────────────────────────────────────────────────────────────────

st.title("🔎 Semantic Search Engine")

st.write(
    "Search the indexed document collection using semantic similarity. "
    "Results are ranked by embedding distance."
)


# ── Collection Statistics ─────────────────────────────────────────────────────

stats = get_collection_stats()

total_chunks = stats["total_chunks"]
unique_sources = stats["unique_sources"]
source_names = stats["source_names"]


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Search Settings")

    st.metric(
        label="Indexed Chunks",
        value=total_chunks,
    )

    st.metric(
        label="Documents",
        value=unique_sources,
    )

    st.divider()

    st.subheader("Result Settings")

    n_results = st.slider(
        "Number of results",
        min_value=1,
        max_value=10,
        value=5,
        step=1,
    )

    distance_threshold = st.slider(
        "Distance threshold",
        min_value=0.0,
        max_value=2.0,
        value=2.0,
        step=0.05,
        help=(
            "Only results with a distance at or below this value "
            "will be shown. Lower distance means greater similarity."
        ),
    )

    st.divider()

    st.subheader("Source Filter")

    selected_sources = st.multiselect(
        "Search specific documents",
        options=source_names,
        help="Leave empty to search all indexed documents.",
    )

    st.divider()

    st.subheader("Index")

    st.write(
        f"Current chunk size: **{DEFAULT_CHUNK_SIZE}** characters"
    )

    st.write(
        f"Current overlap: **{DEFAULT_OVERLAP}** characters"
    )

    if st.button(
        "🔄 Re-index Documents",
        use_container_width=True,
    ):
        with st.spinner("Re-indexing documents..."):
            try:
                ingest(
                    chunk_size=DEFAULT_CHUNK_SIZE,
                    overlap=DEFAULT_OVERLAP,
                )

                st.success("Documents re-indexed successfully.")

                st.rerun()

            except Exception as error:
                st.error(f"Re-indexing failed: {error}")


# ── Search Input ──────────────────────────────────────────────────────────────

query = st.text_input(
    "What would you like to search for?",
    placeholder="Example: How does FastAPI handle data validation?",
)


# ── Search ────────────────────────────────────────────────────────────────────

if query.strip():
    with st.spinner("Searching..."):
        try:
            results = search(
                query=query,
                n_results=n_results,
                sources=selected_sources or None,
                distance_threshold=distance_threshold,
            )

        except Exception as error:
            st.error(f"Search failed: {error}")
            results = []

    st.subheader("Search Results")

    if results:
        st.write(
            f"Found **{len(results)}** relevant result(s)."
        )

        for index, result in enumerate(results, start=1):
            score = result["score"]
            distance = result["distance"]

            if distance <= 0.5:
                relevance = "🟢 High relevance"
            elif distance <= 1.0:
                relevance = "🟡 Moderate relevance"
            else:
                relevance = "🔴 Lower relevance"

            with st.container(border=True):
                st.markdown(
                    f"### {index}. {result['source']}"
                )

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.write(
                        f"**Chunk:** {result['chunk_index']}"
                    )

                with col2:
                    st.write(
                        f"**Distance:** {distance:.4f}"
                    )

                with col3:
                    st.write(
                        f"**Score:** {score:.4f}"
                    )

                st.write(f"**Relevance:** {relevance}")

                st.markdown("**Matched Text:**")

                st.write(result["text"])

    else:
        st.warning(
            "No results matched your search and current filters. "
            "Try a broader query, increase the distance threshold, "
            "or remove the source filter."
        )

else:
    st.info(
        "Enter a question above to search the document collection."
    )