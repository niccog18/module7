import streamlit as st
import chromadb
import os

st.set_page_config(
    page_title="Semantic Search",
    page_icon="🔍",
    layout="wide"
)


# ============================================================
# ChromaDB Setup
# ============================================================

@st.cache_resource
def get_collection():
    client = chromadb.PersistentClient(path="./search_db")
    return client.get_or_create_collection(name="course_docs")


collection = get_collection()


# ============================================================
# Document Loading & Chunking
# ============================================================

def load_and_chunk(directory, chunk_size=400, overlap=50):
    """Load text files and split them into chunks."""

    chunks = []

    for filename in sorted(os.listdir(directory)):

        if not filename.endswith((".txt", ".md")):
            continue

        filepath = os.path.join(directory, filename)

        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        # Split by paragraphs
        paragraphs = [
            p.strip()
            for p in content.split("\n\n")
            if p.strip()
        ]

        for i, para in enumerate(paragraphs):

            chunks.append({
                "text": para,
                "source": filename,
                "chunk_id": f"{filename}_{i}",
                "chunk_index": i,
            })

    return chunks


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:

    st.title("📁 Document Manager")

    # --------------------------------------------------------
    # Re-index Documents
    # --------------------------------------------------------

    if st.button("🔄 Re-index Documents"):

        chunks = load_and_chunk("docs")

        if chunks:

            collection.upsert(
                documents=[c["text"] for c in chunks],

                metadatas=[
                    {
                        "source": c["source"],
                        "chunk_index": str(c["chunk_index"])
                    }
                    for c in chunks
                ],

                ids=[c["chunk_id"] for c in chunks],
            )

            st.success(
                f"Indexed {len(chunks)} chunks from "
                f"{len(set(c['source'] for c in chunks))} files"
            )

            # Refresh the app so the new source list appears
            st.rerun()

        else:

            st.warning(
                "No .txt or .md files found in docs/ folder"
            )

    # --------------------------------------------------------
    # Database Statistics
    # --------------------------------------------------------

    total_documents = collection.count()

    st.metric(
        "Documents in DB",
        total_documents
    )

    # --------------------------------------------------------
    # Find Available Source Files
    # --------------------------------------------------------

    available_sources = []

    if total_documents > 0:

        all_metadata = collection.get(
            include=["metadatas"]
        )["metadatas"]

        available_sources = sorted(
            set(
                metadata["source"]
                for metadata in all_metadata
                if metadata.get("source")
            )
        )

    st.metric(
        "Unique Source Files",
        len(available_sources)
    )

    st.divider()

    # --------------------------------------------------------
    # Source Filter
    # --------------------------------------------------------

    st.subheader("🔎 Search Filters")

    selected_sources = st.multiselect(
        "Filter by source",
        options=available_sources,
        default=available_sources
    )

    # --------------------------------------------------------
    # Number of Results
    # --------------------------------------------------------

    n_results = st.slider(
        "Results to show",
        min_value=1,
        max_value=10,
        value=5
    )


# ============================================================
# Main Search Interface
# ============================================================

st.title("🔍 Semantic Search")

st.write(
    "Search your course documents by meaning, "
    "not just keywords."
)

query = st.text_input(
    "Enter your search query",
    placeholder="How does authentication work?"
)


# ============================================================
# Search
# ============================================================

if query and collection.count() > 0:

    # --------------------------------------------------------
    # Build ChromaDB query
    # --------------------------------------------------------

    query_args = {
        "query_texts": [query],
        "n_results": min(n_results, collection.count())
    }

    # Only apply source filter when specific sources
    # have been selected.
    if selected_sources:

        query_args["where"] = {
            "source": {
                "$in": selected_sources
            }
        }

    # --------------------------------------------------------
    # Run Search
    # --------------------------------------------------------

    results = collection.query(**query_args)

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    # Number of results returned
    result_count = len(documents)

    # --------------------------------------------------------
    # Result Count Display
    # --------------------------------------------------------

    st.subheader(
        f"Showing {result_count} of {collection.count()} "
        f"total documents"
    )

    # --------------------------------------------------------
    # Display Results
    # --------------------------------------------------------

    for i in range(result_count):

        doc = documents[i]
        metadata = metadatas[i]
        distance = distances[i]

        # ----------------------------------------------------
        # Relevance Badge
        # ----------------------------------------------------

        if distance < 0.5:

            relevance = "🟢 High"

        elif distance < 1.0:

            relevance = "🟡 Medium"

        else:

            relevance = "🔴 Low"

        # First 150 characters
        preview = doc[:150]

        if len(doc) > 150:
            preview += "..."

        # ----------------------------------------------------
        # Result Container
        # ----------------------------------------------------

        with st.container():

            col_meta, col_score = st.columns([3, 1])

            with col_meta:

                st.write(
                    f"**{metadata['source']}** "
                    f"— chunk {metadata['chunk_index']}"
                )

            with col_score:

                st.write(
                    f"{relevance} "
                    f"(dist: {distance:.3f})"
                )

            # ------------------------------------------------
            # Preview
            # ------------------------------------------------

            st.write(preview)

            # ------------------------------------------------
            # Expandable Full Text
            # ------------------------------------------------

            with st.expander("📖 View full text"):

                st.write(doc)

            st.divider()


elif query and collection.count() == 0:

    st.info(
        "👈 Click 'Re-index Documents' in the sidebar "
        "to load your documents first."
    )

elif query and not selected_sources:

    st.warning(
        "Please select at least one source file "
        "in the sidebar."
    )