"""
Semantic memory / RAG for the multi-tool agent.

This gives the agent a persistent, local knowledge base built from the
project's own documentation (the learning notes, roadmap, READMEs, and
development log). It is a small, self-contained example of
retrieval-augmented generation:

  1. Seed: chunk a handful of local markdown files and embed each chunk.
  2. Store: persist the chunks + embeddings in a local Chroma collection.
  3. Recall: embed a query, retrieve the most similar chunks, and return
     them as tool output so the agent can ground its answer in them.

Everything here runs locally and offline after the embedding model is
first downloaded: sentence-transformers for embeddings, Chroma for the
vector store. No API key required.
"""

import os

import chromadb
from chromadb.utils import embedding_functions

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
PERSIST_DIR = os.path.join(PROJECT_ROOT, "chroma_db")
COLLECTION_NAME = "project_knowledge"

# Source documents that make up the agent's knowledge base. Paths are
# relative to the repo root (one level above first_agent/).
REPO_ROOT = os.path.dirname(PROJECT_ROOT)
KNOWLEDGE_SOURCES = [
    os.path.join(REPO_ROOT, "agentic-ai-learning.md"),
    os.path.join(REPO_ROOT, "AGENTS_ROADMAP.md"),
    os.path.join(PROJECT_ROOT, "README.md"),
    os.path.join(PROJECT_ROOT, "development-log.md"),
]

_EMBEDDING_FN = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

_client = None
_collection = None


def _get_collection():
    """Lazily create the persistent Chroma client/collection (once per process)."""
    global _client, _collection
    if _collection is not None:
        return _collection

    _client = chromadb.PersistentClient(path=PERSIST_DIR)
    _collection = _client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=_EMBEDDING_FN,
    )
    return _collection


def _chunk_text(text: str, source: str, max_chars: int = 800) -> list[dict]:
    """Split markdown into paragraph-sized chunks, merging small ones up to max_chars."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks = []
    buffer = ""
    for paragraph in paragraphs:
        candidate = f"{buffer}\n\n{paragraph}" if buffer else paragraph
        if len(candidate) > max_chars and buffer:
            chunks.append(buffer)
            buffer = paragraph
        else:
            buffer = candidate
    if buffer:
        chunks.append(buffer)

    return [
        {"id": f"{os.path.basename(source)}::{i}", "text": chunk, "source": os.path.basename(source)}
        for i, chunk in enumerate(chunks)
    ]


def seed_knowledge_base(force: bool = False) -> int:
    """Embed and store the project docs. Skips re-embedding if already populated.

    Returns the number of chunks added (0 if the collection was already
    populated and force=False).
    """
    collection = _get_collection()
    if not force and collection.count() > 0:
        return 0

    if force:
        # Clear existing entries before re-seeding.
        existing = collection.get()
        if existing["ids"]:
            collection.delete(ids=existing["ids"])

    all_chunks = []
    for source in KNOWLEDGE_SOURCES:
        if not os.path.exists(source):
            continue
        with open(source, "r", encoding="utf-8") as f:
            text = f.read()
        all_chunks.extend(_chunk_text(text, source))

    if not all_chunks:
        return 0

    collection.add(
        ids=[c["id"] for c in all_chunks],
        documents=[c["text"] for c in all_chunks],
        metadatas=[{"source": c["source"]} for c in all_chunks],
    )
    return len(all_chunks)


def recall_knowledge(query: str, n_results: int = 3) -> str:
    """Tool entry point: retrieve the most relevant knowledge-base chunks for a query."""
    try:
        collection = _get_collection()
        if collection.count() == 0:
            seed_knowledge_base()
            collection = _get_collection()
            if collection.count() == 0:
                return "Error: knowledge base is empty (no source docs found)"

        results = collection.query(query_texts=[query], n_results=n_results)
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]

        if not documents:
            return f"No relevant knowledge found for '{query}'"

        formatted = []
        for doc, meta in zip(documents, metadatas):
            source = meta.get("source", "unknown") if meta else "unknown"
            formatted.append(f"From {source}: {doc}")
        return "\n\n".join(formatted)
    except Exception as e:
        return f"Error: recall_knowledge failed for '{query}': {e}"


if __name__ == "__main__":
    added = seed_knowledge_base(force=True)
    print(f"Seeded {added} chunks into '{COLLECTION_NAME}' at {PERSIST_DIR}")
    print()
    for q in [
        "Why did the first human-in-the-loop approval attempt fail?",
        "What local model does the math agent use by default?",
    ]:
        print(f"Query: {q}")
        print(recall_knowledge(q))
        print("---")
