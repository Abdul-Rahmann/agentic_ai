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

import hashlib
import os

import chromadb
from chromadb.utils import embedding_functions

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
PERSIST_DIR = os.path.join(PROJECT_ROOT, "chroma_db")
COLLECTION_NAME = "project_knowledge"

# Tracks which version of the source docs was last embedded, so seeding can
# detect staleness (a source file changed since the last index) instead of
# only checking "is the collection empty." Kept outside chroma_db/ itself so
# it's obviously a small sidecar file, not part of the vector store.
FINGERPRINT_PATH = os.path.join(PROJECT_ROOT, ".memory_fingerprint")

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


def _split_into_sections(text: str) -> list[tuple[str, str]]:
    """Split markdown into (paragraph, nearest_heading) pairs.

    Headings (lines starting with '#') are tracked as context and removed
    from the paragraph stream itself, rather than becoming their own tiny
    chunk that a merge step might separate from the content it introduces.
    """
    sections = []
    current_heading = ""
    buffer_lines: list[str] = []

    def _flush():
        if buffer_lines:
            paragraph = "\n".join(buffer_lines).strip()
            if paragraph:
                sections.append((paragraph, current_heading))
            buffer_lines.clear()

    for line in text.split("\n"):
        stripped = line.strip()
        if stripped.startswith("#"):
            _flush()
            current_heading = stripped.lstrip("#").strip()
        elif stripped == "":
            _flush()
        else:
            buffer_lines.append(line)
    _flush()

    return sections


def _chunk_text(text: str, source: str, max_chars: int = 800) -> list[dict]:
    """Split markdown into paragraph-sized chunks, merging small ones up to max_chars.

    Each chunk is tagged with the nearest section heading (e.g. "Phase 2:
    Multi-Tool Agent") so its embedding reflects what topic it belongs to,
    not just its literal words — this matters when a chunk is a short list
    item that only makes sense under its heading.
    """
    sections = _split_into_sections(text)

    chunks: list[tuple[str, str]] = []
    buffer = ""
    buffer_heading = ""
    for paragraph, heading in sections:
        # Flush on a section change so a chunk never spans two headings,
        # even if it would still fit under max_chars.
        if buffer and heading != buffer_heading:
            chunks.append((buffer, buffer_heading))
            buffer = paragraph
            buffer_heading = heading
            continue

        candidate = f"{buffer}\n\n{paragraph}" if buffer else paragraph
        if len(candidate) > max_chars and buffer:
            chunks.append((buffer, buffer_heading))
            buffer = paragraph
            buffer_heading = heading
        else:
            buffer = candidate
            buffer_heading = heading
    if buffer:
        chunks.append((buffer, buffer_heading))

    result = []
    for i, (chunk, heading) in enumerate(chunks):
        tagged_text = f"[Section: {heading}]\n{chunk}" if heading else chunk
        result.append({
            "id": f"{os.path.basename(source)}::{i}",
            "text": tagged_text,
            "source": os.path.basename(source),
        })
    return result


def _compute_fingerprint() -> str:
    """Hash each source file's path + size + mtime, to detect if any doc changed."""
    parts = []
    for source in KNOWLEDGE_SOURCES:
        if os.path.exists(source):
            stat = os.stat(source)
            parts.append(f"{source}:{stat.st_size}:{stat.st_mtime}")
        else:
            parts.append(f"{source}:missing")
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


def _read_stored_fingerprint() -> str | None:
    if os.path.exists(FINGERPRINT_PATH):
        with open(FINGERPRINT_PATH, "r", encoding="utf-8") as f:
            return f.read().strip()
    return None


def _write_fingerprint(fingerprint: str) -> None:
    with open(FINGERPRINT_PATH, "w", encoding="utf-8") as f:
        f.write(fingerprint)


def seed_knowledge_base(force: bool = False) -> int:
    """Embed and store the project docs.

    Skips re-embedding if the collection is already populated AND the
    source docs haven't changed since the last seed (tracked via a
    fingerprint file). This makes staleness self-detecting instead of
    silently depending on whatever happened to run first.

    Returns the number of chunks added (0 if nothing needed to change).
    """
    collection = _get_collection()
    current_fingerprint = _compute_fingerprint()
    stale = _read_stored_fingerprint() != current_fingerprint

    if not force and not stale and collection.count() > 0:
        return 0

    if force or stale:
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
    _write_fingerprint(current_fingerprint)
    return len(all_chunks)


def recall_knowledge(query: str, n_results: int = 4) -> str:
    """Tool entry point: retrieve the most relevant knowledge-base chunks for a query."""
    try:
        # Always check for staleness, not just emptiness — cheap (a few
        # stat() calls + a hash) unless a source doc actually changed, in
        # which case it re-seeds automatically.
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
