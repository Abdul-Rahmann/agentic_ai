"""
Episodic memory for the multi-tool agent.

Unlike memory.py (semantic memory over static project docs), this module
remembers the agent's OWN past runs: what was asked, what it answered, and
whether reflection verified that answer. On a later run of a similar
question, a past unverified (incorrect) answer is surfaced as feedback —
reusing the exact same reflection_feedback mechanism the in-run retry loop
already uses — so the agent doesn't repeat a known mistake across separate
sessions, not just within one.

Uses SQLite for the structured record (question, answer, verified flag,
reflection reason, timestamp) — a good fit for that shape of data. Matching
"is this a similar question" reuses the same sentence-transformers
embedding model memory.py already loads, rather than character-level
string diffing: an earlier version used difflib.SequenceMatcher, but that
scored "What is 15 * 23?" vs "What is 15 times 23?" at only 0.83 similarity
(below the 0.85 threshold) purely because "*" and "times" look nothing
alike character-by-character — the same keyword-vs-meaning gap the RAG
knowledge base runs into, just showing up in a different spot. Embedding
similarity recognizes they mean the same thing regardless of the literal
characters used.
"""

import os
import sqlite3
from datetime import datetime, timezone

import numpy as np

from memory import _EMBEDDING_FN

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(PROJECT_ROOT, "episodes.db")

# How similar (0-1, cosine similarity on sentence embeddings) a past
# question must be to the current one to count as "the same question."
SIMILARITY_THRESHOLD = 0.7


def _get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS episodes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            question TEXT NOT NULL,
            final_answer TEXT,
            verified INTEGER NOT NULL,
            reflection_feedback TEXT
        )
        """
    )
    conn.commit()
    return conn


def _cosine_similarity(a, b) -> float:
    a, b = np.asarray(a), np.asarray(b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def record_episode(
    question: str,
    final_answer: str,
    verified: bool,
    reflection_feedback: str | None = None,
) -> None:
    """Record the outcome of a completed run for future recall."""
    conn = _get_connection()
    try:
        conn.execute(
            "INSERT INTO episodes (timestamp, question, final_answer, verified, reflection_feedback) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                datetime.now(timezone.utc).isoformat(),
                question,
                final_answer,
                1 if verified else 0,
                reflection_feedback,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def recall_similar_episode(question: str, only_unverified: bool = True) -> dict | None:
    """Find the best-matching past episode for a similar question.

    Returns None if nothing is similar enough. If only_unverified is True
    (the default), a match is only returned when that past attempt was
    flagged incorrect — there's nothing useful to warn about for a question
    that was already answered correctly before.
    """
    conn = _get_connection()
    try:
        rows = conn.execute(
            "SELECT timestamp, question, final_answer, verified, reflection_feedback "
            "FROM episodes ORDER BY id DESC"
        ).fetchall()
    finally:
        conn.close()

    candidates = [row for row in rows if not (only_unverified and row[3])]
    if not candidates:
        return None

    query_embedding = _EMBEDDING_FN([question])[0]
    candidate_embeddings = _EMBEDDING_FN([row[1] for row in candidates])

    best_match = None
    best_score = 0.0
    for (timestamp, past_question, final_answer, verified, feedback), embedding in zip(
        candidates, candidate_embeddings
    ):
        score = _cosine_similarity(query_embedding, embedding)
        if score >= SIMILARITY_THRESHOLD and score > best_score:
            best_score = score
            best_match = {
                "timestamp": timestamp,
                "question": past_question,
                "final_answer": final_answer,
                "verified": bool(verified),
                "reflection_feedback": feedback,
                "similarity": score,
            }
    return best_match


def format_episode_feedback(episode: dict) -> str:
    """Turn a recalled episode into a feedback string for the plan/answer prompts."""
    note = (
        f"A similar question was answered incorrectly in a previous run. "
        f"That answer was '{episode['final_answer']}'"
    )
    if episode.get("reflection_feedback"):
        note += f" — the reason was: {episode['reflection_feedback']}"
    note += ". Do not repeat that mistake."
    return note


if __name__ == "__main__":
    # Roundtrip smoke test: record a failure, then recall it via a paraphrase.
    record_episode(
        "What is 15 * 23?",
        "340",
        verified=False,
        reflection_feedback="the calculator returned 345, not 340",
    )
    match = recall_similar_episode("What is 15 times 23?")
    print("Recalled episode:", match)
    if match:
        print("Formatted feedback:", format_episode_feedback(match))
