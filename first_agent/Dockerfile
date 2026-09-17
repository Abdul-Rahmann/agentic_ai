# Container for the agent service (Phase 9).
#
# Build (from the repo root, so the knowledge-base docs are in context):
#   docker build -f first_agent/Dockerfile -t agentic-ai .
#
# Run against OpenAI (simplest — no host dependency):
#   docker run --rm -p 8000:8000 \
#     -e AGENT_API_KEY=secret -e USE_OPENAI=1 -e AGENT_MODEL=gpt-4o-mini \
#     -e OPENAI_API_KEY=sk-... agentic-ai
#
# Run against Ollama on the HOST (Docker Desktop):
#   docker run --rm -p 8000:8000 \
#     -e AGENT_API_KEY=secret \
#     -e OLLAMA_HOST=http://host.docker.internal:11434 agentic-ai
#
# Note: Ollama runs on the host, not in this image — the model is several GB
# and belongs outside the service container. host.docker.internal is how the
# container reaches it on Docker Desktop; on plain Linux you'd use
# --network=host or the host's bridge IP instead.

FROM python:3.12-slim

# Keeps Python from writing .pyc files and buffering logs (so container
# logs actually appear in `docker logs` in real time).
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies first, in their own layer — this is the slow,
# rarely-changing step (torch is multi-GB), so it stays cached across
# rebuilds when only application code changes.
COPY first_agent/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Bake the embedding model into the image rather than downloading it on the
# first request. Without this, the first recall_knowledge call in a fresh
# container pays a multi-second model download — a cold-start penalty that
# would show up as one mysteriously slow request in production.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# The agent's knowledge base is built from the repo's own markdown docs, so
# they have to be in the image for recall_knowledge to have anything to
# retrieve. This is why the build context is the repo root, not first_agent/.
COPY agentic-ai-learning.md AGENTS_ROADMAP.md ./
COPY first_agent/ ./first_agent/

WORKDIR /app/first_agent

# Build the vector index at image-build time, not on the first request.
# Measured: without this, the first recall_knowledge call in a fresh
# container took ~111s vs ~50s warm — roughly a minute of that was
# embedding the knowledge base from scratch. chroma_db/ is correctly
# excluded from the build context as derived data, so the index has to be
# built somewhere; doing it here moves the cost to build time, where it's
# paid once, instead of onto whichever unlucky request arrives first.
RUN python -c "from memory import seed_knowledge_base; print(f'seeded {seed_knowledge_base(force=True)} chunks')"

EXPOSE 8000

# Unauthenticated /health, so this needs no credentials.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health').read()" || exit 1

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
