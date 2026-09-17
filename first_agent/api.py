"""
FastAPI service wrapping the agent — Phase 9 (Production & Deployment).

Exposes the same agent the CLI runs, over HTTP, with the things a service
needs that a CLI doesn't: authentication, cost budgets, concurrency safety,
and a health endpoint.

Run it:

    uvicorn api:app --reload --port 8000          # from inside first_agent/
    AGENT_API_KEY=secret uvicorn api:app --port 8000

Then:

    curl -s localhost:8000/health
    curl -s -X POST localhost:8000/ask \\
        -H 'Content-Type: application/json' \\
        -H 'X-API-Key: secret' \\
        -d '{"question": "What is 15 * 23?"}'

Configuration (all via environment):
    AGENT_API_KEY         Require this key in the X-API-Key header. If unset,
                          auth is DISABLED (fine locally, logged loudly).
    AGENT_MAX_COST_USD    Reject a request whose run would be the one that
                          pushes total spend past this. Default: 1.00.
    AGENT_MODEL / USE_OPENAI    Same as the CLI — picks the backend.
"""

import os
import secrets
import threading
import time

from fastapi import Depends, FastAPI, Header, HTTPException, status
from pydantic import BaseModel, Field

from math_agent import MODEL, USE_OPENAI, run_agent as hand_rolled_run_agent
from langgraph_agent import run_agent as langgraph_run_agent
from multi_agent import run_agent as multi_agent_run_agent
from tracer import estimate_cost_usd, get_token_usage

API_KEY = os.getenv("AGENT_API_KEY")
MAX_COST_USD = float(os.getenv("AGENT_MAX_COST_USD", "1.00"))

IMPLEMENTATIONS = {
    "hand_rolled": hand_rolled_run_agent,
    "langgraph": langgraph_run_agent,
    "multi_agent": multi_agent_run_agent,
}


# -----------------------------------------------------------------------------
# Cost budget
# -----------------------------------------------------------------------------
#
# Deliberately the opposite of the token accumulator in tracer.py: token
# tallies are per-request (ContextVar, isolated), but the budget is shared
# across every request in the process — that's the whole point of a budget.
# Shared mutable state across threads needs a lock; FastAPI runs sync
# endpoints in a threadpool, so these really can be touched concurrently.

_budget_lock = threading.Lock()
_total_spend_usd = 0.0


def _record_spend(amount: float) -> float:
    global _total_spend_usd
    with _budget_lock:
        _total_spend_usd += amount
        return _total_spend_usd


def _current_spend() -> float:
    with _budget_lock:
        return _total_spend_usd


def _budget_exhausted() -> bool:
    return _current_spend() >= MAX_COST_USD


# -----------------------------------------------------------------------------
# Auth
# -----------------------------------------------------------------------------


def require_api_key(x_api_key: str | None = Header(default=None)):
    """Validate the X-API-Key header, if an API key is configured.

    compare_digest rather than == so the check doesn't leak key length or
    content through response timing.
    """
    if API_KEY is None:
        return  # auth disabled — see the startup warning
    if not x_api_key or not secrets.compare_digest(x_api_key, API_KEY):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid X-API-Key header.",
        )


# -----------------------------------------------------------------------------
# Schemas
# -----------------------------------------------------------------------------


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    implementation: str = Field(default="hand_rolled")
    reflect: bool = Field(default=True)
    max_retries: int = Field(default=2, ge=0, le=5)


class AskResponse(BaseModel):
    answer: str
    implementation: str
    model: str
    provider: str
    duration_ms: int
    prompt_tokens: int
    completion_tokens: int
    estimated_cost_usd: float | None
    total_spend_usd: float


# -----------------------------------------------------------------------------
# App
# -----------------------------------------------------------------------------


app = FastAPI(title="Agentic AI Service", version="1.0.0")


@app.on_event("startup")
def _warn_if_unauthenticated():
    if API_KEY is None:
        print(
            "[api] WARNING: AGENT_API_KEY is not set — the /ask endpoint is "
            "UNAUTHENTICATED. Fine for local use; set AGENT_API_KEY before "
            "exposing this anywhere else."
        )
    print(f"[api] provider={'OpenAI' if USE_OPENAI else 'Ollama'} model={MODEL} budget=${MAX_COST_USD:.2f}")


@app.get("/health")
def health():
    """Liveness + current configuration. Deliberately unauthenticated so a
    container healthcheck doesn't need credentials."""
    return {
        "status": "ok",
        "model": MODEL,
        "provider": "OpenAI" if USE_OPENAI else "Ollama",
        "implementations": sorted(IMPLEMENTATIONS),
        "auth_required": API_KEY is not None,
        "budget_usd": MAX_COST_USD,
        "total_spend_usd": round(_current_spend(), 6),
    }


@app.get("/budget", dependencies=[Depends(require_api_key)])
def budget():
    spend = _current_spend()
    return {
        "budget_usd": MAX_COST_USD,
        "total_spend_usd": round(spend, 6),
        "remaining_usd": round(max(0.0, MAX_COST_USD - spend), 6),
        "exhausted": spend >= MAX_COST_USD,
    }


# Defined with `def`, not `async def`, on purpose: run_agent is blocking
# (LLM calls, tool I/O), so FastAPI runs this in its threadpool and other
# requests keep being served. An `async def` here would block the event
# loop for the whole run and serialize every request.
@app.post("/ask", response_model=AskResponse, dependencies=[Depends(require_api_key)])
def ask(request: AskRequest) -> AskResponse:
    if request.implementation not in IMPLEMENTATIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown implementation {request.implementation!r}. Valid: {sorted(IMPLEMENTATIONS)}",
        )

    # Check the budget BEFORE spending, not after — the point of a budget is
    # to stop the next call, not to report that it already overspent.
    if _budget_exhausted():
        # :g rather than a fixed :.2f — a small budget like 0.0005 would
        # otherwise render as "$0.00", making the message read as nonsense.
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail=(
                f"Cost budget exhausted: ${_current_spend():g} of ${MAX_COST_USD:g} spent. "
                "Raise AGENT_MAX_COST_USD to continue."
            ),
        )

    run = IMPLEMENTATIONS[request.implementation]
    kwargs = {
        "verbose": False,
        "trace": True,
        "reflect": request.reflect,
        "max_retries": request.max_retries,
    }
    # The LangGraph-based implementations pause for human approval on
    # external tools; there is no human on an HTTP request, so auto-approve.
    if request.implementation in ("langgraph", "multi_agent"):
        kwargs["auto_approve"] = True

    start = time.perf_counter()
    try:
        answer = run(request.question, **kwargs)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Agent run failed: {e}",
        )
    duration_ms = int((time.perf_counter() - start) * 1000)

    # get_token_usage() is a ContextVar read — this is THIS request's tally,
    # even with other requests running concurrently in sibling threads.
    usage = get_token_usage()
    cost = estimate_cost_usd(MODEL, usage["prompt_tokens"], usage["completion_tokens"])
    total_spend = _record_spend(cost or 0.0)

    return AskResponse(
        answer=answer,
        implementation=request.implementation,
        model=MODEL,
        provider="OpenAI" if USE_OPENAI else "Ollama",
        duration_ms=duration_ms,
        prompt_tokens=usage["prompt_tokens"],
        completion_tokens=usage["completion_tokens"],
        estimated_cost_usd=cost,
        total_spend_usd=round(total_spend, 6),
    )
