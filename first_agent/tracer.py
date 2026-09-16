"""
Structured tracing for the multi-tool agent.

Every run can produce a JSON trace file that records the question, model,
steps, tool calls, results, timings, and final answer. This makes failures
debuggable and allows performance and reliability analysis.
"""

import json
import os
import time
import uuid
from datetime import datetime, timezone

TRACES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "traces")

# Approximate pricing in USD per 1M tokens, (input, output). Hand-maintained
# for the models this project actually uses — verify against the provider's
# current pricing page before trusting this for anything beyond a rough
# estimate; prices drift and this table does not auto-update.
MODEL_PRICING_PER_1M = {
    "gpt-4o-mini": (0.15, 0.60),
}


def estimate_cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float | None:
    """Estimate USD cost from token counts. Returns None for unpriced models
    (e.g. a local Ollama model, which is free) rather than guessing."""
    for name, (input_price, output_price) in MODEL_PRICING_PER_1M.items():
        if name in model:
            return (prompt_tokens * input_price + completion_tokens * output_price) / 1_000_000
    return None


# -----------------------------------------------------------------------------
# Token usage accumulator
# -----------------------------------------------------------------------------
#
# The chat() / _chat() helpers in each agent implementation record usage
# here after every LLM call; run_agent() resets this at the start of a run
# and reads the total at the end. A module-level accumulator (rather than
# threading token counts through every existing tracer.add_step() call)
# keeps this additive: no existing call site needs to change shape.

_token_usage = {"prompt_tokens": 0, "completion_tokens": 0}


def reset_token_usage() -> None:
    global _token_usage
    _token_usage = {"prompt_tokens": 0, "completion_tokens": 0}


def record_token_usage(prompt_tokens: int, completion_tokens: int) -> None:
    _token_usage["prompt_tokens"] += prompt_tokens
    _token_usage["completion_tokens"] += completion_tokens


def get_token_usage() -> dict:
    return dict(_token_usage)


class Trace:
    """Records the execution history of a single agent run."""

    def __init__(self, question: str, model: str, provider: str):
        self.trace_id = str(uuid.uuid4())[:8]
        self.timestamp = datetime.now(timezone.utc).isoformat()
        self.question = question
        self.model = model
        self.provider = provider
        self.steps = []
        self.final_answer = None
        self.total_duration_ms = 0
        self.error = None
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.estimated_cost_usd = None

    def add_step(
        self,
        phase: str,
        llm_output: str,
        tool_name: str | None = None,
        tool_input: str | None = None,
        tool_result: str | None = None,
        duration_ms: int = 0,
        guard_triggered: bool = False,
        guard_reason: str | None = None,
        attempt: int = 0,
    ):
        """Add one step to the trace."""
        step = {
            "step": len(self.steps) + 1,
            "attempt": attempt,
            "phase": phase,
            "llm_output": llm_output,
            "tool_name": tool_name,
            "tool_input": tool_input,
            "tool_result": tool_result,
            "duration_ms": duration_ms,
            "guard_triggered": guard_triggered,
            "guard_reason": guard_reason,
        }
        self.steps.append(step)
        return step

    def finalize(
        self,
        final_answer: str | None,
        total_duration_ms: int,
        error: str | None = None,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
    ):
        self.final_answer = final_answer
        self.total_duration_ms = total_duration_ms
        self.error = error
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.estimated_cost_usd = estimate_cost_usd(self.model, prompt_tokens, completion_tokens)

    def to_dict(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "timestamp": self.timestamp,
            "question": self.question,
            "model": self.model,
            "provider": self.provider,
            "steps": self.steps,
            "final_answer": self.final_answer,
            "total_duration_ms": self.total_duration_ms,
            "error": self.error,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "estimated_cost_usd": self.estimated_cost_usd,
        }

    def write(self, directory: str = TRACES_DIR) -> str:
        """Write the trace to a JSON file and return the path."""
        os.makedirs(directory, exist_ok=True)
        safe_timestamp = self.timestamp.replace(":", "-")
        filename = f"{safe_timestamp}__{self.trace_id}.json"
        path = os.path.join(directory, filename)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        return path


class NullTrace:
    """A trace that records nothing. Used when tracing is disabled."""

    def __init__(self, *args, **kwargs):
        self.error = None

    def add_step(self, *args, **kwargs):
        pass

    def finalize(self, *args, **kwargs):
        pass

    def write(self, *args, **kwargs):
        return None

    def to_dict(self):
        return {}


def current_ms() -> int:
    """Return current time in milliseconds."""
    return int(time.perf_counter() * 1000)
