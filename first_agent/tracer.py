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
    ):
        self.final_answer = final_answer
        self.total_duration_ms = total_duration_ms
        self.error = error

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
