"""
Trace analyzer for the multi-tool agent.

Reads JSON trace files produced by tracer.py and reports:
- total traces and date range
- duration stats (mean, median, min, max, total)
- tokens and estimated cost, broken down by model (traces from before token
  tracking was added report "no token data" rather than a misleading $0)
- per-phase duration stats
- tool usage frequency
- guard triggers and reasons
- errors, unverified answers, and retries

Usage:
    python first_agent/trace_analyzer.py
    python first_agent/trace_analyzer.py --last 20
    python first_agent/trace_analyzer.py --since 2026-08-20
    python first_agent/trace_analyzer.py --slowest --failed
"""

import argparse
import json
import os
import statistics
from collections import Counter
from datetime import datetime, timezone
from glob import glob

TRACES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "traces")


def load_traces(directory: str, since: datetime | None = None, last_n: int | None = None):
    """Load trace JSON files, optionally filtering by date or count."""
    traces = []
    for path in glob(os.path.join(directory, "*.json")):
        try:
            with open(path, "r", encoding="utf-8") as f:
                trace = json.load(f)
            ts = trace.get("timestamp")
            if ts:
                trace["_dt"] = datetime.fromisoformat(ts)
            else:
                trace["_dt"] = datetime.fromtimestamp(os.path.getmtime(path), tz=timezone.utc)
            if since and trace["_dt"] < since:
                continue
            traces.append(trace)
        except Exception as e:
            print(f"Warning: could not read {path}: {e}")

    traces.sort(key=lambda t: t["_dt"])
    if last_n is not None:
        traces = traces[-last_n:]
    return traces


def ms_to_seconds(ms: int) -> float:
    return ms / 1000.0


def format_duration(ms: int) -> str:
    return f"{ms_to_seconds(ms):.2f}s"


def main():
    parser = argparse.ArgumentParser(description="Analyze multi-tool agent traces")
    parser.add_argument("--dir", default=TRACES_DIR, help="Directory containing trace JSON files")
    parser.add_argument("--since", type=str, help="Only include traces on or after this date (YYYY-MM-DD)")
    parser.add_argument("--last", type=int, help="Only include the last N traces")
    parser.add_argument("--slowest", action="store_true", help="List the 5 slowest traces")
    parser.add_argument("--failed", action="store_true", help="List traces with errors or unverified answers")
    args = parser.parse_args()

    since = None
    if args.since:
        since = datetime.strptime(args.since, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    traces = load_traces(args.dir, since=since, last_n=args.last)

    if not traces:
        print("No traces found.")
        return

    total = len(traces)
    durations = [t["total_duration_ms"] for t in traces if t.get("total_duration_ms")]

    first_dt = traces[0]["_dt"]
    last_dt = traces[-1]["_dt"]

    print("=" * 70)
    print("TRACE ANALYZER REPORT")
    print("=" * 70)
    print(f"Total traces: {total}")
    print(f"Date range:   {first_dt.isoformat()} -> {last_dt.isoformat()}")
    print()

    if durations:
        print("-" * 70)
        print("DURATION")
        print("-" * 70)
        print(f"Mean:   {format_duration(statistics.mean(durations))}")
        print(f"Median: {format_duration(statistics.median(durations))}")
        print(f"Min:    {format_duration(min(durations))}")
        print(f"Max:    {format_duration(max(durations))}")
        print(f"Total:  {format_duration(sum(durations))}")
        print()

    token_traces = [t for t in traces if t.get("prompt_tokens") or t.get("completion_tokens")]
    print("-" * 70)
    print("TOKENS & COST")
    print("-" * 70)
    if token_traces:
        by_model: dict[str, dict] = {}
        for t in token_traces:
            model = t.get("model", "unknown")
            entry = by_model.setdefault(model, {"prompt": 0, "completion": 0, "cost": 0.0, "priced": 0, "n": 0})
            entry["n"] += 1
            entry["prompt"] += t.get("prompt_tokens", 0)
            entry["completion"] += t.get("completion_tokens", 0)
            if t.get("estimated_cost_usd") is not None:
                entry["cost"] += t["estimated_cost_usd"]
                entry["priced"] += 1

        total_prompt = sum(e["prompt"] for e in by_model.values())
        total_completion = sum(e["completion"] for e in by_model.values())
        total_cost = sum(e["cost"] for e in by_model.values())
        any_priced = any(e["priced"] for e in by_model.values())
        print(f"  Total prompt tokens:     {total_prompt:,}")
        print(f"  Total completion tokens: {total_completion:,}")
        print(f"  Estimated cost:          ${total_cost:.4f}" if any_priced else "  Estimated cost:          $0 (all traces are unpriced/local models)")
        print()
        for model, e in sorted(by_model.items()):
            cost_str = f"${e['cost']:.4f}" if e["priced"] else "free/local"
            print(f"  {model:20s} n={e['n']:3d}  prompt={e['prompt']:,}  completion={e['completion']:,}  cost={cost_str}")
    else:
        print("  No token data in these traces (written before token tracking was added).")
    print()

    phase_durations: dict[str, list[int]] = {}
    for trace in traces:
        for step in trace.get("steps", []):
            phase = step.get("phase", "unknown")
            phase_durations.setdefault(phase, []).append(step.get("duration_ms", 0))

    if phase_durations:
        print("-" * 70)
        print("DURATION BY PHASE")
        print("-" * 70)
        for phase in sorted(phase_durations):
            times = phase_durations[phase]
            print(
                f"  {phase:15s} mean={format_duration(statistics.mean(times))}  "
                f"median={format_duration(statistics.median(times))}  "
                f"max={format_duration(max(times))}  n={len(times)}"
            )
        print()

    tool_counts: Counter[str] = Counter()
    for trace in traces:
        for step in trace.get("steps", []):
            tool_name = step.get("tool_name")
            if tool_name:
                tool_counts[tool_name] += 1

    if tool_counts:
        print("-" * 70)
        print("TOOL USAGE")
        print("-" * 70)
        for tool, count in tool_counts.most_common():
            print(f"  {tool:20s} {count}")
        print()

    guard_reasons: Counter[str] = Counter()
    guard_count = 0
    for trace in traces:
        for step in trace.get("steps", []):
            if step.get("guard_triggered"):
                guard_count += 1
                reason = step.get("guard_reason") or "unknown"
                guard_reasons[reason[:80]] += 1

    print("-" * 70)
    print("GUARDS")
    print("-" * 70)
    print(f"  Total guard events: {guard_count}")
    for reason, count in guard_reasons.most_common(10):
        print(f"  {count}x {reason}")
    print()

    error_count = sum(1 for t in traces if t.get("error"))
    unverified_count = sum(
        1 for t in traces if t.get("final_answer") and "[Unverified" in str(t["final_answer"])
    )
    retry_count = sum(
        1 for t in traces if any(s.get("attempt", 0) > 0 for s in t.get("steps", []))
    )

    print("-" * 70)
    print("RELIABILITY")
    print("-" * 70)
    print(f"  Errors:             {error_count}")
    print(f"  Unverified answers: {unverified_count}")
    print(f"  Retries observed:   {retry_count}")
    print()

    if args.slowest:
        print("-" * 70)
        print("SLOWEST TRACES")
        print("-" * 70)
        slowest = sorted(traces, key=lambda t: t.get("total_duration_ms", 0), reverse=True)[:5]
        for t in slowest:
            print(f"  {format_duration(t.get('total_duration_ms', 0))}  {t.get('question')[:60]}")
        print()

    if args.failed:
        print("-" * 70)
        print("FAILED / UNVERIFIED TRACES")
        print("-" * 70)
        failed = [t for t in traces if t.get("error") or ("[Unverified" in str(t.get("final_answer")))]
        if not failed:
            print("  None")
        for t in failed:
            print(f"  {t.get('trace_id')}  {t.get('question')[:60]}")
            if t.get("error"):
                print(f"      error: {t['error']}")
            else:
                print(f"      answer: {str(t.get('final_answer'))[:80]}")
        print()

    print("=" * 70)


if __name__ == "__main__":
    main()
