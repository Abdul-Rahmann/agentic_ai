"""
Test short-term memory management: long tool results get summarized before
being stored in tool_history, and tool_history itself gets capped once it
grows past MAX_TOOL_HISTORY_ENTRIES (the last open box in Phase 6).

The LLM call inside _summarize_text is mocked with a fixed, deterministic
response — this test is about the *mechanism* (does summarization trigger
at the right threshold, does the cap collapse correctly), not about
judging real summary quality, which was verified separately by hand
against real recall_knowledge output (see development-log.md, Experiment
14) since it depends on real model behavior, not just plumbing.

Run with:

    python first_agent/test_pruning.py
"""

import sys

sys.path.insert(0, "first_agent")
import math_agent

original_chat = math_agent.chat
summarize_calls = {"count": 0}


def mock_chat(messages: list, temperature: float = 0.0) -> str:
    content = messages[0].get("content", "") if messages else ""
    if "Extract the specific facts" in content:
        summarize_calls["count"] += 1
        return "- summarized fact A\n- summarized fact B"
    return original_chat(messages, temperature)


math_agent.chat = mock_chat


if __name__ == "__main__":
    # --- Test 1: a short result is left untouched, no summarization call. ---
    short_result = "42"
    stored = math_agent._maybe_summarize_tool_result("calculate", "40 + 2", short_result)
    assert stored == short_result, f"Expected short result untouched, got {stored!r}"
    assert summarize_calls["count"] == 0, "Expected no summarization call for a short result"
    print("Test 1 passed: short result left untouched, no LLM call made.")

    # --- Test 2: a long result gets summarized. ---
    long_result = "x" * (math_agent.MAX_TOOL_RESULT_CHARS + 1)
    stored = math_agent._maybe_summarize_tool_result("read_file", "big.txt", long_result)
    assert stored != long_result, "Expected the long result to be replaced by a summary"
    assert summarize_calls["count"] == 1, f"Expected exactly 1 summarization call, got {summarize_calls['count']}"
    print(f"Test 2 passed: long result ({len(long_result)} chars) summarized to {len(stored)} chars.")

    # --- Test 3: an error result is never summarized, even if long. ---
    error_result = "Error: " + "x" * (math_agent.MAX_TOOL_RESULT_CHARS + 1)
    stored = math_agent._maybe_summarize_tool_result("read_file", "missing.txt", error_result)
    assert stored == error_result, "Expected an error result to pass through unchanged"
    assert summarize_calls["count"] == 1, "Expected no new summarization call for an error result"
    print("Test 3 passed: error results are never summarized, regardless of length.")

    # --- Test 4: tool_history at (not over) the cap is left untouched. ---
    at_cap = [("calculate", f"{i}+{i}", str(i * 2)) for i in range(1, math_agent.MAX_TOOL_HISTORY_ENTRIES + 1)]
    pruned = math_agent._prune_tool_history(at_cap)
    assert pruned == at_cap, "Expected history exactly at the cap to be left untouched"
    print(f"Test 4 passed: {len(at_cap)} entries (exactly at the cap) left untouched.")

    # --- Test 5: tool_history over the cap gets collapsed. ---
    over_cap = [("calculate", f"{i}+{i}", str(i * 2)) for i in range(1, math_agent.MAX_TOOL_HISTORY_ENTRIES + 4)]
    pruned = math_agent._prune_tool_history(over_cap)
    keep_recent = math_agent.MAX_TOOL_HISTORY_ENTRIES - 1
    assert len(pruned) == math_agent.MAX_TOOL_HISTORY_ENTRIES, f"Expected {math_agent.MAX_TOOL_HISTORY_ENTRIES} entries, got {len(pruned)}"
    assert pruned[0][0] == "history_summary", "Expected the oldest entries collapsed into a history_summary entry"
    assert pruned[1:] == over_cap[-keep_recent:], "Expected the most recent entries kept in full detail"
    print(f"Test 5 passed: {len(over_cap)} entries collapsed to {len(pruned)} (1 summary + {keep_recent} recent, full detail).")

    print("\nAll pruning tests passed!")
