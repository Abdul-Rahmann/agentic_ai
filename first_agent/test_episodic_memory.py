"""
Test that episodic memory persists a reflection-flagged mistake across
separate run_agent() calls (simulating separate sessions) and that a later
run on a similar (paraphrased) question gets that mistake surfaced as
feedback before the first attempt even starts.

This mocks the LLM the same way test_retry.py does: the plan phase uses the
real model (so tool execution still works), while the answer and reflection
phases are mocked to force a wrong answer flagged as incorrect, with no
retries — so the run ends unverified and gets recorded.

Run with:

    python first_agent/test_episodic_memory.py
"""

import os
import sys

sys.path.insert(0, "first_agent")
import math_agent
import episodic_memory

# Use a throwaway DB so this test doesn't read or pollute the real
# episodes.db from interactive/stress-test runs.
TEST_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_episodes.db")
episodic_memory.DB_PATH = TEST_DB_PATH
if os.path.exists(TEST_DB_PATH):
    os.remove(TEST_DB_PATH)

original_chat = math_agent.chat
captured_prompts: list[str] = []


def mock_chat(messages: list, temperature: float = 0.0) -> str:
    """Mock chat that answers wrong UNLESS the prompt contains episodic
    feedback about a past failure — simulating an LLM that actually uses
    the hint it's given. This proves episodic memory changes the outcome,
    not just that the feedback text reached the prompt.
    """
    content = messages[0].get("content", "") if messages else ""
    captured_prompts.append(content)

    if "Now verify this:" in content:
        if "Proposed answer: 345" in content:
            return "VERIFIED: 345"
        return "INCORRECT: the calculator returned 345, not 340"

    if "Final answer:" in content:
        if "previous run" in content.lower():
            return "345"  # corrected, because it was warned about the past mistake
        return "340"  # wrong, same mistake as before, with no history to learn from

    # Plan phase: use the real model so tool execution still works.
    return original_chat(messages, temperature)


math_agent.chat = mock_chat


if __name__ == "__main__":
    # --- Run 1: a fresh "session" with no episodic history yet. ---
    print("=" * 70)
    print("Run 1: no prior episode should exist")
    print("=" * 70)
    captured_prompts.clear()
    result_1 = math_agent.run_agent("What is 15 * 23?", verbose=True, trace=False, max_retries=0)
    print(f"\nRun 1 result: {result_1!r}")

    episode = episodic_memory.recall_similar_episode("What is 15 * 23?")
    assert episode is not None, "Expected an episode to be recorded after a reflection-flagged failure"
    assert episode["verified"] is False, f"Expected verified=False, got {episode['verified']}"
    print(f"Recorded episode: {episode}")

    # --- Run 2: a new run_agent() call for a similar (paraphrased) question. ---
    print("\n" + "=" * 70)
    print("Run 2: a paraphrased question should recall Run 1's failure")
    print("=" * 70)
    captured_prompts.clear()
    result_2 = math_agent.run_agent("What is 15 times 23?", verbose=True, trace=False, max_retries=0)
    print(f"\nRun 2 result: {result_2!r}")

    plan_prompt = captured_prompts[0] if captured_prompts else ""
    assert "previous run" in plan_prompt.lower(), (
        "Expected the plan prompt on run 2 to include episodic feedback about the past failure, "
        f"but it didn't. First captured prompt:\n{plan_prompt[:500]}"
    )
    assert "345" in result_2, (
        f"Expected run 2 to self-correct to 345 given the episodic hint, got {result_2!r}"
    )
    assert result_1 != result_2, "Expected run 2's outcome to actually differ from run 1's repeated mistake"
    print("\nRun 2 corrected the mistake because of episodic feedback. Confirmed — not just present in the prompt, it changed the outcome.")

    os.remove(TEST_DB_PATH)
    print("\nEpisodic memory test passed!")
