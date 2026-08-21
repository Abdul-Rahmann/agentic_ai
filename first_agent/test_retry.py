"""
Test that the retry loop is exercised when reflection flags an answer.

This script monkey-patches the LLM chat function to simulate a wrong answer on
the first attempt and a correct answer on the retry. The reflection critic is
also mocked to flag the first answer and verify the second.

Run with:

    python first_agent/test_retry.py
"""

import sys

sys.path.insert(0, "first_agent")
import math_agent

original_chat = math_agent.chat

call_counts = {"answer": 0, "reflection": 0}


def mock_chat(messages: list, temperature: float = 0.0) -> str:
    """Mock chat that returns wrong answer first, then correct, with reflection feedback."""
    content = messages[0].get("content", "") if messages else ""

    # Reflection phase: identify by the reflection prompt marker.
    if "Now verify this:" in content:
        call_counts["reflection"] += 1
        if call_counts["reflection"] == 1:
            return "INCORRECT: the calculator returned 345, not 340"
        return "VERIFIED: 345"

    # Answer phase: identify by the final answer marker.
    if "Final answer:" in content:
        call_counts["answer"] += 1
        if call_counts["answer"] == 1:
            return "340"
        return "345"

    # Plan phase: use the real model so the tool execution still works.
    return original_chat(messages, temperature)


math_agent.chat = mock_chat


if __name__ == "__main__":
    result = math_agent.run_agent("What is 15 * 23?", verbose=True, trace=False, max_retries=2)
    print(f"\nFinal result: {result}")
    print(f"Answer calls: {call_counts['answer']}")
    print(f"Reflection calls: {call_counts['reflection']}")

    assert "345" in result, f"Expected 345 in result, got {result!r}"
    assert call_counts["answer"] >= 2, f"Expected at least 2 answer calls, got {call_counts['answer']}"
    assert call_counts["reflection"] >= 2, f"Expected at least 2 reflection calls, got {call_counts['reflection']}"
    print("Retry test passed!")
