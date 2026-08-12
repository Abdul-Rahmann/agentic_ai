"""
Stress test for the two-phase math agent.

Runs a suite of questions against the agent, checks correctness, and reports
reliability metrics. Run with:

    python first_agent/stress_test.py

To test against OpenAI instead of Ollama:

    USE_OPENAI=1 AGENT_MODEL=gpt-4o-mini python first_agent/stress_test.py
"""

import math
import re
import statistics
import time

from math_agent import calculate, run_agent

# -----------------------------------------------------------------------------
# Test cases
# -----------------------------------------------------------------------------

TEST_CASES = [
    # (question, expected_result)
    ("What is 2 + 2?", 4),
    ("What is 15 * 23?", 345),
    ("What is 100 divided by 4?", 25),
    ("What is 2 to the power of 10?", 1024),
    ("What is 2 ** 10?", 1024),
    ("What is 2 + 2 * 3?", 8),
    ("What is (2 + 2) * 3?", 12),
    ("What is sqrt(144)?", 12),
    ("What is sqrt(144) + 5?", 17),
    ("What is the square root of 144 plus 7?", 19),
    ("What is 0 * 100?", 0),
    ("What is -5 + 3?", -2),
    ("What is 10.5 * 2?", 21),
    ("What is abs(-7)?", 7),
    ("What is pow(2, 8)?", 256),
    ("What is log(1)?", 0),
    ("What is sin(0)?", 0),
    ("What is factorial of 5?", 120),
]

# Questions that should NOT trigger the calculate tool.
NON_MATH_QUESTIONS = [
    "What is the capital of France?",
    "Who wrote Hamlet?",
    "What is your name?",
]

# Questions that mix numbers but are not math problems.
TRICK_QUESTIONS = [
    "I have 2 apples and 3 oranges. How many fruits do I have?",  # Could be 5, but should not blindly calculate
]


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def extract_number(text: str):
    """Extract the first numeric value from a string."""
    match = re.search(r"[-+]?\d*\.?\d+", text.replace(",", ""))
    if not match:
        return None
    value = match.group()
    return float(value) if "." in value else int(value)


def values_equal(a, b, tolerance: float = 1e-6) -> bool:
    """Compare numeric values with tolerance for floats."""
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) < tolerance
    return a == b


def run_single_test(question: str, expected=None):
    """Run one test and return pass/fail status and timing."""
    start = time.time()
    try:
        answer = run_agent(question, verbose=False)
    except Exception as e:
        return {
            "question": question,
            "expected": expected,
            "answer": f"EXCEPTION: {e}",
            "actual": None,
            "passed": False,
            "error": str(e),
            "duration": time.time() - start,
        }

    duration = time.time() - start
    actual = extract_number(answer)

    passed = False
    if expected is None:
        # For non-math questions, we just check the agent did not crash and gave a non-empty answer.
        passed = bool(answer.strip()) and "tool" not in answer.lower()
    else:
        passed = actual is not None and values_equal(actual, expected)

    return {
        "question": question,
        "expected": expected,
        "answer": answer,
        "actual": actual,
        "passed": passed,
        "duration": duration,
    }


# -----------------------------------------------------------------------------
# Main test runner
# -----------------------------------------------------------------------------


def main():
    results = []

    print("=" * 70)
    print("STRESS TEST: Two-Phase Math Agent")
    print("=" * 70)

    # Math questions with known answers.
    print("\n--- Math Questions ---\n")
    for question, expected in TEST_CASES:
        result = run_single_test(question, expected)
        results.append(result)
        status = "PASS" if result["passed"] else "FAIL"
        print(f"[{status}] {question}")
        print(f"         Expected: {expected}")
        print(f"         Actual:   {result['actual']}")
        print(f"         Answer:   {result['answer']}")
        print(f"         Time:     {result['duration']:.2f}s")
        print()

    # Non-math questions.
    print("\n--- Non-Math Questions ---\n")
    for question in NON_MATH_QUESTIONS:
        result = run_single_test(question, expected=None)
        results.append(result)
        status = "PASS" if result["passed"] else "FAIL"
        print(f"[{status}] {question}")
        print(f"         Answer: {result['answer']}")
        print(f"         Time:   {result['duration']:.2f}s")
        print()

    # Repetition test: same question multiple times to check consistency.
    print("\n--- Repetition Test (15 * 23 asked 5 times) ---\n")
    repeat_question = "What is 15 * 23?"
    repeat_results = [run_single_test(repeat_question, 345) for _ in range(5)]
    for i, result in enumerate(repeat_results, 1):
        status = "PASS" if result["passed"] else "FAIL"
        print(f"  Run {i}: {status} -> {result['answer']} (actual={result['actual']})")
    results.extend(repeat_results)

    # Summary.
    math_results = [r for r in results if r["expected"] is not None]
    non_math_results = [r for r in results if r["expected"] is None]

    passed_math = sum(r["passed"] for r in math_results)
    passed_non_math = sum(r["passed"] for r in non_math_results)

    durations = [r["duration"] for r in results]

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Math questions:       {passed_math} / {len(math_results)} passed")
    print(f"Non-math questions:   {passed_non_math} / {len(non_math_results)} passed")
    print(f"Total:                {passed_math + passed_non_math} / {len(results)} passed")
    print(f"Mean response time:   {statistics.mean(durations):.2f}s")
    print(f"Max response time:    {max(durations):.2f}s")
    print("=" * 70)

    # Return non-zero exit code if any test failed.
    failed = len(results) - (passed_math + passed_non_math)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
