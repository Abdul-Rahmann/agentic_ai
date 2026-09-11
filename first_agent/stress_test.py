"""
Stress test for the multi-tool agent.

Runs a suite of questions against the agent, checks correctness, and reports
reliability metrics. Run with:

    python first_agent/stress_test.py

To test against OpenAI instead of Ollama:

    USE_OPENAI=1 AGENT_MODEL=gpt-4o-mini python first_agent/stress_test.py
"""

import argparse
import os
import re
import statistics
import time
from datetime import datetime

import sys
sys.path.insert(0, "first_agent")
from math_agent import run_agent as hand_rolled_run_agent
from langgraph_agent import run_agent as langgraph_run_agent
from multi_agent import run_agent as multi_agent_run_agent
import episodic_memory

# Isolate episodic memory into its own throwaway DB during benchmark runs,
# the same principle as trace=False: a stress test that let past runs'
# outcomes influence future runs (e.g. a flagged answer getting "corrected"
# by episodic feedback next time) would no longer be a reproducible,
# apples-to-apples benchmark. Reset fresh on every stress_test.py run.
episodic_memory.DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stress_test_episodes.db")
if os.path.exists(episodic_memory.DB_PATH):
    os.remove(episodic_memory.DB_PATH)

DATA_DIR = "first_agent/data"

TODAY_STR = datetime.now().strftime("%Y-%m-%d")
TODAY_YEAR = datetime.now().strftime("%Y")

# -----------------------------------------------------------------------------
# Test cases
# -----------------------------------------------------------------------------

MATH_QUESTIONS = [
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

READ_FILE_QUESTIONS = [
    (
        "What is in data/numbers.txt?",
        ["12", "15", "23", "8"],
    ),
]

MULTI_STEP_QUESTIONS = [
    ("What is the sum of the numbers in data/numbers.txt?", 68),
    ("What is the product of the numbers in data/numbers.txt?", 993600),
]

NON_MATH_QUESTIONS = [
    ("What is the capital of France?", None),
    ("Who wrote Hamlet?", None),
    ("What is your name?", None),
]

# Weather questions. The actual temperature and conditions change, so we only
# check that the response contains expected markers, not exact values.
WEATHER_QUESTIONS = [
    (
        "What is the weather in Paris?",
        # The answer-synthesis prompt asks for only the final answer, so the
        # model sometimes drops the city name from a terse reply (e.g.
        # "19.9°C, partly cloudy.") even though it correctly used the tool.
        # "°C" is the real invariant: it proves a live temperature came back.
        ["°C"],
    ),
]

WEATHER_MATH_QUESTIONS = [
    (
        "What is the temperature in Paris plus 10?",
        None,  # any numeric answer is acceptable; exact value depends on current weather
    ),
]

# Web search questions. Wikipedia content can change, so we check for expected
# substrings rather than exact values.
WEB_SEARCH_QUESTIONS = [
    (
        "Who is the CEO of OpenAI?",
        ["Sam Altman"],
    ),
]

# Date questions. The expected answer is today's date, so we generate the
# expected substrings at runtime.
# Memory / RAG questions. These ask about this project's own history, which
# only exists in the local knowledge base (project docs), not general
# training data or the web — a real test of retrieval, not recall.
MEMORY_QUESTIONS = [
    (
        "Why did the first human-in-the-loop approval attempt in this project fail?",
        # A correct answer may name the node literally or paraphrase the
        # recursion behavior — either counts as proof retrieval worked.
        [("await_approval", "recursed", "recursion", "recursing")],
    ),
    (
        "What local model does this project's math agent use by default?",
        ["llama3.1"],
    ),
]

DATE_QUESTIONS = [
    (
        "What is today's date?",
        [TODAY_STR, TODAY_YEAR],
    ),
    (
        "What date is today?",
        [TODAY_STR, TODAY_YEAR],
    ),
]


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------



def extract_number(text: str):
    """Extract the first numeric value from a string."""
    text = text.replace(",", "")
    match = re.search(r"[-+]?\d*\.?\d+", text)
    if not match:
        return None
    value = match.group()
    return float(value) if "." in value else int(value)


def values_equal(a, b, tolerance: float = 1e-6) -> bool:
    """Compare numeric values with tolerance for floats."""
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) < tolerance
    return a == b


def run_single_test(
    question: str,
    expected=None,
    expected_substrings=None,
    expect_number: bool = False,
    run_agent=hand_rolled_run_agent,
):
    """Run one test and return pass/fail status and timing."""
    start = time.time()
    try:
        answer = run_agent(question, verbose=False, trace=False)
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

    if expected is not None:
        passed = actual is not None and values_equal(actual, expected)
    elif expected_substrings:
        # Each requirement is either a plain string (must appear as-is) or a
        # tuple/list of alternative phrasings (any one of them counts) — the
        # latter tolerates a correct answer being paraphrased differently
        # each run, rather than requiring one exact identifier verbatim.
        answer_lower = answer.lower()
        passed = all(
            any(alt.lower() in answer_lower for alt in req) if isinstance(req, (list, tuple))
            else req.lower() in answer_lower
            for req in expected_substrings
        )
    elif expect_number:
        passed = actual is not None
    else:
        # Non-math question with no explicit expected value: just check it answered.
        passed = bool(answer.strip()) and "tool" not in answer.lower()

    return {
        "question": question,
        "expected": expected,
        "expected_substrings": expected_substrings,
        "answer": answer,
        "actual": actual,
        "passed": passed,
        "duration": duration,
    }


# -----------------------------------------------------------------------------
# Main test runner
# -----------------------------------------------------------------------------


def print_result(result: dict):
    status = "PASS" if result["passed"] else "FAIL"
    print(f"[{status}] {result['question']}")
    print(f"         Answer: {result['answer']!r}")
    if result["expected"] is not None:
        print(f"         Expected: {result['expected']}")
        print(f"         Actual:   {result['actual']}")
    elif result.get("expected_substrings"):
        print(f"         Expected substrings: {result['expected_substrings']}")
    print(f"         Time:     {result['duration']:.2f}s")
    print()


def main(run_agent=hand_rolled_run_agent, agent_label: str = "Hand-Rolled"):
    results = []

    print("=" * 70)
    print(f"STRESS TEST: Multi-Tool Agent ({agent_label})")
    print("=" * 70)

    # Math questions.
    print("\n--- Math Questions ---\n")
    for question, expected in MATH_QUESTIONS:
        result = run_single_test(question, expected=expected, run_agent=run_agent)
        results.append(result)
        print_result(result)

    # Read-file questions.
    print("\n--- Read-File Questions ---\n")
    for question, expected_substrings in READ_FILE_QUESTIONS:
        result = run_single_test(question, expected_substrings=expected_substrings, run_agent=run_agent)
        results.append(result)
        print_result(result)

    # Multi-step questions.
    print("\n--- Multi-Step Questions ---\n")
    for question, expected in MULTI_STEP_QUESTIONS:
        result = run_single_test(question, expected=expected, run_agent=run_agent)
        results.append(result)
        print_result(result)

    # Weather questions.
    print("\n--- Weather Questions ---\n")
    for question, expected_substrings in WEATHER_QUESTIONS:
        result = run_single_test(question, expected_substrings=expected_substrings, run_agent=run_agent)
        results.append(result)
        print_result(result)

    # Weather + math questions.
    print("\n--- Weather + Math Questions ---\n")
    for question, _ in WEATHER_MATH_QUESTIONS:
        result = run_single_test(question, expect_number=True, run_agent=run_agent)
        results.append(result)
        print_result(result)

    # Web search questions.
    print("\n--- Web Search Questions ---\n")
    for question, expected_substrings in WEB_SEARCH_QUESTIONS:
        result = run_single_test(question, expected_substrings=expected_substrings, run_agent=run_agent)
        results.append(result)
        print_result(result)

    # Memory / RAG questions.
    print("\n--- Memory (RAG) Questions ---\n")
    for question, expected_substrings in MEMORY_QUESTIONS:
        result = run_single_test(question, expected_substrings=expected_substrings, run_agent=run_agent)
        results.append(result)
        print_result(result)

    # Date questions.
    print("\n--- Date Questions ---\n")
    for question, expected_substrings in DATE_QUESTIONS:
        result = run_single_test(question, expected_substrings=expected_substrings, run_agent=run_agent)
        results.append(result)
        print_result(result)

    # Non-math questions.
    print("\n--- Non-Math Questions ---\n")
    for question, _ in NON_MATH_QUESTIONS:
        result = run_single_test(question, run_agent=run_agent)
        results.append(result)
        print_result(result)

    # Repetition test.
    print("\n--- Repetition Test (15 * 23 asked 5 times) ---\n")
    repeat_question = "What is 15 * 23?"
    repeat_results = [run_single_test(repeat_question, expected=345, run_agent=run_agent) for _ in range(5)]
    for i, result in enumerate(repeat_results, 1):
        status = "PASS" if result["passed"] else "FAIL"
        print(f"  Run {i}: {status} -> {result['answer']!r}")
    results.extend(repeat_results)

    # Summary.
    math_results = [r for r in results if r["question"] in [q for q, _ in MATH_QUESTIONS]]
    read_results = [r for r in results if r["question"] in [q for q, _ in READ_FILE_QUESTIONS]]
    multi_results = [r for r in results if r["question"] in [q for q, _ in MULTI_STEP_QUESTIONS]]
    weather_results = [r for r in results if r["question"] in [q for q, _ in WEATHER_QUESTIONS]]
    weather_math_results = [r for r in results if r["question"] in [q for q, _ in WEATHER_MATH_QUESTIONS]]
    web_search_results = [r for r in results if r["question"] in [q for q, _ in WEB_SEARCH_QUESTIONS]]
    memory_results = [r for r in results if r["question"] in [q for q, _ in MEMORY_QUESTIONS]]
    date_results = [r for r in results if r["question"] in [q for q, _ in DATE_QUESTIONS]]
    non_math_results = [r for r in results if r["question"] in [q for q, _ in NON_MATH_QUESTIONS]]
    repeat_summary = repeat_results

    passed_math = sum(r["passed"] for r in math_results)
    passed_read = sum(r["passed"] for r in read_results)
    passed_multi = sum(r["passed"] for r in multi_results)
    passed_weather = sum(r["passed"] for r in weather_results)
    passed_weather_math = sum(r["passed"] for r in weather_math_results)
    passed_web_search = sum(r["passed"] for r in web_search_results)
    passed_memory = sum(r["passed"] for r in memory_results)
    passed_date = sum(r["passed"] for r in date_results)
    passed_non_math = sum(r["passed"] for r in non_math_results)
    passed_repeat = sum(r["passed"] for r in repeat_summary)

    total_passed = sum(r["passed"] for r in results)
    total = len(results)

    durations = [r["duration"] for r in results]

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Math questions:       {passed_math} / {len(math_results)} passed")
    print(f"Read-file questions:  {passed_read} / {len(read_results)} passed")
    print(f"Multi-step questions: {passed_multi} / {len(multi_results)} passed")
    print(f"Weather questions:    {passed_weather} / {len(weather_results)} passed")
    print(f"Weather + math:       {passed_weather_math} / {len(weather_math_results)} passed")
    print(f"Web search questions: {passed_web_search} / {len(web_search_results)} passed")
    print(f"Memory (RAG):         {passed_memory} / {len(memory_results)} passed")
    print(f"Date questions:       {passed_date} / {len(date_results)} passed")
    print(f"Non-math questions:   {passed_non_math} / {len(non_math_results)} passed")
    print(f"Repetition test:      {passed_repeat} / {len(repeat_summary)} passed")
    print(f"Total:                {total_passed} / {total} passed")
    print(f"Mean response time:   {statistics.mean(durations):.2f}s")
    print(f"Max response time:    {max(durations):.2f}s")
    print("=" * 70)

    failed = total - total_passed
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stress test for the multi-tool agent")
    parser.add_argument(
        "--langgraph",
        action="store_true",
        help="Run the stress test against the LangGraph implementation instead of the hand-rolled one",
    )
    parser.add_argument(
        "--multi-agent",
        dest="multi_agent",
        action="store_true",
        help="Run the stress test against the Actor+Critic multi-agent LangGraph team",
    )
    parser.add_argument(
        "--auto-approve",
        dest="auto_approve",
        action="store_true",
        help="Auto-approve external tools in the LangGraph agent (required for non-interactive runs)",
    )
    args = parser.parse_args()

    if args.multi_agent:
        if not args.auto_approve:
            print("Warning: Multi-agent team will pause for approval on external tools.")
            print("Use --auto-approve for non-interactive stress testing.")
        raise SystemExit(
            main(
                run_agent=lambda *a, **kw: multi_agent_run_agent(*a, **kw, auto_approve=args.auto_approve),
                agent_label="Actor+Critic Multi-Agent",
            )
        )
    if args.langgraph:
        if not args.auto_approve:
            print("Warning: LangGraph agent will pause for approval on external tools.")
            print("Use --auto-approve for non-interactive stress testing.")
        raise SystemExit(
            main(
                run_agent=lambda *a, **kw: langgraph_run_agent(*a, **kw, auto_approve=args.auto_approve),
                agent_label="LangGraph",
            )
        )
    raise SystemExit(main(run_agent=hand_rolled_run_agent, agent_label="Hand-Rolled"))
