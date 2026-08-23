# First Agent: Development Log

This document records the design, failures, and fixes for the first agentic AI experiment: a simple math assistant.

---

## Overview

**Goal**: Build the smallest possible agent that demonstrates the full agent loop:

```
Perceive → Plan → Select Tool → Execute → Observe → Reflect/Answer
```

**Agent**: `first_agent/math_agent.py`  
**Tool**: `calculate(expression)` — evaluates a mathematical expression in a restricted Python environment.  
**Model**: `llama3.1:latest` via Ollama (local, free).

---

## Version 1: Single-Loop Agent

### Design

The agent maintained a single conversation with the model. The system prompt instructed the LLM to:

1. Emit a JSON tool call if a calculation was needed: `{"tool": "calculate", "input": "..."}`
2. Emit a plain-text final answer after receiving the tool result.
3. Emit a plain-text answer if no calculation was needed.

The loop parsed the response, executed the tool if needed, appended the result to the conversation history, and iterated up to a max step limit.

### Code shape

```python
messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": question},
]

for step in range(MAX_STEPS):
    response = ollama.chat(model=MODEL, messages=messages)
    tool_call = extract_json(response)

    if tool_call:
        result = calculate(tool_call["input"])
        messages.append({"role": "assistant", "content": response})
        messages.append({"role": "tool", "content": result})
    else:
        return response
```

### Observed failures

```text
Ask a math question: What is 15 * 23
[Step 1] Agent: {"tool": "calculate", "input": "15 * 23"}
[Tool] calculate(15 * 23) = 345
[Step 2] Agent: {"tool": "calculate", "input": "15 * 23"}}
[Tool] calculate(15 * 23) = 345
[Step 3] Agent: 4.35
Final answer: 4.35

Ask a math question: What is 2 ** 10?
[Step 1] Agent: {"tool": "calculate", "input": "2 ** 10"}
[Tool] calculate(2 ** 10) = 1024
[Step 2] Agent: {"tool": "calculate", "input": "5 - 3"}}
[Tool] calculate(5 - 3) = 2
[Step 3] Agent: {"tool": "calculate", "input": "100 / 10"}}
[Tool] calculate(100 / 10) = 10.0
[Step 4] Agent: {"tool": "calculate", "input": "(4 * 5) + 6"}}
[Tool] calculate((4 * 5) + 6) = 26
[Step 5] Agent: The final answer is $\boxed{26}$.
Final answer: The final answer is $\boxed{26}$.

Ask a math question: what is 4 * 4
[Step 1] Agent: {"tool": "calculate", "input": "4 * 4"}
[Tool] calculate(4 * 4) = 16
[Step 2] Agent: 10
Final answer: 10

Ask a math question: what is 2 - 2
[Step 1] Agent: {"tool": "calculate", "input": "2 - 2"}
[Tool] calculate(2 - 2) = 0
[Step 2] Agent: 4 * (8 - 3)
Final answer: 4 * (8 - 3)
```

### Root cause analysis

| Symptom | Root Cause |
|---|---|
| Repeats the tool call after the result is already known | The model could not reliably distinguish between “I need to call a tool” and “I have the tool result and must answer now.” Both roles were squeezed into the same prompt and conversation history. |
| Wrong final answer (`4.35`, `10`) | The model was confused by the accumulating conversation history and produced a generic or hallucinated number. |
| Makes up unrelated calculations (`2 ** 10` → `5 - 3` → `100 / 10`) | The model lost track of the original question and started generating plausible-looking but irrelevant math expressions. |
| Returns an expression instead of a value (`4 * (8 - 3)`) | The output format was ambiguous. The model sometimes treated a plain-text response as a tool input instead of a final answer. |
| Malformed JSON (`{"tool": "calculate", "input": "15 * 23"}}` with extra brace) | Small local models have weaker instruction following for exact output formats. |

The core problem was **ambiguity in the LLM’s role across turns**. The same prompt asked the model to be both a tool-caller and a final-answer-generator, and the same conversation history mixed both modes together. Small models are especially sensitive to this ambiguity.

---

## Version 2: Two-Phase Agent with Guard

### Design changes

Split the single loop into two distinct, single-purpose calls:

1. **Phase 1 — Plan**: Given the question, output either a JSON tool call or a direct answer.
2. **Phase 2 — Answer**: Given the question and the tool result, output the final answer only.

Each phase has its own system prompt and its own conversation. The model never has to switch roles mid-conversation.

### New measures employed

| Measure | Purpose |
|---|---|
| **Two-phase prompts** | Remove role ambiguity. The model is told exactly one task per prompt. |
| **Fresh conversation for each phase** | Prevents the accumulating conversation history from confusing the model. |
| **More few-shot examples** | Show exactly what valid tool-call and final-answer outputs look like. |
| **JSON extraction with regex fallback** | Handle malformed JSON from small local models gracefully. |
| **Guard check** | Verify the final answer contains the computed tool result. If not, return the raw tool result as a fallback. |
| **Optional OpenAI support** | Allow switching to a stronger model (`gpt-4o-mini`) via environment variables if local model performance is still unsatisfactory. |

### Code shape

```python
def run_agent(question: str) -> str:
    # Phase 1: Plan
    plan_output = chat([
        {"role": "system", "content": PLAN_SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ])
    tool_call = extract_json(plan_output)

    if not tool_call:
        return plan_output

    result = calculate(tool_call["input"])

    # Phase 2: Answer
    answer_output = chat([
        {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
        {"role": "user", "content": f"Question: {question}\nTool result: {result}"},
    ])

    # Guard
    if str(result) not in answer_output:
        return result

    return answer_output
```

### Results after the fix

```text
Ask a math question: What is 15 * 23
[Plan] Agent: {"tool": "calculate", "input": "15 * 23"}
[Tool] calculate(15 * 23) = 345
[Answer] Agent: 345
Final answer: 345

Ask a math question: What is 4 * 4
[Plan] Agent: {"tool": "calculate", "input": "4 * 4"}
[Tool] calculate(4 * 4) = 16
[Answer] Agent: 16
Final answer: 16

Ask a math question: What is 2 ** 10?
[Plan] Agent: {"tool": "calculate", "input": "2 ** 10"}
[Tool] calculate(2 ** 10) = 1024
[Answer] Agent: 1024
Final answer: 1024

Ask a math question: What is 2 - 2
[Plan] Agent: {"tool": "calculate", "input": "2 - 2"}
[Tool] calculate(2 - 2) = 0
[Answer] Agent: 0
Final answer: 0
```

All previously failing questions now return the correct answer.

---

## Side-by-side comparison

| Aspect | Version 1 (Single Loop) | Version 2 (Two Phase) |
|---|---|---|
| Number of prompts per question | One shared prompt | Two dedicated prompts |
| Conversation history | Accumulates tool calls and answers | Fresh for each phase |
| Role ambiguity | High: caller + answerer in same prompt | Low: one role per prompt |
| Correctness on test set | ~40% | ~100% |
| JSON format issues | Frequent malformed JSON | Reduced; regex fallback handles edge cases |
| Hallucination of extra calculations | Common | Eliminated |
| Guard against wrong final answer | None | Yes: returns raw tool result if answer does not contain it |
| Model swap support | Ollama only | Ollama + optional OpenAI |

---

## Stress Test Results

A benchmark script (`first_agent/stress_test.py`) was added to measure reliability. The first run used `llama3.1:latest` via Ollama.

### Initial results

```text
Math questions:       21 / 23 passed
Non-math questions:    3 /  3 passed
Repetition test:       5 /  5 passed
Total:                 24 / 26 passed (92%)
Mean response time:    1.49s
Max response time:     7.60s
```

### Failures

| Question | Expected | Got | Root Cause |
|---|---|---|---|
| What is 2 to the power of 10? | 1024 | 8 | Model generated `2 ^ 10`. In Python `^` is bitwise XOR, not exponentiation. `2 XOR 10` = `8`. |
| What is factorial of 5? | 120 | syntax error | Model generated `5!`. Python has no `!` factorial operator. |

### Observations

- The two-phase loop is highly consistent: the repetition test passed 5/5 times.
- The remaining failures are **tool-interface mismatches**, not loop failures. The model produces valid natural-math notation that the Python evaluator does not understand.
- Non-math questions are handled cleanly because the plan prompt allows direct answers.
- The guard correctly caught both cases where the answer phase drifted from the tool result, but it returned the raw (wrong) tool result to the user.

### Fix applied

1. **Expression normalization** in the `calculate` tool:
   - `^` is rewritten to `**` for exponentiation.
   - `n!` is rewritten to `factorial(n)` (exposed from the `math` module).
2. **Better prompt examples** added to the plan prompt for powers and factorials.
3. **Guard improved**: if the tool returns an error, the agent returns a clean error message instead of the raw Python traceback.

### Results after fix

```text
Math questions:       23 / 23 passed
Non-math questions:    3 /  3 passed
Repetition test:       5 /  5 passed
Total:                 26 / 26 passed (100%)
Mean response time:    1.27s
Max response time:     1.71s
```

---

## Experiment 2: Multi-Tool Agent

**Date**: 2026-08-09  
**Goal**: Add a second tool (`read_file`) and evolve the agent from a single-purpose math assistant into a multi-tool agent that can plan sequences of tool calls.

### What was added

- New tool: `read_file(path)` — reads text files within the `first_agent` directory.
- Tool loop: the agent can now call tools repeatedly until it has enough information.
- New sample data file: `first_agent/data/numbers.txt`.
- Updated stress test covering math, file reads, multi-step tasks, and non-math questions.

### Architecture

```
Perceive question
    ↓
Plan: choose a tool or answer directly
    ↓
Execute tool
    ↓
Observe result
    ↓
Re-plan or answer
    ↓
Final answer synthesis
```

### First multi-tool attempt issues

| Symptom | Cause |
|---|---|
| Infinite tool loops for simple questions | The model did not recognize that a single tool result already answered the question. |
| Redundant tool calls (oscillating between read and calculate) | The model re-called tools it had already used. |
| Stray `assistant\n\n` prefix in final answer | Small local model echoed the role label in the answer phase. |

### Fixes applied

1. **Explicit examples in plan prompt** showing when to answer directly after a tool result.
2. **Explicit rule**: do not repeat a tool call already in history.
3. **History-level guard**: the agent is blocked from calling the same tool with the same input twice, not just the immediately previous step.
4. **Answer cleanup**: strips common prefixes like `assistant:` and `Answer:` from the final output.
5. **Safety restriction**: `read_file` only reads files inside the `first_agent` directory.

### Stress test results

```text
Math questions:       23 / 23 passed
Read-file questions:   1 /  1 passed
Multi-step questions:  2 /  2 passed
Non-math questions:    3 /  3 passed
Repetition test:       5 /  5 passed
Total:                29 / 29 passed (100%)
Mean response time:    2.30s
Max response time:     6.80s
```

The `data/numbers.txt` file now contains: `12, 15, 23, 8, 2, 3, 5`.
- Sum: `68`
- Product: `993600`

### Key observations

- Multi-step planning (`read_file` → `calculate` → answer) works with a small local model when the loop and guards are explicit.
- The local model is slow but consistent: 29/29 passed.
- Guards are critical for multi-turn loops. Without them, the agent repeats tools indefinitely.
- The final answer phase is fragile with small models and needs output cleanup.
- Deterministic tools do the actual work; the LLM only decides which tool to use and when.

---

## Experiment 3: Structured Observability / Tracing

**Date**: 2026-08-13  
**Goal**: Add structured tracing so every run produces a machine-readable record of decisions, tool calls, results, timings, and guards.

### What was built

- New module: `first_agent/tracer.py` with:
  - `Trace` class that records question, model, provider, steps, tool calls, results, guard triggers, timings, and final answer.
  - `NullTrace` class used when tracing is disabled.
  - `write()` method that persists each trace as a JSON file in `first_agent/traces/`.
- Updated `math_agent.py` to instrument every phase of `run_agent` and write a trace by default.
- Updated `stress_test.py` to disable tracing (`trace=False`) so benchmark runs do not flood the traces directory.
- Added trace documentation to `README.md`.

### Why it matters

- Reproducibility: every run is recorded exactly, including the raw LLM outputs.
- Debugging: when a failure happens, you can replay the exact sequence of decisions.
- Measurement: traces contain per-step and total latency.
- Safety: traces provide an audit log of what tools were called and what data was accessed.

### Example trace output

```json
{
  "trace_id": "c7e21dac",
  "timestamp": "2026-08-13T13:10:54.217023+00:00",
  "question": "What is the sum of the numbers in data/numbers.txt?",
  "model": "llama3.1:latest",
  "provider": "Ollama",
  "steps": [
    {
      "step": 1,
      "phase": "plan",
      "llm_output": "{\"tool\": \"read_file\", \"input\": \"data/numbers.txt\"}",
      "tool_name": "read_file",
      "tool_input": "data/numbers.txt",
      "tool_result": "12\n15\n23\n8\n2\n3\n5\n",
      "duration_ms": 1005,
      "guard_triggered": false,
      "guard_reason": null
    },
    {
      "step": 2,
      "phase": "plan",
      "llm_output": "{\"tool\": \"calculate\", \"input\": \"12 + 15 + 23 + 8 + 2 + 3 + 5\"}",
      "tool_name": "calculate",
      "tool_input": "12 + 15 + 23 + 8 + 2 + 3 + 5",
      "tool_result": "68",
      "duration_ms": 1882,
      "guard_triggered": false,
      "guard_reason": null
    },
    {
      "step": 3,
      "phase": "plan",
      "llm_output": "68",
      "duration_ms": 643,
      "guard_triggered": true,
      "guard_reason": "model answered directly after tool history; moving to answer phase"
    },
    {
      "step": 4,
      "phase": "answer",
      "llm_output": "68",
      "duration_ms": 645,
      "guard_triggered": false,
      "guard_reason": null
    }
  ],
  "final_answer": "68",
  "total_duration_ms": 4175,
  "error": null
}
```

### Key observations

- Tracing adds minimal overhead but adds huge debugging power.
- It forces the developer to think about what is worth recording at each step.
- Guard events should be explicit in traces so you can see why the loop stopped.
- Stress tests should be able to disable tracing to avoid producing thousands of trace files.

---

## Experiment 4: Reflection / Critic Phase

**Date**: 2026-08-13  
**Goal**: Add a separate critic prompt that verifies the final answer against the question and tool history before returning it to the user.

### What was built

- New reflection prompt in `first_agent/math_agent.py` that acts as a critic:
  - Responds with `VERIFIED: <answer>` if the answer is correct and supported by tool history.
  - Responds with `INCORRECT: <reason>` if the answer is wrong or unsupported.
- New helper functions `_build_reflection_messages` and `_parse_reflection`.
- `run_agent` now has a `reflect=True` parameter.
  - After the final answer phase, the critic prompt is called.
  - If verified, the answer is returned as-is.
  - If incorrect, the answer is returned with an `[Unverified: <reason>]` warning.
- Reflection step is recorded in the JSON trace.
- Stress test runs with reflection enabled (`reflect=True` is the default).

### Why it matters

Reflection is a proven optimization technique for language agents. It adds a second verification layer that can catch errors missed by the answer-generation prompt, especially when the final synthesis drifts from the tool results.

### Example reflection output

```text
[Answer] Agent: 345
[Reflection] Critic: VERIFIED: 345
Final answer: 345
```

If the critic had flagged an error, the output would look like:

```text
[Answer] Agent: 340
[Reflection] Critic: INCORRECT: the tool result is 345, not 340
Final answer: [Unverified: the tool result is 345, not 340] 340
```

### Stress test results with reflection

```text
Math questions:       23 / 23 passed
Read-file questions:   1 /  1 passed
Multi-step questions:  2 /  2 passed
Non-math questions:    3 /  3 passed
Repetition test:       5 /  5 passed
Total:                29 / 29 passed (100%)
Mean response time:    3.12s
Max response time:    7.55s
```

Reflection adds roughly one extra LLM call per question, which increases mean latency from ~2.3s to ~3.1s. All answers were verified by the critic in this run.

### Key observations

- A small local model can act as a critic when the prompt gives clear examples and a strict output format.
- The reflection step itself becomes a recorded phase in the trace, making it auditable.
- Reflection is currently conservative: if the critic flags an answer, we return it with a warning rather than automatically retrying. This avoids compounding model errors.
- The next evolution would be to add a retry loop when reflection detects an error.

---

## Experiment 5: Reflection Retry Loop

**Date**: 2026-08-16  
**Goal**: Evolve reflection from a warning-only verifier into a self-correcting retry loop.

### What was built

- Added `max_retries` parameter to `run_agent` (default: 2).
- Wrapped the plan/answer/reflection sequence in an outer attempt loop.
- When reflection returns `INCORRECT: <reason>`, the reason is appended to a `reflection_feedback` list.
- `_build_plan_messages` and `_build_answer_messages` now include the reflection feedback so subsequent attempts can correct the mistake.
- Tool results are cached across attempts via `tool_history`; tools are not re-executed on retry.
- If the answer is still unverified after `max_retries`, it is returned with an `[Unverified after N retries: ...]` warning.
- Traces record an `attempt` number for each step so the retry sequence is visible in the trace file.
- Added `first_agent/test_retry.py`, a unit-style test that mocks the LLM chat function to force a wrong answer on the first attempt and verifies the retry corrects it.

### Why it matters

Reflection without retry only tells you that something is wrong. Retry turns the critic's feedback into action, which is a key self-improvement pattern in agentic AI. It also demonstrates that the loop structure and tool history must be designed to support re-planning.

### Example retry test output

```text
[Attempt 1]
[Plan 1.1] Agent: {"tool": "calculate", "input": "15 * 23"}
[Tool 1.1] calculate(15 * 23) = 345
[Plan 1.2] Agent: 345
[Answer 1] Agent: 340
[Reflection 1] Critic: INCORRECT: the calculator returned 345, not 340
[Retry] Reflection flagged: the calculator returned 345, not 340

[Attempt 2]
[Plan 2.1] Agent: {"tool": "calculate", "input": "15 * 23"}
[Guard] Repeated tool call detected. Moving to answer phase.
[Answer 2] Agent: 345
[Reflection 2] Critic: VERIFIED: 345
[Retry] Answer verified on attempt 2.

Final result: 345
```

### Stress test results with reflection + retry

```text
Math questions:       23 / 23 passed
Read-file questions:   1 /  1 passed
Multi-step questions:  2 /  2 passed
Non-math questions:    3 /  3 passed
Repetition test:       5 /  5 passed
Total:                29 / 29 passed (100%)
Mean response time:    3.08s
Max response time:    7.51s
```

Because the local model answers correctly on the first attempt for this test suite, no retries were triggered. The retry mechanism is exercised by `test_retry.py` with a mocked LLM.

### Key observations

- Caching tool results across attempts is essential for efficiency and determinism. Re-executing tools on retry could produce different results or side effects.
- Including the critic's feedback in the plan and answer prompts gives the model a concrete correction signal.
- The retry loop re-uses the existing guard against repeated tool calls, which naturally pushes the model toward re-synthesis rather than re-execution.
- Retry only makes sense when the underlying tool results are trustworthy. If the tool itself is wrong, retry will not help.
- Traces now include an `attempt` field, making it easy to see how many tries a question required.

---

## Experiment 6: Adding a Third Tool (Weather via Open-Meteo)

**Date**: 2026-08-16  
**Goal**: Add a third tool that calls a live external API, expanding the agent beyond local/static tools and testing multi-tool planning with real-world data.

### What was added

- New tool: `get_weather(city)` in `first_agent/math_agent.py`.
  - Uses Open-Meteo geocoding and forecast APIs (no API key, stdlib `urllib` only).
  - Converts WMO weather codes into short human-readable descriptions.
  - Returns errors cleanly when a city is not found or the network fails.
- Updated `TOOLS` dict and plan prompt with a weather example.
- Expanded `first_agent/stress_test.py`:
  - `WEATHER_QUESTIONS`: verify that a weather query contains expected substrings (e.g., `"Paris"`, `"°C"`).
  - `WEATHER_MATH_QUESTIONS`: verify that combining a live weather value with math returns a numeric answer.
  - Added `expect_number` flag to `run_single_test` for answers where only the presence of a number matters.

### First external-tool issues

| Symptom | Cause |
|---|---|
| Tests could not assert exact weather values | Real-world temperature and conditions change between runs. |
| Risk of hanging on slow/flaky network | Open-Meteo is generally fast, but any external API call needs a timeout. |
| City name could be ambiguous | Geocoding returns the best match; the response includes the resolved location for transparency. |

### Fixes applied

1. **Fuzzy assertions for weather**: check for expected substrings and numeric presence rather than exact values.
2. **10-second timeout** on both geocoding and forecast HTTP requests.
3. **Resolved location in the tool result** so the final answer is transparent about which city was used.
4. **Plan prompt example** for weather so the local model knows how to format `get_weather` calls.

### Stress test results with three tools

```text
Math questions:         23 / 23 passed
Read-file questions:   1 /  1 passed
Multi-step questions:   2 /  2 passed
Weather questions:     1 /  1 passed
Weather + math:        1 /  1 passed
Non-math questions:    3 /  3 passed
Repetition test:       5 /  5 passed
Total:                31 / 31 passed (100%)
Mean response time:    3.38s
Max response time:     7.58s
```

The weather question adds one external API round-trip (geocoding + forecast), which increases mean latency slightly compared with purely local tools. The local model still plans the correct sequence: call `get_weather`, then either answer directly or feed the result into `calculate` for the weather+math case.

### Key observations

- Adding an external API tool does not require changing the core loop; the existing plan/answer/reflection structure handles it.
- Live data requires tests that assert structure, not exact values.
- Clean error handling in the tool prevents a single API failure from crashing the whole agent.
- The weather+math question confirms the agent can chain heterogeneous tools (`get_weather` → `calculate`) in a single plan.

---

## Experiment 7: Fourth Tool — Web Search via Wikipedia

**Date**: 2026-08-21  
**Goal**: Add a fourth tool that performs general knowledge retrieval, extending the agent from local files, weather, and math into web-scale search.

### What was added

- New tool: `web_search(query)` in `first_agent/math_agent.py`.
  - Uses the Wikipedia search and extracts APIs (no API key, stdlib `urllib` only).
  - Searches for the most relevant article, fetches a short intro extract, and returns a compact text summary.
  - Includes an in-process cache and a polite 0.3s delay to avoid hammering the API.
  - Handles HTTP errors and empty results cleanly.
- Updated `TOOLS` dict and plan prompt with a web-search example.
- Added `WEB_SEARCH_QUESTIONS` to `first_agent/stress_test.py`, using substring assertions because Wikipedia content can change.

### First issues with the new tool

| Symptom | Cause |
|---|---|
| `Who is the CEO of OpenAI?` answer was correct but test failed | Assertion expected both `"Sam Altman"` and `"OpenAI"` in the answer; the model only emitted `"Sam Altman"`. |
| Adding the fourth tool made the plan prompt longer | The previously passing product question (`What is the product of the numbers in data/numbers.txt?`) started failing because the model tried to compute the product in its head and answered `720`. |
| Wikipedia API rate-limiting during manual exploration | Rapid repeated calls from exploratory scripts triggered HTTP 429 errors. |

### Fixes applied

1. **Relaxed web-search test assertion**: check for `"Sam Altman"` only, which is the factual core the question asks for.
2. **Added a product example to the plan prompt**: reinforced that arithmetic over file contents must use `calculate`, not mental math.
3. **Cache and delay in `web_search`**: avoid repeated identical requests and reduce the risk of rate limits.
4. **Clear error messages** for HTTP failures so the agent returns something useful instead of crashing.

### Stress test results with four tools

```text
Math questions:       23 / 23 passed
Read-file questions:   1 /  1 passed
Multi-step questions:  2 /  2 passed
Weather questions:     1 /  1 passed
Weather + math:        1 /  1 passed
Web search questions:  1 /  1 passed
Non-math questions:    3 /  3 passed
Repetition test:       5 /  5 passed
Total:                32 / 32 passed (100%)
Mean response time:    4.21s
Max response time:     8.19s
```

The web-search test adds a Wikipedia API round-trip, so it is one of the slower cases. The overall suite still passes completely.

### Key observations

- Each new tool increases the plan prompt size, which can make existing multi-step examples less salient. Re-adding or reinforcing examples for fragile cases is a cheap fix.
- Free APIs (Wikipedia, Open-Meteo) are great for learning, but they need rate-limit awareness: cache, delay, timeouts, and polite user-agent strings.
- Web search results change, so tests should assert stable structure (names, units, numeric presence) rather than exact text.
- The same loop architecture scales to four distinct tool types: deterministic math, local file I/O, live weather, and knowledge retrieval.

---

## Experiment 7 Update: Hybrid Web Search (DuckDuckGo + Wikipedia)

**Date**: 2026-08-22  
**Goal**: Fix the unreliable Wikipedia-only `web_search` by switching to DuckDuckGo Instant Answer first and falling back to Wikipedia, and correct the reflection critic so it stops hallucinating training cutoffs for live tool results.

### What was changed

- Replaced the Wikipedia-only `web_search(query)` with a **hybrid search**:
  - First calls the DuckDuckGo Instant Answer API (`https://api.duckduckgo.com/?q=...&format=json`).
  - Tries multiple query variants (strips prefixes like `current`, `who is`, `what is`, etc.) when the first attempt is empty.
  - Falls back to Wikipedia if DuckDuckGo has no result.
  - Keeps the in-process cache and returns clean errors.
- Updated the reflection prompt with an explicit rule: **trust live external tool results (weather, web search) over the model's own training knowledge**.

### Why the change was needed

- Wikipedia articles can be stale or omit current details.
- The reflection critic repeatedly rejected correct current facts (e.g., "John Mahama is the president of Ghana") with invented "training data cutoff" reasoning.
- DuckDuckGo Instant Answer alone was too narrow: it handled "President of Ghana" but failed on "CEO of OpenAI" and "Who wrote Hamlet?". Wikipedia covered those gaps.

### Stress test results after the update

```text
Math questions:       23 / 23 passed
Read-file questions:   1 /  1 passed
Multi-step questions:  2 /  2 passed
Weather questions:    1 /  1 passed
Weather + math:       1 /  1 passed
Web search questions:  1 /  1 passed
Non-math questions:    3 /  3 passed
Repetition test:       5 /  5 passed
Total:                32 / 32 passed (100%)
Mean response time:    3.91s
Max response time:     7.10s
```

### Remaining limitation

DuckDuckGo Instant Answer is not a real-time search engine. Queries like *"What date is today?"* either return no result or return stale/cached data. A dedicated `get_date()` tool or a paid live-search API (Tavily, Serper, etc.) is still needed for date/time and breaking-news questions.

---

## Experiment 8: Fifth Tool — Current Date

**Date**: 2026-08-23  
**Goal**: Add a deterministic tool for today's date so the agent can answer real-time date questions reliably, without depending on search.

### What was added

- New tool: `get_current_date()` in `first_agent/math_agent.py`.
  - Returns the system date in `YYYY-MM-DD (DayName)` format (e.g., `2026-08-23 (Sunday)`).
  - No network call, no API key, fully deterministic.
- Updated `TOOLS`, plan prompt examples, and interactive examples.
- Added `DATE_QUESTIONS` to `first_agent/stress_test.py` with runtime-generated expected substrings.

### Why a dedicated date tool matters

Search engines and instant-answer APIs are not built for "today's date":
- DuckDuckGo returns an empty answer or stale cached calendars.
- Wikipedia has no article for "today".
- A small local model will hallucinate a date from its training data.

A simple system-clock tool solves the problem deterministically and demonstrates a general principle: **not every "current" question should be answered by search**. Some questions are best answered by a dedicated, cheap, deterministic tool.

### Stress test results with five tools

```text
Math questions:       23 / 23 passed
Read-file questions:   1 /  1 passed
Multi-step questions:  2 /  2 passed
Weather questions:    1 /  1 passed
Weather + math:       1 /  1 passed
Web search questions:  1 /  1 passed
Date questions:        2 /  2 passed
Non-math questions:    3 /  3 passed
Repetition test:       5 /  5 passed
Total:                34 / 34 passed (100%)
Mean response time:    4.34s
Max response time:    11.75s
```

### Key observations

- A fifth tool did not break any existing tests.
- The model correctly chose `get_current_date()` for date questions instead of `web_search` or `read_file`.
- Deterministic system tools are the cheapest and most reliable way to answer time-sensitive questions.
- The stress test expected substrings are generated at runtime because the expected date changes every day.

---

## General Lessons Learned

1. **The loop is more important than the model size.**  
   A smaller model with a well-structured loop outperforms a larger model with an ambiguous loop.

2. **Separate roles into separate prompts.**  
   When one model has to do multiple jobs, split the jobs into separate calls with fresh context. This is cheaper and more reliable than trying to squeeze everything into one conversation.

3. **Small local models need strict scaffolding.**  
   Few-shot examples, exact output formats, and guard checks are essential. Do not assume the model will infer the right behavior from a vague instruction.

4. **Always verify the final output.**  
   A simple guard (e.g., “does the answer contain the computed result?”) catches a large class of failures for free.

5. **Observability is not optional.**  
   Printing every step made the failure pattern obvious. Without it, debugging would have been guesswork.

6. **Tool results should be deterministic.**  
   The calculator is a pure function. The LLM does not compute the answer; it decides to invoke a deterministic tool. This separation is the heart of agentic AI.

---

## Next Steps

- [x] Add a second tool (`read_file`) and practice multi-tool selection.
- [x] Add a reflection phase where a separate prompt verifies the answer before it is returned.
- [x] Add structured logging so every run produces a JSON trace file for post-hoc analysis.
- [x] Add an explicit retry loop when reflection flags an answer as incorrect.
- [x] Add a third tool (e.g., web search or code execution sandbox).
- [x] Add a fourth tool or capability (e.g., web search, sandboxed code execution, or persistent memory).
- [ ] Expand the benchmark script with more edge cases and adversarial prompts.
- [ ] Build a trace analyzer script that reports pass rate, latency, and common failure modes across many runs.
