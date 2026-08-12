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

- Add a second tool (e.g., `read_file` or `get_current_date`) and practice multi-tool selection.
- Add a reflection phase where a separate prompt verifies the answer before it is returned.
- Add structured logging so every run produces a JSON trace file for post-hoc analysis.
- Expand the benchmark script with more edge cases and adversarial prompts.
