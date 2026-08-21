# First Agent: Multi-Tool Assistant

This agent started as a simple math assistant and evolved into a multi-tool agent that can:

1. Evaluate math expressions with `calculate(expression)`.
2. Read text files with `read_file(path)`.
3. Combine both tools to answer multi-step questions like *“What is the sum of the numbers in `data/numbers.txt`?”*

It demonstrates the full agent loop with **tool selection, execution, and multi-turn planning**.

## Architecture

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
    ↓
Reflect / critic verifies answer
    ↓
(If incorrect: retry with feedback)
    ↓
Return final answer
```

The agent uses a **tool loop** followed by **answer synthesis**, **reflection**, and **retry**:

- **Tool loop**: repeatedly plans and executes tools until the model decides it has enough information.
- **Final answer phase**: synthesizes the gathered tool results into a concise answer.
- **Reflection phase**: a critic prompt verifies the answer against the tool history and question.
- **Retry loop**: if the critic flags the answer, the agent re-runs the plan/answer phase with the feedback, up to a configurable number of retries.
- **Guards**: prevent infinite loops by blocking repeated tool calls and by returning clean errors when a tool fails.

## Files

- `math_agent.py` — the agent implementation
- `tracer.py` — structured trace recorder
- `data/numbers.txt` — sample data file for read-file and multi-step tests
- `stress_test.py` — benchmark suite covering math, file reads, multi-step tasks, and non-math questions
- `test_retry.py` — unit-style test that exercises the reflection retry loop with a mocked LLM
- `traces/` — directory where JSON trace files are written automatically
- `development-log.md` — detailed design/evolution history

## Requirements

- Python 3.10+
- Ollama running locally with `llama3.1:latest` pulled

## Install dependencies

```bash
pip install ollama
```

(Already installed in this environment.)

## Run the agent interactively

```bash
python first_agent/math_agent.py
```

Try questions like:

- `What is 15 * 23?`
- `What is in data/numbers.txt?`
- `What is the sum of the numbers in data/numbers.txt?` (answer: `68`)
- `What is the product of the numbers in data/numbers.txt?` (answer: `993600`)
- `What is the capital of France?`

## Run the stress test

```bash
python first_agent/stress_test.py
```

The current suite covers 29 cases and passes all of them with `llama3.1:latest`.

## Run with OpenAI (optional)

For stronger instruction following and faster responses:

```bash
export USE_OPENAI=1
export AGENT_MODEL=gpt-4o-mini
python first_agent/math_agent.py
```

Make sure `OPENAI_API_KEY` is set in your environment.

## Tool details

### `calculate(expression)`

Evaluates a mathematical expression in a restricted Python environment. Supports:

- Basic arithmetic: `+`, `-`, `*`, `/`, `**`, `^` (normalized to `**`)
- Math functions from the `math` module: `sqrt`, `sin`, `log`, `pow`, etc.
- Factorial: `5!` (normalized to `factorial(5)`)

### `read_file(path)`

Reads a text file relative to the `first_agent` directory. Paths outside this directory are blocked for safety.

## Guards and reliability patterns

- **No repeated tool calls**: the agent is not allowed to call the same tool with the same input twice. This prevents infinite loops.
- **Clean error messages**: if a tool fails, the user sees a readable error instead of a raw Python traceback.
- **Answer cleanup**: removes stray prefixes like `assistant:` or `Answer:` that small local models sometimes emit.
- **Expression normalization**: common math notation (`^`, `!`) is rewritten to valid Python before evaluation.
- **Reflection / critic**: before returning the final answer, a separate prompt verifies it against the tool history. If it flags an issue, the answer is returned with an `[Unverified: ...]` warning.

## Reflection and retry

After the final answer is generated, the agent runs a separate **critic** prompt that verifies the answer against the question and the tool history. This is a reflection / self-criticism step.

- If the critic responds with `VERIFIED: <answer>`, the answer is returned as-is.
- If the critic responds with `INCORRECT: <reason>`, the agent enters a retry loop.
- During retry, the critic's feedback is added to the plan and answer prompts so the next attempt can correct the mistake.
- The retry loop runs up to `max_retries` times (default: 2). If the answer is still unverified after all retries, it is returned with an `[Unverified after N retries: ...]` warning.
- Reflection can be disabled by passing `reflect=False` to `run_agent`.

This pattern catches cases where the final answer synthesis drifted from the tool results and attempts to self-correct.

### Example retry output

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

Run `python first_agent/test_retry.py` to execute this test.

## Observability

Every interactive run automatically writes a JSON trace to `first_agent/traces/`.
The trace records:

- `trace_id` and `timestamp`
- The user question
- The model and provider used
- Every planning step, tool call, tool result, and timing
- Reflection verdicts and guard events
- The final answer and total duration

Example trace file (`first_agent/traces/2026-...__c7e21dac.json`):

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
      "phase": "reflection",
      "llm_output": "VERIFIED: 68",
      "duration_ms": 1028,
      "guard_triggered": false,
      "guard_reason": null
    }
  ],
  "final_answer": "68",
  "total_duration_ms": 4175,
  "error": null
}
```

Traces are disabled during stress testing to keep the benchmark clean.

## Lessons learned

1. Small local models need very explicit prompts and few-shot examples to handle multi-turn tool loops.
2. Splitting planning, answering, and reflection into separate prompts helps the model follow instructions.
3. Guards are essential for production-like reliability: they prevent infinite loops, catch tool failures, and block repeated calls.
4. A deterministic tool (calculator, file reader) should do the actual work; the LLM decides which tool to use and when.
5. Reflection adds a second layer of verification after the final answer synthesis.
6. A retry loop that feeds the critic's reason back into the prompt enables self-correction, but only if the underlying tool results are reliable.
7. Tracing turns opaque failures into replayable records.

## Next steps

1. Add a third tool, such as a web search or a Python code execution sandbox.
2. Evaluate the agent on longer, more ambiguous multi-step tasks.
3. Build a trace analyzer script that reports pass rate, latency, and common failure modes across many runs.
4. Experiment with reflection prompting the agent to choose a *different* tool on retry, not just re-synthesize.
