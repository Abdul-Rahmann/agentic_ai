# First Agent: Simple Math Assistant

This is the smallest possible agent that demonstrates the full loop:

1. **Perceive** the user's math question.
2. **Plan/Reason** whether a calculation is needed.
3. **Select a tool** (`calculate`).
4. **Execute** the tool (evaluate the expression safely).
5. **Observe** the result.
6. **Return** the final answer.

It uses a **two-phase design** to make a small local model reliable:

- **Phase 1 — Plan**: the model decides if a calculation is needed and outputs the JSON tool call.
- **Phase 2 — Answer**: the model receives the tool result and outputs the final answer only.
- **Guard**: if the final answer does not contain the computed result, the agent returns the raw result.

This separation removes the ambiguity that caused the single-prompt version to repeat tool calls and hallucinate. See `development-log.md` for the full iteration history.

## Requirements

- Python 3.10+
- Ollama running locally with `llama3.1:latest` pulled

## Install dependencies

```bash
pip install ollama
```

(Already installed in this environment.)

## Run the agent

```bash
python first_agent/math_agent.py
```

Then try questions like:

- `What is 15 * 23?`
- `What is sqrt(144) + 5?`
- `What is 2 ** 10?`

## What to watch for

Each run prints the agent's internal steps. You should see:

1. **Plan phase**: the agent outputs a JSON tool call like `{"tool": "calculate", "input": "15 * 23"}`.
2. **Tool execution**: the tool prints the computed result.
3. **Answer phase**: the agent returns the final answer.

If the final answer does not contain the tool result, the guard returns the raw result instead.

## Common issues

- **Model does not follow JSON format**: Add more few-shot examples to the plan prompt, or switch to a stronger model (e.g., `gpt-4o-mini`) by setting `USE_OPENAI=1` and `AGENT_MODEL=gpt-4o-mini`.
- **Wrong expression**: The LLM may mis-transcribe the math. More examples in the plan prompt help.
- **Final answer is wrong even though tool result is correct**: The guard catches this by returning the raw tool result. If this happens often, tighten the answer prompt or use a stronger model.
- **Safety note**: `calculate` uses a restricted `eval`. This is safe for arithmetic but should never be exposed to untrusted users or arbitrary code.

## Run with OpenAI (optional)

If you want stronger and more reliable instruction following:

```bash
export USE_OPENAI=1
export AGENT_MODEL=gpt-4o-mini
python first_agent/math_agent.py
```

Make sure `OPENAI_API_KEY` is set in your environment.

## Next steps after this works

1. Add a second tool (e.g., `get_current_date` or `read_file`).
2. Add a reflection step: after getting the tool result, ask the agent to verify the answer.
3. Add memory: keep the conversation history across multiple questions.
4. Add observability: log every step, tool call, and result to a file or tracing dashboard.
