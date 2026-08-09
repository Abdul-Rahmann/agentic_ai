# First Agent: Simple Math Assistant

This is the smallest possible agent that demonstrates the full loop:

1. **Perceive** the user's math question.
2. **Plan/Reason** whether a calculation is needed.
3. **Select a tool** (`calculate`).
4. **Execute** the tool (evaluate the expression safely).
5. **Observe** the result.
6. **Return** the final answer.

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

1. The agent output a JSON tool call like `{"tool": "calculate", "input": "15 * 23"}`.
2. The tool prints the computed result.
3. The agent returns the final answer.

## Common issues

- **Model does not follow JSON format**: If the agent answers without a tool call, the prompt or the model may need adjustment. Try a stronger model (e.g., OpenAI `gpt-4o-mini`) or add more few-shot examples to the system prompt.
- **Wrong expression**: The LLM may mis-transcribe the math. That is normal for small local models. Larger models or more examples help.
- **Safety note**: `calculate` uses a restricted `eval`. This is safe for arithmetic but should never be exposed to untrusted users or arbitrary code.

## Next steps after this works

1. Add a second tool (e.g., `get_current_date` or `read_file`).
2. Add a reflection step: after getting the tool result, ask the agent to verify the answer.
3. Add memory: keep the conversation history across multiple questions.
4. Add observability: log every step, tool call, and result to a file or tracing dashboard.
