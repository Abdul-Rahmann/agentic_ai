"""
Multi-tool agent: a math assistant that can also read files.

This evolves the two-phase design into a general tool loop:

  1. Plan: decide whether to use a tool.
  2. Execute: run the selected tool and record the result.
  3. Repeat: plan again with the new information.
  4. Answer: once no more tools are needed, produce the final answer.

Tools:
  - calculate(expression): evaluates a mathematical expression.
  - read_file(path): reads a text file within the project directory.

Runs locally with Ollama (llama3.1). Set USE_OPENAI=1 to use OpenAI instead.
"""

import json
import math
import os
import re
import time

import ollama

from tracer import NullTrace, Trace, current_ms

MODEL = os.getenv("AGENT_MODEL", "llama3.1:latest")
USE_OPENAI = os.getenv("USE_OPENAI", "0") == "1"
MAX_TOOL_STEPS = 5

# Root directory for file reads. Absolute paths outside this are blocked.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


# -----------------------------------------------------------------------------
# Tools
# -----------------------------------------------------------------------------


def calculate(expression: str) -> str:
    """Evaluate a math expression in a restricted environment."""
    expression = _normalize_expression(expression)
    safe_globals = {"__builtins__": {}}
    safe_locals = {
        name: getattr(math, name)
        for name in dir(math)
        if not name.startswith("_")
    }
    safe_locals.update({
        "abs": abs,
        "round": round,
        "max": max,
        "min": min,
        "sum": sum,
        "pow": pow,
    })
    try:
        result = eval(expression, safe_globals, safe_locals)
        return str(result)
    except Exception as e:
        return f"Error: {e}"


def _normalize_expression(expression: str) -> str:
    """Rewrite common math notation into valid Python syntax."""
    # n! -> factorial(n)
    expression = re.sub(r"(\d+)!", r"factorial(\1)", expression)
    # ^ -> ** for exponentiation (common math notation, not Python XOR)
    expression = expression.replace("^", "**")
    return expression


def read_file(path: str) -> str:
    """Read a text file within the project directory."""
    target = os.path.abspath(os.path.join(PROJECT_ROOT, path))
    if not target.startswith(PROJECT_ROOT):
        return "Error: path is outside the allowed project directory"
    if not os.path.exists(target):
        return f"Error: file not found: {path}"
    if not os.path.isfile(target):
        return f"Error: not a file: {path}"
    try:
        with open(target, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Error: {e}"


TOOLS = {
    "calculate": calculate,
    "read_file": read_file,
}


# -----------------------------------------------------------------------------
# LLM helpers
# -----------------------------------------------------------------------------


def extract_json(text: str):
    """Try to find and parse a JSON object in the LLM response."""
    match = re.search(r"\{.*?\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            return None
    return None


def chat(messages: list, temperature: float = 0.0) -> str:
    """Call the configured LLM and return the assistant's content."""
    if USE_OPENAI:
        import openai
        response = openai.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=temperature,
        )
        return response.choices[0].message.content.strip()

    response = ollama.chat(model=MODEL, messages=messages, options={"temperature": temperature})
    return response["message"]["content"].strip()


# -----------------------------------------------------------------------------
# Prompts
# -----------------------------------------------------------------------------


_PLAN_SYSTEM_PROMPT = """You are a helpful assistant. You have access to these tools:

- calculate(expression): evaluates a mathematical expression and returns the result.
- read_file(path): reads the contents of a text file within the project directory.

Rules:
1. If you need to use a tool, respond with ONLY a JSON object in this exact format:
   {"tool": "calculate", "input": "<expression>"}
   or
   {"tool": "read_file", "input": "<path>"}
2. If you already have enough information to answer the user's question, respond with the final answer in plain text.
3. Do NOT repeat a tool call you have already made. Use the result you already have.
4. Do not include any explanation, code blocks, or markdown outside the JSON or the final answer.

Examples:

User: What is 2 + 2?
Assistant: {"tool": "calculate", "input": "2 + 2"}
Tool result: 4
Assistant: 4

User: What is in data/numbers.txt?
Assistant: {"tool": "read_file", "input": "data/numbers.txt"}
Tool result: 12\n15\n23\n8\n2\n3\n5
Assistant: 12\n15\n23\n8\n2\n3\n5

User: What is the sum of the numbers in data/numbers.txt?
Assistant: {"tool": "read_file", "input": "data/numbers.txt"}
Tool result: 12\n15\n23\n8\n2\n3\n5
Assistant: {"tool": "calculate", "input": "12 + 15 + 23 + 8 + 2 + 3 + 5"}
Tool result: 68
Assistant: 68

User: What is the capital of France?
Assistant: Paris
"""


def _build_plan_messages(question: str, tool_history: list) -> list:
    """Build the messages for the planning step, including tool history."""
    content = _PLAN_SYSTEM_PROMPT
    if tool_history:
        content += "\n\nYou have already used tools. Here are the results:\n"
        for tool_name, tool_input, tool_result in tool_history:
            content += f"\n{tool_name}({tool_input}) -> {tool_result}"
        content += "\n\nIf you need another tool, use JSON. If you have enough information, answer directly."

    return [
        {"role": "system", "content": content},
        {"role": "user", "content": question},
    ]


def _build_answer_messages(question: str, tool_history: list) -> list:
    """Build the messages for the final answer step."""
    content = "Use the tool results below to answer the question. Output ONLY the final answer in plain text, with no explanation.\n\n"
    content += "Tool results:\n"
    for tool_name, tool_input, tool_result in tool_history:
        content += f"\n  {tool_name}({tool_input}) -> {tool_result}"
    content += f"\n\nQuestion: {question}\n\nFinal answer:"

    return [
        {"role": "user", "content": content},
    ]


def _clean_answer(text: str) -> str:
    """Remove common role prefixes and blank lines that small models may emit."""
    text = text.strip()
    prefixes = ("assistant", "Assistant", "answer:", "Answer:", "final answer:", "Final answer:")
    for prefix in prefixes:
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix):].strip()
            # Also strip a leading colon or newline if present.
            text = text.lstrip(":\n").strip()
            break
    return text


# -----------------------------------------------------------------------------
# Agent loop
# -----------------------------------------------------------------------------


def run_agent(question: str, verbose: bool = False, trace: bool = True) -> str:
    provider = "OpenAI" if USE_OPENAI else "Ollama"
    tracer = Trace(question, MODEL, provider) if trace else NullTrace()

    run_start = current_ms()
    tool_history = []
    final_answer = None

    try:
        # Tool loop: plan and execute until no tool is needed.
        for step in range(MAX_TOOL_STEPS):
            plan_messages = _build_plan_messages(question, tool_history)
            plan_start = current_ms()
            plan_output = chat(plan_messages)
            plan_duration = current_ms() - plan_start

            if verbose:
                print(f"[Plan {step + 1}] Agent: {plan_output}")

            tool_call = extract_json(plan_output)

            # No tool call: the model thinks it can answer directly.
            if not tool_call or tool_call.get("tool") not in TOOLS:
                if not tool_history:
                    # No tools used yet; treat this as the direct answer.
                    final_answer = _clean_answer(plan_output)
                    tracer.add_step(
                        phase="direct_answer",
                        llm_output=plan_output,
                        duration_ms=plan_duration,
                    )
                    break
                # Tools were used; move to the final answer phase.
                tracer.add_step(
                    phase="plan",
                    llm_output=plan_output,
                    duration_ms=plan_duration,
                    guard_triggered=True,
                    guard_reason="model answered directly after tool history; moving to answer phase",
                )
                break

            tool_name = tool_call["tool"]
            tool_input = tool_call.get("input", "")

            # Guard: do not repeat any tool call that already appears in history.
            if any(name == tool_name and inp == tool_input for name, inp, _ in tool_history):
                if verbose:
                    print(f"[Guard] Repeated tool call detected. Moving to answer phase.")
                tracer.add_step(
                    phase="plan",
                    llm_output=plan_output,
                    tool_name=tool_name,
                    tool_input=tool_input,
                    duration_ms=plan_duration,
                    guard_triggered=True,
                    guard_reason="repeated tool call blocked",
                )
                break

            tool_fn = TOOLS[tool_name]
            tool_start = current_ms()
            result = tool_fn(tool_input)
            tool_duration = current_ms() - tool_start

            if verbose:
                print(f"[Tool {step + 1}] {tool_name}({tool_input}) = {result}")

            tracer.add_step(
                phase="plan",
                llm_output=plan_output,
                tool_name=tool_name,
                tool_input=tool_input,
                tool_result=result,
                duration_ms=plan_duration + tool_duration,
            )

            if result.startswith("Error:"):
                if verbose:
                    print(f"[Guard] Tool failed. Returning error message.")
                final_answer = f"I could not use {tool_name}. {result}"
                tracer.add_step(
                    phase="error",
                    llm_output="",
                    guard_triggered=True,
                    guard_reason=f"tool {tool_name} returned error",
                )
                break

            tool_history.append((tool_name, tool_input, result))

        # Final answer phase: synthesize the gathered information.
        if final_answer is None:
            answer_messages = _build_answer_messages(question, tool_history)
            answer_start = current_ms()
            answer_output = chat(answer_messages)
            answer_duration = current_ms() - answer_start
            answer_output = _clean_answer(answer_output)
            if verbose:
                print(f"[Answer] Agent: {answer_output}")
            final_answer = answer_output
            tracer.add_step(
                phase="answer",
                llm_output=answer_output,
                duration_ms=answer_duration,
            )

    except Exception as e:
        final_answer = f"Agent error: {e}"
        tracer.error = str(e)
        if verbose:
            print(f"[Error] {e}")

    total_duration = current_ms() - run_start
    tracer.finalize(final_answer, total_duration, tracer.error)
    trace_path = tracer.write()

    if trace_path and verbose:
        print(f"[Trace] written to {trace_path}")

    return final_answer


# -----------------------------------------------------------------------------
# Interactive entry point
# -----------------------------------------------------------------------------


if __name__ == "__main__":
    provider = "OpenAI" if USE_OPENAI else "Ollama"
    print(f"Multi-Tool Agent ({provider} / {MODEL})")
    print("Type 'exit' to quit.\n")

    try:
        while True:
            question = input("Ask a question: ").strip()
            if question.lower() in {"exit", "quit"}:
                break
            if not question:
                continue

            answer = run_agent(question, verbose=True)
            print(f"\nFinal answer: {answer}\n")
    except EOFError:
        print("\nExiting (no more input).")
