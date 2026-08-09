"""
Minimal agent: a math assistant that uses a tool.

Demonstrates the core agent loop:
  1. Perceive input (user question)
  2. Plan and reason (LLM decides if a tool is needed)
  3. Select tool (calculate)
  4. Execute action (run the math expression)
  5. Evaluate feedback (LLM sees the result and returns final answer)
  6. Control flow (iterate up to a max step limit)

Runs locally with Ollama (llama3.1).
"""

import json
import math
import re

import ollama

MODEL = "llama3.1:latest"
MAX_STEPS = 5

SYSTEM_PROMPT = """You are a helpful math assistant.

You have access to one tool:
- calculate(expression): evaluates a mathematical expression and returns the result.

Rules:
1. If the user asks a question that requires calculation, respond with ONLY a JSON object in this exact format:
   {"tool": "calculate", "input": "<math expression>"}
2. After you receive the tool result, respond with the final answer in plain text. Do not call the tool again.
3. If no calculation is needed, respond directly in plain text.
4. Do not include any explanation outside the JSON or the final answer.

Examples:
User: What is 2 + 2?
Assistant: {"tool": "calculate", "input": "2 + 2"}
Tool result: 4
Assistant: 4

User: What is the capital of France?
Assistant: Paris
"""


def calculate(expression: str) -> str:
    """Evaluate a math expression in a restricted environment."""
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


def extract_json(text: str):
    """Try to find and parse a JSON object in the LLM response."""
    match = re.search(r"\{.*?\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            return None
    return None


def run_agent(question: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    for step in range(MAX_STEPS):
        response = ollama.chat(model=MODEL, messages=messages)
        content = response["message"]["content"].strip()

        print(f"[Step {step + 1}] Agent: {content}")

        tool_call = extract_json(content)

        if tool_call and tool_call.get("tool") == "calculate":
            expression = tool_call.get("input", "")
            result = calculate(expression)
            print(f"[Tool] calculate({expression}) = {result}")

            messages.append({"role": "assistant", "content": content})
            messages.append({"role": "tool", "content": result})
        else:
            # No tool call: this is the final answer.
            return content

    return "Max steps reached without a final answer."


if __name__ == "__main__":
    print("Simple Math Agent")
    print("Type 'exit' to quit.\n")

    try:
        while True:
            question = input("Ask a math question: ").strip()
            if question.lower() in {"exit", "quit"}:
                break
            if not question:
                continue

            answer = run_agent(question)
            print(f"\nFinal answer: {answer}\n")
    except EOFError:
        print("\nExiting (no more input).")
