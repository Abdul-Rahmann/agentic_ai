"""
Minimal agent: a math assistant that uses a tool.

Demonstrates the core agent loop with a two-phase design to make a small local
model more reliable:

  Phase 1 - Plan: decide whether a calculation is needed and which expression to use.
  Phase 2 - Answer: given the tool result, produce the final answer.

This is less flexible than a single open-ended loop, but much easier to debug
and a better starting point for learning.

Runs locally with Ollama (llama3.1). Set USE_OPENAI=1 to use OpenAI instead.
"""

import json
import math
import os
import re

import ollama

MODEL = os.getenv("AGENT_MODEL", "llama3.1:latest")
USE_OPENAI = os.getenv("USE_OPENAI", "0") == "1"

# -----------------------------------------------------------------------------
# Tool: calculate
# -----------------------------------------------------------------------------


def normalize_expression(expression: str) -> str:
    """Rewrite common math notation into valid Python syntax."""
    # n! -> factorial(n)  (factorial is imported from math in the evaluator)
    expression = re.sub(r"(\d+)!", r"factorial(\1)", expression)
    # ^ -> ** for exponentiation (common math notation, not Python XOR)
    expression = expression.replace("^", "**")
    return expression


def calculate(expression: str) -> str:
    """Evaluate a math expression in a restricted environment."""
    expression = normalize_expression(expression)
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
# Two-phase agent
# -----------------------------------------------------------------------------


PLAN_SYSTEM_PROMPT = """You are a math assistant. Your job is to decide whether a user question requires a calculation.

Rules:
1. If the question requires math, output ONLY a JSON object in this exact format:
   {"tool": "calculate", "input": "<math expression>"}
2. If the question does NOT require math, output the answer directly in plain text.
3. Do not include any explanation, code block, or extra text.

Examples:

User: What is 2 + 2?
Assistant: {"tool": "calculate", "input": "2 + 2"}

User: What is 15 * 23?
Assistant: {"tool": "calculate", "input": "15 * 23"}

User: What is 2 to the power of 10?
Assistant: {"tool": "calculate", "input": "2 ** 10"}

User: What is factorial of 5?
Assistant: {"tool": "calculate", "input": "5!"}

User: What is the capital of France?
Assistant: Paris
"""


ANSWER_SYSTEM_PROMPT = """You are a math assistant. You have already used a calculator tool to get the result of a user's question.

Rules:
1. Output ONLY the final answer in plain text.
2. Do not include the calculation, JSON, code blocks, or explanation.
3. Do not call the tool again.

Example:
Question: What is 2 + 2?
Tool result: 4
Assistant: 4
"""


def run_agent(question: str, verbose: bool = False) -> str:
    # Phase 1: plan / decide tool.
    plan_messages = [
        {"role": "system", "content": PLAN_SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    plan_output = chat(plan_messages)
    if verbose:
        print(f"[Plan] Agent: {plan_output}")

    tool_call = extract_json(plan_output)

    if not tool_call or tool_call.get("tool") != "calculate":
        # No math needed or model chose to answer directly.
        return plan_output

    expression = tool_call.get("input", "")
    result = calculate(expression)
    if verbose:
        print(f"[Tool] calculate({expression}) = {result}")

    # If the tool itself failed, return a clean error instead of confusing the answer phase.
    if result.startswith("Error:"):
        if verbose:
            print(f"[Guard] Tool failed. Returning error message.")
        return f"I could not calculate that. The expression '{expression}' is not valid Python math syntax."

    # Phase 2: answer given the tool result.
    answer_messages = [
        {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Question: {question}\nTool result: {result}",
        },
    ]
    answer_output = chat(answer_messages)
    if verbose:
        print(f"[Answer] Agent: {answer_output}")

    # Guard: if the final answer does not contain the computed result, return the raw result.
    if str(result) not in answer_output:
        if verbose:
            print(f"[Guard] Final answer did not contain tool result. Returning raw result.")
        return result

    return answer_output


# -----------------------------------------------------------------------------
# Interactive entry point
# -----------------------------------------------------------------------------


if __name__ == "__main__":
    provider = "OpenAI" if USE_OPENAI else "Ollama"
    print(f"Simple Math Agent ({provider} / {MODEL})")
    print("Type 'exit' to quit.\n")

    try:
        while True:
            question = input("Ask a math question: ").strip()
            if question.lower() in {"exit", "quit"}:
                break
            if not question:
                continue

            answer = run_agent(question, verbose=True)
            print(f"\nFinal answer: {answer}\n")
    except EOFError:
        print("\nExiting (no more input).")
