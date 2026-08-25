"""
Multi-tool agent implemented in LangGraph.

This rebuilds the hand-rolled agent from math_agent.py using LangGraph's
state-machine abstractions: nodes, edges, and shared state.

The goal is to keep the same tools, prompts, and behavior so the two
implementations can be compared directly.
"""

import json
import os
import sys

# Reuse the tools and prompts from the hand-rolled agent.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from math_agent import (
    TOOLS,
    _build_plan_messages,
    _build_answer_messages,
    _build_reflection_messages,
    _clean_answer,
    _parse_reflection,
    extract_json,
)
from tracer import Trace, NullTrace, current_ms

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from typing import TypedDict

# LLM configuration mirrors math_agent.py.
MODEL = os.getenv("AGENT_MODEL", "llama3.1:latest")
USE_OPENAI = os.getenv("USE_OPENAI", "0") == "1"
MAX_TOOL_STEPS = 5

# Create one LLM instance at module load to avoid recreating it per call.
if USE_OPENAI:
    from langchain_openai import ChatOpenAI
    _LLM = ChatOpenAI(model=MODEL, temperature=0.0)
else:
    from langchain_ollama import ChatOllama
    _LLM = ChatOllama(model=MODEL, temperature=0.0)


# -----------------------------------------------------------------------------
# State
# -----------------------------------------------------------------------------


class AgentState(TypedDict):
    """Shared state passed between LangGraph nodes."""

    question: str
    tool_history: list
    reflection_feedback: list[str]
    final_answer: str | None
    error: str | None
    attempts: int
    max_retries: int
    step_count: int
    done: bool
    pending_tool: dict | None
    reflect: bool
    trace: Trace | NullTrace


# -----------------------------------------------------------------------------
# LLM helper
# -----------------------------------------------------------------------------


def _chat(messages: list, temperature: float = 0.0) -> str:
    """Call the configured LLM via LangChain and return the assistant's content."""
    lc_messages = []
    for m in messages:
        role = m.get("role")
        content = m.get("content", "")
        if role == "system":
            lc_messages.append(SystemMessage(content=content))
        elif role == "user":
            lc_messages.append(HumanMessage(content=content))
        elif role == "assistant":
            lc_messages.append(AIMessage(content=content))
        else:
            lc_messages.append(HumanMessage(content=content))

    response = _LLM.invoke(lc_messages)
    return response.content.strip()


# -----------------------------------------------------------------------------
# Nodes
# -----------------------------------------------------------------------------


def plan_node(state: AgentState) -> dict:
    """Decide whether to call a tool or move to the answer phase."""
    question = state["question"]
    tool_history = state["tool_history"]
    reflection_feedback = state["reflection_feedback"]
    step_count = state["step_count"]
    attempts = state["attempts"]
    trace = state["trace"]

    if step_count >= MAX_TOOL_STEPS:
        final_answer = "Max steps reached without a final answer."
        trace.add_step(
            phase="error",
            llm_output="",
            guard_triggered=True,
            guard_reason="max tool steps reached",
            attempt=attempts,
        )
        return {"final_answer": final_answer, "done": True}

    plan_messages = _build_plan_messages(question, tool_history, reflection_feedback)
    plan_start = current_ms()
    plan_output = _chat(plan_messages)
    plan_duration = current_ms() - plan_start

    tool_call = extract_json(plan_output)

    # No tool call: the model thinks it can answer directly.
    if not tool_call or tool_call.get("tool") not in TOOLS:
        if not tool_history:
            # No tools used yet; treat this as the direct answer.
            final_answer = _clean_answer(plan_output)
            trace.add_step(
                phase="direct_answer",
                llm_output=plan_output,
                duration_ms=plan_duration,
                attempt=attempts,
            )
            return {"final_answer": final_answer, "done": True}

        # Tools were used; move to the final answer phase.
        trace.add_step(
            phase="plan",
            llm_output=plan_output,
            duration_ms=plan_duration,
            attempt=attempts,
            guard_triggered=True,
            guard_reason="model answered directly after tool history; moving to answer phase",
        )
        return {"pending_tool": None}

    tool_name = tool_call["tool"]
    tool_input = tool_call.get("input", "")

    # Guard: do not repeat any tool call that already appears in history.
    if any(name == tool_name and inp == tool_input for name, inp, _ in tool_history):
        trace.add_step(
            phase="plan",
            llm_output=plan_output,
            tool_name=tool_name,
            tool_input=tool_input,
            duration_ms=plan_duration,
            attempt=attempts,
            guard_triggered=True,
            guard_reason="repeated tool call blocked",
        )
        return {"pending_tool": None}

    trace.add_step(
        phase="plan",
        llm_output=plan_output,
        tool_name=tool_name,
        tool_input=tool_input,
        duration_ms=plan_duration,
        attempt=attempts,
    )
    return {"pending_tool": {"tool": tool_name, "input": tool_input}, "step_count": step_count + 1}


def execute_node(state: AgentState) -> dict:
    """Run the pending tool and append its result to tool_history."""
    pending = state["pending_tool"]
    tool_name = pending["tool"]
    tool_input = pending["input"]
    tool_history = state["tool_history"]
    attempts = state["attempts"]
    trace = state["trace"]

    tool_fn = TOOLS[tool_name]
    tool_start = current_ms()
    result = tool_fn(tool_input)
    tool_duration = current_ms() - tool_start

    trace.add_step(
        phase="plan",
        llm_output=json.dumps(pending),
        tool_name=tool_name,
        tool_input=tool_input,
        tool_result=result,
        duration_ms=tool_duration,
        attempt=attempts,
    )

    if result.startswith("Error:"):
        final_answer = f"I could not use {tool_name}. {result}"
        trace.add_step(
            phase="error",
            llm_output="",
            guard_triggered=True,
            guard_reason=f"tool {tool_name} returned error",
            attempt=attempts,
        )
        return {
            "tool_history": tool_history + [(tool_name, tool_input, result)],
            "final_answer": final_answer,
            "done": True,
            "pending_tool": None,
        }

    return {
        "tool_history": tool_history + [(tool_name, tool_input, result)],
        "pending_tool": None,
    }


def answer_node(state: AgentState) -> dict:
    """Synthesize the final answer from tool results."""
    question = state["question"]
    tool_history = state["tool_history"]
    reflection_feedback = state["reflection_feedback"]
    attempts = state["attempts"]
    trace = state["trace"]

    answer_messages = _build_answer_messages(question, tool_history, reflection_feedback)
    answer_start = current_ms()
    answer_output = _chat(answer_messages)
    answer_duration = current_ms() - answer_start
    answer_output = _clean_answer(answer_output)

    trace.add_step(
        phase="answer",
        llm_output=answer_output,
        duration_ms=answer_duration,
        attempt=attempts,
    )

    return {"final_answer": answer_output}


def reflect_node(state: AgentState) -> dict:
    """Verify the final answer and decide whether to retry or finish."""
    if not state["reflect"]:
        return {"done": True}

    question = state["question"]
    tool_history = state["tool_history"]
    final_answer = state["final_answer"]
    reflection_feedback = state["reflection_feedback"]
    attempts = state["attempts"]
    max_retries = state["max_retries"]
    trace = state["trace"]

    reflection_messages = _build_reflection_messages(question, tool_history, final_answer)
    reflection_start = current_ms()
    reflection_output = _chat(reflection_messages)
    reflection_output = _clean_answer(reflection_output)
    reflection_duration = current_ms() - reflection_start

    verified, reflection_text = _parse_reflection(reflection_output)
    trace.add_step(
        phase="reflection",
        llm_output=reflection_output,
        duration_ms=reflection_duration,
        guard_triggered=not verified,
        guard_reason=None if verified else reflection_text,
        attempt=attempts,
    )

    if verified:
        return {"done": True}

    # Reflection flagged the answer. Add feedback and retry if possible.
    new_feedback = reflection_feedback + [reflection_text]
    if attempts < max_retries:
        return {
            "reflection_feedback": new_feedback,
            "attempts": attempts + 1,
            "final_answer": None,
        }

    # No more retries. Return the answer with a warning.
    final = f"[Unverified after {max_retries} retries: {reflection_text}] {final_answer}"
    return {"final_answer": final, "done": True}


# -----------------------------------------------------------------------------
# Conditional edges
# -----------------------------------------------------------------------------


def route_after_plan(state: AgentState) -> str:
    if state.get("done"):
        return END
    if state.get("pending_tool"):
        return "execute"
    return "answer"


def route_after_reflect(state: AgentState) -> str:
    if state.get("done"):
        return END
    return "plan"


# -----------------------------------------------------------------------------
# Graph builder
# -----------------------------------------------------------------------------


def build_graph():
    builder = StateGraph(AgentState)
    builder.add_node("plan", plan_node)
    builder.add_node("execute", execute_node)
    builder.add_node("answer", answer_node)
    builder.add_node("reflect", reflect_node)

    builder.set_entry_point("plan")
    builder.add_conditional_edges(
        "plan",
        route_after_plan,
        {"execute": "execute", "answer": "answer", END: END},
    )
    builder.add_edge("execute", "plan")
    builder.add_edge("answer", "reflect")
    builder.add_conditional_edges(
        "reflect",
        route_after_reflect,
        {"plan": "plan", END: END},
    )
    return builder.compile()


# -----------------------------------------------------------------------------
# Public entry point
# -----------------------------------------------------------------------------


def run_agent(
    question: str,
    verbose: bool = False,
    trace: bool = True,
    reflect: bool = True,
    max_retries: int = 2,
) -> str:
    """Run the LangGraph agent on a question and return the final answer."""
    provider = "OpenAI" if USE_OPENAI else "Ollama"
    tracer = Trace(question, MODEL, provider) if trace else NullTrace()

    run_start = current_ms()
    initial_state = {
        "question": question,
        "tool_history": [],
        "reflection_feedback": [],
        "final_answer": None,
        "error": None,
        "attempts": 0,
        "max_retries": max_retries,
        "step_count": 0,
        "done": False,
        "pending_tool": None,
        "reflect": reflect,
        "trace": tracer,
    }

    graph = build_graph()

    try:
        final_state = graph.invoke(initial_state)
    except Exception as e:
        final_state = {"final_answer": f"Agent error: {e}", "error": str(e)}
        tracer.error = str(e)
        if verbose:
            print(f"[Error] {e}")

    final_answer = final_state.get("final_answer") or "No answer produced."
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
    print(f"LangGraph Multi-Tool Agent ({provider} / {MODEL})")
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
