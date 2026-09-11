"""
Two-agent team: Actor + Critic, in LangGraph.

This is Phase 7 (Multi-Agent) of the roadmap. Rather than one agent
reviewing its own answer (that's all langgraph_agent.py's reflect_node
ever was — a second LLM call with a lightweight VERIFIED/INCORRECT prompt,
barely distinguishable from "the same agent double-checking itself"), this
gives the reviewer a genuinely separate role:

  - Actor: plans, calls tools, and answers. Reuses plan_node, execute_node,
    and answer_node UNCHANGED from langgraph_agent.py — same tools, same
    prompts, same human-approval gate. Nothing about the Actor's own
    behavior changes.
  - Critic: reviews the Actor's answer against a structured checklist
    (numeric consistency, tool relevance, format match) rather than a
    single yes/no judgment call, and is explicitly instructed to treat the
    Actor's answer with skepticism rather than charity.

The two graphs (langgraph_agent.py's single-agent version and this one)
differ in exactly one node, so any behavior difference on the same
question is attributable to the Critic, not a confound elsewhere.
"""

import json
import os
import uuid

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from langgraph_agent import (
    AgentState,
    MODEL,
    USE_OPENAI,
    _chat,
    _print_stream_events,
    plan_node,
    execute_node,
    answer_node,
    route_after_plan,
    route_after_execute,
)
from math_agent import _clean_answer
from tracer import Trace, NullTrace, current_ms


# -----------------------------------------------------------------------------
# Critic
# -----------------------------------------------------------------------------


_CRITIC_SYSTEM_PROMPT = """You are the Critic in a two-agent team. A separate Actor agent just answered a question using tools. Your job is to check the answer against the checklist below.

Default to APPROVE. Only respond REVISE if you can cite ONE SPECIFIC, CONCRETE mismatch: an exact number that doesn't match the tool history, a result clearly taken from the wrong sub-question, or a format the question EXPLICITLY demanded (e.g. it said "just the number") that the answer does not follow. Do not invent a stricter interpretation of the question than what was literally asked, and do not require a format the question never requested.

Checklist:
1. Numeric consistency: if the answer contains a number, does it match a number from the tool history (after any necessary arithmetic)? Watch for the answer citing a different sub-question's result when multiple similar results appear in history.
2. Tool relevance: does the answer rely on the correct tool result for THIS specific question, not a different sub-question's result?
3. Format match: ONLY flag this if the question explicitly specified a format (e.g. "just the number, no explanation") and the answer violates that exact instruction. A complete, correctly-worded sentence is NOT a format violation just because it could theoretically be shorter.
4. If there is no tool history, verify using general knowledge. Trust live external tool results (get_weather, web_search) over your own training knowledge.

Respond with ONLY:
APPROVE: <answer>
or
REVISE: <the one specific, concrete mismatch — an exact wrong number or an exact unmet format instruction>

Do not include any explanation, code, or markdown outside the required format.

Examples:

Question: What is 15 * 23?
Tool history:
  calculate(15 * 23) -> 345
Proposed answer: 345
Response: APPROVE: 345

Question: What is 8 * 7, and separately what is 63 / 9?
Tool history:
  calculate(8 * 7) -> 56
  calculate(63 / 9) -> 7
Proposed answer: For 8 * 7 the answer is 7.
Response: REVISE: checklist item 2 (tool relevance) failed — 7 is the result of 63 / 9, not 8 * 7. The correct value for 8 * 7 is 56.

Question: What is the weather in Paris?
Tool history:
  get_weather(Paris) -> Current weather in Paris, France: 15°C, partly cloudy.
Proposed answer: It is 15°C and partly cloudy in Paris.
Response: APPROVE: It is 15°C and partly cloudy in Paris.

Question: What is the capital of France?
Tool history: (none)
Proposed answer: Paris
Response: APPROVE: Paris
"""


def _build_critic_messages(question: str, tool_history: list, proposed_answer: str) -> list:
    content = _CRITIC_SYSTEM_PROMPT
    content += "\n\nNow review this:\n\nTool history:\n"
    if tool_history:
        for tool_name, tool_input, tool_result in tool_history:
            content += f"\n  {tool_name}({tool_input}) -> {tool_result}"
    else:
        content += "\n  (none)"
    content += f"\n\nQuestion: {question}\nProposed answer: {proposed_answer}\n\nResponse:"
    return [{"role": "system", "content": content}]


def _parse_critique(critique_output: str) -> tuple[bool, str]:
    """Parse the critic's output. Returns (approved, answer_or_reason)."""
    text = critique_output.strip()
    if text.upper().startswith("APPROVE:"):
        return True, text[len("APPROVE:"):].strip()
    if text.upper().startswith("REVISE:"):
        return False, text[len("REVISE:"):].strip()
    # Fallback: assume approved if there's no explicit REVISE marker.
    return True, text


def critic_node(state: AgentState) -> dict:
    """Review the Actor's answer against the checklist and decide whether to retry."""
    if not state["reflect"]:
        return {"done": True}

    question = state["question"]
    tool_history = state["tool_history"]
    final_answer = state["final_answer"]
    reflection_feedback = state["reflection_feedback"]
    attempts = state["attempts"]
    max_retries = state["max_retries"]
    trace_events = state["trace_events"]

    critique_messages = _build_critic_messages(question, tool_history, final_answer)
    critique_start = current_ms()
    critique_output = _chat(critique_messages)
    critique_output = _clean_answer(critique_output)
    critique_duration = current_ms() - critique_start

    approved, critique_text = _parse_critique(critique_output)
    trace_events.append(
        {
            "phase": "critique",
            "llm_output": critique_output,
            "duration_ms": critique_duration,
            "guard_triggered": not approved,
            "guard_reason": None if approved else critique_text,
            "attempt": attempts,
        }
    )

    if approved:
        return {"done": True}

    # Critic flagged the answer. Add feedback and retry if possible.
    new_feedback = reflection_feedback + [critique_text]
    if attempts < max_retries:
        return {
            "reflection_feedback": new_feedback,
            "attempts": attempts + 1,
            "final_answer": None,
        }

    # No more retries. Return the answer with a warning.
    final = f"[Unapproved by Critic after {max_retries} retries: {critique_text}] {final_answer}"
    return {"final_answer": final, "done": True}


def route_after_critic(state: AgentState) -> str:
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
    builder.add_node("critic", critic_node)

    builder.set_entry_point("plan")
    builder.add_conditional_edges(
        "plan",
        route_after_plan,
        {"execute": "execute", "answer": "answer", END: END},
    )
    builder.add_conditional_edges(
        "execute",
        route_after_execute,
        {"plan": "plan"},
    )
    builder.add_edge("answer", "critic")
    builder.add_conditional_edges(
        "critic",
        route_after_critic,
        {"plan": "plan", END: END},
    )
    return builder.compile(checkpointer=MemorySaver())


# -----------------------------------------------------------------------------
# Public entry point
# -----------------------------------------------------------------------------


def run_agent(
    question: str,
    verbose: bool = False,
    trace: bool = True,
    reflect: bool = True,
    max_retries: int = 2,
    auto_approve: bool = False,
) -> str:
    """Run the two-agent (Actor + Critic) LangGraph team on a question."""
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
        "approved_tools": [],
        "auto_approve": auto_approve,
        "reflect": reflect,
        "trace_events": [],
    }

    graph = build_graph()
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 25}

    final_state: dict = {}
    try:
        state = initial_state
        while True:
            events = list(graph.stream(state, config, stream_mode="values"))
            if not events:
                break
            final_state = events[-1]

            if verbose:
                _print_stream_events(events)

            interrupts = final_state.get("__interrupt__")
            if interrupts and not final_state.get("auto_approve"):
                pending = interrupts[0].value
                tool_name = pending["tool"]
                tool_input = pending["input"]
                response = input(f"Approve {tool_name}({tool_input!r})? [y/N]: ").strip()
                state = Command(resume=response)
                continue

            if final_state.get("done"):
                break

            break
    except Exception as e:
        final_state = {"final_answer": f"Agent error: {e}", "error": str(e)}
        tracer.error = str(e)
        if verbose:
            print(f"[Error] {e}")

    final_answer = final_state.get("final_answer") or "No answer produced."

    trace_events = final_state.get("trace_events", [])
    for event in trace_events:
        tracer.add_step(**event)

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
    print(f"Actor + Critic Multi-Agent Team ({provider} / {MODEL})")
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
