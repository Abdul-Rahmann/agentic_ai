# Agentic AI: Learning Notes

> A living document for learning, building, and optimizing autonomous AI agents.  
> Update this file as new concepts, frameworks, experiments, and references are encountered.
>
> **For the complete step-by-step learning path, agents, features, tools, and assessment checklist, see [`AGENTS_ROADMAP.md`](./AGENTS_ROADMAP.md).**

---

## Table of Contents

- [Agentic AI: Learning Notes](#agentic-ai-learning-notes)
  - [Table of Contents](#table-of-contents)
  - [What is Agentic AI?](#what-is-agentic-ai)
    - [Key Characteristics](#key-characteristics)
  - [Practical Roadmap: How to Start Building Agents](#practical-roadmap-how-to-start-building-agents)
  - [Core Anatomy of an AI Agent](#core-anatomy-of-an-ai-agent)
    - [Planning Techniques](#planning-techniques)
  - [The Operational Loop](#the-operational-loop)
  - [System Topologies](#system-topologies)
    - [Single-Agent Architecture](#single-agent-architecture)
    - [Multi-Agent Architecture](#multi-agent-architecture)
  - [How to Build an Agent](#how-to-build-an-agent)
    - [Recommended Build Workflow](#recommended-build-workflow)
  - [Development Frameworks](#development-frameworks)
  - [Optimization Approaches](#optimization-approaches)
    - [Mechanism-Level Optimizations](#mechanism-level-optimizations)
    - [System-Level Optimizations](#system-level-optimizations)
    - [Prompt Engineering First](#prompt-engineering-first)
  - [90-Day Learning Plan](#90-day-learning-plan)
  - [Project Ideas](#project-ideas)
    - [Beginner](#beginner)
    - [Intermediate](#intermediate)
    - [Advanced](#advanced)
  - [Production Checklist](#production-checklist)
  - [Glossary](#glossary)
  - [References \& Resources](#references--resources)
    - [Key Survey Papers](#key-survey-papers)
    - [Frameworks to Explore](#frameworks-to-explore)
    - [Learning Materials](#learning-materials)
  - [Build Log](#build-log)
  - [To Add / Next Topics](#to-add--next-topics)
  - [Changelog](#changelog)

---

## What is Agentic AI?

Agentic AI refers to autonomous systems that perceive inputs, reason about goals, plan actions, execute them via tools, and reflect on outcomes. Unlike passive chatbots that respond to one-shot prompts, agentic AI systems act as **active workflow executors** operating in closed loops.

### Key Characteristics

- **Autonomy**: Pursues objectives over multiple steps without constant human input.
- **Planning**: Breaks goals into sub-goals, evaluates strategies, and adapts.
- **Tool Use**: Calls external functions, APIs, code interpreters, databases, and search engines.
- **Feedback Loops**: Observes results, retries, reflects, and updates memory.
- **Memory**: Maintains short-term context and, optionally, long-term knowledge.
- **Governance**: Operates within guardrails, budgets, and approval workflows.

---

## Practical Roadmap: How to Start Building Agents

### 1. Start Small and Concrete

Build a **single-agent tool user** for one narrow, high-frequency task before thinking about multi-agent orchestration.

Good first projects:

- A coding agent that reads a repo, runs tests, fixes a specific bug class
- A research agent that searches the web, extracts structured data, and writes a markdown report
- A data-processing agent that takes a file, validates it, transforms it, and writes back results

**Why**: narrow tasks are easier to debug, cheaper to run, and let you learn the control loop without distributed-system complexity.

### 2. Learn the Core Anatomy by Building It

Implement the closed loop yourself at least once, even if a framework abstracts it later:

| Component | What to build first |
|---|---|
| **Perception / Input** | Parse user intent into structured goal + constraints |
| **Planning** | Generate a step-by-step plan; decompose into sub-goals |
| **Tool use** | Give the agent callable functions (search, code execution, APIs, files) with strict schemas |
| **Action** | Execute the chosen tool, observe result |
| **Reflection** | Evaluate whether the result satisfies the sub-goal |
| **Memory** | Maintain short-term context; optionally persist trajectories and outcomes |
| **Control flow** | Re-plan if a step fails, succeeds, or returns unexpected output |

Starting from scratch with Python + an LLM SDK teaches you what frameworks hide. Once you hit boilerplate, move to a framework.

### 3. Pick a Framework Based on Your Use Case

- **LangChain / LangGraph**: good for flexible tool-calling agents and state-machine-like workflows
- **AutoGen**: good for multi-agent conversation patterns
- **CrewAI**: good for role-based multi-agent teams
- **MetaGPT**: good for software-engineering multi-agent teams
- **Plain LLM SDK + custom orchestration**: often enough for a single-agent MVP

**Recommendation**: build the first version in plain code, then adopt LangGraph if the state machine becomes messy or CrewAI/AutoGen if you genuinely need multiple agents.

### 4. Optimize in This Order

1. **Prompt and role clarity first.** A well-defined system prompt, output format, and role usually beats a bigger model.
2. **Add tool schemas and validation.** Constrain tool inputs/outputs. Make tools idempotent, least-privilege, and bounded (timeouts, retries, budgets).
3. **Add planning and decomposition.** Use chain-of-thought, sub-goal breakdown, and re-planning on failure.
4. **Add reflection and verification.** A separate verification step or “critic” check catches errors the actor misses.
5. **Add memory carefully.** Start with in-context short-term memory. Add persistent memory only when you have a clear hygiene strategy — vector DBs can pollute context with irrelevant retrieval.
6. **Add multi-agent only after single-agent is reliable.** Coordination overhead is high; most tasks do not need it.

### 5. Use Proven Optimization Techniques

- **Reflection / self-critic**: ask the agent to review its own plan and output before returning
- **Self-consistency**: run the same reasoning multiple times and vote or aggregate
- **Search and backtracking**: when a plan fails, search alternatives rather than giving up
- **In-context learning / few-shot examples**: show successful trajectories in the prompt
- **Retrieval-augmented generation (RAG)**: ground tool outputs and decisions in retrievable evidence
- **Hybrid neuro-symbolic**: for high-stakes tasks, combine LLM reasoning with symbolic checks, rule-based guards, and deterministic verifiers
- **Human-in-the-loop**: keep approval gates for irreversible or high-risk actions

### 6. Treat Reliability as a System Problem

Base model capability matters, but the stack around it matters more:

- **Observability**: log every decision, tool call, observation, and state transition
- **Telemetry**: track success rate, latency, cost, retries, and tool failure rates
- **Guardrails**: content filtering, action approval workflows, runtime policy enforcement
- **Testing**: evaluate end-to-end task success under realistic constraints, not just output fluency
- **Benchmarks**: define success criteria, robustness, security, latency, and cost before optimizing

### 7. Suggested 90-Day Learning Plan

| Phase | Focus | Deliverable |
|---|---|---|
| Weeks 1–2 | Build a plain-LLM agent loop | Agent that plans and calls one tool |
| Weeks 3–4 | Add tools, reflection, memory | Agent that completes a 3–5 step workflow reliably |
| Weeks 5–6 | Use a framework | Rebuild in LangGraph or CrewAI |
| Weeks 7–8 | Add evaluation and observability | Success benchmark + tracing dashboard |
| Weeks 9–10 | Add guardrails and human approval | Safe deployment to a constrained environment |
| Weeks 11–12 | Optional multi-agent | Two-agent collaboration on a decomposed task |

### Bottom Line

Do not start with multi-agent systems, persistent memory, or autonomous self-improvement. Start with one well-scoped agent that can plan, use tools, verify results, and log its behavior. Make it reliable and observable. Only then expand.

---

## Core Anatomy of an AI Agent

| Component | Function | Implementation Notes |
|---|---|---|
| **Cognition / Planning** | Interprets input, reasons, decomposes goals, and decides next steps | Powered by an LLM or VLM; uses chain-of-thought, reflection, and sub-goal decomposition |
| **Memory** | Stores past experiences, context, and acquired knowledge | Short-term working memory + long-term episodic/semantic storage (e.g., vector DB, graph DB) |
| **Perception / Reflection** | Interprets sensory-like inputs and evaluates outcomes | Handles text, audio, video, tool outputs, and error signals |
| **Action / Tools** | Translates decisions into real-world effects | APIs, code execution, search, calendars, databases, file system operations |

### Planning Techniques

- Chain-of-thought (CoT) reasoning
- Sub-goal decomposition
- Reflection and self-criticism
- Self-consistency and voting
- Search and backtracking
- ReAct-style reasoning (reason + act)
- Tree-of-thoughts

---

## The Operational Loop

A typical agent follows a closed loop:

```
Perceive Input
    ↓
Plan & Reason
    ↓
Select Tool
    ↓
Execute Action
    ↓
Evaluate Feedback
    ↓
Update Memory
    ↓
(repeat or terminate)
```

Each iteration should be logged. Observability is essential from day one.

---

## System Topologies

### Single-Agent Architecture

One LLM performs all reasoning, planning, memory, and tool execution. Best for narrow, well-defined tasks.

**Pros**: Simple to debug, cheaper, faster to iterate.  
**Cons**: Limited to one reasoning style and one set of tools.

### Multi-Agent Architecture

Two or more agents, each with a persona, tools, and responsibilities, coordinated vertically or horizontally.

**Common patterns**:
- **Vertical**: A planner delegates to specialized executors (e.g., coder, tester, reviewer).
- **Horizontal**: Peers collaborate, debate, or vote on a shared outcome.
- **Hierarchical**: A manager agent breaks tasks and assigns subtasks to worker agents.

**Use when**: The problem genuinely decomposes into distinct expertise areas, or when verification by an independent critic improves reliability.

---

## How to Build an Agent

### Recommended Build Workflow

1. **Define the task and scope**  
   Choose a narrow, high-frequency workflow with clear success criteria.

2. **Build the core loop in plain code first**  
   Use Python + an LLM SDK to implement perception, planning, tool selection, execution, and reflection. This teaches what frameworks hide.

3. **Design strict tool interfaces**  
   Every tool should have:
   - A clear JSON schema
   - Idempotency where possible
   - Least-privilege access
   - Timeouts and retry logic
   - Budget and termination rules

4. **Add memory incrementally**  
   Start with short-term in-context memory. Add persistent memory only when provenance and retrieval hygiene are understood.

5. **Add reflection and verification**  
   Before returning an answer, have the agent verify whether the result satisfies the original goal.

6. **Instrument everything**  
   Log every decision, tool call, observation, cost, and latency.

7. **Benchmark end-to-end outcomes**  
   Measure success rate, robustness, cost, and latency under realistic constraints.

8. **Deploy with governance**  
   Add guardrails, approval workflows, monitoring, and human oversight.

9. **Iterate and expand**  
   Only move to multi-agent or persistent memory after the single-agent loop is reliable.

---

## Development Frameworks

| Framework | Best For | Notes |
|---|---|---|
| **LangChain** | General tool-using agents, chaining, retrieval | Large ecosystem; good for prototyping |
| **LangGraph** | State-machine-like agent workflows, complex control flow | Strong for cyclical reasoning and conditional transitions |
| **AutoGen** | Multi-agent conversational patterns | Agents debate, code, and review each other |
| **CrewAI** | Role-based multi-agent teams | Simple abstraction for agents with roles and tasks |
| **MetaGPT** | Software engineering multi-agent teams | Agents emulate a software company (PM, architect, engineer, QA) |
| **Plain LLM SDK + custom orchestration** | Single-agent MVPs | Often the fastest path to learning and control |

**Recommendation**: Build the first MVP in plain code. Move to LangGraph if state management becomes complex. Move to AutoGen/CrewAI only if multi-agent collaboration is genuinely required.

---

## Optimization Approaches

### Mechanism-Level Optimizations

- **Reflection / self-criticism**: Ask the model to evaluate its own plan and output.
- **Self-consistency**: Generate multiple reasoning paths and vote or aggregate.
- **Search and backtracking**: When a step fails, explore alternatives.
- **In-context learning**: Provide few-shot examples of successful trajectories.
- **ReAct reasoning**: Interleave reasoning and tool use explicitly.
- **Retrieval-augmented generation (RAG)**: Ground decisions in external evidence.
- **Reinforcement learning / imitation learning**: For long-horizon tasks with clear rewards.
- **Hybrid neuro-symbolic systems**: Combine LLM reasoning with deterministic rules, verifiers, and symbolic checks.

### System-Level Optimizations

- **Modular components**: Separate planner, tool router, executor, critic, memory, and safety monitor.
- **Tool constraints**: Strict schemas, idempotency, least privilege, budgets, timeouts.
- **Governance**: Runtime policy enforcement, content filtering, action approval.
- **Observability**: Decision trace logging, behavior monitoring, cost tracking.
- **Evaluation**: Test under realistic constraints, not just ideal conditions.

### Prompt Engineering First

Before upgrading the model or adding complexity, optimize:

- System role definition
- Output format and schemas
- Few-shot examples
- Step-by-step instructions
- Explicit constraints and edge cases

A well-prompted smaller model often outperforms a poorly prompted larger model.

---

## 90-Day Learning Plan

| Phase | Weeks | Focus | Deliverable |
|---|---|---|---|
| **Foundation** | 1–2 | Build a plain-LLM agent loop | Single agent that plans and calls one tool |
| **Capability** | 3–4 | Add tools, reflection, memory | Agent completes a 3–5 step workflow reliably |
| **Frameworks** | 5–6 | Rebuild in a framework | Equivalent agent in LangGraph or CrewAI |
| **Evaluation** | 7–8 | Add observability and benchmarks | Success benchmark + tracing dashboard |
| **Governance** | 9–10 | Add guardrails and human approval | Safe deployment to a constrained environment |
| **Scale** | 11–12 | Optional multi-agent | Two or more agents collaborate on a decomposed task |

---

## Project Ideas

### Beginner
- Web research assistant: search, summarize, and cite sources in markdown.
- File validator: read CSV/JSON, validate schema, report errors.
- Coding assistant: read a repo, run tests, fix a specific bug class.

### Intermediate
- Task planner with calendar integration: parse natural language requests, check availability, schedule events.
- Document pipeline: ingest PDFs, extract structured data, classify, and write to a database.
- Code review agent: review diffs, check style, run tests, and summarize findings.

### Advanced
- Multi-agent software team: product manager, architect, coder, and QA collaborate to implement a feature.
- Autonomous research agent: generate hypotheses, search literature, synthesize reports, and self-critique.
- Self-improving agent: log trajectories, learn from failures, and update its own prompts or tool selection policy.

---

## Production Checklist

Before deploying an agent to a real environment:

- [ ] Success criteria defined and benchmarked
- [ ] Tool schemas validated and versioned
- [ ] Idempotency, timeouts, retries, and budgets configured
- [ ] Least-privilege permissions enforced
- [ ] Decision trace logging enabled
- [ ] Cost and latency monitored
- [ ] Content filtering and safety guardrails in place
- [ ] Human approval workflow for irreversible or high-risk actions
- [ ] Failure modes and fallback behavior documented
- [ ] Rollback plan defined
- [ ] Privacy and compliance requirements reviewed

---

## Glossary

| Term | Definition |
|---|---|
| **Agent** | An autonomous system that perceives, plans, acts, and reflects |
| **Agentic AI** | AI systems that operate autonomously in closed loops to achieve goals |
| **Chain-of-thought (CoT)** | Prompting the model to show intermediate reasoning steps |
| **ReAct** | A pattern that interleaves reasoning and action/tool use |
| **Tool use** | The agent's ability to call external functions or APIs |
| **Reflection** | The agent evaluating its own output or plan |
| **Multi-agent** | A system with two or more collaborating agents |
| **Guardrails** | Rules and constraints that keep agent behavior safe and aligned |
| **Observability** | Logging, tracing, and monitoring of agent behavior |
| **RAG** | Retrieval-augmented generation: grounding outputs in retrieved documents |

---

## References & Resources

### Key Survey Papers
- Liu et al. (2025) — agent architecture and components
- Xu et al. (2026) — planning, memory, and evaluation
- Qu et al. (2025) — planning, reflection, and memory
- Fang et al. (2025) — LLM agents as workflow executors
- Masterman et al. (2024) — tool use and single vs. multi-agent systems
- Sapkota et al. (2025) — multi-agent collaboration
- Swamy et al. (2025) — observability, governance, and deployment
- Nowaczyk et al. (2025) — reliability through modularization
- Kumar et al. (2025) — design principles and governance
- Allmendinger et al. (2026) — agentic AI orchestration
- Thakur et al. (2026) — safety, security, and interoperability
- Hosseini et al. (2025) — frameworks and adoption barriers
- Hughes et al. (2025) — safety, accountability, and real-world deployment
- Sumers et al. (2023) — language agents as modular systems
- Bharti et al. (2025) — retrieval-augmented and tool-augmented designs

### Frameworks to Explore
- [LangChain](https://www.langchain.com/)
- [LangGraph](https://langchain-ai.github.io/langgraph/)
- [AutoGen](https://microsoft.github.io/autogen/)
- [CrewAI](https://www.crewai.com/)
- [MetaGPT](https://github.com/geekan/MetaGPT)

### Learning Materials
- [OpenAI Function Calling Guide](https://platform.openai.com/docs/guides/function-calling)
- [Anthropic Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents)
- [ReAct Paper](https://arxiv.org/abs/2210.03629)
- [LangGraph State Machine Tutorial](https://langchain-ai.github.io/langgraph/tutorials/introduction/)

---

## Build Log

A record of hands-on experiments, prototypes, and lessons learned.

### Experiment 1: Simple Math Agent

**Date**: 2026-08-09  
**Location**: `first_agent/math_agent.py`  
**Goal**: Build the smallest possible agent that demonstrates the full loop (perceive, plan, tool use, execute, reflect, answer).

**What was built**:
- A single-agent math assistant using a local `llama3.1` model via Ollama.
- One tool: `calculate(expression)` which evaluates a restricted arithmetic expression.
- A hand-rolled control loop that parses JSON tool calls from the LLM, executes the tool, feeds back the result, and returns the final answer.
- Graceful EOF handling for non-interactive testing.

**What the loop looks like in practice**:

```
User: What is the square root of 144 plus 7?
[Step 1] Agent: {"tool": "calculate", "input": "sqrt(144) + 7"}
[Tool] calculate(sqrt(144) + 7) = 19.0
[Step 2] Agent: 19.0
Final answer: 19.0
```

**Key observations**:
- The model sometimes needs very explicit prompting to follow the JSON tool-call format and stop after one tool use.
- Extracting JSON with a regex fallback is necessary because small local models can produce malformed JSON or extra text.
- Even a tiny local model can complete a useful task when the loop is explicit and the tool is deterministic.
- This demonstrates the core idea: **the LLM does not compute the answer; it decides to use a deterministic tool, and the tool does the work.**
- A **two-phase design** (first plan the tool call, then answer from the result) dramatically improves reliability with small local models.
- A simple **guard** (check that the final answer contains the tool result) catches the remaining cases where the model drifts.

**What was changed**:
- Replaced the single open-ended loop with a two-phase loop: `plan` and `answer`.
- Added a guard that returns the raw tool result if the final answer does not contain it.
- Added optional OpenAI support via `USE_OPENAI=1` environment variable.

---

### Experiment 2: Multi-Tool Agent (calculate + read_file)

**Date**: 2026-08-09  
**Location**: `first_agent/math_agent.py`, `first_agent/stress_test.py`  
**Goal**: Add a second tool and evolve the agent from a single-purpose math assistant into a multi-tool agent that can plan sequences of tool calls.

**What was built**:
- Added `read_file(path)` tool that reads text files within the `first_agent` directory.
- Replaced the two-phase plan/answer design with a **tool loop + final answer phase**:
  - Plan: decide which tool to use or answer directly.
  - Execute: run the tool and record the result.
  - Re-plan: repeat until the model has enough information.
  - Answer: synthesize a final answer from the tool history.
- Added a guard that prevents the exact same tool call from being executed twice.
- Added safety restriction: `read_file` only reads files inside the project directory.
- Added sample data file `first_agent/data/numbers.txt`.
- Expanded the stress test to cover math, file reads, multi-step tasks, and non-math questions.

**Stress test results**:

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

**Key observations**:
- Multi-step planning (`read_file` → `calculate` → answer) works with a small local model when the loop and guards are explicit.
- Small local models struggle to know when to stop calling tools. Explicit rules and history guards are essential.
- Final answer synthesis is fragile with small models; output cleanup (removing `assistant:` prefixes) is necessary.
- Deterministic tools do the actual work; the LLM only decides which tool to use and when.

**Next steps for this experiment**:
- Add a reflection phase where a separate prompt verifies the final answer.
- Add structured logging to record every tool call and decision.
- Add a third tool (e.g., web search or code execution sandbox).
- Test more adversarial and ambiguous multi-step prompts.

---

### Experiment 3: Structured Observability / Tracing

**Date**: 2026-08-13  
**Location**: `first_agent/tracer.py`, `first_agent/math_agent.py`, `first_agent/stress_test.py`  
**Goal**: Add structured tracing so every run produces a machine-readable record of decisions, tool calls, results, timings, and guard events.

**What was built**:
- A `tracer.py` module with a `Trace` class that records:
  - `trace_id`, `timestamp`, `question`, `model`, `provider`
  - Every step: phase, raw LLM output, tool name/input/result, duration, guard triggers
  - Final answer, total duration, and any error
- A `NullTrace` class to disable tracing during benchmarks.
- JSON trace files written to `first_agent/traces/` automatically on every interactive run.
- Stress test updated to disable tracing (`trace=False`) to avoid clutter.

**Why it matters**:
- Reproducibility: every run is recorded exactly.
- Debugging: you can replay the exact sequence of decisions when a failure occurs.
- Measurement: per-step and total latency are captured automatically.
- Safety/audit: every tool call and data access is logged.

**Example trace**:

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

**Key observations**:
- Tracing adds minimal overhead but dramatically improves debuggability.
- Recording raw LLM outputs is essential; you cannot reconstruct the failure from the final answer alone.
- Guard events should be explicit in traces so you can see why the loop stopped.
- Benchmarks should be able to disable tracing to avoid generating thousands of files.

---

### Experiment 4: Reflection / Critic Phase

**Date**: 2026-08-13  
**Location**: `first_agent/math_agent.py`  
**Goal**: Add a separate critic prompt that verifies the final answer against the question and tool history before returning it to the user.

**What was built**:
- A reflection/critic prompt in `math_agent.py` that responds with either:
  - `VERIFIED: <answer>` — answer is correct and supported by tool history.
  - `INCORRECT: <reason>` — answer is wrong or unsupported.
- `_build_reflection_messages` and `_parse_reflection` helpers.
- `run_agent` gained a `reflect=True` parameter.
- Reflection step is recorded in the JSON trace as phase `reflection`.
- If the critic flags an answer, the agent returns it with an `[Unverified: ...]` warning.
- Stress test now runs with reflection enabled by default.

**Why it matters**:
- Reflection is a proven optimization technique for language agents.
- It adds a second verification layer after final answer synthesis.
- It catches drift between the tool results and the final answer.

**Stress test results with reflection**:

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

Reflection added roughly one extra LLM call per question, raising mean latency from ~2.3s to ~3.1s. All answers were verified in this run.

**Key observations**:
- A small local model can act as a critic when given clear examples and a strict output format.
- Reflection is currently conservative: flagged answers are returned with a warning rather than triggering an automatic retry. This avoids compounding model errors.
- The next evolution would be an explicit retry loop when the critic detects an error.

---

### Experiment 5: Reflection Retry Loop

**Date**: 2026-08-16  
**Location**: `first_agent/math_agent.py`, `first_agent/test_retry.py`  
**Goal**: Evolve reflection from a warning-only verifier into a self-correcting retry loop.

**What was built**:
- Added `max_retries` parameter to `run_agent` (default: 2).
- Wrapped plan/answer/reflection in an outer attempt loop.
- When reflection returns `INCORRECT: <reason>`, the reason is appended to `reflection_feedback` and included in the plan and answer prompts for the next attempt.
- Tool results are cached across attempts via `tool_history`; tools are not re-executed on retry.
- Traces record an `attempt` number per step so retry sequences are visible.
- Added `test_retry.py`, which mocks the LLM to force a wrong answer on the first attempt and verifies the retry corrects it.

**Why it matters**:
- Reflection without retry only tells you something is wrong; retry turns the critic's feedback into action.
- It demonstrates that the agent loop must be designed to support re-planning and that tool results should be deterministic/cacheable.
- It is a concrete self-improvement pattern used in more advanced agent systems.

**Example retry test output**:

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

**Stress test results with reflection + retry**:

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

**Key observations**:
- Caching tool results across attempts is essential for efficiency and determinism.
- The critic's feedback must be surfaced to the plan and answer prompts to have a corrective effect.
- The existing guard against repeated tool calls naturally pushes the model toward re-synthesis on retry.
- Retry only helps when the underlying tool results are trustworthy.
- Traces now include an `attempt` field, making retry sequences auditable.

---

### Experiment 6: Multi-Tool Agent with Live Weather Tool

**Date**: 2026-08-16  
**Location**: `first_agent/math_agent.py`, `first_agent/stress_test.py`  
**Goal**: Add a third tool that calls an external API and verify the agent can combine it with existing tools in multi-step plans.

**What was built**:
- Added `get_weather(city)` tool using Open-Meteo (geocoding + current forecast, no API key).
- Updated the plan prompt with a weather example.
- Expanded the stress test with `WEATHER_QUESTIONS` and `WEATHER_MATH_QUESTIONS`.
- Added `expect_number` flag to `run_single_test` for cases where only a numeric answer can be asserted (live weather + math).

**Stress test results**:

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

**Key observations**:
- The existing plan/answer/reflection loop handled the new tool without structural changes.
- Live data requires fuzzy test assertions (substrings, numeric presence) rather than exact values.
- External API tools need timeouts and clean error handling to avoid crashing the agent.
- Combining a live API value with math (`temperature in Paris plus 10`) confirms the agent can chain heterogeneous tools.

**Next steps for this experiment**:
- Add a fourth tool (web search, sandboxed code execution, or persistent memory).
- Make tools configurable and add cost/budget-aware routing.
- Test adversarial prompts and ambiguous multi-step tasks.

---

### Experiment 7: Four-Tool Agent with Hybrid Web Search

**Date**: 2026-08-21 / updated 2026-08-22  
**Location**: `first_agent/math_agent.py`, `first_agent/stress_test.py`  
**Goal**: Add a general knowledge retrieval tool and verify the agent can route among four distinct tools. Then fix reliability issues by switching to a DuckDuckGo-first + Wikipedia-fallback search and tightening the reflection prompt.

**What was built**:
- Added `web_search(query)` tool using DuckDuckGo Instant Answer first, falling back to Wikipedia (no API key, stdlib `urllib`, in-process cache, query variants).
- Updated the plan prompt with a web-search example and reinforced the existing product example after a regression.
- Added `WEB_SEARCH_QUESTIONS` to the stress test using substring assertions.
- Updated the reflection prompt with an explicit rule to trust live external tool results over training knowledge.

**Stress test results**:

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
Mean response time:    3.91s
Max response time:     7.10s
```

**Key observations**:
- The same loop handled a fourth tool type without structural changes.
- Free APIs need rate-limit awareness: cache, timeout, polite user-agent, and fuzzy test assertions.
- Adding tools makes the plan prompt longer, which can weaken existing examples. Re-adding a targeted few-shot example fixed the product regression.
- DuckDuckGo Instant Answer is great for factual/entity queries (e.g., "President of Ghana") but cannot answer real-time questions like "today's date".
- Wikipedia fallback is essential for coverage on queries DuckDuckGo misses (e.g., "CEO of OpenAI", "Who wrote Hamlet").
- The reflection critic must be told explicitly to trust live tool results; otherwise it invents training-data cutoffs and rejects correct current facts.

**Next steps for this experiment**:
- Add a dedicated date/time tool for real-time date questions.
- Add search+math multi-step questions (e.g., population of France divided by 10).
- Build a trace analyzer script.
- Make tool selection configurable and add cost/budget-aware routing.
- Test adversarial and ambiguous prompts.

---

### Experiment 8: Five-Tool Agent with Current Date

**Date**: 2026-08-23  
**Location**: `first_agent/math_agent.py`, `first_agent/stress_test.py`  
**Goal**: Add a deterministic date tool and verify the agent uses it for real-time date questions instead of unreliable search.

**What was built**:
- Added `get_current_date()` tool using Python's `datetime` (no network, no key).
- Updated plan prompt with a date example.
- Added `DATE_QUESTIONS` to the stress test with runtime-generated expected substrings.

**Stress test results**:

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

**Key observations**:
- Not every real-time question should be solved by search; a cheap deterministic tool is often better.
- The model correctly routed date questions to `get_current_date()` after adding a clear plan-prompt example.
- Stress-test assertions for date must be generated at runtime because the expected value changes daily.

**Next steps for this experiment**:
- Add a `get_current_time()` tool for exact clock time.
- Add search+math multi-step questions.
- ~~Build a trace analyzer script.~~ Done.
- Make tool selection configurable and add budget-aware routing.

---

### Experiment 9: Trace Analyzer


**Date**: 2026-08-23  
**Location**: `first_agent/trace_analyzer.py`, `first_agent/traces/`  
**Goal**: Turn JSON trace files into an actionable observability report.

**What was built**:
- Added `trace_analyzer.py` that reads `first_agent/traces/*.json`.
- Reports duration stats, per-phase latency, tool usage, guard events, errors, unverified answers, and retries.
- Supports filters: `--last N`, `--since YYYY-MM-DD`, `--slowest`, `--failed`.

**Sample findings from the first run**:
- 36 traces spanning 10 days.
- 6 unverified answers, all from "Who is the president of ..." questions before the reflection prompt was fixed.
- `plan` phase dominates both in count and max latency.
- `get_weather` was the most-used tool in the corpus.

**Key observations**:
- Observability is more than logging — it is turning logs into metrics.
- A small analyzer script makes failure patterns visible without opening individual JSON files.
- Guard-event summaries reveal where the model struggles (e.g., repeated tool calls, incorrect synthesis).

**Next steps for this experiment**:
- Add pass/fail classification if expected answers are added to traces.
- Add cost estimation (per-provider token counts) when available.
- Export the report to markdown or JSON for CI integration.

---

### Experiment 10: Rebuilding the Agent in LangGraph

**Date**: 2026-08-23  
**Location**: `first_agent/langgraph_agent.py`, `first_agent/stress_test.py`  
**Goal**: Port the hand-rolled agent to LangGraph and compare the two implementations.

**What was built**:
- Created `langgraph_agent.py` with `AgentState`, `plan_node`, `execute_node`, `answer_node`, `reflect_node`, and conditional edges.
- Reused tools and prompts from `math_agent.py`.
- Added `--langgraph` flag to `stress_test.py`.

**Stress test comparison**:

| Implementation | Pass rate | Mean latency | Max latency |
|---|---|---|---|
| Hand-rolled | 34 / 34 (100%) | ~3.91s | ~6.41s |
| LangGraph | 34 / 34 (100%) | ~4.00s | ~7.34s |

**Key observations**:
- The hand-rolled agent was already a state machine in disguise.
- LangGraph makes nodes, state, and routing explicit.
- Framework overhead is small compared to LLM inference time.
- Reusing tools and prompts made the port straightforward and validated that prompts are the main reliability driver.
- LangGraph opens the door to persistence, human-in-the-loop, and multi-agent coordination.

**Next steps for this experiment**:
- Add LangGraph persistence/checkpointing.
- Add a human-in-the-loop approval gate before external tools.
- Explore multi-agent patterns.

---

### Experiment 11: Human-in-the-Loop Approval in LangGraph

**Date**: 2026-08-29
**Location**: `first_agent/langgraph_agent.py`, `first_agent/stress_test.py`
**Goal**: Add a human approval gate before external tools (`web_search`, `get_weather`) in the LangGraph agent.

**What was built**:
- Added `APPROVAL_REQUIRED_TOOLS`, `approved_tools`, and `auto_approve` to the LangGraph agent.
- Used `langgraph.types.interrupt()` inside `execute_node` to pause before external tools.
- Updated the outer `run_agent` loop to detect `__interrupt__` events, prompt the user, and resume with `Command(resume=response)`.
- Added `--auto-approve` to `stress_test.py` for headless LangGraph runs.

**First attempt failed**:
- A manual checkpoint using `pending_approval` state + `await_approval` node recursed infinitely (`execute -> await_approval -> execute`) before the outer `input()` was called.

**Fix**:
- Replaced the manual checkpoint with LangGraph's native `interrupt()`. This suspends execution mid-node and exposes a resume API.

**Approval behavior**:
| User input | Result |
|---|---|
| `y` | Tool executes and the agent continues. |
| anything else | Agent returns `Tool <name> was not approved.` and stops. |

**Stress test results**:

| Implementation | Pass rate | Mean latency | Max latency |
|---|---|---|---|
| Hand-rolled | 34 / 34 (100%) | ~4.03s | ~6.76s |
| LangGraph (auto-approve) | 34 / 34 (100%) | ~4.29s | ~12.93s |

**Key observations**:
- Native interrupts are the right primitive for human-in-the-loop; hand-rolled checkpoints are fragile.
- Approval gates are a governance/safety layer, not just UX.
- `auto_approve` is needed for automated tests but should be off by default in interactive use.

**Next steps for this experiment**:
- Add per-tool approval policies.
- Add a timeout to the approval prompt for headless defaults.
- Combine with LangGraph persistence so paused approvals survive restarts.

---

### Experiment 12: Semantic Memory / RAG over the Project's Own Docs

**Date**: 2026-08-29
**Location**: `first_agent/memory.py`, `first_agent/math_agent.py`, `first_agent/stress_test.py`
**Goal**: Give the agent a local knowledge base and a `recall_knowledge` tool (first piece of Phase 6: Memory & RAG).

**What was built**:
- `memory.py`: local RAG store using `chromadb` (persistent) + `sentence-transformers` (`all-MiniLM-L6-v2`, local, no API key).
- Chunks and embeds this repo's own docs (`agentic-ai-learning.md`, `AGENTS_ROADMAP.md`, both READMEs, `development-log.md`); seeds once, persists to `first_agent/chroma_db/` (gitignored, rebuildable).
- `recall_knowledge(query)` added to `TOOLS` and the plan prompt, with a rule distinguishing it from `web_search` (this project's own history vs. general/live facts).
- `langgraph_agent.py` picked up the tool automatically (shared `TOOLS`/prompts from `math_agent.py`).
- Added `MEMORY_QUESTIONS` to the stress test — answers exist only in this repo's docs, so a pass proves retrieval, not recall.

**Environment fix along the way**: weather/web-search stress test failures turned out to be an unrelated SSL certificate gap in this Python.framework install (never ran `Install Certificates.command`), not a regression. Fixed by running that script.

**Stress test results** (both implementations, 36 cases = 34 existing + 2 memory):

| Implementation | Pass rate | Mean latency | Max latency |
|---|---|---|---|
| Hand-rolled | 36 / 36 (100%) | ~4.79s | ~14.86s |
| LangGraph (auto-approve) | 36 / 36 (100%) | ~4.59s | ~13.11s |

**Key observations**:
- A vector store is just another tool in the existing loop — no structural change needed.
- Seeding once and persisting keeps repeated runs cheap.
- Using the project's own docs as the knowledge base makes retrieval genuinely verifiable: the facts exist nowhere else.
- Don't assume the latest change caused a failure — check whether it's environmental first.

**Next steps for this experiment**:
- Episodic memory (SQLite): store and recall past runs/corrections.
- Prune/summarize long tool histories (short-term memory).
- Try local knowledge first, web search as fallback.

---

### Experiment 13: Episodic Memory

**Date**: 2026-09-03
**Location**: `first_agent/episodic_memory.py`, `first_agent/math_agent.py`, `first_agent/test_episodic_memory.py`
**Goal**: Give the agent memory of its own past runs, so a reflection-flagged mistake gets avoided on a later, separate run of a similar question — not just within one run's retry loop (the other half of Phase 6).

**What was built**:
- `episodic_memory.py`: SQLite-backed `record_episode()` / `recall_similar_episode()`. Wired into `math_agent.py`'s `run_agent()` by seeding `reflection_feedback` with a past failure's explanation before the first attempt — reusing the existing retry-loop mechanism rather than building a parallel one.
- `test_episodic_memory.py`: proves the outcome actually changes across two separate `run_agent()` calls (a mocked "wrong answer" run gets recorded, then a paraphrased question in a new call self-corrects because of the recalled feedback), not just that text reached the prompt.
- Not yet ported to `langgraph_agent.py`.

**Two bugs found immediately by testing**:
1. Character-level similarity (`difflib`) scored `"15 * 23"` vs `"15 times 23"` at 0.83 — below threshold — the same keyword-vs-meaning gap `memory.py` already solved. Fixed by reusing `memory.py`'s embedding model for cosine similarity instead (0.896 for the same pair).
2. A relative path for the benchmark's isolated episode DB only worked when run from the repo root; run from inside `first_agent/`, it crashed every single stress-test question (0/36) because the recall call sat outside `run_agent`'s `try/except`. Fixed the path (anchor to `__file__`, like every other path in this codebase) and wrapped both the recall and record calls defensively, matching how every tool call already degrades instead of crashing.

**Stress test results**: both implementations pass 36/36 (LangGraph doesn't have episodic memory yet — this just confirms nothing broke).

**Key observations**:
- The same "lexical similarity isn't semantic similarity" problem showed up in a second, unrelated place in one session — worth treating as a general rule.
- New logic should be wrapped as defensively as the code around it by default, not assumed safe.
- Reusing an existing mechanism (`reflection_feedback`) kept the integration to a few lines with zero prompt-building changes.

**Next steps for this experiment**:
- Port episodic memory to the LangGraph agent.
- Short-term memory (summarize/prune long tool histories) — the last open box in Phase 6.

---

### Experiment 14: Short-Term Memory (Summarize/Prune Tool History)

**Date**: 2026-09-06
**Location**: `first_agent/math_agent.py`, `first_agent/test_pruning.py`
**Goal**: Close the last open box in Phase 6 — keep long tool results and long `tool_history` from bloating every future prompt within a run.

**What was built**:
- A tool result over `MAX_TOOL_RESULT_CHARS` (1000, chosen to catch `recall_knowledge`'s ~2280-char typical output while leaving `web_search`'s ~820 chars alone) gets summarized before being stored in `tool_history`. Error results are never summarized.
- `tool_history` over `MAX_TOOL_HISTORY_ENTRIES` (6) gets its oldest entries collapsed into one summary, keeping the 5 most recent in full.
- The trace still records the full raw result — pruning only affects what feeds back into future prompts, not observability.
- `test_pruning.py`: mocked, deterministic tests of the mechanism (threshold, cap, error passthrough).

**Two real prompt failures found by testing on actual data**:
1. A plain "summarize in 2-4 sentences" prompt fed the real `recall_knowledge` output for a stress-test question and produced a generic paragraph that **dropped the literal `await_approval` identifier and the actual answer**. Fixed by reframing as fact extraction (a bullet list, every bullet must contain a concrete detail) with a worked example.
2. That fix didn't generalize: a direct unit test on collapsing 8 `calculate` calls showed the summary preserving *inputs* but **dropping every result value** — because the only example was prose-shaped, not `name(input) -> result`-shaped. Fixed with an explicit rule plus a second example in that exact shape.

**Stress test results**: both implementations pass 36/36 (LangGraph doesn't have this yet). Max response time got noisier (17–41s vs. 13–18s before) — the real cost of an extra LLM call whenever summarization triggers.

**Key observations**:
- The same "a general rule needs a worked example in the same shape as the real input" lesson from `memory.py` showed up again in a new form.
- "Summarize" and "extract facts" produce different failure modes in a small model — summarization drifts toward describing the topic, fact extraction stays anchored to specifics. Ask for facts when the specifics are the answer.
- The `_prune_tool_history` bug was caught by a targeted unit test, not the stress suite — no current question triggers 6+ tool calls, a reminder that passing tests only prove the paths they exercise.

**Next steps for this experiment**:
- Port short-term pruning to the LangGraph agent.
- Phase 6 is now fully checked off. Natural next phase: Phase 7 (Multi-Agent).

---

## To Add / Next Topics

Use this section to track future additions to the notes.

- [ ] Concrete code example: minimal ReAct agent in Python
- [ ] Tool schema design patterns and best practices
- [ ] Memory design: short-term vs. long-term vs. episodic vs. semantic
- [ ] Vector DB and RAG integration patterns for agents
- [ ] Multi-agent coordination patterns (vertical, horizontal, hierarchical)
- [ ] Evaluation frameworks: SWE-bench, AgentBench, WebArena, GAIA
- [ ] Cost and latency optimization strategies
- [ ] Safety and alignment: red-teaming, adversarial robustness
- [ ] Deployment patterns: sync API, async workers, streaming
- [ ] Case study: build and evaluate a real agent end-to-end
- [ ] Notes from specific framework tutorials or experiments
- [ ] Comparison of agentic frameworks after hands-on use

---

## Changelog

| Date | Change |
|---|---|
| 2026-08-09 | Initial version: captured definitions, anatomy, build workflow, optimization, 90-day plan, and references from Consensus research and discussion. |
| 2026-08-09 | Added "Practical Roadmap: How to Start Building Agents" section with the 7-step start-to-build guide. |
| 2026-08-09 | Added "Build Log" section and recorded Experiment 1: Simple Math Agent using Ollama and a hand-rolled tool loop. |
| 2026-08-09 | Added stress test for the math agent; fixed expression normalization for `^` and `!`; achieved 26/26 pass rate. |
| 2026-08-09 | Evolved agent to multi-tool (calculate + read_file); added tool loop, guards, and multi-step stress test; achieved 29/29 pass rate. |
| 2026-08-09 | Added more numbers (2, 3, 5) to `data/numbers.txt`; updated stress test expected values; still 29/29 passing. |
| 2026-08-13 | Added structured tracing (`tracer.py`, `traces/`); every interactive run now writes a JSON trace; stress test disables tracing. |
| 2026-08-13 | Added reflection/critic phase; `run_agent` now verifies answers with a separate prompt before returning; still 29/29 passing. |
| 2026-08-16 | Added reflection retry loop with `max_retries`, cached tool results, and `test_retry.py`; stress test still 29/29 passing. |
| 2026-08-16 | Added third tool `get_weather(city)` using Open-Meteo; expanded stress test to 31 cases (weather and weather+math); all 31 passing. |
| 2026-08-21 | Added fourth tool `web_search(query)` using Wikipedia API; expanded stress test to 32 cases; all 32 passing. |
| 2026-08-22 | Switched `web_search` to DuckDuckGo Instant Answer with Wikipedia fallback; fixed reflection prompt to trust live tool results; stress test still 32/32. |
| 2026-08-23 | Added fifth tool `get_current_date()`; expanded stress test to 34 cases; all 34 passing. |
| 2026-08-23 | Added `trace_analyzer.py` to summarize trace files into latency, tool usage, guard, and reliability reports. |
| 2026-08-23 | Rebuilt agent in LangGraph (`langgraph_agent.py`); added `--langgraph` stress-test flag; both implementations pass 34/34. |
| 2026-08-29 | Added human-in-the-loop approval to the LangGraph agent using `interrupt()`; added `--auto-approve` to `stress_test.py`; both hand-rolled and LangGraph (auto-approve) pass 34/34. |
| 2026-08-29 | Added local semantic memory / RAG (`memory.py`, `recall_knowledge` tool) over the project's own docs using Chroma + sentence-transformers; expanded stress test to 36 cases; fixed an unrelated SSL cert environment issue; both implementations pass 36/36. |
| 2026-08-29 | Fixed two `memory.py` gaps found by manual testing: chunking is now heading-aware (never spans two sections) for better retrieval, and seeding is fingerprint-based (auto re-seeds on doc changes, not just on an empty collection). Also fixed a brittle stress-test assertion pattern (one exact substring required) that was flagging factually-correct, differently-phrased answers as failures. |
| 2026-09-03 | Added episodic memory (`episodic_memory.py`, SQLite): the agent now recalls a reflection-flagged mistake from a past run and avoids repeating it in a new one. Fixed a similarity-matching bug (character-level diffing scored a valid paraphrase too low; switched to embedding-based cosine similarity) and a crash bug (relative DB path only worked from the repo root, and the recall call wasn't defensively wrapped) found while testing. Hand-rolled agent only so far; both implementations still pass 36/36. |
| 2026-09-06 | Added short-term memory management: long tool results get summarized before entering `tool_history`, and `tool_history` itself gets capped with older entries collapsed. Fixed two prompt failures found by testing on real data (narrative summarization dropped the key identifier; the fix didn't generalize to `name(input) -> result`-shaped tool history until a matching example was added). Phase 6 is now fully checked off. Hand-rolled agent only so far; both implementations still pass 36/36. |

---

> **Tip**: Treat this file as a working notebook. After each experiment, paper read, or framework deep dive, add a concise section, update the checklist, and record what worked and what did not.
