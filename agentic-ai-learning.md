# Agentic AI: Learning Notes

> A living document for learning, building, and optimizing autonomous AI agents.  
> Update this file as new concepts, frameworks, experiments, and references are encountered.

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

---

> **Tip**: Treat this file as a working notebook. After each experiment, paper read, or framework deep dive, add a concise section, update the checklist, and record what worked and what did not.
