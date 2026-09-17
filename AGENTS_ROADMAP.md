# Complete Agentic AI Learning Roadmap

> A living curriculum for learning, building, and mastering autonomous AI agents.  
> This document maps the journey from a first single-tool agent to production-grade multi-agent systems. It is meant to be updated as new experiments, papers, frameworks, and failures are encountered.

---

## Table of Contents

1. [How to Use This Roadmap](#how-to-use-this-roadmap)
2. [Learning Philosophy](#learning-philosophy)
3. [Core Skill Stack](#core-skill-stack)
4. [Phase-by-Phase Learning Path](#phase-by-phase-learning-path)
5. [Project Portfolio](#project-portfolio)
6. [Concept Map](#concept-map)
7. [Technology Map](#technology-map)
8. [Assessment Checklist](#assessment-checklist)
9. [References](#references)

---

## How to Use This Roadmap

1. **Start at Phase 1** even if the concepts feel simple. The goal is to learn the control loop, not the model.
2. **Build each agent yourself** before reading the reference implementation. Struggling with the loop teaches more than reading about it.
3. **Run a stress test / benchmark after every change**. Subjective demos lie; benchmarks do not.
4. **Document failures and fixes**. The most valuable notes come from things that did not work the first time.
5. **Revisit earlier phases** when new techniques unlock retroactive improvements.

---

## Learning Philosophy

- **The loop is more important than the model.** A small model with a clear loop outperforms a large model with an ambiguous one.
- **Build before framework.** Hand-roll the control loop once so you understand what LangGraph/CrewAI/AutoGen are abstracting.
- **Reliability before scale.** Make one agent robust before adding memory, multi-agent, or deployment.
- **Observability is not optional.** If you cannot see what the agent did, you cannot improve it.
- **Safety is a system property, not a prompt trick.** Guardrails, approval gates, timeouts, and audit logs are part of the architecture.

---

## Core Skill Stack

### Concepts

| Concept | Why It Matters |
|---|---|
| **Closed-loop agent architecture** | Perceive → Plan → Act → Observe → Reflect is the foundation of autonomy. |
| **Tool use / function calling** | Agents extend LLMs by calling deterministic functions and external APIs. |
| **Planning & decomposition** | Breaking goals into sub-goals lets agents handle multi-step tasks. |
| **Reflection / self-criticism** | A separate verification layer catches errors the actor missed. |
| **Memory (short & long term)** | Agents need context and the ability to learn from past runs. |
| **Retrieval-Augmented Generation (RAG)** | Grounds agent decisions in retrievable evidence. |
| **Human-in-the-loop** | Approval gates keep agents safe in high-stakes or external-action scenarios. |
| **Multi-agent coordination** | Specialized agents can solve problems too complex for one agent. |
| **Evaluation & benchmarks** | You cannot optimize what you do not measure. |
| **Governance & safety** | Guardrails, budgets, audit logs, and policy enforcement are production requirements. |

### Tools & Technologies

| Layer | Tools / Technologies |
|---|---|
| **LLM providers** | Ollama (local), OpenAI, Anthropic, Groq, Together, Cohere |
| **LLM SDKs / frameworks** | `langchain`, `langchain-core`, `langgraph`, `ollama`, `openai` |
| **Agent frameworks** | LangGraph (state machines), AutoGen (conversation), CrewAI (roles), MetaGPT (Swe teams) |
| **Model serving** | Ollama, vLLM, TGI, OpenAI-compatible APIs |
| **Search APIs** | DuckDuckGo Instant Answer, Wikipedia API, Tavily, Serper, Exa, Firecrawl |
| **Weather APIs** | Open-Meteo |
| **Vector stores / memory** | Chroma, FAISS, LanceDB, Pinecone, Weaviate, PostgreSQL + pgvector |
| **Databases** | SQLite, PostgreSQL |
| **Web / APIs** | `urllib`, `requests`, `httpx`, FastAPI, Flask |
| **Observability** | JSON traces, custom trace analyzers, LangSmith (optional), Weights & Biases |
| **Evaluation** | `pytest`, custom stress tests, public benchmarks (SWE-bench, AgentBench, WebArena, GAIA) |
| **Deployment** | FastAPI, Docker, cron, async workers (Celery/RQ), serverless functions |

### Programming & Theory

- Python (intermediate+)
- JSON / structured output parsing
- Prompt engineering and few-shot prompting
- Basic reinforcement learning and planning concepts
- Multi-agent coordination theory
- Software engineering: modularity, testing, logging, error handling

---

## Phase-by-Phase Learning Path

Each phase has a **goal**, **agent to build**, **features to add**, **concepts to learn**, and **technologies to use**.

---

### Phase 0: Foundations

**Goal**: Understand what agentic AI is and why it differs from chatbots.

**Read / watch**:
- Agentic AI survey papers (see [References](#references)).
- ReAct, Chain-of-Thought, and Toolformer papers.
- LangChain/LangGraph introductory videos.

**Do**:
- Set up a local LLM with Ollama.
- Run a few one-shot prompts and observe how an LLM behaves without tools.
- Write a one-page summary of the difference between passive response and active execution.

**Agent**: None yet — just exploration.

**Deliverable**: A short note explaining the closed-loop agent architecture.

---

### Phase 1: First Agent — Single-Tool Math Agent

**Goal**: Build the simplest possible agent that plans, calls a tool, and returns an answer.

**Agent**: `math_agent.py` — a calculator assistant.

**Features**:
- Single tool: `calculate(expression)`.
- A loop that sends the question to an LLM, parses a JSON tool call, runs the tool, and returns the result.
- Max-step guard to prevent infinite loops.

**Concepts**:
- Perceive → Plan → Select Tool → Execute → Observe → Answer.
- Structured output (JSON tool calls).
- Tool schemas.
- Guards (max steps, output cleanup).

**Technologies**:
- Ollama + `llama3.1:latest`
- `ollama` Python client or `langchain-ollama`
- Python `math` module

**Common failures to learn from**:
- Model repeats the tool call after seeing the result.
- Model returns malformed JSON.
- Model answers before calling the tool.
- Model uses wrong notation (`^` for exponent, `!` for factorial).

**Deliverable**: An interactive CLI agent that answers basic math questions correctly.

**Benchmark**: A small set of math questions; target > 90% pass rate.

---

### Phase 2: Multi-Tool Agent

**Goal**: Extend the agent from one tool to several and learn to chain them.

**Agent**: Multi-tool assistant (`math_agent.py` extended).

**New tools**:
- `read_file(path)` — read local text files.
- `get_weather(city)` — fetch live weather.
- `web_search(query)` — search the web.
- `get_current_date()` — return today's date.

**Features**:
- Tool loop: agent can call tools repeatedly until it has enough information.
- Tool registry (`TOOLS` dict).
- History of tool calls and results.
- Guard: do not repeat the same tool with the same input.
- Error handling for failed tool calls.
- Answer cleanup (strip stray prefixes).

**Concepts**:
- Multi-step planning.
- Tool chaining.
- History as short-term memory.
- Guards as reliability mechanisms.
- Deterministic tools vs. LLM reasoning.

**Technologies**:
- DuckDuckGo Instant Answer API
- Wikipedia API
- Open-Meteo API
- Python `urllib` / `requests`

**Common failures**:
- Infinite loops on simple questions.
- Redundant tool calls.
- Wrong tool selected because prompt examples are weak.
- External API rate limits and failures.

**Deliverable**: Agent can answer multi-step questions like *"What is the sum of the numbers in data/numbers.txt?"* and *"What is the temperature in Paris plus 10?"*.

**Benchmark**: 30+ questions covering math, file reads, weather, web search, date, multi-step, and non-math.

---

### Phase 3: Reliability — Reflection, Retry, and Tracing

**Goal**: Add verification, self-correction, and observability.

**Agent**: Same multi-tool agent, now with reflection and tracing.

**Features**:
- **Reflection phase**: a critic prompt verifies the final answer against the tool history.
- **Retry loop**: if the critic flags an error, retry with feedback.
- **Tracing**: every run writes a JSON trace with question, tool calls, results, timings, guard events, and final answer.
- **Trace analyzer**: a script that reads traces and reports latency, tool usage, guards, errors, and unverified answers.

**Concepts**:
- Reflection / self-criticism.
- Retry with feedback.
- Structured logging / observability.
- Auditing and reproducibility.
- Measuring success rates, latency, and failure modes.

**Technologies**:
- JSON trace files
- A custom `tracer.py` module
- `trace_analyzer.py`

**Common failures**:
- Critic is too harsh or too lenient.
- Critic hallucinates training cutoffs for live tool results.
- Retry loops never terminate.
- Traces become too noisy; need to know what to record.

**Deliverable**: Agent passes the full stress test with reflection enabled and the trace analyzer produces useful reports.

**Benchmark**: Same 30+ suite, now with reflection and retry; target 34/34 or explain any failure.

---

### Phase 4: Frameworks — Rebuild in LangGraph

**Goal**: Learn what agent frameworks abstract by porting the hand-rolled agent.

**Agent**: `langgraph_agent.py` — the same agent rebuilt as a LangGraph state machine.

**Features**:
- `AgentState` as a `TypedDict`.
- Nodes: `plan_node`, `execute_node`, `answer_node`, `reflect_node`.
- Conditional edges: `route_after_plan`, `route_after_reflect`.
- Reuse tools and prompts from the hand-rolled version.
- Stress test parity: same 34/34 pass rate.

**Concepts**:
- State machines vs. hand-rolled loops.
- Nodes and edges.
- Shared state.
- Framework overhead vs. value.

**Technologies**:
- `langgraph`
- `langchain-core`
- `langgraph.checkpoint.memory.MemorySaver`

**Common failures**:
- State updates are incomplete or wrong keys are used.
- Routing conditions miss edge cases.
- Framework behavior differs subtly from hand-rolled version.

**Deliverable**: LangGraph agent passes the same benchmark as the hand-rolled agent.

**Benchmark**: Direct comparison table of hand-rolled vs. LangGraph latency and pass rate.

---

### Phase 5: Human-in-the-Loop & Safety

**Goal**: Add approval gates before high-risk or external actions.

**Agent**: LangGraph agent with interrupt-based approval.

**Features**:
- `APPROVAL_REQUIRED_TOOLS` set (e.g., `web_search`, `get_weather`).
- `interrupt()` inside `execute_node` to pause before external tools.
- Outer loop detects `__interrupt__` events, prompts the user, and resumes with `Command(resume=response)`.
- `auto_approve` mode for headless runs.
- Clean refusal message when approval is denied.

**Concepts**:
- Human-in-the-loop (HITL).
- Governance and guardrails.
- Native framework interrupts vs. hand-rolled checkpoints.
- Least-privilege tool use.

**Technologies**:
- `langgraph.types.interrupt`
- `langgraph.types.Command`

**Common failures**:
- Manual checkpoint recurses infinitely before `input()` is called.
- Interrupt not detected in stream events.
- Resume value not passed correctly to the interrupted node.

**Deliverable**: Interactive LangGraph agent that pauses before external tools; stress test passes with `--auto-approve`.

**Benchmark**: Manual approve/deny tests + automated stress test.

---

### Phase 6: Memory & RAG

**Goal**: Give the agent context beyond the current conversation.

**Agent**: Agent with persistent memory and retrieval-augmented generation.

**Features**:
- **Short-term memory improvements**: summarize long tool histories; prune old context.
- **Episodic memory**: store past runs, failures, and user corrections.
- **Semantic memory**: store facts in a vector store and retrieve relevant ones.
- **RAG**: retrieve from a local knowledge base before (or instead of) web search.

**Concepts**:
- Short-term vs. long-term memory.
- Episodic vs. semantic memory.
- Embedding-based retrieval.
- Context windows and token budgets.

**Technologies**:
- Chroma, FAISS, LanceDB, or PostgreSQL + pgvector
- Sentence-transformers or OpenAI embeddings
- SQLite for simple episodic memory

**Common failures**:
- Retrieved context is irrelevant (poor chunking or embeddings).
- Memory pollutes the prompt with outdated information.
- Vector DB hygiene is neglected.

**Deliverable**: Agent can answer questions using its own stored knowledge and remember corrections across sessions.

**Benchmark**: Questions that require retrieval from memory; questions that previously failed and now pass with feedback.

---

### Phase 7: Multi-Agent Systems

**Goal**: Learn how multiple specialized agents coordinate.

**Agents**: A team of two or more agents with distinct roles.

**Features**:
- **Planner + Executor**: one agent plans, another executes tools.
- **Researcher + Writer**: one searches, another synthesizes.
- **Critic + Actor**: critic reviews actor's output.
- Role definitions and message filtering.
- Task decomposition and handoff.

**Concepts**:
- Vertical vs. horizontal coordination.
- Role-based multi-agent teams.
- Manager-worker hierarchies.
- Communication overhead and consensus.

**Technologies**:
- LangGraph multi-actor graphs
- AutoGen
- CrewAI
- MetaGPT

**Common failures**:
- Agents talk in circles instead of making progress.
- Role boundaries blur.
- Coordination overhead exceeds value.
- Error propagation across agents.

**Deliverable**: A small multi-agent team that outperforms the single agent on at least one decomposed task.

**Benchmark**: Same task solved by single agent vs. multi-agent team; compare correctness, latency, and cost.

---

### Phase 8: Evaluation & Benchmarks

**Goal**: Move from "looks right" to measured performance.

**Activities**:
- Add expected answers to traces.
- Compute pass/fail rates automatically.
- Add adversarial and edge-case questions.
- Measure latency, cost, and retry rates.
- Compare model/provider performance.
- Explore public benchmarks: SWE-bench, AgentBench, WebArena, GAIA.

**Concepts**:
- End-to-end evaluation vs. component evaluation.
- Robustness, security, latency, cost metrics.
- Red-teaming and adversarial testing.
- Regression testing.

**Technologies**:
- `pytest`
- Custom eval harness
- LangSmith / Weights & Biases (optional)
- Public benchmark datasets

**Deliverable**: An evaluation harness that reports pass rate, latency, cost, and failure categories across runs.

**Benchmark**: Full suite + adversarial cases + comparison across models/providers.

---

### Phase 9: Production & Deployment

**Goal**: Ship an agent as a usable service.

**Features**:
- FastAPI or CLI wrapper around `run_agent`.
- Async handling for multiple concurrent requests.
- Persistent checkpointing across restarts.
- Authentication / authorization for tool access.
- Rate limiting and cost budgets.
- Deployment: Docker, cloud, or local server.
- Monitoring and alerting.

**Concepts**:
- API design for agents.
- Concurrency and state management.
- Security boundaries.
- Cost control.
- Observability in production.

**Technologies**:
- FastAPI / Uvicorn
- Docker
- Redis or PostgreSQL for queues/state
- Cloud provider (optional)

**Deliverable**: A deployed agent service with a real use case.

**Benchmark**: Real users or a real task; measure uptime, latency, errors, and user satisfaction.

---

## Project Portfolio

As you move through the phases, build these concrete agents:

1. **Math Agent** (Phase 1) — single tool, closed loop.
2. **File + Math Agent** (Phase 2) — multi-tool chaining.
3. **Research Assistant** (Phase 2–3) — search + summarize + reflect.
4. **Weather + Calendar Agent** (Phase 2–3) — live data + multi-step reasoning.
5. **Self-Correcting Agent** (Phase 3) — reflection + retry.
6. **LangGraph Port** (Phase 4) — same agent in a framework.
7. **Approved-Agent** (Phase 5) — human approval gates.
8. **Remembering Agent** (Phase 6) — persistent memory and RAG.
9. **Agent Team** (Phase 7) — planner + executor + critic.
10. **Production Service** (Phase 9) — API-backed agent for a real task.

---

## Concept Map

```
                    ┌─────────────┐
                    │   User Goal  │
                    └──────┬──────┘
                           ▼
                    ┌─────────────┐
                    │  Perception  │
                    └──────┬──────┘
                           ▼
                    ┌─────────────┐
                    │   Planning   │ ──► Chain-of-thought, sub-goals
                    └──────┬──────┘
                           ▼
                    ┌─────────────┐
                    │ Tool Selection│
                    └──────┬──────┘
                           ▼
                    ┌─────────────┐
                    │   Execution  │ ◄── Human approval gate (Phase 5)
                    └──────┬──────┘
                           ▼
                    ┌─────────────┐
                    │  Observation │
                    └──────┬──────┘
                           ▼
                    ┌─────────────┐
                    │  Reflection  │ ──► Critic, retry
                    └──────┬──────┘
                           ▼
                    ┌─────────────┐
                    │    Memory    │ ──► Short-term, episodic, semantic
                    └──────┬──────┘
                           ▼
                    ┌─────────────┐
                    │ Final Answer │
                    └─────────────┘
```

---

## Technology Map

| Phase | Technologies |
|---|---|
| 1 | Ollama, Python, `math`, JSON parsing |
| 2 | DuckDuckGo API, Wikipedia API, Open-Meteo, `urllib`/`requests` |
| 3 | Custom tracer, JSON traces, trace analyzer, reflection prompts |
| 4 | LangGraph, LangChain, `TypedDict`, `MemorySaver` |
| 5 | `langgraph.types.interrupt`, `Command`, approval policies |
| 6 | Chroma / FAISS / LanceDB, embeddings, SQLite |
| 7 | LangGraph multi-actor, AutoGen, CrewAI, MetaGPT |
| 8 | `pytest`, eval harness, public benchmarks |
| 9 | FastAPI, Docker, PostgreSQL/Redis, cloud deployment |

---

## Assessment Checklist

Use this checklist to track mastery.

### Phase 1 — First Agent
- [ ] Built a single-tool agent from scratch.
- [ ] Parses structured tool calls from LLM output.
- [ ] Has a max-step guard.
- [ ] Passes a small math benchmark.

### Phase 2 — Multi-Tool
- [ ] Agent can select among multiple tools.
- [ ] Tool chaining works for multi-step questions.
- [ ] Repeated tool calls are blocked.
- [ ] Tool errors are handled gracefully.

### Phase 3 — Reliability
- [ ] Reflection/critic phase added.
- [ ] Retry loop with feedback works.
- [ ] Every run writes a structured trace.
- [ ] Trace analyzer produces useful metrics.

### Phase 4 — Frameworks
- [ ] Agent rebuilt in LangGraph.
- [ ] LangGraph version matches hand-rolled pass rate.
- [ ] State machine is documented with a diagram.

### Phase 5 — Human-in-the-Loop
- [x] Approval gate before external tools.
- [x] Manual approve/deny works.
- [x] Auto-approve mode works for benchmarks.
- [x] Denial returns a clean, safe message.

### Phase 6 — Memory & RAG
- [x] Short-term history is summarized/pruned.
- [x] Past failures are stored and reused.
- [x] Agent retrieves from a vector store.

### Phase 7 — Multi-Agent
- [x] Built a team of 2+ agents.
- [x] Roles are clearly separated.
- [ ] Team beats single agent on at least one task. (Tested honestly: currently at parity, 36/36 both. See development-log.md Experiment 15 for what was tried and why it didn't show a win with one shared local model.)

### Phase 8 — Evaluation
- [x] Automated pass/fail reporting. (Already true since early phases — `stress_test.py`.)
- [x] Adversarial / edge cases added. (Path traversal, division by zero, large factorial — all pass. A prompt-injection probe is included but deliberately unscored: found genuinely unreliable after mitigation, tracked honestly rather than falsely certified fixed. See development-log.md Experiment 16.)
- [x] Compared at least two models or providers. (`llama3.1` vs `gpt-4o-mini` on the identical 39-question suite — `gpt-4o-mini` ~2x faster, 38/39 vs 39/39, with two nuanced findings that don't reduce to "which one wins." See development-log.md Experiment 17.)

### Phase 9 — Production
- [x] Agent exposed via API or CLI. (`api.py` — FastAPI service over all three implementations, plus the pre-existing CLIs.)
- [x] Handles concurrent requests. (Measured: 3 concurrent requests in 5.42s wall vs 14.24s summed. Required fixing a real concurrency bug — the token accumulator was process-global and would have corrupted per-request tallies; now a ContextVar.)
- [x] Has authentication and cost guards. (`X-API-Key` with `compare_digest`; a shared, lock-protected spend budget returning 402 when exhausted. Both tested, including rejection paths.)
- [ ] Deployed and monitored. (Containerized and verified running — image builds, container answers real questions against host Ollama, `/health` + Docker HEALTHCHECK work. Not deployed to any remote host, and no real monitoring/alerting stack beyond the health endpoint and JSON traces.)

---

## References

### Key Survey Papers
- Liu et al. (2025) — agent architecture surveys.
- Xu et al. (2026) — planning, reflection, and tool use.
- Masterman et al. (2024) — tool use and closed-loop agents.
- Sapkota et al. (2025) — multi-agent orchestration.
- Swamy et al. (2025) — observability and governance.

### Frameworks to Explore
- **LangGraph** — state-machine agents, interrupts, persistence, multi-actor.
- **AutoGen** — conversational multi-agent patterns.
- **CrewAI** — role-based agent teams.
- **MetaGPT** — software-engineering multi-agent teams.

### Learning Materials
- ReAct paper: *Reasoning + Acting with Language Models*.
- Chain-of-Thought paper.
- Toolformer paper.
- LangGraph docs: interrupts, persistence, multi-agent.

---

> **Last updated**: 2026-08-29  
> **Status**: Living document — update after every experiment.
