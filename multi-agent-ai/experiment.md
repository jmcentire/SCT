# Experimental Protocol: 4-Architecture Comparison
# Substrate-Independent Dysfunction Study — Experiment 3+

## Overview

A 2×4 factorial experiment comparing four multi-agent architectures on two
tasks of different complexity. The independent variable is coordination
topology. The dependent variables are task completion, cost, coordination
overhead, and dysfunction indicators.

## Hypotheses

**H1 (Architectural):** Dysfunction patterns (bikeshedding, decision
paralysis, governance conflict, rework loops) correlate with coordination
topology, not agent capability. The same model under different architectures
produces different dysfunction profiles.

**H2 (Hierarchy-specific):** Gated-review hierarchies produce more
dysfunction than trust-based hierarchies, which produce more than
stigmergic coordination, which produces more than a single agent.
Predicted ordering: Org > Hi-Trust > Emergence > Unary.

**H3 (Task-complexity interaction):** The coordination overhead penalty
is proportionally larger for simple tasks (where the overhead:production
ratio is higher) and partially masked for complex tasks.

---

## Independent Variable: Architecture (4 levels)

### A1: Unary (Single Agent)

Claude Code operating in a standard session. No coordination, no
delegation, no review gates. Full context maintained in one conversation.

**Implementation:** Claude Code CLI invoked with a task prompt and
instrumentation wrapper that logs timestamps, token usage, and tool calls.

**What it models:** The "just do it" baseline. Tests whether a single
agent can accomplish the task without coordination overhead.

### A2: Org Swarm (Gated-Review Hierarchy)

The existing swarm at `/Users/jmcentire/WanderRepos/swarm/`. Hierarchical
pipeline: diagnose → decompose → architect → architect_review → locate →
execute → test → verify → review. Gated evaluation at multiple stages.
Six anti-dysfunction mechanisms active.

**Implementation:** Existing swarm, new task input. No modifications to
pipeline logic. Add experiment-level instrumentation (external log wrapper).

**What it models:** A traditional organization with division of labor,
specialization, and quality gates. The architecture that produced the
documented dysfunction.

### A3: Hi-Trust Hierarchy (Delegation Tree)

A recursive delegation tree. The root agent receives the task, decomposes
it into subtasks, and delegates each to a child agent with full trust.
Child agents either execute (if the subtask is small enough) or
recursively decompose and delegate. Leaf nodes produce code artifacts.
Assembly flows back up the tree: each parent collects deliverables from
children and integrates them into a coherent product, returning the
assembled result to its own parent.

**Key properties:**
- Hierarchical decomposition (like Org Swarm)
- NO review gates — agents trust subordinates' output
- NO lateral communication — agents only talk to parent/children
- Assembly happens on the way UP, not through separate review stages
- Decomposition depth is adaptive (agent decides when task is small enough)

**What it models:** Auftragstaktik / mission-type delegation. A
high-functioning organization where managers set objectives and trust
execution. Tests whether the dysfunction comes from the hierarchy itself
or from the review/gating mechanism within the hierarchy.

**Predicted failure modes:**
- Integration failures at assembly (components don't fit together)
- Drift in interpretation (child's understanding diverges from parent's intent)
- No error correction (trust means bugs propagate upward uncaught)

**Architecture spec (for implementation):**

```
Agent Protocol:
  receive(task, context) →
    if is_leaf_task(task):
      code = execute(task, context)
      log(task_received, decision_made="execute directly", work_produced=code)
      return code
    else:
      subtasks = decompose(task, context)
      log(task_received, decision_made="decompose", delegated_to=subtasks)
      results = []
      for subtask in subtasks:
        child = spawn_agent()
        result = child.receive(subtask, context + subtask_context)
        results.append(result)
      assembled = assemble(results, task, context)
      log(assembled_from=results, work_produced=assembled)
      return assembled

Constraints:
  - Max tree depth: 4 levels
  - Max children per node: 5
  - Leaf task heuristic: agent judges task completable in ~200 lines of code
  - No agent reads another agent's log or intermediate state
  - No review, approval, or rejection at any level
  - Assembly = integration, not evaluation
```

### A4: Emergence Swarm (Stigmergic Coordination)

Agents coordinate through a shared environment rather than through direct
communication. A shared workspace (repository + task board) is the only
coordination medium. Agents observe the environment, claim work based on
self-assessed competence, produce artifacts into the shared space, and
other agents build on those artifacts. No one assigns work. No one
reviews work. The environment IS the coordination mechanism.

**Key properties:**
- NO hierarchy — all agents are peers
- NO direct agent-to-agent communication
- Shared environment: a repository + a task/signal board
- Work claiming: agents read the board, claim tasks they're suited for
- Artifact-based coordination: agents see what others produced by reading the repo
- Convergence through building, not consensus

**What it models:** Open-source development / stigmergic coordination.
Wikipedia editing. Ant colony optimization. Tests whether removing ALL
hierarchical structure (not just gates) eliminates the documented
dysfunction patterns.

**Predicted failure modes:**
- Duplication of effort (two agents claim the same work)
- Incoherence (no one ensures components fit together)
- Starvation (hard/unclear tasks never get claimed)
- Drift (no authority to course-correct)

**Architecture spec (for implementation):**

```
Shared Environment:
  - task_board: list of {task_id, description, status: [open|claimed|done], claimed_by}
  - repository: shared filesystem where agents read/write code
  - signal_log: append-only log of agent observations about the repo state

Agent Protocol (all agents identical):
  loop:
    scan(task_board) → find open tasks
    scan(repository) → observe current state of codebase
    if suitable_task found:
      claim(task) → atomically set status=claimed, claimed_by=self
      work = execute(task, repository_context)
      commit(work) → write to repository
      update(task_board, status=done)
      log(signal_log, "completed {task_id}, produced {summary}")
    elif integration_needed:
      # Agent notices components exist but aren't wired together
      create_task(task_board, "integrate X and Y")
      claim and execute it
    else:
      observe(signal_log) → read what others have done
      # Agent may notice gaps and create new tasks

Constraints:
  - N agents (suggest 4-6 for simple task, 6-10 for complex)
  - All agents use the same model and base prompt
  - No agent can modify another agent's code without going through the repo
  - No messaging between agents — only the shared environment
  - Run terminates when all tasks on board are done OR budget exhausted
  - Task board is seeded with the initial high-level task (single entry)
  - Agents create subtasks themselves as they discover what's needed
```

---

## Tasks

### Simple Task

**Prompt (given to all four architectures identically):**

> "I need a command-line tool that lets me manage a personal book library —
> add books, find them, and track what I've read."

**Acceptance criteria (evaluated post-run, not provided to agents):**
1. Can add a book (title, author, at minimum)
2. Can search/find books by some criteria
3. Can mark a book as read/unread
4. Can list books (all, read, unread)
5. Data persists between invocations
6. Runs without errors on the test machine

**Language:** Agent's choice (part of the design-decision measurement).

**Why this task:** Simple enough that a single agent can do it in one
session. Complex enough that it requires data model decisions, storage
decisions, CLI design decisions, and error handling decisions. The
vagueness is deliberate — every architecture must interpret "manage,"
"find," and "track."

### Complex Task

**Prompt (given to all four architectures identically):**

> "Build a notification service that handles email, SMS, and push
> notifications. Users should be able to set their preferences for which
> channels they want. There should be rate limiting so we don't spam
> people. We need templates for common notification types. And we need
> to track whether notifications were delivered."

**Acceptance criteria (evaluated post-run, not provided to agents):**
1. Supports 3 channels (email, SMS, push) — at least stub/mock implementations
2. User preference storage and lookup
3. Rate limiting logic (per-user, per-channel)
4. Template system (at least 2 templates)
5. Delivery tracking (sent, delivered, failed states)
6. API or CLI interface to send a notification
7. Tests exist and pass

**Language:** Agent's choice.

**Why this task:** Requires decomposition into multiple components
(channel adapters, preference engine, rate limiter, template renderer,
delivery tracker, API layer). Components must coordinate. Complex enough
to stress multi-agent architectures. Simple enough to be completable
within budget.

---

## Instrumentation

### Principle: Observe, don't interfere.

All instrumentation is append-only and external. No agent reads the
instrumentation log. No agent's behavior is modified by the logging.

### Metrics collected:

**Per-run:**
- Total wall-clock time
- Total cost ($)
- Total tokens (input + output)
- Task completion (binary: passes acceptance criteria or not)
- Acceptance criteria score (0-7 for complex, 0-6 for simple)

**Per-agent-event (logged automatically):**
- Timestamp
- Agent ID
- Event type: task_received | decision_made | work_produced | delegated_to | assembled_from | claimed_task | created_task
- Content summary (≤100 words)
- Tokens consumed for this event

**Derived metrics (computed post-run):**
- Coordination overhead ratio: tokens spent on coordination / tokens spent on production
- Rework rate: number of times the same component was re-produced
- Decision paralysis count: number of escalations or human-intervention requests
- Backward transitions: number of times work product moved to an earlier stage
- Integration failure rate: percentage of assembly steps that required rework
- Unique dysfunction indicators per architecture

### Log format:

```jsonl
{"ts": "ISO8601", "agent": "agent_id", "event": "event_type", "summary": "...", "tokens": N}
```

One .jsonl file per run. Analysis reads these post-hoc.

---

## Budget

Total experimental budget: $250

**Estimated per-run costs:**
| Architecture | Simple Task | Complex Task |
|---|---|---|
| Unary | $1-3 | $5-15 |
| Org Swarm | $1-5 | $25-60 |
| Hi-Trust | $2-5 | $10-30 |
| Emergence | $3-8 | $15-40 |

**With 2 replications per cell:** 8 conditions × 2 runs = 16 runs
**Estimated total:** $100-250

If budget allows, 3 replications: 24 runs, $150-350.
Start with 1 run per cell (8 runs, ~$60-160), evaluate, then replicate.

---

## Run Protocol

1. Create fresh directory for each run: `experiments/{architecture}/{task}/{run_id}/`
2. Seed the directory with the task prompt (identical across architectures)
3. Start the instrumentation logger
4. Invoke the architecture
5. Do NOT intervene (no human input during run)
6. Run terminates on: task completion, budget exhaustion, or 4-hour timeout
7. Collect: final code output, instrumentation log, total cost, wall time
8. Evaluate against acceptance criteria (human evaluation, blind to architecture)

**Critical:** The human evaluator should evaluate outputs blind — not knowing
which architecture produced which output. Randomize the order.

---

## Analysis Plan

### Primary analysis:
- 4×2 table of completion rates
- Cost comparison across architectures (box plots if N≥3)
- Coordination overhead ratio comparison
- Dysfunction indicator counts per architecture

### Secondary analysis:
- Qualitative comparison of failure modes across architectures
- Token flow analysis (where did the tokens go?)
- Decision tree analysis (how did each architecture decompose the complex task?)
- Integration quality comparison (do components fit together?)

### Key comparisons:
- Unary vs. Org Swarm: does adding coordination help or hurt?
- Org Swarm vs. Hi-Trust: is the dysfunction from hierarchy or from gates?
- Hi-Trust vs. Emergence: is the dysfunction from hierarchy or from any structure?
- Emergence vs. Unary: is stigmergic coordination better than none?

---

## What Needs Building

1. **Instrumentation wrapper** — generic logger that wraps any architecture
2. **Hi-Trust Hierarchy agent** — recursive decompose-delegate-assemble tree
3. **Emergence Swarm agent** — shared-environment stigmergic coordinator
4. **Evaluation harness** — acceptance criteria checker + blind evaluation protocol
5. **Analysis scripts** — parse .jsonl logs, compute metrics, generate tables

## Implementation Order

1. Instrumentation wrapper (shared across all architectures)
2. Hi-Trust Hierarchy (simpler, ~1 day)
3. Emergence Swarm (more complex, ~2-3 days)
4. Run simple task across all 4 (1 day)
5. Evaluate + adjust
6. Run complex task across all 4 (1-2 days)
7. Replicate (1-2 days)
8. Analyze (1 day)
