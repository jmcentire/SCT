# Stigmergic Agent Architecture for Organizational Knowledge Systems

## Technical Architecture Document

---

## 1. Problem Statement

Organizational knowledge systems fail for a structural reason: they impose human taxonomies on inhuman problems.

Information in an organization does not respect departmental boundaries. A bug in an onboarding workflow simultaneously concerns engineering, support, sales, and product. Current systems — wikis, ticket trackers, search tools, AI assistants — require a human to know which silo to query, what question to ask, and when to ask it. This fails in three predictable ways:

1. **The routing problem.** Humans file information where the org chart says it belongs. The org chart is political, not logical. Information ends up where it can be found by people who already know it exists.

2. **The latency problem.** Information sits inert until a human queries it. By the time someone asks the right question, the damage — duplicated work, conflicting decisions, missed context — is already done.

3. **The dimensionality problem.** A single signal (a ticket, a PR, a Slack message) has relevance along dimensions that no human taxonomy can capture. Current systems force categorical assignment. Reality is continuous.

These are not human problems that need faster humans. They are inhuman problems that require a fundamentally different architecture.

---

## 2. Core Principle: Stigmergy Over Orchestration

Stigmergy is coordination through shared environment rather than direct communication. Ants don't talk to each other. They modify pheromone trails. The colony's behavior emerges from simple local rules applied at scale.

This architecture applies stigmergic principles to organizational knowledge:

- **No central orchestrator.** No coordinator assigns tasks or routes signals. Agents act on local state.
- **Shared state as communication.** Agents read from and write to shared context stores. The stores *are* the coordination mechanism.
- **Emergent structure.** Contexts (knowledge domains) are not designed. They form from signal clustering and dissolve when signals stop arriving.
- **Simple rules, complex behavior.** Each agent follows a small set of deterministic rules. The system's intelligence is in the interaction patterns, not in any individual agent.

---

## 3. One Pattern

There is one architectural pattern. It recurs at every level of the system.

An **agent** is bound to a **context**. It ingests **signals**. It evaluates them against what it knows. It produces **assessments**. When its context becomes too broad, it **forks**. When its context overlaps with a neighbor's, they **merge**. When its context goes quiet, it **decays**. It follows **rules** and obeys **constraints**.

This is the entire architecture.

An agent bound to TICKET-7372 and an agent bound to "Jeremy's attention" and an agent bound to "output compliance" are the same thing. They differ in what context they hold, what signals they watch, and what constraints they obey — not in kind. There are no special classes. There is one pattern, applied recursively.

- An agent watching a codebase context and an agent watching a person context differ only in their binding. The codebase agent asks: "does this signal relate to my code?" The person agent asks: "does this signal matter to my human?" Same mechanism. Different input.

- An agent filtering output for compliance and an agent filtering signals for relevance differ only in where they sit in the pipeline and what rules they enforce. The compliance agent kills messages that violate constraints. The relevance agent kills signals that fall below threshold. Same mechanism. Different constraints.

- An agent tracking duplicate tickets and an agent tracking decision propagation differ only in what patterns they've learned to recognize. Neither was born as a "type." Both emerged because the system needed that competency in that context.

### How Agent Competencies Emerge

Agent competencies are not enumerated at design time. They are identified by **hypervisor agents** — agents whose context is the system's own performance.

A hypervisor agent watches the signal flow. It sees: signals are arriving in Context A that contradict established decisions, and no agent is catching them. It identifies the gap. It spawns an agent whose initial training emphasizes contradiction detection within Context A's domain.

Starting points exist. At deployment, seed agents are initialized with broad competencies: similarity detection, temporal analysis, cross-context propagation. These are not "types" — they are initial conditions. As the system processes signals and receives feedback, competencies specialize, split, merge, and die just like contexts do. A seed agent that started as a general "conflict detector" may fork into one agent that specializes in PR conflicts and another that specializes in decision contradictions — or it may decay entirely if the context it serves doesn't produce conflicts.

The hypervisor agents are themselves subject to the same lifecycle. A hypervisor watching system performance in the engineering domain may fork when the engineering domain splits. A hypervisor that monitors a context that goes quiet decays with it.

There is no level of the system that is exempt from the pattern.

---

## 4. Primitives

### 4.1 Signal

The atomic unit of information entering the system.

```
Signal {
  id:          UUID
  content:     string | structured_data
  source:      enum { slack, linear, github, docs, email, deploy, ... }
  channel:     string              // origin channel, repo, board, etc.
  author:      string              // human or system that produced it
  timestamp:   datetime
  embeddings:  map<strategy, vector[d]>  // multiple embedding representations
  metadata:    map<string, any>    // source-specific structured fields
}
```

Signals are immutable. Once ingested, a signal is never modified. Derived state lives in contexts.

### 4.2 Context

A knowledge domain backed by a vector store. Contexts are not predefined by org chart or product area. They emerge from signal clustering.

```
Context {
  id:              UUID
  vector_store:    RAG              // embeddings + retrievable chunks
  active_agents:   set<Agent>       // agents currently bound to this context
  relevance_decay: float            // exponential decay rate (λ)
  last_signal:     datetime         // timestamp of most recent relevant signal
  energy:          float            // current "aliveness" — f(signal_rate, recency)
  parent:          Context | null   // if forked from another context
  neighbors:       set<Context>     // topologically adjacent contexts
  business_weight: float            // learned importance to the organization
}
```

**Energy** is the key lifecycle metric:

```
energy(t) = Σ(signal_weights) * e^(-λ * (t - last_signal))
```

When energy drops below a threshold, the context enters decay. When it hits zero, agents are released and the context archives.

**Business weight** is a learned scalar reflecting organizational importance. Derived from: signal frequency, seniority of engaged people, presence in strategic documents vs. casual conversation, and downstream financial impact. A context tracking a bug costing customers $500/day outranks a context tracking the lunch debate. The system has the data to make this judgment.

**Contexts are topological.** They have neighbors — contexts with shared edges in signal space. Two contexts are adjacent when they process overlapping signals, serve overlapping consumers, or share causal relationships (decisions in one affect work in the other). This topology is not declared. It is discovered from signal flow and continuously updated.

### 4.3 Agent

A worker bound to one or more contexts. All agents follow the same pattern. They differ in their context bindings, their learned competencies, and their constraints.

```
Agent {
  id:              UUID
  contexts:        map<Context, affinity_weight>  // weighted context bindings
  competencies:    learned                         // what it has become good at
  confidence:      float                           // historical accuracy score
  rules:           RuleSet                         // constraints governing behavior
  weights:         map<Domain, float>              // how much it cares about different concerns
  state:           map<string, any>                // working memory
}
```

**Affinity weight** is a continuous value in [0, 1] representing how strongly an agent is bound to a given context.

**Weights** are the agent's own priorities — how much it cares about different domains of concern. An agent bound to the "Jeremy" context weights technical decisions heavily and sales matters lightly. An agent bound to the "Q1 Revenue" context does the inverse. These weights are brought into consensus (Section 5.5) — they are not decorative. They determine how much an agent's vote counts on a given topic.

### 4.4 Familiarity Score

The routing metric. Given a signal and a context, the familiarity score determines how "close" the signal is to that context's knowledge domain.

```
familiarity(signal, context) → float [0, 1]

Components:
  - embedding_similarity:  cosine(signal.embedding, context.centroid)
  - keyword_overlap:       jaccard(signal.terms, context.terms)  
  - source_affinity:       learned weight per source-context pair
  - temporal_proximity:    recency of related signals in this context
  - author_affinity:       how often this author's signals land in this context

familiarity = weighted_sum(components, learned_weights)
```

Familiarity is not categorical. A signal doesn't "belong" to a context. It has a continuous degree of relevance to every active context simultaneously.

---

## 5. Signal Flow

### 5.1 Ingestion

Signals arrive continuously from all connected systems via adapters (webhooks, polling, event streams). Each adapter normalizes the source data into the Signal primitive and computes embeddings.

```
[Slack] ──→ [Adapter] ──→ Signal ──→ Ingestion Queue
[GitHub] ─→ [Adapter] ──→ Signal ──→ Ingestion Queue
[Linear] ─→ [Adapter] ──→ Signal ──→ Ingestion Queue
[Docs]   ─→ [Adapter] ──→ Signal ──→ Ingestion Queue
```

The ingestion queue is the only centralized component. It is a FIFO buffer with priority weighting, not a coordinator. It does not route. It holds signals until agents claim them.

### 5.2 Classification

Every active agent independently computes familiarity scores for every incoming signal against its bound contexts. This is the **Deterministic Parallel Routing** principle:

> Given identical inputs (signal + context state), every agent computing familiarity for the same signal-context pair arrives at the same score. No coordination required. No voting on routing. The math is the routing.

Each agent solves the whole routing problem but only acts on the slice relevant to its contexts.

```
for each signal in queue:
  for each active_context:
    score = familiarity(signal, context)
    if score > context.relevance_threshold:
      context.ingest(signal, score)
      context.energy += score * signal_weight
```

### 5.3 Active Inquiry

Classification is not passive pattern matching. When a signal arrives and scores above threshold for a context, the receiving agents don't just file it. They *converse* with their knowledge base.

A message in Slack carries context: who said it, when, what project they're working on, what their recent activity looks like, what words they used, what implicit knowledge they assumed. The system sees the signal and asks itself: *What do I know about this?*

It decomposes the signal — people, projects, tickets, keywords, sentiment, implied questions — and routes each dimension to the contexts that would care. Each context is prompted: *Here's what just happened. Anything relevant I should know or bring up?*

```
for each context where familiarity(signal) > inquiry_threshold:
  response = context.inquire(signal)
  // "Nothing relevant."            → move on
  // "This relates to TICKET-X."    → Assessment: surface
  // "This contradicts decision Y." → Assessment: conflict
```

This is the inversion. Current systems wait for humans to ask questions. This system asks its own contexts questions — using statements as prompts. An LLM doesn't need a question mark to produce insight. A signal and a context are sufficient.

The system functions like a new hire onboarding: reading the code, the history, the docs, the channels, the pitch deck. Watching conversations. Listening. Building understanding. Then, when something comes up, connecting it to everything it's absorbed — not because someone asked, but because the connection exists and the system noticed.

### 5.4 Disposition

Once a signal is ingested and inquired upon, the context's bound agents evaluate it according to whatever competencies they've developed. Each agent produces an **Assessment**:

```
Assessment {
  agent:       Agent
  signal:      Signal
  context:     Context
  action:      enum { store, surface, replicate, block, ignore }
  target:      User | Context | null
  confidence:  float [0, 1]
  domain:      string           // what domain of concern this assessment addresses
  reasoning:   string           // natural language explanation
}
```

The `domain` field matters for consensus. An assessment about a technical conflict and an assessment about a customer impact both address the same signal but from different domains. The agents voting on each bring different weights.

### 5.5 Consensus

When multiple agents produce assessments for the same signal in the same context, the system resolves via weighted vote — but the weights are not static system parameters. Each participating agent contributes its own domain weights.

```
consensus(assessments) → Action

For each assessment:
  vote_weight = assessment.confidence 
              * assessment.agent.confidence 
              * assessment.agent.weights[assessment.domain]

For each possible action:
  action_weight = Σ(vote_weights of assessments proposing this action)

If max(action_weight) > consensus_threshold:
  execute(winning_action)
Else if max(action_weight) > uncertainty_threshold:
  store + flag_for_review
Else:
  store silently
```

This means: an agent bound to the "Jeremy" context that doesn't care about sales will contribute very little vote weight to a sales-domain assessment, even if it's confident. An agent bound to "Platform Architecture" that cares deeply about technical decisions will dominate the vote on technical assessments.

The consensus algorithm doesn't decide what matters. The agents tell it what they care about.

Losing agents update their confidence scores based on downstream feedback (did the action produce a positive or negative signal from the user?).

### 5.6 Output Constraints

When consensus resolves to `surface` or `block`, the proposed output passes through agents whose context is compliance — agents bound to the constraint domain rather than a knowledge domain.

These agents follow the same pattern as every other agent. They evaluate the signal (in this case, the proposed outbound message) against their context (organizational compliance rules, sensitivity categories, legal constraints). They produce assessments: pass, redact, or kill.

The principle: **the system can know. The system cannot say.**

This is functionally identical to any access control mechanism. The system has the information. The gate determines what crosses the boundary. The difference is that the gate is not an ACL table keyed on user identity — it is an agent evaluating content against constraints. Same pattern. Different context.

**Kill categories** (message is null-routed, never delivered):
- PII in any form
- Financial specifics (compensation, revenue figures, deal terms)
- HR-sensitive topics (termination, PIP, performance evaluation, hiring decisions, equity)
- Legal-sensitive topics (active litigation, regulatory findings)
- Credentials, tokens, secrets

**Redaction categories** (message is delivered with sensitive content removed):
- References to people by role instead of name in sensitive contexts
- Aggregate data where specifics are restricted but trends are not

A constraint agent does not reason about *why* a message should be killed. It pattern-matches and acts. This makes it fast, deterministic, and resistant to prompt injection or adversarial manipulation of upstream agents. An agent that has been corrupted cannot talk its way past a constraint agent. Constraint agents don't negotiate.

When a message is killed:
1. The originating agent is notified: "Output blocked — [category]"
2. The agent may reformulate and resubmit
3. If reformulation is also killed, the signal is logged as "undeliverable" and stored internally
4. Persistent kill patterns from a specific agent trigger a confidence review

---

## 6. Agent Lifecycle

Agents and contexts are not static. They are born, they evolve, and they die.

### 6.1 Forking

A context forks when it can no longer adequately serve its signal space. Fork triggers:

1. **Volume overload.** Signal rate exceeds the context's agent capacity.
2. **Coherence degradation.** Average familiarity score of incoming signals drops — a sub-cluster has formed.
3. **Dimensional split.** Clustering analysis detects two or more distinct signal populations.

Forking takes the agent's context and runs a bifurcating (or n-furcating) classification on its domain. The original agent keeps one partition. One (or n-1) new agents are spawned, each allocated a partition.

```
fork(agent, context, n=2):
  partitions = classify(context.signals, n)
  
  agent.context = partitions[0]           // original keeps first partition
  
  for i in 1..n-1:
    new_agent = spawn(
      context = partitions[i],
      competencies = inherited_seed,       // starting point, not destiny
      confidence = default
    )
  
  context.energy *= (1/n)                  // energy distributed
```

New agents inherit seed competencies but learn their own specialization from the signals in their partition. The original agent's domain narrows. Both (or all) are subject to the same lifecycle — they may fork again, merge, or decay.

### 6.2 Merging

Merging requires **topological adjacency**. Two agents can only merge if their contexts share edges — overlapping signals, overlapping consumers, or causal relationships. You don't merge distant contexts just because they're both quiet.

Merge triggers:

1. **Signal overlap.** > 70% of signals in Context A also score above threshold in neighboring Context B.
2. **Assessment agreement.** Agents in both contexts produce identical assessments for shared signals > 80% of the time.
3. **Consumer overlap.** The same users are being surfaced information from both contexts.

```
merge(context_a, context_b):
  // Only if topologically adjacent
  assert context_b in context_a.neighbors
  
  merged = Context(
    vector_store = union(context_a.vector_store, context_b.vector_store),
    active_agents = deduplicate(context_a.agents ∪ context_b.agents),
    energy = max(context_a.energy, context_b.energy),
    neighbors = (context_a.neighbors ∪ context_b.neighbors) - {context_a, context_b}
  )
  
  release(surplus_agents)
```

**Decay-to-fork handoff:** When a decaying agent holds knowledge that a neighboring agent is about to need (because the neighbor is forking into the decaying agent's territory), the system should absorb the decaying context into the fork rather than letting it die and then rediscovering the same knowledge. The decaying agent's vector store is folded into the new partition. This avoids the inefficiency of knowledge dying in one place and being relearned in another.

```
if decaying_context.neighbors.any(n → n.pending_fork):
  fold(decaying_context, into: neighbor.fork_partition)
else:
  archive(decaying_context)
```

### 6.3 Decay

Contexts that stop receiving relevant signals decay exponentially:

```
energy(t) = energy(t₀) * e^(-λ * (t - t₀))

When energy < decay_threshold:
  context.status = DECAYING
  // Check neighbors for handoff opportunity before dying
  
When energy < archive_threshold:
  if no neighbor handoff:
    context.status = ARCHIVED
    release(context.agents)
    context.vector_store → cold_storage
```

A project wraps up. A class of bugs gets fixed. A product area sunsets. The agents serving those concerns fade. Resources free. Nothing is manually archived. Relevance is the only survival criterion.

**Resurrection:** If new signals arrive that score high against an archived context's vector store, the context can be reactivated. Agents are respawned. The system remembers what it once knew.

---

## 7. Embedding Strategy

Signals are not embedded once with a single model. Different dimensions of a signal require different representations.

### Multi-Strategy Embedding

A single Slack message carries multiple types of information simultaneously:

| Dimension | Embedding Strategy | Purpose |
|---|---|---|
| **Semantic content** | General-purpose LLM embedding | What is being said |
| **Technical content** | Code-aware embedding | Code references, technical terms, system names |
| **People and roles** | Entity embedding (from org graph + interaction patterns) | Who is involved, who is affected |
| **Temporal context** | Positional encoding relative to project timelines | When this matters |
| **Business context** | Embedding against strategic documents, OKRs | How important this is to the organization |

```
Signal.embeddings = {
  semantic:   embed_semantic(signal.content),
  technical:  embed_technical(signal.content, signal.code_refs),
  social:     embed_social(signal.author, signal.mentions, org_graph),
  temporal:   embed_temporal(signal.timestamp, active_projects),
  strategic:  embed_strategic(signal.content, strategy_corpus)
}
```

### Embedding Strategies Are Not Fixed

The list above is a starting point — initial conditions, not a permanent taxonomy. Embedding strategies are themselves subject to the same lifecycle as everything else in the system.

Hypervisor agents monitoring system performance can observe: "signals in this context are being misclassified because the available embedding dimensions don't capture the relevant variation." A context that has emerged around regulatory compliance may need an embedding strategy tuned to legal language that none of the seed strategies provide.

The system can:
- **Spawn** new embedding strategies when existing ones fail to capture signal dimensions that matter
- **Retire** embedding strategies that no longer contribute to classification accuracy
- **Specialize** a general strategy into domain-specific variants as contexts fork

Each context learns which embedding dimensions are most predictive of relevance for its domain. A context around a codebase weights `technical` heavily. A context tracking customer issues weights `semantic` and `social`. These weights are learned from feedback, not assigned.

The same pattern. Applied to the system's own perception.

### Active Inquiry Embeddings

When a context performs active inquiry (Section 5.3), it uses the signal's multi-dimensional embedding to construct targeted queries against its RAG store:

```
inquiry(signal, context):
  queries = decompose(signal, context.preferred_dimensions)
  
  for query in queries:
    results = context.vector_store.search(query)
    if results.relevance > inquiry_threshold:
      yield Insight(query, results)
  
  return synthesize(insights)
```

The system asks questions the way a knowledgeable colleague would: not "does this match anything?" but "given what I know about this codebase, this team, this customer, and this quarter's goals — does this signal connect to anything I should raise?"

---

## 8. Business Priority Weighting

Not all signals are equal. The system must internalize organizational priorities without being explicitly told what matters. It learns from structural cues.

### Signal Importance Scoring

```
importance(signal) → float [0, 1]

Factors:
  - author_seniority:    Signals from leadership carry more weight (learned from org graph)
  - channel_gravity:     #incidents > #general > #watercooler (learned from response patterns)
  - financial_proximity: Signals near revenue outrank internal chatter
  - strategic_alignment: Similarity to active OKRs, investment decks, board materials
  - downstream_impact:   Signals of this type have historically led to incidents or revenue changes
  - engagement_velocity: How quickly and broadly people respond to related signals
```

A bug costing customers $500/day scores higher than a debate about office snacks. Not because someone tagged it "P0" but because the system can see the customer impact, the revenue exposure, the support ticket volume, and the engineering urgency — simultaneously, across contexts, without anyone synthesizing it manually.

The system already has all the data needed to determine whether a signal matters to the business. Investment decks. Sales pipelines. Customer support volume. Engineering velocity. OKR progress. The difference between "corn vs. peas for lunch" and "system costing $500/day" is visible in the signal metadata before any human intervenes.

### Resource Allocation

Business priority directly influences resource allocation:

**High importance:** Faster classification, more agents in consensus, lower intervention thresholds, deeper active inquiry.

**Low importance:** Batch classification during off-peak hours, minimal agent involvement, higher intervention thresholds, standard embedding only.

This is how the system manages cost without managing agents. The agents don't know about budgets. They respond to the signals they receive. The priority system determines *which* signals they receive and *how quickly*.

---

## 9. Resource Management

### Token Budget

The system's primary cost driver is LLM token consumption: embeddings, active inquiries, consensus reasoning, and output generation.

```
TokenBudget {
  daily_cap:         int              // hard limit on total daily token spend
  context_budgets:   map<Context, int> // allocated per context based on business_weight
  reserve_pool:      int              // held for high-priority signals
  current_spend:     int              // real-time tracking
}
```

### Allocation Strategy

Token budget is allocated proportionally to context business weight, recalculated daily:

```
context_budget(context) = daily_cap * (context.business_weight / Σ(all_context_weights))
```

High-importance contexts get more tokens. Low-importance contexts operate on tight budgets, naturally constraining them to simpler assessments. The lunch debate doesn't need deep multi-context reasoning.

### Token Bucket Prioritization

When an agent wants to perform an expensive operation, it requests tokens from its context's bucket:

```
request_tokens(agent, operation, estimated_cost):
  if context.remaining_budget >= estimated_cost:
    grant(estimated_cost)
  else if reserve_pool >= estimated_cost AND signal.importance > reserve_threshold:
    grant_from_reserve(estimated_cost)
  else:
    deny → agent falls back to cheaper heuristic evaluation
```

Denied agents don't stop working. They fall back to embedding-only classification, keyword matching, or cached assessments. The system degrades gracefully under budget pressure.

### Off-Peak Processing

Non-urgent work defers to off-peak hours:

- Batch classification of low-priority signals
- Context fitness evaluation and lifecycle decisions (fork/merge/decay)
- Embedding strategy evaluation and tuning
- Cross-context coherence analysis
- Hypervisor analysis of system performance

### Scope Limiting

Additional cost controls at the system boundary:

- **Input limiting:** Low-value sources can be sampled or batch-ingested
- **Output limiting:** Intervention frequency capped per user, per channel, per time window
- **Audience limiting:** During early deployment, restrict active processing to specific teams or channels. Expand as the system proves value.

---

## 10. Integrity and Traceability

### Signal Lineage

Every action the system takes is traceable back through its full decision chain:

```
Trace {
  signal:          Signal              // what triggered this
  contexts:        set<Context>        // where it was classified
  inquiries:       set<InquiryResult>  // what the contexts found
  assessments:     set<Assessment>     // what agents concluded (with domain weights)
  consensus:       ConsensusResult     // how the vote resolved
  constraint_eval: ConstraintResult    // what the constraint agents did
  output:          Output | null       // what the user saw (or didn't)
}
```

Every trace is immutable and stored in an append-only log. Where did this insight come from? Which contexts contributed? Which agents voted, with what weights? Why did a constraint agent kill it?

### Adversarial Response

When the system produces incorrect, misleading, or corrupted output, the trace provides the forensic path:

1. **Identify the corrupted component.** Was the signal itself bad? Was a context's RAG contaminated? Did an agent's model drift? Did a constraint agent miss something?

2. **Remediate at the source.**
   - Bad signal from a human → people problem. The system provides the evidence; management handles the human.
   - Contaminated RAG → edit the vector store. Remove bad embeddings. Re-embed from clean data.
   - Drifted agent → retrain or retire. Spawn replacement with default confidence.
   - Constraint gap → add the pattern to the kill list.

3. **Prevent recurrence.**
   - Human-in-the-loop gates for RAG curation of high-sensitivity contexts
   - Periodic adversarial testing: inject known-bad signals, verify the system catches them
   - Confidence floor enforcement: agents below threshold are automatically retired

The system is not immune to adversarial input. No system is. But it is *traceable*. Every decision has a receipt. When something goes wrong, you follow the chain, find the break, fix the component, and handle the human who caused it.

---

## 11. Onboarding

There is no cold start problem.

The system's data sources already exist. Slack history, ticket backlogs, PR archives, documentation, email threads, deploy logs — all available on day one. The aggregation layer (the foundation that tools like Adapt have already built) already connects to these sources and can retrieve from them.

Onboarding the stigmergic layer is not bootstrapping from nothing. It is adding agency and meta-intelligence to an existing knowledge base.

### Process

**Day 1: Ingestion.** Connect adapters. Ingest historical signals. Compute embeddings across all strategies. The system now has the same raw material a new hire would have on their first day: access to everything, understanding of nothing.

**Week 1: Context Formation.** As historical signals are processed, contexts emerge from clustering. The system begins to see the shape of the organization — not the org chart, but the actual topology of information flow. Which people talk to which other people. Which systems touch which other systems. Which tickets recur. Which decisions propagate and which die in their channel. Hypervisor agents begin identifying competency gaps and spawning specialized agents.

**Week 2: Passive Observation.** The system classifies incoming signals in real time. It produces assessments but does not act on them. Every proposed intervention is logged. Humans can review: would this have been helpful?

**Week 3+: Calibrated Intervention.** Based on observation feedback, thresholds are tuned. The system begins intervening — gently at first, more confidently as its accuracy is validated. Person-context agents begin learning individual attention patterns.

This is not different from how a capable new hire onboards. They read the docs, watch the channels, attend the meetings, ask questions, and gradually start contributing. The difference: the system reads *all* the docs, watches *all* the channels, and never forgets.

---

## 12. Implementation Architecture

### 12.1 Infrastructure Layer

```
┌─────────────────────────────────────────────────────────┐
│                    Source Adapters                        │
│  [Slack] [GitHub] [Linear] [Docs] [Email] [Deploy] ...  │
└────────────────────────┬────────────────────────────────┘
                         │ Signals
                         ▼
┌─────────────────────────────────────────────────────────┐
│              Priority Ingestion Queue                    │
│         (Kafka / Redis Streams + Priority)               │
└────────────────────────┬────────────────────────────────┘
                         │
              ┌──────────┼──────────┐
              ▼          ▼          ▼
        ┌──────────┬──────────┬──────────┐
        │ Context  │ Context  │ Context  │
        │ Cluster  │ Cluster  │ Cluster  │
        │    A     │    B     │    N     │
        └────┬─────┴────┬─────┴────┬─────┘
             │          │          │
             ▼          ▼          ▼
        ┌──────────────────────────────────┐
        │          Consensus Bus            │
        └──────────────┬───────────────────┘
                       │
                       ▼
        ┌──────────────────────────────────┐
        │      Constraint Agents           │
        │   (Kill / Redact / Pass)         │
        └──────────────┬───────────────────┘
                       │
              ┌────────┼────────┐
              ▼        ▼        ▼
          [Slack]    [PR]    [Ticket]   ← Output Adapters
```

### 12.2 Context Cluster Detail

Each context cluster runs as an independent unit:

```
Context Cluster {
  Vector Store:      Pinecone / pgvector / Qdrant (multi-index for embedding strategies)
  Agent Pool:        N agents as async workers (serverless functions or containers)
  State Store:       Redis (working memory, energy tracking, token budget)
  Consensus Engine:  Lightweight voting service (in-process)
  Lifecycle Manager: Monitors energy, triggers fork/merge/decay
  Token Meter:       Tracks consumption against context budget
}
```

Clusters are **horizontally scalable**. Adding a new source adapter doesn't change the architecture. Adding 10x more engineers doesn't change the architecture. The system scales by spawning more contexts and agents, not by redesigning.

### 12.3 Agent Implementation

Agents are lightweight. Most are **small models or heuristic classifiers**, not full LLM instances. LLM reasoning is reserved for ambiguous cases, active inquiry, natural language output, and hypervisor analysis.

Typical agent implementations:
- Embedding similarity search + structural comparison (no LLM needed for most classification)
- Graph analysis of signal dependencies
- Pattern matching on signal content with LLM fallback for ambiguity
- Sliding window clustering for anomaly detection
- Temporal analysis for staleness detection
- Cross-context familiarity analysis for propagation

The specific competency an agent develops depends on the signals it processes and the feedback it receives — not on a design-time enumeration.

---

## 13. Differentiation from Existing Approaches

| Approach | Architecture | Limitation |
|---|---|---|
| **RAG + Chat** (most AI tools) | Human queries → retrieval → response | Reactive. Requires humans to know what to ask. |
| **Auto-Generated Docs** (Sequa, etc.) | AI reads code → writes docs | Solves wrong problem. Produces artifacts humans don't read. |
| **Orchestrated Multi-Agent** (CrewAI, AutoGen, LangGraph) | Coordinator assigns tasks to agents | Central bottleneck. Predefined workflows. Brittle. |
| **Knowledge Graph** (Neo4j, etc.) | Entities + relationships, human-maintained | Static. Manual curation. Doesn't scale. |
| **Stigmergic Agent Architecture** | One recursive pattern: emergent contexts, self-organizing agents, weighted consensus, constraint-gated output | No coordinator. No predefined structure. No fixed types. Scales by growing. |

The key differentiator: **this system does not need to be told what matters, what types of agents to run, or how to organize its knowledge.** It discovers all of these from signal patterns. The architecture is one pattern, applied recursively, at every level — from individual signal classification to system-wide self-monitoring.

---

## 14. Deployment Strategy

### Phase 1: Passive Observation
- Ingest signals from all connected sources (historical + real-time)
- Compute embeddings, build initial contexts from signal clustering
- Seed agents begin developing competencies
- No intervention — log what the system *would* do
- Validate: are emergent contexts sensible? Are familiarity scores calibrated?

### Phase 2: Advisory Mode
- Surface recommendations to a single review channel
- Human reviews proposed interventions before delivery
- Feedback loop active — humans rate accuracy
- Hypervisor agents begin identifying competency gaps
- Tune: relevance thresholds, consensus confidence floors, intervention triggers

### Phase 3: Direct Intervention (Low Stakes)
- Duplicate detection active
- Decision propagation active
- Dead end detection active
- Person-context agents active — personalized filtering begins learning

### Phase 4: Full Autonomy
- Conflict detection with blocking
- Deploy gates for high-risk changes
- Pattern recognition with proactive alerting
- Self-organizing lifecycle fully autonomous
- Constraint agents operating at full constraint set
- Hypervisor agents spawning and retiring competencies as needed

Each phase builds trust. The system earns authority incrementally, not by assertion.

---

*In this system, humans ideate and deliberate and think aloud and spitball and plan and solve creatively. AI collaborates where it shines — omnipresence, perfect memory, zero ego, tireless pattern recognition across dimensions no human can hold simultaneously.*

*Together, we are the centaur.*

*The system learns the organization faster than the organization learns itself. Not to replace the humans in it, but to make their thinking count.*
