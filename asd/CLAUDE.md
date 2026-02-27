# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

This repository is one leg of a **trifecta**: (1) a patentable solution, (2) a proof-of-concept implementation, and (3) an academic paper — all centered on **Ambient Structure Discovery (ASD)**. This repo contains the theory, papers, and research. The working code lives at `/Users/jmcentire/WanderRepos/tools/stigmergy`.

## Core Thesis

Organizations fail not because they lack information, but because they cannot perceive what they don't know they're missing. The ASD system is a **stigmergic mesh** — a decentralized, self-organizing architecture that continuously ingests work artifacts (sematectonic traces), develops emergent topological specialization, and surfaces structural patterns without requiring anyone to formulate a query.

## Key Concepts (Required Context)

- **Second-Order Ignorance (2OI)**: Unknown unknowns. The system's entire purpose is converting 2OI into 1OI (known unknowns) and then 0OI (knowledge). Armour's Five Orders of Ignorance is the epistemic taxonomy.
- **Dysmemic Pressure**: A compound selection force (strategic communication degradation + adverse selection in idea markets + transmission bias) that causes organizations to converge on collective delusion. Dysmemes are signals optimized for internal fitness rather than correspondence with reality.
- **Stigmergy**: Coordination through shared environment rather than direct communication. The architecture uses sematectonic stigmergy (the work product itself is the signal) rather than marker-based stigmergy (explicit tags/metadata).
- **One Pattern**: The architecture has a single recursive pattern — an agent bound to a context, ingesting signals, producing assessments, forking/merging/decaying. Supervisors, control layers, and hypervisors are all the same pattern applied at different scales. There are no special classes.
- **Babbling Equilibrium**: Crawford-Sobel result — when incentive bias b >= 1/4, communication carries zero information. This is the mathematical limit the mesh is designed to bypass.
- **Spectral Right-Shift**: Structural anomalies cause high-frequency energy in the graph Laplacian spectrum. This is the mesh's detection mechanism for hidden organizational dysfunction.

## The Proof-of-Concept Code

The implementation lives at **`/Users/jmcentire/WanderRepos/tools/stigmergy`**.

### Tech Stack
- **Python 3.12+**, Pydantic for data validation, NumPy for numerics
- **Build**: Hatchling (`pip install -e .`)
- **CLI**: `stigmergy init | run | config | status`
- **LLM**: Anthropic Claude (optional; falls back to stub heuristics)
- **Tests**: 485 passing tests, pytest with pytest-asyncio (`asyncio_mode = "auto"`)

### Common Commands
```bash
cd /Users/jmcentire/WanderRepos/tools/stigmergy
. .venv/bin/activate

# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_mesh.py -v

# Run by keyword
pytest -k "test_fork" tests/

# Install for development
pip install -e ".[dev]"

# Install with Anthropic integration
pip install -e ".[cli]"

# Run with mock data
stigmergy run --once

# Run with live GitHub data
stigmergy run --once --live
```

### Code Architecture
The implementation directly realizes the ART-based stigmergic mesh described in the papers:

- **`src/stigmergy/mesh/`** — Core mesh: `mesh.py` (ART network, BFS routing), `worker.py` (ART category nodes), `topology.py` (neighbor selection), `fingerprints.py` (activation patterns), `insights.py` (pattern detection)
- **`src/stigmergy/core/`** — Algorithms: `familiarity.py` (5-component ART match function), `consensus.py` (weighted voting), `energy.py` (exponential decay lifecycle), `lifecycle.py` (fork/merge/decay)
- **`src/stigmergy/primitives/`** — Data types: `signal.py`, `context.py`, `agent.py`, `assessment.py`
- **`src/stigmergy/adapters/`** — Source integrations: GitHub, Linear, Slack (mock + live)
- **`src/stigmergy/constraints/`** — Output filtering: PII/credential redaction and killing
- **`src/stigmergy/structures/`** — Performance: Bloom filters, LSH, SimHash, ring buffers, tries
- **`src/stigmergy/services/`** — LLM, embedding, vector store, token budget
- **`src/stigmergy/cli/`** — CLI entry point, config, budget tracking, live adapters
- **`src/stigmergy/policy/`** — Policy engine, spectral analysis, budget enforcement

### Key Design Patterns in Code
- **Stop-on-first-accept**: BFS routing, first worker above threshold takes the signal (Simon's satisficing)
- **Complement coding**: Full workers raise vigilance thresholds (prevents monopoly)
- **Match-based learning**: Weights update only on acceptance (ART stability guarantee)
- **Three learning modes**: `rag_indexed` (full storage), `context_summarized` (compressed), `weight_shifted` (lossy impression)
- **Immutable signals**: Signals are frozen Pydantic models; derived state lives in contexts
- **Async throughout**: All I/O is async; tests use pytest-asyncio auto mode

### Test Configuration (pyproject.toml)
```
[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
addopts = "-x -q"
```

## Document Map (This Repository)

### Primary Papers (by Jeremy McEntire)
- **`asd.tex`** — The formal academic paper (LaTeX). "Ambient Structure Discovery via Stigmergic Mesh." This is the authoritative document.
- **`DysmemicPressure.txt`** — Companion paper on selection dynamics in organizational information environments.
- **`RightingtheShip.txt`** — Paper on the succession paradox — why replacing leaders fails when the information architecture is preserved.
- **`UnifiedTheory.txt`** — "Compression, Selection, and Organizational Self-Deception" — substrate independence claim (cognition, organizations, AI, academia all exhibit the same mechanism).
- **`RigorousInquiry.txt`** — Paper on rigorous inquiry as the counter-mechanism to dysmemic pressure. Three properties: independence, specificity, propagation.

### Architecture Documents
- **`stigmergic.md`** — Technical architecture spec for the stigmergic agent system. Defines primitives (Signal, Context, Agent), the one-pattern principle, competitive routing, consensus, control layer, and token budget model.
- **`emergent_architecture.txt`** — Earlier version of the architecture doc, superseded by `stigmergic.md`.

### Theoretical Research Reports (AI-generated deep research)
- **`ADSClaims.txt`** — Theoretical audit verifying claims in `asd.tex` against primary literature.
- **`TechValidationRept.txt`** — Forensic validation of specific mathematical claims. Confirms ART 2/3 Rule, N* formula, babbling threshold, BWGNN.
- **`TheoreticalFoundations.txt`** — Stigmergy, SOMs, spectral methods, 2OI costs, external observer problem.
- **`TheoreticalFoundations2.txt`** — ART, Crawford-Sobel, Liberti-Mian, spectral right-shift, Flyvbjerg power laws.
- **`ADSTrends.txt`** — Cross-disciplinary foundations: ART, VSM, Dretske semantics, rate-distortion theory.
- **`ArchitectureUnknowns.txt`** — Maps the theoretical neighborhood: ART vigilance, hierarchical friction, VSM, power laws.
- **`ASDThreads.txt`** — Six research threads synthesized: stigmergy, SOMs/GNG, spectral methods, 2OI empirics.

### Theoretical Extensions
- **`Orthogonality.txt`** — Orthogonal evaluation as partial escape from self-referential evaluation failure (Lawvere's fixed-point theorem, relocation thesis). Key result: exogeneity is a spectrum; the price of orthogonality is substantive thinness.
- **`Unnamed.txt`** — "The unnamed theorem" — why systems cannot grade their own exams. Lawvere unifying Godel/Tarski/Cantor/Turing across 8 domains. Identifies the gap between static self-reference (formal systems) and reactive self-reference (strategic agents).
- **`Alignment.txt`** — Analysis across the four quadrants (Discovery, Normalized Deviance, Noise, Ambient).

### Dysfunction Paper (Subdirectory)
- **`dysfunction/paper.tex`** — "The Organizational Physics of Multi-Agent AI: Substrate-Independent Dysfunction in Autonomous Software Engineering Swarms." Empirical paper demonstrating that multi-agent AI systems exhibit identical organizational dysfunction to human organizations. Uses data from the swarm deployment at `/Users/jmcentire/WanderRepos/swarm/arch/.swarm/state.json`. Has its own `CLAUDE.md` with detailed context.
- **`dysfunction/references.bib`** — Bibliography (35 citations spanning Crawford-Sobel, Goodhart, Lawvere, Akerlof, Liberti-Mian, plus recent 2025 AI multi-agent empirics).

### Other
- **`conversation.txt`** — Conversation log from the collaborative development of the architecture.
- **`sum.txt`** — Summary mapping theory across quadrants.
- **`The Cage and the Mirror.txt`** / **`The Key and the Current.txt`** — Longer-form works (book-length).

## Key Theoretical Pillars

These are the formal results that underpin everything:

1. **Crawford-Sobel (1982)**: N* = floor((1 + sqrt(1 + 2/b)) / 2). At b >= 1/4, babbling equilibrium.
2. **ART Stability (Grossberg/Carpenter)**: Match-based learning with vigilance parameter rho guarantees stable category formation in non-stationary environments. The 2/3 Rule prevents hallucinated patterns.
3. **Conant-Ashby (1970)**: Every good regulator must be a model of the system it regulates. The mesh must maintain requisite variety.
4. **Beer's VSM**: System 4 (Intelligence) is the locus of ambient discovery. The 3-4 homeostat balances stability and plasticity. Algedonic signals bypass hierarchy.
5. **Flyvbjerg**: IT project overruns follow power law with alpha ~ 1.0 (infinite mean and variance). Standard risk management is mathematically inapplicable.
6. **Spectral Right-Shift (Tang et al. 2022)**: Anomalies shift spectral energy toward high frequencies in graph Laplacian. BWGNN uses Beta wavelet band-pass filters to detect this.
7. **Elliott Threshold (~25)**: Beyond ~25 participants, social negotiation collapses; stigmergic coordination becomes necessary.
8. **Liberti-Mian**: Hierarchies exhibit a structural break at Level 3 where sensitivity to soft information collapses to near-zero.

## The Swarm (Empirical Source for Dysfunction Paper)

The multi-agent coding swarm at `/Users/jmcentire/WanderRepos/swarm/` provides the empirical evidence for the dysfunction paper. Key files for the substrate-independence argument:

- **`arch/.swarm/state.json`** — Complete audit trail: 89 stages, $57.43, 7.17M tokens, 18 hours. Shows bikeshedding (factual=0, subjective=15-23), governance conflicts, backward pipeline oscillation, verification theater (tests=0/0).
- **`src/swarm/agents/review.py`** — Six anti-dysfunction mechanisms in code. Critical because the prompts encode anti-dysfunction, not dysfunction. The dysfunction emerged despite countermeasures.
- **`src/swarm/scoring.py`** — Seven proxy metrics (confidence, diff_size, coverage, risk_count, coherence, guardian, issue_count). None measure whether code does what it's supposed to. Goodhart confirmation.
- **`src/swarm/control.py`** — Lyapunov stability monitor: V(x) with 8-metric StateVector, dV/dt via linear regression, oscillation detection via sign-flip counting.
- **`docs/isomorphism_agency.txt`** — Core isomorphism thesis: Shannon channel capacity, Information Bottleneck, Ashby's requisite variety apply identically to human hierarchies and AI swarms.
- **`docs/emergence-synthesis.md`** — Five traps (Cage, Legibility, Context, Scale, Fiduciary) + solutions (Mirror, Emergence, stigmergy).

## Related Books by Author

- **`The Cage and the Mirror`** — Organizational dysfunction via Godelian incompleteness. The book-length argument underlying the ASD theoretical framework.
- **`Privacy: Architecture of Forgetting`** — Cryptographic architecture for a privacy-preserving internet. Six independently adoptable components. Located at `/Users/jmcentire/Personal/Privacy/`. Has its own CLAUDE.md.
- **`Applied Synthesis`** (3rd ed.) — Perceptual blindness.
- **`Uncommon Leadership`** — Leadership theory.
- **`Emergence: A Programming Paradigm`** — Constraint-based agent systems. Paper at `mcentire2025e` in dysfunction/references.bib.

## Working Notes

- The LaTeX paper (`asd.tex`) is the canonical theoretical source. Other documents are research inputs, earlier drafts, or explorations that fed into it.
- The architecture in `stigmergic.md` reflects the "one pattern" insight — resist creating taxonomies or special classes. Everything is the same agent-context-signal pattern applied recursively.
- The 9.9% per-layer information loss figure (attributed to Liberti-Mian/Petersen) has been flagged as a likely mis-citation in `TechValidationRept.txt`. The phenomenon is valid but the specific number is suspect.
- `emergent_architecture.txt` is superseded by `stigmergic.md` — the former has artificial distinctions (agent types, separate supervisor class, ACL) that were collapsed in the latter.
- The dysfunction paper (N=1) needs replication, formal bridge between theory and data (compute Crawford-Sobel `b`), and citation verification before submission beyond arXiv.
- Several 2025-2026 citations in the dysfunction paper were sourced from `isomorphism_agency.txt` (itself AI-generated research), not verified against primary sources. Verify before publication.

## LaTeX Compilation

```
pdflatex asd.tex
bibtex asd
pdflatex asd.tex
pdflatex asd.tex
```

Note: A `.bib` file is referenced but may not yet exist in the repository. Citations use `natbib` with `\citep{}` commands.

## Kindex Knowledge Graph

A persistent knowledge graph (`kin`) indexes conversations, projects, and intellectual work across all repos. It hooks into Claude Code automatically (SessionStart, PreCompact). Docs: https://jmcentire.github.io/kindex/

```bash
kin search "asd-patent"          # Hybrid search (FTS + graph)
kin context asd-patent           # Pull related context
kin add "<insight>"              # Capture discoveries
kin link <a> <b> <rel> --why "<reason>"  # Create edges
```

Legacy Conv vault (459 nodes, richer historical data): `~/Personal/Projects/Conv/`
