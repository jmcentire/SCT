# Structural Compression Theory

**Data, code, and experimental results for the monograph:**

*Structural Compression Theory: A Unified Information-Theoretic Account of Organizational Dysfunction, Creativity, and Substrate-Independent Selection Dynamics*

Jeremy McEntire, 2026

## Overview

Every system that coordinates at scale compresses information. The compression creates gaps between what the system sees and what is real. Selection fills those gaps -- not with truth, but with whatever survives the selection environment. The result is drift: toward internal consistency, away from external accuracy. The sequence is as reliable as entropy and as avoidable.

This repository contains the code, data, and experimental results supporting the monograph's claims. The physics does not care what the agents are made of. It cares how they coordinate.

## Repository Structure

### `activation-space/`
Neural network experiments (Chapters 9-10, Appendices G-J, P). Ensemble collapse, constellation composition, stochastic resonance, CMS detection, and structural transfer studies at scales from 124M to 7B parameters.

- `src/` -- Library code for fingerprinting, composition, CMS, regime detection, and speculative training
- `paper3_*.py` -- Ensemble collapse and constellation composition experiments
- `paper4_*.py` -- Stochastic resonance and structural transfer experiments
- `paper5_*.py` -- CMS detection experiments
- `phase*.py` -- Earlier-phase validation experiments
- `results/` -- Experimental outputs (large checkpoint directories excluded)

### `multi-agent-ai/`
Multi-agent AI dysfunction experiments (Chapter 8, Appendix F). Four coordination architectures given identical tasks, demonstrating that organizational dysfunction emerges from coordination topology, not agent properties.

- `experiments/` -- Experiment code and run outputs for unary, swarm, hierarchical, and high-trust architectures
- `experiment.md` -- Experimental protocol and design

### `organizational/`
Organizational evidence (Chapters 1, 10, Appendices A-B).

- `variance-compression/` -- SEC filing linguistic analysis
- `nasa-variance/` -- NASA budget document linguistic analysis (FY1961-2025), analysis scripts and extracted metrics. Raw PDFs excluded (publicly available from NASA History Office).
- `noc-variance/` -- Northrop Grumman 10-K linguistic analysis. Raw filings excluded (publicly available from SEC EDGAR).
- `case-studies/` -- Apple, aerospace, and CEO language case studies
- `forced-ranking/` -- Forced ranking simulation with code and results

### `communicative-variance/`
Communicative variance and generative lossy channel formalization (Chapter 6, Appendix C). Simulation code, results, and paper.

### `strategic-rdp/`
Strategic Rate-Distortion-Perception tradeoff (Chapter 6, Appendix E). Formal extension of Blau-Michaeli to organizational communication under Crawford-Sobel strategic constraints.

### `emergence-calculus/`
Design calculus for emergence (Chapter 13, Appendix O). Lambda estimation studies and compositional well-formedness validation.

### `asd/`
Ambient Structure Discovery (Chapter 12, Appendix L). Stigmergic mesh implementation and patent documentation.

## Relationship to the Monograph

Each directory maps to specific chapters and appendices. The monograph derives its claims from formal results (rate-distortion theory, Crawford-Sobel, Kosko's forbidden interval theorem) and tests them empirically across four substrates:

| Substrate | Evidence | Directory |
|-----------|----------|-----------|
| Multi-agent AI | Coordination dysfunction without human psychology | `multi-agent-ai/` |
| Neural networks | Activation-space compression dynamics at 7B parameters | `activation-space/` |
| Organizations | SEC filing entropy, NASA linguistic metrics, case studies | `organizational/` |
| Simulations | Forced ranking misallocation, emergence dynamics | `organizational/forced-ranking/`, `emergence-calculus/` |

## Reproducibility Notes

- Neural network experiments were run on NVIDIA A100 and A6000 GPUs via Vast.ai. Setup scripts (`setup_vastai.sh`, etc.) are included.
- Multi-agent experiments use Claude API calls. Experiment logs and complete outputs are preserved in `experiments/runs/`.
- Organizational analyses use publicly available data (SEC EDGAR, NASA History Office). Analysis scripts and extracted metrics are included; raw source documents are excluded to manage repository size.

## Citation

McEntire, J. (2026). *Structural Compression Theory: A Unified Information-Theoretic Account of Organizational Dysfunction, Creativity, and Substrate-Independent Selection Dynamics.* SSRN.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18808428.svg)](https://doi.org/10.5281/zenodo.18808428)

## License

Code: MIT License
Data and analysis outputs: CC BY 4.0
