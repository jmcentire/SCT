# Briefing: Communicative Variance Theory → Constellation Composition SR

## For the Claude Code instance working in ~/Code/AI

This document bridges two research streams running in parallel:
1. **CommunicativeVariance** (`~/Personal/Research/CommunicativeVariance/`): A formal information-theoretic framework for when noise produces net benefit in hierarchical systems
2. **Constellation Composition** (`~/Code/AI/`): Paper 3's stochastic resonance experiment on model fingerprint decomposition at high collinearity

---

## The Theory (Theorem 1): Five Sufficient Conditions for Net-Beneficial Noise

We proved (with simulation validation across SR and non-SR mechanisms) that noise at level N produces net benefit at level N+1 when ALL five conditions hold:

| Condition | Formal | Plain English |
|-----------|--------|---------------|
| **C1: Suboptimal** | P_{N+1}(0) < P_{N+1}^{max} | The system isn't reaching its potential without noise |
| **C2: Nonlinear** | Jensen gap > 0 | The response function is nonlinear (E[f(x+noise)] ≠ f(x)) |
| **C3: Accessible** | Noise can reach improvement region | The noise distribution can push the system toward better states |
| **C4: Weighting** | β·ΔP_{N+1} > α·ΔP_N at some σ | The gain at the higher level outweighs the cost at the lower level |
| **C5: Robustness** | P_N degrades gracefully | The lower level doesn't catastrophically fail under moderate noise |

**Key finding**: This is NOT just stochastic resonance. We validated across:
- Threshold detectors (classic SR) ✓
- Sigmoid detectors ✓
- Simulated annealing (landscape escape) ✓
- Ensemble diversity (error decorrelation) ✓
- Multi-armed bandit (exploration-exploitation) ✓

Zero counterexamples in 500 Monte Carlo random parameter sweeps.

**The inverted-U is universal**: there always exists an optimal noise level σ* > 0 when all conditions are met, and excessive noise always destroys the benefit.

---

## Mapping to Constellation Composition at High Collinearity

The Paper 3 stochastic resonance experiment (`paper3_stochastic_resonance.py`) maps perfectly:

### Level N: Centroid Fidelity
- **What it is**: The accuracy of domain centroid vectors (averages of specialist activation fingerprints)
- **How noise degrades it**: Adding Gaussian noise to centroids before Gram-Schmidt distorts the centroid representation
- **Degradation model**: Approximately linear for small noise (centroid + noise still close to centroid), potentially catastrophic for large noise (Gram-Schmidt produces garbage basis)

### Level N+1: Composition Quality
- **What it is**: Cross-domain win rate (vs generalist and vs task arithmetic)
- **How noise helps**: At high collinearity (0.973 at 7B), orthogonal components have tiny norms. The discriminative signal is below the effective detection threshold of the decomposition. Noise can push weak orthogonal signals above this threshold.
- **The nonlinearity**: Gram-Schmidt orthogonalization is nonlinear (involves projection, subtraction, normalization). The decomposition weights depend nonlinearly on the basis orientation. This creates the Jensen gap needed for C2.

### Condition Check

| Condition | Status at 7B (collinearity 0.973) | Evidence |
|-----------|------|---------|
| C1: Suboptimal | **MET** | Cross-domain vs TA dropped to 60.7% (from 93.3% at 3B). The clean signal fails. |
| C2: Nonlinear | **MET** | Gram-Schmidt + L2 normalization + weight decomposition are all nonlinear |
| C3: Accessible | **MET** | Gaussian noise can perturb basis vectors in any direction |
| C4: Weighting | **TESTABLE** | If noise-assisted win rate exceeds baseline, C4 is confirmed empirically |
| C5: Robustness | **LIKELY MET** | Centroids are averages over many activation vectors — inherently robust |

### Predictions from Theorem 1

**The critical prediction is #4 — the null result. Without it, the rest could be confounded with simple regularization or dithering.**

1. An **inverted-U** in win rate as noise fraction increases (0.001 → 0.1) at high collinearity (7B, 3B)
2. σ* should be **larger at higher collinearity** (7B > 3B > 1.5B), because the signal deficit (C1) is larger
3. The **ensemble averaging** approach (5 noise realizations, average weights) should outperform single-shot, because averaging reduces noise variance while preserving the SR benefit
4. **At low collinearity (0.5B, collinearity 0.906), noise should NOT help.** The clean signal already achieves 100% cross-domain win rate — C1 is not met. If noise helps everywhere regardless of whether C1 is met, the mechanism is dithering/regularization, not subthreshold signal detection. If noise helps specifically where C1 is met and fails where C1 is not met, that is a controlled dissociation confirming the theoretical mechanism. **This prediction must appear in any abstract or summary. It is the falsification condition doing actual work.**

### Falsification

The theory is falsified if:
- All conditions are met AND the inverted-U doesn't appear (noise only hurts or only helps without peaking)
- The optimal noise level doesn't scale with collinearity
- Noise helps equally at 0.5B (C1 not met) and 7B (C1 met) — would indicate the mechanism is not subthreshold detection

### What a null result means

If SR does NOT help at 7B, that does not falsify the theory — it means the 7B degradation is **geometric collapse** (the specialist information is genuinely absent from the orthogonal complement) rather than **threshold failure** (the information is present but below detection). Both are useful findings for Paper 3. The SR experiment is a diagnostic: it distinguishes between "the signal isn't there" and "the signal is there but too weak to detect." Frame it that way.

---

## Specific Suggestions for the SR Experiment

### 1. Two noise calibration strategies (run both, report which fits)

**Empirically motivated (more defensible):**
`σ = k * (collinearity - 0.9)` where k is a scale factor tuned on one model, tested on others. This is directly motivated by the data — no theoretical assumptions about potential wells.

**Theoretically motivated (use with care):**
The Gammaitoni formula `D_opt = ΔV/2` from the Kramers escape rate applies to bistable potential wells with known barrier heights. The "barrier height" here would be the effective detection threshold of Gram-Schmidt, which is NOT directly measurable as a potential energy. If you use this framing, you must either (a) measure ΔV empirically from the orthogonal component norm distributions, or (b) frame it explicitly as an analogy rather than a direct application. A reviewer who knows the SR literature will test this immediately. The empirical calibration is safer; report the Gammaitoni fit as a secondary analysis.

### 2. Test the Jensen gap directly

Before and after adding noise, compute the orthogonal component norms. If the mean norm increases with moderate noise (because noise pushes near-zero components above zero), that's the Jensen gap in action.

### 3. The σ* scaling curve is the publishable figure

If prediction 2 holds — optimal noise fraction increases monotonically with collinearity across 0.5B, 1.5B, 3B, 7B — that is a single figure that tells the whole story:
- x-axis: collinearity
- y-axis: optimal noise fraction σ*
- Four points, one curve

**That figure appears in both papers.** CommunicativeVariance cites it as empirical validation of C1 scaling. Paper 3 cites it as the design prescription for noise-assisted decomposition at scale.

### 4. Report the inverted-U shape explicitly

The six noise fractions in the current script [0.001, 0.005, 0.01, 0.02, 0.05, 0.1] should capture the U. If the best is at the boundary (0.1), extend the sweep. The theory guarantees a peak and decline.

### 5. Run the 0.5B null condition

This is not optional. Without the low-collinearity null, the experiment cannot distinguish SR from dithering. Even if it means re-running the 0.5B model through `paper3_stochastic_resonance.py`, it's the most important control.

---

## How These Papers Should Cite Each Other

**CommunicativeVariance is the theory paper. Paper 3 is an empirical instance. The citation runs one direction.**

- **Paper 3 cites CommunicativeVariance** for the theoretical grounding of the SR experiment: "We apply the sufficient conditions framework of [CommunicativeVariance] to predict when noise injection improves fingerprint-based model composition..."
- **CommunicativeVariance mentions the activation space experiment** in a "computational validation" subsection or as a corollary application, but does not depend on it: "As a computational validation in a domain outside the organizational literature, [Paper 3] demonstrates that Condition 1 predicts when noise-assisted decomposition succeeds in high-collinearity activation spaces..."

This way CommunicativeVariance stands independently. Paper 3 gains theoretical depth without creating a circular dependency.

---

## Key Files in CommunicativeVariance

| File | What it contains |
|------|-----------------|
| `atlan_formalization.md` | Theorem 1 statement, proof sketch, corollaries, simulation results |
| `simulation_level_crossing_v2.py` | SR-class tests (threshold, sigmoid, polynomial, linear null) |
| `simulation_non_sr_mechanisms.py` | Non-SR generality tests (optimization, ensemble, bandit) |
| `simulation_results_v2.json` | Full numerical results from SR tests |
| `simulation_results_non_sr.json` | Full numerical results from non-SR tests |
| `section2_formal_foundation.md` | Formal chain: Crawford-Sobel → Blau-Michaeli → Atlan → Kosko |
| `gen_lossy.md` | Full 36K treatise on the generative lossy channel theory |

---

## The Bigger Picture: 7B Degradation is a Detection Problem, Not Geometric Collapse

This is the key reframing. The 7B cross-domain win rate dropping to 60.7% from 3B's 93.3% looks like the geometric approach breaking down at scale. But if SR helps, it means **the domain markers are subthreshold at high collinearity, not absent.** The information is there — the decomposition just can't detect it because the orthogonal component norms are below the effective noise floor of the Gram-Schmidt process.

That's a much more optimistic finding than geometric collapse. It means there's a fix: noise injection at the fingerprinting/decomposition stage rather than retraining at smaller scales or abandoning the approach. The 7B specialist knowledge is encoded in the activation space; we just need a better detector.

If the SR experiment validates this:
- **CommunicativeVariance gets a computational, reproducible demonstration** of C1 (subthreshold signal + noise → detection) in a domain no one has connected to the organizational creativity literature
- **Paper 3 gets a principled fix** for the collinearity scaling problem, grounded in a formal theorem rather than ad-hoc noise injection

If it doesn't:
- **CommunicativeVariance still stands** on its simulation validations (SR, non-SR, Monte Carlo)
- **Paper 3 learns something equally valuable**: the 7B problem is geometric collapse, not threshold failure, which redirects the research toward different solutions (e.g., nonlinear decomposition methods rather than noise injection)

Both projects are instances of: **a lossy channel forces the receiver to reconstruct, and the reconstruction can exceed the fidelity of the original signal when the right conditions are met.** In organizations, the lossy channel is strategic communication under misaligned incentives. In model composition, the lossy channel is high-dimensional projection under near-collinearity. Same mechanism, same conditions, same inverted-U.
