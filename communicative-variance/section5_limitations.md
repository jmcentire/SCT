# Section 5: Limitations, Falsification, and Open Questions

## 5.1 What the Theory Claims and Does Not Claim

The framework claims that five conditions are jointly *sufficient* for net-beneficial noise in a two-level system. It does not claim they are necessary. It does not claim that noise is generally good. It does not claim that the same optimal noise level applies across substrates. It claims: check these five conditions; if all hold, moderate noise helps; if any fails, no guarantee.

The framework also claims that the same lossy-channel mechanism produces both organizational dysfunction and creative emergence, with the valence determined by the selection environment. It does not claim that dysfunction and creativity are identical — it claims they share a root cause. The design problem is channeling, not elimination.

## 5.2 Falsification Conditions

The theory is falsified by any of the following:

**F1: Counterexample to sufficiency.** A system where all five conditions (C1-C5) are verified to hold and the inverted-U does not appear — noise monotonically hurts, or monotonically helps without peaking. Zero counterexamples have been found across 500 Monte Carlo configurations and six mechanisms. But sufficiency claims are universal quantifiers: one confirmed counterexample invalidates the claim.

**F2: Mechanism-independent benefit.** A system where noise produces the inverted-U benefit without *any* of the five conditions holding — no suboptimality (C1), no nonlinearity (C2), no accessibility (C3), no favorable cost-benefit ratio (C4), and no robustness (C5). This would show the conditions are not even useful as indicators. The 8.9% of Monte Carlo configurations that showed benefit with at least one condition violated had at most one or two conditions violated, not all five. A true F2 falsification requires benefit with zero conditions met.

**F3: Symmetric valence across selection regimes.** Two systems with identical channel properties (same compression, same residual, same noise level) but different selection criteria (one convergent, one divergent) that produce the *same* valence of output. The framework predicts that selection criterion alone determines whether the generative residual produces dysfunction or novelty. If convergent and divergent selection produce identical outputs, the dual-valence claim (Proposition 1) fails.

**F4: Noise-independent SR in the activation space experiment.** The companion experiment on activation space decomposition (Section 4.5) generates a specific falsifiable prediction: noise should help at high collinearity (7B, C1 met) and should *not* help at low collinearity (0.5B, C1 not met). If noise helps equally at both scales — or at neither — the dissociation fails, and the mechanism is either simple regularization (helps everywhere) or irrelevant (helps nowhere).

**F5: Beneficial noise in a linear system.** If noise produces net system benefit in a system where the integration function is strictly linear (not affine-with-offset, but truly linear), C2 is violated and the Jensen gap is identically zero. The computational null model (linear detector) shows exactly zero benefit. A real-world counterexample would overturn the claim that nonlinearity is essential.

## 5.3 Limitations of the Computational Validation

### Sufficient vs. necessary

The conditions are sufficient but not necessary. Approximately 9% of configurations with at least one condition violated still showed benefit. This means the conditions define a *conservative* boundary. Systems outside the boundary may still benefit from noise, but the theory does not guarantee it. Tightening the gap between sufficient and necessary conditions is an open problem.

### The ensemble C4 test failure

The ensemble diversity mechanism's C4 violation test ($\alpha = 0.95$, $\beta = 0.05$) did not effectively induce violation. The hard XOR problem makes all individual predictors weak regardless of the weighting parameter — noise still helps because even heavily weighted weak individuals benefit from error decorrelation. The theory's prediction (extreme fidelity weighting should suppress benefit) is correct; the test simply failed to create the intended condition. This does not affect any other result, but it means the C4 boundary for ensemble mechanisms remains empirically untested.

### C2 steepness threshold

The gap between sigmoid (works) and polynomial (does not work) identifies a transition in C2, but the computational work does not quantify the minimum steepness required. The connection to Chapeau-Blondeau's (1997) generalized SR for arbitrary nonlinearities suggests that the critical feature is the slope at threshold relative to noise variance. A full characterization of the C2 boundary would require systematic variation of steepness across detector types — a natural extension of the current simulation framework.

### Crawford-Sobel simplifications

The CS simulation implements the standard one-sender-one-receiver model with uniform priors. Real organizational hierarchies involve multiple senders, multiple receivers, heterogeneous priors, repeated interaction, and reputation effects. The sharp babbling transition at $b \geq 0.087$ is a property of the one-shot uniform model; multi-round games with reputation produce smoother degradation. The simulation validates the qualitative mechanism (bias degrades communication monotonically toward babbling), not the quantitative threshold.

## 5.4 Open Questions

### O1: Multi-level cascades

The theorem covers two levels. Organizations have many. If noise at level 1 benefits level 2 via Theorem 1, does level 2's now-noisier output benefit level 3 via a second application of the theorem? The conditions may compose, but the noise at level 2 is no longer exogenous — it is a function of the level-1 noise that was itself optimized. The interaction between optimized noise at adjacent levels is not addressed.

This connects to Rosas et al. (2024) on hierarchical emergence via computational mechanics: strong lumpability of $\epsilon$-machines as the composition criterion for multi-level information processing. If the level-crossing at each interface satisfies lumpability, the cascade may compose cleanly. If not, the optimal noise at each level may interfere.

### O2: Dynamic environments

The theorem assumes static conditions — the threshold $\theta$, the signal $s$, and the degradation profile $P_N(\sigma)$ do not change during the noise sweep. In non-stationary environments where $\theta$ shifts, the forbidden interval shifts, and a previously beneficial noise level may become harmful. The rate at which the system can track a moving threshold and adapt $\sigma^*$ is an open control-theoretic question.

### O3: Quantitative $\sigma^*$ prediction

The theorem guarantees the *existence* of $\sigma^* > 0$ but does not provide a closed-form expression for its value. For threshold detectors, the Gammaitoni formula $D_{opt} = \Delta V / 2$ from the Kramers escape rate applies when the barrier height $\Delta V$ is known. For general nonlinearities and non-SR mechanisms, no analogue exists. The empirical calibration approach ($\sigma = k \cdot f(\text{operating point})$ where $k$ is tuned on one system and tested on others) is currently the most defensible prescription. A general closed-form would move the theory from "noise helps here" to "add exactly this much."

### O4: Connection to causal emergence

Hoel et al. (2013) proved that effective information can peak at the macro level when coarse-graining reduces noise and degeneracy. This is a parallel formalization of level-crossing: causal emergence $> 0$ if and only if the macro mechanisms are more deterministic and less degenerate than the micro mechanisms. The relationship between our Theorem 1 (conditions under which noise at level N benefits level N+1) and Hoel's result (conditions under which macro-level causation is stronger than micro-level causation) is suggestive but unproved. If Theorem 1 can be derived as a special case of positive causal emergence, the framework gains a deeper foundation. If not, the two results may address different aspects of level-crossing.

### O5: Tightening C4

The integral formulation of C4 ($\exists \sigma: \beta \cdot \Delta P_{N+1}(\sigma) > \alpha \cdot \Delta P_N(\sigma)$) is correct but hard to verify a priori without running the noise sweep. The original marginal condition ($\beta \cdot \partial P_{N+1}/\partial\sigma > \alpha \cdot |\partial P_N/\partial\sigma|$ at $\sigma \to 0^+$) is easier to check but fails for threshold detectors where benefit onset is delayed. A condition that is both easy to check and correct for all detector types would be practically valuable. The gap between the two formulations is an algebraic question about the relationship between $\beta/\alpha$, the degradation rate $L$, and the detector's response-onset profile.

### O6: The Bjork mechanism

The inverted-U in desirable difficulties (Bjork and Bjork 2011) — where introducing friction improves long-term retention at the cost of short-term performance — resembles the theorem's structure but may operate through a different mechanism than threshold detection. The "storage strength / retrieval strength" dissociation is not obviously a Jensen-gap phenomenon. If it is (storage strength provides a locally convex response to retrieval difficulty), the inverted-U universality claim extends to learning. If it is not, the framework describes a *family* of related theorems — all featuring suboptimality + nonlinearity + noise — rather than a single theorem. Distinguishing these cases requires formalizing Bjork's model information-theoretically, which is an open problem in the learning sciences.

## 5.5 What a Null Result Means

The framework makes precise predictions. Some of those predictions will fail. When they do, the failure is informative:

- If the activation space SR experiment (Section 4.5) fails to show benefit at high collinearity, the 7B degradation is **geometric collapse** (the domain-specific information is genuinely absent from the orthogonal complement), not threshold failure (the information is there but too weak to detect). This redirects the engineering effort: nonlinear decomposition methods rather than noise injection.

- If the inverted-U fails to appear in a system where all five conditions appear to be met, the most likely explanation is that one of the conditions was incorrectly assessed — typically C2 (the nonlinearity is not steep enough) or C4 (the cost-benefit ratio does not actually favor the higher level). The conditions are checkable, and the failure mode is diagnosable.

- If noise helps at 0.5B (C1 not met), the mechanism is regularization or dithering, not subthreshold signal detection. The framework is not falsified — dithering is a well-known phenomenon — but the activation space instance would not validate Theorem 1 specifically. The theory would lose one piece of evidence, not its formal structure.

The framework is designed to be wrong precisely. A theory that cannot be wrong cannot be useful. The falsification conditions (F1-F5) and the null-result interpretations above are the theory's commitment to testability.
