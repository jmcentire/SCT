# Derivation of the Contraction Parameter lambda (v2)

## Revision Notes

v2 incorporates all criticisms from three antagonistic review agents. Key changes:
- Global contraction proved directly for 1D (not just local via Jacobian)
- Gaps between Gaussian model and actual mesh honestly classified
- Mixture (multi-worker) gap explicitly acknowledged and bridging path stated
- Fork modeled as state-dependent perturbation, not constant additive noise
- Threshold/truncation claim downgraded from "proved" to "conjectured with supporting argument"
- d > 1 covariance Jacobian claim weakened to bound (not equality)

---

## 1. The 1D Gaussian Model (Fully Rigorous)

### 1.1 Setup

Output space R. Acceptability distribution Q_t = N(mu_t, sigma^2_t).

**Scoring function:** kappa(o) proportional to exp(-o^2 / (2 sigma^2_kappa)), centered at mu_* = 0 (WLOG by translation).

**Lifecycle variance injection:** sigma^2_L > 0, representing the aggregate effect of fork events, external signal arrival, and agent non-determinism.

**The composed map T:** (delta, sigma^2) -> (delta', sigma'^2) where delta = mu - mu_*:

    delta' = delta * r(sigma^2)       where r(sigma^2) = sigma^2_kappa / (sigma^2 + sigma^2_kappa)
    sigma'^2 = h(sigma^2)             where h(sigma^2) = sigma^2 sigma^2_kappa / (sigma^2 + sigma^2_kappa) + sigma^2_L

(Derived from the Gaussian product formula. Verified independently by three reviewers.)

### 1.2 Global Contraction of the Variance Map

**Proposition 1.** The variance map h: (0, infinity) -> (0, infinity) is a global contraction on any invariant interval [sigma^2_L, M].

*Proof.* Compute h'(sigma^2) for all sigma^2 > 0:

    h'(sigma^2) = sigma^2_kappa^2 / (sigma^2 + sigma^2_kappa)^2 = r(sigma^2)^2

For all sigma^2 > 0: 0 < r(sigma^2) < 1, so 0 < h'(sigma^2) < 1.

Therefore h is strictly increasing with derivative strictly less than 1 everywhere on (0, infinity). By the mean value theorem, for any sigma^2_a, sigma^2_b > 0:

    |h(sigma^2_a) - h(sigma^2_b)| = |h'(xi)| |sigma^2_a - sigma^2_b| < |sigma^2_a - sigma^2_b|

for some xi between sigma^2_a and sigma^2_b. This is the strict contraction condition.

Additionally, h maps (0, infinity) into [sigma^2_L, infinity) since h(sigma^2) >= sigma^2_L for all sigma^2 > 0. And h(sigma^2) < sigma^2_kappa + sigma^2_L for all sigma^2 > 0 (since sigma^2 sigma^2_kappa / (sigma^2 + sigma^2_kappa) < sigma^2_kappa). So h maps the compact interval [sigma^2_L, sigma^2_kappa + sigma^2_L] into itself.

The restriction of h to [sigma^2_L, sigma^2_kappa + sigma^2_L] is a contraction on a complete metric space (closed interval with the absolute value metric). By Banach's theorem, h has a unique fixed point sigma^2_* in this interval, and h^n(sigma^2_0) -> sigma^2_* for any sigma^2_0 in the interval.

**Moreover**, since h maps (0, infinity) into [sigma^2_L, sigma^2_kappa + sigma^2_L], any initial sigma^2_0 > 0 enters the invariant interval after one step. So convergence is global on (0, infinity). []

**Remark.** This is a GLOBAL contraction argument, not a Jacobian/local argument. It does not require computing eigenvalues. The contraction holds everywhere, not just near the fixed point.

### 1.3 Fixed Point (Explicit)

The fixed-point equation h(sigma^2_*) = sigma^2_* gives the quadratic:

    sigma^2_*^2 - sigma^2_L sigma^2_* - sigma^2_L sigma^2_kappa = 0

Unique positive root:

    sigma^2_* = (sigma^2_L + sqrt(sigma^2_L^2 + 4 sigma^2_L sigma^2_kappa)) / 2

Verified: discriminant > 0 since sigma^2_L, sigma^2_kappa > 0. The negative root is negative (infeasible). Uniqueness confirmed.

### 1.4 Contraction Rate at the Fixed Point

The contraction rate for the variance map at the fixed point is:

    lambda_var = h'(sigma^2_*) = r(sigma^2_*)^2 = (sigma^2_kappa / (sigma^2_* + sigma^2_kappa))^2

The contraction rate for the mean map, once sigma^2 has converged to sigma^2_*, is:

    lambda_mean = r(sigma^2_*) = sigma^2_kappa / (sigma^2_* + sigma^2_kappa)

Since lambda_var = lambda_mean^2 < lambda_mean, the dominant contraction rate is:

    lambda_* = lambda_mean = sigma^2_kappa / (sigma^2_* + sigma^2_kappa)

### 1.5 Explicit Lambda

    lambda_* = 2 sigma^2_kappa / (sigma^2_L + sqrt(sigma^2_L^2 + 4 sigma^2_L sigma^2_kappa) + 2 sigma^2_kappa)

**Theorem 1 (1D Gaussian Contraction).** For the Q-dynamics map T with Gaussian scoring (precision 1/sigma^2_kappa > 0) and lifecycle variance injection (sigma^2_L > 0):

(i) T has a unique fixed point (delta_* = 0, sigma^2_*) with sigma^2_* given by the quadratic formula above.

(ii) The variance map h is a global contraction on (0, infinity) with sigma^2_t -> sigma^2_* for any initial sigma^2_0 > 0.

(iii) Once sigma^2 has converged, the mean map contracts delta at geometric rate lambda_* = sigma^2_kappa / (sigma^2_* + sigma^2_kappa) < 1.

(iv) lambda_* < 1 whenever sigma^2_kappa > 0 (scoring has curvature) and sigma^2_L > 0 (lifecycle injects variance).

(v) The full trajectory (delta_t, sigma^2_t) converges to (0, sigma^2_*) from any initial condition (delta_0, sigma^2_0) with sigma^2_0 > 0.

*Proof of (v).* By (ii), sigma^2_t -> sigma^2_*. Once sigma^2_t is within epsilon of sigma^2_*, the mean contraction rate satisfies |r(sigma^2_t)| <= lambda_* + O(epsilon). So |delta_t| decreases geometrically at a rate approaching lambda_*. The combined convergence follows. []

**Remark on global vs. local.** Statement (v) is GLOBAL convergence, proved directly from h'(s) < 1 for all s > 0. The Jacobian analysis (eigenvalues r and r^2) gives the same lambda_* but only proves local convergence. The global argument is stronger and simpler.

### 1.6 Parameter Dependence

| Parameter change | Effect on lambda_* | Interpretation |
|---|---|---|
| sigma^2_kappa -> 0 (sharper scoring) | lambda_* -> 0 | Stronger selection = faster convergence |
| sigma^2_kappa -> infinity (flatter scoring) | lambda_* -> 1 | Weaker selection = slower convergence |
| sigma^2_L -> 0 (less lifecycle noise) | lambda_* -> 1 | Less variance injection = fixed point approaches degeneracy |
| sigma^2_L -> infinity (more lifecycle noise) | lambda_* -> 0 | More noise = faster contraction, but Q* has high variance |

**Quantitative bound for convergence time.** The system reaches W_1(Q_t, Q_*) < epsilon when t > log(epsilon / W_1(Q_0, Q_*)) / log(lambda_*). For lambda_* = 0.9, reaching epsilon = 0.01 from a distance of 1 requires t > 44 steps. For lambda_* = 0.99, t > 460 steps.

---

## 2. Extension to d > 1 (Partial)

### 2.1 What Extends Directly

The mean contraction generalizes cleanly. In d dimensions:

    delta' = (I + Sigma Lambda)^{-1} delta

where Sigma and Lambda are d x d positive definite matrices. The eigenvalues of (I + Sigma Lambda)^{-1} are 1/(1 + gamma_i) where gamma_i are the eigenvalues of Sigma^{1/2} Lambda Sigma^{1/2} (which is symmetric PD, hence has real positive eigenvalues, and is similar to Sigma Lambda). The contraction rate for the mean is:

    lambda_mean = max_i 1/(1 + gamma_i) = 1/(1 + gamma_min)

where gamma_min = lambda_min(Sigma^{1/2} Lambda Sigma^{1/2}) >= lambda_min(Sigma) lambda_min(Lambda) > 0.

So lambda_mean < 1 in d dimensions under the same conditions.

### 2.2 What Requires Additional Work

The covariance map in d dimensions is Sigma -> (Sigma^{-1} + Lambda)^{-1} + Sigma_L. The Frechet derivative of f(Sigma) = (Sigma^{-1} + Lambda)^{-1} is:

    Df(Sigma)[H] = (Sigma^{-1} + Lambda)^{-1} Sigma^{-1} H Sigma^{-1} (Sigma^{-1} + Lambda)^{-1}

This is a linear map on the space of d x d symmetric matrices. Its spectral radius equals the square of the mean contraction rate WHEN Sigma and Lambda commute (simultaneously diagonalizable), because the problem decouples into d independent 1D problems. For the non-commuting case, the spectral radius is bounded above by lambda_mean^2, but the exact value requires computing the operator norm of Df, which depends on the chosen matrix norm.

**Status:** The d > 1 covariance contraction is proved for the commuting case and bounded for the general case. Full proof for non-commuting case requires additional analysis of the Frechet derivative, or alternatively a direct Lyapunov argument on the matrix Riccati-type equation.

### 2.3 Global Contraction in d > 1

The 1D global argument (h'(s) < 1 for all s > 0) does not trivially extend. In d dimensions, the covariance map acts on the cone of positive definite matrices. Global contraction on this cone requires showing that the map is a contraction in a suitable metric (Thompson metric, Bures metric, or trace-class operator norm). This is plausible but unproved in the current derivation.

**Status:** Global contraction in d > 1 is an open problem. Local contraction (spectral radius of Jacobian < 1) is established. The gap between local and global is the same gap as in the 1D case but without the easy fix (monotone 1D argument does not generalize to matrix-valued maps).

---

## 3. Gaps Between the Gaussian Model and the Actual Mesh

These are HONEST acknowledgments of where the Gaussian model does not capture the actual mesh dynamics. Each gap is classified by severity and bridging path.

### 3.1 Scoring Function Is Not Gaussian (BRIDGEABLE)

The mesh familiarity function is a weighted sum of cosine similarity, Jaccard overlap, empirical frequencies, exponential decay, and credibility scores. This is bounded in [0, 1], continuous, and (for the embedding component) has positive curvature near the centroid.

**The gap:** The Gaussian product formula (Q_{t+1} proportional to kappa * Q_t) requires both factors to be Gaussian for closed-form updates. With a non-Gaussian kappa, the product is not Gaussian.

**Bridging path:** Near the fixed point, the second-order Taylor expansion of log(kappa) at its maximum gives an effective Gaussian scoring function with precision Lambda_eff = -D^2 log(kappa). The contraction rate depends on Lambda_eff to leading order, with higher-order corrections. The Gaussian lambda_* is an approximation to the true contraction rate, valid when Q_t is concentrated near the scoring maximum (i.e., near the fixed point).

**Status:** Local approximation is standard but unproved for this specific system.

### 3.2 Competitive Allocation vs. Gibbs Reweighting (BRIDGEABLE)

The mesh uses winner-take-all routing (BFS order, first worker above vigilance threshold accepts). The proof models this as Gibbs reweighting (proportional resampling by scoring function).

**The gap:** Winner-take-all is path-dependent and order-dependent. Gibbs reweighting is not.

**Bridging path:** In the mean-field limit (large worker population), the competitive allocation converges to the Gibbs reweighting. This is a standard result in evolutionary game theory / mean-field games. The correction term scales as O(1/K) where K is the number of workers.

**Status:** Standard technique; application to this specific system is unproved.

### 3.3 Fork Creates Mixtures, Not Additive Noise (CRITICAL, BRIDGEABLE)

Fork splits one worker into two with different centroids. This creates a mixture component, injecting between-component variance that depends on the current state.

**The gap:** The constant Sigma_L model does not capture state-dependent variance injection. The between-component variance from fork can be much larger or smaller than Sigma_L depending on how the signals partition.

**Bridging path (moment-closure):** Define mu_t and Sigma_t as the mean and covariance of the MIXTURE of all workers. Show that fork events perturb the mixture moments by amounts bounded (in expectation) by a constant that depends on the fork threshold and the signal distribution. The key constraint: fork is triggered when signal_count > 100 (volume) or avg_familiarity < 0.50 (coherence degradation). The coherence trigger means fork fires when within-cluster variance is large, which bounds the between-component variance injected by the split.

Specifically: if a worker forks because avg_familiarity < 0.50, its signals are poorly concentrated. The resulting children's centroids will differ by an amount proportional to the within-cluster spread. The between-component variance after fork is O(sigma^2_within), which is bounded by the coherence threshold. This gives a state-dependent but bounded Sigma_L(state) <= C for some constant C determined by the coherence threshold.

**Status:** Argument is plausible but requires proof. The key inequality (between-component variance bounded by coherence threshold) needs to be derived from the familiarity scoring function.

### 3.4 Q_t Is a Mixture, Not a Single Gaussian (FUNDAMENTAL, BRIDGEABLE)

The actual Q_t is a mixture over K workers. The Gaussian model is a moment-closure approximation.

**The gap:** The dynamics of a mixture (changing number of components, varying weights, correlated component updates via routing) differ qualitatively from the dynamics of a single Gaussian.

**Bridging path:** The Wasserstein contraction of the full mixture dynamics requires showing that the operator Phi_A contracts in W_1 on the space of all mixture distributions. The argument structure:

(a) Each component (worker) contracts toward its local mean at rate lambda_* (from the 1D Gaussian theorem, applied to each worker's distribution of accepted signals).

(b) The routing ensures that the local means converge to the means of the Voronoi cells of the input distribution.

(c) Fork/merge/decay adjust the number of components to match the number of modes in the input distribution.

(d) The overall W_1 distance between two mixture distributions decreases because each component contracts and the component structure converges.

This is a multi-scale contraction argument: fast contraction within components (the Gaussian theorem), slow convergence of the component structure (the ART stability result from Carpenter & Grossberg). The combined rate is dominated by the slow component.

**Status:** Unproved but the mathematical tools exist (mean-field theory, ART convergence, multi-scale analysis).

### 3.5 Truncation by Vigilance Threshold (SIGNIFICANT, CONJECTURED)

The vigilance threshold creates a hard gate. Signals below threshold are rejected. This makes the effective scoring function discontinuous.

**The gap:** The Gaussian product formula breaks at the truncation boundary. The resulting distribution is truncated Gaussian, not Gaussian.

**Conjecture:** Truncation strengthens contraction (increases effective precision, decreases lambda). Supporting argument: truncation removes the tails of the scoring function, concentrating the reweighted distribution further than the untruncated version would. The Gaussian lambda_* is therefore an upper bound on the actual contraction rate.

**Status:** Conjectured with supporting heuristic argument. Not proved. A counterexample would require the truncation boundary to interact pathologically with the fixed point. This is unlikely when the vigilance threshold is well below the scoring function's maximum (as in the mesh, where rho ranges from 0.15 to 0.80 and the scoring function peaks at 1.0).

### 3.6 Stationarity (BRIDGEABLE)

The proof assumes stationary input distribution. Real organizations change.

**Bridging path:** Standard tracking theorem for time-varying contractions. If the scoring parameters change at rate epsilon per step, the trajectory stays within O(epsilon / (1 - lambda_*)) of the instantaneous fixed point.

**Status:** Standard result; application is straightforward.

---

## 4. What Is Proved, What Is Conjectured, What Is Open

### PROVED (rigorous, all steps verified):

- **Theorem 1 (1D Gaussian Contraction):** For Gaussian scoring and constant additive lifecycle, the map T is a global contraction with explicit lambda_* = sigma^2_kappa / (sigma^2_* + sigma^2_kappa) < 1 whenever sigma^2_kappa > 0 and sigma^2_L > 0.

- **Explicit fixed point:** sigma^2_* = (sigma^2_L + sqrt(sigma^2_L^2 + 4 sigma^2_L sigma^2_kappa)) / 2.

- **Global convergence in 1D:** From any initial sigma^2_0 > 0, the variance converges to sigma^2_* and the mean converges to mu_* at geometric rate lambda_*.

- **d > 1 mean contraction:** lambda_mean = 1/(1 + lambda_min(Sigma^{1/2} Lambda Sigma^{1/2})) < 1.

### PROVED UNDER CONDITIONS (rigorous but scope-limited):

- **d > 1 covariance contraction for commuting case:** When Sigma and Lambda commute, the covariance map contracts at rate lambda_mean^2.

### CONJECTURED WITH SUPPORTING ARGUMENT:

- **Truncation strengthens contraction:** The vigilance threshold increases effective scoring precision. The Gaussian lambda_* is an upper bound on the actual contraction rate.

### OPEN (requires substantial additional work):

- **d > 1 global contraction:** The matrix-valued variance map plausibly contracts globally on the cone of positive definite matrices, but the 1D monotone argument does not generalize. Requires analysis in Thompson metric or direct Lyapunov argument.

- **Mixture (multi-worker) contraction:** Q_t is a mixture distribution. The Gaussian theorem applies to each component. The full system's contraction requires a multi-scale argument.

- **Fork as moment-bounded perturbation:** Fork injects state-dependent variance. The bound Sigma_L(state) <= C requires deriving from the coherence threshold.

- **Competitive allocation mean-field limit:** The mesh's winner-take-all routing converges to Gibbs reweighting in the large-worker limit.

- **d > 1 covariance contraction (non-commuting case):** Requires analysis of the Frechet derivative's operator norm or a direct Lyapunov argument.

---

## 5. Empirical Studies to Close the Gaps

### Study 1: Direct Lambda Estimation

**Method:** Run the mesh on the same signal stream from K >= 10 different random initial conditions (different initial worker topologies). At each time step, compute the empirical distribution Q_t (histogram of worker centroids in embedding space). Compute W_1(Q_t, Q_t') for all pairs of runs. Fit lambda from the decay: W_1(Q_t, Q_t') ~ lambda^t * W_1(Q_0, Q_0').

**Prediction:** lambda decreases geometrically. The fitted lambda should be < 1.

**What this tests:** Global contraction of the FULL system (mixture, competitive allocation, truncation, fork — all included). If the empirical lambda < 1, the Gaussian idealization captures the essential dynamics even though the details differ. If lambda >= 1 or the decay is non-geometric, the idealization fails.

**Feasibility:** Requires K * (signal count) mesh runs. Each run costs ~$1. Total: ~$10-50.

### Study 2: Predicted vs. Measured Lambda

**Method:** From the mesh's scoring function parameters, estimate sigma^2_kappa (effective scoring variance) by fitting a Gaussian to the familiarity score as a function of distance from centroid. Estimate sigma^2_L (effective lifecycle variance) from the per-step variance change in the distribution of worker centroids. Compute lambda_predicted from the formula. Compare with lambda_measured from Study 1.

**Prediction:** lambda_predicted and lambda_measured should agree within 10-20% if the Gaussian idealization is adequate. Systematic overestimation of lambda (lambda_predicted > lambda_measured) would confirm the truncation conjecture (truncation strengthens contraction).

**What this tests:** Validity of the Gaussian idealization. If the two lambdas agree, the Gaussian model is adequate. If they disagree, the non-Gaussian effects (truncation, competitive allocation, mixture structure) dominate.

**Feasibility:** Same runs as Study 1 plus post-hoc analysis of scoring function profiles. No additional cost.

### Study 3: Selection Balance and Cage Verification

**Method:** Vary the mesh's discovery/retrieval classification threshold (which controls beta, the selection balance). At beta ~ 0 (all signals treated as retrieval): measure whether Q_t collapses to repetitive, Frame-confirming findings (Cage prediction from Theorem F). At beta > 0 (discovery signals admitted): measure whether Q_t includes novel pattern types (generative preservation from Theorem G).

**Specific protocol:**
- Run 1 (beta ~ 0): Disable discovery classification. All signals treated as familiar retrieval.
- Run 2 (beta ~ 0.5): Normal operation with discovery/retrieval classification.
- Run 3 (beta ~ 1): Bias toward novelty — lower the familiarity threshold for discovery classification.
- Measure: entropy of the finding distribution, number of distinct finding categories, fraction of findings that reference patterns not present in any single signal.

**Prediction:** Run 1 produces low-entropy, repetitive findings (Cage). Run 2 produces moderate entropy with both confirmatory and novel findings. Run 3 produces high entropy but potentially incoherent findings (beyond the inverted-U peak).

**What this tests:** Theorem F (Cage as beta -> 0 attractor) and Theorem G (generative preservation at beta > 0).

**Feasibility:** Three mesh runs on the same data stream. Total: ~$3.

### Study 4: Convergence Rate vs. Scoring Curvature

**Method:** Vary the vigilance parameters (base_threshold, max_threshold) which control the effective scoring precision. Higher vigilance = sharper effective scoring = smaller effective sigma^2_kappa = smaller lambda_*. Measure convergence rate at each setting.

**Prediction:** lambda_measured should decrease monotonically with effective scoring precision. The relationship should follow the formula lambda_* = sigma^2_kappa / (sigma^2_* + sigma^2_kappa).

**What this tests:** The functional dependence of lambda on scoring parameters. If the formula holds, the Gaussian model captures the parametric structure correctly even if the exact functional form differs.

**Feasibility:** 5-10 mesh runs with different vigilance settings. Total: ~$5-10.

### Study 5: Fork Variance Injection

**Method:** Instrument the fork events. For each fork, record: parent centroid, child 1 centroid, child 2 centroid, parent variance, child 1 variance, child 2 variance. Compute the between-component variance ||mu_1 - mu_2||^2 / 4 and compare with the within-component variance (parent sigma^2).

**Prediction:** Between-component variance should be bounded by a constant related to the coherence threshold (0.50). Fork events triggered by volume (signal_count > 100) should inject less between-component variance than fork events triggered by coherence degradation (avg_familiarity < 0.50).

**What this tests:** Whether fork variance injection is bounded (required for the moment-closure bridge) and whether the coherence threshold provides the bounding mechanism.

**Feasibility:** Instrumentation of existing fork code. No additional runs needed — can be measured from any mesh deployment.

---

## 6. Summary

The 1D Gaussian contraction is a genuine theorem with an explicit, verified lambda. The extension to the actual mesh requires bridging five gaps, each addressable but unproved. The empirical studies are designed to test whether the bridges hold in practice. If Studies 1-2 show lambda_measured < 1 and lambda_predicted ~ lambda_measured, the Gaussian idealization is validated empirically even where the formal bridge is incomplete. If they disagree, the disagreement identifies which gaps are load-bearing.

The honest status is:

**WF4 for the Gaussian idealization:** THEOREM (Theorem 1, proved).
**WF4 for the actual mesh:** CONJECTURED with explicit derivation path and five testable predictions.
**WF4 for general constraint architectures:** OPEN; requires specifying the scoring function class and lifecycle rules.
