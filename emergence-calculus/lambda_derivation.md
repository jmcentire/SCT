# Derivation of the Contraction Parameter lambda

## Goal

Derive lambda < 1 for the Q-dynamics operator Phi_A from the properties of the scoring function and lifecycle rules, rather than assuming it.

## Setup: Gaussian Idealization

Output space R^d. Model the acceptability distribution as Gaussian:
Q_t = N(mu_t, Sigma_t).

The dynamics have two components:
1. **Scoring-selection step**: reweight Q_t by the scoring function kappa, renormalize
2. **Lifecycle step**: inject variance from fork/external signals/agent non-determinism

### The Scoring-Selection Operator

The scoring function kappa is modeled as Gaussian with precision Lambda (positive definite) centered at mu_* (the scoring target):

    kappa(o) = C * exp(-1/2 (o - mu_*)^T Lambda (o - mu_*))

Under convergent selection: mu_* = mu_F (Frame center).
Under mixed selection: mu_* depends on the selection balance beta.

The reweighted distribution is:

    Q_{score}(o) = kappa(o) Q_t(o) / Z_t

Since both are Gaussian, the product is Gaussian:

    kappa(o) * q_t(o) proportional to exp(-1/2 [o^T(Lambda + Sigma_t^{-1})o - 2o^T(Lambda mu_* + Sigma_t^{-1} mu_t) + const])

Completing the square:

    Q_{score} = N(mu_{score}, Sigma_{score})

where:

    Sigma_{score}^{-1} = Sigma_t^{-1} + Lambda
    mu_{score} = Sigma_{score} (Sigma_t^{-1} mu_t + Lambda mu_*)

### The Lifecycle Operator

Fork, external signals, and agent non-determinism inject variance. Model as additive:

    Q_{t+1} = N(mu_{t+1}, Sigma_{t+1})

where:

    mu_{t+1} = mu_{score}
    Sigma_{t+1} = Sigma_{score} + Sigma_L

with Sigma_L > 0 (positive definite) representing lifecycle variance injection. This is bounded by dissipation (WF3): lifecycle cannot inject unbounded variance.

### The Composed Map T

The full update T: (mu, Sigma) -> (mu', Sigma') is:

    Sigma' = (Sigma^{-1} + Lambda)^{-1} + Sigma_L
    mu' = (Sigma^{-1} + Lambda)^{-1} (Sigma^{-1} mu + Lambda mu_*)

## Derivation in 1D (d = 1)

For clarity, start with d = 1. All matrices become scalars. Write sigma^2 for variance, lambda_k for the scoring precision (1/sigma^2_kappa), and sigma^2_L for lifecycle variance.

### The update map

    sigma'^2 = sigma^2 * sigma^2_kappa / (sigma^2 + sigma^2_kappa) + sigma^2_L
    mu' = mu * sigma^2_kappa / (sigma^2 + sigma^2_kappa) + mu_* * sigma^2 / (sigma^2 + sigma^2_kappa)

Let delta = mu - mu_* (deviation from scoring target). Then:

    delta' = mu' - mu_* = (mu - mu_*) * sigma^2_kappa / (sigma^2 + sigma^2_kappa) = delta * sigma^2_kappa / (sigma^2 + sigma^2_kappa)

So:

    delta' = delta * r(sigma^2)    where r(sigma^2) = sigma^2_kappa / (sigma^2 + sigma^2_kappa)

### The Jacobian

The map T: (delta, sigma^2) -> (delta', sigma'^2) has Jacobian:

    J = | dr(delta')/d(delta)    dr(delta')/d(sigma^2)  |
        | d(sigma'^2)/d(delta)   d(sigma'^2)/d(sigma^2) |

Computing each entry:

    d(delta')/d(delta) = r(sigma^2) = sigma^2_kappa / (sigma^2 + sigma^2_kappa)

    d(delta')/d(sigma^2) = delta * d/d(sigma^2)[sigma^2_kappa/(sigma^2 + sigma^2_kappa)]
                         = -delta * sigma^2_kappa / (sigma^2 + sigma^2_kappa)^2

    d(sigma'^2)/d(delta) = 0    [variance update does not depend on mean]

    d(sigma'^2)/d(sigma^2) = d/d(sigma^2)[sigma^2 sigma^2_kappa / (sigma^2 + sigma^2_kappa)]
                           = sigma^2_kappa^2 / (sigma^2 + sigma^2_kappa)^2
                           = r(sigma^2)^2

So:

    J = | r          -delta * r / (sigma^2 + sigma^2_kappa) |
        | 0          r^2                                     |

where r = sigma^2_kappa / (sigma^2 + sigma^2_kappa).

### Eigenvalues

J is upper triangular. The eigenvalues are the diagonal entries:

    eigenvalue_1 = r = sigma^2_kappa / (sigma^2 + sigma^2_kappa)
    eigenvalue_2 = r^2 = (sigma^2_kappa / (sigma^2 + sigma^2_kappa))^2

Both are in (0, 1) since sigma^2 > 0 and sigma^2_kappa > 0.

### Spectral radius

    rho(J) = max(eigenvalue_1, eigenvalue_2) = eigenvalue_1 = r

since r^2 < r for r in (0, 1).

### THE CONTRACTION PARAMETER

    lambda = sigma^2_kappa / (sigma^2 + sigma^2_kappa) = 1 / (1 + sigma^2/sigma^2_kappa)

This is strictly less than 1 whenever sigma^2 > 0 (the distribution has positive variance) and sigma^2_kappa > 0 (the scoring function has finite variance, i.e., positive curvature).

## Fixed Point

At the fixed point, sigma'^2 = sigma^2 = sigma^2_*:

    sigma^2_* = sigma^2_* sigma^2_kappa / (sigma^2_* + sigma^2_kappa) + sigma^2_L

Rearranging:

    sigma^2_* - sigma^2_L = sigma^2_* sigma^2_kappa / (sigma^2_* + sigma^2_kappa)
    (sigma^2_* - sigma^2_L)(sigma^2_* + sigma^2_kappa) = sigma^2_* sigma^2_kappa
    sigma^2_*^2 + sigma^2_* sigma^2_kappa - sigma^2_L sigma^2_* - sigma^2_L sigma^2_kappa = sigma^2_* sigma^2_kappa
    sigma^2_*^2 - sigma^2_L sigma^2_* - sigma^2_L sigma^2_kappa = 0

Quadratic in sigma^2_*:

    sigma^2_* = (sigma^2_L + sqrt(sigma^2_L^2 + 4 sigma^2_L sigma^2_kappa)) / 2

Taking the positive root. This is always positive when sigma^2_L > 0 and sigma^2_kappa > 0.

### Contraction rate at the fixed point

    lambda_* = sigma^2_kappa / (sigma^2_* + sigma^2_kappa)
             = 2 sigma^2_kappa / (sigma^2_L + sqrt(sigma^2_L^2 + 4 sigma^2_L sigma^2_kappa) + 2 sigma^2_kappa)

### Properties of lambda_*

1. **lambda_* in (0, 1)** always, when sigma^2_L > 0 and sigma^2_kappa > 0. PROVED.

2. **Dependence on scoring precision (1/sigma^2_kappa):**
   - As sigma^2_kappa -> 0 (sharp scoring): lambda_* -> 0 (strong contraction)
   - As sigma^2_kappa -> infinity (flat scoring): lambda_* -> 1 (weak contraction)
   Sharp scoring = fast convergence.

3. **Dependence on lifecycle variance sigma^2_L:**
   - As sigma^2_L -> 0: sigma^2_* -> 0, lambda_* -> 1 (contraction degenerates)
     This is the degenerate case: no variance injection means the distribution collapses
     to a point mass. Contraction requires something to contract.
   - As sigma^2_L -> infinity: sigma^2_* -> infinity, lambda_* -> 0 (strong contraction)
     But Q* has huge variance — the system is noisy.

4. **Dependence on strategic RDP parameters:**
   - The perception tolerance P constrains sigma^2_kappa: tighter P -> smaller sigma^2_kappa -> smaller lambda_*
   - The bias b shifts mu_* away from the "true" mean, affecting the fixed-point mean delta_*
     but NOT the contraction rate (lambda depends on variances only, not means)

## The Domain of Validity

lambda_* < 1 requires:

(V1) sigma^2_kappa < infinity: the scoring function has positive curvature (is not flat).
     Equivalently: the scoring precision Lambda > 0.

(V2) sigma^2_L > 0: lifecycle injects positive variance.
     Equivalently: fork, external signals, or agent non-determinism add noise.

(V3) sigma^2_* < infinity: the fixed-point variance is finite.
     This follows from (V1) and (V2) via the quadratic formula.

Under these three conditions, WF4 is a THEOREM for the Gaussian case, with an explicit
contraction parameter lambda_* = sigma^2_kappa / (sigma^2_* + sigma^2_kappa).

## Extension to d > 1

For d-dimensional output space, the Jacobian is a 2-block upper triangular matrix (in the (delta, Sigma) parameterization). The mean block contracts with rate:

    lambda_mean = max eigenvalue of (I + Sigma Lambda)^{-1}

For the covariance block, the map Sigma -> (Sigma^{-1} + Lambda)^{-1} + Sigma_L has Jacobian with spectral radius:

    lambda_cov = (max eigenvalue of (I + Sigma Lambda)^{-1})^2

Since the full Jacobian is upper triangular (covariance does not depend on mean), the spectral radius is:

    rho(J) = max(lambda_mean, lambda_cov) = lambda_mean

because lambda_cov = lambda_mean^2 < lambda_mean for lambda_mean in (0, 1).

The contraction rate in d dimensions is:

    lambda_* = 1 / (1 + lambda_min(Sigma_* Lambda))

where lambda_min denotes the smallest eigenvalue. For commuting Sigma_* and Lambda:
lambda_min(Sigma_* Lambda) = lambda_min(Sigma_*) * lambda_min(Lambda) > 0.

For non-commuting case: lambda_min(Sigma_* Lambda) = lambda_min(Sigma_*^{1/2} Lambda Sigma_*^{1/2}) >= lambda_min(Sigma_*) lambda_min(Lambda) > 0.

(Proof: lambda_min(A^{1/2} B A^{1/2}) = min_x x^T A^{1/2} B A^{1/2} x / x^T x.
Substituting y = A^{1/2} x: = min_y y^T B y / y^T A^{-1} y >= lambda_min(B) * min_y y^T y / y^T A^{-1} y = lambda_min(B) / lambda_max(A^{-1}) = lambda_min(B) * lambda_min(A).)

So lambda_* < 1 in d dimensions under the same conditions (V1)-(V3).

## Connection to the Mesh

The mesh familiarity scoring function is:

    F(signal, context) = sum(w_i * c_i) / sum(w_i)

with 6 components weighted (0.35, 0.20, 0.15, 0.10, 0.10, 0.10).

This is NOT Gaussian. It's a weighted sum of:
- Cosine similarity (bounded, continuous)
- Jaccard overlap (bounded, continuous)
- Empirical frequencies (bounded, discrete)
- Exponential decay (bounded, continuous)
- Crawford-Sobel credibility (bounded, discrete)

The Gaussian model captures the STRUCTURE of the update but not the exact functional form. The mapping from mesh parameters to Gaussian parameters is:

- sigma^2_kappa relates to the "sharpness" of the familiarity scoring function — how quickly scores decrease as signals move away from the worker's specialization. For the embedding similarity component (weight 0.35, cosine similarity), this is determined by the embedding dimension and the concentration of the centroid.

- sigma^2_L relates to the rate at which fork and new signal arrival inject variance into the distribution of worker specializations. Fork creates a new worker that starts with half the parent's signals (lower fullness, lower vigilance, more accepting of diverse signals). New signal arrival shifts centroids.

- The vigilance threshold rho acts as a hard gate, creating a threshold nonlinearity. Signals below rho are rejected entirely. In the Gaussian idealization, this corresponds to truncation of the scoring function below a cutoff — which makes kappa non-Gaussian (it's a truncated Gaussian).

## The Threshold Effect

The adaptive vigilance rho(fullness) = 0.15 + 0.65 * fullness^2 creates a state-dependent threshold. This modifies the contraction analysis:

For signals above threshold (accepted):
- The effective scoring function is kappa(o) * I(F(o) >= rho), where I is the indicator
- The truncation makes the scoring MORE concentrated (fewer signals accepted)
- This INCREASES the effective scoring precision -> DECREASES lambda

For signals below threshold (rejected):
- They are routed to the next worker or dropped
- They do not affect the current worker's centroid
- This means the current worker's distribution is insulated from far-away signals

The threshold effect STRENGTHENS contraction, not weakens it. The Gaussian derivation gives an UPPER BOUND on lambda: the actual lambda is at most the Gaussian lambda, because truncation only concentrates further.

## What Needs Empirical Verification

The Gaussian derivation gives lambda_* as a function of (sigma^2_kappa, sigma^2_L). These are idealized parameters. The empirical studies should:

1. **Estimate sigma^2_kappa from the mesh**: Run the mesh with known input distributions. Measure how the familiarity score decreases with distance from the centroid. Fit a Gaussian to the score profile. The fitted sigma^2_kappa is the effective scoring variance.

2. **Estimate sigma^2_L from the mesh**: Measure the variance injection from fork events and new signal acceptance. Compute the per-step variance added to the distribution of worker centroids.

3. **Measure lambda directly**: Run the mesh from multiple initial conditions on the same input stream. Track W_1(Q_t, Q_t') between runs. Fit lambda from the decay rate.

4. **Compare predicted vs. measured lambda**: Use the estimated (sigma^2_kappa, sigma^2_L) to predict lambda_* from the formula. Compare with the directly measured lambda. If they agree, the Gaussian idealization is adequate. If they disagree, the non-Gaussian effects (threshold, discrete components) matter.

## Summary

**Theorem (Gaussian Contraction).** For the Q-dynamics map T: (mu, Sigma) -> (mu', Sigma') with Gaussian scoring (precision Lambda > 0) and lifecycle variance injection (Sigma_L > 0):

(i) T has a unique fixed point (mu_*, Sigma_*) where mu_* = the scoring center and Sigma_* is the positive root of Sigma^2 - Sigma_L Sigma - Sigma_L sigma^2_kappa = 0 (in 1D).

(ii) The Jacobian of T at the fixed point has spectral radius lambda_* = sigma^2_kappa / (sigma^2_* + sigma^2_kappa) < 1.

(iii) For any initial Q_0 = N(mu_0, sigma^2_0) in a neighborhood of the fixed point, Q_t -> Q_* at geometric rate lambda_*.

(iv) lambda_* is explicitly:

    lambda_* = 2 sigma^2_kappa / (sigma^2_L + sqrt(sigma^2_L^2 + 4 sigma^2_L sigma^2_kappa) + 2 sigma^2_kappa)

(v) lambda_* < 1 iff sigma^2_kappa < infinity and sigma^2_L > 0.

**This is a theorem, not an assumption.**

The conditions (V1: positive scoring curvature, V2: positive lifecycle variance, V3: finite fixed-point variance) are checkable properties of the constraint architecture. WF4 is not an axiom added to the well-formedness conditions. It is a consequence of the scoring function having curvature and the lifecycle injecting variance.
