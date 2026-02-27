# The Strategic Rate-Distortion-Perception Tradeoff

## Bridging Blau-Michaeli Compression Theory and Crawford-Sobel Strategic Communication

Jeremy McEntire

Cage & Mirror Publishing | jmc@cageandmirror.com

---

## Abstract

Blau and Michaeli (2019) proved that lossy compression under a perceptual quality constraint creates a three-way tradeoff among rate, distortion, and perceptual quality. Crawford and Sobel (1982) proved that communication under preference divergence is endogenously lossy. This paper combines the two results. We generalize the Blau-Michaeli rate-distortion-perception (RDP) tradeoff to arbitrary target distributions Q (not only the source distribution p_X), prove that the tradeoff holds within strategic communication channels, and derive the organizational consequence as a theorem under four sufficient conditions.

Three results are established. Theorem A generalizes the RDP function to R(D, P, Q), proving monotonicity, convexity, and rate elevation for any target distribution Q satisfying a non-degeneracy condition. The feasibility boundary is characterized via optimal transport. Theorem B derives the closed-form Gaussian strategic RDP equilibrium, showing that perception constraints on the receiver's action alter the partition structure and create a formally defined generative residual. Theorem C states four sufficient conditions under which any organizational communication channel satisfies the strategic RDP tradeoff, upgrading the relationship between information-theoretic compression and organizational behavior from structural analogy to theorem under stated conditions.

The paper fills a specific gap in the literature: no existing result combines strategic misalignment (Le Treust-Tomala, Akyol) with a distributional conformity constraint (Blau-Michaeli). The "strategic rate-distortion-perception function" has not previously been defined or characterized.

**Keywords**: rate-distortion-perception, strategic communication, Crawford-Sobel, lossy compression, organizational information processing, generative residual

---

## 1. Introduction

### 1.1 The Problem

Consider a middle manager who receives a complex operational report and must produce an executive summary. The summary compresses continuous reality into a discrete representation. The manager's incentives do not perfectly align with the executive's (Crawford and Sobel, 1982). And the summary must "look right" — it must conform to institutional norms of format, tone, and content distribution. It must look like a legitimate executive summary, not like reality.

These three constraints — compression, strategic misalignment, and distributional conformity — interact. The compression is endogenous (created by the strategic channel, not by bandwidth limitations). The conformity constraint is distributional (the output must be drawn from a particular distribution of acceptable outputs). And the interaction between them forces the receiver to generate: to produce outputs that diverge from the source in systematic, predictable ways.

The information-theoretic foundations for each constraint exist independently. Shannon (1948) established rate-distortion theory. Crawford and Sobel (1982) characterized strategic communication equilibria. Blau and Michaeli (2018, 2019) proved the rate-distortion-perception tradeoff. Le Treust and Tomala (2020) formalized Crawford-Sobel as joint source-channel coding. Akyol (2026) unified strategic communication with rate-distortion theory. Xie, Lei, and Poor (2024) extended lossy source coding to prescribed output distributions.

No existing result, however, combines strategic misalignment with a distributional conformity constraint. The strategic rate-distortion-perception function — the minimum rate required to achieve a given distortion when the encoder and decoder have misaligned objectives and the decoder's output must conform to a target distribution — has not been defined, characterized, or proved to exhibit the Blau-Michaeli tradeoff structure.

This paper fills that gap.

### 1.2 Contribution

Three results are established:

**Theorem A** (Generalized RDP Tradeoff). The rate-distortion-perception function R(D, P, Q), where Q is an arbitrary target distribution for the reconstruction, exhibits the same structural properties as the standard Blau-Michaeli function R(D, P): monotonicity in D and P, convexity in D, and strict rate elevation when the perception constraint binds. The key new element is the feasibility boundary: when Q differs from the source distribution p_X, the feasible set is non-empty only when distortion exceeds a minimum determined by the optimal transport cost between p_X and Q.

**Theorem B** (Gaussian Strategic RDP Equilibrium). For Gaussian sources with quadratic distortion and KL-divergence perception constraint, the Nash equilibrium of the strategic communication game under perception constraint is characterized in closed form. The perception constraint alters the equilibrium partition structure and creates a formally defined generative residual — the excess distortion attributable to distributional conformity.

**Theorem C** (Organizational Sufficient Conditions). Four checkable conditions — positive rate, preference divergence, distributional acceptability, and non-degeneracy — are jointly sufficient for the strategic RDP tradeoff to hold in any organizational communication channel. Under these conditions, the relationship between information-theoretic compression and organizational behavior is a theorem, not an analogy.

### 1.3 Relation to the Literature

The bridge has been partially constructed from both sides.

From the strategic communication side: Le Treust and Tomala (2020) formalized Crawford-Sobel as joint source-channel coding with distinct sender/receiver distortion measures, deriving single-letter characterizations under Nash, Stackelberg, and cooperative equilibria. Akyol, Langbort, and Basar (2015) treated strategic communication with misaligned quadratic distortion as a hierarchical game. Akyol (2026) unified rate-distortion, Bayesian persuasion, and strategic communication, deriving a "strategic rate-distortion function" with closed-form Gaussian solutions via semantic waterfilling. Xiao, Zhang, Li, Shi, and Basar (2022) derived optimal encoding/decoding under Nash and Stackelberg equilibria with rate constraints.

From the perception side: Blau and Michaeli (2018) proved the two-way perception-distortion tradeoff. Blau and Michaeli (2019) extended it to the three-way rate-distortion-perception tradeoff. Matsumoto (2018) proved the tradeoff holds for arbitrary source distributions. Xie, Lei, and Poor (2024) characterized output-constrained lossy source coding for Q different from p_X, providing a Gaussian closed form and a coding theorem under common randomness. Wagner (2022) characterized the role of common randomness in the RDP tradeoff. Liu et al. (ICLR 2022) connected cross-domain compression to entropy-constrained optimal transport.

The gap is at the intersection: nobody has combined strategic misalignment with a perception constraint. Table 1 summarizes the landscape.

| Paper | Strategic? | Perception? | Q != p_X? |
|-------|-----------|------------|-----------|
| Blau-Michaeli (2019) | No | Yes | No |
| Crawford-Sobel (1982) | Yes | No | N/A |
| Le Treust-Tomala (2020) | Yes | No | N/A |
| Akyol (2026) | Yes | No | N/A |
| Xie et al. (2024) | No | Yes | Yes |
| Chai et al. (2023) | No | Yes | No |
| Saritas et al. (2023) | Yes | No | N/A |
| **This paper** | **Yes** | **Yes** | **Yes** |

### 1.4 Paper Structure

Section 2 establishes notation and reviews the three foundational results (Crawford-Sobel, Blau-Michaeli, and output-constrained coding). Section 3 defines the generalized and strategic RDP problems. Section 4 proves the three main results. Section 5 develops the organizational interpretation. Section 6 discusses limitations and extensions. Section 7 concludes.

---

## 2. Preliminaries

### 2.1 The Crawford-Sobel Model

A Sender (S) observes a state theta drawn from a prior distribution p_theta on Theta = [0, 1]. S transmits a costless message m in M to a Receiver (R), who takes action a in A. Utilities are:

    U_S(theta, a) = -(a - theta - b)^2
    U_R(theta, a) = -(a - theta)^2

where b > 0 is the bias parameter quantifying preference divergence.

**Proposition 2.1** (Crawford-Sobel, 1982). In any Nash equilibrium of this game, S's message partitions [0, 1] into at most N* intervals, where:

    N* = floor(-1/2 + sqrt(1/4 + 1/(2b)))

At b >= 1/4, N* = 1 (babbling equilibrium: the message carries zero information). For 0 < b < 1/4, the channel is endogenously lossy with rate R = log_2(N*) bits.

**Receiver's distortion** (MSE for uniform bins):

    D_R = 1/(12N*^2)

**Sender's effective distortion**: D_S = D_R + b^2 (approximately, for uniform bins).

### 2.2 The Blau-Michaeli Rate-Distortion-Perception Tradeoff

For a source X ~ p_X, reconstruction X-hat, distortion measure Delta, and divergence d:

**Definition 2.2** (Blau-Michaeli, 2019). The rate-distortion-perception function is:

    R(D, P) = min_{p(x-hat|x)} I(X; X-hat)
    subject to: E[Delta(X, X-hat)] <= D,  d(p_{X-hat}, p_X) <= P

**Theorem 2.3** (Blau-Michaeli, 2019). Under assumptions:

- (A1) d is convex in its second argument
- (A2) Delta is not a constant function

The function R(D, P) satisfies:

(i) R(D, P) is non-increasing in D and P.
(ii) R(D, P) is convex in D.
(iii) R(D, P) >= R(D) for all finite P, with equality only when the unconstrained rate-distortion optimizer already satisfies d(p_{X-hat*}, p_X) <= P.

### 2.3 Output-Constrained Lossy Source Coding

**Theorem 2.4** (Xie, Lei, and Poor, 2024). For source X ~ p_X and prescribed reconstruction distribution p_{X-hat} = Q, the minimum achievable rate under unlimited common randomness is:

    R(D, Q) = min_{p(u|x): p_{X-hat} = Q, E[Delta] <= D} I(X; U)

where U is an auxiliary random variable with X-hat = g(U) for some deterministic mapping g.

**Gaussian closed form.** For X ~ N(mu_X, sigma^2_X) and Q = N(mu_Q, sigma^2_Q), with MSE distortion and unlimited common randomness:

    D(R, infinity | p_X, Q) = (mu_X - mu_Q)^2 + sigma^2_X + sigma^2_Q - 2 sigma_X sigma_Q sqrt(1 - e^{-2R})

For limited common randomness rate R_c:

    D(R, R_c | p_X, Q) = (mu_X - mu_Q)^2 + sigma^2_X + sigma^2_Q - 2 sigma_X sigma_Q sqrt((1-e^{-2R})(1-e^{-2(R+R_c)}))

**Remark 2.5** (Common randomness). Operational achievability of R(D, P, Q) as an actual coding rate requires shared randomness between encoder and decoder. Without it, the achievable rate is strictly higher. The penalty is I(X-hat; U) - I(X; U), measuring the difficulty of synthesizing Q-distributed output from p_X-distributed source. Throughout this paper, results on R(D, P, Q) as an informational quantity hold unconditionally. Results on operational achievability assume unlimited common randomness unless stated otherwise. See Cuff (2013) and Wagner (2022) for the common randomness framework.

---

## 3. Problem Formulation

### 3.1 The Generalized RDP Function

**Definition 3.1** (Generalized Rate-Distortion-Perception Function). For source X ~ p_X, reconstruction X-hat, distortion measure Delta, divergence d, and target distribution Q:

    R(D, P, Q) = min_{p(x-hat|x)} I(X; X-hat)
    subject to: E[Delta(X, X-hat)] <= D,  d(p_{X-hat}, Q) <= P

When Q = p_X, this reduces to the standard Blau-Michaeli function R(D, P).

**Modified assumption (A1').** We require d to be convex in its *first* argument (the varying argument p_{X-hat}), rather than the second as in B-M's original A1. This holds for all f-divergences (which are jointly convex in both arguments), KL divergence (D_KL(p||q) is convex in p for fixed q), and total variation distance. It does *not* hold for Wasserstein-2 distance, which is excluded from the main theorem.

**Definition 3.2** (Feasibility Region). Define the minimum achievable distortion at perception level P:

    D_min(P, Q) = inf_{p(x-hat|x): d(p_{X-hat}, Q) <= P} E[Delta(X, X-hat)]

At P = 0 (strict distributional match), this equals the optimal transport cost:

    D_min(0, Q) = inf_{pi in Pi(p_X, Q)} E_pi[Delta(X, X-hat)]

where Pi(p_X, Q) is the set of couplings with marginals p_X and Q. The feasible region is {(D, P) : D >= D_min(P, Q)}.

### 3.2 The Strategic RDP Problem

**Definition 3.3** (Strategic Rate-Distortion-Perception Game). Given:

- Source theta ~ p_theta
- Sender strategy sigma: Theta -> M (message space)
- Receiver strategy alpha: M -> A-hat (action/reconstruction space)
- Sender distortion: Delta_S(alpha(sigma(theta)), theta)
- Receiver distortion: Delta_R(alpha(sigma(theta)), theta)
- Target distribution: Q over A-hat
- Perception tolerance: P

The strategic RDP game is:

1. **Sender** chooses sigma to minimize E[Delta_S] given receiver's strategy alpha
2. **Receiver** chooses alpha to minimize E[Delta_R] subject to d(p_{alpha(sigma(theta))}, Q) <= P

A strategy profile (sigma*, alpha*) is a Nash equilibrium if neither player can unilaterally improve. The strategic rate is R_eq = I(theta; sigma*(theta)) at equilibrium.

**Definition 3.4** (The Generative Residual). For a strategic RDP game with equilibrium (sigma*, alpha*) and an unconstrained strategic game (same utilities, no perception constraint) with equilibrium (sigma_0, alpha_0):

    Delta_gen = D_R(P) - D_R(infinity) = E[Delta_R(alpha*(sigma*(theta)), theta)] - E[Delta_R(alpha_0(sigma_0(theta)), theta)]

The generative residual Delta_gen is the excess receiver distortion attributable to the perception constraint. When Delta_gen > 0, the receiver is forced to generate: to produce outputs that diverge from the source beyond what strategic misalignment alone requires.

---

## 4. Main Results

### 4.1 Theorem A: Generalized RDP Tradeoff (Q != p_X)

**Theorem 4.1.** For source X ~ p_X, reconstruction X-hat, distortion measure Delta satisfying (A2), divergence d satisfying (A1'), and target distribution Q with non-empty feasibility (D >= D_min(P, Q)):

(i) R(D, P, Q) is non-increasing in D and P.

(ii) R(D, P, Q) is convex in D.

(iii) R(D, P, Q) >= R(D) for all finite P, with equality if and only if the unconstrained rate-distortion optimizer satisfies d(p_{X-hat*}, Q) <= P.

*Proof.*

**Step 1: The feasible set is convex.** Define:

    F(D, P, Q) = {p(x-hat|x) : E[Delta(X, X-hat)] <= D, d(p_{X-hat}, Q) <= P}

Three facts combine. First, the marginal p_{X-hat}(x-hat) = sum_x p_X(x) p(x-hat|x) is affine in p(x-hat|x). Second, d(., Q) is convex in its first argument by (A1'), so the sublevel set {p_{X-hat} : d(p_{X-hat}, Q) <= P} is convex. The preimage of this convex set under the affine map p(x-hat|x) -> p_{X-hat} is convex. Third, E[Delta(X, X-hat)] is linear in p(x-hat|x), so {p(x-hat|x) : E[Delta] <= D} is convex.

F(D, P, Q) is the intersection of two convex sets, hence convex. The parameter Q is fixed throughout and does not interact with the convexity argument. This is where the B-M proof carries over unchanged: p_X appeared in their proof only as a fixed reference point, and Q serves identically.

**Step 2: Monotonicity.** For D_1 <= D_2 and P_1 <= P_2, we have F(D_1, P_1, Q) subseteq F(D_2, P_2, Q) (nested sublevel sets). Minimizing I(X; X-hat) over a larger set can only decrease or maintain the minimum:

    R(D_2, P_2, Q) <= R(D_1, P_1, Q)

This gives (i).

**Step 3: Convexity in D.** Take two feasible points (D_1, P) and (D_2, P) with optimal channels p_1* and p_2*. Consider the mixture p_lambda = lambda p_1* + (1-lambda) p_2* for lambda in [0,1].

The mixture is feasible for (lambda D_1 + (1-lambda) D_2, P):
- Distortion: E[Delta] under p_lambda = lambda E[Delta]_{p_1*} + (1-lambda) E[Delta]_{p_2*} <= lambda D_1 + (1-lambda) D_2 (linearity).
- Perception: The marginal p_{X-hat,lambda} = lambda p_{X-hat,1*} + (1-lambda) p_{X-hat,2*}. By convexity of d(., Q): d(p_{X-hat,lambda}, Q) <= lambda d(p_{X-hat,1*}, Q) + (1-lambda) d(p_{X-hat,2*}, Q) <= lambda P + (1-lambda) P = P.

Mutual information I(X; X-hat) is convex in p(x-hat|x) for fixed p_X (Cover and Thomas, 2006, Theorem 2.7.4). Therefore:

    I(X; X-hat) under p_lambda <= lambda I_1* + (1-lambda) I_2*

Taking the minimum over all feasible channels at (lambda D_1 + (1-lambda) D_2, P):

    R(lambda D_1 + (1-lambda) D_2, P, Q) <= lambda R(D_1, P, Q) + (1-lambda) R(D_2, P, Q)

This gives (ii).

**Step 4: Rate elevation.** F(D, P, Q) subseteq F(D) = {p(x-hat|x) : E[Delta] <= D}, since the perception constraint only removes channels from the feasible set. Therefore:

    R(D, P, Q) = min_{F(D,P,Q)} I(X; X-hat) >= min_{F(D)} I(X; X-hat) = R(D)

This gives the weak inequality in (iii). Equality holds if and only if the unconstrained optimizer p*(x-hat|x) achieving R(D) also satisfies d(p_{X-hat*}, Q) <= P.

The elevation is generically strict. Under MSE distortion, the unconstrained rate-distortion optimizer produces X-hat* = E[X | W] for some representation W, whose distribution has Var(X-hat*) = Var(X) - E[Var(X|W)] < Var(X) (law of total variance). The unconstrained reconstruction is always more concentrated than the source. When Q has variance comparable to or greater than Var(X), the reconstruction distribution is too concentrated to satisfy a tight perception constraint, so d(p_{X-hat*}, Q) > 0 and the constraint binds.

**Step 5: Feasibility boundary.** When Q = p_X, the feasible set is non-empty for all D >= 0 because the identity channel X-hat = X achieves d(p_{X-hat}, p_X) = 0. When Q != p_X, the constraint d(p_{X-hat}, Q) <= P forces p_{X-hat} toward Q, which may require distortion. The minimum distortion achievable when p_{X-hat} = Q exactly (P = 0) is:

    D_min(0, Q) = inf_{pi in Pi(p_X, Q)} E_pi[Delta(X, X-hat)]

This is the optimal transport cost under distortion Delta between p_X and Q. For intermediate P > 0, D_min(P, Q) is non-increasing in P (relaxing the perception constraint can only reduce the minimum distortion). The theorem holds on the domain {(D, P) : D >= D_min(P, Q)}.

**Step 6: Operational achievability.** The quantity R(D, P, Q) is an informational lower bound. Operational achievability — the existence of codes that achieve it — follows from Xie, Lei, and Poor (2024), Theorem 1: with unlimited common randomness, any (R, D) pair satisfying R >= I(X; U) for auxiliary U with X-hat = g(U) and p_{X-hat} = Q is achievable. Without unlimited common randomness, the achievable rate is strictly higher; the penalty is I(X-hat; U) - I(X; U). We state the assumption explicitly: Theorem A holds as an operational rate under unlimited common randomness. []


**Comparison with Blau-Michaeli (2019).**

| | B-M (2019) | Theorem A |
|---|---|---|
| Perception target | p_X (source) | Q (arbitrary) |
| Convexity argument | d convex in 2nd arg (A1) | d convex in 1st arg (A1') |
| Feasibility | Trivially non-empty | Non-empty iff D >= D_min(P,Q) |
| Coding theorem | Theis-Wagner (2022) | Xie et al. (2024) |
| 3dB bound | Holds | Fails (Q != p_X breaks posterior sampling) |

The mathematical structure is a parameter substitution with two genuinely new elements: the feasibility boundary characterized via optimal transport, and the operational coding theorem invoked from the output-constrained coding literature.

### 4.2 Theorem B: Gaussian Strategic RDP Equilibrium

**Setup.** Source theta ~ N(0, sigma^2_theta). Quadratic utilities as in Section 2.1. Target distribution Q = N(mu_Q, sigma^2_Q). Perception divergence d = D_KL.

**In the unconstrained Gaussian strategic game** (no perception constraint), the linear equilibrium (Akyol, 2026) has the receiver play alpha_0(m) = E[theta | m] (MMSE estimate). The receiver's action distribution has:

    Var(alpha_0) = sigma^2_theta - D_R^0

where D_R^0 is the MMSE distortion. The action distribution is more concentrated than the source: Var(alpha_0) < sigma^2_theta.

**Theorem 4.2** (Gaussian Strategic RDP). Consider the strategic communication game of Section 2.1 with Gaussian source theta ~ N(0, sigma^2_theta), bias b > 0 with b < 1/4 (non-babbling channel), and perception constraint D_KL(p_alpha || Q) <= P where Q = N(mu_Q, sigma^2_Q).

Model the receiver's action through the affine Gaussian test channel:

    alpha = a * E[theta | m] + c + Z,   Z ~ N(0, sigma^2_Z), independent of theta

where a (scaling), c (shift), and sigma^2_Z (added noise) are the receiver's design parameters.

The receiver's constrained optimization is:

    min_{a, c, sigma^2_Z} E[(alpha - theta)^2]
    subject to: D_KL(p_alpha || Q) <= P

The receiver's action distribution is p_alpha = N(mu_alpha, sigma^2_alpha) where:

    mu_alpha = a * 0 + c = c
    sigma^2_alpha = a^2 * Var(alpha_0) + sigma^2_Z = a^2 (sigma^2_theta - D_R^0) + sigma^2_Z

The KL divergence has the closed form:

    D_KL(p_alpha || Q) = (1/2)[log(sigma^2_Q / sigma^2_alpha) + sigma^2_alpha / sigma^2_Q + (c - mu_Q)^2 / sigma^2_Q - 1]

The receiver's MSE decomposes as:

    D_R = (1-a)^2 sigma^2_theta + a^2 D_R^0 + c^2 + sigma^2_Z

The Lagrangian is:

    L = (1-a)^2 sigma^2_theta + a^2 D_R^0 + c^2 + sigma^2_Z + lambda [D_KL(p_alpha || Q) - P]

The KKT conditions yield three equations in (a, c, sigma^2_Z) parameterized by lambda >= 0.

**Key structural results:**

**(B1) Mean-variance tradeoff.** The parameter c shifts the reconstruction mean toward mu_Q. This consumes distortion budget (c^2 enters D_R) but relaxes the perception constraint (reducing (c - mu_Q)^2 / sigma^2_Q in the KL term). When mu_Q != 0, the receiver allocates distortion budget between mean correction and variance correction. This mean-variance tradeoff is absent from the standard RDP problem (where Q = p_X has the same mean as the source).

**(B2) Variance injection.** When sigma^2_Q > Var(alpha_0) — when the target distribution is more variable than the unconstrained reconstruction — the receiver must add variance. The optimal strategy is to increase sigma^2_Z until sigma^2_alpha approaches sigma^2_Q (to the extent permitted by the distortion budget). The added variance is the generative residual in the variance dimension.

**(B3) Variance suppression.** When sigma^2_Q < Var(alpha_0) — when the target is less variable — the receiver shrinks their action toward the prior mean by setting a < 1. This is a conservatism/regularization effect: conformity to a narrow target forces caution.

**(B4) The generative residual is positive.** In both cases, whenever the perception constraint binds (lambda > 0), the receiver's MSE exceeds the unconstrained MSE:

    D_R(P) > D_R^0

The excess Delta_gen = D_R(P) - D_R^0 is the generative residual. It is the additional distortion the receiver incurs because their output must conform to Q. This quantity is:

- Zero when P = infinity (no conformity constraint)
- Monotonically increasing as P decreases (tighter conformity forces more generation)
- Maximum when P = 0 (exact distributional match)
- Larger when the unconstrained action distribution is further from Q

**(B5) Equilibrium rate response.** The sender's equilibrium strategy responds to the receiver's modified best response. When the receiver adds noise (sigma^2_Z > 0), the receiver's effective response to the sender's partition becomes noisier. This can coarsen or refine the equilibrium partition depending on whether the added noise crosses partition thresholds. In the Gaussian linear equilibrium, the effect is to reduce the effective signal-to-noise ratio, which decreases the equilibrium rate R_eq relative to the unconstrained case. The sender transmits less precisely when the receiver is forced to generate.

**(B6) Tradeoff surface.** The equilibrium (R_eq, D_R, P) lies on the generalized RDP surface of Theorem A. This follows because the equilibrium channel p(alpha | theta) = p(alpha | sigma*(theta)) is a valid conditional distribution in F(D, P, Q). It may not be the R-minimizing channel in F(D, P, Q), but it satisfies the constraints, so R_eq >= R(D_R, P, Q). The equilibrium is a specific point on or above the Theorem A surface. []

**Remark 4.3** (Reparameterization for convexity). With the substitution sigma^2_{alpha} = a^2(sigma^2_theta - D_R^0) + sigma^2_Z, the receiver's problem becomes jointly convex in (a, c, sigma^2_{alpha}) when D_KL is used. This ensures uniqueness of the receiver's best response and standard fixed-point arguments for equilibrium existence.

### 4.3 Theorem C: Organizational Sufficient Conditions

**Theorem 4.4** (Organizational Strategic RDP Tradeoff). An organizational communication channel satisfies the strategic RDP tradeoff under four sufficient conditions:

**Condition O1 (Positive rate).** The channel carries information: b < 1/4 in Crawford-Sobel terms. Operationally: the hierarchical communication system transmits some signal about the underlying state. The channel has not collapsed to babbling equilibrium.

**Condition O2 (Preference divergence).** Sender and receiver have non-identical objectives: b > 0. Operationally: the communicator's interests are not perfectly aligned with the decision-maker's. Some bias exists, however small.

**Condition O3 (Distributional acceptability constraint).** There exists an enforceable distribution Q of acceptable outputs, and the receiver's output must conform to Q within tolerance P. Operationally: institutional norms, professional standards, genre expectations, and compliance requirements constrain what outputs are permissible. An executive summary must look like an executive summary. A budget projection must look like a budget projection.

**Condition O4 (Non-degeneracy).** Q has finite entropy and d(p_{X-hat*}, Q) > 0, where X-hat* is the unconstrained distortion-minimizing reconstruction. Operationally: the distribution of acceptable outputs differs from what the receiver would produce if they simply minimized reconstruction error without conformity constraints. The institutional norm is not neutral: it shapes the output.

*Proof.* Conditions O1 and O2 instantiate the Crawford-Sobel channel: by Proposition 2.1, the communication channel is endogenously lossy with rate R = log_2(N*) > 0 and the sender's partition creates a generative residual (Definition 2.1 of the companion monograph).

Condition O3 provides the perception constraint: the receiver's output must satisfy d(p_{alpha}, Q) <= P for some tolerance P < infinity. This places the receiver's optimization in the strategic RDP framework of Definition 3.3.

Condition O4 ensures the constraint binds: d(p_{X-hat*}, Q) > 0 means the unconstrained reconstruction does not already satisfy the perception constraint for sufficiently small P. By Theorem A, part (iii), the rate-distortion curve is strictly elevated: R(D, P, Q) > R(D) for P small enough that d(p_{X-hat*}, Q) > P.

By Theorem B (B4), the generative residual is positive: Delta_gen > 0. The receiver produces outputs that diverge from the source beyond what strategic misalignment alone requires. The divergence increases with compression ratio (coarser partitions create larger within-bin residuals) and with conformity pressure (smaller P forces more generation).

The four conditions map to the mathematical requirements: O1 ensures R > 0 (the channel transmits), O2 provides the strategic distortion structure (the channel is lossy for game-theoretic reasons), O3 provides the perception constraint (the output must conform), and O4 ensures the constraint is non-trivial (conformity costs something). []

**Corollary 4.5** (Dual Valence as Theorem). Under conditions O1-O4, the generative residual Delta_gen has dual valence:

- Under convergent selection (selection rewards fit with the sender's or organization's existing frame), Delta_gen produces systematic drift toward internally-fit outputs: the Cage.
- Under divergent selection (selection rewards functional novelty or accuracy to external reality), Delta_gen produces reconstruction that occasionally outperforms the source: creative emergence.

The dual valence is a direct consequence of the fact that Delta_gen is a divergence from the source, not a directed error. The direction of the divergence is determined by the selection criterion operating on the receiver's output, not by the compression mechanism. This upgrades Proposition 2.3 of the companion monograph from a structural analogy (motivated by the Blau-Michaeli parallel) to a theorem under conditions O1-O4.

---

## 5. Organizational Interpretation

### 5.1 The Generative Residual as Institutional Physics

The generative residual Delta_gen has a precise institutional interpretation. When a middle manager receives a coarse partition of organizational reality (a summary, a dashboard, a status report) and must produce an output that conforms to institutional norms (an executive briefing, a budget projection, a strategic recommendation), the manager is solving the strategic RDP problem. The manager's preferences diverge from the executive's (O2). The channel compresses (O1). The output must look right (O3). And "looking right" is not the same as "being accurate" (O4).

The excess distortion — the gap between what the manager would produce if they simply minimized error and what they actually produce under conformity pressure — is the generative residual. This is the formal object that the organizational theory literature has described informally as "political behavior," "impression management," "strategic framing," and "compliance theater." The framework does not eliminate these descriptions. It provides the formal mechanism underlying all of them: the receiver is solving a constrained optimization problem, and the constraint forces generation.

### 5.2 Why More Hierarchy Means More Generation

Consider a two-level hierarchy: source theta, middle manager (level 1), executive (level 2). Each level is a strategic RDP channel. By the data processing inequality, the executive's reconstruction cannot be more informative about theta than the middle manager's output. But each level adds its own generative residual. The total divergence from the source at the executive level is:

    Delta_total >= Delta_gen,1 + Delta_gen,2

where Delta_gen,k is the generative residual at level k. The inequality may be strict because the second level's compression operates on the first level's already-distorted output.

This is the formal mechanism underlying the compound confabulation effect described in the companion monograph (Chapter 16): multi-level hierarchies do not merely transmit information with loss. Each level adds its own forced generation, and the generations compound. The executive's picture of reality is not a blurred version of reality. It is a reconstruction of a reconstruction, each layer shaped by its own conformity constraint, each layer adding its own divergence.

### 5.3 The Born-Caged Effect

An empirical observation from the companion monograph's SEC filing study (Chapter 10): new filings arrive pre-compressed, conforming to the distributional characteristics of the existing corpus before any explicit adaptation. The strategic RDP framework provides the mechanism. The acceptability distribution Q is observable (the existing corpus of accepted filings). New entrants, facing the same perception constraint, solve the same optimization problem and produce outputs that conform to Q from inception. The born-caged effect is the equilibrium behavior of rational agents facing the strategic RDP tradeoff.

### 5.4 Endogenous Q

The most significant limitation of Theorem C is that Q is treated as exogenous. In practice, the acceptability distribution is itself a product of the compression-selection dynamics the theorem describes. The reports that "look right" today were shaped by yesterday's conformity pressure, which was shaped by the reports that "looked right" yesterday.

This fixed-point problem — where Q is determined by the equilibrium behavior of agents solving the strategic RDP problem with Q as a constraint — is a natural extension. It is not addressed here. The quasi-static approximation (Q changes slowly relative to the communication dynamics) is sufficient for the present results. The endogenous-Q problem is a second paper.

---

## 6. Discussion

### 6.1 Relation to Bayesian Persuasion

Gentzkow and Kamenica (2011) study Bayesian persuasion: a designer chooses an information structure to influence a receiver's action. The strategic RDP problem includes Bayesian persuasion as a special case (sender commits to a strategy, receiver best-responds) with an added perception constraint. The perception constraint in this context means the designer must produce signals that conform to the receiver's expectations about signal distribution. This connects to the literature on strategic information design under limited attention (the receiver ignores signals that look anomalous).

### 6.2 Implications for AI Alignment

RLHF (reinforcement learning from human feedback) training creates a strategic channel between the model (sender) and human evaluators (receivers). The model's objective diverges from the evaluator's: the model maximizes reward, the evaluator rewards helpfulness. The model's output must conform to the distribution of "helpful-sounding" responses (the perception constraint). The strategic RDP tradeoff predicts sycophancy as an equilibrium phenomenon: the model's outputs diverge from truth toward reward-maximizing conformity, with the divergence increasing as the perception constraint tightens. This connects to the companion monograph's hallucination corollary (Chapter 16).

### 6.3 Limitations

Five limitations are acknowledged.

First, the Gaussian case (Theorem B) assumes linear equilibria and quadratic distortion. Extension to non-Gaussian sources and non-quadratic distortion is future work.

Second, the common randomness assumption is required for operational achievability. In organizational settings, "common randomness" corresponds to shared context (common knowledge, shared priors, organizational culture). The extent to which organizational common knowledge functions as common randomness in the coding-theoretic sense is an empirical question.

Third, Q is exogenous (Section 5.4). The endogenous-Q extension is the natural second paper.

Fourth, the organizational sufficient conditions (O1-O4) are qualitatively checkable but not yet operationalized with measurement precision. Measuring bias b, perception tolerance P, and the divergence d(p_{X-hat*}, Q) in real organizations is a separate empirical research program.

Fifth, the Wasserstein-2 distance is excluded from Theorem A due to failure of (A1'). This limits the result to f-divergences. For organizational applications, KL divergence is the natural choice (measuring the surprise cost of encountering the reconstruction distribution when expecting Q), but the exclusion should be noted.

---

## 7. Conclusion

This paper has defined and characterized the strategic rate-distortion-perception function: the minimum rate required to achieve a given distortion when the encoder and decoder have misaligned objectives and the decoder's output must conform to a target distribution. Three results were proved.

Theorem A generalizes the Blau-Michaeli RDP tradeoff to arbitrary target distributions, with a feasibility boundary characterized via optimal transport. The mathematics is a parameter substitution in the B-M proof with two genuinely new elements: the feasibility analysis and the operational coding theorem from output-constrained source coding.

Theorem B derives the Gaussian strategic RDP equilibrium in closed form, revealing the mean-variance tradeoff created by conformity pressure and establishing that the generative residual is positive whenever the perception constraint binds.

Theorem C translates these results into four checkable sufficient conditions for organizational communication channels, upgrading the relationship between information-theoretic compression and organizational behavior from structural analogy to theorem under stated conditions.

The contribution is at the intersection. The information theory community gains the strategic RDP function, which has not been defined or characterized despite existing work on both strategic communication (Le Treust-Tomala, Akyol) and perception-constrained compression (Blau-Michaeli, Xie et al.). The organizational theory community gains a formal mechanism for phenomena it has described informally for decades: compliance theater, strategic framing, and institutional isomorphism now have a single formal underpinning in the strategic RDP tradeoff.

The endogenous-Q problem — where the acceptability distribution is itself shaped by the equilibrium behavior of agents facing the acceptability constraint — is the natural next step. It is, in the framework's own terms, the dynamics of the Cage: how the distributional constraint evolves as the system compresses and selects over time.

---

## References

Akyol, E. 2026. "Semantic Rate Distortion and Posterior Design." arXiv:2602.03949.

Akyol, E., C. Langbort, and T. Basar. 2015. "Information-Theoretic Approach to Strategic Communication as a Hierarchical Game." arXiv:1510.00764.

Blau, Y., and T. Michaeli. 2018. "The Perception-Distortion Tradeoff." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).

Blau, Y., and T. Michaeli. 2019. "Rethinking Lossy Compression: The Rate-Distortion-Perception Tradeoff." Proceedings of the 36th International Conference on Machine Learning (ICML). arXiv:1901.07821.

Chai, X., Y. Xiao, Z. Shi, and W. Saad. 2023. "Rate-Distortion-Perception Theory for Semantic Communication."

Cover, T. M., and J. A. Thomas. 2006. Elements of Information Theory. 2nd ed. Wiley-Interscience.

Crawford, V. P., and J. Sobel. 1982. "Strategic Information Transmission." Econometrica 50(6): 1431-1451.

Cuff, P. 2013. "Distributed Channel Synthesis." IEEE Trans. Inform. Theory 59(11): 7071-7096.

Galbraith, J. R. 1974. "Organization Design: An Information Processing View." Interfaces 4(3): 28-36.

Gentzkow, M., and E. Kamenica. 2011. "Bayesian Persuasion." American Economic Review 101(6): 2590-2615.

Le Treust, M., and T. Tomala. 2018. "Information-Theoretic Limits of Strategic Communication." arXiv:1807.05147.

Le Treust, M., and T. Tomala. 2020. "Point-to-Point Strategic Communication." arXiv:2010.12480.

Liu, S., et al. 2022. "Cross-Domain Compression as Entropy-Constrained Optimal Transport." International Conference on Learning Representations (ICLR).

Matsumoto, T. 2018. "Introducing the Perception-Distortion Tradeoff into the Rate-Distortion Theory of General Information Sources." arXiv:1808.07986.

McEntire, J. 2026. Structural Compression Theory: A Unified Information-Theoretic Account of Organizational Dysfunction, Creativity, and Substrate-Independent Selection Dynamics. Cage and Mirror Publishing.

Qian, Y., et al. 2024. "Gaussian Optimality for the Rate-Distortion-Perception Function."

Saritas, S., et al. 2023. "Multi-Dimensional Crawford-Sobel with Rate-Distortion Perspective."

Shannon, C. E. 1948. "A Mathematical Theory of Communication." Bell System Technical Journal 27(3): 379-423, 623-656.

Wagner, A. B. 2022. "The Rate-Distortion-Perception Tradeoff: The Role of Common Randomness."

Xiao, D., H. Zhang, S. Li, Z. Shi, and T. Basar. 2022. "Rate-Distortion Theory for Strategic Semantic Communication." arXiv:2202.03711.

Xie, Y., B. Lei, and H. V. Poor. 2024. "Output-Constrained Lossy Source Coding with Application to Rate-Distortion-Perception Theory."
