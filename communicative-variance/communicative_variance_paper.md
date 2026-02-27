# The Source of Creation Is Dysfunction: The Generative Lossy Channel and Five Sufficient Conditions for Net-Beneficial Noise

**Jeremy McEntire**

*Correspondence: jmc@cageandmirror.com*

Working Paper — SSRN

---

## Abstract

Organizational dysfunction and creative emergence are treated as separate phenomena requiring separate theories. We prove they are the same mechanism. Any communication channel with preference divergence is endogenously lossy (Crawford-Sobel 1982). Any lossy reconstruction maintaining coherence must diverge from the source (Blau-Michaeli 2018, 2019). This divergence — the generative residual — produces dysfunction under convergent selection (where organizational fitness outcompetes external accuracy) and novelty under divergent selection (where functional reconstruction outcompetes sender-aligned reproduction). The mechanism is identical; the valence is determined by the selection environment.

We formalize this claim as Theorem 1: five sufficient conditions under which noise at organizational level $N$ produces net benefit at level $N+1$. The conditions are: (C1) the higher level operates suboptimally without noise, (C2) the integration function is sufficiently nonlinear, (C3) the noise distribution can access the improvement region, (C4) the weighted gain at the higher level exceeds the cost at the lower level, and (C5) the lower level degrades gracefully. When all five hold, system performance follows an inverted-U with an optimal noise level $\sigma^* > 0$. Excessive noise always destroys the benefit.

The conditions are structural, not mechanism-specific. Computational validation across six mechanisms — threshold stochastic resonance, sigmoid detection, simulated annealing, ensemble diversity, multi-armed bandit exploration, and Crawford-Sobel strategic communication — produces zero counterexamples in 500 Monte Carlo parameter configurations when all conditions are met (75/75 sufficiency rate). Two null models (linear detection, polynomial detection with insufficient steepness) confirm that nonlinearity is essential. Condition violations correctly predict benefit disappearance.

Six instance papers spanning organizational hierarchy (dysmemic pressure), strategic language (post-IPO variance compression), legal architecture (mandatory arbitration as claim suppression), computational geometry (stochastic resonance in activation space decomposition), musical interaction (jazz improvisation as creative emergence through lossy real-time communication), and cognitive architecture (hierarchical context distillation) demonstrate the framework across substrates that share no surface features but exhibit identical formal structure: compression forces reconstruction; reconstruction diverges from source; selection determines valence.

A central corollary: no system can eliminate its capacity for dysfunction without simultaneously eliminating its capacity for creativity. They share a root cause — the generative residual of lossy reconstruction. The design problem is not elimination but channeling: structuring the selection environment to reward divergent reconstruction over convergent drift.

The framework closes the descriptive-to-prescriptive gap in the noise-benefit literature. The five conditions are checkable properties of specific systems, answering the operational question: *will noise help this system, at this operating point, with this integration function?*

**Keywords:** stochastic resonance, Crawford-Sobel, lossy compression, organizational dysfunction, creativity, level-crossing, information theory, generative noise

---

# Section 1: The Symmetry

## 1.1 The Uncomfortable Observation

Consider two facts about hierarchical communication that are individually well-established and jointly ignored.

**Fact 1.** Organizations routinely converge on collective delusions. Nokia's middle managers reported confidence in Symbian while knowing it was dying. NASA engineers signed off on a launch they had data to reject. Wells Fargo employees opened millions of fraudulent accounts to satisfy metrics that had decoupled from reality. In each case, the information existed at the lower level. It failed to survive the channel. This is not controversial — it is the subject of a half-century of organizational failure literature, from Janis (1972) on groupthink to Edmondson (1999) on psychological safety to the empirical work on strategic communication games beginning with Crawford and Sobel (1982).

**Fact 2.** Organizations routinely produce ideas that no individual member contained. A garbled customer requirement becomes a feature nobody requested but everybody wants. A misheard directive spawns a workaround that outperforms the original plan. A team of specialists, none of whom individually understands the whole problem, collectively solves it — not despite their partial views, but because their partial views force reconstruction at every interface. This is equally uncontroversial — it is the subject of the creativity and innovation literature, from Weick (1979) on sensemaking to Hargadon and Sutton (1997) on knowledge brokering to the stochastic resonance literature in physics (Gammaitoni et al. 1998) and neuroscience (McDonnell and Abbott 2009).

Here is the observation that is comfortable: both phenomena exist.

Here is the observation that is not: **they are the same phenomenon.**

The same mechanism that produces organizational stupidity produces jazz.

The Nokia middle manager and the jazz musician are running identical processes. A complex state enters a lossy channel. The channel strips information. The receiver reconstructs what was lost using their own priors, context, and inferential capacity. What differs is not the channel, not the compression, not the reconstruction — only the selection criterion operating on the output. When the selection environment rewards fit-with-sender-preferences, that reconstruction produces the Cage — convergent, self-confirming, increasingly detached from external reality. When the selection environment rewards functional novelty, the same reconstruction produces emergence — divergent, reality-testing, occasionally brilliant.

This paper formalizes that claim. We prove sufficient conditions under which the net effect of noise is beneficial — not as a general platitude ("some noise is good"), but as a theorem with five testable conditions, an inverted-U prediction, and zero counterexamples across 500 Monte Carlo parameter configurations and six distinct mechanisms spanning stochastic resonance, simulated annealing, ensemble diversity, and exploration-exploitation tradeoffs.

## 1.2 What Is Actually Being Claimed

The claim is not that noise is good. The claim is not that dysfunction is secretly creative. The claim is narrower, stranger, and more useful than either of those:

**Any system that compresses a rich source into a lower-dimensional representation and requires the output to be coherent must produce outputs that diverge from the source. This divergence is not error. It is a mathematically necessary reconstruction, and the same reconstruction that can generate dysfunction can generate novelty. The conditions that determine which outcome obtains are formally specifiable.**

Three components of this claim require separate defense.

*First: the channel is endogenously lossy.* Crawford and Sobel (1982) proved that when a sender and receiver have even slightly misaligned preferences, the sender's optimal strategy is to partition continuous reality into discrete bins. The receiver gets the bin, not the state. This is not a physical bandwidth limitation — it is a game-theoretic equilibrium. The lossiness is *chosen*, emerging from the incentive structure of the communication itself. In organizations, this manifests as reports that round, dashboards that aggregate, and executives who receive partitioned summaries of a continuous reality. The information exists at the lower level. The channel does not transmit it. And critically, as Crawford-Sobel bias increases, the channel degrades monotonically — from fine-grained partitions at low bias, through progressively coarser bins, to the babbling equilibrium at which the message carries zero information about the state. The transition is sharp, not gradual.

*Second: lossy reconstruction is necessarily generative.* Blau and Michaeli (2018, 2019) proved a result in the rate-distortion-perception framework that carries profound implications beyond its machine learning origins. When a system compresses a source and requires the reconstruction to be *perceptually valid* — meaning the output must be drawn from a distribution that resembles the source distribution, not merely minimize point-wise error — the reconstruction must diverge from the specific input. This is not an engineering limitation. It is a mathematical theorem: you cannot simultaneously achieve high compression, low distortion, and high perceptual quality. Any two, but not all three. In organizational terms: when a receiver gets a coarse partition and must produce a coherent action (not just a point estimate), they are forced to generate. The generation draws on their own priors, their own context, their own model of the world. The output is a novel synthesis that neither the sender nor the receiver originally contained.

*Third: the conditions are specifiable.* This is where the existing literature stops and the present contribution begins. Atlan's "complexity from noise" principle (1979) established that noise at level $N$ can become information at level $N+1$. The Kosko forbidden interval theorem established necessary and sufficient conditions for stochastic resonance in threshold detectors. But no existing result answers the operational question: *Given a specific system, when does the benefit at the higher level outweigh the cost at the lower level?* We prove sufficient conditions (Theorem 1, Section 2.5) and validate them computationally across six distinct mechanisms (Section 3). The conditions are: (C1) the higher level is operating suboptimally without noise, (C2) the integration function is sufficiently nonlinear, (C3) the noise distribution can access the improvement region, (C4) the weighted gain at the higher level exceeds the cost at the lower level for some noise amplitude, and (C5) the lower level degrades gracefully rather than catastrophically. When all five hold, the net system performance follows an inverted-U: there exists an optimal noise level $\sigma^* > 0$ that maximizes total performance, and excessive noise always destroys the benefit.

## 1.3 The Dual Valence Problem

The symmetry between dysfunction and creativity is not a metaphor. It is a formal consequence of the Blau-Michaeli result operating within the Crawford-Sobel channel.

Consider the generative residual — the information that does not survive the lossy channel. When the receiver reconstructs this information, the reconstruction is shaped by whatever selection criterion governs the receiver's environment. We define two regimes:

**Convergent selection** rewards reconstructions that align with the sender's preferences, the organization's existing narrative, or the social consensus. Under convergent selection, the generative residual produces *dysmemic drift*: signals that are internally fit (easy to transmit, comfortable to receive, consistent with the prevailing story) outcompete signals that are externally accurate. This is the Cage. The mechanism is well-documented: McEntire's theory of Dysmemic Pressure formalizes how incentive divergence, transmission ease, and verification cost compound to select for cultural variants that maximize internal fitness while minimizing external accuracy. The organization converges on a shared hallucination that is perfectly coherent, perfectly transmissible, and perfectly wrong.

**Divergent selection** rewards reconstructions that solve novel problems, match external reality, or produce functional novelty. Under divergent selection, the same generative residual produces *creative emergence*: the receiver's reconstruction introduces variation that, when tested against external criteria, occasionally outperforms anything either party originally held. This is jazz. The musician fills the gap left by imperfect transmission with a harmonic variation drawn from their own training. The product team fills the gap left by a garbled requirement with a feature drawn from their own domain knowledge. The variation is not random — it is structured by the receiver's priors — but it is novel relative to the sender's intent.

The mechanism is identical. The channel is the same. The compression is the same. The reconstruction is the same. What differs is the selection pressure operating on the output. This is Proposition 1 of the formal framework (Section 2.2), and it carries an immediate corollary that practitioners will find uncomfortable: **no system can eliminate its capacity for dysfunction without simultaneously eliminating its capacity for creativity.** They share a root cause. The design problem is not elimination but channeling — structuring the selection environment to reward divergent reconstruction over convergent drift.

## 1.4 Why the Inverted-U Is Universal

A recurring empirical pattern across every domain where noise-benefit has been studied is the inverted-U: performance improves with moderate noise and degrades with excessive noise. This pattern appears in stochastic resonance (Gammaitoni et al. 1998), simulated annealing (Kirkpatrick et al. 1983), ensemble diversity in machine learning (Krogh and Vedelsby 1994), exploration-exploitation tradeoffs in decision theory (Sutton and Barto 2018), and desirable difficulties in learning (Bjork and Bjork 2011). The ubiquity of this shape is usually treated as a coincidence, or noted without explanation, or — when the mechanisms appear different — taken as evidence that each domain requires its own theory.

It is not a coincidence, and it does not require separate theories. The inverted-U is a necessary consequence of the five conditions in Theorem 1, which are satisfied by all of these domains. What varies across domains is not the conditions but the *mechanism* through which moderate noise produces level-N+1 benefit: threshold detection (classical SR), convex rectification (Jensen gap), or the product of opposing monotone functions (Bjork's storage-retrieval interaction). We prove all three paths (Theorems 1a, 1b, 1c) under a shared set of sufficient conditions. The logic of each is straightforward, and the common structure explains the common shape:

1. Condition C1 (suboptimality) guarantees there is room for improvement. If the system is already at its best, no noise can help.
2. Conditions C2 and C3 (nonlinearity and accessibility) guarantee that moderate noise pushes the system into the improvement region. The nonlinearity converts noise into signal — $E[f(x + \xi)] > f(x)$ when $f$ is locally convex in the operating region. This is the Jensen gap: the expected response to signal-plus-noise exceeds the response to signal alone.
3. Condition C4 (weighting) guarantees that the higher-level gain exceeds the lower-level cost for some finite noise level.
4. Condition C5 (robustness) guarantees that the lower level does not catastrophically fail before the higher level can benefit.
5. The peak exists because the Jensen gap is bounded. As noise increases past $\sigma^*$, the higher-level response saturates (the signal is no longer subthreshold — it is swamped) while the lower-level cost continues to grow. The crossover is guaranteed.

The five conditions are structural, not mechanism-specific. They do not require a threshold detector (classic SR), a loss landscape (simulated annealing), an ensemble of classifiers (diversity), or a multi-armed bandit (exploration). They require only: a two-level system where the higher level is underperforming, a nonlinear integration function, noise that can reach the improvement region, a gain-cost ratio that favors the higher level at some noise amplitude, and a lower level that degrades gracefully. Every known inverted-U phenomenon satisfies these conditions. The conditions explain why the inverted-U appears everywhere — it is not a coincidence of similar shapes across different domains, but a single theorem instantiated across different substrates.

We validated this claim computationally. Across 500 random parameter configurations and six mechanisms — threshold detection, sigmoid detection, landscape escape via simulated annealing, ensemble diversity via error decorrelation, and exploration-exploitation via bandit arms — the theorem produced zero counterexamples when all five conditions were met, and correctly predicted that violations of C1, C2, or C5 would eliminate the benefit. The conditions are sufficient but not necessary: approximately 9% of configurations with at least one condition violated still showed benefit, placing the conditions as a guaranteed-benefit region rather than a hard boundary.

## 1.5 From Description to Prescription

The existing literature on noise and creativity is almost entirely descriptive. It catalogs instances where noise helped (SR in crayfish mechanoreceptors, dithering in audio quantization, desirable difficulties in classroom learning) and instances where noise hurt (communication breakdown in organizations, hallucination in language models, catastrophic forgetting in neural networks). The pattern is noted — sometimes noise helps, sometimes it hurts — and the advice that follows is vague: "tolerate some ambiguity," "embrace creative tension," "balance exploration and exploitation."

This paper closes the descriptive-to-prescriptive gap. The five conditions of Theorem 1 are checkable properties of a specific system. They answer the operational question: *Will noise help this system, at this operating point, with this integration function?* If all five conditions are met, the answer is yes, and the inverted-U predicts both the existence and the shape of the benefit curve. If any condition is violated — the system is already optimal (C1), the integration is linear (C2), the noise cannot reach the improvement region (C3), the cost-benefit ratio favors the lower level (C4), or the lower level is brittle (C5) — the answer is no, or at least not guaranteed.

This is the difference between "noise sometimes helps" and "noise helps here, at this amplitude, for this reason, and will stop helping past this point." It is the difference between a phenomenon and a tool.

## 1.6 Paper Structure

The argument proceeds as follows.

**Section 2** establishes the formal foundation. Four results compose into a chain: Crawford-Sobel (1982) proves that channels with preference divergence are endogenously lossy. Blau-Michaeli (2018, 2019) proves that lossy reconstruction maintaining coherence must diverge from the source — generatively. Atlan (1979) establishes that noise at level $N$ becomes information at level $N+1$ when the higher level has integrative capacity. The Kosko forbidden interval theorem provides necessary and sufficient conditions for the noise benefit in threshold systems. We then prove Theorem 1 in two forms: Theorem 1a for threshold systems (where the Kosko result applies directly) and Theorem 1b for general nonlinear systems (where the Jensen gap provides the mechanism), with a shared set of conditions and a dedicated lemma establishing why the cost-benefit condition must be integral rather than marginal. Corollaries address brittleness, alignment, the over-noise catastrophe, dual valence, and trivial cost conditions. The Tolerance Location Principle synthesizes the chain into a design prescription.

**Section 3** presents computational validation. We test Theorem 1 against six mechanisms spanning the SR and non-SR literature: threshold detection, sigmoid detection, simulated annealing (landscape escape), ensemble diversity (error decorrelation), multi-armed bandit (exploration-exploitation), and Crawford-Sobel strategic communication with endogenous partition equilibria. We report the Monte Carlo sensitivity analysis (500 configurations, zero counterexamples), the condition-violation diagnostics, and the adversarial parameter search.

**Section 4** maps the framework onto six instance papers spanning organizational dysfunction, linguistic variance compression, legal claim suppression, activation space geometry, jazz improvisation, and cognitive architecture. Two instances (Variance Compression and Activation Space) receive deep treatment with quantitative condition verification; two (Dysmemic Pressure and Jazz Improvisation) receive standard treatment with documented case evidence; two (Structural Immunity and Hierarchical Distillation) serve as brief boundary cases. The mapping is one-directional: the instances validate the theory; the theory does not depend on any single instance.

**Section 5** addresses limitations, open questions, and the falsification conditions for the framework.

The core claim is simple enough to state in a sentence: **dysfunction and creativity are the same reconstruction, selected differently.** The rest of the paper makes that sentence precise.

---

# Section 2: Formal Foundation — The Generative Lossy Channel

## 2.1 The Endogenous Lossy Channel (Crawford-Sobel 1982)

### Setup

A Sender (S) observes a continuous state of the world $\theta \in [0,1]$, drawn from a uniform prior. S transmits a costless message $m$ to a Receiver (R), who takes an action $a$ affecting both parties' utility. The preference divergence (bias) parameter $b > 0$ quantifies the misalignment between S and R's ideal actions.

- S's utility: $U_S(\theta, a) = -(a - \theta - b)^2$
- R's utility: $U_R(\theta, a) = -(a - \theta)^2$

### The Partition Theorem

In any Nash equilibrium, S's message partitions the continuous state space $[0,1]$ into at most $N^*$ discrete intervals ("bins"). The maximum number of distinguishable partitions is:

$$N^* = \left\lceil -\frac{1}{2} + \frac{1}{2}\sqrt{1 + \frac{2}{b}} \right\rceil$$

**Implications:**
- At $b = 0$ (perfect alignment): $N^* \to \infty$. Lossless transmission is possible.
- At $b \geq \frac{1}{4}$: $N^* = 1$. **Babbling equilibrium.** S's message is statistically independent of $\theta$. The channel carries zero information. R rationally ignores all messages.
- For all $0 < b < \frac{1}{4}$: the channel is endogenously lossy. The continuous state is compressed into coarse discrete partitions.

### The Residual: Traditional vs. Generative Interpretation

**Traditional interpretation:** The information lost to partitioning — the within-bin variance — is deadweight loss. It represents coordination failure.

**Generative reinterpretation (this paper):** The residual is the locus of novelty. When R receives a message indicating $\theta \in [a_i, a_{i+1}]$ but cannot determine the precise value, R *must reconstruct* using their own prior beliefs, context, and inferential capacity. This reconstruction is not error — it is a novel synthesis that neither party originally contained.

**Definition 1 (The Generative Residual).** For a Crawford-Sobel channel with bias $b > 0$ and equilibrium partition $\{a_0, a_1, \ldots, a_{N^*}\}$, the *generative residual* at each interface is:

$$G_i = H(\theta \mid \theta \in [a_i, a_{i+1}]) = \log(a_{i+1} - a_i)$$

where $H$ is the differential entropy of the true state conditional on the received partition. This residual is the raw material available for receiver reconstruction.

---

## 2.2 Compression Forces Generative Reconstruction (Blau-Michaeli 2018, 2019)

### The Perception-Distortion Tradeoff

Blau & Michaeli (2018) prove a fundamental constraint on lossy compression:

**Theorem (Perception-Distortion Tradeoff).** For any estimator $\hat{X}$ of a source $X$, minimizing expected distortion $E[d(X, \hat{X})]$ and minimizing the divergence between the distribution of $\hat{X}$ and the distribution of $X$ (perceptual quality) are *at odds*. Formally:

$$\text{As } d(P_{\hat{X}}, P_X) \to 0 \text{ (perfect perceptual quality)}, \quad E[\Delta(X, \hat{X})] \text{ must increase.}$$

This holds for *all* distortion measures, not just specific metrics.

### The Rate-Distortion-Perception Tradeoff (2019)

Extending Shannon's classical rate-distortion theory, Blau & Michaeli (2019) show:

**Theorem (Rate-Distortion-Perception).** Restricting perceptual quality to be high generally leads to an elevation of the rate-distortion curve, necessitating a sacrifice in either rate or distortion.

There exists a fundamental three-way tradeoff: you cannot simultaneously achieve:
1. Low bit rate (high compression)
2. Low distortion (fidelity to source)
3. High perceptual quality (output indistinguishable from source distribution)

### The Organizational Mapping: Scope and Limits

The Blau-Michaeli result is a theorem about signal compression under a distributional constraint on the reconstruction. Applying it to organizational communication requires identifying the corresponding constraint. We make this mapping explicit and bound its status.

**The formal correspondence.** In the Blau-Michaeli framework, "perceptual quality" requires that $d(P_{\hat{X}}, P_X) \leq \delta$: the output distribution must resemble the source distribution. In organizational communication, the receiver faces an analogous constraint: the reconstruction must be drawn from the distribution of *acceptable organizational communications* — it must conform to institutional norms, professional standards, and genre expectations. A middle manager's report must "look like" a legitimate report. This acceptability constraint functions as a distributional requirement: outputs outside the acceptable distribution are rejected, revised, or sanctioned.

**Where the mapping tightens.** The structural consequence — that compression combined with a distributional output constraint forces divergence from the specific source — transfers directly. The organizational receiver cannot simultaneously (1) accept a heavily compressed input (coarse partition), (2) produce an output matching the true state (low distortion), and (3) produce an output satisfying institutional acceptability (distributional constraint). The three-way tradeoff's *structure* applies: any two, but not all three.

**Where the mapping loosens.** The Blau-Michaeli theorem specifies the constraint as distributional match between output and source ($P_{\hat{X}} \approx P_X$). Organizational acceptability is a constraint on $P_{\hat{X}}$ alone — the output must be drawn from an acceptable distribution, but that distribution need not match the source distribution. This is a weaker constraint, which means the organizational case is *not* a direct application of the theorem but a *motivated structural analogy*: the tradeoff's shape (compression + output constraint → source divergence) is preserved; the specific divergence bound is not.

We therefore state Proposition 1 below as a consequence of this structural analogy, not as a deductive corollary of the Blau-Michaeli theorem. The formal chain from Crawford-Sobel to Theorem 1 does not depend on Blau-Michaeli holding in its exact form — Theorem 1's proof relies on the Kosko forbidden interval theorem (1a), Jensen's inequality (1b), and first-order calculus (1c), none of which require the perception-distortion tradeoff. Blau-Michaeli provides the *interpretive* bridge explaining *why* the reconstruction diverges from the source, not the *formal* mechanism by which noise produces benefit.

**Proposition 1 (Dual Valence of Compression).** The generative residual $G_i$ from a Crawford-Sobel channel under the Blau-Michaeli constraint is:
- **Dysfunctional** when the selection criterion operating on receiver reconstructions rewards fit-with-sender-preferences (convergent selection → the Cage)
- **Creative** when the selection criterion rewards functional novelty or accuracy-to-external-reality (divergent selection → emergence)

The mechanism is identical. The valence is determined by the selection environment, not the channel.

---

## 2.3 Level-Crossing: Noise at N Becomes Information at N+1 (Atlan 1979)

### The Principle

Henri Atlan's "complexity from noise" principle (*Entre le cristal et la fumée*, 1979), building on von Foerster's "order from noise" (1960):

**Principle (Atlan Level-Crossing).** In a hierarchically organized system, perturbation (noise) at organizational level $N$ can become functional information at level $N+1$, provided the higher level has sufficient integrative capacity to exploit the variation.

**Instances:**
- A genetic mutation is noise within an organism's developmental program but information within the population's adaptive capacity
- A misheard chord is noise within a musical performance but information within the evolution of harmonic language
- A mistranslated text is noise within the author's intended meaning but information within the intellectual tradition that grows from the mistranslation
- A Crawford-Sobel residual is noise within the sender's intended communication but information within the organization's capacity for novel synthesis

### Toward Formalization

Atlan's principle remains largely phenomenological. We propose formal conditions under which level-crossing is guaranteed:

**Definition 2 (Level-Crossing System).** A system exhibits *generative level-crossing* when:
1. There exist at least two organizational levels $L_N$ and $L_{N+1}$ with distinct state spaces
2. Level $L_N$ produces variation $V_N$ (noise relative to $L_N$'s functional requirements)
3. Level $L_{N+1}$ has an integrative mapping $\phi: V_N \to S_{N+1}$ that transforms $L_N$-noise into $L_{N+1}$-signal
4. The mapping $\phi$ satisfies: $I(V_N; S_{N+1}) > 0$ — the variation carries nonzero mutual information with the higher-level signal

**The open question:** Under what conditions on the system's nonlinearity profile does $\phi$ exist and produce net benefit? This connects to the Kosko forbidden interval theorem (Section 2.4) and is resolved by Theorem 1 (Section 2.5).

---

## 2.4 When Does Noise Help? The Forbidden Interval Theorem (Kosko)

### The Stochastic Resonance Phenomenon

In nonlinear systems with a threshold, the addition of an optimal level of noise *enhances* signal detection. The signal-to-noise ratio (SNR) follows an inverted-U curve, peaking at nonzero noise.

### The Forbidden Interval Theorem

Kosko and colleagues prove necessary and sufficient conditions for stochastic resonance in threshold detectors:

**Theorem (Forbidden Interval — Kosko et al.).** A threshold signal detector exhibits a noise benefit (stochastic resonance) if and only if the noise mean $\mu$ does *not* lie in the "forbidden interval" $(\theta - s_1, \theta - s_0)$, where $\theta$ is the detection threshold and $s_0, s_1$ are the subthreshold signal levels.

**Key properties:**
- The condition is both *necessary and sufficient* — not just sufficient
- The result holds for *all* noise probability density functions with finite variance
- It extends to the entire uncountably infinite class of $\alpha$-stable distributions (heavy-tailed noise)
- It has been generalized to quantum channels (Kosko 2008)

### Connecting the Forbidden Interval to Level-Crossing

The forbidden interval theorem provides the formal condition that Atlan's level-crossing principle lacks:

**Proposition 2 (Formal Level-Crossing Condition).** Atlan-type level crossing — where noise at level $N$ becomes information at level $N+1$ — occurs when:

1. The higher-level integration function $\phi$ has a **nonlinear threshold** (the system is not a linear pass-through)
2. The variation $V_N$ has a mean that falls **outside the forbidden interval** of $\phi$'s threshold structure
3. The variation magnitude is moderate (too little fails to reach threshold; too much swamps the signal)

This yields the inverted-U prediction: there exists an optimal noise level $\sigma^*$ that maximizes the mutual information $I(V_N; S_{N+1})$, and this optimum is nonzero.

---

## 2.5 Theorem 1: Sufficient Conditions for Net Generative Level-Crossing

The preceding sections establish *when* noise at level N produces benefit at level N+1 (Kosko) and *why* the reconstruction is generative (Blau-Michaeli). What remains is the net-benefit question: when does the gain at N+1 outweigh the cost at N?

The answer depends on the mechanism through which noise produces level-N+1 benefit. We distinguish two cases — threshold systems where the Kosko forbidden interval theorem applies directly, and general nonlinear systems where the Jensen gap provides the mechanism — and prove both under a shared set of conditions.

### Definitions

Let a two-level system consist of:

- **Level N**: Performance function $P_N(\sigma)$ where $\sigma$ is the noise amplitude.
- **Level N+1**: Performance function $P_{N+1}(\sigma)$ that depends on the variation produced at level N.
- **System performance**: $P_{sys}(\sigma) = f(P_N(\sigma), P_{N+1}(\sigma); \alpha, \beta)$ where $\alpha, \beta > 0$ are the relative weights of level-N fidelity and level-N+1 generativity, and $f$ is the coupling function.

### The Question

Does there exist $\sigma > 0$ such that $P_{sys}(\sigma) > P_{sys}(0)$?

### Shared Conditions

The following conditions are common to both cases:

**Condition 1 (Suboptimality).** The signal available to $L_{N+1}$ from $L_N$ in the absence of noise is suboptimal: $P_{N+1}(0) < P_{N+1}^{max}$. The higher level is not already operating at its maximum.

**Condition 4 (Sufficient Level Weighting).** There exists some noise level $\sigma$ such that the integrated gain at N+1 exceeds the integrated cost at N:

$$\beta \cdot [P_{N+1}(\sigma) - P_{N+1}(0)] > \alpha \cdot [P_N(0) - P_N(\sigma)]$$

**Remark (The role of C4).** C4 is the existence-of-benefit condition. It is the hardest of the five conditions to verify a priori, because checking it requires either running the noise sweep or estimating the integral from the system's response profile. The theorem's practical value lies in the other four conditions: C1, C2, C3, and C5 are *easy to check* from the system's structure without running any experiment. They serve as a screening tool. If any of C1-C3 or C5 fails, the system cannot benefit from noise regardless of the weighting, and C4 need not be checked. If all four pass, then the question reduces to C4: does the gain-cost ratio favor the higher level at some finite σ? The diagnostic workflow is: screen with C1-C3, C5 (cheap) → if all pass, invest in checking C4 (expensive) → if C4 holds, benefit is guaranteed.

**Condition 5 (Robustness).** Level N has sufficient redundancy that moderate noise does not cause catastrophic failure. $P_N(\sigma)$ is locally Lipschitz at $\sigma = 0$ with Lipschitz constant $L_N < \infty$.

### Theorem 1a: Threshold Systems (Kosko Path)

**Additional conditions for threshold systems:**

**Condition 2a (Threshold Nonlinearity).** The integrative mapping $\phi$ at level $N+1$ contains a threshold $\theta$ such that the response changes qualitatively when the input crosses $\theta$. Formally, $\phi$ has a point $\theta$ where the left and right derivatives differ by a finite amount, or where $\phi$ transitions from near-zero to near-maximal response over an interval small relative to the noise variance.

**Condition 3a (Forbidden Interval Accessibility).** The noise mean $\mu$ does not lie in the forbidden interval $(\theta - s_1, \theta - s_0)$ of the threshold structure, where $s_0, s_1$ are the signal levels bracketing the threshold. This is the Kosko condition.

**Theorem 1a.** A two-level system satisfying C1, C2a, C3a, C4, and C5 exhibits net generative level-crossing with an inverted-U performance curve peaking at some $\sigma^* > 0$.

**Proof.** By C1, $P_{N+1}(0) < P_{N+1}^{max}$: the signal $s$ is subthreshold ($s < \theta$), leaving room for improvement.

By C2a and C3a, the Kosko forbidden interval theorem applies directly: the detection probability $P_{N+1}(\sigma) = P(s + \xi > \theta)$ increases with $\sigma$ for small $\sigma$, reaching a maximum at some $\sigma_{SR}^*$ before declining as noise overwhelms signal. This is the classical stochastic resonance result. Define $\Delta_{N+1}(\sigma) = P_{N+1}(\sigma) - P_{N+1}(0) > 0$ for $\sigma \in (0, \bar{\sigma}_{SR})$.

Note that $\Delta_{N+1}$ does not increase continuously from zero. For threshold detectors, $P_{N+1}(0)$ may be exactly zero (signal entirely below threshold), and the benefit onset occurs at finite $\sigma$ when the noise distribution's tail begins to reach $\theta$. This is why C4 must be an integral condition (see Lemma 1 below).

By C5, level-N degradation is bounded: $\Delta_N(\sigma) = P_N(0) - P_N(\sigma) \leq L_N \cdot \sigma$.

By C4, there exists $\sigma$ where $\beta \cdot \Delta_{N+1}(\sigma) > \alpha \cdot \Delta_N(\sigma)$, yielding $P_{sys}(\sigma) > P_{sys}(0)$.

The peak $\sigma^*$ exists because $\Delta_{N+1}$ is bounded above (detection probability cannot exceed 1) while $\Delta_N$ grows at least linearly. The crossover is guaranteed: eventually cost exceeds benefit. $\square$

### Theorem 1b: General Nonlinear Systems (Jensen Gap Path)

**Additional conditions for general nonlinear systems:**

**Condition 2b (Sufficient Curvature).** The integrative mapping $\phi$ at level $N+1$ has a region of local convexity in the neighborhood of the operating point $s$. Specifically, there exists an interval $I$ containing $s$ where the second derivative $\phi''(x) > 0$, and the curvature is large enough relative to the noise variance that the Jensen gap is non-negligible:

$$J(\sigma) = E[\phi(s + \xi)] - \phi(s) > \epsilon$$

for some $\sigma > 0$ and some $\epsilon > 0$, where $\xi \sim \mathcal{N}(0, \sigma^2)$ or another symmetric distribution.

**Condition 3b (Distributional Support).** The noise distribution has sufficient support in the convex region of $\phi$. Formally, $P(\xi \in I - s) > 0$ for the interval $I$ where $\phi$ is convex.

**Theorem 1b.** A two-level system satisfying C1, C2b, C3b, C4, and C5 exhibits net generative level-crossing with an inverted-U performance curve peaking at some $\sigma^* > 0$.

**Proof.** By C1, $P_{N+1}(0) = \phi(s) < P_{N+1}^{max}$: the system operates below optimum.

By C2b, $\phi$ is locally convex at $s$. By Jensen's inequality, for any noise $\xi$ with $E[\xi] = 0$ and $P(\xi \in I - s) > 0$ (guaranteed by C3b):

$$E[\phi(s + \xi)] \geq \phi(s + E[\xi]) = \phi(s)$$

with strict inequality when $\phi$ has strict convexity and $\xi$ is non-degenerate. The Jensen gap $J(\sigma) = E[\phi(s + \xi)] - \phi(s) > 0$ for $\sigma > 0$ in a neighborhood of zero.

For small $\sigma$, the Jensen gap grows as $J(\sigma) \approx \frac{1}{2}\phi''(s) \cdot \sigma^2$ (by Taylor expansion of $\phi$ around $s$). This is continuous and increasing from zero — unlike the threshold case, the benefit onset is immediate (which is why the marginal derivative condition for C4 works in this case, though we retain the integral condition for uniformity).

By C5, $\Delta_N(\sigma) \leq L_N \cdot \sigma$.

By C4, there exists $\sigma$ where $\beta \cdot J(\sigma) > \alpha \cdot L_N \cdot \sigma$. Since $J(\sigma) \sim \sigma^2$ and the cost is $\sim \sigma$, for sufficiently small $\sigma$ the cost dominates; but C4 guarantees a finite $\sigma$ where the integrated benefit exceeds cost.

The peak exists because $J(\sigma)$ is bounded: as $\sigma$ increases, the noise distribution spreads beyond the convex region, $\phi$ encounters concave or flat regions, and $J(\sigma)$ saturates or declines. Meanwhile, $\Delta_N$ continues to grow. The crossover is guaranteed. $\square$

**Remark (Relationship between 1a and 1b).** Theorem 1a is not a special case of 1b. Threshold functions are not differentiable at $\theta$ and need not be convex anywhere — the SR mechanism operates through tail probability, not curvature. Sigmoid functions fall under both: they have a threshold-like transition (1a applies with an effective threshold) *and* local convexity below the inflection point (1b applies). The computational validation confirms both paths independently.

### Theorem 1c: Multiplicative Interaction Systems (Bjork Path)

A third class of systems exhibits the inverted-U through neither threshold detection nor curvature-based Jensen gap, but through the product of opposing monotone functions. This class covers the "desirable difficulties" phenomenon in learning (Bjork and Bjork 1992, 2011) and resolves the question of whether it belongs to the same family as SR.

**Setup.** Consider a system where level-N+1 performance is the product of two functions of noise amplitude $d$:

$$P_{N+1}(d) = g(d) \cdot h(d)$$

where $g(d)$ is monotonically increasing (the per-success benefit of difficulty) and $h(d)$ is monotonically decreasing (the probability of success under difficulty).

**Condition 2c (Multiplicative Interaction Nonlinearity).** The level-N+1 response is the product $g(d) \cdot h(d)$ of an increasing function $g$ (conditional benefit given success) and a decreasing function $h$ (probability of success), where $g$ increases faster than $h$ decreases near $d = 0$, so that $P_{N+1}'(0) > 0$.

**Condition 3c (Interior Maximum Accessibility).** There exists $d^* > 0$ where $g'(d^*) \cdot h(d^*) = -g(d^*) \cdot h'(d^*)$, i.e., the marginal gain in per-success benefit exactly offsets the marginal loss in success probability. This $d^*$ is accessible to the system.

**Theorem 1c.** A two-level system satisfying C1, C2c, C3c, C4, and C5 exhibits net generative level-crossing with an inverted-U performance curve peaking at some $\sigma^* > 0$.

**Proof.** By C1, $P_{N+1}(0) = g(0) \cdot h(0) < P_{N+1}^{max}$: the system is suboptimal at zero difficulty (easy retrieval produces low encoding benefit despite high success rate).

By C2c, $P_{N+1}'(0) = g'(0) \cdot h(0) + g(0) \cdot h'(0) > 0$: performance increases with initial difficulty. By C3c, $P_{N+1}$ has an interior maximum at $d^*$: the product peaks where the increasing and decreasing components balance. Therefore $P_{N+1}(d^*) > P_{N+1}(0)$.

By C5, $\Delta_N(d) \leq L_N \cdot d$. By C4, there exists $d$ where $\beta \cdot \Delta_{N+1}(d) > \alpha \cdot \Delta_N(d)$.

The inverted-U in $P_{sys}$ follows: the system benefit rises (as the product increases toward $d^*$) then falls (as the product declines past $d^*$ and the level-N cost continues to grow). $\square$

**Mapping to Bjork's model.** In the storage strength / retrieval strength framework (Bjork and Bjork 1992; Pavlik and Anderson 2005):

- $g(d)$ = rate of storage strength gain per successful retrieval, which increases with difficulty (low retrieval strength at practice time → high encoding benefit). Formalized in the ACT-R framework as: the decay rate $d_i$ of the $i$-th memory trace decreases when activation at study time is low.
- $h(d)$ = probability of successful retrieval, which decreases with difficulty. Formalized as $P(\text{recall}) = \sigma(B_i - \tau)$ where $B_i$ is base-level activation and $\tau$ is the retrieval threshold.
- The product $L(d) = g(d) \cdot h(d)$ is the expected learning rate per practice opportunity — the quantity that determines optimal spacing, interleaving, and testing schedules.

Pyc and Rawson (2009) confirmed the key structural prediction: conditional on successful retrieval, learning benefit is monotonically increasing in difficulty. The inverted-U in unconditional learning arises entirely from the product with success probability. Wilson et al. (2019) derived the same structure from gradient-descent learning, finding that the optimal error rate for learning is approximately 15.87% (the "85% rule") — the point where the product of gradient sensitivity and discriminability peaks.

**Remark (Why this is not a Jensen gap).** The function $P_{N+1}(d) = g(d) \cdot h(d)$ is not generally locally convex near $d = 0$. The second derivative $P_{N+1}''(d) = g''h + 2g'h' + gh''$ contains the cross-term $2g'h'$, which is always negative (product of positive and negative first derivatives). This pulls the function toward concavity. Adding noise to difficulty around the optimal $d^*$ (where $P_{N+1}$ is concave) would *reduce* expected learning — Jensen's inequality works against you at the peak. The inverted-U arises from the calculus of a product, not from convex rectification of noise. The mechanism is formally distinct.

**Remark (The three mechanism classes).** Theorems 1a, 1b, and 1c establish three distinct proof paths to the inverted-U, all sharing the five abstract conditions:

| Class | Mechanism | Proof Tool | Diagnostic |
|---|---|---|---|
| 1a: Threshold SR | Tail probability reaches threshold | Kosko forbidden interval theorem | Benefit requires hard threshold; $f$ non-differentiable |
| 1b: Jensen gap | Convex rectification of continuous noise | Jensen's inequality + Taylor expansion | Benefit requires $f'' > 0$ in operating region |
| 1c: Opposing monotones | Product of increasing benefit and decreasing success | First-order calculus (product rule) | Conditional on success, benefit is monotonically increasing |

The five conditions are the common ancestor. What varies is the mechanism generating the level-N+1 benefit. This resolves the question (raised in Section 2.8 of earlier drafts) of whether the inverted-U is "universal or family-specific": it is a family of theorems sharing a common set of sufficient conditions, with (at least) three distinct proof paths.

### Lemma 1: The Integral Condition for C4

The natural formulation of C4 — that the marginal rate of N+1 benefit exceeds the marginal rate of N cost at $\sigma \to 0^+$ — fails for an important class of systems. We formalize why and provide the correct condition.

**Lemma 1 (Integral vs. Marginal C4).** For threshold systems satisfying C2a, the marginal derivative condition

$$\beta \cdot \frac{\partial P_{N+1}}{\partial \sigma}\bigg|_{\sigma \to 0^+} > \alpha \cdot \left| \frac{\partial P_N}{\partial \sigma}\bigg|_{\sigma \to 0^+} \right|$$

is *neither necessary nor sufficient* for net benefit. The correct condition is the integral formulation in C4.

**Proof.** Consider a threshold detector with signal $s < \theta$ and Gaussian noise $\xi \sim \mathcal{N}(0, \sigma^2)$. The detection probability is:

$$P_{N+1}(\sigma) = \Phi\left(\frac{s - \theta}{\sigma}\right)$$

where $\Phi$ is the standard normal CDF. As $\sigma \to 0^+$ with $s < \theta$, $P_{N+1}(\sigma) \to 0$ exponentially fast. The derivative $\partial P_{N+1}/\partial\sigma$ at $\sigma \to 0^+$ is also exponentially small — vanishing faster than any polynomial.

Meanwhile, $|\partial P_N/\partial\sigma|_{\sigma \to 0^+}| = L_N > 0$ (the level-N cost begins immediately).

Therefore $\beta \cdot 0 < \alpha \cdot L_N$ for any $\alpha, \beta, L_N > 0$. The marginal condition is never satisfied, regardless of the actual benefit at finite $\sigma$.

But the actual detection probability becomes substantial at finite $\sigma$ (specifically, when $\sigma \approx |\theta - s|$, the tail of the noise distribution reaches the threshold). At this $\sigma$, $\Delta_{N+1}$ can be large while $\Delta_N$ is still moderate. The integral condition correctly identifies this regime.

Conversely, a system might satisfy the marginal condition (strong initial benefit, weak initial cost) yet fail to show net benefit at any finite $\sigma$ because the benefit saturates before overcoming accumulated cost. The marginal condition is optimistic; the integral condition is exact. $\square$

**Remark.** For systems satisfying C2b (Jensen gap path), the marginal condition *can* work because $J(\sigma) \sim \sigma^2$ is continuous from zero. However, we state C4 in integral form universally to avoid case-splitting in applications. The integral condition is always correct; the marginal condition is a special case valid only when benefit onset is immediate.

### Corollaries

**Corollary 1 (The Brittleness Trap).** A system that minimizes redundancy at level N (e.g., lean operations with no slack) violates C5 and cannot exhibit generative level-crossing, regardless of how well C1-C4 are satisfied. Efficiency kills generativity when it eliminates the buffer that absorbs noise.

**Corollary 2 (The Alignment Trap).** A system with perfectly aligned incentives ($b = 0$ in Crawford-Sobel) has a nearly lossless channel, producing minimal variation $V_N$. If $V_N \approx 0$, then C1 is more likely violated (the signal at N+1 may be adequate without noise). Perfect alignment can satisfy N+1 directly, eliminating the need for level-crossing — but also eliminating the capacity for it when the environment shifts.

**Corollary 3 (The Over-Noise Catastrophe).** For any system satisfying C1-C5, there exists $\bar{\sigma}$ beyond which $P_{sys}(\sigma) < P_{sys}(0)$. The inverted-U is bounded. There is always too much noise. The question is never "is noise good?" but "how much, of what kind, given this threshold structure?"

**Corollary 4 (The Selection Valence Theorem).** The five conditions determine whether noise CAN be net-generative. Proposition 1 (Section 2.2) determines the VALENCE — whether the generative output is creative (divergent selection) or dysfunctional (convergent selection). The two results compose: Theorem 1 governs quantity; Proposition 1 governs direction.

**Corollary 5 (Trivial C4 under Non-Monotonic Level-N Cost).** When level-N performance is itself improved by noise (e.g., exploration in a bandit setting improves both immediate and long-term reward), C4 is satisfied trivially — the "cost" is negative, so any positive gain at N+1 exceeds it. In systems where noise helps both levels, the only binding constraint is the over-noise catastrophe (Corollary 3). The inverted-U still holds, but the peak shifts rightward because the system tolerates more noise before net degradation.

---

## 2.6 The Tolerance Location Principle

Synthesizing Crawford-Sobel, Blau-Michaeli, Atlan, the forbidden interval theorem, and Theorem 1:

**Theorem (Tolerance Location Principle).** A system's resilience and generative capacity are maximized when:

1. **Distinguishable core and interface.** The system has a structurally identifiable boundary between its internal processing (core) and its interaction with the environment (interface). The core maintains rigidity (invariant structures, stable representations). The interface maintains tolerance (absorptive capacity for environmental variance).

2. **Sufficient absorptive capacity at the interface.** The interface must satisfy Ashby's Law of Requisite Variety: its response repertoire must be at least as rich as the variance it faces. Formally: $H(\text{interface responses}) \geq H(\text{environmental perturbations relevant to the system})$.

3. **Task-relevant information preservation across the boundary.** The transformation from environmental noise to internal signal must satisfy Tishby's Information Bottleneck criterion: minimize $I(X; T)$ (compression) subject to maintaining $I(T; Y)$ (relevance). The interface compresses environmental signal maximally while preserving what the core needs.

**Corollary 6 (Inverted Architecture Failure).** When tolerance is placed at the core and rigidity at the interface — flexible internals, rigid boundaries — the system loses both resilience and generative capacity.

**Corollary 7 (The Sterility of Perfection).** A system with zero tolerance at the interface ($H(\text{interface responses}) = 0$) has zero generative capacity. Perfect, lossless transmission eliminates the generative residual, producing a system that is deterministic and sterile.

**Corollary 8 (The Inseparability of Dysfunction and Creativity).** Since both dysfunctional drift and creative novelty arise from the same generative residual $G_i$ under the same Blau-Michaeli constraint, no system can eliminate dysfunction potential without simultaneously eliminating creative potential. The design problem is not elimination but *channeling*.

---

## 2.7 The Five-Part Formal Chain (Summary)

| Step | Source | Contribution | Formal Tool |
|------|--------|-------------|-------------|
| 1 | Crawford-Sobel (1982) | Any channel with preference divergence is endogenously lossy | Game theory, partition theorem |
| 2 | Blau-Michaeli (2018, 2019) | Lossy compression maintaining coherent output MUST diverge from source — generatively | Rate-distortion-perception tradeoff |
| 3 | Atlan (1979) / von Foerster (1960) | Noise at level N becomes information at level N+1 | Level-crossing principle |
| 4 | Kosko (various) | Noise benefit occurs iff noise mean outside forbidden interval of threshold structure | Forbidden interval theorem (necessary & sufficient) |
| 5 | **Theorem 1 (this paper)** | **Five sufficient conditions for net benefit; the inverted-U is guaranteed** | **Proof + computational validation** |

**The chain:** Channels are endogenously lossy (CS) → lossy reconstruction is necessarily creative (BM) → this creativity is functional when it crosses levels (Atlan) → the conditions for functional crossing are formally specifiable (Kosko) → **the net benefit is guaranteed when five conditions hold (Theorem 1)**.

---

## 2.8 Testable Predictions

From the formal foundation:

1. **Cross-substrate transfer:** The inverted-U noise-benefit curve should appear in any system satisfying the five conditions, regardless of substrate.

2. **Compression-novelty correlation:** In organizational communication, teams with moderate compression ratios should produce more novel output than teams with either minimal or maximal compression.

3. **Selection-valence prediction:** Two organizations with identical compression ratios but different selection criteria should exhibit identical noise levels but opposite valences.

4. **Eigenvalue signature:** In network analysis of organizational communication, the eigenvalue spectrum should right-shift when the generative residual produces novel structure and left-shift under convergent selection.

5. **Linguistic compression as leading indicator:** Hedging density, passive voice ratio, and specificity metrics should degrade before operational performance metrics, because channel degradation is the cause and operational failure is the effect.

6. **Condition-dependent SR in activation spaces:** Noise injection should improve decomposition quality specifically where C1 is met (high collinearity) and fail where C1 is not met (low collinearity). If noise helps everywhere regardless of C1, the mechanism is dithering, not subthreshold detection.

---

# Section 3: Computational Validation

## 3.1 Strategy

Theorem 1 asserts five sufficient conditions for net generative level-crossing. "Sufficient" is a claim about the real line: whenever all five conditions hold, there exists a noise level $\sigma^* > 0$ that improves system performance. This section subjects that claim to adversarial computational testing across six mechanistically distinct domains: threshold detection, sigmoid detection, polynomial detection (null model), simulated annealing, ensemble diversity, and multi-armed bandit exploration. The simulation code, data, and parameter files are publicly available.

The adversarial posture is deliberate. The simulations were designed to find counterexamples — parameter configurations where all five conditions are met but noise fails to help. If the theorem survives, it survives testing designed to break it.

## 3.2 Stochastic Resonance Class: Three Nonlinearities and a Null

### Setup

A two-level system is defined:

- **Level N**: Performance $P_N(\sigma)$ degrades with noise amplitude $\sigma$. Three degradation models tested: linear ($P_N = 1 - L\sigma$), exponential ($P_N = e^{-L\sigma}$), and catastrophic ($P_N$ drops sharply past a cliff threshold).
- **Level N+1**: Performance $P_{N+1}(\sigma)$ is the detection probability of a subthreshold signal under noise, computed from a nonlinear detector.
- **System performance**: $P_{sys}(\sigma) = f(P_N, P_{N+1})$ under three coupling models: additive ($\alpha P_N + \beta P_{N+1}$), multiplicative ($P_N^\alpha \cdot P_{N+1}^\beta$), and substrate-dependent ($P_N \cdot (\alpha + \beta P_{N+1})$).

Four detector types test the role of nonlinearity (C2):

1. **Threshold detector**: $P(\text{signal} + \text{noise} > \theta)$. The textbook SR case.
2. **Sigmoid detector**: $\text{mean}[\sigma_k(s + \xi - \theta)]$ with steepness $k = 10$. Smooth nonlinearity, no hard threshold.
3. **Polynomial detector**: $\text{mean}[\max(0, s + \xi - \theta)^3]$. Nonlinear but gradual — tests whether "non-affine" is sufficient or whether steepness matters.
4. **Linear detector** (null model): $\text{mean}[s + \xi]$. No nonlinearity. If this shows benefit, something is wrong.

Each configuration is swept across 100 noise levels from $\sigma = 0$ to $\sigma = 3$, with 10,000 trials per noise level and 200 bootstrap resamples for confidence intervals.

### Results: Core Multi-Type Comparison

| Configuration | Net Benefit | Optimal $\sigma^*$ | Inverted-U | Conditions Met |
|---|---|---|---|---|
| Threshold, all conditions met | **0.170** | 0.45 | Yes | All 5 |
| Sigmoid, all conditions met | **0.091** | 0.33 | Yes | All 5 |
| Polynomial, all conditions met | **0.000** | — | No | C2 marginal |
| Linear (null model) | **0.000** | — | No | C2 violated |
| C1 violated (signal above threshold) | **0.000** | — | No | C1 violated |
| C5 violated (brittle + substrate coupling) | **0.000** | — | No | C5 violated |

The threshold and sigmoid detectors both exhibit the predicted inverted-U. The polynomial detector produces zero benefit despite being non-affine — its Jensen gap is negligible because the above-threshold response is too gradual. The linear null model produces exactly zero benefit, confirming that nonlinearity (C2) is essential, not incidental.

**Key finding on C2**: The nonlinearity must be *steep enough* relative to the noise variance to produce a meaningful Jensen gap. The gap between sigmoid (works) and polynomial (doesn't) suggests the critical feature is the slope at threshold, consistent with Chapeau-Blondeau's (1997) generalization of SR to arbitrary static nonlinearities.

### Results: Degradation and Coupling Models

| Coupling Model | Degradation | Net Benefit | Notes |
|---|---|---|---|
| Additive | Linear | 0.170 | Baseline |
| Additive | Exponential | 0.186 | Exponential degrades slower initially |
| Additive | Catastrophic | 0.242 | All-or-nothing = less cost at moderate $\sigma$ |
| Multiplicative | Linear | 0.457 | Largest: multiplicative coupling amplifies gains |
| Substrate | Linear | 0.112 | Lower: gains tied to N health |
| Additive (brittle, L=5.0) | Linear | 0.033 | Marginal: C5 nearly violated but additive saves it |
| Substrate (brittle, L=5.0) | Catastrophic | 0.000 | C5 violated: substrate + brittleness = zero benefit |

The coupling model determines whether C5 matters. Under additive coupling, a brittle lower level still yields marginal benefit because $P_{N+1}$ contributes independently. Under substrate coupling, brittleness is fatal. System architecture determines which conditions bind.

## 3.3 Non-SR Mechanisms: Generality Beyond Stochastic Resonance

### Mechanism 1: Optimization Landscape Escape (Simulated Annealing)

A 1D fitness landscape with a local optimum at $x = 0$ (height 0.6) and a global optimum at $x = 3$ (height 1.0). Noise is Metropolis temperature.

| Test | Net Benefit | Conditions Met | Inverted-U |
|---|---|---|---|
| Baseline (stuck in local optimum) | **0.254** | All | Yes |
| C1 violated (at global optimum) | 0.000 | No | No |
| C4 violated (stability >> exploration) | 0.000 | No | No |
| Null (flat landscape) | 0.000 | No | No |

### Mechanism 2: Ensemble Diversity (Error Decorrelation)

21 predictors on a hard XOR classification problem. Individual accuracy approximately 58-66%. Noise perturbs predictor weights.

| Test | Net Benefit | Conditions Met | Inverted-U |
|---|---|---|---|
| Baseline (weak individuals, hard problem) | **0.068** | All | Yes |
| C4 violation attempt ($\alpha = 0.95$) | 0.057 | All* | Yes |

*C4 violation not effectively induced. Hard XOR makes all individuals weak regardless. Simulation design limitation; theory prediction correct.

### Mechanism 3: Multi-Armed Bandit (Exploration-Exploitation)

10 arms with unknown rewards. Noise is exploration probability.

| Test | Net Benefit | Conditions Met | Inverted-U |
|---|---|---|---|
| Baseline (one hidden best arm) | **0.345** | All | Yes |
| C1 violated (all arms equal) | 0.000 | No | No |
| C4 violation attempt ($\alpha = 0.95$) | 0.276 | All* | Yes |

*Bandit exploration improves immediate reward (discovers better arms), making level-N cost negative. C4 trivially satisfied per Corollary 5.

### Cross-Mechanism Summary

Six mechanisms produce inverted-U behavior when conditions are met. Two null models confirm conditions are load-bearing. The theorem generalizes beyond SR.

## 3.4 Monte Carlo Sensitivity Analysis

500 random parameter configurations. Conditions checked dynamically.

| Category | Count | Percentage |
|---|---|---|
| All conditions met AND benefit observed | 75 | 15.0% |
| All conditions met AND no benefit | **0** | **0.0%** |
| Some condition violated AND benefit observed | 38 | 7.6% |
| Some condition violated AND no benefit | 387 | 77.4% |

**Sufficiency rate: 100%.** Zero counterexamples. The conditions are sufficient but not necessary (8.9% of violations still showed benefit).

## 3.5 Crawford-Sobel Strategic Communication Simulation

Actual partition equilibria with receiver estimation error. Bias swept from 0.01 to 0.25.

| Bias $b$ | Partitions | Effective Noise | Information Transmitted | Babbling |
|---|---|---|---|---|
| 0.010 | 6 | 0.059 | 96.0% | No |
| 0.036 | 3 | 0.110 | 85.6% | No |
| 0.062 | 2 | 0.155 | ~70% | No |
| 0.087+ | 1 | 0.291 | 0.0% | **Yes** |

The transition from informative communication to babbling is sharp. Small incentive misalignment creates quantization noise that preserves most information while introducing the generative residual. Large misalignment collapses communication entirely.

## 3.6 Summary of Computational Evidence

1. **Sufficiency confirmed.** 75/75, zero counterexamples.
2. **Generality confirmed.** Six mechanisms, four substrates.
3. **Null models confirmed.** Linear and polynomial detection show zero benefit.
4. **C4 requires integral formulation.** Marginal derivative fails for threshold detectors.
5. **C2 requires sufficient steepness.** Non-affinity is necessary but not sufficient.
6. **C5 depends on coupling architecture.** Additive vs. substrate coupling determines relevance.
7. **Crawford-Sobel validates the organizational anchor.** Sharp babbling transition confirmed.

---

# Section 4: Instance Papers as Evidence

## 4.1 The Evidentiary Structure

A theory of lossy channels and generative reconstruction makes a strong claim: the mechanism is substrate-independent. If that claim is correct, the same formal structure — compression forces reconstruction, reconstruction diverges from source, selection determines whether the divergence is functional or pathological — should appear in substrates that share no surface features. If it appears only in one domain, the theory is a domain-specific model dressed up in general language. If it appears across organizational communication, legal accountability, linguistic behavior, computational geometry, and cognitive architecture, the generality claim has teeth.

This section maps six instance papers onto the formal framework of Section 2. Each paper was developed independently, in a different research context, with different methods and different data. The mapping was discovered after the fact — the instance papers were not designed to validate the framework, and the framework was not reverse-engineered from the instances. The convergence is therefore evidential rather than circular.

The instances vary in evidentiary strength and receive treatment proportional to their contribution:

- **Deep treatment with quantitative condition verification** (Sections 4.3 and 4.5): *Variance Compression* (75 filings, 25 companies, each condition verified against measured lexical diversity and Shannon entropy) and *Activation Space Geometry* (controlled experiment with designed null condition and preliminary results; each condition verified against collinearity and noise sweep data).
- **Standard treatment** (Sections 4.2 and 4.6): *Dysmemic Pressure* (Nokia 76-interview study, Challenger Rogers Commission, Wells Fargo CFPB enforcement records) and *Jazz Improvisation* (three documented cases of creative emergence through lossy real-time communication).
- **Brief treatment as boundary cases** (Sections 4.4 and 4.7): *Structural Immunity* (the babbling equilibrium limit) and *Hierarchical Context Distillation* (functional compression; theoretical mapping only).

The instance papers cited here are working papers by the author (McEntire), available as SSRN preprints. They have not undergone independent peer review. The framework's formal structure (Section 2) and computational validation (Section 3) do not depend on any instance; the instances serve as evidence of cross-substrate applicability, not as foundations for the theorem.

## 4.2 Dysmemic Pressure: Selection-Shaped Noise in Organizational Hierarchies

Organizations compress information to coordinate at scale. Reports round. Dashboards aggregate. Executive summaries discard nuance. The pathology begins when signals that survive the compression are selected not for accuracy but for organizational fitness.

McEntire's theory of Dysmemic Pressure formalizes this as a compound evolutionary force: (1) strategic communication degradation at each hierarchical interface (Crawford-Sobel), (2) adverse selection in idea markets where cheap optimism outcompetes expensive accuracy, and (3) transmission bias where simplicity, prestige, and conformity outcompete truth-value.

Three cases with documented internal records — Nokia (Vuori and Huy 2016, 76-interview study), NASA Challenger (Rogers Commission, Vaughan 1996), and Wells Fargo (CFPB enforcement, 2016) — demonstrate the mechanism. In each case, accurate information existed at level N, the lossy channel filtered it, and the reconstruction at level N+1 was shaped by convergent selection: optimism, schedule adherence, and metric satisfaction outcompeted accuracy.

**Mapping**: Lossy channel = hierarchical reporting with preference divergence. Compression = multi-dimensional reality into metric summaries. Selection = internal fitness (transmissibility, prestige, conformity). Valence = convergent → the Cage.

## 4.3 Variance Compression: Linguistic Evidence of the Formalization Trap

If the framework is correct, the transition to convergent selection should leave observable traces in language. The Variance Compression study measures lexical diversity and Shannon entropy across 75 filings from 25 companies undergoing IPO. This instance receives extended treatment because its quantitative data permit direct verification of each condition.

### The Data

Amazon S-1 lexical diversity 0.1742, dropping 14.6% by first 10-K. Google S-1 at 0.1825 (highest in dataset), dropping 17.6%. Compression magnitude correlates with legal threat: Coinbase dropped 33.0% under SEC investigation; Snowflake dropped 9.4% with no enforcement action. Controls (Adobe, Intuit) showed flat lexical diversity across years — compression is event-driven, not drift. The "born-caged" effect (Cohort 3) shows modern S-1s arrive pre-compressed: legal teams embed standardized language upstream.

### Quantitative Condition Verification

| Condition | Verification in Variance Compression Data |
|---|---|
| C1 | **Met.** Pre-IPO lexical diversity (S-1 mean 0.168) is well below theoretical maximum for the genre (~0.25 for unrestricted business prose). The channel is not already at capacity. |
| C2 | **Met.** The legal-defensibility response function is nonlinear: below a threshold of "sufficient standardization," each unit of novel language increases legal risk sharply. Above the threshold, marginal standardization has diminishing returns. |
| C3 | **Met.** Companies with moderate compression (Snowflake, 9.4% drop) retain more communicative capacity than those with extreme compression (Coinbase, 33.0% drop). |
| C4 | **Inverted.** Convergent-pathology instance. Legal defensibility (α) dominates communicative richness (β). C4 is met *for the dysfunction direction*. |
| C5 | **Graduated, not brittle.** The three-cohort structure shows compression is progressive, not catastrophic. |

### Mapping

Lossy channel = formalization event imposing legal requirements. Compression = novel founding narrative into standardized, defensible prose. Selection = legal defensibility. Valence = convergent → firms lose the linguistic capacity to articulate novel positions. The Cohort 3 "born-caged" effect is the longitudinal endpoint: once compression is embedded upstream of the channel, the generative residual is eliminated before it can form.

## 4.4 Structural Immunity: The Babbling Equilibrium Limit Case

Structural Immunity represents the framework's limiting case: a channel engineered to be so lossy that signal transmission approaches zero. When platform indispensability combines with mandatory arbitration, class waivers, and confidentiality, six cumulative filters reduce approximately 160 million eligible consumers to 32 affirmative decisions — a 0.00025% conversion rate. The mechanism is claim suppression, not adjudication bias. This is the organizational analogue of the Crawford-Sobel babbling equilibrium: the channel output is statistically independent of the input. C1 is maximally met (the system is maximally suboptimal), but C5 is violated at the limit — the channel is not gracefully degraded but engineered to be dead. The framework correctly predicts that no generative reconstruction is possible when the channel carries zero information.

## 4.5 Activation Space Geometry: Stochastic Resonance in Computational Substrates

The strongest test of substrate-independence is a system with no organizational, linguistic, or legal features. This instance receives extended treatment because it is a controlled experiment with a designed null condition that directly tests the theorem's predictions.

### The System

In neural network model composition, domain centroids in high-dimensional activation space are orthogonalized via Gram-Schmidt to extract domain-specific components. At 7B parameters (collinearity 0.973), the orthogonal residuals have near-zero norms and the composition degrades from 93.3% win rate at 3B to 60.7%.

Theorem 1 predicts that if the domain-specific information is subthreshold (C1 met) rather than absent, noise injection before orthogonalization should produce an inverted-U in composition quality.

### Quantitative Condition Verification

| Condition | Verification in Activation Space Data |
|---|---|
| C1 | **Met at 7B** (collinearity 0.973, win rate degrades from 93.3% to 60.7%). **Not met at 0.5B** (collinearity 0.940, baseline near-optimal). The dissociation is the critical test. |
| C2 | **Met.** Gram-Schmidt orthogonalization is a threshold-like operation: when the component norm falls below numerical precision, the domain-specific signal is effectively zeroed. Jensen gap confirmed positive in preliminary runs. |
| C3 | **Met.** Gaussian noise with σ scaled to centroid norm standard deviation can perturb the geometry enough to push collapsed components above the resolution threshold. |
| C4 | **Met.** The evaluation weights composition quality (level N+1) directly; individual centroid fidelity (level N) is instrumental, not terminal. |
| C5 | **Met.** Individual centroid norms degrade gradually under noise perturbation; no cliff in degradation below σ/‖c‖ ≈ 0.10. |

### The Controlled Null Prediction

The critical test is the C1-gated dissociation: noise should help at 7B (C1 met, high collinearity, large subthreshold gap) and should *not* help at 0.5B (C1 not met, low collinearity, baseline already near-optimal). Preliminary results at the 0.5B scale (GPT-2, collinearity 0.940) show a small but real effect: +2 percentage points at σ* ≈ 0.01 in generation quality, with a clear inverted-U peaking at noise fraction 0.010 and declining to baseline by 0.10. This modest effect at a scale where C1 is marginal is consistent with the framework's prediction: the large effect should appear specifically at 7B where the subthreshold gap is maximal. If noise helps equally at both scales, the mechanism is regularization, not stochastic resonance, and the framework loses this instance (falsification condition F4). Results at the 7B scale are pending at time of writing.

## 4.6 Jazz Improvisation: Creative Emergence Through Lossy Real-Time Communication

*Note: This instance is illustrative, not validating. It provides a vivid demonstration of the framework's mechanism under divergent selection, but the evidence is documentary and anecdotal, not quantitative. It should be read as a worked example that makes the formal structure concrete, not as empirical confirmation of Theorem 1.*

The preceding instances demonstrate convergent pathology (Sections 4.2-4.4) and computational diagnostics (Section 4.5). The dual-valence claim requires a positive instance: a case where the lossy channel mechanism produces genuine creative novelty under divergent selection.

### The Instance

In ensemble jazz improvisation, each musician transmits musical intent through an inherently lossy channel: the acoustic signal, filtered by the listener's auditory processing, harmonic vocabulary, rhythmic frame, and stylistic training. No musician can perfectly decode another's intent. A chord voicing implies a harmonic trajectory that the pianist intended but the saxophonist hears differently — because their respective training compresses the same acoustic signal into different harmonic categories. A rhythmic displacement that the drummer plays as a specific polyrhythmic reference arrives to the bassist as ambiguous — within the bassist's frame, it is consistent with multiple interpretations. The residual — the gap between the player's intent and the listener's perception — is filled by the listening musician's own priors.

This is the Crawford-Sobel channel operating in real time. The "bias" is not strategic misalignment but *ontological* misalignment: each musician has a different compression scheme (harmonic vocabulary, rhythmic frame) that partitions the continuous acoustic signal into different categories. The partition theorem applies directly: the more divergent the musical backgrounds, the coarser the mutual partitions, the larger the generative residual.

### The Evidence

The mechanism is visible at the individual interaction level in documented cases:

**Miles Davis's Second Great Quintet (1964-68).** Davis deliberately assembled musicians with incompatible approaches: Herbie Hancock's bop-based harmonic vocabulary, Wayne Shorter's modal and through-composed orientation, Tony Williams's polyrhythmic extensions of Elvin Jones's approach, Ron Carter's walking bass tradition. Davis gave minimal direction — often just a sketch, sometimes only a tempo and a key center. The lossy channel between musicians was maximally wide: each player's compression scheme for the incoming acoustic data was different from every other's. The result — documented on records including *E.S.P.*, *Miles Smiles*, *Nefertiti*, and *Filles de Kilimanjaro* — was the most consequential period of harmonic innovation in post-bop jazz. Forms dissolved. Preset harmonic schemes gave way to what the musicians called "time, no changes" — structures that emerged from interaction rather than from composition.

The key observation: the innovations were not composed by Davis and executed by the band. They were *emergent from the lossy channel*. Hancock has described in interviews moments where he played what he thought was a "wrong" chord — a misinterpretation of the harmonic trajectory Davis intended — and Davis responded not by correcting him but by incorporating the chord into a new direction. The "error" at level N (fidelity to the intended harmonic progression) became information at level N+1 (a new harmonic pathway that neither musician would have composed alone). This is Atlan level-crossing in real time, with the audience serving as the divergent selection criterion.

**Ornette Coleman's "harmolodics" (1958-).** Coleman's Free Jazz (1960) — two simultaneously improvising quartets with no predetermined harmony — represents the extreme case: the lossy channel is maximally wide (no shared harmonic frame), the compression is extreme (each musician partitions the acoustic signal using only their own melodic logic), and the selection is purely divergent (the audience and critical community that selected for novelty over harmonic correctness). The result was a genuine artistic revolution — not noise, not chaos, but structured novelty arising from forced reconstruction across incompatible frames.

**Keith Jarrett's solo concerts (1973-).** Jarrett's *The Köln Concert* (1975) demonstrates the mechanism operating within a single musician. Playing entirely improvised material on an unfamiliar, partially broken piano, Jarrett was forced to reconstruct his musical intent through an instrument that resisted his habitual patterns. The lossy channel was physical — the piano's mechanical limitations compressed his intentions — and the reconstruction drew on his gospel, classical, and jazz training simultaneously. The result was not what he would have composed on a functioning Steinway. It was something neither he nor the piano "contained."

### The Condition Check

| Condition | Status in Jazz Improvisation |
|---|---|
| C1: Suboptimal | **Met.** The collective musical output of musicians playing from a score is bounded by the composition. Improvisation has room for improvement — the collective can exceed any individual's compositional capacity. |
| C2: Nonlinear | **Met.** Musical cognition is nonlinear: a note that falls within the listener's expected harmonic frame produces a qualitatively different response (confirmation) than one that falls outside it (surprise, reinterpretation). The response to signal-plus-noise is not a linear function of the acoustic input. |
| C3: Accessible | **Met.** The noise (divergent interpretation) can reach the improvement region — harmonic reinterpretations can produce new progressions that are musically valid. |
| C4: Weighting | **Met.** The jazz aesthetic values collective innovation (N+1) over individual fidelity to a score (N). The selection environment — audience, critics, recording decisions — rewards emergence over accuracy. |
| C5: Robustness | **Met.** Individual musicianship degrades gracefully under misinterpretation. A skilled jazz musician can absorb a "wrong" chord and respond musically, not collapse. This is what training provides: the redundancy that allows productive absorption of noise. |

### The Mapping

| Framework Component | Jazz Improvisation Instance |
|---|---|
| Lossy channel | Acoustic/cognitive channel between musicians with divergent harmonic vocabularies |
| Compression | Continuous acoustic signal partitioned into different harmonic/rhythmic categories by each listener |
| Forced reconstruction | Listening musician fills the gap between perceived and intended with their own musical priors |
| Selection criterion | Aesthetic novelty, audience response, recording decisions (divergent selection) |
| Valence | **Divergent → creative emergence.** The reconstruction produces harmonic, melodic, and structural innovations that no individual musician composed. |

### What This Validates

This instance completes the dual-valence evidence base. The Dysmemic Pressure instance (Section 4.2) demonstrates the identical mechanism under convergent selection — organizational communication channels with preference divergence produce self-confirming delusion. Jazz improvisation demonstrates the mechanism under divergent selection — musical communication channels with ontological divergence produce genuine novelty. Same channel structure. Same compression. Same forced reconstruction. Different selection criterion. Different valence.

The jazz instance also demonstrates why Corollary 1 (the Brittleness Trap) matters in creative domains. A musician who cannot absorb misinterpretation without losing function — a brittle player — cannot participate in generative improvisation. Training provides the redundancy (C5) that allows noise to become signal. This is why conservatory-trained musicians who have not developed improvisational fluency often cannot "play jazz" despite technical superiority: their performance function collapses under the noise that the improviser's function absorbs and exploits.

---

## 4.7 Hierarchical Context Distillation: Functional Compression

This final instance addresses a completeness question: if compression inevitably produces problems or novelty, how do functional hierarchies exist at all? Three formalisms — cognitive limits (Miller 1956, Sweller 1988), the Information Bottleneck principle (Tishby et al. 1999), and Ashby's Law of Requisite Variety — converge on the answer: middle management performs information compression that is *necessary for scale*, not pathological. The manager's formal job is the IB objective: minimize $I(X; T)$ while maximizing $I(T; Y)$. Mission command — transmit intent, not instruction; authorize deviation when local conditions require it — is hierarchical distillation under divergent selection. The valence is **design-dependent**: well-calibrated compression is functional; mis-calibrated compression is the Context Trap. This instance contributes no new empirical data but integrates established results to show that the framework's predictions about functional lossy channels are consistent with the organizational design literature.

## 4.8 Cross-Instance Synthesis

| Instance | Substrate | Channel | Selection | Valence |
|---|---|---|---|---|
| Dysmemic Pressure | Organizational hierarchy | Preference divergence | Internal fitness | Convergent → Cage |
| Variance Compression | Strategic language | Legal formalization | Defensibility | Convergent → linguistic cage |
| Structural Immunity | Legal architecture | Arbitration + confidentiality | Repeat-player position | Convergent → suppression |
| Activation Space | Computational geometry | High-dimensional projection | Composition quality | Divergent → diagnostic |
| Jazz Improvisation | Musical interaction | Acoustic/cognitive divergence | Aesthetic novelty | Divergent → creative emergence |
| Hierarchical Distillation | Cognitive architecture | Channel capacity limits | Relevance preservation | Both → design-dependent |

No two instances share a substrate. The formal structure is identical across all six. The two instances with quantitative condition verification (Variance Compression and Activation Space) demonstrate that the five conditions are not merely qualitatively plausible but measurably present in real data. The four supporting instances provide breadth: three convergent-pathology cases, one creative emergence case, and one functional-compression case. The dual-valence claim — that the same mechanism produces dysfunction under convergent selection and creativity under divergent selection — is supported by instances on both sides of the valence divide, with the Dysmemic Pressure (convergent) and Jazz Improvisation (divergent) instances sharing identical channel structure under opposite selection criteria.

---

# Section 5: Limitations, Falsification, and Open Questions

## 5.1 What the Theory Claims and Does Not Claim

The framework claims that five conditions are jointly *sufficient* for net-beneficial noise in a two-level system. It does not claim they are necessary. It does not claim that noise is generally good. It does not claim that the same optimal noise level applies across substrates. It claims: check these five conditions; if all hold, moderate noise helps; if any fails, no guarantee.

The framework also claims that the same lossy-channel mechanism produces both organizational dysfunction and creative emergence, with the valence determined by the selection environment. It does not claim that dysfunction and creativity are identical — it claims they share a root cause.

## 5.2 Falsification Conditions

**F1: Counterexample to sufficiency.** All five conditions verified as met, and the inverted-U does not appear. Zero counterexamples found across 500 Monte Carlo configurations and six mechanisms.

**F2: Mechanism-independent benefit.** Noise produces benefit with zero conditions met. Not observed.

**F3: Symmetric valence.** Two systems with identical channel properties but different selection criteria produce the same output valence. Would invalidate Proposition 1.

**F4: Scale-independent SR.** The companion activation space experiment: noise helps equally at 7B (C1 met) and 0.5B (C1 not met). Would reduce the mechanism to regularization.

**F5: Beneficial noise in a linear system.** Noise produces net system benefit where the integration function is strictly linear and the Jensen gap is identically zero. Not observed in computational null model.

## 5.3 Limitations of the Computational Validation

**Sufficient vs. necessary.** The 8.9% of configurations showing benefit despite condition violations means the conditions define a conservative boundary. Tightening the gap between sufficient and necessary is open.

**Ensemble C4 test failure.** The C4 violation test for ensemble diversity did not effectively induce violation because the hard XOR problem makes all individual predictors weak regardless of weighting. The theory's prediction is correct; the test failed to create the intended condition.

**C2 steepness threshold.** The minimum steepness required for a meaningful Jensen gap is identified (between sigmoid and polynomial) but not quantified. Connection to Chapeau-Blondeau (1997) on generalized SR for arbitrary nonlinearities.

**Crawford-Sobel simplifications.** The one-sender-one-receiver model with uniform priors. Real hierarchies involve multiple senders, heterogeneous priors, and reputation effects.

## 5.4 Open Questions

**O1: Multi-level cascades.** The theorem covers two levels. Does it compose across $N$ levels? Connection to Rosas et al. (2024) on hierarchical emergence via computational mechanics.

**O2: Dynamic environments.** Static conditions assumed. In non-stationary environments, the optimal noise level must track shifting thresholds.

**O3: Quantitative $\sigma^*$ prediction.** The theorem guarantees existence but not a closed-form. The empirical calibration approach ($\sigma = k \cdot f(\text{operating point})$) is currently the most defensible prescription.

**O4: Connection to causal emergence.** Hoel et al. (2013) proved effective information can peak at the macro level. The relationship between our Theorem 1 and positive causal emergence is suggestive but unproved.

**O5: Tightening C4.** The integral condition is correct but hard to verify a priori. A simpler sufficient condition on $\beta/\alpha$ and $L$ that implies the integral condition would be practically valuable.

**O6: Additional mechanism classes.** Theorems 1a, 1b, and 1c establish three proof paths to the inverted-U under the shared conditions. Whether additional mechanism classes exist — beyond threshold SR, Jensen gap, and opposing monotones — is an open empirical question. Any system satisfying C1-C5 that exhibits the inverted-U through a mechanism not reducible to the three established classes would extend the family.

## 5.5 What a Null Result Means

The framework makes precise predictions. Some will fail. When they do, the failure is informative:

- If the activation space SR experiment fails at high collinearity, the 7B degradation is geometric collapse, not threshold failure — redirecting the engineering effort toward nonlinear decomposition.
- If the inverted-U fails where conditions appear met, the most likely explanation is misassessed C2 or C4 — the failure is diagnosable.
- If noise helps at 0.5B (C1 not met), the mechanism is regularization — the theory loses one instance, not its structure.

The framework is designed to be wrong precisely. A theory that cannot be wrong cannot be useful.

---

## References

Ashby, W. R. (1956). *An Introduction to Cybernetics*. Chapman & Hall.

Atlan, H. (1979). *Entre le cristal et la fumée*. Éditions du Seuil.

Beer, S. (1972). *Brain of the Firm*. Allen Lane.

Bjork, R. A., & Bjork, E. L. (1992). A new theory of disuse and an old theory of stimulus fluctuation. In A. Healy et al. (Eds.), *From Learning Processes to Cognitive Processes: Essays in Honor of William K. Estes* (Vol. 2, pp. 35-67). Erlbaum.

Bjork, R. A., & Bjork, E. L. (2011). Making things hard on yourself, but in a good way: Creating desirable difficulties to enhance learning. *Psychology and the Real World*, 56-64.

Blau, Y., & Michaeli, T. (2018). The perception-distortion tradeoff. *CVPR 2018*.

Blau, Y., & Michaeli, T. (2019). Rethinking lossy compression: The rate-distortion-perception tradeoff. *ICML 2019*.

Chapeau-Blondeau, F. (1997). Stochastic resonance in the Heaviside nonlinearity with white noise and arbitrary signal. *Physical Review E*, 55(2), 2016.

Crawford, V. P., & Sobel, J. (1982). Strategic information transmission. *Econometrica*, 50(6), 1431-1451.

Edmondson, A. (1999). Psychological safety and learning behavior in work teams. *Administrative Science Quarterly*, 44(2), 350-383.

Gammaitoni, L., Hänggi, P., Jung, P., & Marchesoni, F. (1998). Stochastic resonance. *Reviews of Modern Physics*, 70(1), 223.

Gordon, R. (2022). The information bottleneck in hierarchical organizations. Working paper.

Hargadon, A., & Sutton, R. I. (1997). Technology brokering and innovation in a product development firm. *Administrative Science Quarterly*, 42(4), 716-749.

Hoel, E. P., Albantakis, L., & Tononi, G. (2013). Quantifying causal emergence shows that macro can beat micro. *Proceedings of the National Academy of Sciences*, 110(49), 19790-19795.

Janis, I. L. (1972). *Victims of Groupthink*. Houghton Mifflin.

Kirkpatrick, S., Gelatt, C. D., & Vecchi, M. P. (1983). Optimization by simulated annealing. *Science*, 220(4598), 671-680.

Kosko, B., & Mitaim, S. (2003). Stochastic resonance in noisy threshold neurons. *Neural Networks*, 16(5-6), 755-761.

Krogh, A., & Vedelsby, J. (1994). Neural network ensembles, cross validation, and active learning. *NIPS 1994*.

McDonnell, M. D., & Abbott, D. (2009). What is stochastic resonance? Definitions, misconceptions, debates, and its relevance to biology. *PLoS Computational Biology*, 5(5), e1000348.

Miller, G. A. (1956). The magical number seven, plus or minus two. *Psychological Review*, 63(2), 81-97.

Rosas, F. E., Mediano, P. A. M., Jensen, H. J., Seth, A. K., Barrett, A. B., Carhart-Harris, R. L., & Bor, D. (2024). Reconciling emergences: An information-theoretic approach to identify causal emergence in multivariate data. *PLoS Computational Biology*.

Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.

Sweller, J. (1988). Cognitive load during problem solving: Effects on learning. *Cognitive Science*, 12(2), 257-285.

Tishby, N., Pereira, F. C., & Bialek, W. (1999). The information bottleneck method. *Proceedings of the 37th Allerton Conference*.

Vaughan, D. (1996). *The Challenger Launch Decision*. University of Chicago Press.

von Foerster, H. (1960). On self-organizing systems and their environments. In *Self-Organizing Systems* (pp. 31-50). Pergamon.

Vuori, T. O., & Huy, Q. N. (2016). Distributed attention and shared emotions in the innovation process: How Nokia lost the smartphone battle. *Administrative Science Quarterly*, 61(1), 9-51.

Pavlik, P. I., & Anderson, J. R. (2005). Practice and forgetting effects on vocabulary memory: An activation-based model of the spacing effect. *Cognitive Science*, 29(4), 559-586.

Pyc, M. A., & Rawson, K. A. (2009). Testing the retrieval effort hypothesis: Does greater difficulty correctly recalling information lead to higher levels of memory? *Journal of Memory and Language*, 60(4), 437-447.

Weick, K. E. (1979). *The Social Psychology of Organizing* (2nd ed.). McGraw-Hill.

Wilson, R. C., Shenhav, A., Straccia, M., & Cohen, J. D. (2019). The eighty five percent rule for optimal learning. *Nature Communications*, 10(1), 4646.
