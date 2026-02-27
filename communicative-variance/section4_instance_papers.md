# Section 3: Instance Papers as Evidence

## 3.1 The Evidentiary Structure

A theory of lossy channels and generative reconstruction makes a strong claim: the mechanism is substrate-independent. If that claim is correct, the same formal structure — compression forces reconstruction, reconstruction diverges from source, selection determines whether the divergence is functional or pathological — should appear in substrates that share no surface features. If it appears only in one domain, the theory is a domain-specific model dressed up in general language. If it appears across organizational communication, legal accountability, linguistic behavior, computational geometry, and cognitive architecture, the generality claim has teeth.

This section maps five instance papers onto the formal framework of Section 2. Each paper was developed independently, in a different research context, with different methods and different data. The mapping was discovered after the fact — the instance papers were not designed to validate the framework, and the framework was not reverse-engineered from the instances. The convergence is therefore evidential rather than circular.

The mapping is one-directional. Each instance validates the framework. The framework does not depend on any single instance. If any instance fails to map cleanly, the framework loses one piece of evidence but not its formal foundation, which rests on the Crawford-Sobel, Blau-Michaeli, Atlan, and Kosko results independently.

---

## 3.2 Dysmemic Pressure: Selection-Shaped Noise in Organizational Hierarchies

### The Instance

Organizations compress information to coordinate at scale. Reports round. Dashboards aggregate. Executive summaries discard nuance. This compression is not pathological — it is physically and cognitively necessary (Section 3.6 below). The pathology begins when signals that survive the compression are selected not for accuracy but for organizational fitness.

McEntire's theory of Dysmemic Pressure formalizes this as a compound evolutionary force operating on three dynamics:

1. **Strategic communication degradation** (Crawford-Sobel): At each hierarchical interface, preference divergence between sender and receiver degrades transmission precision. Engineers soften bad news. Managers filter for palatability. Executives present optimistically to boards. The compound effect across multiple interfaces can approach babbling equilibrium — messages statistically independent of the true state.

2. **Adverse selection in idea markets** (Akerlof parallel): Producing accurate assessments of complex situations is expensive. Producing optimism is cheap. Receivers cannot reliably distinguish accurate from cheap signals at the point of consumption. Cheap signals flood the market. Accurate signals withdraw. The market settles at noise.

3. **Transmission bias** (Boyd-Richerson): Ideas spread based on simplicity, prestige association, and conformity pressure — properties independent of truth-value. Emotionally satisfying falsehoods outcompete complex accurate signals because the selection criterion is transmissibility, not correspondence.

A *dysmeme* is defined as a cultural variant — a signal, practice, metric, or narrative — that is highly fit for internal transmission but structurally misaligned with external reality. Dysmemic pressure is the compound force that selects for these variants: incentive divergence amplifies misreporting, transmission ease rewards simplicity over accuracy, and verification cost prevents auditing.

### The Evidence

The theory is tested against three cases with documented internal communication records:

**Nokia** (Vuori and Huy 2016, 76-interview study): Engineers knew Symbian could not compete with iOS. Middle managers transmitted optimistic projections because the selection environment punished bearers of bad news. Top management received a picture of the competitive landscape that was internally consistent, transmissible, and wrong. The information existed at level N. It did not survive the channel. The reconstructed picture at level N+1 was a dysmeme: high fitness, low correspondence.

**NASA Challenger** (Rogers Commission, Vaughan 1996): Engineers documented O-ring erosion risks. Each successful launch with anomalies shifted the baseline — normalized deviance, a transmission bias that rewarded continuity over alarm. At the final decision point, program leadership's directive to "put on your management hat" captures the selection criterion explicitly: evaluate the signal for organizational fitness, not engineering accuracy. The lossy channel (hierarchical reporting under schedule pressure) produced a reconstruction (launch is safe) that was convergent with sender preferences and catastrophically wrong.

**Wells Fargo** (CFPB enforcement action, 2016): Cross-sell metrics compressed the multi-dimensional reality of customer relationships into a single integer. Employees who opened fraudulent accounts met the metric. Employees who maintained legitimate relationships did not. The metric — the compressed signal — selected for fraud, producing 3.5 million fictitious accounts. The selection environment was not criminal hiring; it was rational response to a compression scheme that discarded the dimensions on which fraud and service differ.

### The Mapping

| Framework Component | Dysmemic Pressure Instance |
|---|---|
| Lossy channel | Hierarchical reporting with preference divergence at each interface |
| Compression | Multi-dimensional reality → metric-friendly summaries |
| Forced reconstruction | Receivers fill gaps with priors shaped by organizational incentives |
| Selection criterion | Internal fitness (transmissibility, prestige, conformity) |
| Valence | **Convergent → dysfunction.** The Cage: self-confirming, increasingly decoupled from external reality |

### What This Validates

This instance validates the dual-valence claim (Proposition 1) under convergent selection. The channel is endogenously lossy (Crawford-Sobel interfaces at each hierarchical level). The reconstruction is generative (receivers synthesize coherent narratives from partitioned data). The selection environment rewards fit-over-truth. The result is organizational self-deception as a *stable equilibrium*, not a moral failure — precisely as the framework predicts.

---

## 3.3 Variance Compression: Linguistic Evidence of the Formalization Trap

### The Instance

If the framework is correct, the transition from lossy-but-generative communication to convergent-selection communication should leave observable traces in language. The Variance Compression study tests this by measuring what happens to strategic language when organizations undergo a formalization event: the initial public offering.

Pre-IPO, a company's founding narrative is high-variance by necessity. The firm is explaining a novel business model to investors who have no existing frame for it. The language must be creative, exploratory, and specific to the firm's unique position. Post-IPO, the same firm must satisfy fiduciary obligations — demonstrating prudence through documented analysis, industry-standard justifications, and quantifiable metrics. The legal selection environment shifts from "convince investors this is new" to "convince regulators this is sound."

### The Evidence

The study analyzes 75 filings (S-1 prospectuses and subsequent 10-K annual reports) from 25 companies across five cohorts spanning 1995-2023, measuring two metrics:

**Lexical diversity** (unique words / total words): Higher values indicate richer, less repetitive vocabulary.

**Shannon entropy** (base-2 information uncertainty): Higher values indicate less predictable, more varied term usage.

Key findings across cohorts:

**The baseline effect** (Cohort 1, Pre-SOX 1995-1999): Amazon's S-1 (1997) showed lexical diversity of 0.1742 and Shannon entropy of 11.61 — high variance, the language of a company inventing a category. By the first 10-K, lexical diversity dropped 14.6% to 0.1488. eBay showed a 17.0% drop. Control companies (Adobe, Intuit) already public showed flat lexical diversity (0.1305 ± 0.0004 across three years), confirming that the compression is event-driven, not temporal drift.

**The legal amplifier** (Cohort 4, High-Liability 2020-2023): Coinbase's S-1 (2021) showed the highest lexical diversity in the dataset: 0.1944 — the language of a company evangelizing a new asset class. Under SEC investigation post-IPO, the first 10-K collapsed 33.0% to 0.1302. Robinhood, amid Congressional hearings after GameStop, dropped 30.0%. The magnitude of compression correlates with legal threat intensity, not firm age or industry maturity.

**The born-caged effect** (Cohort 3, 2015-2020): Snowflake's S-1 (2020) showed lexical diversity of only 0.1495 — lower than 1990s S-1s before any post-IPO compression had occurred. Legal teams and investment banks now embed standardized language *before* the formalization event. Modern firms arrive at IPO already partially compressed. The Cage has learned to pre-select.

**The founder insulation effect** (Cohort 5, Dual-Class Structures): Meta's S-1 showed only 5.8% lexical diversity decline post-IPO, versus 15.3% for LinkedIn (comparable firm, no dual-class). Snap showed 7.2% — the mildest compression in the dataset. Dual-class control structures insulate from shareholder pressure but do not prevent compression entirely; they attenuate it.

### The Mapping

| Framework Component | Variance Compression Instance |
|---|---|
| Lossy channel | Formalization event (IPO) imposes legal requirements on strategic language |
| Compression | High-variance founding narrative → low-variance, metric-focused, legally defensible prose |
| Forced reconstruction | Firms reconstruct strategy through standardized safe-harbor language |
| Selection criterion | Legal defensibility, fiduciary demonstrability |
| Valence | **Convergent → dysfunction.** Firms lose the linguistic capacity to articulate novel strategic positions |

### What This Validates

This instance provides *quantitative, longitudinal* evidence of the compression mechanism. The framework predicts that increased selection pressure for convergence should produce measurable variance reduction. The data confirm this: compression magnitude correlates with legal exposure intensity (Coinbase -33% vs. Snowflake -9.4%), not with firm fundamentals. The born-caged effect demonstrates that the selection environment can anticipate and pre-compress — a prediction the framework makes but that purely behavioral theories of organizational conformity do not.

---

## 3.4 Structural Immunity: When the Channel Suppresses the Signal Entirely

### The Instance

The framework predicts that lossy channels under convergent selection produce dysfunction. Structural Immunity represents the limiting case: a channel engineered to be so lossy that signal transmission approaches zero.

When platform indispensability combines with pre-dispute mandatory arbitration and class action waivers, the result is a dispute resolution architecture in which valid claims are systematically filtered at every stage of a conversion pipeline — from harm, to recognized claim, to active dispute, to filed case, to adjudicated outcome. The paper terms this "structural immunity": not adjudication bias (deciding cases wrongly) but claim suppression (preventing cases from existing as legal events).

### The Evidence

The pipeline collapse is quantified:

**Consumer finance**: Approximately 160 million consumers were eligible for class settlement relief worth $2.7 billion. Under arbitration-only regimes, 32 affirmative decisions were rendered, totaling less than $400,000. Conversion rate: approximately 0.00025% of the eligible population.

**Employment arbitration**: The "overwhelming majority" of covered claims never reach any forum. Estimated filing rates: approximately 1 in 100,000, despite ubiquitous arbitration clauses.

Six cumulative filters produce this collapse:

1. **Mandatory pre-dispute arbitration**: Eliminates court access. Consumer must accept arbitration to use indispensable platform.
2. **Class action waivers**: Eliminates aggregation. Individual pursuit of small-dollar claims is economically irrational.
3. **Confidentiality**: Prevents pattern recognition. Each dispute resolves invisibly. No journalist, regulator, or academic can identify systematic misconduct from individually sealed outcomes.
4. **Repeat-player arbitrator effects**: Corporations track arbitrator behavior across hundreds of cases and strike unfavorable arbitrators from future panels. Research on approximately 9,000 FINRA securities arbitrations (Egan et al. 2025) finds that random arbitrator assignment would increase average consumer awards by approximately $60,000.
5. **Truncated discovery**: Compressed evidence procedures make complex fraud unprovable. The defendant holds all information; the plaintiff can develop almost none.
6. **Operational edge closure**: Customer support authority to resolve is stripped from every customer-facing node; maximum escalation yields $5-20.

### The Mapping

| Framework Component | Structural Immunity Instance |
|---|---|
| Lossy channel | Mandatory arbitration + confidentiality + class waivers |
| Compression | Multi-dimensional complaint → single-integer decision (resolved for $X or $0), sealed |
| Forced reconstruction | Platform reconstructs each harm as isolated transaction error; consumer reconstructs as "not worth pursuing" |
| Selection criterion | Repeat-player structural position; information asymmetry; economic rationality of non-filing |
| Valence | **Convergent → dysfunction at the limit.** The channel approaches babbling equilibrium: output is statistically independent of input. Harm volume has no effect on accountability volume. |

### What This Validates

This instance validates the framework's prediction about extreme convergent selection. Crawford-Sobel predicts that as bias increases, channel capacity degrades monotonically toward babbling. Structural Immunity demonstrates the organizational analogue: when every filter in the dispute pipeline selects for non-transmission, the system reaches a state where the output (legal accountability) is effectively independent of the input (actual harm). The channel is not merely lossy — it is engineered to be maximally lossy while maintaining the surface appearance of access. This is the Cage operating at the interface between firm and legal system, with confidentiality preventing the Mirror function (external observation that could make the compression visible).

---

## 3.5 Activation Space Geometry: Stochastic Resonance in Computational Substrates

### The Instance

The framework claims substrate-independence. The strongest test of that claim is a substrate with no organizational, linguistic, or legal features — a purely computational system where the mechanism must operate through mathematical structure alone.

In neural network model composition, the problem is this: given multiple specialist models trained on different domains, extract the domain-specific knowledge from each and combine it into a single model that outperforms any individual specialist across all domains. The approach (detailed in companion work) uses activation space fingerprinting: collect each specialist's activation patterns across held-out data, compute domain centroids, apply Gram-Schmidt orthogonalization to decompose shared versus domain-specific components, and compose using the orthogonal (domain-specific) residuals.

This works well at moderate model scales. At 3B parameters, the decomposition achieves 93.3% cross-domain win rate against task arithmetic baselines. But at 7B parameters, cosine similarity between domain centroids reaches 0.973 (collinearity), and the cross-domain win rate collapses to 60.7%. The domain-specific signal — the orthogonal residual after removing shared components — has norms near zero. The decomposition cannot detect it.

The question is: is the domain-specific information absent (geometric collapse — the specialists genuinely converge at scale, and there is nothing domain-specific left to extract), or is it present but subthreshold (the information exists in the orthogonal complement but is below the effective detection threshold of the Gram-Schmidt process)?

### The Theoretical Prediction

Theorem 1 generates a precise prediction. If the five conditions are met — the system is suboptimal without noise (C1), the integration function is nonlinear (C2), the noise can reach the improvement region (C3), the gain outweighs the cost (C4), and the lower level degrades gracefully (C5) — then injecting noise into the centroids before orthogonalization should produce an inverted-U in composition quality. The optimal noise level $\sigma^*$ should be nonzero and should scale with collinearity (more collinearity means larger signal deficit means more noise needed to push subthreshold components above the detection threshold).

The condition check:

| Condition | Status at 7B (collinearity 0.973) |
|---|---|
| C1: Suboptimal | **Met.** Win rate dropped from 93.3% to 60.7%. Clean signal fails. |
| C2: Nonlinear | **Met.** Gram-Schmidt orthogonalization involves projection, subtraction, and normalization — all nonlinear. The decomposition weights depend nonlinearly on basis orientation. |
| C3: Accessible | **Met.** Gaussian noise can perturb centroids in any direction, including directions that increase orthogonal component norms. |
| C4: Weighting | **Testable.** If noise-assisted win rate exceeds baseline, C4 is confirmed empirically. |
| C5: Robustness | **Likely met.** Centroids are averages over many activation vectors — inherently robust to moderate perturbation. |

The experiment also generates a critical *null prediction*. At 0.5B parameters (collinearity 0.906), the clean decomposition achieves near-perfect cross-domain performance. C1 is not met — the system is already operating well without noise. Theorem 1 predicts that noise should *not* help at 0.5B. If it does, the mechanism is simple regularization or dithering, not subthreshold signal detection. If noise helps specifically where C1 is met (7B) and fails where C1 is not met (0.5B), that controlled dissociation confirms the theoretical mechanism. **This null prediction is the falsification condition doing actual work.**

### The Mapping

| Framework Component | Activation Space Instance |
|---|---|
| Lossy channel | High-dimensional projection under near-collinearity |
| Compression | Domain-specific signal compressed to near-zero orthogonal norms by Gram-Schmidt |
| Forced reconstruction | Decomposition must reconstruct domain-specific weights from near-zero residuals |
| Selection criterion | Cross-domain composition quality (win rate against baselines) |
| Valence | **Divergent → diagnostic.** If SR helps: detection problem (information is there but subthreshold). If SR fails: geometric collapse (information is genuinely absent). |

### What This Validates

This instance tests the framework in a domain with no organizational, linguistic, or social features. The "channel" is high-dimensional vector projection. The "compression" is mathematical orthogonalization. The "noise" is Gaussian perturbation of centroids. If the inverted-U appears where predicted (high collinearity) and is absent where predicted (low collinearity), the framework has demonstrated cross-substrate transfer from organizational communication theory to computational linear algebra — the strongest possible evidence for substrate-independence.

The publishable figure from this instance: optimal noise fraction $\sigma^*$ plotted against collinearity across model scales (0.5B, 1.5B, 3B, 7B). If the four points trace a monotonically increasing curve, one figure tells the whole story of Theorem 1 in a domain no one has connected to the organizational creativity literature.

---

## 3.6 Hierarchical Context Distillation: Why Compression Is Necessary, Not Pathological

### The Instance

The preceding four instances demonstrate the framework under conditions where compression produces dysfunction (Sections 3.2-3.4) or serves as a diagnostic tool (Section 3.5). This final instance addresses a question the framework must answer to be complete: if compression inevitably produces dysfunction or creative divergence, how do functional hierarchies exist at all?

The Hierarchical Context Distillation paper provides the answer by formalizing what middle management actually does in information-theoretic terms. Scale creates a "Context Trap": executives have strategic coherence but lack local resolution; frontline workers have local fidelity but lack strategic context. Neither can function on the other's information without compression.

### The Formal Structure

Three independent formalisms converge on the same conclusion:

**Cognitive limits** (Miller 1956, Sweller 1988): Working memory holds 7±2 chunks. A CEO attempting to hold the state of a 1,000-person organization exceeds chunk capacity. A frontline engineer burdened with full corporate strategy suffers extraneous cognitive load that reduces tactical performance. Hierarchy solves this by making "department" a single chunk at the executive level and "corporate strategy" a single chunk at the frontline level. Each level operates within cognitive capacity because the adjacent level has compressed the relevant information.

**Information Bottleneck** (Tishby et al. 1999, applied to hierarchy by Gordon 2022): The middle manager's formal job description is the Information Bottleneck objective: minimize $I(X; T)$ (compress the input — 1,000 bug reports into a summary) while maximizing $I(T; Y)$ (preserve relevance to the strategic target — is the product launch on track?). This is not metaphor. It is a literal application of the same variational principle that governs representation learning in deep neural networks: each layer abstracts features (generalizations) from the layer below while preserving task-relevant information for the layer above.

**Requisite Variety** (Ashby 1956, Beer 1972): The environment has high variety (complexity). The executive has low variety (limited attention). Ashby's Law requires that the system's regulatory variety match the environmental variety. Hierarchy achieves this by distributing the variety budget: middle managers absorb tactical variety (solving problems before they reach the executive) and amplify strategic variety (converting low-variety commands into high-variety tactical actions). Beer's "algedonic signal" provides the safety valve: a dedicated channel bypassing standard compression for survival-critical alerts that cannot afford lossy transmission.

### The Four Operations

The middle manager performs four compression operations:

1. **Interpret**: Separate strategic imperative from passing anxiety in downward signals. (Not everything the CEO says is equally load-bearing.)
2. **Translate**: Convert abstract intent into concrete local terms. (Strategic goals become engineering tasks.)
3. **Project** (downward): Provide calibrated strategic context without cognitive overload. (Enough "why" for frontline autonomy; not so much "why" that it paralyzes.)
4. **Regulate** (upward): Compress operational data into signals meaningful to executives. (Transform 1,000 data points into a decision-relevant summary.)

### The Mapping

| Framework Component | Hierarchical Distillation Instance |
|---|---|
| Lossy channel | Each hierarchical layer compresses information for the adjacent level |
| Compression | Upward: 1,000 bug reports → "systemic risk in payments." Downward: "Expand to Asia" → "localize payment gateway" |
| Forced reconstruction | Executive reconstructs frontline reality from compressed summary. Frontline reconstructs strategy from translated intent. |
| Selection criterion | Compression calibration: relevance-preserving vs. relevance-destroying |
| Valence | **Both.** Well-calibrated compression = functional hierarchy. Mis-calibrated compression = the Context Trap (executives blind, frontline confused). |

### What This Validates

This instance validates the framework's completeness by demonstrating that compression is not inherently pathological. The Cage (Section 3.2), variance compression (Section 3.3), and structural immunity (Section 3.4) show what happens when selection pressure drives reconstruction toward convergence. Hierarchical distillation shows what happens when the compression is *designed* — when the selection criterion is relevance preservation rather than organizational fitness or legal defensibility. The framework accommodates both outcomes through Proposition 1: the mechanism is the same; the valence depends on the selection environment.

This instance also connects to the Mirror (McEntire): the organizational architecture that makes compression visible and manages it intentionally rather than denying it. Mission command — transmit intent, not instruction; authorize deviation when local conditions require it — is hierarchical distillation operating under divergent selection. The subordinate's reconstruction of the commander's intent, shaped by local information the commander does not have, is the generative residual operating functionally.

---

## 3.7 Cross-Instance Synthesis

The five instances span:

| Instance | Substrate | Channel | Selection | Valence |
|---|---|---|---|---|
| Dysmemic Pressure | Organizational hierarchy | Preference divergence at interfaces | Internal fitness | Convergent → Cage |
| Variance Compression | Strategic language | Legal formalization | Defensibility | Convergent → linguistic cage |
| Structural Immunity | Legal architecture | Arbitration + confidentiality | Repeat-player position | Convergent → claim suppression |
| Activation Space | Computational geometry | High-dimensional projection | Composition quality | Divergent → diagnostic |
| Hierarchical Distillation | Cognitive architecture | Channel capacity limits | Relevance preservation | Both → design-dependent |

No two instances share a substrate. The organizational cases involve human communication and incentives. The linguistic case involves measurable textual features. The legal case involves institutional architecture. The computational case involves linear algebra on activation vectors. The cognitive case involves information-theoretic limits on working memory.

Yet the formal structure is identical across all five:

1. A complex source must be compressed to be transmitted (Crawford-Sobel, Shannon, Tishby).
2. The compression is lossy — information is discarded (partitioning, aggregation, projection, filtering).
3. The receiver must reconstruct to act — and the reconstruction necessarily diverges from the source (Blau-Michaeli).
4. The direction of divergence is determined by the selection criterion operating on the receiver's output (Proposition 1).
5. When conditions C1-C5 are met, moderate noise at the lower level produces net benefit at the higher level (Theorem 1), following an inverted-U.

The instances also form a completeness argument. Dysmemic Pressure, Variance Compression, and Structural Immunity show the convergent-selection pathology — the Cage in three different substrates. Activation Space Geometry shows the framework operating as a diagnostic tool in a purely computational substrate. Hierarchical Distillation shows the framework's accommodation of functional compression — not all lossiness is pathological; the valence depends on the selection environment.

Together, they demonstrate that the framework is not a theory of organizational dysfunction, nor a theory of computational noise injection, nor a theory of linguistic behavior. It is a theory of what happens when lossy channels force reconstruction, and what determines whether the reconstruction creates or destroys.
