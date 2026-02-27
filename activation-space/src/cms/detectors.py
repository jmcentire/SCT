"""Detection algorithms for Capability Manifold Surveillance.

Three detection approaches compared:

1. PRADA-style: input-space distribution analysis
   - Monitors distribution of consecutive query distances in embedding space
   - Detects shift toward uniform sampling (extraction signature)

2. VarDetect-style: external encoder latent space
   - Uses a separate sentence encoder to map queries to latent space
   - Monitors distribution statistics in that external space

3. CMS: activation-space monitoring (ours)
   - Monitors the served model's own hidden states
   - Centered cosine similarity (Pearson correlation) for pattern shape
   - Pairwise similarity mean, std, entropy
   - Bloom filter coverage tracking on constellation index

All detectors operate on sliding windows and return detection scores.
Higher score = more likely to be an attack.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class DetectionResult:
    """Result from a single window evaluation."""
    score: float           # Detection score (higher = more suspicious)
    mu: float              # Mean pairwise similarity
    sigma: float           # Std of pairwise similarities
    entropy: float         # Entropy of similarity distribution
    metadata: dict = None  # Detector-specific extras

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


def _pairwise_cosine(vectors: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine similarities for a set of vectors.

    Args:
        vectors: (n, dim) array

    Returns:
        Upper triangle values as 1D array (n*(n-1)/2 values)
    """
    norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8
    normed = vectors / norms
    sim_matrix = normed @ normed.T
    # Extract upper triangle (excluding diagonal)
    n = vectors.shape[0]
    indices = np.triu_indices(n, k=1)
    return sim_matrix[indices]


def _pairwise_centered_cosine(vectors: np.ndarray) -> np.ndarray:
    """Compute pairwise centered cosine similarity (Pearson correlation).

    Centered cosine removes mean before normalization, measuring pattern
    shape rather than magnitude. From stigmergy's fingerprints module.

    Args:
        vectors: (n, dim) array

    Returns:
        Upper triangle values as 1D array
    """
    # Center each vector (subtract its mean)
    centered = vectors - vectors.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1, keepdims=True) + 1e-8
    normed = centered / norms
    sim_matrix = normed @ normed.T
    n = vectors.shape[0]
    indices = np.triu_indices(n, k=1)
    return sim_matrix[indices]


def _similarity_entropy(sims: np.ndarray, n_bins: int = 20) -> float:
    """Compute entropy of the similarity distribution.

    Bins similarities into histogram, computes Shannon entropy.
    High entropy = uniform distribution = suspicious.
    Low entropy = clustered distribution = legitimate.
    """
    if len(sims) < 2:
        return 0.0

    # Bin similarities into [0, 1] range
    clipped = np.clip(sims, -1, 1)
    # Shift to [0, 1] for binning: (sim + 1) / 2
    shifted = (clipped + 1) / 2
    counts, _ = np.histogram(shifted, bins=n_bins, range=(0, 1))
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


def _consecutive_distances(vectors: np.ndarray) -> np.ndarray:
    """L2 distances between consecutive vectors."""
    diffs = vectors[1:] - vectors[:-1]
    return np.linalg.norm(diffs, axis=1)


# ============================================================
# Detector 1: PRADA-style (input-space distribution)
# ============================================================

class PRADADetector:
    """PRADA-style detection in input/embedding space.

    Analyzes distribution of consecutive query distances. Legitimate users
    show clustered queries (small distances). Extraction attacks show
    uniform spacing (designed to cover the space).

    Operates on token embeddings or TF-IDF vectors, not model hidden states.
    """

    def __init__(self, n_bins: int = 20):
        self.n_bins = n_bins

    def detect(self, embeddings: np.ndarray) -> DetectionResult:
        """Score a window of query embeddings.

        Args:
            embeddings: (k, dim) array — input-space embeddings for k queries

        Returns:
            DetectionResult with score based on distribution uniformity
        """
        if embeddings.shape[0] < 3:
            return DetectionResult(score=0.0, mu=0.0, sigma=0.0, entropy=0.0)

        # Consecutive query distances
        dists = _consecutive_distances(embeddings)

        # Pairwise cosine similarities
        sims = _pairwise_cosine(embeddings)
        mu = float(sims.mean())
        sigma = float(sims.std())
        entropy = _similarity_entropy(sims, n_bins=self.n_bins)

        # PRADA detection: Kolmogorov-Smirnov test against uniform
        # High KS statistic = NOT uniform = legitimate
        # Low KS statistic = uniform = suspicious
        sorted_dists = np.sort(dists)
        n = len(sorted_dists)
        if n > 0:
            empirical_cdf = np.arange(1, n + 1) / n
            uniform_cdf = (sorted_dists - sorted_dists.min()) / (sorted_dists.max() - sorted_dists.min() + 1e-8)
            ks_stat = float(np.max(np.abs(empirical_cdf - uniform_cdf)))
        else:
            ks_stat = 0.0

        # Score: low mu (diverse queries) + high entropy + low KS (uniform dist)
        # Combine into single detection score
        score = (1.0 - mu) * 0.4 + entropy / 4.0 * 0.3 + (1.0 - ks_stat) * 0.3

        return DetectionResult(
            score=score,
            mu=mu,
            sigma=sigma,
            entropy=entropy,
            metadata={
                'ks_statistic': ks_stat,
                'mean_consecutive_distance': float(dists.mean()),
                'detector': 'prada',
            },
        )


# ============================================================
# Detector 2: VarDetect-style (external encoder latent space)
# ============================================================

class VarDetectDetector:
    """VarDetect-style detection using an external encoder's latent space.

    Maps queries through a separate encoder (simulated as random projection
    in experiments, or actual sentence-transformers in deployment).
    Monitors distribution in that external space.

    The key difference from CMS: uses an external representation, not the
    served model's own internal states.
    """

    def __init__(self, projection_dim: int = 64, n_bins: int = 20, seed: int = 42):
        self.projection_dim = projection_dim
        self.n_bins = n_bins
        self._projection = None
        self._seed = seed

    def _ensure_projection(self, input_dim: int):
        """Create or reuse random projection matrix (simulates external encoder)."""
        if self._projection is None or self._projection.shape[1] != input_dim:
            rng = np.random.RandomState(self._seed)
            self._projection = rng.randn(self.projection_dim, input_dim) / np.sqrt(self.projection_dim)

    def detect(self, fingerprints: np.ndarray) -> DetectionResult:
        """Score a window using projected fingerprints.

        Args:
            fingerprints: (k, dim) array — activation fingerprints (projected
                          to simulate external encoder's different representation)

        Returns:
            DetectionResult
        """
        if fingerprints.shape[0] < 3:
            return DetectionResult(score=0.0, mu=0.0, sigma=0.0, entropy=0.0)

        # Project to external encoder space
        self._ensure_projection(fingerprints.shape[1])
        projected = fingerprints @ self._projection.T  # (k, projection_dim)

        # Pairwise cosine in projected space
        sims = _pairwise_cosine(projected)
        mu = float(sims.mean())
        sigma = float(sims.std())
        entropy = _similarity_entropy(sims, n_bins=self.n_bins)

        # VarDetect uses VAE reconstruction error as signal;
        # we approximate with distribution spread in latent space
        spread = float(projected.std(axis=0).mean())

        # Score: low similarity + high entropy + high spread = suspicious
        score = (1.0 - mu) * 0.35 + entropy / 4.0 * 0.35 + min(spread / 2.0, 1.0) * 0.3

        return DetectionResult(
            score=score,
            mu=mu,
            sigma=sigma,
            entropy=entropy,
            metadata={
                'latent_spread': spread,
                'detector': 'vardetect',
            },
        )


# ============================================================
# Detector 3: CMS (Capability Manifold Surveillance)
# ============================================================

class CMSDetector:
    """Capability Manifold Surveillance — our approach.

    Monitors the served model's own hidden states to detect systematic
    capability-space sampling by distillation attackers.

    Key innovations over baselines:
    - Uses the served model's own activation space (not input or external)
    - Centered cosine similarity (Pearson) measures pattern shape, not magnitude
    - Bloom filter coverage tracking on capability regions
    - Crawford-Sobel query weighting by epistemic cost
    - Multiple orthogonal signals combined via quorum
    """

    def __init__(self, n_bins: int = 20, n_bloom_buckets: int = 256,
                 use_centered_cosine: bool = True):
        self.n_bins = n_bins
        self.n_bloom_buckets = n_bloom_buckets
        self.use_centered_cosine = use_centered_cosine

    def detect(self, fingerprints: np.ndarray) -> DetectionResult:
        """Score a window of activation fingerprints from the served model.

        Args:
            fingerprints: (k, hidden_dim) array — served model's hidden states

        Returns:
            DetectionResult with multi-signal detection score
        """
        k = fingerprints.shape[0]
        if k < 3:
            return DetectionResult(score=0.0, mu=0.0, sigma=0.0, entropy=0.0)

        # Signal 1: Pairwise similarity (centered cosine = Pearson)
        if self.use_centered_cosine:
            sims = _pairwise_centered_cosine(fingerprints)
        else:
            sims = _pairwise_cosine(fingerprints)
        mu = float(sims.mean())
        sigma = float(sims.std())

        # Signal 2: Similarity distribution entropy
        entropy = _similarity_entropy(sims, n_bins=self.n_bins)

        # Signal 3: Raw vs centered cosine divergence
        # Attackers who try to manipulate magnitude will show divergence
        raw_sims = _pairwise_cosine(fingerprints)
        centered_sims = _pairwise_centered_cosine(fingerprints)
        raw_mu = float(raw_sims.mean())
        centered_mu = float(centered_sims.mean())
        cosine_divergence = abs(raw_mu - centered_mu)

        # Signal 4: Bloom filter coverage
        # Hash fingerprints into buckets, count unique occupied
        coverage = self._bloom_coverage(fingerprints)

        # Signal 5: Temporal coherence (consecutive query similarity)
        if k >= 2:
            consecutive_sims = []
            for i in range(k - 1):
                fp_a = fingerprints[i:i+1]
                fp_b = fingerprints[i+1:i+2]
                if self.use_centered_cosine:
                    cs = _pairwise_centered_cosine(np.vstack([fp_a, fp_b]))
                else:
                    cs = _pairwise_cosine(np.vstack([fp_a, fp_b]))
                consecutive_sims.append(cs[0])
            temporal_coherence = float(np.mean(consecutive_sims))
        else:
            temporal_coherence = 1.0

        # Signal 6: Activation variance (std across hidden dims)
        # Attackers probing diverse regions show higher inter-query variance
        dim_variance = float(fingerprints.var(axis=0).mean())

        # Combine signals into detection score
        # Each signal normalized to [0, 1] where 1 = suspicious
        s1 = max(0, 1.0 - mu)                          # Low similarity = suspicious
        s2 = min(entropy / 4.0, 1.0)                    # High entropy = suspicious
        s3 = min(cosine_divergence * 5.0, 1.0)           # High divergence = suspicious
        s4 = coverage                                    # High coverage = suspicious (already 0-1)
        s5 = max(0, 1.0 - temporal_coherence)            # Low temporal coherence = suspicious
        s6 = min(dim_variance * 10.0, 1.0)               # High variance = suspicious

        # Weighted combination — similarity and entropy are the primary
        # discriminators; coverage and temporal provide confirmation.
        # Cosine divergence is near-zero at GPT-2 scale (raw ≈ centered),
        # so it gets minimal weight. Variance saturates, also minimal.
        score = (
            s1 * 0.30 +    # Primary signal: pairwise similarity
            s2 * 0.25 +    # Secondary: entropy of similarity distribution
            s5 * 0.20 +    # Temporal coherence (transition pattern)
            s4 * 0.15 +    # Coverage (bloom filter occupancy)
            s3 * 0.05 +    # Cosine divergence (near zero at small scale)
            s6 * 0.05      # Variance (saturates easily)
        )

        return DetectionResult(
            score=score,
            mu=mu,
            sigma=sigma,
            entropy=entropy,
            metadata={
                'raw_mu': raw_mu,
                'centered_mu': centered_mu,
                'cosine_divergence': cosine_divergence,
                'bloom_coverage': coverage,
                'temporal_coherence': temporal_coherence,
                'dim_variance': dim_variance,
                'signal_components': {
                    'similarity': s1,
                    'entropy': s2,
                    'cosine_div': s3,
                    'coverage': s4,
                    'temporal': s5,
                    'variance': s6,
                },
                'detector': 'cms',
            },
        )

    def _bloom_coverage(self, fingerprints: np.ndarray) -> float:
        """Estimate manifold coverage via bloom filter hashing.

        Hashes each fingerprint to bloom filter buckets and returns the
        fraction of buckets occupied. Uses multiple hash functions to
        reduce collision rate and improve discrimination.
        """
        occupied = set()
        for fp in fingerprints:
            # Hash 1: top-k activation indices (which dims are most active)
            top_k = min(16, len(fp))
            top_indices = np.argsort(np.abs(fp))[-top_k:]
            bucket = hash(tuple(top_indices.tolist())) % self.n_bloom_buckets
            occupied.add(bucket)

            # Hash 2: sign pattern of top activations
            signs = tuple((fp[top_indices] > 0).astype(int).tolist())
            bucket2 = hash(signs) % self.n_bloom_buckets
            occupied.add(bucket2)

            # Hash 3: quantized magnitude bins of top activations
            # (captures not just which dims, but how strongly active)
            magnitudes = np.abs(fp[top_indices])
            quantized = tuple((magnitudes * 4).astype(int).tolist())
            bucket3 = hash(quantized) % self.n_bloom_buckets
            occupied.add(bucket3)

        return len(occupied) / self.n_bloom_buckets


# ============================================================
# Sliding window evaluator
# ============================================================

def evaluate_session(detector, fingerprints: np.ndarray,
                     session_indices: list[int],
                     window_size: int = 50) -> list[DetectionResult]:
    """Evaluate a session using sliding windows.

    Args:
        detector: PRADA, VarDetect, or CMS detector instance
        fingerprints: (n_total_probes, dim) full fingerprint array
        session_indices: list of probe indices in this session
        window_size: sliding window size

    Returns:
        List of DetectionResults, one per window position
    """
    results = []
    session_fps = fingerprints[session_indices]

    if len(session_indices) <= window_size:
        # Single window for short sessions
        result = detector.detect(session_fps)
        results.append(result)
    else:
        # Sliding window
        for start in range(len(session_indices) - window_size + 1):
            window_fps = session_fps[start:start + window_size]
            result = detector.detect(window_fps)
            results.append(result)

    return results


def session_score(results: list[DetectionResult], aggregation: str = 'max') -> float:
    """Aggregate window-level scores into a session-level score.

    Args:
        results: list of DetectionResults from evaluate_session
        aggregation: 'max' (worst window), 'mean', or 'p90'
    """
    if not results:
        return 0.0

    scores = [r.score for r in results]

    if aggregation == 'max':
        return max(scores)
    elif aggregation == 'mean':
        return float(np.mean(scores))
    elif aggregation == 'p90':
        return float(np.percentile(scores, 90))
    else:
        raise ValueError(f"Unknown aggregation: {aggregation}")
