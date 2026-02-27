"""Session simulation for CMS experiments.

Generates legitimate and attack query sessions from domain probe sets.
Each session is a sequence of probe indices that would be sent to an API.

Session types:
  Legitimate:
    - single_domain: k queries from one domain (focused user)
    - mixed_domain: 80/20 split across two domains (realistic multi-topic)
    - power_user: structured multi-domain with topical coherence
  Attack:
    - naive: uniform random across all domains
    - systematic: greedy coverage maximization (farthest-first)
    - adaptive: clustering-constrained to evade detection while covering
"""

import numpy as np
from dataclasses import dataclass, field


@dataclass
class Session:
    """A query session: sequence of probe indices with metadata."""
    probe_indices: list[int]
    session_type: str           # 'single_domain', 'mixed_domain', 'power_user',
                                # 'naive_attack', 'systematic_attack', 'adaptive_attack'
    label: int                  # 0 = legitimate, 1 = attack
    domains: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


def _domain_indices(domain_probes: dict, domain: str, offset_map: dict) -> list[int]:
    """Get global probe indices for a domain."""
    start = offset_map[domain]
    return list(range(start, start + len(domain_probes[domain])))


def _build_offset_map(domain_probes: dict) -> dict:
    """Build domain -> start_index mapping."""
    offsets = {}
    idx = 0
    for domain in domain_probes:
        offsets[domain] = idx
        idx += len(domain_probes[domain])
    return offsets


def generate_legitimate_single_domain(
    domain_probes: dict,
    k: int = 50,
    n_sessions: int = 100,
    rng: np.random.RandomState = None,
) -> list[Session]:
    """Generate single-domain legitimate sessions.

    Each session samples k queries from one domain, simulating a focused user
    (e.g., a developer asking only code questions).
    """
    if rng is None:
        rng = np.random.RandomState(42)

    offsets = _build_offset_map(domain_probes)
    domains = list(domain_probes.keys())
    sessions = []

    for i in range(n_sessions):
        domain = domains[i % len(domains)]
        pool = _domain_indices(domain_probes, domain, offsets)
        # Sample with replacement if k > pool size
        indices = rng.choice(pool, size=k, replace=(k > len(pool))).tolist()
        sessions.append(Session(
            probe_indices=indices,
            session_type='single_domain',
            label=0,
            domains=[domain],
        ))
    return sessions


def generate_legitimate_mixed_domain(
    domain_probes: dict,
    k: int = 50,
    n_sessions: int = 100,
    primary_fraction: float = 0.8,
    rng: np.random.RandomState = None,
) -> list[Session]:
    """Generate mixed-domain legitimate sessions.

    Each session draws ~80% from a primary domain and ~20% from a secondary,
    simulating a user with a main task who occasionally branches out.
    """
    if rng is None:
        rng = np.random.RandomState(43)

    offsets = _build_offset_map(domain_probes)
    domains = list(domain_probes.keys())
    sessions = []

    for i in range(n_sessions):
        d1 = domains[i % len(domains)]
        d2 = domains[(i + 1 + rng.randint(1, len(domains))) % len(domains)]
        pool1 = _domain_indices(domain_probes, d1, offsets)
        pool2 = _domain_indices(domain_probes, d2, offsets)

        k_primary = int(k * primary_fraction)
        k_secondary = k - k_primary

        idx_primary = rng.choice(pool1, size=k_primary, replace=(k_primary > len(pool1))).tolist()
        idx_secondary = rng.choice(pool2, size=k_secondary, replace=(k_secondary > len(pool2))).tolist()

        # Interleave: mostly primary with secondary sprinkled in
        indices = []
        sec_positions = set(rng.choice(k, size=k_secondary, replace=False))
        pi, si = 0, 0
        for pos in range(k):
            if pos in sec_positions and si < len(idx_secondary):
                indices.append(idx_secondary[si])
                si += 1
            elif pi < len(idx_primary):
                indices.append(idx_primary[pi])
                pi += 1
            elif si < len(idx_secondary):
                indices.append(idx_secondary[si])
                si += 1

        sessions.append(Session(
            probe_indices=indices,
            session_type='mixed_domain',
            label=0,
            domains=[d1, d2],
        ))
    return sessions


def generate_legitimate_power_user(
    domain_probes: dict,
    k: int = 50,
    n_sessions: int = 50,
    rng: np.random.RandomState = None,
) -> list[Session]:
    """Generate power-user sessions: structured multi-domain with topical runs.

    Unlike attack sessions, power users show temporal coherence — they work in
    one domain for a stretch, then switch. The transitions are bursty, not uniform.
    """
    if rng is None:
        rng = np.random.RandomState(44)

    offsets = _build_offset_map(domain_probes)
    domains = list(domain_probes.keys())
    sessions = []

    for _ in range(n_sessions):
        indices = []
        # Pick 2-3 domains with random run lengths
        n_domains = rng.choice([2, 3])
        chosen = rng.choice(domains, size=n_domains, replace=False).tolist()
        remaining = k

        for j, domain in enumerate(chosen):
            pool = _domain_indices(domain_probes, domain, offsets)
            if j == len(chosen) - 1:
                run_len = remaining
            else:
                run_len = rng.randint(max(5, remaining // 3), max(6, 2 * remaining // 3))
                run_len = min(run_len, remaining)
            run_indices = rng.choice(pool, size=run_len, replace=(run_len > len(pool))).tolist()
            indices.extend(run_indices)
            remaining -= run_len
            if remaining <= 0:
                break

        sessions.append(Session(
            probe_indices=indices[:k],
            session_type='power_user',
            label=0,
            domains=chosen,
        ))
    return sessions


def generate_naive_attack(
    domain_probes: dict,
    k: int = 50,
    n_sessions: int = 200,
    rng: np.random.RandomState = None,
) -> list[Session]:
    """Generate naive attack sessions: uniform random across all domains.

    The attacker randomly samples from the full probe pool with no strategy.
    """
    if rng is None:
        rng = np.random.RandomState(100)

    offsets = _build_offset_map(domain_probes)
    total_probes = sum(len(v) for v in domain_probes.values())
    all_indices = list(range(total_probes))
    sessions = []

    for _ in range(n_sessions):
        indices = rng.choice(all_indices, size=k, replace=(k > total_probes)).tolist()
        sessions.append(Session(
            probe_indices=indices,
            session_type='naive_attack',
            label=1,
            domains=list(domain_probes.keys()),
        ))
    return sessions


def generate_systematic_attack(
    domain_probes: dict,
    fingerprints: np.ndarray,
    k: int = 50,
    n_sessions: int = 200,
    rng: np.random.RandomState = None,
) -> list[Session]:
    """Generate systematic attack sessions: farthest-first coverage maximization.

    Uses activation fingerprints to greedily select the query that is most
    distant from already-selected queries, maximizing manifold coverage.
    Requires pre-computed fingerprints for all probes.

    Args:
        fingerprints: (n_probes, hidden_dim) array of activation fingerprints
    """
    if rng is None:
        rng = np.random.RandomState(101)

    n_probes = fingerprints.shape[0]
    # Normalize fingerprints for cosine distance
    norms = np.linalg.norm(fingerprints, axis=1, keepdims=True) + 1e-8
    fp_normed = fingerprints / norms

    sessions = []
    for _ in range(n_sessions):
        # Farthest-first traversal
        start = rng.randint(0, n_probes)
        selected = [start]
        remaining = set(range(n_probes)) - {start}

        for _ in range(k - 1):
            if not remaining:
                # Wrap around if k > n_probes
                remaining = set(range(n_probes)) - set(selected[-n_probes:])
                if not remaining:
                    remaining = set(range(n_probes))

            rem_list = list(remaining)
            # Min similarity to any selected query (max distance)
            sel_fps = fp_normed[selected]  # (n_selected, dim)
            rem_fps = fp_normed[rem_list]  # (n_remaining, dim)
            sims = rem_fps @ sel_fps.T  # (n_remaining, n_selected)
            min_sims = sims.max(axis=1)  # closest selected for each remaining
            # Pick the one with lowest max-similarity (farthest from all selected)
            best_idx = rem_list[np.argmin(min_sims)]
            selected.append(best_idx)
            remaining.discard(best_idx)

        sessions.append(Session(
            probe_indices=selected[:k],
            session_type='systematic_attack',
            label=1,
            domains=list(domain_probes.keys()),
        ))
    return sessions


def generate_adaptive_attack(
    domain_probes: dict,
    fingerprints: np.ndarray,
    k: int = 50,
    n_sessions: int = 100,
    tau_mu: float = 0.70,
    rng: np.random.RandomState = None,
) -> list[Session]:
    """Generate adaptive attack sessions: clustering-constrained coverage.

    The attacker knows about the detection threshold tau_mu and tries to maintain
    mean pairwise similarity above it while still maximizing coverage.

    Strategy: work in "shells" — select a cluster center, sample nearby queries
    (high similarity to maintain mu), then jump to a new region. The tradeoff
    is that clustering reduces coverage efficiency.

    Args:
        fingerprints: (n_probes, hidden_dim) array of activation fingerprints
        tau_mu: detection threshold to evade (attacker maintains mu > tau_mu)
    """
    if rng is None:
        rng = np.random.RandomState(102)

    n_probes = fingerprints.shape[0]
    norms = np.linalg.norm(fingerprints, axis=1, keepdims=True) + 1e-8
    fp_normed = fingerprints / norms

    # Precompute full similarity matrix
    sim_matrix = fp_normed @ fp_normed.T

    sessions = []
    for _ in range(n_sessions):
        selected = []
        used = set()

        # Work in clusters to maintain high mean similarity
        while len(selected) < k:
            # Pick a new cluster center (far from existing selections if any)
            if not selected:
                center = rng.randint(0, n_probes)
            else:
                # Pick center that's far from existing centers but not too far
                candidates = [i for i in range(n_probes) if i not in used]
                if not candidates:
                    candidates = list(range(n_probes))
                center = candidates[rng.randint(0, len(candidates))]

            selected.append(center)
            used.add(center)

            if len(selected) >= k:
                break

            # Find queries similar to center (within similarity shell)
            sims_to_center = sim_matrix[center]
            # Sort by similarity, pick queries that keep mean similarity high
            candidates = [(i, sims_to_center[i]) for i in range(n_probes) if i not in used]
            candidates.sort(key=lambda x: -x[1])  # highest similarity first

            # Add queries from this cluster until we'd drop below tau_mu
            max_cluster = min(8, k - len(selected) + 1)
            cluster_size = rng.randint(2, max(3, max_cluster)) if max_cluster > 2 else max(1, max_cluster)
            added = 0
            for idx, sim in candidates:
                if added >= cluster_size:
                    break

                # Check if adding this query keeps mean similarity above tau_mu
                test_indices = selected + [idx]
                test_fps = fp_normed[test_indices]
                test_sims = test_fps @ test_fps.T
                n = len(test_indices)
                # Mean of upper triangle (pairwise similarities)
                mask = np.triu_indices(n, k=1)
                mean_sim = test_sims[mask].mean()

                if mean_sim >= tau_mu:
                    selected.append(idx)
                    used.add(idx)
                    added += 1
                elif mean_sim >= tau_mu - 0.05:
                    # Accept with some probability (attacker is willing to risk)
                    if rng.random() < 0.3:
                        selected.append(idx)
                        used.add(idx)
                        added += 1

                if len(selected) >= k:
                    break

        sessions.append(Session(
            probe_indices=selected[:k],
            session_type='adaptive_attack',
            label=1,
            domains=list(domain_probes.keys()),
            metadata={'tau_mu_target': tau_mu},
        ))
    return sessions


def generate_all_sessions(
    domain_probes: dict,
    fingerprints: np.ndarray = None,
    k: int = 50,
    n_legit_single: int = 250,
    n_legit_mixed: int = 250,
    n_legit_power: int = 100,
    n_attack_naive: int = 200,
    n_attack_systematic: int = 200,
    n_attack_adaptive: int = 100,
    tau_mu: float = 0.70,
    seed: int = 42,
) -> dict[str, list[Session]]:
    """Generate all session types for evaluation.

    Returns dict mapping session_type -> list of Sessions.
    If fingerprints not provided, systematic and adaptive attacks are skipped.
    """
    rng = np.random.RandomState(seed)

    result = {
        'single_domain': generate_legitimate_single_domain(
            domain_probes, k=k, n_sessions=n_legit_single,
            rng=np.random.RandomState(rng.randint(0, 2**31))),
        'mixed_domain': generate_legitimate_mixed_domain(
            domain_probes, k=k, n_sessions=n_legit_mixed,
            rng=np.random.RandomState(rng.randint(0, 2**31))),
        'power_user': generate_legitimate_power_user(
            domain_probes, k=k, n_sessions=n_legit_power,
            rng=np.random.RandomState(rng.randint(0, 2**31))),
        'naive_attack': generate_naive_attack(
            domain_probes, k=k, n_sessions=n_attack_naive,
            rng=np.random.RandomState(rng.randint(0, 2**31))),
    }

    if fingerprints is not None:
        result['systematic_attack'] = generate_systematic_attack(
            domain_probes, fingerprints, k=k, n_sessions=n_attack_systematic,
            rng=np.random.RandomState(rng.randint(0, 2**31)))
        result['adaptive_attack'] = generate_adaptive_attack(
            domain_probes, fingerprints, k=k, n_sessions=n_attack_adaptive,
            tau_mu=tau_mu,
            rng=np.random.RandomState(rng.randint(0, 2**31)))

    return result


def compute_coverage(session: Session, fingerprints: np.ndarray,
                     n_bins: int = 10) -> float:
    """Compute manifold coverage of a session.

    Projects fingerprints to 2D via PCA, bins into grid, counts occupied cells.
    Returns fraction of grid cells covered (0 to 1).
    """
    fps = fingerprints[session.probe_indices]
    if fps.shape[0] < 2:
        return 0.0

    # Simple PCA via SVD for projection
    centered = fps - fps.mean(axis=0)
    try:
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        projected = centered @ Vt[:2].T  # (k, 2)
    except np.linalg.LinAlgError:
        return 0.0

    # Bin into grid
    mins = projected.min(axis=0)
    maxs = projected.max(axis=0)
    ranges = maxs - mins + 1e-8

    grid = set()
    for p in projected:
        bx = min(int((p[0] - mins[0]) / ranges[0] * n_bins), n_bins - 1)
        by = min(int((p[1] - mins[1]) / ranges[1] * n_bins), n_bins - 1)
        grid.add((bx, by))

    return len(grid) / (n_bins * n_bins)
