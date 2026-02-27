"""PoH-style continuous scoring for extraction detection.

Adapts the Privacy book's Proof-of-Human logarithmic effort model to
distillation detection. Instead of binary flag, accumulates a continuous
extraction-likelihood score where each anomalous window contributes evidence.

The score compounds over time with temporal decay, implementing the
"structural protection" philosophy: detection cost scales logarithmically
for the defender but exponentially for the attacker trying to evade.

Score thresholds map to response intensities:
  0.0 - 0.3: Normal monitoring
  0.3 - 0.5: Increased monitoring (log all queries)
  0.5 - 0.7: Rate limiting applied
  0.7 - 0.85: Manual review flagged
  0.85 - 1.0: Block / suspend account
"""

import numpy as np
from dataclasses import dataclass, field
from .detectors import DetectionResult


@dataclass
class ScoringState:
    """Persistent state for an account's extraction score."""
    score: float = 0.0
    n_windows: int = 0
    n_anomalous: int = 0
    history: list[float] = field(default_factory=list)
    response_level: str = 'normal'

    @property
    def score_log(self) -> float:
        """Logarithmic score for display (higher = more suspicious)."""
        if self.score <= 0:
            return 0.0
        return float(-np.log10(1.0 - min(self.score, 0.999)))


RESPONSE_LEVELS = [
    (0.0, 'normal'),
    (0.3, 'increased_monitoring'),
    (0.5, 'rate_limiting'),
    (0.7, 'manual_review'),
    (0.85, 'block'),
]


def _get_response_level(score: float) -> str:
    """Map score to response level."""
    level = 'normal'
    for threshold, name in RESPONSE_LEVELS:
        if score >= threshold:
            level = name
    return level


class ExtractionScorer:
    """Continuous extraction-likelihood scorer.

    Each window evaluation contributes evidence to a running score.
    Score accumulates via: score = 1 - prod(1 - p_i) for each window,
    matching the PoH independence model.

    Temporal decay ensures old evidence fades, requiring sustained
    anomalous behavior for high scores.
    """

    def __init__(
        self,
        anomaly_threshold: float = 0.35,
        decay_half_life: int = 50,        # windows until half-life
        base_weight: float = 0.20,        # base contribution per anomalous window
        cost_escalation: float = 2.0,     # consecutive anomalies increase weight
    ):
        """
        Args:
            anomaly_threshold: detector score above this counts as anomalous.
                Set based on observed legitimate session score distribution
                (typically ~p95 of legitimate scores).
            decay_half_life: number of windows for score half-life
            base_weight: base probability contribution per anomalous window
            cost_escalation: multiplier for consecutive anomalies
        """
        self.anomaly_threshold = anomaly_threshold
        self.decay_rate = np.log(2) / decay_half_life
        self.base_weight = base_weight
        self.cost_escalation = cost_escalation

    def update(self, state: ScoringState, detection: DetectionResult) -> ScoringState:
        """Update account score with new window observation.

        Returns updated ScoringState (mutates in place for efficiency).
        """
        state.n_windows += 1

        # Temporal decay: reduce existing score
        state.score *= np.exp(-self.decay_rate)

        if detection.score >= self.anomaly_threshold:
            state.n_anomalous += 1

            # Compute evidence weight
            # Base weight scaled by how anomalous this window is
            severity = (detection.score - self.anomaly_threshold) / (1.0 - self.anomaly_threshold + 1e-8)

            # Consecutive anomaly escalation
            recent_anomalies = sum(1 for s in state.history[-10:] if s >= self.anomaly_threshold)
            escalation = self.cost_escalation ** min(recent_anomalies, 5)

            p_i = min(self.base_weight * (1 + severity) * escalation, 0.5)

            # PoH accumulation: score = 1 - (1 - score) * (1 - p_i)
            state.score = 1.0 - (1.0 - state.score) * (1.0 - p_i)

        state.history.append(detection.score)
        state.score = min(state.score, 0.999)  # Asymptotic bound
        state.response_level = _get_response_level(state.score)

        return state

    def score_session(self, detections: list[DetectionResult]) -> ScoringState:
        """Score a full session from a sequence of window detections.

        Returns final ScoringState after processing all windows.
        """
        state = ScoringState()
        for d in detections:
            self.update(state, d)
        return state

    def score_sessions_batch(self, all_detections: list[list[DetectionResult]]) -> list[ScoringState]:
        """Score multiple sessions."""
        return [self.score_session(dets) for dets in all_detections]


class MultiSignalScorer:
    """Multi-signal scorer combining multiple detection dimensions.

    Inspired by PoH's multi-signal approach: email + phone + device + typing
    patterns each contribute independently. Here we combine:
    - Similarity signal (primary)
    - Entropy signal
    - Coverage signal
    - Temporal coherence signal

    Each signal has independent weight and threshold.
    """

    def __init__(self):
        # Thresholds calibrated from observed legitimate session distributions:
        # similarity ~0.11-0.14, entropy ~0.18-0.25, temporal ~0.11-0.13,
        # coverage ~0.15-0.25 (with 256-bucket bloom filter)
        self.signals = {
            'similarity': {'threshold': 0.20, 'weight': 0.30},
            'entropy': {'threshold': 0.30, 'weight': 0.25},
            'temporal': {'threshold': 0.20, 'weight': 0.25},
            'coverage': {'threshold': 0.30, 'weight': 0.20},
        }

    def score_detection(self, detection: DetectionResult) -> dict:
        """Decompose a detection into per-signal scores."""
        components = detection.metadata.get('signal_components', {})
        if not components:
            return {'composite': detection.score, 'signals': {}}

        signal_scores = {}
        composite = 0.0
        for name, config in self.signals.items():
            raw = components.get(name, 0.0)
            signal_scores[name] = {
                'raw': raw,
                'anomalous': raw >= config['threshold'],
                'weight': config['weight'],
            }
            composite += raw * config['weight']

        # Quorum: require at least 2 signals anomalous for high confidence
        n_anomalous = sum(1 for s in signal_scores.values() if s['anomalous'])
        quorum_bonus = 0.1 * max(0, n_anomalous - 1)
        composite = min(composite + quorum_bonus, 1.0)

        return {
            'composite': composite,
            'signals': signal_scores,
            'n_anomalous_signals': n_anomalous,
            'quorum_met': n_anomalous >= 2,
        }
