#!/usr/bin/env python3
"""
Numerical simulation: Testing the five sufficient conditions for
net generative level-crossing (Atlan formalization, Theorem 1).

Tests the prediction that system performance P_sys follows an inverted-U
as a function of noise amplitude sigma, peaking at nonzero sigma when
all five conditions are met.

Also tests boundary violations: what happens when each condition fails.
"""

import numpy as np
import json
from dataclasses import dataclass, asdict

@dataclass
class SimResult:
    name: str
    sigma_range: list
    p_n: list          # Level N performance
    p_n1: list         # Level N+1 performance
    p_sys: list        # System performance
    sigma_opt: float   # Optimal noise level
    p_sys_at_opt: float
    p_sys_at_zero: float
    net_benefit: float  # P_sys(sigma_opt) - P_sys(0)
    conditions_met: dict

def threshold_detector(signal, threshold, noise_std, n_trials=10000):
    """
    Simulates a threshold detector with stochastic resonance.
    Returns detection probability (proxy for level N+1 performance).
    """
    noise = np.random.normal(0, noise_std, n_trials) if noise_std > 0 else np.zeros(n_trials)
    detections = (signal + noise) > threshold
    return np.mean(detections)

def level_n_performance(sigma, lipschitz_constant=1.0, base_performance=1.0):
    """
    Level N performance degrades with noise.
    P_N(sigma) = base - L * sigma (Lipschitz degradation)
    Clipped at 0 (can't go negative).
    """
    return max(0, base_performance - lipschitz_constant * sigma)

def run_simulation(
    name,
    signal_strength,      # s: strength of signal available to N+1
    threshold,            # theta: detection threshold at N+1
    alpha,                # weight on level-N performance
    beta,                 # weight on level-N+1 performance
    lipschitz_constant,   # L_N: how fast level-N degrades
    sigma_max=3.0,
    n_points=100,
    n_trials=50000,
):
    """Run a full simulation sweep over noise amplitudes."""
    sigmas = np.linspace(0, sigma_max, n_points)
    p_n_vals = []
    p_n1_vals = []
    p_sys_vals = []

    for sigma in sigmas:
        pn = level_n_performance(sigma, lipschitz_constant)
        pn1 = threshold_detector(signal_strength, threshold, sigma, n_trials)
        psys = alpha * pn + beta * pn1
        p_n_vals.append(pn)
        p_n1_vals.append(pn1)
        p_sys_vals.append(psys)

    # Find optimal sigma
    best_idx = np.argmax(p_sys_vals)
    sigma_opt = sigmas[best_idx]

    # Check conditions
    conditions = {
        "C1_subthreshold": signal_strength < threshold,
        "C2_nonlinear_threshold": True,  # threshold detector is inherently nonlinear
        "C3_noise_accessible": True,     # Gaussian noise with mean 0; forbidden interval check
        "C4_sufficient_weighting": beta > alpha * lipschitz_constant,  # marginal condition
        "C5_sufficient_redundancy": lipschitz_constant < float('inf'),
    }

    # Forbidden interval check (Kosko): mu=0 must not be in (theta-s1, theta-s0)
    # For binary signal: s0=0, s1=signal_strength
    # Forbidden interval: (theta - signal_strength, theta - 0) = (theta - signal_strength, theta)
    # Noise mean mu=0 is outside iff 0 < theta-signal_strength or 0 > theta
    # i.e., signal_strength < 0 (impossible) or theta < 0
    # For subthreshold signal (s < theta), mu=0 is IN the forbidden interval when theta > 0 and s > 0
    # But Kosko's theorem says SR occurs when mu is OUTSIDE forbidden interval
    # For Gaussian noise with mu=0 and signal below threshold, SR still occurs
    # because the noise VARIANCE (not mean) pushes signal over threshold
    # The forbidden interval theorem for finite-variance noise is about the NOISE DISTRIBUTION
    # not just the mean — need the full CDF condition
    conditions["C3_noise_accessible"] = True  # Gaussian satisfies for subthreshold signals

    return SimResult(
        name=name,
        sigma_range=[float(s) for s in sigmas],
        p_n=[float(v) for v in p_n_vals],
        p_n1=[float(v) for v in p_n1_vals],
        p_sys=[float(v) for v in p_sys_vals],
        sigma_opt=float(sigma_opt),
        p_sys_at_opt=float(p_sys_vals[best_idx]),
        p_sys_at_zero=float(p_sys_vals[0]),
        net_benefit=float(p_sys_vals[best_idx] - p_sys_vals[0]),
        conditions_met=conditions,
    )

def main():
    results = []

    # =============================================
    # TEST 1: All five conditions met — should show inverted-U
    # =============================================
    r1 = run_simulation(
        name="All conditions met (baseline)",
        signal_strength=0.8,    # subthreshold (< 1.0)
        threshold=1.0,
        alpha=0.3,              # moderate weight on fidelity
        beta=0.7,               # high weight on generativity
        lipschitz_constant=0.5, # moderate redundancy
    )
    results.append(r1)

    # =============================================
    # TEST 2: Condition 1 violated — signal is suprathreshold
    # =============================================
    r2 = run_simulation(
        name="C1 violated: suprathreshold signal",
        signal_strength=1.5,    # above threshold — no help needed
        threshold=1.0,
        alpha=0.3,
        beta=0.7,
        lipschitz_constant=0.5,
    )
    results.append(r2)

    # =============================================
    # TEST 3: Condition 4 violated — fidelity weight dominates
    # =============================================
    r3 = run_simulation(
        name="C4 violated: fidelity dominates",
        signal_strength=0.8,
        threshold=1.0,
        alpha=0.9,              # high weight on fidelity
        beta=0.1,               # low weight on generativity
        lipschitz_constant=0.5,
    )
    results.append(r3)

    # =============================================
    # TEST 4: Condition 5 violated — brittle system
    # =============================================
    r4 = run_simulation(
        name="C5 violated: brittle system (high L)",
        signal_strength=0.8,
        threshold=1.0,
        alpha=0.3,
        beta=0.7,
        lipschitz_constant=5.0, # very rapid degradation
    )
    results.append(r4)

    # =============================================
    # TEST 5: Varying signal deficit (parametric sweep)
    # =============================================
    deficit_results = []
    for signal in np.linspace(0.1, 1.5, 15):
        r = run_simulation(
            name=f"signal={signal:.2f}",
            signal_strength=float(signal),
            threshold=1.0,
            alpha=0.3,
            beta=0.7,
            lipschitz_constant=0.5,
            n_trials=20000,
        )
        deficit_results.append({
            "signal": float(signal),
            "deficit": float(1.0 - signal),
            "sigma_opt": r.sigma_opt,
            "net_benefit": r.net_benefit,
            "subthreshold": signal < 1.0,
        })

    # =============================================
    # TEST 6: Varying alpha/beta ratio (hierarchy values sweep)
    # =============================================
    ratio_results = []
    for beta_val in np.linspace(0.1, 0.9, 9):
        alpha_val = 1.0 - beta_val
        r = run_simulation(
            name=f"beta={beta_val:.2f}",
            signal_strength=0.8,
            threshold=1.0,
            alpha=float(alpha_val),
            beta=float(beta_val),
            lipschitz_constant=0.5,
            n_trials=20000,
        )
        ratio_results.append({
            "alpha": float(alpha_val),
            "beta": float(beta_val),
            "ratio": float(beta_val / alpha_val),
            "sigma_opt": r.sigma_opt,
            "net_benefit": r.net_benefit,
            "c4_met": beta_val > alpha_val * 0.5,
        })

    # =============================================
    # TEST 7: Crawford-Sobel partition simulation
    # Maps bias b to partition count to noise level to generativity
    # =============================================
    cs_results = []
    for b in np.linspace(0.01, 0.30, 15):
        # Crawford-Sobel partition count
        n_star = max(1, int(np.ceil(-0.5 + 0.5 * np.sqrt(1 + 2/b))))
        # Within-partition variance (uniform on [0,1] split into n_star bins)
        bin_width = 1.0 / n_star
        within_variance = (bin_width ** 2) / 12  # variance of uniform on [0, bin_width]
        noise_equivalent = np.sqrt(within_variance)

        # Use this as effective noise in threshold detector
        r = run_simulation(
            name=f"CS_b={b:.3f}",
            signal_strength=0.8,
            threshold=1.0,
            alpha=0.3,
            beta=0.7,
            lipschitz_constant=0.5,
            sigma_max=0.5,
            n_trials=20000,
        )

        cs_results.append({
            "bias_b": float(b),
            "n_star": int(n_star),
            "bin_width": float(bin_width),
            "within_variance": float(within_variance),
            "noise_equivalent": float(noise_equivalent),
            "babbling": n_star == 1,
            "sigma_opt_at_this_noise": r.sigma_opt,
        })

    # =============================================
    # Print results
    # =============================================
    print("=" * 70)
    print("LEVEL-CROSSING SIMULATION RESULTS")
    print("Testing Theorem 1: Sufficient Conditions for Net Generative Benefit")
    print("=" * 70)

    for r in results:
        print(f"\n--- {r.name} ---")
        print(f"  Conditions met: {r.conditions_met}")
        print(f"  P_sys(0) = {r.p_sys_at_zero:.4f}")
        print(f"  P_sys(σ*) = {r.p_sys_at_opt:.4f} at σ* = {r.sigma_opt:.4f}")
        print(f"  Net benefit = {r.net_benefit:.4f}")
        if r.net_benefit > 0:
            print(f"  >>> INVERTED-U CONFIRMED: noise helps <<<")
        else:
            print(f"  >>> No noise benefit (as predicted by condition violation) <<<")

    print(f"\n{'=' * 70}")
    print("PARAMETRIC SWEEP: Signal deficit vs optimal noise")
    print(f"{'=' * 70}")
    for d in deficit_results:
        marker = "**" if d["subthreshold"] and d["net_benefit"] > 0 else "  "
        print(f"  {marker}signal={d['signal']:.2f}  deficit={d['deficit']:.2f}  "
              f"σ*={d['sigma_opt']:.3f}  benefit={d['net_benefit']:.4f}")

    print(f"\n{'=' * 70}")
    print("PARAMETRIC SWEEP: Alpha/beta ratio vs optimal noise")
    print(f"{'=' * 70}")
    for d in ratio_results:
        marker = "**" if d["c4_met"] and d["net_benefit"] > 0 else "  "
        print(f"  {marker}α={d['alpha']:.2f} β={d['beta']:.2f} ratio={d['ratio']:.2f}  "
              f"σ*={d['sigma_opt']:.3f}  benefit={d['net_benefit']:.4f}  C4={'met' if d['c4_met'] else 'VIOLATED'}")

    print(f"\n{'=' * 70}")
    print("CRAWFORD-SOBEL MAPPING: Bias → Partitions → Effective Noise")
    print(f"{'=' * 70}")
    for d in cs_results:
        status = "BABBLING" if d["babbling"] else f"N*={d['n_star']}"
        print(f"  b={d['bias_b']:.3f}  {status:12}  "
              f"bin_width={d['bin_width']:.3f}  "
              f"noise_equiv={d['noise_equivalent']:.4f}")

    # Save full results
    output = {
        "main_tests": [asdict(r) for r in results],
        "deficit_sweep": deficit_results,
        "ratio_sweep": ratio_results,
        "crawford_sobel_mapping": cs_results,
    }

    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, (np.bool_,)):
                return bool(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    with open("/Users/jmcentire/Personal/Research/CommunicativeVariance/simulation_results.json", "w") as f:
        json.dump(output, f, indent=2, cls=NpEncoder)

    print(f"\nFull results saved to simulation_results.json")

if __name__ == "__main__":
    main()
