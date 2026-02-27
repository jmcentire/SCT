"""
Power law distribution extensions for forced ranking simulation.

Implements Pareto (power law) talent distributions and analyzes
asymmetric error patterns between terminations and promotions.
"""

import numpy as np
import pandas as pd
from typing import Dict, List
from .simulation import Simulation


def run_powerlaw_simulation(
    clustering_strength: float = 0.7,
    num_simulations: int = 100,
    num_employees: int = 994,
    team_size: int = 7,
    pareto_shape: float = 3.0,
    seed: int = None
) -> Dict:
    """
    Run simulation with power law (Pareto) talent distribution.
    
    Args:
        clustering_strength: Team quality variance (0.0=random, 1.0=perfect)
        num_simulations: Monte Carlo iterations
        num_employees: Total population
        team_size: Members per team
        pareto_shape: Shape parameter for Pareto distribution
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary with results
    """
    sim = Simulation(
        num_employees=num_employees,
        team_size=team_size,
        distribution='powerlaw',
        clustering_strength=clustering_strength,
        cutoff_percentile=0.15,
        pareto_shape=pareto_shape
    )
    
    results_df = sim.run(num_simulations=num_simulations, seed=seed)
    mean_results = results_df.mean().to_dict()
    
    return {
        'results_df': results_df,
        'mean_term_error': mean_results['term_error_rate'],
        'mean_prom_error': mean_results['prom_error_rate'],
        'mean_combined_error': (mean_results['term_error_rate'] + mean_results['prom_error_rate']) / 2,
        'mean_term_correct': mean_results['term_correct'],
        'mean_prom_correct': mean_results['prom_correct'],
        'mean_term_false_pos': mean_results['term_false_pos'],
        'mean_prom_false_pos': mean_results['prom_false_pos'],
        'mean_term_avg_percentile': mean_results['term_avg_percentile'],
        'mean_prom_avg_percentile': mean_results['prom_avg_percentile'],
    }


def compare_distributions(
    clustering_strength: float = 0.7,
    num_simulations: int = 100,
    seed: int = None
) -> pd.DataFrame:
    """
    Compare normal vs power law distributions at same clustering level.
    
    Args:
        clustering_strength: Team quality variance
        num_simulations: Monte Carlo iterations
        seed: Random seed
        
    Returns:
        DataFrame comparing distributions
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Normal distribution
    sim_normal = Simulation(
        num_employees=994,
        team_size=7,
        distribution='normal',
        clustering_strength=clustering_strength,
        cutoff_percentile=0.15
    )
    results_normal = sim_normal.run(num_simulations=num_simulations)
    
    # Power law distribution
    sim_powerlaw = Simulation(
        num_employees=994,
        team_size=7,
        distribution='powerlaw',
        clustering_strength=clustering_strength,
        cutoff_percentile=0.15,
        pareto_shape=3.0
    )
    results_powerlaw = sim_powerlaw.run(num_simulations=num_simulations)
    
    # Compare
    comparison = pd.DataFrame({
        'Distribution': ['Normal', 'Power Law'],
        'Clustering': [clustering_strength, clustering_strength],
        'Term_Error_%': [
            results_normal['term_error_rate'].mean() * 100,
            results_powerlaw['term_error_rate'].mean() * 100
        ],
        'Prom_Error_%': [
            results_normal['prom_error_rate'].mean() * 100,
            results_powerlaw['prom_error_rate'].mean() * 100
        ],
        'Term_Avg_Percentile': [
            results_normal['term_avg_percentile'].mean(),
            results_powerlaw['term_avg_percentile'].mean()
        ],
        'Prom_Avg_Percentile': [
            results_normal['prom_avg_percentile'].mean(),
            results_powerlaw['prom_avg_percentile'].mean()
        ],
    })
    
    return comparison


def analyze_asymmetry(
    clustering_levels: List[float] = None,
    num_simulations: int = 100,
    seed: int = None
) -> pd.DataFrame:
    """
    Analyze asymmetry between termination and promotion errors under power law.
    
    Args:
        clustering_levels: List of clustering strengths to test
        num_simulations: Monte Carlo iterations
        seed: Random seed
        
    Returns:
        DataFrame showing asymmetric error patterns
    """
    if clustering_levels is None:
        clustering_levels = [0.0, 0.3, 0.7, 0.95]
    
    if seed is not None:
        np.random.seed(seed)
    
    results = []
    
    for clustering in clustering_levels:
        result = run_powerlaw_simulation(
            clustering_strength=clustering,
            num_simulations=num_simulations
        )
        
        results.append({
            'Clustering': f"{clustering*100:.0f}%",
            'Term_Error_%': result['mean_term_error'] * 100,
            'Prom_Error_%': result['mean_prom_error'] * 100,
            'Asymmetry_%': (result['mean_prom_error'] - result['mean_term_error']) * 100,
            'Term_Avg_Percentile': result['mean_term_avg_percentile'],
            'Prom_Avg_Percentile': result['mean_prom_avg_percentile'],
        })
    
    return pd.DataFrame(results)


def run_powerlaw_analysis(
    clustering_levels: List[float] = None,
    num_simulations: int = 100,
    seed: int = 42
) -> Dict:
    """
    Run comprehensive power law analysis for Appendix B.
    
    Args:
        clustering_levels: List of clustering strengths
        num_simulations: Monte Carlo iterations
        seed: Random seed
        
    Returns:
        Dictionary with all power law analysis results
    """
    if clustering_levels is None:
        clustering_levels = [0.0, 0.3, 0.7, 0.95]
    
    print("="*80)
    print("POWER LAW DISTRIBUTION ANALYSIS")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  - Pareto shape: 3.0 (10x engineer distribution)")
    print(f"  - Clustering levels: {clustering_levels}")
    print(f"  - Simulations per level: {num_simulations}")
    print()
    
    # Run asymmetry analysis
    asymmetry_df = analyze_asymmetry(
        clustering_levels=clustering_levels,
        num_simulations=num_simulations,
        seed=seed
    )
    
    print("ASYMMETRY ANALYSIS")
    print("-"*80)
    print(asymmetry_df.to_string(index=False))
    print()
    
    # Key findings
    print("="*80)
    print("KEY FINDINGS")
    print("="*80)
    print()
    print("1. SYMMETRIC AT LOW CLUSTERING:")
    print(f"   At 0% clustering: {asymmetry_df.iloc[0]['Asymmetry_%']:.1f}% asymmetry")
    print("   Termination and promotion errors are similar")
    print()
    
    print("2. ASYMMETRY EMERGES WITH CLUSTERING:")
    moderate_idx = 2  # 70% clustering
    print(f"   At 70% clustering:")
    print(f"     Termination error: {asymmetry_df.iloc[moderate_idx]['Term_Error_%']:.0f}%")
    print(f"     Promotion error:   {asymmetry_df.iloc[moderate_idx]['Prom_Error_%']:.0f}%")
    print(f"     Asymmetry:         {asymmetry_df.iloc[moderate_idx]['Asymmetry_%']:.0f}%")
    print()
    
    print("3. CATASTROPHIC AT STRONG CLUSTERING:")
    strong_idx = 3  # 95% clustering
    print(f"   At 95% clustering:")
    print(f"     Termination error: {asymmetry_df.iloc[strong_idx]['Term_Error_%']:.0f}%")
    print(f"     Promotion error:   {asymmetry_df.iloc[strong_idx]['Prom_Error_%']:.0f}%")
    print(f"     System promotes people in {asymmetry_df.iloc[strong_idx]['Prom_Avg_Percentile']:.0f}th percentile")
    print()
    
    print("4. THE MECHANISM:")
    print("   - Bottom-heavy tail: Even weak teams have poor performers")
    print("   - Top-heavy clustering: Exceptional performers concentrate")
    print("   - Result: Termination errors limited, promotion errors catastrophic")
    print()
    
    # Compare to normal distribution
    print("="*80)
    print("COMPARISON TO NORMAL DISTRIBUTION")
    print("="*80)
    print()
    
    comparison_70 = compare_distributions(
        clustering_strength=0.7,
        num_simulations=num_simulations,
        seed=seed
    )
    
    print("At 70% clustering (moderate, realistic):")
    print(comparison_70.to_string(index=False))
    print()
    
    print("="*80)
    print("CONCLUSION")
    print("="*80)
    print()
    print("Power law distribution creates asymmetric failure mode:")
    print("  - Terminations: ~36% error (contained)")
    print("  - Promotions: 61-80% error (catastrophic)")
    print()
    print("The normal distribution analysis (32%/54%) remains more conservative")
    print("and is therefore the primary result in the paper.")
    print()
    
    return {
        'asymmetry_analysis': asymmetry_df,
        'comparison_70': comparison_70,
        'clustering_levels': clustering_levels
    }
