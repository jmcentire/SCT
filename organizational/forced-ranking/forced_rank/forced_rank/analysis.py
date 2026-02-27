"""
Statistical analysis functions for forced ranking simulation results.

Provides comparison, sensitivity analysis, and reporting utilities.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from .simulation import Simulation


def compare_scenarios(
    num_simulations: int = 100,
    num_employees: int = 994,
    team_size: int = 7,
    seed: Optional[int] = None
) -> Dict:
    """
    Compare random vs biased assignment scenarios.
    
    Args:
        num_simulations: Number of Monte Carlo iterations
        num_employees: Total population
        team_size: Members per team
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary with comparison results and dataframes
    """
    from .simulation import run_simulation
    
    results_random = run_simulation(
        use_bias=False,
        num_simulations=num_simulations,
        num_employees=num_employees,
        team_size=team_size,
        seed=seed
    )
    
    results_biased = run_simulation(
        use_bias=True,
        num_simulations=num_simulations,
        num_employees=num_employees,
        team_size=team_size,
        seed=seed
    )
    
    # Create comparison table
    comparison_df = pd.DataFrame({
        'Scenario': ['Random', 'Biased'],
        'Term_Error_%': [
            results_random['mean_term_error'] * 100,
            results_biased['mean_term_error'] * 100
        ],
        'Prom_Error_%': [
            results_random['mean_prom_error'] * 100,
            results_biased['mean_prom_error'] * 100
        ],
        'Combined_Error_%': [
            results_random['mean_combined_error'] * 100,
            results_biased['mean_combined_error'] * 100
        ],
        'Term_False_Pos': [
            results_random['mean_term_false_pos'],
            results_biased['mean_term_false_pos']
        ],
        'Prom_False_Pos': [
            results_random['mean_prom_false_pos'],
            results_biased['mean_prom_false_pos']
        ],
    })
    
    return {
        'random': results_random,
        'biased': results_biased,
        'comparison_table': comparison_df,
        'error_increase': results_biased['mean_combined_error'] - results_random['mean_combined_error']
    }


def sensitivity_analysis(
    param: str = 'clustering_strength',
    values: Optional[List[float]] = None,
    num_simulations: int = 100,
    seed: Optional[int] = None
) -> pd.DataFrame:
    """
    Run sensitivity analysis by varying a parameter.
    
    Args:
        param: Parameter to vary ('clustering_strength', 'team_size', 'cutoff_percentile')
        values: List of values to test (None for default ranges)
        num_simulations: Monte Carlo iterations per configuration
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with results across parameter values
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Default value ranges
    if values is None:
        if param == 'clustering_strength':
            values = [0.0, 0.3, 0.5, 0.7, 0.9]
        elif param == 'team_size':
            values = [5, 6, 7, 8, 9, 10]
        elif param == 'cutoff_percentile':
            values = [0.10, 0.15, 0.20]
        else:
            raise ValueError(f"Unknown parameter: {param}")
    
    results = []
    
    for value in values:
        # Set up simulation with varied parameter
        kwargs = {
            'num_employees': 994,
            'team_size': 7,
            'distribution': 'normal',
            'clustering_strength': 0.7,
            'cutoff_percentile': 0.15
        }
        
        if param == 'clustering_strength':
            kwargs['clustering_strength'] = value
        elif param == 'team_size':
            kwargs['team_size'] = int(value)
            kwargs['num_employees'] = int(value) * 142  # Maintain 142 teams
        elif param == 'cutoff_percentile':
            kwargs['cutoff_percentile'] = value
        
        sim = Simulation(**kwargs)
        results_df = sim.run(num_simulations=num_simulations)
        
        # Calculate means
        mean_results = results_df.mean()
        
        results.append({
            'param_value': value,
            'term_error': mean_results['term_error_rate'],
            'prom_error': mean_results['prom_error_rate'],
            'combined_error': (mean_results['term_error_rate'] + mean_results['prom_error_rate']) / 2,
            'term_false_pos': mean_results['term_false_pos'],
            'prom_false_pos': mean_results['prom_false_pos'],
            'term_avg_percentile': mean_results['term_avg_percentile'],
            'prom_avg_percentile': mean_results['prom_avg_percentile'],
        })
    
    return pd.DataFrame(results)


def calculate_confidence_intervals(
    results_df: pd.DataFrame,
    confidence: float = 0.95
) -> Dict:
    """
    Calculate confidence intervals for error rates.
    
    Args:
        results_df: DataFrame from Simulation.run()
        confidence: Confidence level (default 0.95)
        
    Returns:
        Dictionary with means and confidence intervals
    """
    alpha = 1 - confidence
    
    metrics = ['term_error_rate', 'prom_error_rate']
    ci_results = {}
    
    for metric in metrics:
        values = results_df[metric].values
        mean = np.mean(values)
        se = np.std(values) / np.sqrt(len(values))
        
        # Use t-distribution for CI
        from scipy import stats
        t_crit = stats.t.ppf(1 - alpha/2, len(values) - 1)
        margin = t_crit * se
        
        ci_results[metric] = {
            'mean': mean,
            'se': se,
            'ci_lower': mean - margin,
            'ci_upper': mean + margin,
            'confidence': confidence
        }
    
    return ci_results


def generate_summary_statistics(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate comprehensive summary statistics for simulation results.
    
    Args:
        results_df: DataFrame from Simulation.run()
        
    Returns:
        DataFrame with summary statistics
    """
    metrics = [
        'term_error_rate', 'prom_error_rate',
        'term_false_pos', 'prom_false_pos',
        'term_avg_percentile', 'prom_avg_percentile'
    ]
    
    summary = []
    for metric in metrics:
        values = results_df[metric].values
        summary.append({
            'Metric': metric,
            'Mean': np.mean(values),
            'Median': np.median(values),
            'Std': np.std(values),
            'Min': np.min(values),
            'Max': np.max(values),
        })
    
    return pd.DataFrame(summary)


def test_statistical_significance(
    results_random: pd.DataFrame,
    results_biased: pd.DataFrame
) -> Dict:
    """
    Test whether biased assignment error rates are significantly different from random.
    
    Args:
        results_random: DataFrame from random assignment simulation
        results_biased: DataFrame from biased assignment simulation
        
    Returns:
        Dictionary with t-test results
    """
    from scipy import stats
    
    # Test termination error rates
    term_t, term_p = stats.ttest_ind(
        results_random['term_error_rate'],
        results_biased['term_error_rate']
    )
    
    # Test promotion error rates
    prom_t, prom_p = stats.ttest_ind(
        results_random['prom_error_rate'],
        results_biased['prom_error_rate']
    )
    
    return {
        'termination': {
            't_statistic': term_t,
            'p_value': term_p,
            'significant': term_p < 0.05,
            'mean_random': results_random['term_error_rate'].mean(),
            'mean_biased': results_biased['term_error_rate'].mean(),
        },
        'promotion': {
            't_statistic': prom_t,
            'p_value': prom_p,
            'significant': prom_p < 0.05,
            'mean_random': results_random['prom_error_rate'].mean(),
            'mean_biased': results_biased['prom_error_rate'].mean(),
        }
    }
