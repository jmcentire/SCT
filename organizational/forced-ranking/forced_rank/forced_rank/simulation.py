"""
Core simulation engine for forced ranking analysis.

This module implements agent-based simulation of forced distribution
performance management systems, demonstrating systematic classification
errors that emerge from evaluating global populations using local frames.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Literal
from scipy.stats import percentileofscore

class Simulation:
    """
    Agent-based simulation of forced ranking system.
    
    Models an organization with employees distributed across teams,
    applies forced ranking within teams, and measures classification
    errors against ground truth.
    
    Attributes:
        num_employees: Total number of employees
        team_size: Number of employees per team
        num_teams: Calculated number of teams
        distribution: Talent distribution ('normal' or 'powerlaw')
        clustering_strength: Team quality variance (0.0=random, 1.0=perfect)
        cutoff_percentile: What fraction to fire/promote (default 0.15)
    """
    
    def __init__(
        self,
        num_employees: int = 994,
        team_size: int = 7,
        distribution: Literal['normal', 'powerlaw'] = 'normal',
        clustering_strength: float = 0.7,
        cutoff_percentile: float = 0.15,
        pareto_shape: float = 3.0
    ):
        """
        Initialize simulation parameters.
        
        Args:
            num_employees: Total population (default 994)
            team_size: Members per team (default 7)
            distribution: 'normal' or 'powerlaw' (default 'normal')
            clustering_strength: 0.0 (random) to 1.0 (perfect clustering)
            cutoff_percentile: Fraction to fire/promote (default 0.15)
            pareto_shape: Shape parameter for power law (default 3.0)
        """
        self.num_employees = num_employees
        self.team_size = team_size
        self.num_teams = num_employees // team_size
        self.distribution = distribution
        self.clustering_strength = clustering_strength
        self.cutoff_percentile = cutoff_percentile
        self.pareto_shape = pareto_shape
        
        # Validate parameters
        if num_employees < 100:
            raise ValueError("num_employees must be >= 100")
        if team_size < 3 or team_size > 20:
            raise ValueError("team_size must be between 3 and 20")
        if clustering_strength < 0.0 or clustering_strength > 1.0:
            raise ValueError("clustering_strength must be between 0.0 and 1.0")
        if cutoff_percentile < 0.05 or cutoff_percentile > 0.30:
            raise ValueError("cutoff_percentile must be between 0.05 and 0.30")
    
    def generate_talents_and_teams(self) -> tuple:
        """
        Generate talent distribution and team assignments.
        
        For biased assignment, uses hierarchical model:
        - Team means ~ N(0, σ_team)
        - Members ~ N(team_mean, σ_within)
        - Where σ_team² + σ_within² = 1
        
        Returns:
            Tuple of (talents, team_ids)
        """
        if self.clustering_strength == 0.0:
            # Random assignment
            if self.distribution == 'normal':
                talents = np.random.normal(0, 1, self.num_employees)
            elif self.distribution == 'powerlaw':
                raw_scores = np.random.pareto(self.pareto_shape, self.num_employees)
                talents = 100 * (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min())
            
            team_ids = np.arange(self.num_employees) // self.team_size
            np.random.shuffle(team_ids)
            
            return talents, team_ids
        
        else:
            # Biased assignment using hierarchical model
            sigma_team = self.clustering_strength
            sigma_within = np.sqrt(max(0.0, 1.0 - sigma_team**2))
            
            # Generate team means
            team_means = np.random.normal(0, sigma_team, self.num_teams)
            
            # Assign employees to teams (in order)
            team_ids = np.repeat(np.arange(self.num_teams), self.team_size)[:self.num_employees]
            
            # Generate talents from team means
            talents = np.zeros(self.num_employees)
            for i in range(self.num_employees):
                team_mean = team_means[team_ids[i]]
                talents[i] = np.random.normal(team_mean, sigma_within)
            
            return talents, team_ids
    
    def run_single(self) -> Dict:
        """
        Run a single simulation iteration.
        
        Returns:
            Dictionary with classification results and error rates
        """
        # Generate talents and assign teams
        talents, team_ids = self.generate_talents_and_teams()
        
        # Create dataframe
        df = pd.DataFrame({
            'talent': talents,
            'team_id': team_ids,
            'global_percentile': [percentileofscore(talents, t) for t in talents]
        })
        
        # Calculate ground truth thresholds
        bottom_threshold = np.percentile(talents, self.cutoff_percentile * 100)
        top_threshold = np.percentile(talents, (1 - self.cutoff_percentile) * 100)
        
        df['true_bottom'] = df['talent'] <= bottom_threshold
        df['true_top'] = df['talent'] >= top_threshold
        
        # Apply forced ranking within teams
        terminations = df.loc[df.groupby('team_id')['talent'].idxmin()]
        promotions = df.loc[df.groupby('team_id')['talent'].idxmax()]
        
        # Calculate classification metrics
        
        # Terminations
        term_correct = int((terminations['true_bottom']).sum())
        term_false_pos = int((~terminations['true_bottom']).sum())
        term_false_neg = len(set(df[df['true_bottom']].index) - set(terminations.index))
        term_error_rate = term_false_pos / len(terminations)
        term_avg_percentile = float(terminations['global_percentile'].mean())
        
        # Promotions
        prom_correct = int((promotions['true_top']).sum())
        prom_false_pos = int((~promotions['true_top']).sum())
        prom_false_neg = len(set(df[df['true_top']].index) - set(promotions.index))
        prom_error_rate = prom_false_pos / len(promotions)
        prom_avg_percentile = float(promotions['global_percentile'].mean())
        
        return {
            'term_total': len(terminations),
            'term_correct': term_correct,
            'term_false_pos': term_false_pos,
            'term_false_neg': term_false_neg,
            'term_error_rate': term_error_rate,
            'term_avg_percentile': term_avg_percentile,
            'prom_total': len(promotions),
            'prom_correct': prom_correct,
            'prom_false_pos': prom_false_pos,
            'prom_false_neg': prom_false_neg,
            'prom_error_rate': prom_error_rate,
            'prom_avg_percentile': prom_avg_percentile,
        }
    
    def run(self, num_simulations: int = 100, seed: Optional[int] = None) -> pd.DataFrame:
        """
        Run multiple simulation iterations and aggregate results.
        
        Args:
            num_simulations: Number of Monte Carlo iterations
            seed: Random seed for reproducibility
            
        Returns:
            DataFrame with results from all simulations
        """
        if seed is not None:
            np.random.seed(seed)
        
        results = []
        for i in range(num_simulations):
            results.append(self.run_single())
        
        return pd.DataFrame(results)

def run_simulation(
    use_bias: bool = False,
    num_simulations: int = 100,
    num_employees: int = 994,
    team_size: int = 7,
    seed: Optional[int] = None
) -> Dict:
    """
    Convenience function to run simulation with standard parameters.
    
    This matches the methodology from the paper:
    - Random assignment (use_bias=False): clustering_strength = 0.0
    - Biased assignment (use_bias=True): clustering_strength = 0.7
    
    Args:
	use_bias: Use realistic team clustering (default False)
	num_simulations: Number of Monte Carlo iterations (default 100)
	num_employees: Total population (default 994)
	team_size: Members per team (default 7)
	seed: Random seed for reproducibility
	
    Returns:
	Dictionary with mean results and full DataFrame
    """
    clustering = 0.7 if use_bias else 0.0
    
    sim = Simulation(
	num_employees=num_employees,
	team_size=team_size,
	distribution='normal',
	clustering_strength=clustering,
	cutoff_percentile=0.15
    )
    
    results_df = sim.run(num_simulations=num_simulations, seed=seed)
    
    # Calculate means
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


def print_results(results: Dict, label: str = "Results"):
    """
    Print simulation results in readable format.
    
    Args:
	results: Dictionary from run_simulation()
	label: Label for this result set
    """
    print("="*80)
    print(f"{label}")
    print("="*80)
    print(f"\nTERMINATIONS:")
    print(f"  Correct:            {results['mean_term_correct']:.0f} ({(1-results['mean_term_error'])*100:.0f}%)")
    print(f"  False Positives:    {results['mean_term_false_pos']:.0f} ({results['mean_term_error']*100:.0f}%)")
    print(f"  Avg Percentile:     {results['mean_term_avg_percentile']:.1f}")
    
    print(f"\nPROMOTIONS:")
    print(f"  Correct:            {results['mean_prom_correct']:.0f} ({(1-results['mean_prom_error'])*100:.0f}%)")
    print(f"  False Positives:    {results['mean_prom_false_pos']:.0f} ({results['mean_prom_error']*100:.0f}%)")
    print(f"  Avg Percentile:     {results['mean_prom_avg_percentile']:.1f}")
    
    print(f"\nCOMBINED ERROR RATE: {results['mean_combined_error']*100:.0f}%")
    print()
