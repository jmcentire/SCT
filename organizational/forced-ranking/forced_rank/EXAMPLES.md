# Usage Examples

## Quick Start

### Run Paper Results

```python
from forced_rank import run_simulation, print_results

# Random assignment (32% error)
results_random = run_simulation(use_bias=False, num_simulations=100, seed=42)
print_results(results_random, "Random Assignment")

# Biased assignment (54% error)
results_biased = run_simulation(use_bias=True, num_simulations=100, seed=42)
print_results(results_biased, "Biased Assignment")
```

### Compare Scenarios

```python
from forced_rank.analysis import compare_scenarios

results = compare_scenarios(num_simulations=100, seed=42)

# View comparison table
print(results['comparison_table'])

# Access individual results
print(f"Random error: {results['random']['mean_combined_error']*100:.0f}%")
print(f"Biased error: {results['biased']['mean_combined_error']*100:.0f}%")
print(f"Increase: {results['error_increase']*100:.0f}%")
```

## Advanced Usage

### Custom Configuration

```python
from forced_rank import Simulation

# Create custom simulation
sim = Simulation(
    num_employees=1000,
    team_size=10,
    distribution='normal',
    clustering_strength=0.5,
    cutoff_percentile=0.10
)

# Run simulations
results_df = sim.run(num_simulations=100, seed=42)

# Analyze results
mean_error = results_df['term_error_rate'].mean()
print(f"Mean termination error: {mean_error*100:.1f}%")

# Access detailed results
print(results_df.describe())
```

### Sensitivity Analysis

```python
from forced_rank.analysis import sensitivity_analysis

# Test different clustering levels
results = sensitivity_analysis(
    param='clustering_strength',
    values=[0.0, 0.3, 0.5, 0.7, 0.9],
    num_simulations=100,
    seed=42
)

print(results)
# Shows error rates across clustering strengths
```

### Test Different Team Sizes

```python
from forced_rank.analysis import sensitivity_analysis

# Test team sizes from 5 to 10
results = sensitivity_analysis(
    param='team_size',
    values=[5, 6, 7, 8, 9, 10],
    num_simulations=100,
    seed=42
)

print(results)
```

### Test Different Cutoff Percentiles

```python
from forced_rank.analysis import sensitivity_analysis

# Test 10%, 15%, 20% cutoffs
results = sensitivity_analysis(
    param='cutoff_percentile',
    values=[0.10, 0.15, 0.20],
    num_simulations=100,
    seed=42
)

print(results)
```

## Power Law Distribution

### Basic Power Law Simulation

```python
from forced_rank.powerlaw import run_powerlaw_simulation

# Run with strong clustering
results = run_powerlaw_simulation(
    clustering_strength=0.95,
    num_simulations=100,
    seed=42
)

print(f"Termination error: {results['mean_term_error']*100:.0f}%")
print(f"Promotion error: {results['mean_prom_error']*100:.0f}%")
```

### Compare Normal vs Power Law

```python
from forced_rank.powerlaw import compare_distributions

# Compare at 70% clustering
comparison = compare_distributions(
    clustering_strength=0.7,
    num_simulations=100,
    seed=42
)

print(comparison)
```

### Analyze Asymmetry

```python
from forced_rank.powerlaw import analyze_asymmetry

# Test multiple clustering levels
results = analyze_asymmetry(
    clustering_levels=[0.0, 0.3, 0.7, 0.95],
    num_simulations=100,
    seed=42
)

print(results)
# Shows how promotion errors diverge from termination errors
```

### Full Power Law Analysis (Appendix B)

```python
from forced_rank.powerlaw import run_powerlaw_analysis

# Run comprehensive analysis
results = run_powerlaw_analysis(
    clustering_levels=[0.0, 0.3, 0.7, 0.95],
    num_simulations=100,
    seed=42
)

# Access results
print(results['asymmetry_analysis'])
print(results['comparison_70'])
```

## Statistical Analysis

### Calculate Confidence Intervals

```python
from forced_rank import Simulation
from forced_rank.analysis import calculate_confidence_intervals

# Run simulation
sim = Simulation(clustering_strength=0.7)
results_df = sim.run(num_simulations=100, seed=42)

# Calculate 95% confidence intervals
ci = calculate_confidence_intervals(results_df, confidence=0.95)

print(f"Termination error: {ci['term_error_rate']['mean']*100:.1f}% "
      f"[{ci['term_error_rate']['ci_lower']*100:.1f}%, "
      f"{ci['term_error_rate']['ci_upper']*100:.1f}%]")
```

### Generate Summary Statistics

```python
from forced_rank import Simulation
from forced_rank.analysis import generate_summary_statistics

# Run simulation
sim = Simulation(clustering_strength=0.7)
results_df = sim.run(num_simulations=100, seed=42)

# Get summary stats
summary = generate_summary_statistics(results_df)
print(summary)
```

### Test Statistical Significance

```python
from forced_rank import Simulation
from forced_rank.analysis import test_statistical_significance

# Run both scenarios
sim_random = Simulation(clustering_strength=0.0)
results_random = sim_random.run(num_simulations=100, seed=42)

sim_biased = Simulation(clustering_strength=0.7)
results_biased = sim_biased.run(num_simulations=100, seed=42)

# Test significance
sig_test = test_statistical_significance(results_random, results_biased)

print(f"Termination: p = {sig_test['termination']['p_value']:.4f}")
print(f"Promotion: p = {sig_test['promotion']['p_value']:.4f}")
print(f"Significant: {sig_test['termination']['significant']}")
```

## Command Line Interface

### Basic Usage

```bash
# Run standard analysis
python scripts/run_analysis.py

# Run sensitivity analysis
python scripts/run_sensitivity.py

# Run power law analysis
python scripts/run_powerlaw.py
```

### With Custom Parameters

```bash
# Run with custom simulation count
python scripts/run_analysis.py --simulations 1000

# Run sensitivity with custom range
python scripts/run_sensitivity.py --param clustering_strength --values 0.0 0.5 1.0
```

## Reproducible Research

### Set Random Seed

```python
from forced_rank import run_simulation

# Always use seed for reproducibility
results = run_simulation(use_bias=True, num_simulations=100, seed=42)

# Results will be identical across runs
```

### Save Results for Later

```python
from forced_rank import Simulation
import pandas as pd

# Run simulation
sim = Simulation(clustering_strength=0.7)
results_df = sim.run(num_simulations=100, seed=42)

# Save to CSV
results_df.to_csv('my_results.csv', index=False)

# Load later
loaded_df = pd.read_csv('my_results.csv')
```

## Batch Processing

### Test Multiple Configurations

```python
from forced_rank import Simulation
import pandas as pd

configurations = [
    {'clustering': 0.0, 'team_size': 7},
    {'clustering': 0.3, 'team_size': 7},
    {'clustering': 0.7, 'team_size': 7},
    {'clustering': 0.7, 'team_size': 5},
    {'clustering': 0.7, 'team_size': 10},
]

all_results = []

for i, config in enumerate(configurations):
    print(f"Running configuration {i+1}/{len(configurations)}...")
    
    sim = Simulation(
        num_employees=config['team_size'] * 142,
        team_size=config['team_size'],
        clustering_strength=config['clustering']
    )
    
    results_df = sim.run(num_simulations=100, seed=42)
    means = results_df.mean()
    
    all_results.append({
        'config': str(config),
        'term_error': means['term_error_rate'],
        'prom_error': means['prom_error_rate'],
    })

# Convert to dataframe
batch_results = pd.DataFrame(all_results)
print(batch_results)
```

## Integration with Other Tools

### Export for Visualization

```python
from forced_rank import Simulation
import matplotlib.pyplot as plt

# Run simulation
sim = Simulation(clustering_strength=0.7)
results_df = sim.run(num_simulations=100, seed=42)

# Plot distribution of error rates
plt.figure(figsize=(10, 6))
plt.hist(results_df['term_error_rate'], bins=20, alpha=0.5, label='Terminations')
plt.hist(results_df['prom_error_rate'], bins=20, alpha=0.5, label='Promotions')
plt.xlabel('Error Rate')
plt.ylabel('Frequency')
plt.title('Distribution of Error Rates (100 simulations)')
plt.legend()
plt.savefig('error_distribution.png')
```

### Integration with Jupyter Notebooks

```python
# In Jupyter notebook
from forced_rank import run_simulation, print_results
import pandas as pd

# Run analysis
results = run_simulation(use_bias=True, num_simulations=100, seed=42)

# Display in notebook
display(results['results_df'].head())
display(results['results_df'].describe())

# Create interactive plots
import plotly.express as px
fig = px.histogram(results['results_df'], x='term_error_rate', 
                   title='Termination Error Rate Distribution')
fig.show()
```

## Troubleshooting

### Memory Issues with Large Simulations

```python
# Instead of running 10,000 simulations at once
# Run in batches and aggregate

from forced_rank import Simulation
import pandas as pd

batch_size = 100
total_sims = 10000
num_batches = total_sims // batch_size

all_means = []

for batch in range(num_batches):
    sim = Simulation(clustering_strength=0.7)
    results_df = sim.run(num_simulations=batch_size, seed=42+batch)
    all_means.append(results_df.mean())
    
# Aggregate across batches
final_mean = pd.DataFrame(all_means).mean()
print(final_mean)
```

### Validation Errors

```python
from forced_rank import Simulation

try:
    # This will fail validation
    sim = Simulation(num_employees=50)  # Too small
except ValueError as e:
    print(f"Validation error: {e}")
    # Create valid simulation instead
    sim = Simulation(num_employees=100)
```

## Performance Optimization

### Vectorized Operations

The simulation already uses vectorized NumPy operations internally. For even better performance:

```python
# Use multiprocessing for independent simulations
from multiprocessing import Pool
from forced_rank import Simulation

def run_single(seed):
    sim = Simulation(clustering_strength=0.7)
    results_df = sim.run(num_simulations=1, seed=seed)
    return results_df.iloc[0].to_dict()

# Run 100 simulations in parallel
with Pool(processes=4) as pool:
    results = pool.map(run_single, range(100))

# Convert to DataFrame
import pandas as pd
results_df = pd.DataFrame(results)
```

## Complete Analysis Pipeline

```python
#!/usr/bin/env python3
"""Complete analysis pipeline."""

from forced_rank import run_simulation, print_results
from forced_rank.analysis import compare_scenarios, sensitivity_analysis
from forced_rank.powerlaw import run_powerlaw_analysis

# 1. Run paper results
print("="*80)
print("PAPER RESULTS")
print("="*80)

comparison = compare_scenarios(num_simulations=100, seed=42)
print(comparison['comparison_table'])

# 2. Sensitivity analysis
print("\n" + "="*80)
print("SENSITIVITY ANALYSIS")
print("="*80)

clustering_sens = sensitivity_analysis(
    param='clustering_strength',
    values=[0.0, 0.3, 0.5, 0.7, 0.9],
    num_simulations=100,
    seed=42
)
print(clustering_sens)

# 3. Power law analysis
print("\n" + "="*80)
print("POWER LAW ANALYSIS")
print("="*80)

powerlaw_results = run_powerlaw_analysis(
    clustering_levels=[0.0, 0.3, 0.7, 0.95],
    num_simulations=100,
    seed=42
)

# 4. Save everything
comparison['comparison_table'].to_csv('data/comparison.csv', index=False)
clustering_sens.to_csv('data/sensitivity.csv', index=False)
powerlaw_results['asymmetry_analysis'].to_csv('data/powerlaw.csv', index=False)

print("\n✓ Analysis complete. Results saved to data/")
```
