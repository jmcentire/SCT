# Simulation Parameters Quick Reference

## Default Configuration

**Paper Standard:**
```python
num_employees = 994          # Total population
team_size = 7                # Members per team  
num_teams = 142              # Calculated: 994 / 7
distribution = 'normal'      # N(0,1) bell curve
clustering_strength = 0.7    # For biased scenario
cutoff_percentile = 0.15     # Fire/promote 15%
num_simulations = 100        # Monte Carlo iterations
```

## Core Parameters

### `num_employees`
- **Default:** 994
- **Range:** 100-10,000
- **Description:** Total population size
- **Note:** Should be divisible by `team_size`

### `team_size`
- **Default:** 7
- **Range:** 3-20
- **Description:** Number of members per team
- **Note:** Smaller teams increase variance, larger teams reduce it

### `num_teams`
- **Calculated:** `num_employees / team_size`
- **Paper value:** 142
- **Description:** Total number of teams

### `use_bias`
- **Default:** False
- **Options:** True/False
- **Description:** 
  - `False`: Random assignment (σ_team = 0.0)
  - `True`: Biased assignment (σ_team = 0.7)

### `clustering_strength`
- **Default:** 0.7 (for biased)
- **Range:** 0.0-1.0
- **Description:** Team quality variance
  - `0.0`: Random assignment (no clustering)
  - `0.3`: Weak clustering
  - `0.7`: Moderate clustering (paper default)
  - `0.9`: Strong clustering
  - `1.0`: Perfect sorting by talent

### `distribution`
- **Default:** 'normal'
- **Options:** 'normal', 'powerlaw'
- **Description:** Talent distribution type
  - `'normal'`: Standard normal N(0,1) - paper default
  - `'powerlaw'`: Pareto distribution - Appendix B

### `cutoff_percentile`
- **Default:** 0.15
- **Range:** 0.05-0.30
- **Description:** What fraction to fire/promote
  - `0.10`: Fire/promote bottom/top 10%
  - `0.15`: Fire/promote bottom/top 15% (paper default)
  - `0.20`: Fire/promote bottom/top 20%

### `pareto_shape`
- **Default:** 3.0
- **Range:** 1.5-5.0
- **Description:** Shape parameter for power law distribution
- **Only applies when:** `distribution='powerlaw'`
- **Interpretation:**
  - Lower values (1.5-2.5): More extreme inequality (heavier tail)
  - `3.0`: Standard "10x engineer" distribution
  - Higher values (3.5-5.0): Less extreme inequality

### `num_simulations`
- **Default:** 100
- **Range:** 1-10,000
- **Description:** Number of Monte Carlo iterations
- **Note:** More iterations = more accurate means, but slower runtime

### `seed`
- **Default:** None (random)
- **Type:** Integer or None
- **Description:** Random seed for reproducibility
- **Paper value:** 42 (for reproducible results)

## Variance Decomposition (Biased Assignment)

The biased assignment implements hierarchical normal model:

```
Team means ~ N(0, σ_team)
Team members ~ N(team_mean, σ_within)

Where: σ_team² + σ_within² = 1.0
```

**For σ_team = 0.7:**
- σ_within = 0.714
- Calculation: 0.7² + 0.714² ≈ 1.0
- Maintains overall N(0,1) distribution

## Example Configurations

### Paper Results (Random Assignment)
```python
Simulation(
    num_employees=994,
    team_size=7,
    distribution='normal',
    clustering_strength=0.0,  # Random
    cutoff_percentile=0.15
).run(num_simulations=100, seed=42)
```
**Expected result:** ~32% error

### Paper Results (Biased Assignment)
```python
Simulation(
    num_employees=994,
    team_size=7,
    distribution='normal',
    clustering_strength=0.7,  # Moderate clustering
    cutoff_percentile=0.15
).run(num_simulations=100, seed=42)
```
**Expected result:** ~54% error

### Power Law (Appendix B)
```python
Simulation(
    num_employees=994,
    team_size=7,
    distribution='powerlaw',
    clustering_strength=0.95,  # Strong clustering
    cutoff_percentile=0.15,
    pareto_shape=3.0
).run(num_simulations=100, seed=42)
```
**Expected result:** ~80% promotion error

### Sensitivity Test (Team Size)
```python
for size in [5, 6, 7, 8, 9, 10]:
    Simulation(
        num_employees=size * 142,  # Maintain 142 teams
        team_size=size,
        distribution='normal',
        clustering_strength=0.7,
        cutoff_percentile=0.15
    ).run(num_simulations=100)
```

## Parameter Relationships

### Error Rate Drivers

**Primary driver:** `clustering_strength`
- 0.0 (random) → ~32% error
- 0.3 (weak) → ~36% error
- 0.7 (moderate) → ~54% error
- 0.9 (strong) → ~58% error

**Secondary driver:** `team_size`
- Smaller teams → higher variance → higher error
- Larger teams → lower variance → lower error
- Effect is modest (~5% change across 5-10 size range)

**Tertiary driver:** `cutoff_percentile`
- Narrower cutoffs (5-10%) → slightly higher error
- Wider cutoffs (20-30%) → slightly lower error
- Effect is small (~3% change)

### Distribution Effects

**Normal distribution:**
- Symmetric errors (terminations ≈ promotions)
- Errors increase with clustering

**Power law distribution:**
- Asymmetric errors (promotions >> terminations)
- Errors increase dramatically with strong clustering
- Shows "trapped talent" phenomenon

## Performance Notes

**Runtime scales with:**
- `num_employees × num_simulations`
- Typical: 994 employees × 100 simulations ≈ 2-3 seconds
- Large: 10,000 employees × 1,000 simulations ≈ 5-10 minutes

**Memory scales with:**
- `num_simulations` (stores all results)
- Typical: ~1 MB for 100 simulations
- Large: ~100 MB for 10,000 simulations

## Validation

All parameters are validated on initialization:
- `num_employees >= 100`
- `3 <= team_size <= 20`
- `0.0 <= clustering_strength <= 1.0`
- `0.05 <= cutoff_percentile <= 0.30`
- `num_simulations >= 1`

Invalid parameters raise `ValueError` with descriptive message.
