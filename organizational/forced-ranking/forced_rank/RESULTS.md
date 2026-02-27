# Forced Ranking Simulation

Agent-based simulation demonstrating systematic classification errors in forced distribution performance management systems.

## Overview

This repository contains the simulation code supporting the empirical analysis in *The Cage and the Mirror: Engineering Capability Within Organizational Constraints* (McEntire, 2025). The simulation models a technology organization with ~1,000 engineers distributed across teams and demonstrates that forced ranking systems—which evaluate employees relative to teammates and apply fixed distribution requirements—produce error rates of 32-54% even under idealized conditions.

## Key Findings

- **Random Assignment (Best Case):** 32% error rate
- **Realistic Conditions (Team Quality Variance):** 54% error rate  
- **Power Law Distribution + Strong Clustering:** Up to 80% error rate on promotions

These errors are not implementation failures—they are mathematical necessities that emerge from evaluating a global population using local frames.

## What is Forced Ranking?

Forced ranking (also called "stack ranking" or "rank and yank") mandates that managers:
1. Evaluate employees relative to their teammates
2. Apply predetermined performance categories (e.g., top 15%, middle 70%, bottom 15%)
3. Fire the bottom performers and promote the top performers from each team

The system assumes that small teams are representative samples of the company's global talent distribution. This assumption is false, and the consequences are severe.

## Installation

### Requirements

- Python 3.8+
- NumPy
- Pandas
- Matplotlib
- SciPy
- Seaborn (optional, for visualizations)

### Setup

```bash
git clone https://github.com/yourusername/forced_rank.git
cd forced_rank
pip install -r requirements.txt
```

## Quick Start

### Run Basic Simulation

```python
from forced_rank import run_simulation, print_results

# Random assignment (best case)
results_random = run_simulation(use_bias=False, num_simulations=100)
print_results(results_random, "Random Assignment")

# Biased assignment (realistic)
results_biased = run_simulation(use_bias=True, num_simulations=100)
print_results(results_biased, "Biased Assignment")
```

### Run Full Analysis

```bash
python run_analysis.py
```

This generates:
- Summary statistics for random and biased scenarios
- Comparison tables
- Visualization plots (saved to `outputs/`)

## Simulation Methodology

### Setup

- **Population:** 994 employees
- **Team Structure:** 142 teams of 7 members each
- **Talent Distribution:** Standard normal N(0,1) by default
- **Evaluation Method:** Forced ranking within teams (bottom ~15% fired, top ~15% promoted)

### Two Scenarios

**Random Assignment (Baseline):**
- Employees randomly assigned to teams
- No hiring bias, managerial quality differences, or favoritism
- Represents best-case conditions for forced ranking

**Biased Assignment (Realistic):**
- Team quality varies using hierarchical normal model:
  - Team means drawn from N(0, 0.7)
  - Team members drawn from N(team_mean, 0.714)
  - Overall maintains N(0,1) distribution
- Simulates differential managerial capability (strong managers attract/retain talent)

### Error Measurement

We compare forced ranking outcomes to ground truth:

**False Positive:** Employee labeled for termination/promotion but NOT in true global bottom/top 15%

**False Negative:** Employee in true global bottom/top 15% but NOT labeled

**Error Rate:** False Positives / Total Labeled

## Repository Structure

```
forced_rank/
├── README.md
├── LICENSE
├── requirements.txt
├── setup.py
├── forced_rank/
│   ├── __init__.py
│   ├── simulation.py      # Core simulation engine
│   ├── analysis.py        # Statistical analysis functions
│   ├── visualization.py   # Plotting utilities
│   └── powerlaw.py        # Power law distribution variants
├── scripts/
│   ├── run_analysis.py    # Main analysis script
│   ├── run_sensitivity.py # Parameter sensitivity tests
│   └── run_powerlaw.py    # Power law distribution analysis
├── tests/
│   ├── test_simulation.py
│   └── test_analysis.py
├── outputs/               # Generated plots and results
└── data/                 # Simulation results (generated)
```

## Usage Examples

### Custom Parameters

```python
from forced_rank import Simulation

# Create custom simulation
sim = Simulation(
    num_employees=1000,
    team_size=10,
    num_simulations=100,
    distribution='normal',  # or 'powerlaw'
    clustering_strength=0.7
)

# Run and analyze
results = sim.run()
sim.plot_results(save_path='outputs/custom_results.png')
```

### Sensitivity Analysis

```python
from forced_rank import sensitivity_analysis

# Test multiple clustering levels
results = sensitivity_analysis(
    clustering_range=[0.0, 0.3, 0.5, 0.7, 0.9],
    num_simulations=100
)

# Results show error rates across clustering strengths
```

### Power Law Analysis

```python
from forced_rank.powerlaw import run_powerlaw_analysis

# Compare Normal vs Power Law distributions
results = run_powerlaw_analysis(
    clustering_levels=[0.0, 0.3, 0.7, 0.95],
    num_simulations=100
)

# Demonstrates asymmetric error patterns (terminations vs promotions)
```

## Key Results

### Random Assignment (32% Error)

| Metric | Terminations | Promotions |
|--------|--------------|------------|
| Total Labeled | 142 | 142 |
| Correct | 97 (68%) | 96 (68%) |
| False Positives | 45 (32%) | 46 (32%) |
| False Negatives | 52 | 53 |

### Biased Assignment (54% Error)

| Metric | Terminations | Promotions |
|--------|--------------|------------|
| Total Labeled | 142 | 142 |
| Correct | 65 (46%) | 66 (46%) |
| False Positives | 77 (54%) | 76 (54%) |
| False Negatives | 84 | 83 |

### Power Law + Strong Clustering (80% Promotion Error)

Under power law talent distributions with strong clustering (95%), promotion error rates reach 80%—the system promotes individuals in the 56th percentile while claiming to identify the top 15%.

## Interpretation

**Why 32% error with random assignment?**

Even with no bias, small teams have high composition variance. By chance, some teams draw stronger members, others draw weaker members. When you fire the "worst person on a strong team," you often fire someone who is globally above average. When you promote the "best person on a weak team," you often promote someone who is globally below average.

**Why 54% error with realistic clustering?**

Strong managers attract and retain talent, creating team quality variance. This amplifies the frame problem:
- Strong teams get punished (their "bottom performer" is globally competent)
- Weak teams get rewarded (their "top performer" is globally mediocre)
- The system cannot see this because it evaluates locally

**Why worse than random?**

Random firing/promotion (coin flips) produces 50% accuracy by definition. Forced ranking produces 46% accuracy (worse than random) because the errors are **systematically biased** against people on strong teams and in favor of people on weak teams.

## Conservative Assumptions

All simulation assumptions favor forced ranking:

1. **Perfect talent visibility:** We assume managers can perfectly assess talent (no measurement error)
2. **Normal distribution:** We use bell curve talent (HR's assumption) not power law (empirical reality)
3. **Static teams:** No turnover, politics, or gaming
4. **Single period:** No compounding multi-year effects
5. **Moderate clustering:** σ_team = 0.7 represents realistic but not extreme team variance

Real-world implementations perform **worse** than these simulations.

## Extensions

The repository includes extensions for:

- **Power Law Distributions:** Pareto talent distributions with varying shape parameters
- **Multi-Period Dynamics:** Simulating brain drain and adverse selection over time
- **Team Size Sensitivity:** Testing with teams of 5, 6, 8, 9, 10 members
- **Alternative Cutoffs:** Testing 10% and 20% thresholds instead of 15%
- **Calibration Mechanisms:** Testing whether cross-team calibration reduces errors (spoiler: it doesn't)

## Citation

If you use this simulation in research or analysis, please cite:

```
McEntire, J. (2025). The Cage and the Mirror: Engineering Capability Within 
Organizational Constraints. Self-published.

McEntire, J. (2025). Forced Ranking Simulation [Software]. 
GitHub: https://github.com/yourusername/forced_rank
```

## Contributing

Contributions welcome! Areas of interest:

- Alternative talent distributions (Beta, Exponential, Bimodal)
- Real-world validation data
- Additional visualization methods
- Performance optimizations
- Extensions to other evaluation systems (360 reviews, OKRs, etc.)

Please open an issue before submitting major changes.

## License

MIT License - see LICENSE file for details

## Related Work

### Academic Research

- O'Boyle, E., & Aguinis, H. (2012). The best and the rest: Revisiting the norm of normality of individual performance. *Personnel Psychology*, 65(1), 79-119.
- Scullen, S. E., Bergey, P. K., & Aiman-Smith, L. (2005). Forced distribution rating systems and the improvement of workforce potential. *Personnel Psychology*, 58(1), 1-32.
- Schleicher, D. J., Bachiochi, P. D., & Palladino, E. A. (2018). Putting the system into performance management systems. *Industrial and Organizational Psychology*, 11(1), 111-145.

### Industry Coverage

- Eichenwald, K. (2012). Microsoft's lost decade. *Vanity Fair*.
- Competitive edits: How Microsoft lost its mojo. *Harvard Business Review*.

### Book Website

For complete documentation, datasets, and analysis:
- **Website:** [cageandmirror.com](https://cageandmirror.com)
- **Book:** Available January 2025

## Contact

Questions, comments, or collaboration inquiries:
- **Author:** Jeremy McEntire
- **Email:** [your-email]
- **Website:** [cageandmirror.com](https://cageandmirror.com)
- **LinkedIn:** [your-linkedin]

## Acknowledgments

This simulation was developed as part of research for *The Cage and the Mirror*. Thanks to early readers who provided feedback on methodology and interpretation.

---

**Disclaimer:** This simulation is for research and educational purposes. Organizations should not use forced ranking systems regardless of implementation quality. The errors demonstrated here are structural, not procedural.
