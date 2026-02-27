# Contributing to Forced Rank Simulation

Thank you for your interest in contributing! This project welcomes contributions from researchers, practitioners, and anyone interested in understanding organizational evaluation systems.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally
3. **Create a branch** for your changes
4. **Make your changes** with clear commit messages
5. **Test your changes** (run existing tests, add new ones)
6. **Submit a pull request**

## Development Setup

```bash
# Clone your fork
git clone https://github.com/yourusername/forced_rank.git
cd forced_rank

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/
```

## Areas for Contribution

### High Priority

- **Real-world validation data:** Case studies, anonymized company data
- **Alternative distributions:** Beta, Exponential, Bimodal talent models
- **Multi-period dynamics:** Brain drain, adverse selection simulations
- **Calibration mechanisms:** Testing whether cross-team calibration helps

### Medium Priority

- **Performance optimizations:** Vectorization, parallel processing
- **Additional visualizations:** Interactive plots, dashboards
- **Documentation improvements:** Examples, tutorials, explanations
- **Test coverage:** Edge cases, integration tests

### Nice to Have

- **Web interface:** Browser-based simulation runner
- **Jupyter notebooks:** Interactive analysis examples
- **Video explanations:** Visual walkthroughs of key concepts
- **Translations:** Documentation in other languages

## Code Style

We follow standard Python conventions:

- **PEP 8** for code style
- **Black** for formatting (line length: 100)
- **Type hints** for function signatures
- **Docstrings** for public functions (Google style)

```python
def run_simulation(
    num_employees: int = 994,
    team_size: int = 7,
    use_bias: bool = False
) -> dict:
    """
    Run forced ranking simulation.
    
    Args:
        num_employees: Total number of employees to simulate
        team_size: Number of employees per team
        use_bias: Whether to use biased (realistic) team assignment
        
    Returns:
        Dictionary containing error rates and classification results
    """
    pass
```

## Testing

All contributions should include tests:

```python
# tests/test_simulation.py
def test_random_assignment_error_rate():
    """Test that random assignment produces ~32% error rate."""
    results = run_simulation(use_bias=False, num_simulations=100)
    assert 0.30 <= results['error_rate'] <= 0.34
```

Run tests before submitting:

```bash
pytest tests/ -v
pytest tests/ --cov=forced_rank  # With coverage
```

## Commit Messages

Use clear, descriptive commit messages:

**Good:**
```
Add power law distribution support

- Implement Pareto distribution with configurable shape parameter
- Add tests for power law vs normal distribution comparison
- Update documentation with power law examples
```

**Bad:**
```
fix stuff
update code
```

## Pull Request Process

1. **Update documentation** if you change functionality
2. **Add tests** for new features
3. **Update CHANGELOG.md** with your changes
4. **Ensure all tests pass** locally
5. **Describe your changes** clearly in the PR

### PR Template

```markdown
## Description
Brief description of what this PR does

## Motivation
Why is this change needed?

## Changes
- Bullet list of changes

## Testing
How was this tested?

## Checklist
- [ ] Tests pass locally
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] Code follows style guide
```

## Research Contributions

If you're contributing research findings:

1. **Share methodology:** Explain your approach clearly
2. **Provide data:** Share datasets (anonymized if necessary)
3. **Document assumptions:** Be explicit about what you're modeling
4. **Show reproducibility:** Include code to replicate results

## Questions?

- **Open an issue** for bugs or feature requests
- **Start a discussion** for questions or ideas
- **Email the maintainer:** [your-email]
- **Visit the website:** [cageandmirror.com](https://cageandmirror.com)

## Code of Conduct

### Our Standards

- **Be respectful** of differing viewpoints and experiences
- **Be collaborative** - we're all trying to understand these systems better
- **Be constructive** - focus on what's right, not who's right
- **Be open** to feedback and willing to adapt

### Unacceptable Behavior

- Harassment, discrimination, or personal attacks
- Publishing others' private information without permission
- Trolling, insulting comments, or sustained disruption
- Any conduct which could reasonably be considered inappropriate

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Recognition

Contributors will be acknowledged in:
- CONTRIBUTORS.md file
- Release notes
- Book acknowledgments (for significant contributions)

Thank you for helping improve this project!
