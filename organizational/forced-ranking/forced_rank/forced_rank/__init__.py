"""
Forced Ranking Simulation

Agent-based simulation demonstrating systematic classification errors
in forced distribution performance management systems.

Supporting research for "The Cage and the Mirror: Engineering Capability 
Within Organizational Constraints" (McEntire, 2025).
"""

__version__ = "1.0.0"
__author__ = "Jeremy McEntire"
__email__ = "your-email@example.com"

from .simulation import (
    Simulation,
    run_simulation,
    print_results,
)

__all__ = [
    'Simulation',
    'run_simulation',
    'print_results',
]
