"""
IFT LIGO Analysis Package
=======================

Search for f¹ gravitational wave signature in LIGO GWTC-3 data

Main modules:
    - config: Global configuration
    - template: Gravitational wave templates (GR vs IFT)
    - matched_filter: Matched filter analysis
    - statistical_analysis: Bayesian inference
    - gwtc3_analysis: GWTC-3 reanalysis
    - plotting: Visualization
"""

__version__ = "1.0.0"
__author__ = "Juan Diego Vicente Gabancho"
__date__ = "April 2, 2026"
__license__ = "MIT"

# Import main classes
from .config import *
from .template import (
    GravitationalWaveTemplate,
    WaveformComparison
)
from .matched_filter import (
    MatchedFilter,
    SignalInjectionStudy
)
from .statistical_analysis import (
    BayesianAnalysis,
    HypothesisTesting,
    SensitivityAnalysis
)
from .gwtc3_analysis import (
    GWTC3Event,
    GWTC3Reanalysis
)
from .plotting import LIGOPlots

# Convenience function
def run_complete_analysis():
    """
    Run complete GWTC-3 f¹ signature search
    
    Returns:
        Results dictionary
    """
    
    print("="*80)
    print("LIGO f¹ SIGNATURE SEARCH - COMPLETE ANALYSIS")
    print("="*80)
    
    # Initialize reanalysis
    reanalysis = GWTC3Reanalysis(n_events=14)
    
    # Load GWTC-3 events
    print("\n1. Loading GWTC-3 events...")
    events = reanalysis.load_gwtc3_events()
    
    # Analyze each event
    print("\n2. Analyzing individual events...")
    for i, event in enumerate(events, 1):
        print(f"   {i}. {event.name}...", end=" ", flush=True)
        reanalysis.analyze_single_event(event)
        print("✓")
    
    # Combined analysis
    print("\n3. Performing Bayesian combined analysis...")
    kappa_array, posterior = reanalysis.combined_analysis()
    
    # Compute constraints
    print("4. Computing constraints...")
    constraints = reanalysis.compute_constraints(kappa_array, posterior)
    
    # Print summary
    reanalysis.print_summary()
    
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"\nBest-fit κ: {constraints['best_fit']:.3e}")
    print(f"68% C.L.:   [{constraints['1sigma_lower']:.3e}, {constraints['1sigma_upper']:.3e}]")
    print(f"Significance of κ ≠ 0: {constraints['significance_nonzero']:.2f}σ")
    
    if constraints['significance_nonzero'] > 3.0:
        print("\n✓✓✓ DETECTION: f¹ SIGNATURE FOUND AT >3σ ✓✓✓")
    else:
        print("\n✗ No significant detection (limit on κ)")
    
    print("\n" + "="*80)
    
    return {
        'events': events,
        'kappa_array': kappa_array,
        'posterior': posterior,
        'constraints': constraints
    }

__all__ = [
    'GravitationalWaveTemplate',
    'WaveformComparison',
    'MatchedFilter',
    'SignalInjectionStudy',
    'BayesianAnalysis',
    'HypothesisTesting',
    'SensitivityAnalysis',
    'GWTC3Event',
    'GWTC3Reanalysis',
    'LIGOPlots',
    'run_complete_analysis'
]
