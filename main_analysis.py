#!/usr/bin/env python3
"""
GWTC-3 f¹ Signature Search - Main Analysis Script
================================================

Execute complete analysis pipeline for IFT f¹ signature
"""

import sys
import numpy as np
from datetime import datetime

# Add to path
sys.path.insert(0, '.')

from ift_ligo import (
    GWTC3Reanalysis, BayesianAnalysis, LIGOPlots,
    GravitationalWaveTemplate, MatchedFilter
)
from ift_ligo.config import *

def main():
    """Run complete analysis"""
    
    print("\n" + "="*80)
    print("INFORMATION FIELD THEORY - LIGO f¹ SIGNATURE SEARCH")
    print("="*80)
    print(f"\nTimestamp: {datetime.now().isoformat()}")
    print(f"Analysis: GWTC-3 Reanalysis for f¹ Signature Detection")
    
    # ========================================================================
    # PHASE 1: INITIALIZATION
    # ========================================================================
    
    print("\n" + "="*80)
    print("PHASE 1: INITIALIZATION")
    print("="*80)
    
    # Create reanalysis object
    reanalysis = GWTC3Reanalysis(n_events=14)
    
    # Load GWTC-3 events
    print("\nLoading GWTC-3 events...")
    events = reanalysis.load_gwtc3_events()
    print(f"✓ Loaded {len(events)} events from gravitational wave catalog")
    
    # Print event summary
    print("\nEvent Summary:")
    print(f"{'Event':<15} {'m1 (M☉)':<12} {'m2 (M☉)':<12} {'Distance (Mpc)':<15} {'SNR':<10}")
    print("-"*65)
    for event in events:
        print(f"{event.name:<15} {event.m1:<12.1f} {event.m2:<12.1f} {event.distance:<15.0f} {event.snr_network:<10.1f}")
    
    # ========================================================================
    # PHASE 2: INDIVIDUAL EVENT ANALYSIS
    # ========================================================================
    
    print("\n" + "="*80)
    print("PHASE 2: INDIVIDUAL EVENT ANALYSIS")
    print("="*80)
    
    print(f"\nAnalyzing {len(events)} events for f¹ signature...\n")
    
    for i, event in enumerate(events, 1):
        print(f"[{i:2d}/{len(events)}] {event.name}...", end=" ", flush=True)
        
        # Analyze event
        results = reanalysis.analyze_single_event(event)
        
        # Extract key results
        snr_gr = results['snr_gr']
        snr_ift = results['snr_ift']
        
        # SNR at κ=0 and max κ
        idx_zero = np.argmin(np.abs(results['kappa_values']))
        snr_at_max_kappa = snr_ift[-1]
        snr_improvement_pct = (snr_at_max_kappa / snr_gr - 1) * 100
        
        print(f"SNR: {snr_gr:.1f} (GR) → {snr_at_max_kappa:.1f} (IFT) [{snr_improvement_pct:+.2f}%] ✓")
    
    print(f"\n✓ Individual event analysis complete")
    
    # ========================================================================
    # PHASE 3: BAYESIAN COMBINED ANALYSIS
    # ========================================================================
    
    print("\n" + "="*80)
    print("PHASE 3: BAYESIAN COMBINED ANALYSIS")
    print("="*80)
    
    print("\nPerforming multi-event Bayesian inference...")
    
    # Combined analysis
    kappa_array, posterior = reanalysis.combined_analysis()
    
    print(f"✓ Analyzed parameter space: κ ∈ [{kappa_array[0]:.2e}, {kappa_array[-1]:.2e}]")
    print(f"✓ Grid points: {len(kappa_array)}")
    
    # ========================================================================
    # PHASE 4: CONSTRAINT EXTRACTION
    # ========================================================================
    
    print("\n" + "="*80)
    print("PHASE 4: CONSTRAINT EXTRACTION")
    print("="*80)
    
    print("\nComputing confidence intervals and significance...")
    
    constraints = reanalysis.compute_constraints(kappa_array, posterior)
    
    print(f"\n✓ Constraints computed:")
    print(f"  Best-fit κ:           {constraints['best_fit']:.3e}")
    print(f"  68% C.L. (1σ):        [{constraints['1sigma_lower']:.3e}, {constraints['1sigma_upper']:.3e}]")
    print(f"  95% C.L. (2σ):        [{constraints['2sigma_lower']:.3e}, {constraints['2sigma_upper']:.3e}]")
    print(f"  Significance (κ≠0):   {constraints['significance_nonzero']:.2f}σ")
    print(f"  P(κ > 0):             {constraints['p_positive']*100:.1f}%")
    print(f"  P(κ < 0):             {constraints['p_negative']*100:.1f}%")
    
    # ========================================================================
    # PHASE 5: RESULTS INTERPRETATION
    # ========================================================================
    
    print("\n" + "="*80)
    print("PHASE 5: RESULTS INTERPRETATION")
    print("="*80)
    
    sigma = constraints['significance_nonzero']
    
    if sigma > 5.0:
        result_status = "✓✓✓ STRONG DETECTION ✓✓✓"
        print(f"\n{result_status}")
        print(f"The f¹ signature is DETECTED at {sigma:.1f}σ significance!")
        print(f"This strongly supports Information Field Theory.")
    elif sigma > 3.0:
        result_status = "✓✓ DETECTION ✓✓"
        print(f"\n{result_status}")
        print(f"The f¹ signature is DETECTED at {sigma:.1f}σ significance.")
        print(f"Additional data should confirm IFT predictions.")
    elif sigma > 2.0:
        result_status = "✓ EVIDENCE"
        print(f"\n{result_status}")
        print(f"Possible evidence for f¹ signature at {sigma:.1f}σ.")
        print(f"More events needed for confirmation.")
    else:
        result_status = "CONSTRAINT"
        print(f"\nNo significant detection. Constraining κ to:")
        print(f"  κ < {abs(constraints['1sigma_upper']):.2e} (68% C.L.)")
        print(f"  κ < {abs(constraints['2sigma_upper']):.2e} (95% C.L.)")
    
    # ========================================================================
    # PHASE 6: VISUALIZATION
    # ========================================================================
    
    print("\n" + "="*80)
    print("PHASE 6: VISUALIZATION")
    print("="*80)
    
    print("\nGenerating publication-quality figures...")
    
    plots = LIGOPlots(output_dir='plots/')
    
    # Posterior distribution
    fig1 = plots.posterior_κ(kappa_array, posterior, constraints)
    print("✓ Posterior distribution figure")
    
    # Detection power
    sensitivity = BayesianAnalysis()
    kappa_test = np.linspace(-1e-13, 1e-13, 100)
    power = np.array([sensitivity.likelihood_single_event(10, 10+k*0.1, 1.0) 
                     for k in kappa_test])
    
    fig2 = plots.detection_power(kappa_test, power / np.max(power))
    print("✓ Detection power figure")
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    
    print(f"""
╔════════════════════════════════════════════════════════════════════════════╗
║                    IFT f¹ SIGNATURE SEARCH RESULTS                        ║
╠════════════════════════════════════════════════════════════════════════════╣
║                                                                            ║
║  Status:                    {result_status:<43}  ║
║  Significance:              {sigma:.2f}σ                                         ║
║  Best-fit κ:                {constraints['best_fit']:.3e}                                ║
║  68% Confidence Interval:   [{constraints['1sigma_lower']:.3e}, {constraints['1sigma_upper']:.3e}]  ║
║                                                                            ║
║  Number of events:          {len(events):<41}  ║
║  Total network SNR²:        {sum([e.snr_network**2 for e in events]):<41.0f}  ║
║                                                                            ║
║  Analysis date:             {datetime.now().strftime('%Y-%m-%d %H:%M:%S'):<40}  ║
║                                                                            ║
║  Next steps:                                                               ║
║  ├─ 2026-2027: LIGO O5 (50+ events)                                      ║
║  ├─ 2027-2030: Enhanced sensitivity analysis                             ║
║  └─ 2030+: Potential IFT confirmation or stronger constraints            ║
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Save results
    print("\nSaving results...")
    reanalysis.save_results('results/gwtc3_f1_search_results.json')
    print("✓ Results saved to results/gwtc3_f1_search_results.json")
    print("✓ Plots saved to plots/ directory")
    
    print("\n" + "="*80)
    print("✓ ANALYSIS COMPLETE")
    print("="*80 + "\n")
    
    return {
        'events': events,
        'kappa_array': kappa_array,
        'posterior': posterior,
        'constraints': constraints,
        'status': result_status
    }

if __name__ == "__main__":
    results = main()
