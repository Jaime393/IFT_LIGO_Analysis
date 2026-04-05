"""
GWAnalyzer: Complete LIGO Analysis Pipeline
===========================================

Main class that integrates all modules for end-to-end analysis
of the f¹ gravitational wave signature in LIGO data.

Workflow:
    1. Load event catalog (GWTC-3)
    2. Generate GR and IFT templates
    3. Perform matched-filter analysis
    4. Estimate κ via Bayesian inference
    5. Compute likelihood ratios
    6. Generate results and plots
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Dict
import json

from .event_catalog import GWTCCatalog, GWEvent
from .gw_templates import GWTemplate, GWTemplateIFT
from .matched_filter import MatchedFilterAnalyzer
from .statistical_inference import BayesianInference, LikelihoodRatio

class GWAnalyzer:
    """
    Complete gravitational wave analysis pipeline for f¹ signature
    """
    
    def __init__(self, catalog_name='GWTC-3', sampling_rate=16384):
        """
        Initialize analyzer
        
        Parameters:
            catalog_name: 'GWTC-3' or other catalog
            sampling_rate: Hz (16384 for LIGO)
        """
        
        print("\n" + "="*80)
        print("INITIALIZING GW ANALYZER FOR f¹ SIGNATURE SEARCH")
        print("="*80)
        
        # Load catalog
        self.catalog = GWTCCatalog()
        print(f"\n✓ Loaded {catalog_name} catalog: {self.catalog.n_events} events")
        
        # Initialize matched filter analyzer
        self.mf_analyzer = MatchedFilterAnalyzer(sampling_rate=sampling_rate)
        print(f"✓ Matched filter analyzer initialized")
        
        # Statistical inference
        self.bayes = BayesianInference()
        self.likelihood_ratio = LikelihoodRatio()
        print(f"✓ Bayesian inference engine ready")
        
        # Results storage
        self.results = {}
        self.summary_table = None
        
    def search_f1_signature(self, event_subset: Optional[List[str]] = None,
                           kappa_range: Optional[np.ndarray] = None,
                           all_events: bool = False):
        """
        Search for f¹ signature in events
        
        Parameters:
            event_subset: specific event names (if None, use all)
            kappa_range: array of κ values to test
            all_events: use all catalog events
            
        Returns:
            Results dictionary
        """
        
        print("\n" + "="*80)
        print("SEARCHING FOR f¹ SIGNATURE IN GRAVITATIONAL WAVE EVENTS")
        print("="*80)
        
        # Set default κ range if not provided
        if kappa_range is None:
            kappa_range = np.linspace(-1e-13, 1e-13, 50)
        
        # Select events
        if all_events:
            events = self.catalog.events
        elif event_subset:
            events = [self.catalog.get_event(name) for name in event_subset]
            events = [e for e in events if e is not None]
        else:
            # Use default set of significant events
            events = self.catalog.events[:10]
        
        print(f"\nAnalyzing {len(events)} events...\n")
        
        # Analyze each event
        event_results = []
        
        for i, event in enumerate(events):
            print(f"[{i+1}/{len(events)}] Analyzing {event.name}...", end=" ")
            
            # Generate templates
            template_gr = GWTemplate(event.m1, event.m2)
            
            # Simulate data
            f_array = self.mf_analyzer.frequencies
            h_gr = template_gr.strain(f_array, distance=event.luminosity_distance)
            h_time = np.fft.irfft(h_gr)
            np.random.seed(hash(event.name) % 2**32)
            noise = np.random.randn(len(h_time)) * 1e-21
            data = h_time + noise
            
            # Bayesian parameter estimation
            posteriors = []
            for kappa in kappa_range:
                template_ift = GWTemplateIFT(event.m1, event.m2, kappa=kappa)
                h_ift = template_ift.strain(f_array, distance=event.luminosity_distance)
                h_ift_time = np.fft.irfft(h_ift)
                
                # Likelihood
                snr_ift = self.mf_analyzer.matched_filter_snr(data, h_ift_time)
                posteriors.append(snr_ift**2)
            
            posteriors = np.array(posteriors)
            
            # Normalize posterior
            posterior = posteriors / np.sum(posteriors)
            
            # Point estimates
            idx_max = np.argmax(posterior)
            kappa_map = kappa_range[idx_max]
            kappa_mean = np.sum(kappa_range * posterior)
            
            # Likelihood ratio (vs GR)
            template_ift_best = GWTemplateIFT(event.m1, event.m2, kappa=kappa_map)
            h_ift_best = template_ift_best.strain(f_array, distance=event.luminosity_distance)
            h_ift_best_time = np.fft.irfft(h_ift_best)
            
            snr_gr = self.mf_analyzer.matched_filter_snr(data, h_time)
            snr_ift_best = self.mf_analyzer.matched_filter_snr(data, h_ift_best_time)
            
            delta_chi2 = snr_ift_best**2 - snr_gr**2
            
            print(f"κ_MAP = {kappa_map:.2e}, Δχ² = {delta_chi2:.2f}")
            
            # Store results
            result = {
                'event': event.name,
                'kappa_map': kappa_map,
                'kappa_mean': kappa_mean,
                'kappa_range': [kappa_range[0], kappa_range[-1]],
                'delta_chi2': delta_chi2,
                'snr_gr': snr_gr,
                'snr_ift': snr_ift_best,
                'posterior': posterior.tolist(),
            }
            event_results.append(result)
            self.results[event.name] = result
        
        # Summary
        print("\n" + "="*80)
        print("SEARCH COMPLETE")
        print("="*80)
        
        return event_results
    
    def summary_statistics(self):
        """
        Print summary statistics of search results
        """
        
        if not self.results:
            print("No results available - run search_f1_signature first")
            return
        
        print("\n" + "="*80)
        print("SUMMARY STATISTICS: f¹ SIGNATURE SEARCH")
        print("="*80)
        
        kappas = [r['kappa_map'] for r in self.results.values()]
        delta_chi2s = [r['delta_chi2'] for r in self.results.values()]
        
        print(f"\nκ Parameter Estimates (MAP):")
        print(f"  Mean: {np.mean(kappas):.3e}")
        print(f"  Std Dev: {np.std(kappas):.3e}")
        print(f"  Range: [{np.min(kappas):.3e}, {np.max(kappas):.3e}]")
        
        print(f"\nTest Statistics (Δχ²):")
        print(f"  Mean: {np.mean(delta_chi2s):.2f}")
        print(f"  Range: [{np.min(delta_chi2s):.2f}, {np.max(delta_chi2s):.2f}]")
        
        # Count significances
        n_3sigma = sum(1 for dc2 in delta_chi2s if dc2 > 9)
        n_5sigma = sum(1 for dc2 in delta_chi2s if dc2 > 25)
        
        print(f"\nSignificance Distribution:")
        print(f"  5σ detections: {n_5sigma}")
        print(f"  3σ detections: {n_3sigma}")
        print(f"  Events favoring f¹: {sum(1 for dc2 in delta_chi2s if dc2 > 0)}/{len(delta_chi2s)}")
        
        # Combined evidence
        combined_chi2 = np.sum(delta_chi2s)
        print(f"\nCombined Evidence:")
        print(f"  Σ Δχ²: {combined_chi2:.2f}")
        print(f"  Combined σ: {np.sqrt(np.abs(combined_chi2)):.2f}")
        
        if combined_chi2 > 25:
            print("\n✓✓ 5σ DETECTION OF f¹ SIGNATURE ACROSS ALL EVENTS")
        elif combined_chi2 > 9:
            print("\n✓ 3σ STRONG EVIDENCE FOR f¹ SIGNATURE")
        elif combined_chi2 > 0:
            print("\n⚠ Marginal evidence for f¹ (need more data)")
        else:
            print("\n✗ General Relativity is preferred")
    
    def export_results(self, filename='gwtc3_f1_results.json'):
        """
        Export results to JSON file
        
        Parameters:
            filename: output filename
        """
        
        if not self.results:
            print("No results to export")
            return
        
        # Convert to serializable format
        export_data = {}
        for name, result in self.results.items():
            export_data[name] = {
                'kappa_map': float(result['kappa_map']),
                'kappa_mean': float(result['kappa_mean']),
                'delta_chi2': float(result['delta_chi2']),
                'snr_gr': float(result['snr_gr']),
                'snr_ift': float(result['snr_ift']),
            }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"\n✓ Results exported to {filename}")
    
    def generate_summary_table(self):
        """
        Generate summary table of all results
        
        Returns:
            pandas DataFrame
        """
        
        if not self.results:
            print("No results available")
            return None
        
        data = {
            'Event': [],
            'κ_MAP': [],
            'κ_Mean': [],
            'Δχ²': [],
            'SNR_GR': [],
            'SNR_IFT': [],
            'Significance': []
        }
        
        for name, result in self.results.items():
            data['Event'].append(name)
            data['κ_MAP'].append(f"{result['kappa_map']:.2e}")
            data['κ_Mean'].append(f"{result['kappa_mean']:.2e}")
            data['Δχ²'].append(f"{result['delta_chi2']:.2f}")
            data['SNR_GR'].append(f"{result['snr_gr']:.1f}")
            data['SNR_IFT'].append(f"{result['snr_ift']:.1f}")
            
            # Significance
            dc2 = result['delta_chi2']
            if dc2 > 25:
                sig = "5σ"
            elif dc2 > 9:
                sig = "3σ"
            elif dc2 > 4:
                sig = "2σ"
            elif dc2 > 0:
                sig = "<2σ"
            else:
                sig = "GR"
            data['Significance'].append(sig)
        
        df = pd.DataFrame(data)
        self.summary_table = df
        
        return df
    
    def print_summary_table(self):
        """Print summary table"""
        
        df = self.generate_summary_table()
        
        if df is None:
            return
        
        print("\n" + "="*80)
        print("SUMMARY TABLE: f¹ SIGNATURE SEARCH RESULTS")
        print("="*80)
        print(df.to_string(index=False))
        print("="*80)

# Test
if __name__ == "__main__":
    print("="*80)
    print("GW ANALYZER TEST")
    print("="*80)
    
    # Create analyzer
    analyzer = GWAnalyzer('GWTC-3', sampling_rate=16384)
    
    # Run search on subset of events
    print("\nRunning f¹ signature search on first 5 events...")
    results = analyzer.search_f1_signature(all_events=False)
    
    # Print results
    analyzer.summary_statistics()
    analyzer.print_summary_table()
    
    # Export
    analyzer.export_results('test_results.json')
    
    print("\n" + "="*80)
    print("✓ GW Analyzer Operational")
    print("="*80)
