"""
Plotting and Visualization Module
=================================

Generate publication-quality figures for f¹ signature search
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import os

# Set publication quality defaults
rcParams['font.size'] = 11
rcParams['font.family'] = 'serif'
rcParams['figure.dpi'] = 150
rcParams['savefig.dpi'] = 300

class LIGOPlots:
    """
    Generate plots for gravitational wave analysis
    """
    
    def __init__(self, output_dir='plots/'):
        """
        Initialize plotting module
        
        Parameters:
            output_dir: Directory for saving plots
        """
        
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def strain_comparison(self, t, strain_gr, strain_ift, 
                         event_name="GW150914", save=True):
        """
        Compare strain waveforms: GR vs IFT
        
        Parameters:
            t: Time array
            strain_gr: Strain for GR
            strain_ift: Strain for IFT
            event_name: Event name for title
            save: Save to file
        """
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # Top: waveforms
        ax = axes[0]
        ax.plot(t, strain_gr, 'b-', label='GR', linewidth=1.5)
        ax.plot(t, strain_ift, 'r--', label='IFT (f¹)', linewidth=1.5, alpha=0.8)
        ax.set_ylabel('Strain', fontsize=12)
        ax.set_title(f'Gravitational Wave Comparison: {event_name}', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11, loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # Bottom: difference
        ax = axes[1]
        delta_strain = strain_ift - strain_gr
        ax.plot(t, delta_strain, 'g-', linewidth=1.5)
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('Δ Strain (IFT - GR)', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            filename = f"{self.output_dir}strain_comparison_{event_name}.pdf"
            plt.savefig(filename, bbox_inches='tight')
            print(f"✓ Saved: {filename}")
        
        return fig
    
    def matched_filter_snr(self, t, snr_gr, snr_ift, 
                          event_name="GW150914", save=True):
        """
        Compare matched filter SNR: GR vs IFT
        
        Parameters:
            t: Time array
            snr_gr: SNR for GR
            snr_ift: SNR for IFT
            event_name: Event name
            save: Save to file
        """
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.plot(t, snr_gr, 'b-', label='GR SNR', linewidth=2)
        ax.plot(t, snr_ift, 'r-', label='IFT SNR (f¹)', linewidth=2, alpha=0.8)
        
        # Mark SNR peaks
        peak_gr = np.max(snr_gr)
        peak_ift = np.max(snr_ift)
        t_peak_gr = t[np.argmax(snr_gr)]
        t_peak_ift = t[np.argmax(snr_ift)]
        
        ax.axhline(5.0, color='gray', linestyle=':', linewidth=1, label='Detection threshold')
        ax.plot(t_peak_gr, peak_gr, 'bo', markersize=10, label=f'GR peak: {peak_gr:.1f}')
        ax.plot(t_peak_ift, peak_ift, 'rs', markersize=10, label=f'IFT peak: {peak_ift:.1f}')
        
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('Signal-to-Noise Ratio (SNR)', fontsize=12)
        ax.set_title(f'Matched Filter SNR: {event_name}', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11, loc='upper right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            filename = f"{self.output_dir}snr_comparison_{event_name}.pdf"
            plt.savefig(filename, bbox_inches='tight')
            print(f"✓ Saved: {filename}")
        
        return fig
    
    def posterior_κ(self, kappa, posterior, constraints=None, save=True):
        """
        Plot posterior distribution of κ
        
        Parameters:
            kappa: κ array
            posterior: Posterior probability
            constraints: Dictionary with confidence intervals
            save: Save to file
        """
        
        fig, ax = plt.subplots(figsize=(11, 7))
        
        # Posterior
        ax.fill_between(kappa, posterior, alpha=0.3, color='blue', label='Posterior')
        ax.plot(kappa, posterior, 'b-', linewidth=2)
        
        if constraints:
            # Best fit
            best = constraints['best_fit']
            ax.axvline(best, color='red', linestyle='--', linewidth=2, label=f"Best fit: κ = {best:.2e}")
            
            # 1σ interval
            lower_1s = constraints['1sigma_lower']
            upper_1s = constraints['1sigma_upper']
            ax.axvspan(lower_1s, upper_1s, alpha=0.1, color='red', label='68% C.L.')
            
            # 2σ interval
            lower_2s = constraints['2sigma_lower']
            upper_2s = constraints['2sigma_upper']
            ax.axvspan(lower_2s, upper_2s, alpha=0.05, color='red')
        
        # GR limit
        ax.axvline(0, color='black', linestyle='-', linewidth=1.5, label='GR prediction (κ=0)')
        
        ax.set_xlabel('Coupling constant κ', fontsize=12)
        ax.set_ylabel('Probability density', fontsize=12)
        ax.set_title('Bayesian Posterior: f¹ Coupling Parameter', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11, loc='upper right')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save:
            filename = f"{self.output_dir}posterior_kappa.pdf"
            plt.savefig(filename, bbox_inches='tight')
            print(f"✓ Saved: {filename}")
        
        return fig
    
    def constraints_by_event(self, events_data, save=True):
        """
        Plot κ constraints from individual events
        
        Parameters:
            events_data: List of (event_name, best_fit, error) tuples
            save: Save to file
        """
        
        names = [d[0] for d in events_data]
        best_fits = [d[1] for d in events_data]
        errors = [d[2] for d in events_data]
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Plot constraints
        y_pos = np.arange(len(names))
        ax.errorbar(best_fits, y_pos, xerr=errors, fmt='o', markersize=8, 
                   capsize=5, capthick=2, linewidth=2, color='blue')
        
        # GR line
        ax.axvline(0, color='red', linestyle='--', linewidth=2, label='GR prediction')
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names)
        ax.set_xlabel('Coupling constant κ', fontsize=12)
        ax.set_title('f¹ Coupling Constraints: Individual Events', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        if save:
            filename = f"{self.output_dir}constraints_by_event.pdf"
            plt.savefig(filename, bbox_inches='tight')
            print(f"✓ Saved: {filename}")
        
        return fig
    
    def frequency_evolution(self, t_gr, f_gr, t_ift, f_ift, 
                           event_name="GW150914", save=True):
        """
        Compare frequency evolution: GR vs IFT
        
        Parameters:
            t_gr, f_gr: GR time and frequency
            t_ift, f_ift: IFT time and frequency
            event_name: Event name
            save: Save to file
        """
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 9))
        
        # Top: frequency evolution
        ax = axes[0]
        ax.plot(t_gr, f_gr, 'b-', label='GR', linewidth=2)
        ax.plot(t_ift, f_ift, 'r--', label='IFT (f¹)', linewidth=2, alpha=0.8)
        ax.set_ylabel('Frequency (Hz)', fontsize=12)
        ax.set_yscale('log')
        ax.set_title(f'Frequency Evolution: {event_name}', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, which='both')
        
        # Bottom: relative difference
        ax = axes[1]
        # Interpolate to same time grid
        f_gr_interp = np.interp(t_ift, t_gr, f_gr)
        rel_diff = (f_ift - f_gr_interp) / f_gr_interp * 100
        
        ax.plot(t_ift, rel_diff, 'g-', linewidth=2)
        ax.axhline(0, color='black', linestyle=':', linewidth=1)
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('Relative difference (%)', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            filename = f"{self.output_dir}frequency_evolution_{event_name}.pdf"
            plt.savefig(filename, bbox_inches='tight')
            print(f"✓ Saved: {filename}")
        
        return fig
    
    def detection_power(self, kappa_values, power, significance=3.0, save=True):
        """
        Plot detection power vs κ
        
        Parameters:
            kappa_values: κ array
            power: Detection power (0 to 1)
            significance: Significance level
            save: Save to file
        """
        
        fig, ax = plt.subplots(figsize=(11, 7))
        
        ax.plot(kappa_values * 1e14, power * 100, 'b-', linewidth=2.5, 
               label=f'{significance}σ significance')
        ax.fill_between(kappa_values * 1e14, power * 100, alpha=0.3, color='blue')
        
        # 50% power
        idx_half = np.argmin(np.abs(power - 0.5))
        kappa_half = kappa_values[idx_half]
        ax.axvline(kappa_half, color='red', linestyle='--', linewidth=2,
                  label=f'κ = {kappa_half:.2e} (50% power)')
        
        ax.axhline(50, color='gray', linestyle=':', linewidth=1)
        
        ax.set_xlabel('Coupling constant κ (×10⁻¹⁴)', fontsize=12)
        ax.set_ylabel('Detection Power (%)', fontsize=12)
        ax.set_title('f¹ Signature Detection Power (GWTC-3)', fontsize=14, fontweight='bold')
        ax.set_ylim([0, 105])
        ax.legend(fontsize=11, loc='lower right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            filename = f"{self.output_dir}detection_power.pdf"
            plt.savefig(filename, bbox_inches='tight')
            print(f"✓ Saved: {filename}")
        
        return fig
    
    def summary_figure(self, kappa, posterior, constraints, snr_comparison, save=True):
        """
        Create comprehensive summary figure (2×2)
        
        Parameters:
            kappa: κ array
            posterior: Posterior distribution
            constraints: Constraint dictionary
            snr_comparison: (snr_gr, snr_ift) tuple
            save: Save to file
        """
        
        fig = plt.figure(figsize=(14, 10))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        # Top left: Posterior
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.fill_between(kappa, posterior, alpha=0.3, color='blue')
        ax1.plot(kappa, posterior, 'b-', linewidth=2)
        ax1.axvline(constraints['best_fit'], color='red', linestyle='--', linewidth=2)
        ax1.set_xlabel('κ', fontsize=11)
        ax1.set_ylabel('P(κ|data)', fontsize=11)
        ax1.set_title('Posterior Distribution', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Top right: Constraints table
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.axis('off')
        
        table_data = [
            ['Best fit', f"{constraints['best_fit']:.2e}"],
            ['1σ lower', f"{constraints['1sigma_lower']:.2e}"],
            ['1σ upper', f"{constraints['1sigma_upper']:.2e}"],
            ['2σ lower', f"{constraints['2sigma_lower']:.2e}"],
            ['2σ upper', f"{constraints['2sigma_upper']:.2e}"],
            ['Significance', f"{constraints['significance_nonzero']:.2f}σ"],
        ]
        
        table = ax2.table(cellText=table_data, cellLoc='center', 
                         colLabels=['Parameter', 'Value'],
                         loc='center', bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        ax2.set_title('Results Summary', fontsize=12, fontweight='bold', pad=20)
        
        # Bottom left: SNR comparison
        ax3 = fig.add_subplot(gs[1, 0])
        snr_gr, snr_ift = snr_comparison
        events = ['Event ' + str(i) for i in range(len(snr_gr))]
        x = np.arange(len(events))
        width = 0.35
        
        ax3.bar(x - width/2, snr_gr, width, label='GR', alpha=0.8)
        ax3.bar(x + width/2, snr_ift, width, label='IFT (f¹)', alpha=0.8)
        ax3.axhline(5, color='red', linestyle='--', linewidth=1, alpha=0.5)
        
        ax3.set_ylabel('SNR', fontsize=11)
        ax3.set_title('SNR Comparison', fontsize=12, fontweight='bold')
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Bottom right: text summary
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.axis('off')
        
        summary_text = f"""
GWTC-3 f¹ Signature Search Results

Event Count: {len(snr_gr)}
Detection Significance: {constraints['significance_nonzero']:.2f}σ

κ = {constraints['best_fit']:.2e} ± {(constraints['1sigma_upper']-constraints['1sigma_lower'])/2:.2e}

Status: {'DETECTION' if constraints['significance_nonzero'] > 3.0 else 'CONSTRAINT'}
        """
        
        ax4.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
                verticalalignment='center', bbox=dict(boxstyle='round', 
                facecolor='wheat', alpha=0.3))
        
        fig.suptitle('f¹ Signature Search: Summary Results', fontsize=16, fontweight='bold', y=0.98)
        
        if save:
            filename = f"{self.output_dir}summary_figure.pdf"
            plt.savefig(filename, bbox_inches='tight')
            print(f"✓ Saved: {filename}")
        
        return fig

# Test
if __name__ == "__main__":
    print("="*80)
    print("PLOTTING MODULE TEST")
    print("="*80)
    
    plots = LIGOPlots()
    
    # Create test data
    t = np.linspace(0, 1, 1000)
    strain_gr = 0.1 * np.sin(2*np.pi*100*t) * np.exp(-t/0.5)
    strain_ift = strain_gr * 1.02  # 2% difference
    
    # Generate a plot
    fig = plots.strain_comparison(t, strain_gr, strain_ift, 
                                  event_name="GW150914")
    
    print("\n✓ Plotting module operational")
    print("✓ Generated test figures")
