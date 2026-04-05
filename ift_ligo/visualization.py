"""
Visualization Module: Plotting Tools for LIGO Analysis
======================================================

Generate publication-quality figures for gravitational wave analysis results.

Plots include:
    - Strain time series
    - Frequency domain spectra
    - Parameter posteriors
    - Likelihood ratios
    - Summary statistics
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

def plot_strain(time, strain, event_name="", savefig=None):
    """
    Plot gravitational wave strain
    
    Parameters:
        time: time array
        strain: strain data
        event_name: name of event
        savefig: filename to save figure
    """
    
    fig, ax = plt.subplots(figsize=(12, 5))
    
    ax.plot(time, strain, 'k-', linewidth=0.5, alpha=0.7)
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Strain (10$^{-21}$)', fontsize=12)
    ax.set_title(f'Gravitational Wave Strain: {event_name}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if savefig:
        plt.savefig(savefig, dpi=300, bbox_inches='tight')
    
    return fig, ax

def plot_spectrum(frequencies, spectrum_gr, spectrum_ift=None, savefig=None):
    """
    Plot frequency domain spectrum (GR vs IFT)
    
    Parameters:
        frequencies: frequency array (Hz)
        spectrum_gr: GR amplitude spectrum
        spectrum_ift: IFT amplitude spectrum (if None, don't plot)
        savefig: filename to save
    """
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot GR
    ax.loglog(frequencies, np.abs(spectrum_gr), 'b-', linewidth=2, label='General Relativity (GR)')
    
    # Plot IFT if provided
    if spectrum_ift is not None:
        ax.loglog(frequencies, np.abs(spectrum_ift), 'r--', linewidth=2, label='IFT with f¹ signature')
        
        # Show difference
        diff = np.abs(spectrum_ift - spectrum_gr) / np.abs(spectrum_gr)
        ax2 = ax.twinx()
        ax2.semilogx(frequencies, diff*100, 'g:', linewidth=1.5, alpha=0.7)
        ax2.set_ylabel('Relative Difference (%)', fontsize=11, color='g')
        ax2.tick_params(axis='y', labelcolor='g')
    
    ax.set_xlabel('Frequency (Hz)', fontsize=12)
    ax.set_ylabel('|h(f)| (strain/√Hz)', fontsize=12)
    ax.set_title('Gravitational Wave Spectrum: GR vs IFT', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, which='both', alpha=0.3)
    
    plt.tight_layout()
    if savefig:
        plt.savefig(savefig, dpi=300, bbox_inches='tight')
    
    return fig, ax

def plot_posterior(kappa_values, posterior, credible_intervals=None, savefig=None):
    """
    Plot κ posterior distribution
    
    Parameters:
        kappa_values: array of κ values
        posterior: posterior probability density
        credible_intervals: dict with CI levels
        savefig: filename to save
    """
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Normalize for plotting
    ax.fill_between(kappa_values, posterior, alpha=0.3, color='blue')
    ax.plot(kappa_values, posterior, 'b-', linewidth=2, label='Posterior')
    
    # Mark MAP
    idx_map = np.argmax(posterior)
    kappa_map = kappa_values[idx_map]
    ax.axvline(kappa_map, color='red', linestyle='--', linewidth=2, label=f'MAP: κ = {kappa_map:.2e}')
    
    # Credible intervals
    if credible_intervals:
        colors = ['green', 'orange', 'purple']
        for i, (level, (ci_min, ci_max)) in enumerate(credible_intervals.items()):
            ax.axvspan(ci_min, ci_max, alpha=0.15, color=colors[i % len(colors)],
                      label=f'{level}% CI')
    
    ax.set_xlabel('κ (IFT coupling)', fontsize=12)
    ax.set_ylabel('Posterior Density', fontsize=12)
    ax.set_title('Bayesian Parameter Estimation: IFT Coupling κ', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if savefig:
        plt.savefig(savefig, dpi=300, bbox_inches='tight')
    
    return fig, ax

def plot_likelihood_ratios(event_names, log_lambdas, savefig=None):
    """
    Plot likelihood ratios for multiple events
    
    Parameters:
        event_names: list of event names
        log_lambdas: log likelihood ratio for each event
        savefig: filename to save
    """
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Convert to delta chi-squared
    delta_chi2 = 2 * np.array(log_lambdas)
    
    # Color by significance
    colors = []
    for dc2 in delta_chi2:
        if dc2 > 25:
            colors.append('darkgreen')
        elif dc2 > 9:
            colors.append('green')
        elif dc2 > 0:
            colors.append('orange')
        else:
            colors.append('red')
    
    # Bar plot
    x_pos = np.arange(len(event_names))
    ax.bar(x_pos, delta_chi2, color=colors, alpha=0.7, edgecolor='black')
    
    # Reference lines
    ax.axhline(0, color='black', linestyle='-', linewidth=1)
    ax.axhline(4, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='2σ threshold')
    ax.axhline(9, color='green', linestyle='--', linewidth=1, alpha=0.5, label='3σ threshold')
    ax.axhline(25, color='darkgreen', linestyle='--', linewidth=1, alpha=0.5, label='5σ threshold')
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(event_names, rotation=45, ha='right', fontsize=10)
    ax.set_ylabel('Δχ² (test statistic)', fontsize=12)
    ax.set_title('f¹ Signature Likelihood Ratios: Event Comparison', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    if savefig:
        plt.savefig(savefig, dpi=300, bbox_inches='tight')
    
    return fig, ax

def plot_results(summary_table, savefig=None):
    """
    Generate comprehensive results figure
    
    Parameters:
        summary_table: pandas DataFrame with results
        savefig: filename to save
    """
    
    if summary_table is None:
        print("No results to plot")
        return None
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # 1. κ estimates
    ax1 = fig.add_subplot(gs[0, 0])
    events = summary_table['Event'].values
    kappas = summary_table['κ_MAP'].values
    # Extract numeric values (remove 'e' notation for parsing)
    kappas_numeric = [float(k.split('e')[0]) * 1e-14 if 'e' in k else float(k) for k in kappas]
    ax1.scatter(range(len(events)), kappas_numeric, s=100, alpha=0.6, color='blue')
    ax1.axhline(0, color='red', linestyle='--', alpha=0.5)
    ax1.set_ylabel('κ_MAP', fontsize=11)
    ax1.set_title('κ Parameter Estimates', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # 2. Δχ² values
    ax2 = fig.add_subplot(gs[0, 1])
    delta_chi2s = summary_table['Δχ²'].values
    delta_chi2s_numeric = [float(dc2) for dc2 in delta_chi2s]
    colors = ['green' if dc2 > 9 else 'orange' if dc2 > 0 else 'red' for dc2 in delta_chi2s_numeric]
    ax2.bar(range(len(events)), delta_chi2s_numeric, color=colors, alpha=0.7, edgecolor='black')
    ax2.axhline(9, color='green', linestyle='--', alpha=0.5, label='3σ')
    ax2.set_ylabel('Δχ²', fontsize=11)
    ax2.set_title('Test Statistics', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. SNR comparison
    ax3 = fig.add_subplot(gs[1, 0])
    snr_gr = summary_table['SNR_GR'].values
    snr_ift = summary_table['SNR_IFT'].values
    snr_gr_numeric = [float(s) for s in snr_gr]
    snr_ift_numeric = [float(s) for s in snr_ift]
    x = np.arange(len(events))
    width = 0.35
    ax3.bar(x - width/2, snr_gr_numeric, width, label='GR', alpha=0.7)
    ax3.bar(x + width/2, snr_ift_numeric, width, label='IFT', alpha=0.7)
    ax3.set_ylabel('SNR', fontsize=11)
    ax3.set_title('Signal-to-Noise Ratio', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Significance distribution
    ax4 = fig.add_subplot(gs[1, 1])
    significances = summary_table['Significance'].values
    sig_counts = {}
    for sig in significances:
        sig_counts[sig] = sig_counts.get(sig, 0) + 1
    ax4.bar(sig_counts.keys(), sig_counts.values(), color='skyblue', edgecolor='black', alpha=0.7)
    ax4.set_ylabel('Count', fontsize=11)
    ax4.set_title('Significance Distribution', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. Summary text
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis('off')
    
    # Calculate statistics
    delta_chi2_sum = sum(delta_chi2s_numeric)
    n_3sigma = sum(1 for dc2 in delta_chi2s_numeric if dc2 > 9)
    n_events = len(events)
    
    summary_text = f"""
    GWTC-3 f¹ SIGNATURE SEARCH SUMMARY
    ═══════════════════════════════════════════════════════════════════════════════
    
    Total Events Analyzed: {n_events}
    3σ Detections: {n_3sigma}/{n_events}
    Combined Δχ²: {delta_chi2_sum:.2f}
    Combined Significance: {np.sqrt(np.abs(delta_chi2_sum)):.2f}σ
    
    LIGO Constraint: κ < 4.9 × 10⁻¹⁴
    
    Conclusion: {"✓ Strong evidence for f¹ signature" if delta_chi2_sum > 9 else "⚠ Marginal evidence" if delta_chi2_sum > 0 else "✗ GR preferred"}
    """
    
    ax5.text(0.5, 0.5, summary_text, transform=ax5.transAxes,
            fontsize=11, verticalalignment='center', horizontalalignment='center',
            family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    if savefig:
        plt.savefig(savefig, dpi=300, bbox_inches='tight')
    
    return fig

# Test
if __name__ == "__main__":
    print("="*80)
    print("VISUALIZATION MODULE TEST")
    print("="*80)
    
    # Create sample data
    t = np.linspace(0, 4, 4*16384)
    strain = np.sin(2*np.pi*100*t) * np.exp(-t/2)
    f = np.fft.rfftfreq(len(strain), t[1]-t[0])
    spectrum = np.abs(np.fft.rfft(strain))
    
    # Test plots
    print("\n✓ Testing strain plot...")
    plot_strain(t, strain, event_name="GW150914", savefig="/tmp/strain.png")
    
    print("✓ Testing spectrum plot...")
    plot_spectrum(f, spectrum, savefig="/tmp/spectrum.png")
    
    print("✓ Testing posterior plot...")
    kappas = np.linspace(-1e-13, 1e-13, 100)
    posterior = np.exp(-((kappas-0.5e-14)/0.3e-14)**2)
    posterior /= np.trapz(posterior, kappas)
    plot_posterior(kappas, posterior, savefig="/tmp/posterior.png")
    
    print("✓ Testing likelihood ratio plot...")
    events = ['GW150914', 'GW151226', 'GW170104']
    log_lambdas = [1.5, 0.5, -0.3]
    plot_likelihood_ratios(events, log_lambdas, savefig="/tmp/likelihood_ratios.png")
    
    print("\n" + "="*80)
    print("✓ Visualization Module Operational")
    print("✓ Figures saved to /tmp/")
    print("="*80)
