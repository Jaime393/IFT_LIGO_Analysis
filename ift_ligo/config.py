"""
LIGO f¹ Signature Search - Configuration Module
==============================================

Global configuration for gravitational wave analysis
"""

import numpy as np

# ============================================================================
# PHYSICAL CONSTANTS
# ============================================================================

# Fundamental constants (SI units)
HBAR = 1.054571817e-34  # J·s
C = 299792458  # m/s
G = 6.67430e-11  # m³/(kg·s²)

# ============================================================================
# LIGO PARAMETERS
# ============================================================================

# LIGO detector configuration
LIGO_HANFORD_LAT = 46.455  # degrees
LIGO_HANFORD_LON = -119.408  # degrees
LIGO_LIVINGSTON_LAT = 30.494  # degrees
LIGO_LIVINGSTON_LON = -90.739  # degrees

# Detector arms
LIGO_ARM_LENGTH = 4000  # meters

# Sensitivity curve (approximate)
LIGO_LOW_FREQ = 20  # Hz (lower sensitivity cutoff)
LIGO_HIGH_FREQ = 8000  # Hz (upper sensitivity cutoff)
LIGO_OPTIMAL_FREQ = 150  # Hz (best sensitivity)

# ============================================================================
# GWTC-3 CATALOG PARAMETERS
# ============================================================================

# GWTC-3 contains 70 events from O3 run
GWTC3_N_EVENTS = 70

# Events used in analysis (subset for testing)
GWTC3_EVENTS = [
    "GW190412", "GW190814", "GW190915", "GW190917", "GW190930",
    "GW191103", "GW191105", "GW191109", "GW191113", "GW191129",
    "GW191204", "GW191219", "GW200105", "GW200115",
    "GW200129", "GW200202", "GW200209", "GW200219",
    "GW200224", "GW200225", "GW200311", "GW200316",
    "GW200425", "GW200426", "GW200519", "GW200520",
    "GW200602", "GW200606", "GW200612", "GW200630",
    # ... (70 total events available)
]

# Number of events for analysis
N_EVENTS_ANALYSIS = 14  # Start with subset

# ============================================================================
# IFT PARAMETERS
# ============================================================================

# IFT coupling constant (small correction)
KAPPA_TCI = 1.8e-14  # Dimensionless coupling strength

# f¹ signature parameters
F_REF = 100  # Reference frequency (Hz)
F_MIN = 20  # Minimum frequency for analysis (Hz)
F_MAX = 8000  # Maximum frequency (Hz)

# Bayesian analysis: Prior range for κ
KAPPA_MIN = -1e-13
KAPPA_MAX = 1e-13
N_KAPPA_SAMPLES = 1000

# ============================================================================
# DATA PROCESSING
# ============================================================================

# Sampling rate
SAMPLE_RATE = 16384  # Hz (standard LIGO)

# Whitening
WHITEN_WINDOW = 4  # seconds
WHITEN_FFT = 8192  # samples

# Band-pass filtering
BUTTERWORTH_ORDER = 4

# ============================================================================
# MATCHED FILTER PARAMETERS
# ============================================================================

# Template bank parameters
N_TEMPLATES_GR = 200  # Number of GR templates
N_TEMPLATES_TCI_PER_KAPPA = 50  # TCI templates per κ value

# SNR threshold for signal detection
SNR_THRESHOLD = 5.0  # Standard LIGO threshold

# False alarm probability
FAP_THRESHOLD = 1e-5

# ============================================================================
# STATISTICAL PARAMETERS
# ============================================================================

# Bayesian inference
N_WALKERS = 32  # MCMC walkers
N_STEPS = 1000  # MCMC steps
N_BURNIN = 100  # Burnin steps

# Significance testing
N_INJECTIONS = 1000  # Number of signal injections for calibration
SIGNIFICANCE_THRESHOLD = 3.0  # σ (3-sigma)

# ============================================================================
# OUTPUT PARAMETERS
# ============================================================================

# Result storage
RESULTS_DIR = "results/"
PLOTS_DIR = "plots/"
DATA_DIR = "data/"

# File names
GWTC3_CATALOG_FILE = "gwtc3_events.txt"
RESULTS_CSV = "gwtc3_kappa_constraints.csv"
LIKELIHOOD_CSV = "likelihood_ratio_analysis.csv"
BAYESIAN_POSTERIOR = "posterior_kappa.txt"

# ============================================================================
# ANALYSIS OPTIONS
# ============================================================================

# Enable different analysis modes
ANALYZE_STRAIN_DATA = True  # Use actual LIGO strain
INJECT_SIGNALS = False  # Add injected signals
BAYESIAN_INFERENCE = True  # Run Bayesian analysis
PLOT_RESULTS = True  # Generate plots

# Verbosity
VERBOSE = True
DEBUG = False

# ============================================================================
# DOCUMENTATION STRINGS
# ============================================================================

"""
CONFIGURATION GUIDE
===================

1. KAPPA_TCI: IFT coupling strength
   - Standard Model limit: κ → 0
   - IFT prediction: κ ~ 10⁻¹⁴
   - Testable range: κ ∈ [-10⁻¹³, 10⁻¹³]

2. F_REF: Reference frequency for f¹ scaling
   - Standard: 100 Hz
   - Should be within LIGO sensitivity band

3. SNR_THRESHOLD: Signal-to-noise ratio cutoff
   - Standard LIGO: SNR > 5
   - Needed for confident detection

4. SIGNIFICANCE_THRESHOLD: Detection significance
   - 3σ: ~99.7% confidence
   - Particle physics standard: 5σ

5. Data files should be in data/ directory
6. Results will be saved in results/ directory
7. Plots will be saved in plots/ directory
"""
