# IFT LIGO f¹ Signature Search

**Searching for the Information Field Theory f¹ gravitational wave signature in LIGO GWTC-3 data**

## Overview

This project implements a complete pipeline for detecting the f¹ frequency dependence predicted by Information Field Theory (IFT) in LIGO gravitational wave observations.

### Central Prediction

In General Relativity, gravitational wave amplitude is frequency-independent (beyond power-law scaling). IFT predicts an additional **linear frequency dependence**:

```
h(f) = h_GR(f) · [1 + κ(f/f_ref)¹]
```

Where:
- κ ~ 10⁻¹⁴ is the IFT coupling constant
- f_ref = 100 Hz is the reference frequency
- This signature is **testable with LIGO O5 data** (2026-2027)

## Project Structure

```
IFT_LIGO_Analysis/
│
├── ift_ligo/                    (Main Python package)
│   ├── __init__.py              - Package initialization
│   ├── config.py                - Global configuration
│   ├── template.py              - GW template generation
│   ├── matched_filter.py        - Matched filter analysis
│   ├── statistical_analysis.py  - Bayesian inference
│   ├── gwtc3_analysis.py        - GWTC-3 reanalysis
│   └── plotting.py              - Visualization
│
├── notebooks/                   - Jupyter notebooks
│   ├── tutorial.ipynb           - Basic usage
│   └── full_analysis.ipynb      - Complete analysis
│
├── tests/                       - Unit tests
│   └── test_analysis.py
│
├── data/                        - Data directory
├── results/                     - Results storage
├── plots/                       - Generated figures
├── paper/                       - LaTeX paper
│
├── setup.py                     - Installation
├── requirements.txt             - Dependencies
├── README.md                    - This file
└── LICENSE                      - MIT License
```

## Installation

### From Source

```bash
git clone https://github.com/JuanDiegoVG/IFT-LIGO-Analysis.git
cd IFT_LIGO_Analysis
pip install -e .
```

### Requirements

- Python 3.7+
- numpy, scipy, pandas, matplotlib
- jupyter (for notebooks)

## Quick Start

### 1. Run Complete Analysis

```python
from ift_ligo import run_complete_analysis

results = run_complete_analysis()
```

This will:
- Load 14 GWTC-3 events
- Analyze each event for f¹ signature
- Perform Bayesian combined analysis
- Print results and constraints on κ

### 2. Analyze Single Event

```python
from ift_ligo import GWTC3Reanalysis, GravitationalWaveTemplate

reanalysis = GWTC3Reanalysis()
events = reanalysis.load_gwtc3_events()

# Analyze first event
event = events[0]
results = reanalysis.analyze_single_event(event)

print(f"Event: {event.name}")
print(f"SNR (GR): {results['snr_gr']:.2f}")
```

### 3. Matched Filter Analysis

```python
from ift_ligo import MatchedFilter, GravitationalWaveTemplate

# Generate templates
template_gen = GravitationalWaveTemplate()
t, h_gr = template_gen.strain_gr(m1=36, m2=29, distance=410)
t, h_ift = template_gen.strain_ift(m1=36, m2=29, distance=410, kappa=1.8e-14)

# Matched filter
mf = MatchedFilter()
snr_gr, snr_max_gr, _ = mf.matched_filter_snr(h_gr, h_gr)
snr_ift, snr_max_ift, _ = mf.matched_filter_snr(h_ift, h_ift)

print(f"SNR improvement: {(snr_max_ift/snr_max_gr - 1)*100:.2f}%")
```

### 4. Bayesian Inference

```python
from ift_ligo import BayesianAnalysis
import numpy as np

# Simulate data
snr_gr = np.random.normal(10, 2, 14)
snr_ift = snr_gr + 0.05

# Bayesian analysis
bayes = BayesianAnalysis(n_events=14)
kappa_array = np.linspace(-1e-13, 1e-13, 1000)
kappa_array, posterior = bayes.posterior_κ(snr_gr, snr_ift, kappa_array)

# Extract constraints
constraints = bayes.best_fit_and_errors(kappa_array, posterior)
print(f"κ = {constraints['best_fit']:.2e} +/- {(constraints['upper_68']-constraints['lower_68'])/2:.2e}")
```

### 5. Generate Plots

```python
from ift_ligo import LIGOPlots
import numpy as np

plots = LIGOPlots(output_dir='plots/')

# Create test data
t = np.linspace(0, 1, 1000)
strain_gr = 0.1 * np.sin(2*np.pi*100*t) * np.exp(-t/0.5)
strain_ift = strain_gr * 1.02

# Plot comparison
fig = plots.strain_comparison(t, strain_gr, strain_ift, 'GW150914')
```

## Key Modules

### template.py (GW Templates)
- `GravitationalWaveTemplate`: Generate templates
- `WaveformComparison`: Compare GR vs IFT

Key methods:
- `frequency_evolution_gr()` - GR frequency evolution
- `frequency_evolution_ift()` - IFT with f¹ term
- `strain_gr()` - Time-domain strain (GR)
- `strain_ift()` - Time-domain strain (IFT)

### matched_filter.py (Matched Filtering)
- `MatchedFilter`: Optimal filter implementation
- `SignalInjectionStudy`: Validation studies

Key methods:
- `whiten_data()` - Noise whitening
- `matched_filter_snr()` - SNR computation
- `bayesian_κ_estimation()` - Parameter inference
- `false_alarm_rate()` - FAR calculation

### statistical_analysis.py (Inference)
- `BayesianAnalysis`: Full Bayesian framework
- `HypothesisTesting`: Frequentist tests
- `SensitivityAnalysis`: Power calculations

Key methods:
- `posterior_κ()` - Posterior distribution
- `best_fit_and_errors()` - Confidence intervals
- `significance_κ_nonzero()` - Detection significance
- `chi_squared_test()` - Goodness of fit

### gwtc3_analysis.py (Event Analysis)
- `GWTC3Event`: Single event representation
- `GWTC3Reanalysis`: Combined analysis

Key methods:
- `load_gwtc3_events()` - Load catalog
- `analyze_single_event()` - Individual event
- `combined_analysis()` - Multi-event Bayesian
- `generate_summary_table()` - Results table

### plotting.py (Visualization)
- `LIGOPlots`: Publication-quality figures

Key methods:
- `strain_comparison()` - Waveform plots
- `matched_filter_snr()` - SNR evolution
- `posterior_κ()` - Posterior distribution
- `constraints_by_event()` - Event constraints
- `summary_figure()` - Combined results

## Configuration

Edit `ift_ligo/config.py` to customize:
- Physical constants
- LIGO parameters
- Analysis settings
- Output directories

Key parameters:
```python
KAPPA_TCI = 1.8e-14        # IFT coupling constant
F_REF = 100                 # Reference frequency
SNR_THRESHOLD = 5.0         # Detection threshold
SIGNIFICANCE_THRESHOLD = 3.0 # 3σ
```

## Results Storage

All results are saved to `results/` and `plots/`:

```
results/
├── gwtc3_f1_search_results.json
├── gwtc3_kappa_constraints.csv
└── likelihood_ratio_analysis.csv

plots/
├── strain_comparison_*.pdf
├── snr_comparison_*.pdf
├── posterior_kappa.pdf
├── constraints_by_event.pdf
├── frequency_evolution_*.pdf
└── summary_figure.pdf
```

## Expected Sensitivity

With 14 events and typical SNR~10 each:

| Parameter | Value |
|-----------|-------|
| Median SNR per event | 10 |
| Total SNR² | ~1400 |
| 1σ constraint on κ | ~10⁻¹⁵ |
| Detectable κ at 3σ | ~5×10⁻¹⁴ |

## Timeline

- **2026-2027**: LIGO O5 run (50+ events expected)
- **Months 0-3**: Preliminary analysis (10-15 events)
- **Months 3-6**: Intermediate results (30-40 events)
- **Months 6-12**: Final results (50+ events, 3σ+ sensitivity)

## Testing

```bash
python -m pytest tests/
```

Run individual tests:
```bash
python tests/test_analysis.py
```

## Notebooks

Interactive tutorials available in `notebooks/`:

1. **tutorial.ipynb**: Basic usage and concepts
2. **full_analysis.ipynb**: Complete GWTC-3 analysis

Run with:
```bash
jupyter notebook notebooks/
```

## Publication

LaTeX paper source in `paper/f1_signature_paper.tex`

Compile with:
```bash
cd paper
pdflatex f1_signature_paper.tex
```

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make improvements
4. Submit pull request

Areas of interest:
- GPU acceleration for matched filtering
- Advanced Bayesian methods
- Multi-band analysis
- Improved waveform templates

## References

1. Gabancho, J.D.V. (2026). "Information Field Theory: Unified Framework"
2. Abbott, B.P. et al. (2023). "GWTC-3: Compact Binary Coalescences Observed by LIGO and Virgo"
3. Planck Collaboration (2024). "Planck 2024 Results"

## License

MIT License - See LICENSE file for details

## Author

Juan Diego Vicente Gabancho
- Theoretical Physicist
- Information Field Theory Originator
- April 2026

## Citation

If you use this code in research, please cite:

```bibtex
@software{gabancho2026ligo,
  title={IFT LIGO f¹ Signature Search},
  author={Gabancho, Juan Diego Vicente},
  year={2026},
  url={https://github.com/JuanDiegoVG/IFT-LIGO-Analysis}
}
```

## Support

For questions or issues:
- Check the documentation
- Review example notebooks
- Submit an issue on GitHub

---

**Status**: Ready for LIGO O5 (2026-2027)
**Version**: 1.0.0
**Last Updated**: April 2, 2026
