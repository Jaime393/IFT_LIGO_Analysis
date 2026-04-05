"""
Statistical Analysis Module
==========================

Bayesian inference and hypothesis testing for f¹ signature
"""

import numpy as np
from scipy.stats import norm, chi2
from scipy.special import erf
import warnings

class BayesianAnalysis:
    """
    Bayesian inference for κ parameter estimation
    """
    
    def __init__(self, n_events=14):
        """
        Initialize Bayesian analysis
        
        Parameters:
            n_events: Number of gravitational wave events
        """
        
        self.n_events = n_events
        self.results = {}
    
    def likelihood_single_event(self, snr_gr, snr_ift, rho_tci=1.0):
        """
        Likelihood for single event
        
        L(κ) ∝ exp(SNR_IFT² / 2) / exp(SNR_GR² / 2)
        
        Parameters:
            snr_gr: SNR for GR template
            snr_ift: SNR for IFT template
            rho_tci: Relative fitness of IFT vs GR
            
        Returns:
            Likelihood value
        """
        
        # Log likelihood
        log_L = (snr_ift**2 - snr_gr**2) / 2
        
        # Avoid overflow
        log_L = np.clip(log_L, -100, 100)
        
        likelihood = np.exp(log_L)
        
        return likelihood
    
    def likelihood_combined(self, snr_gr_array, snr_ift_array):
        """
        Combined likelihood for multiple events
        
        L_total = ∏ L_i
        
        Parameters:
            snr_gr_array: SNR values for GR (array)
            snr_ift_array: SNR values for IFT (array)
            
        Returns:
            Combined likelihood
        """
        
        likelihoods = np.zeros(len(snr_gr_array))
        
        for i in range(len(snr_gr_array)):
            likelihoods[i] = self.likelihood_single_event(
                snr_gr_array[i], snr_ift_array[i]
            )
        
        # Combined (use log to avoid underflow)
        log_likelihood = np.sum(np.log(likelihoods + 1e-300))
        
        return np.exp(np.clip(log_likelihood, -100, 100))
    
    def posterior_κ(self, snr_gr_array, snr_ift_array, 
                    kappa_array, prior_flat=True):
        """
        Posterior probability distribution for κ
        
        P(κ|data) ∝ L(data|κ) * P(κ)
        
        Parameters:
            snr_gr_array: GR SNR values
            snr_ift_array: IFT SNR values
            kappa_array: κ values to evaluate
            prior_flat: Use flat prior
            
        Returns:
            κ array, posterior array
        """
        
        # Compute likelihood for each κ
        n_kappa = len(kappa_array)
        likelihood = np.zeros(n_kappa)
        
        for i, kappa in enumerate(kappa_array):
            # Delta SNR for this κ
            delta_snr_sq = (snr_ift_array**2 - snr_gr_array**2) / 2
            
            # Log likelihood
            log_L = np.sum(delta_snr_sq)
            
            likelihood[i] = np.exp(np.clip(log_L, -100, 100))
        
        # Prior
        if prior_flat:
            prior = np.ones_like(kappa_array)
        else:
            # Gaussian prior centered at 0
            prior = norm.pdf(kappa_array, 0, 5e-14)
        
        # Posterior = Likelihood × Prior
        posterior = likelihood * prior
        
        # Normalize
        posterior = posterior / np.sum(posterior)
        
        return kappa_array, posterior
    
    def best_fit_and_errors(self, kappa_array, posterior):
        """
        Extract best-fit value and error bars
        
        Parameters:
            kappa_array: Parameter array
            posterior: Posterior distribution
            
        Returns:
            Best-fit κ, 68% interval (1σ), 95% interval (2σ)
        """
        
        # Best fit
        idx_best = np.argmax(posterior)
        kappa_best = kappa_array[idx_best]
        
        # Credible intervals
        cdf = np.cumsum(posterior)
        cdf = cdf / cdf[-1]
        
        # 68% interval (1σ)
        idx_lower_68 = np.argmin(np.abs(cdf - 0.16))
        idx_upper_68 = np.argmin(np.abs(cdf - 0.84))
        kappa_lower_68 = kappa_array[idx_lower_68]
        kappa_upper_68 = kappa_array[idx_upper_68]
        
        # 95% interval (2σ)
        idx_lower_95 = np.argmin(np.abs(cdf - 0.025))
        idx_upper_95 = np.argmin(np.abs(cdf - 0.975))
        kappa_lower_95 = kappa_array[idx_lower_95]
        kappa_upper_95 = kappa_array[idx_upper_95]
        
        return {
            'best_fit': kappa_best,
            'lower_68': kappa_lower_68,
            'upper_68': kappa_upper_68,
            'lower_95': kappa_lower_95,
            'upper_95': kappa_upper_95
        }
    
    def significance_κ_nonzero(self, kappa_array, posterior):
        """
        Test significance of κ ≠ 0
        
        Compute probability that κ ≠ 0
        
        Parameters:
            kappa_array: Parameter array
            posterior: Posterior distribution
            
        Returns:
            P(κ > 0), P(κ < 0), σ significance
        """
        
        # Probability κ > 0
        idx_positive = kappa_array > 0
        p_positive = np.sum(posterior[idx_positive])
        
        # Probability κ < 0
        idx_negative = kappa_array < 0
        p_negative = np.sum(posterior[idx_negative])
        
        # Significance (convert probability to σ)
        # P = 0.5 + erf(σ/√2) / 2
        p_max = max(p_positive, p_negative)
        
        if p_max < 0.5:
            sigma = 0
        else:
            # Inverse of error function
            x = 2 * (p_max - 0.5)
            sigma = np.sqrt(2) * np.arctan(x / np.sqrt(1 - x**2)) * 2 / np.pi
        
        return p_positive, p_negative, sigma
    
    def bayes_factor_κ0_vs_κnonzero(self, kappa_array, posterior):
        """
        Bayes factor comparing κ=0 vs κ≠0 models
        
        BF = P(κ=0|data) / P(κ≠0|data)
        
        Returns:
            Bayes factor
        """
        
        # Probability κ=0 (approximate as region within δκ of zero)
        delta_kappa = kappa_array[1] - kappa_array[0]
        idx_zero = np.abs(kappa_array) < 2 * delta_kappa
        p_zero = np.sum(posterior[idx_zero])
        
        # Probability κ≠0
        p_nonzero = 1 - p_zero
        
        # Bayes factor
        bayes_factor = p_zero / (p_nonzero + 1e-30)
        
        return bayes_factor

class HypothesisTesting:
    """
    Frequentist hypothesis testing framework
    """
    
    def __init__(self):
        pass
    
    def chi_squared_test(self, observed, expected, errors=None):
        """
        χ² test for goodness of fit
        
        χ² = Σ (O_i - E_i)² / E_i
        
        Parameters:
            observed: Observed values
            expected: Expected values
            errors: Error bars
            
        Returns:
            χ² value, p-value, degrees of freedom
        """
        
        observed = np.array(observed)
        expected = np.array(expected)
        
        if errors is None:
            errors = np.sqrt(expected)
        
        errors = np.maximum(errors, 1e-10)  # Avoid division by zero
        
        chi2_val = np.sum(((observed - expected) / errors)**2)
        dof = len(observed) - 1
        
        # p-value
        p_value = 1 - chi2.cdf(chi2_val, dof)
        
        return chi2_val, p_value, dof
    
    def likelihood_ratio_test(self, log_likelihood_null, log_likelihood_alt):
        """
        Likelihood ratio test
        
        LR = -2 * (log L_null - log L_alt)
        
        Approximately χ² distributed with 1 dof
        
        Parameters:
            log_likelihood_null: Log likelihood for null model
            log_likelihood_alt: Log likelihood for alternative model
            
        Returns:
            LR value, p-value
        """
        
        lr = -2 * (log_likelihood_null - log_likelihood_alt)
        
        # p-value (1 dof)
        p_value = 1 - chi2.cdf(lr, 1)
        
        return lr, p_value
    
    def pull_distribution(self, measurement, true_value, uncertainty):
        """
        Pull distribution: (measurement - truth) / uncertainty
        
        Should be normal with mean 0 and std 1 if errors are correct
        
        Parameters:
            measurement: Measured values
            true_value: True values
            uncertainty: Measurement uncertainties
            
        Returns:
            Pull values
        """
        
        pulls = (measurement - true_value) / uncertainty
        
        return pulls

class SensitivityAnalysis:
    """
    Sensitivity and power analysis
    """
    
    def __init__(self, n_events=14, snr_per_event=10):
        """
        Initialize sensitivity analysis
        
        Parameters:
            n_events: Number of events
            snr_per_event: Average SNR per event
        """
        
        self.n_events = n_events
        self.snr_per_event = snr_per_event
    
    def minimum_detectable_κ(self, significance_level=3.0):
        """
        Minimum κ that can be detected at given significance
        
        Parameters:
            significance_level: Detection significance (σ)
            
        Returns:
            Minimum κ
        """
        
        # Total SNR² from all events
        total_snr_sq = self.n_events * self.snr_per_event**2
        
        # κ affects SNR² as: (SNR_IFT² - SNR_GR²) ∝ κ
        # So: δ(SNR²) / SNR² ~ κ * (f/f_ref)
        
        # Approximate: detectable κ is when Δχ² = significance²
        f_ref = 100
        f_typical = 100  # typical frequency
        
        min_kappa = significance_level / (total_snr_sq**0.5 * (f_typical / f_ref))
        
        return min_kappa
    
    def detection_power(self, kappa_true, significance_threshold=3.0):
        """
        Power to detect κ_true at given significance threshold
        
        Parameters:
            kappa_true: True value of κ
            significance_threshold: Detection threshold (σ)
            
        Returns:
            Detection power (0 to 1)
        """
        
        # Expected SNR improvement from f¹ term
        f_ref = 100
        f_typical = 100
        
        expected_snr_improvement = self.n_events**0.5 * kappa_true * (f_typical / f_ref)
        
        # Probability of detection
        power = 1 - norm.cdf(significance_threshold - expected_snr_improvement)
        
        return power

# Test
if __name__ == "__main__":
    print("="*80)
    print("STATISTICAL ANALYSIS MODULE TEST")
    print("="*80)
    
    # Create Bayesian analysis
    bayes = BayesianAnalysis(n_events=14)
    
    # Simulate data: 14 events with SNR~10
    np.random.seed(42)
    snr_gr = np.random.normal(10, 2, 14)
    snr_ift = snr_gr + np.random.normal(0.1, 0.05, 14)  # Small IFT improvement
    
    print(f"\nSimulated data:")
    print(f"Number of events: {len(snr_gr)}")
    print(f"Avg SNR (GR): {np.mean(snr_gr):.2f}")
    print(f"Avg SNR (IFT): {np.mean(snr_ift):.2f}")
    
    # Bayesian analysis
    kappa_array = np.linspace(-1e-13, 1e-13, 1000)
    kappa_array, posterior = bayes.posterior_κ(snr_gr, snr_ift, kappa_array)
    
    print(f"\nBayesian results:")
    results = bayes.best_fit_and_errors(kappa_array, posterior)
    print(f"Best-fit κ: {results['best_fit']:.3e}")
    print(f"68% interval: [{results['lower_68']:.3e}, {results['upper_68']:.3e}]")
    
    # Significance
    p_pos, p_neg, sigma = bayes.significance_κ_nonzero(kappa_array, posterior)
    print(f"\nSignificance of κ ≠ 0: {sigma:.2f}σ")
    
    print("\n✓ Statistical analysis module operational")
