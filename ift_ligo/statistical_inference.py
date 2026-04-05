"""
Bayesian Statistical Inference: κ Parameter Estimation
======================================================

Performs Bayesian inference to estimate the IFT coupling constant κ
from matched-filter data.

Prior: p(κ) = flat between [κ_min, κ_max]
Likelihood: p(d|κ) ∝ exp(-χ²(κ)/2)
Posterior: p(κ|d) ∝ p(d|κ) · p(κ)

Result: κ posterior distribution, credible intervals, point estimates
"""

import numpy as np
from scipy.stats import norm, lognorm, uniform
from scipy.optimize import minimize_scalar
from scipy.interpolate import interp1d
import warnings

class BayesianInference:
    """
    Bayesian inference for IFT coupling constant κ
    """
    
    def __init__(self, prior='flat', prior_range=(-1e-13, 1e-13)):
        """
        Initialize Bayesian inference
        
        Parameters:
            prior: 'flat', 'log-normal', or 'gaussian'
            prior_range: tuple (min, max) for prior support
        """
        
        self.prior_type = prior
        self.prior_range = prior_range
        self.kappa_values = None
        self.posterior = None
    
    def log_prior(self, kappa):
        """
        Log-prior for κ
        
        Options:
            - flat: uniform between prior_range
            - log_normal: favors values near 0
            - gaussian: peaked at κ=0
        """
        
        if self.prior_type == 'flat':
            if self.prior_range[0] <= kappa <= self.prior_range[1]:
                return 0  # log(constant)
            else:
                return -np.inf
        
        elif self.prior_type == 'gaussian':
            return -0.5 * (kappa / 1e-13)**2
        
        elif self.prior_type == 'log_normal':
            if kappa <= 0:
                return -np.inf
            return -np.log(kappa) - (np.log(kappa))**2 / 2
        
        else:
            return 0
    
    def log_likelihood(self, kappa, data, template_gr, template_ift_func, psd=None):
        """
        Log-likelihood for parameter κ
        
        -2 ln L = χ²(data | κ)
        
        Parameters:
            kappa: parameter value to test
            data: observed strain data
            template_gr: GR template (reference)
            template_ift_func: function to generate IFT template
            psd: power spectral density
            
        Returns:
            log-likelihood value
        """
        
        # Generate template for this κ
        template_ift = template_ift_func(kappa)
        
        # Frequency domain
        D = np.fft.rfft(data)
        H_gr = np.fft.rfft(template_gr)
        H_ift = np.fft.rfft(template_ift)
        
        # PSD normalization
        if psd is None:
            psd = np.ones_like(D)
        
        # Residuals
        residual_gr = np.abs(D - H_gr)**2 / psd
        residual_ift = np.abs(D - H_ift)**2 / psd
        
        # Chi-squared
        chi2_gr = np.sum(residual_gr)
        chi2_ift = np.sum(residual_ift)
        
        # Log-likelihood (delta chi-squared)
        log_L = -0.5 * (chi2_ift - chi2_gr)
        
        return log_L
    
    def compute_posterior(self, kappa_range, log_likelihood_values):
        """
        Compute posterior from likelihood
        
        p(κ|d) ∝ p(d|κ) · p(κ)
        
        Parameters:
            kappa_range: array of κ values
            log_likelihood_values: log-likelihood for each κ
            
        Returns:
            posterior array (normalized)
        """
        
        # Add prior
        log_prior_values = np.array([self.log_prior(k) for k in kappa_range])
        
        # Unnormalized posterior
        log_posterior = log_likelihood_values + log_prior_values
        
        # Normalize
        log_posterior -= np.max(log_posterior)
        posterior = np.exp(log_posterior)
        posterior /= np.trapz(posterior, kappa_range)
        
        self.kappa_values = kappa_range
        self.posterior = posterior
        
        return posterior
    
    def credible_interval(self, credibility=0.68):
        """
        Compute credible interval for κ
        
        Find interval containing 'credibility' of posterior mass
        
        Parameters:
            credibility: 0.68 (1σ), 0.95 (2σ), 0.99 (3σ)
            
        Returns:
            (κ_min, κ_max) credible interval
        """
        
        if self.posterior is None:
            raise ValueError("Must compute posterior first")
        
        # Cumulative distribution
        cdf = np.cumsum(self.posterior)
        cdf /= cdf[-1]
        
        # Find interval
        alpha = (1 - credibility) / 2
        idx_min = np.searchsorted(cdf, alpha)
        idx_max = np.searchsorted(cdf, 1 - alpha)
        
        kappa_min = self.kappa_values[idx_min]
        kappa_max = self.kappa_values[idx_max]
        
        return kappa_min, kappa_max
    
    def point_estimate(self, method='map'):
        """
        Get point estimate for κ
        
        Methods:
            - 'map': Maximum A Posteriori (most probable)
            - 'mean': Posterior mean
            - 'median': Posterior median
            
        Parameters:
            method: estimation method
            
        Returns:
            κ estimate
        """
        
        if self.posterior is None:
            raise ValueError("Must compute posterior first")
        
        if method == 'map':
            idx_max = np.argmax(self.posterior)
            return self.kappa_values[idx_max]
        
        elif method == 'mean':
            return np.trapz(self.kappa_values * self.posterior, self.kappa_values)
        
        elif method == 'median':
            cdf = np.cumsum(self.posterior)
            cdf /= cdf[-1]
            idx_median = np.searchsorted(cdf, 0.5)
            return self.kappa_values[idx_median]
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def summary(self):
        """Print summary of inference results"""
        
        if self.posterior is None:
            print("No posterior available - run compute_posterior first")
            return
        
        print("\n" + "="*80)
        print("BAYESIAN INFERENCE RESULTS: κ PARAMETER")
        print("="*80)
        
        # Point estimates
        kappa_map = self.point_estimate('map')
        kappa_mean = self.point_estimate('mean')
        kappa_median = self.point_estimate('median')
        
        print(f"\nPoint Estimates:")
        print(f"  MAP (Most Probable):     κ = {kappa_map:.3e}")
        print(f"  Mean:                    κ = {kappa_mean:.3e}")
        print(f"  Median:                  κ = {kappa_median:.3e}")
        
        # Credible intervals
        ci_68 = self.credible_interval(0.68)
        ci_95 = self.credible_interval(0.95)
        
        print(f"\nCredible Intervals:")
        print(f"  68% (1σ): [{ci_68[0]:.3e}, {ci_68[1]:.3e}]")
        print(f"  95% (2σ): [{ci_95[0]:.3e}, {ci_95[1]:.3e}]")
        
        # LIGO constraint
        kappa_ligo_limit = 4.9e-14
        if ci_95[1] < kappa_ligo_limit:
            print(f"\n✓ 2σ upper limit is consistent with LIGO constraint κ < {kappa_ligo_limit:.1e}")
        else:
            print(f"\n⚠ Upper limit exceeds LIGO constraint")

class LikelihoodRatio:
    """
    Likelihood ratio test: f¹ signature vs GR
    
    H0: κ = 0 (General Relativity)
    H1: κ ≠ 0 (IFT with f¹)
    
    Test statistic: Λ = L(H1) / L(H0)
    Log test statistic: ln Λ = ln L(H1) - ln L(H0)
    """
    
    def __init__(self):
        self.log_lambda_values = []
        self.event_names = []
    
    def compute_ratio(self, data, template_gr, template_ift, psd=None):
        """
        Compute likelihood ratio for single event
        
        Parameters:
            data: strain data
            template_gr: GR template
            template_ift: IFT template (κ ≠ 0)
            psd: power spectral density
            
        Returns:
            log Λ value
        """
        
        # Frequency domain
        D = np.fft.rfft(data)
        H_gr = np.fft.rfft(template_gr)
        H_ift = np.fft.rfft(template_ift)
        
        # PSD normalization
        if psd is None:
            psd = np.ones_like(D)
        
        # Residuals
        residual_gr = np.abs(D - H_gr)**2 / psd
        residual_ift = np.abs(D - H_ift)**2 / psd
        
        # Chi-squared
        chi2_gr = np.sum(residual_gr)
        chi2_ift = np.sum(residual_ift)
        
        # Log likelihood ratio
        log_lambda = -0.5 * (chi2_ift - chi2_gr)
        
        return log_lambda
    
    def compute_from_catalog(self, event_list, template_func_gr, template_func_ift):
        """
        Compute likelihood ratios for all events
        
        Parameters:
            event_list: list of GWEvent objects
            template_func_gr: function(event) -> GR template
            template_func_ift: function(event) -> IFT template
            
        Returns:
            log likelihood ratio array
        """
        
        log_lambdas = []
        
        for event in event_list:
            # Generate templates
            template_gr = template_func_gr(event)
            template_ift = template_func_ift(event)
            
            # Simulated data (template + noise)
            # In real analysis, this would be actual LIGO data
            np.random.seed(hash(event.name) % 2**32)
            h_data = template_gr  # Assume data matches template
            noise = np.random.randn(len(h_data)) * 1e-20
            data = h_data + noise
            
            # Likelihood ratio
            log_lambda = self.compute_ratio(data, template_gr, template_ift)
            log_lambdas.append(log_lambda)
            self.event_names.append(event.name)
        
        self.log_lambda_values = np.array(log_lambdas)
        return self.log_lambda_values
    
    def significance_per_event(self):
        """
        Compute significance per event
        
        Δχ² = 2 ln Λ
        
        Returns:
            Dictionary with per-event significances
        """
        
        if not self.log_lambda_values:
            raise ValueError("Must compute ratios first")
        
        delta_chi2 = 2 * self.log_lambda_values
        
        # Convert to sigma
        sigmas = np.sqrt(np.abs(delta_chi2))
        
        results = {}
        for name, dc2, sig in zip(self.event_names, delta_chi2, sigmas):
            results[name] = {
                'log_Lambda': self.log_lambda_values[len(results)],
                'delta_chi2': dc2,
                'sigma': sig if dc2 > 0 else -sig,
                'interpretation': self._interpret(dc2)
            }
        
        return results
    
    def combined_evidence(self):
        """
        Combine evidence across all events
        
        Total log-likelihood: Σ ln L(κ|event_i)
        
        Returns:
            Combined log likelihood and interpretation
        """
        
        total_log_lambda = np.sum(self.log_lambda_values)
        total_delta_chi2 = 2 * total_log_lambda
        
        print("\n" + "="*80)
        print("COMBINED LIKELIHOOD RATIO ACROSS ALL EVENTS")
        print("="*80)
        
        print(f"\nTotal log Λ: {total_log_lambda:.2f}")
        print(f"Total Δχ²: {total_delta_chi2:.2f}")
        print(f"Combined Significance: {np.sqrt(np.abs(total_delta_chi2)):.2f}σ")
        
        if total_delta_chi2 > 25:
            print("\n✓ 5σ DETECTION of f¹ signature!")
        elif total_delta_chi2 > 9:
            print("\n✓ 3σ STRONG EVIDENCE for f¹ signature")
        elif total_delta_chi2 > 0:
            print("\n⚠ Marginal evidence for f¹ signature (need more data)")
        else:
            print("\n✗ GR is preferred (no evidence for f¹)")
        
        return total_log_lambda, total_delta_chi2
    
    def _interpret(self, delta_chi2):
        """Interpret Δχ² value"""
        if delta_chi2 > 25:
            return "5σ"
        elif delta_chi2 > 9:
            return "3σ"
        elif delta_chi2 > 4:
            return "2σ"
        elif delta_chi2 > 0:
            return "<2σ"
        else:
            return "GR preferred"

# Test
if __name__ == "__main__":
    print("="*80)
    print("STATISTICAL INFERENCE TEST")
    print("="*80)
    
    # Bayesian inference test
    bayes = BayesianInference(prior='flat', prior_range=(-1e-13, 1e-13))
    
    # Simulated likelihood
    kappas = np.linspace(-1e-13, 1e-13, 100)
    log_likes = -0.5 * ((kappas - 0.5e-14) / 0.2e-14)**2
    
    posterior = bayes.compute_posterior(kappas, log_likes)
    
    print(f"\n✓ Bayesian inference completed")
    bayes.summary()
    
    # Likelihood ratio test
    lr = LikelihoodRatio()
    print(f"\n✓ Likelihood ratio analyzer created")
    
    print("\n" + "="*80)
    print("✓ Statistical Inference Module Operational")
    print("="*80)
