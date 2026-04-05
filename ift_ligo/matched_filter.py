"""
Matched Filter Analysis Module
==============================

Optimal filter for gravitational wave detection and κ estimation
"""

import numpy as np
from scipy.signal import correlate, get_window
from scipy.fft import fft, ifft, rfft, irfft
from scipy import stats
import warnings

class MatchedFilter:
    """
    Matched filter implementation for gravitational wave analysis
    """
    
    def __init__(self, sample_rate=16384, f_min=20, f_max=8000):
        """
        Initialize matched filter
        
        Parameters:
            sample_rate: Sampling rate (Hz)
            f_min: Minimum frequency (Hz)
            f_max: Maximum frequency (Hz)
        """
        
        self.sample_rate = sample_rate
        self.f_min = f_min
        self.f_max = f_max
        self.dt = 1.0 / sample_rate
    
    def whiten_data(self, strain, segment_length=4):
        """
        Whiten strain data to make noise spectrum flat
        
        Parameters:
            strain: Time-domain strain data
            segment_length: Whitening segment length (seconds)
            
        Returns:
            Whitened strain data
        """
        
        # Compute power spectral density
        n_samples = len(strain)
        n_segments = int(len(strain) / (segment_length * self.sample_rate))
        
        # Simple whitening: divide by amplitude in frequency domain
        strain_fft = rfft(strain)
        frequencies = np.fft.rfftfreq(n_samples, self.dt)
        
        # Power spectral density
        psd = np.abs(strain_fft)**2
        psd = np.maximum(psd, 1e-30)  # Avoid division by zero
        
        # Smooth PSD
        from scipy.ndimage import uniform_filter1d
        psd_smooth = uniform_filter1d(psd, size=100)
        
        # Whiten
        strain_fft_white = strain_fft / np.sqrt(psd_smooth)
        
        # Back to time domain
        strain_white = irfft(strain_fft_white, n=n_samples)
        
        # Normalize
        strain_white = strain_white / np.std(strain_white)
        
        return strain_white, frequencies, psd_smooth
    
    def inner_product(self, data1, data2, psd=None):
        """
        Optimal inner product accounting for noise PSD
        
        <d|h> = ∫ d(f) h*(f) / S_n(f) df
        
        Parameters:
            data1, data2: Time domain signals or frequency domain
            psd: Power spectral density
            
        Returns:
            Inner product (scalar)
        """
        
        # Fourier transform
        data1_fft = rfft(data1)
        data2_fft = rfft(data2)
        
        if psd is None:
            # Simple inner product
            ip = np.sum(np.real(np.conj(data1_fft) * data2_fft))
        else:
            # Weighted by 1/PSD
            ip = np.sum(np.real(np.conj(data1_fft) * data2_fft) / psd)
        
        return ip / len(data1)
    
    def matched_filter_snr(self, strain, template, psd=None):
        """
        Compute matched filter SNR
        
        SNR(t) = |d(t) * h(t)| / sqrt(<h|h>)
        
        Parameters:
            strain: Strain data (time domain)
            template: Template waveform (time domain)
            psd: Power spectral density
            
        Returns:
            SNR time series, SNR maximum
        """
        
        # Normalize template by its inner product with itself
        template_norm = np.sqrt(self.inner_product(template, template, psd))
        template_normalized = template / (template_norm + 1e-30)
        
        # Matched filter = correlation with normalized template
        if psd is None:
            # Simple correlation
            snr = correlate(strain, template_normalized, mode='same')
        else:
            # Frequency domain filtering
            strain_fft = rfft(strain)
            template_fft = rfft(template_normalized)
            
            # Apply 1/PSD weighting
            filtered = strain_fft * np.conj(template_fft) / (psd + 1e-30)
            snr = np.abs(irfft(filtered, n=len(strain)))
        
        # Normalize
        snr = snr / (np.std(strain) + 1e-30)
        
        snr_max = np.max(snr)
        snr_time = np.argmax(snr) * self.dt
        
        return snr, snr_max, snr_time
    
    def bayesian_κ_estimation(self, strain, template_gr, template_ift_func, 
                              kappa_range=(-1e-13, 1e-13), n_samples=1000):
        """
        Bayesian inference to estimate κ parameter
        
        Compute likelihood ratio: L(κ) = P(data | κ) / P(data | κ=0)
        
        Parameters:
            strain: Strain data
            template_gr: GR template (κ=0)
            template_ift_func: Function to generate IFT templates
            kappa_range: Range of κ values
            n_samples: Number of κ samples
            
        Returns:
            κ array, likelihood array, posterior array
        """
        
        kappa_array = np.linspace(kappa_range[0], kappa_range[1], n_samples)
        
        # Compute likelihood for each κ
        likelihood = np.zeros(n_samples)
        
        for i, kappa in enumerate(kappa_array):
            # Generate IFT template
            template_ift = template_ift_func(kappa)
            
            # SNR for GR
            snr_gr, _, _ = self.matched_filter_snr(strain, template_gr)
            snr_gr_max = np.max(snr_gr)
            
            # SNR for IFT
            snr_ift, _, _ = self.matched_filter_snr(strain, template_ift)
            snr_ift_max = np.max(snr_ift)
            
            # Likelihood ratio (simplified)
            # L ∝ exp(SNR_IFT² / 2 - SNR_GR² / 2)
            delta_snr_sq = (snr_ift_max**2 - snr_gr_max**2) / 2
            likelihood[i] = np.exp(np.clip(delta_snr_sq, -100, 100))
        
        # Normalize to posterior
        posterior = likelihood / np.sum(likelihood)
        
        return kappa_array, likelihood, posterior
    
    def confidence_interval(self, kappa_array, posterior, confidence=0.68):
        """
        Extract confidence interval from posterior
        
        Parameters:
            kappa_array: Parameter array
            posterior: Posterior probability distribution
            confidence: Confidence level (default 68% = 1σ)
            
        Returns:
            Best-fit κ, lower bound, upper bound
        """
        
        # Best fit (maximum posterior)
        idx_best = np.argmax(posterior)
        kappa_best = kappa_array[idx_best]
        
        # Cumulative distribution
        cdf = np.cumsum(posterior)
        cdf = cdf / cdf[-1]
        
        # Confidence interval
        alpha = (1 - confidence) / 2
        idx_lower = np.argmin(np.abs(cdf - alpha))
        idx_upper = np.argmin(np.abs(cdf - (1 - alpha)))
        
        kappa_lower = kappa_array[idx_lower]
        kappa_upper = kappa_array[idx_upper]
        
        return kappa_best, kappa_lower, kappa_upper
    
    def false_alarm_rate(self, snr_array, snr_threshold=5.0):
        """
        Estimate false alarm rate from noise-only data
        
        Parameters:
            snr_array: SNR values from noise data
            snr_threshold: SNR threshold
            
        Returns:
            False alarm rate (false alarms per unit time)
        """
        
        n_above_threshold = np.sum(snr_array > snr_threshold)
        duration = len(snr_array) * self.dt
        
        far = n_above_threshold / duration
        
        return far
    
    def significance_test(self, snr_signal, snr_noise_mean=0, snr_noise_std=1):
        """
        Calculate detection significance (σ level)
        
        Parameters:
            snr_signal: SNR of detected signal
            snr_noise_mean: Mean SNR of noise
            snr_noise_std: Std dev of noise SNR
            
        Returns:
            Significance in σ
        """
        
        significance = (snr_signal - snr_noise_mean) / snr_noise_std
        
        return significance

class SignalInjectionStudy:
    """
    Injection study to validate detection pipeline
    """
    
    def __init__(self, sample_rate=16384):
        self.matched_filter = MatchedFilter(sample_rate=sample_rate)
        self.sample_rate = sample_rate
        self.dt = 1.0 / sample_rate
    
    def add_signal_to_noise(self, noise, signal, snr_target=10.0):
        """
        Add signal to noise at target SNR
        
        Parameters:
            noise: Noise time series
            signal: Signal time series
            snr_target: Target SNR
            
        Returns:
            Data with signal (noise + injected signal)
        """
        
        # Scale signal to achieve target SNR
        signal_power = np.mean(signal**2)
        noise_power = np.mean(noise**2)
        
        scale = snr_target * np.sqrt(noise_power / signal_power)
        
        data_with_signal = noise + scale * signal
        
        return data_with_signal
    
    def injection_recovery_test(self, noise_data, signal_template, 
                                snr_levels=np.linspace(5, 50, 10)):
        """
        Test signal recovery at various SNR levels
        
        Parameters:
            noise_data: Noise time series
            signal_template: Signal template
            snr_levels: SNR levels to test
            
        Returns:
            Recovery efficiency vs SNR
        """
        
        recovery_rate = np.zeros_like(snr_levels)
        
        for i, snr_target in enumerate(snr_levels):
            # Inject signal
            data = self.add_signal_to_noise(noise_data, signal_template, snr_target)
            
            # Matched filter
            snr, snr_max, _ = self.matched_filter.matched_filter_snr(
                data, signal_template
            )
            
            # Recovery: if max SNR > 5, signal detected
            if snr_max > 5.0:
                recovery_rate[i] = 1.0
            else:
                recovery_rate[i] = 0.0
        
        return snr_levels, recovery_rate

# Test
if __name__ == "__main__":
    print("="*80)
    print("MATCHED FILTER ANALYSIS MODULE TEST")
    print("="*80)
    
    # Create matched filter
    mf = MatchedFilter()
    
    # Generate test data
    duration = 1.0  # seconds
    n_samples = int(duration * mf.sample_rate)
    t = np.arange(n_samples) * mf.dt
    
    # Simple template (sine wave)
    f_signal = 100  # Hz
    template = np.sin(2 * np.pi * f_signal * t)
    
    # Noise
    noise = np.random.normal(0, 0.1, n_samples)
    
    # Data with signal
    data = noise + 0.5 * template
    
    print(f"\nTest setup:")
    print(f"Duration: {duration} s")
    print(f"Sample rate: {mf.sample_rate} Hz")
    print(f"Signal frequency: {f_signal} Hz")
    
    # Matched filter
    snr, snr_max, snr_time = mf.matched_filter_snr(data, template)
    
    print(f"\nMatched filter results:")
    print(f"Max SNR: {snr_max:.2f}")
    print(f"Time of max SNR: {snr_time:.4f} s")
    
    print("\n✓ Matched filter module operational")
