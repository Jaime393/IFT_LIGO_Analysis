"""
Gravitational Wave Template Module
==================================

Waveform templates for General Relativity and IFT
for matched-filter searches
"""

import numpy as np
from scipy.integrate import odeint
from scipy.interpolate import interp1d
import warnings

class GravitationalWaveTemplate:
    """
    Generate gravitational wave templates for matched-filter analysis
    """
    
    def __init__(self):
        """Initialize constants"""
        self.c = 299792458  # m/s
        self.G = 6.67430e-11  # m³/(kg·s²)
        self.M_sun = 1.989e30  # kg
        self.pc = 3.086e16  # m (parsec)
        
    def chirp_mass_from_masses(self, m1, m2):
        """
        Calculate chirp mass from component masses
        
        M_c = (m1 * m2)^(3/5) / (m1 + m2)^(1/5)
        
        Parameters:
            m1, m2: Component masses (solar masses)
            
        Returns:
            Chirp mass (solar masses)
        """
        m1_kg = m1 * self.M_sun
        m2_kg = m2 * self.M_sun
        
        M_c = ((m1_kg * m2_kg)**(3/5)) / ((m1_kg + m2_kg)**(1/5))
        
        return M_c / self.M_sun
    
    def symmetric_mass_ratio(self, m1, m2):
        """
        Symmetric mass ratio η = m1*m2 / (m1+m2)²
        """
        M = m1 + m2
        return (m1 * m2) / (M**2)
    
    def frequency_evolution_gr(self, f_start, m1, m2, t_length=1.0):
        """
        Frequency evolution under General Relativity
        
        df/dt = (96π/5) * (G M_c / c³)^(5/3) * f^(11/3)
        
        Parameters:
            f_start: Starting frequency (Hz)
            m1, m2: Component masses (solar masses)
            t_length: Time duration (seconds)
            
        Returns:
            Time array, frequency array
        """
        
        # Chirp mass in SI units
        M_c = self.chirp_mass_from_masses(m1, m2)
        M_c_kg = M_c * self.M_sun
        
        # GR chirp rate
        K_gr = (96 * np.pi / 5) * ((self.G * M_c_kg / self.c**3)**(5/3))
        
        # Solve ODE: df/dt = K * f^(11/3)
        def df_dt(f, t):
            if f > 0:
                return K_gr * (f**(11/3))
            return 0
        
        # Time array
        t = np.linspace(0, t_length, 10000)
        
        # Solve
        f_evolution = odeint(df_dt, f_start, t)[:, 0]
        
        # Clip at merger frequency (approximation)
        f_merger = 4400 / (m1 + m2)  # Approximate merger frequency
        f_evolution = np.minimum(f_evolution, f_merger)
        
        return t, f_evolution
    
    def frequency_evolution_ift(self, f_start, m1, m2, kappa=1.8e-14, t_length=1.0):
        """
        Frequency evolution with IFT f¹ correction
        
        df/dt = K_GR * f^(11/3) * [1 + κ(f/f_ref)¹]
        
        Parameters:
            f_start: Starting frequency (Hz)
            m1, m2: Component masses (solar masses)
            kappa: IFT coupling constant
            t_length: Time duration (seconds)
            
        Returns:
            Time array, frequency array
        """
        
        # Chirp mass
        M_c = self.chirp_mass_from_masses(m1, m2)
        M_c_kg = M_c * self.M_sun
        
        # GR chirp rate
        K_gr = (96 * np.pi / 5) * ((self.G * M_c_kg / self.c**3)**(5/3))
        
        f_ref = 100  # Reference frequency
        
        # Solve ODE with IFT correction
        def df_dt(f, t):
            if f > 0:
                gr_term = K_gr * (f**(11/3))
                correction = 1 + kappa * (f / f_ref)**1
                return gr_term * correction
            return 0
        
        # Time array
        t = np.linspace(0, t_length, 10000)
        
        # Solve
        f_evolution = odeint(df_dt, f_start, t)[:, 0]
        
        # Clip at merger
        f_merger = 4400 / (m1 + m2)
        f_evolution = np.minimum(f_evolution, f_merger)
        
        return t, f_evolution
    
    def strain_gr(self, m1, m2, distance_mpc=100, f_start=20, duration=1.0):
        """
        Generate strain waveform for binary merger (General Relativity)
        
        Simple approximation: amplitude * cos(2π ∫ f(t) dt)
        
        Parameters:
            m1, m2: Component masses (solar masses)
            distance_mpc: Distance (megaparsecs)
            f_start: Starting frequency (Hz)
            duration: Duration (seconds)
            
        Returns:
            Time array, strain array
        """
        
        # Frequency evolution
        t, f = self.frequency_evolution_gr(f_start, m1, m2, duration)
        
        # Amplitude scaling with distance
        # A ~ G * M_c^(5/3) / (c² * d) * f^(2/3)
        M_c = self.chirp_mass_from_masses(m1, m2)
        distance = distance_mpc * self.pc * 1e6  # Convert to meters
        
        A_scale = (self.G * (M_c * self.M_sun)**(5/3)) / (self.c**2 * distance)
        amplitude = A_scale * (f**(2/3))
        
        # Phase: Φ(t) = 2π ∫ f(t') dt'
        phase = 2 * np.pi * np.cumsum(f) * np.gradient(t)
        
        # Strain (simplified)
        h = amplitude * np.cos(phase)
        
        return t, h
    
    def strain_ift(self, m1, m2, distance_mpc=100, f_start=20, 
                   kappa=1.8e-14, duration=1.0):
        """
        Generate strain waveform with IFT f¹ correction
        
        Parameters:
            m1, m2: Component masses (solar masses)
            distance_mpc: Distance (megaparsecs)
            f_start: Starting frequency (Hz)
            kappa: IFT coupling constant
            duration: Duration (seconds)
            
        Returns:
            Time array, strain array
        """
        
        # Frequency evolution with IFT
        t, f = self.frequency_evolution_ift(f_start, m1, m2, kappa, duration)
        
        # Amplitude
        M_c = self.chirp_mass_from_masses(m1, m2)
        distance = distance_mpc * self.pc * 1e6
        
        A_scale = (self.G * (M_c * self.M_sun)**(5/3)) / (self.c**2 * distance)
        amplitude = A_scale * (f**(2/3))
        
        # Phase with IFT
        phase = 2 * np.pi * np.cumsum(f) * np.gradient(t)
        
        # Strain
        h = amplitude * np.cos(phase)
        
        return t, h
    
    def frequency_domain_template_gr(self, f_array, m1, m2, distance_mpc=100):
        """
        Frequency domain waveform (Fourier transform of strain)
        
        Uses TaylorF2 approximation for speed
        
        Parameters:
            f_array: Frequency array (Hz)
            m1, m2: Component masses (solar masses)
            distance_mpc: Distance (megaparsecs)
            
        Returns:
            Complex amplitude array
        """
        
        # Chirp mass
        M_c = self.chirp_mass_from_masses(m1, m2)
        eta = self.symmetric_mass_ratio(m1, m2)
        
        # TaylorF2 amplitude
        A0 = (1/distance_mpc) * np.sqrt((5 * eta) / (96 * np.pi**(8/3)))
        
        # Phase (TaylorF2)
        f_0 = 100  # Reference frequency
        psi = 2 * np.pi * f_array**(-5/3) * (3/256) * ((self.G * M_c * self.M_sun / self.c**3)**(-5/3))
        
        # Amplitude modulation
        amplitude = A0 * (f_array / f_0)**(-7/6)
        
        # Complex template
        h_f = amplitude * np.exp(1j * psi)
        
        return h_f
    
    def frequency_domain_template_ift(self, f_array, m1, m2, 
                                      distance_mpc=100, kappa=1.8e-14):
        """
        Frequency domain template with IFT f¹ correction
        
        H(f) = H_GR(f) * exp(iδΦ(f))
        
        Where δΦ ∝ κ(f/f_ref)¹
        """
        
        # GR template
        h_gr = self.frequency_domain_template_gr(f_array, m1, m2, distance_mpc)
        
        # IFT phase correction
        f_ref = 100
        delta_phase = 2 * np.pi * kappa * (f_array / f_ref)**1
        
        # Apply correction
        h_ift = h_gr * np.exp(1j * delta_phase)
        
        return h_ift
    
    def amplitude_correction_factor(self, f, kappa, f_ref=100):
        """
        Amplitude correction from IFT
        
        h(f) ∝ h_GR(f) * [1 + κ(f/f_ref)¹]
        
        Parameters:
            f: Frequency (Hz)
            kappa: Coupling constant
            f_ref: Reference frequency
            
        Returns:
            Correction factor
        """
        
        correction = 1 + kappa * (f / f_ref)**1
        
        return correction

class WaveformComparison:
    """
    Compare GR and IFT waveforms
    """
    
    def __init__(self):
        self.template = GravitationalWaveTemplate()
    
    def gr_vs_ift_frequency_evolution(self, m1, m2, kappa=1.8e-14, 
                                      f_start=20, duration=1.0):
        """
        Compare frequency evolution: GR vs IFT
        """
        
        t_gr, f_gr = self.template.frequency_evolution_gr(f_start, m1, m2, duration)
        t_ift, f_ift = self.template.frequency_evolution_ift(
            f_start, m1, m2, kappa, duration
        )
        
        return t_gr, f_gr, t_ift, f_ift
    
    def frequency_difference(self, m1, m2, kappa=1.8e-14, f_start=20):
        """
        Frequency difference between IFT and GR
        
        ΔF / F_GR = [κ(f/f_ref)¹] / [f^(11/3) term]
        """
        
        # At reference frequency
        f_ref = 100
        
        # Relative difference
        rel_diff = kappa * (f_ref / f_ref)**1
        
        return rel_diff
    
    def merger_time_difference(self, m1, m2, kappa=1.8e-14):
        """
        Time-to-merger difference: IFT vs GR
        """
        
        t_gr, f_gr, t_ift, f_ift = self.gr_vs_ift_frequency_evolution(
            m1, m2, kappa
        )
        
        # Time when frequency reaches 1000 Hz (near merger)
        f_merger = 1000
        
        idx_gr = np.argmin(np.abs(f_gr - f_merger))
        idx_ift = np.argmin(np.abs(f_ift - f_merger))
        
        t_merger_gr = t_gr[idx_gr]
        t_merger_ift = t_ift[idx_ift]
        
        delta_t = t_merger_ift - t_merger_gr
        
        return delta_t, t_merger_gr, t_merger_ift

# Test
if __name__ == "__main__":
    print("="*80)
    print("GRAVITATIONAL WAVE TEMPLATE MODULE TEST")
    print("="*80)
    
    # Create template
    template = GravitationalWaveTemplate()
    
    # Example: GW150914-like event
    m1 = 36  # Solar masses
    m2 = 29
    distance = 410  # Megaparsecs
    
    print(f"\nExample: Binary black hole merger")
    print(f"Masses: {m1} M☉ + {m2} M☉")
    print(f"Distance: {distance} Mpc")
    
    # Chirp mass
    M_c = template.chirp_mass_from_masses(m1, m2)
    print(f"\nChirp mass: {M_c:.2f} M☉")
    
    # Frequency evolution
    t_gr, f_gr = template.frequency_evolution_gr(20, m1, m2, 1.0)
    t_ift, f_ift = template.frequency_evolution_ift(20, m1, m2, 1.8e-14, 1.0)
    
    print(f"\nGR evolution: f(0) = {f_gr[0]:.1f} Hz → f(1s) = {f_gr[-1]:.1f} Hz")
    print(f"IFT evolution: f(0) = {f_ift[0]:.1f} Hz → f(1s) = {f_ift[-1]:.1f} Hz")
    print(f"Difference: {(f_ift[-1] - f_gr[-1]) / f_gr[-1] * 100:.3f}%")
    
    print("\n✓ Template module operational")
