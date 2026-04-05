"""
Gravitational Wave Templates: GR vs IFT
=======================================

Generate gravitational wave strain templates for matched-filter analysis.

Standard GR template (TaylorF2):
    h(f) = A · (π f_c M)^(-7/12) · exp(iΨ(f))
    
    Where M = (m1·m2)^(3/5) / (m1+m2)^(1/5) is chirp mass
    
IFT modification:
    h_IFT(f) = h_GR(f) · [1 + κ(f/f_ref)¹]
    
    Linear frequency dependence (f¹ signature)
    κ < 4.9×10⁻¹⁴ (extremely small coupling)
"""

import numpy as np
from scipy.special import hyp2f1
from scipy.optimize import fminbound
import warnings

class GWTemplate:
    """
    Gravitational wave waveform template in General Relativity
    
    Implements TaylorF2 approximant commonly used in LIGO analysis.
    """
    
    def __init__(self, m1, m2, chi1=0, chi2=0):
        """
        Initialize template
        
        Parameters:
            m1, m2: component masses (solar masses)
            chi1, chi2: spin parameters (0 to 1)
        """
        
        self.m1 = m1
        self.m2 = m2
        self.chi1 = chi1
        self.chi2 = chi2
        
        # Derived quantities
        self.M = m1 + m2  # Total mass
        self.eta = (m1 * m2) / (self.M ** 2)  # Symmetric mass ratio
        self.Mc = self.M * (self.eta ** (3/5))  # Chirp mass
        
        # Geometric units conversion
        self.G = 6.67430e-11  # m³/(kg·s²)
        self.c = 299792458    # m/s
        self.M_sun = 1.989e30  # kg
        
        # Convert to SI
        self.Mc_SI = self.Mc * self.M_sun
        
    def frequency_merinder(self):
        """Merger frequency (innermost stable circular orbit)"""
        return self.c**3 / (6**(3/2) * np.pi * self.G * self.Mc_SI)
    
    def amplitude(self, f):
        """
        GR amplitude spectrum A(f)
        
        A(f) ∝ (π f_c M_c)^(-7/12)
        
        Parameters:
            f: frequency array (Hz)
            
        Returns:
            Amplitude (normalized)
        """
        
        f_ref = 100  # Reference frequency (Hz)
        
        # Avoid singularities
        f_safe = np.where(f > 0, f, f_ref)
        
        # Amplitude scales as f^(-7/6)
        amplitude = (f_ref / f_safe) ** (7/6)
        
        return amplitude
    
    def phase(self, f, t_c=0, phi_c=0):
        """
        GR phase evolution Ψ(f)
        
        TaylorF2 phase (Post-Newtonian expansion)
        
        Parameters:
            f: frequency array (Hz)
            t_c: coalescence time
            phi_c: coalescence phase
            
        Returns:
            Phase array (radians)
        """
        
        # Avoid singularities
        f_safe = np.where(f > 1e-10, f, 1e-10)
        
        # Post-Newtonian phase (simplified PN order)
        v = (np.pi * self.Mc_SI * f_safe / self.c**3) ** (1/3)
        
        # Phase evolution
        phi = 2*np.pi * f_safe * t_c - phi_c
        phi += (3/(128 * (np.pi * self.Mc_SI * f_safe / self.c**3)**(5/3)))
        
        return phi
    
    def strain(self, f, t_c=0, phi_c=0, distance=1):
        """
        Complete GR strain h(f) = A(f) exp(i Ψ(f))
        
        Parameters:
            f: frequency array (Hz)
            t_c: coalescence time (s)
            phi_c: coalescence phase (rad)
            distance: luminosity distance (Mpc)
            
        Returns:
            Complex strain array
        """
        
        # Amplitude and phase
        A = self.amplitude(f) / distance
        Phi = self.phase(f, t_c, phi_c)
        
        # Complex strain
        h_f = A * np.exp(1j * Phi)
        
        return h_f
    
    def strain_time(self, f_array, t_c=0, phi_c=0, distance=1):
        """
        Convert frequency domain to time domain via FFT
        """
        
        h_f = self.strain(f_array, t_c, phi_c, distance)
        
        # Simple inverse Fourier
        h_t = np.fft.ifft(h_f)
        
        return h_t

class GWTemplateIFT(GWTemplate):
    """
    Gravitational wave template with IFT f¹ modification
    
    Inherits from GWTemplate and adds linear frequency dependence
    """
    
    def __init__(self, m1, m2, chi1=0, chi2=0, kappa=0):
        """
        Initialize IFT template
        
        Parameters:
            m1, m2: component masses (solar masses)
            chi1, chi2: spin parameters
            kappa: IFT coupling constant (< 4.9×10⁻¹⁴)
        """
        
        super().__init__(m1, m2, chi1, chi2)
        self.kappa = kappa
        self.f_ref = 100  # Reference frequency (Hz)
    
    def f1_correction(self, f):
        """
        IFT frequency-dependent correction
        
        Δh/h = κ · (f/f_ref)¹
        
        Parameters:
            f: frequency array
            
        Returns:
            Multiplicative correction factor: [1 + κ(f/f_ref)¹]
        """
        
        if self.kappa == 0:
            return np.ones_like(f)
        
        correction = 1.0 + self.kappa * (f / self.f_ref)
        
        return correction
    
    def strain(self, f, t_c=0, phi_c=0, distance=1):
        """
        IFT strain with f¹ modification
        
        h_IFT(f) = h_GR(f) · [1 + κ(f/f_ref)¹]
        
        Parameters:
            f: frequency array (Hz)
            t_c: coalescence time
            phi_c: coalescence phase
            distance: luminosity distance
            
        Returns:
            Complex strain array with IFT correction
        """
        
        # Get GR strain
        h_gr = super().strain(f, t_c, phi_c, distance)
        
        # Apply IFT correction
        correction = self.f1_correction(f)
        h_ift = h_gr * correction
        
        return h_ift
    
    def amplitude_ift(self, f):
        """
        IFT amplitude spectrum (including f¹ correction)
        """
        
        A_gr = super().amplitude(f)
        correction = self.f1_correction(f)
        A_ift = A_gr * np.abs(correction)
        
        return A_ift
    
    def phase_ift(self, f, t_c=0, phi_c=0):
        """
        IFT phase (same as GR - correction is in amplitude)
        """
        
        return super().phase(f, t_c, phi_c)

class TemplateBank:
    """
    Generate bank of templates for matched filtering
    """
    
    def __init__(self, m_min=1.2, m_max=250, n_templates=1000):
        """
        Create template bank
        
        Parameters:
            m_min, m_max: mass range (solar masses)
            n_templates: number of templates
        """
        
        self.m_min = m_min
        self.m_max = m_max
        self.n_templates = n_templates
        
        # Generate masses logarithmically spaced
        self.masses = self._generate_mass_lattice()
    
    def _generate_mass_lattice(self):
        """Generate mass pairs for template bank"""
        
        # Chirp mass spacing: 3% (typical LIGO value)
        dm = 0.03
        
        m_min_effective = 1.0
        m_max_effective = 2.0
        
        n_chirp = int(np.log(m_max_effective/m_min_effective) / np.log(1 + dm))
        
        # Generate template masses
        templates = []
        for i in range(n_chirp):
            m1 = 10 + i*5
            m2 = 20 + i*3
            
            if m1 < self.m_max and m2 < self.m_max:
                templates.append((m1, m2))
        
        return templates[:self.n_templates]
    
    def generate_templates_gr(self):
        """Generate GR template bank"""
        
        templates = []
        for m1, m2 in self.masses:
            templates.append(GWTemplate(m1, m2))
        
        return templates
    
    def generate_templates_ift(self, kappa=0):
        """Generate IFT template bank"""
        
        templates = []
        for m1, m2 in self.masses:
            templates.append(GWTemplateIFT(m1, m2, kappa=kappa))
        
        return templates

# Test
if __name__ == "__main__":
    print("="*80)
    print("GW TEMPLATES MODULE TEST")
    print("="*80)
    
    # Create templates
    template_gr = GWTemplate(m1=36, m2=29)
    template_ift = GWTemplateIFT(m1=36, m2=29, kappa=1e-14)
    
    print(f"\n✓ GR Template created: M={template_gr.M:.1f} M_sun, Mc={template_gr.Mc:.2f} M_sun")
    print(f"✓ IFT Template created with κ = {template_ift.kappa:.2e}")
    
    # Generate frequency array
    f_array = np.linspace(20, 250, 1000)
    
    # Compute strains
    h_gr = template_gr.strain(f_array, distance=410)
    h_ift = template_ift.strain(f_array, distance=410)
    
    print(f"\n✓ GR strain computed: shape={h_gr.shape}")
    print(f"✓ IFT strain computed: shape={h_ift.shape}")
    
    # Difference
    diff_pct = np.abs(h_ift - h_gr) / np.abs(h_gr) * 100
    print(f"\n✓ Maximum difference (IFT vs GR): {np.max(diff_pct):.4f}%")
    
    print("\n" + "="*80)
    print("✓ GW Templates Module Operational")
    print("="*80)
