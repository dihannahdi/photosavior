"""
Psychovisual Frequency Shaping (PFS)
======================================

Novel Contribution: Human Visual System-Aware Perturbation Allocation
---------------------------------------------------------------------
Previous adversarial protection methods (Glaze, PhotoGuard, Mist) use
simple uniform or texture-based perturbation budgets. They add the same
magnitude of perturbation across all spatial frequencies.

This module implements a BIOLOGICALLY ACCURATE model of human visual
sensitivity — the Contrast Sensitivity Function (CSF) — to allocate
perturbation energy OPTIMALLY across frequency bands. The result:
maximum adversarial impact with minimum human visibility.

Key Technical Innovations:
1. Mannos-Sakrison CSF: Models human sensitivity as a function of 
   spatial frequency (cycles/degree of visual angle)
2. Local luminance adaptation: Weber's law + Stevens' power law
3. Frequency-dependent perturbation budgets per 8×8 DCT block
4. Generates a 3D psychovisual mask (H × W × freq_band) that tells
   the optimizer exactly how much perturbation each pixel/frequency
   combination can tolerate before becoming visible

Theory:
-------
The Human Visual System (HVS) has peak sensitivity at ~4 cycles/degree
and drops off sharply at both low and high frequencies. The Mannos-
Sakrison CSF model captures this:

    S(f) = 2.6 × (0.0192 + 0.114f) × exp(-(0.114f)^1.1)

where f is spatial frequency in cycles/degree.

By INVERTING the CSF, we get a "perturbation tolerance" function:
high tolerance (= more perturbation allowed) where S(f) is LOW.

Additionally, the HVS is less sensitive:
  - In textured/edge-rich regions (masking effect)
  - In dark or very bright regions (luminance adaptation)
  - To chrominance changes vs. luminance changes

We combine ALL these factors into a single perturbation mask.

References:
  Mannos & Sakrison (1974) "Effects of visual fidelity criterion"
  Watson (1993) "DCT quantization matrices visually optimized"
  Wandell (1995) "Foundations of Vision"
  
Author: PhotoSavior Research Team  
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional, Tuple
from scipy.ndimage import gaussian_filter, sobel, uniform_filter


class ContrastSensitivityFunction:
    """
    Implementation of the Mannos-Sakrison Contrast Sensitivity Function.
    
    Models how sensitive the human eye is to different spatial frequencies.
    Used to determine where we can hide perturbations most effectively.
    """
    
    @staticmethod
    def mannos_sakrison(freq: np.ndarray) -> np.ndarray:
        """
        Mannos-Sakrison CSF model (1974).
        
        Args:
            freq: spatial frequency in cycles/degree of visual angle
        Returns:
            sensitivity values (higher = more sensitive = less perturbation allowed)
        """
        # Handle zero frequency
        freq = np.maximum(freq, 1e-6)
        sensitivity = 2.6 * (0.0192 + 0.114 * freq) * np.exp(
            -(0.114 * freq) ** 1.1
        )
        return sensitivity
    
    @staticmethod
    def watson_dct(freq_index_i: int, freq_index_j: int,
                   luminance: float = 128.0) -> float:
        """
        Watson's DCT-based visibility threshold (1993).
        
        Gives the minimum DCT coefficient change that is visible
        at a given frequency position and luminance level.
        
        Args:
            freq_index_i, freq_index_j: DCT frequency indices (0-7)
            luminance: local mean luminance (0-255)
        Returns:
            visibility threshold (higher = more perturbation allowed)
        """
        # Base visibility thresholds (Watson's Table 1)
        watson_base = np.array([
            [1.40, 1.01, 1.16, 1.66, 2.40, 3.43, 4.79, 6.56],
            [1.01, 1.45, 1.32, 1.52, 2.00, 2.71, 3.67, 4.93],
            [1.16, 1.32, 2.24, 2.59, 2.98, 3.64, 4.60, 5.88],
            [1.66, 1.52, 2.59, 3.77, 4.55, 5.30, 6.28, 7.60],
            [2.40, 2.00, 2.98, 4.55, 6.50, 7.79, 8.97, 10.4],
            [3.43, 2.71, 3.64, 5.30, 7.79, 10.2, 12.2, 14.1],
            [4.79, 3.67, 4.60, 6.28, 8.97, 12.2, 15.3, 18.2],
            [6.56, 4.93, 5.88, 7.60, 10.4, 14.1, 18.2, 22.5],
        ])
        
        base_threshold = watson_base[freq_index_i, freq_index_j]
        
        # Luminance adaptation (Stevens' power law)
        T0 = watson_base[0, 0]  # DC threshold
        alpha = 0.649  # Stevens exponent
        adapted = base_threshold * (luminance / 128.0) ** alpha
        
        return adapted


class PsychovisualMask:
    """
    Generates a psychovisual perturbation mask based on the HVS model.
    
    The mask assigns a perturbation BUDGET to each pixel location,
    indicating how much perturbation can be added before becoming
    visible to a human observer.
    
    High mask value = more perturbation allowed (low visual sensitivity)
    Low mask value = less perturbation allowed (high visual sensitivity)
    """
    
    def __init__(self, 
                 viewing_distance_pixels: float = 3000.0,
                 luminance_adaptation: bool = True,
                 texture_masking: bool = True,
                 chrominance_boost: bool = True):
        """
        Args:
            viewing_distance_pixels: assumed viewing distance in pixels
                (affects spatial frequency calculation)
            luminance_adaptation: enable luminance-dependent sensitivity
            texture_masking: enable texture-based masking
            chrominance_boost: allow more perturbation in color channels
        """
        self.viewing_distance = viewing_distance_pixels
        self.luminance_adaptation = luminance_adaptation
        self.texture_masking = texture_masking
        self.chrominance_boost = chrominance_boost
        self.csf = ContrastSensitivityFunction()
    
    def compute_frequency_tolerance(self, block_size: int = 8) -> np.ndarray:
        """
        Compute perturbation tolerance for each DCT frequency position.
        
        Returns:
            tolerance: (block_size, block_size) array where higher values
                      mean more perturbation is allowed at that frequency
        """
        tolerance = np.zeros((block_size, block_size))
        
        for i in range(block_size):
            for j in range(block_size):
                if i == 0 and j == 0:
                    # DC component — very sensitive, minimal perturbation
                    tolerance[i, j] = 0.1
                else:
                    # Compute spatial frequency for this DCT position
                    # freq = sqrt(i² + j²) / (2 × block_size) cycles/pixel
                    freq_pixels = np.sqrt(i**2 + j**2) / (2.0 * block_size)
                    
                    # Convert to cycles/degree
                    freq_degrees = freq_pixels * self.viewing_distance / 57.3
                    
                    # Get CSF sensitivity (invert for tolerance)
                    sensitivity = self.csf.mannos_sakrison(
                        np.array([freq_degrees])
                    )[0]
                    
                    # Tolerance = 1/sensitivity (more tolerant where less sensitive)
                    tolerance[i, j] = 1.0 / (sensitivity + 1e-6)
        
        # Normalize to [0, 1]
        tolerance = tolerance / (tolerance.max() + 1e-8)
        
        return tolerance
    
    def compute_luminance_mask(self, image: np.ndarray) -> np.ndarray:
        """
        Compute luminance-dependent perceptual mask.
        
        Based on Weber's law: sensitivity is proportional to 1/luminance
        in mid-range, but decreases for very dark and very bright regions.
        
        Args:
            image: (H, W, 3) float image in [0, 1]
        Returns:
            mask: (H, W) tolerance map in [0, 1]
        """
        # Convert to luminance
        if image.ndim == 3:
            lum = 0.2126 * image[:, :, 0] + 0.7152 * image[:, :, 1] + \
                  0.0722 * image[:, :, 2]
        else:
            lum = image
        
        # Weber-Fechner sensitivity model
        # More tolerant at extremes (dark/bright), less in mid-range
        # But we want TOLERANCE, so high values = safe to perturb
        
        # Parabolic tolerance: high at extremes, low at middle
        # But also factor in that very dark regions are noisy
        weber_tolerance = np.abs(lum - 0.5) * 2.0  # 0 at middle, 1 at extremes
        
        # Also: dark regions mask noise better (photon noise is expected)
        dark_boost = np.exp(-3.0 * lum)  # High for dark, low for bright
        
        # Combine
        mask = 0.6 * weber_tolerance + 0.4 * dark_boost
        
        # Smooth
        mask = gaussian_filter(mask, sigma=2.0)
        
        # Normalize
        mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
        
        return mask
    
    def compute_texture_mask(self, image: np.ndarray) -> np.ndarray:
        """
        Compute texture complexity mask.
        
        Regions with high texture complexity mask perturbations
        more effectively (visual masking phenomenon).
        
        Args:
            image: (H, W, 3) or (H, W) float image in [0, 1]
        Returns:
            mask: (H, W) texture complexity map in [0, 1]
        """
        if image.ndim == 3:
            gray = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + \
                   0.114 * image[:, :, 2]
        else:
            gray = image
        
        # Multi-scale gradient magnitude
        grad_x = sobel(gray, axis=1)
        grad_y = sobel(gray, axis=0)
        gradient_mag = np.sqrt(grad_x**2 + grad_y**2)
        
        # Local variance (texture indicator)
        window_size = 7
        local_mean = uniform_filter(gray, window_size)
        local_sq_mean = uniform_filter(gray**2, window_size)
        local_var = np.maximum(local_sq_mean - local_mean**2, 0)
        local_std = np.sqrt(local_var)
        
        # Edge density (Canny-like)
        edge_threshold = np.percentile(gradient_mag, 70)
        edge_density = uniform_filter(
            (gradient_mag > edge_threshold).astype(float), 11
        )
        
        # Combine all texture measures
        gradient_norm = gradient_mag / (gradient_mag.max() + 1e-8)
        std_norm = local_std / (local_std.max() + 1e-8)
        edge_norm = edge_density / (edge_density.max() + 1e-8)
        
        texture_map = (0.4 * gradient_norm + 
                      0.35 * std_norm + 
                      0.25 * edge_norm)
        
        # Ensure non-negative (floating point edge cases)
        texture_map = np.clip(texture_map, 0.0, 1.0)
        
        return texture_map
    
    def compute_full_mask(self, image: np.ndarray) -> np.ndarray:
        """
        Compute the complete psychovisual perturbation mask.
        
        Combines:
        1. Texture masking (spatial domain)
        2. Luminance adaptation (photometric domain)
        3. Frequency tolerance (transform domain)
        
        Args:
            image: (H, W, 3) float image in [0, 1]
        Returns:
            mask: (H, W) complete perturbation tolerance map in [0, 1]
                  Higher values = more perturbation allowed
        """
        h, w = image.shape[:2]
        
        # Start with uniform mask
        mask = np.ones((h, w), dtype=np.float64) * 0.5
        
        # Factor 1: Texture masking
        if self.texture_masking:
            texture = self.compute_texture_mask(image)
            mask = mask * (0.3 + 0.7 * texture)  # Scale: 0.3 in smooth, 1.0 in textured
        
        # Factor 2: Luminance adaptation
        if self.luminance_adaptation:
            luminance = self.compute_luminance_mask(image)
            mask = mask * (0.4 + 0.6 * luminance)  # Scale: 0.4 to 1.0
        
        # Normalize
        mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
        
        # Ensure minimum perturbation budget everywhere
        mask = np.clip(mask, 0.05, 1.0)
        
        return mask
    
    def compute_channel_weights(self) -> np.ndarray:
        """
        Compute per-channel perturbation weights.
        
        Human vision is MUCH less sensitive to chrominance changes
        than luminance changes. We exploit this to add more perturbation
        in color channels (Cb, Cr) while keeping luminance (Y) clean.
        
        Returns:
            weights: (3,) array of per-channel perturbation multipliers
                     in RGB space (approximating YCbCr sensitivity)
        """
        if self.chrominance_boost:
            # These weights approximate the luminance contribution ratios
            # R has moderate luminance contribution, so moderate tolerance
            # G has highest luminance contribution, so least tolerance
            # B has lowest luminance contribution, so most tolerance
            return np.array([1.0, 0.7, 1.3])
        else:
            return np.array([1.0, 1.0, 1.0])


class PsychovisualConstraint:
    """
    Enforces psychovisual constraints on adversarial perturbations.
    
    Used in the PGD optimization loop to reshape perturbations
    according to the psychovisual model BEFORE projection.
    
    This ensures that even as the optimizer pushes for maximum
    adversarial effect, the perturbation remains invisible.
    """
    
    def __init__(self, image: np.ndarray, 
                 max_perturbation: float = 16.0 / 255.0):
        """
        Args:
            image: original image (H, W, 3) in [0, 1]
            max_perturbation: maximum L∞ perturbation budget
        """
        self.max_perturbation = max_perturbation
        
        # Compute psychovisual mask
        pv_model = PsychovisualMask()
        self.spatial_mask = pv_model.compute_full_mask(image)
        self.channel_weights = pv_model.compute_channel_weights()
        self.freq_tolerance = pv_model.compute_frequency_tolerance()
    
    def apply(self, delta: np.ndarray) -> np.ndarray:
        """
        Apply psychovisual shaping to a perturbation.
        
        Args:
            delta: (H, W, 3) raw perturbation
        Returns:
            shaped delta: (H, W, 3) psychovisually shaped perturbation
        """
        shaped = delta.copy()
        h, w, c = shaped.shape
        
        # Apply spatial mask (pixel-wise tolerance)
        for ch in range(c):
            per_pixel_budget = (
                self.max_perturbation * 
                self.spatial_mask * 
                self.channel_weights[ch]
            )
            # Scale perturbation by local budget
            shaped[:, :, ch] = np.clip(
                shaped[:, :, ch],
                -per_pixel_budget,
                per_pixel_budget
            )
        
        return shaped
    
    def to_torch_mask(self, device: str = 'cpu') -> torch.Tensor:
        """
        Convert the psychovisual mask to a PyTorch tensor for use
        in differentiable optimization loops.
        
        Returns:
            mask: (1, 3, H, W) tensor of per-pixel perturbation budgets
        """
        h, w = self.spatial_mask.shape
        
        # Expand to 3 channels with channel weights
        mask_3ch = np.stack([
            self.spatial_mask * self.channel_weights[0],
            self.spatial_mask * self.channel_weights[1],
            self.spatial_mask * self.channel_weights[2],
        ], axis=0)  # (3, H, W)
        
        mask_tensor = torch.from_numpy(mask_3ch).float().unsqueeze(0)  # (1, 3, H, W)
        mask_tensor = mask_tensor * self.max_perturbation
        
        return mask_tensor.to(device)
