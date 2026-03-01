"""
Texture-Adaptive Perturbation Masking (TAPM)
=============================================

A perception-aware perturbation allocation system based on the
Human Visual System (HVS) masking model.

Academic Foundation:
-------------------
The Human Visual System has variable sensitivity across image regions:
  - High sensitivity in smooth/uniform areas (changes easily visible)
  - Low sensitivity in textured/edge-rich areas (changes hidden)
  - Variable sensitivity based on luminance (Weber's Law)
  - Lower sensitivity to chrominance changes than luminance

This module implements a sophisticated masking model that:
  1. Estimates local texture complexity via gradient magnitude
  2. Detects edges using Canny-like operators
  3. Computes local luminance variance
  4. Combines these into a perceptual importance map
  5. Allocates more perturbation budget to high-texture areas

This ensures maximum disruption to AI with minimum visibility to humans.

References:
  - Watson (1993): DCT quantization visibility model
  - Wandell (1995): Foundations of Vision
  - Larson & Chandler (2010): Most Apparent Distortion metric
"""

import numpy as np
from scipy.ndimage import gaussian_filter, sobel, uniform_filter
from typing import Tuple


class TextureAnalyzer:
    """
    Analyzes local texture complexity to create a perceptual mask.

    High values = high texture = safe to add more perturbation.
    Low values = smooth area = minimum perturbation allowed.
    """

    def __init__(self, sigma_gradient: float = 1.5,
                 sigma_variance: float = 3.0,
                 variance_window: int = 7):
        self.sigma_gradient = sigma_gradient
        self.sigma_variance = sigma_variance
        self.variance_window = variance_window

    def compute_texture_map(self, image: np.ndarray) -> np.ndarray:
        """
        Compute a texture complexity map.

        Parameters
        ----------
        image : np.ndarray
            Input image as float64 in [0, 1], shape (H, W, C).

        Returns
        -------
        np.ndarray
            Texture map in [0, 1], shape (H, W). Higher = more textured.
        """
        # Convert to grayscale for texture analysis
        if image.ndim == 3:
            gray = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + \
                   0.114 * image[:, :, 2]
        else:
            gray = image

        # 1. Gradient magnitude (edge detection)
        grad_x = sobel(gray, axis=1)
        grad_y = sobel(gray, axis=0)
        gradient_mag = np.sqrt(grad_x**2 + grad_y**2)
        gradient_mag = gaussian_filter(gradient_mag, self.sigma_gradient)

        # 2. Local variance (texture measure)
        local_mean = uniform_filter(gray, self.variance_window)
        local_sq_mean = uniform_filter(gray**2, self.variance_window)
        local_var = np.maximum(local_sq_mean - local_mean**2, 0)
        local_var = np.sqrt(local_var)  # Standard deviation

        # 3. Luminance-based masking (Weber's law)
        # Higher luminance = lower sensitivity to noise
        luminance_mask = np.clip(gray, 0.05, 0.95)
        # Weber fraction: sensitivity ~ 1/luminance in mid-range
        weber_mask = luminance_mask * (1.0 - luminance_mask)
        weber_mask = weber_mask / (weber_mask.max() + 1e-8)

        # 4. Combine components
        # Normalize each component
        gradient_norm = gradient_mag / (gradient_mag.max() + 1e-8)
        variance_norm = local_var / (local_var.max() + 1e-8)

        # Weighted combination (boost gradient for more differentiation)
        texture_map = (0.5 * gradient_norm +
                       0.35 * variance_norm +
                       0.15 * weber_mask)

        # Smooth the map to avoid harsh transitions
        texture_map = gaussian_filter(texture_map, self.sigma_variance)

        # Normalize to [0, 1]
        texture_map = (texture_map - texture_map.min()) / \
                      (texture_map.max() - texture_map.min() + 1e-8)

        return texture_map


class PerceptualMask:
    """
    Creates a full perceptual masking map that includes:
      - Texture complexity
      - Color channel sensitivity (chrominance vs luminance)
      - Content-adaptive noise floor

    The mask values represent how much perturbation can be added
    at each pixel without being perceptually detectable.
    """

    def __init__(self, min_strength: float = 0.15,
                 max_strength: float = 1.0,
                 chroma_boost: float = 1.5):
        self.min_strength = min_strength
        self.max_strength = max_strength
        self.chroma_boost = chroma_boost
        self.texture_analyzer = TextureAnalyzer()

    def compute_mask(self, image: np.ndarray) -> np.ndarray:
        """
        Compute per-pixel, per-channel perturbation strength mask.

        Parameters
        ----------
        image : np.ndarray
            Input image as float64 in [0, 1], shape (H, W, C).

        Returns
        -------
        np.ndarray
            Mask in [min_strength, max_strength], shape (H, W, C).
        """
        h, w, c = image.shape

        # Get texture map
        texture_map = self.texture_analyzer.compute_texture_map(image)

        # Scale to [min_strength, max_strength]
        strength_range = self.max_strength - self.min_strength
        base_mask = self.min_strength + texture_map * strength_range

        # Create per-channel mask
        mask = np.zeros_like(image)

        # Channel sensitivity: humans are less sensitive to chrominance
        # In RGB, approximate by boosting R and B channels
        channel_weights = np.array([self.chroma_boost, 1.0, self.chroma_boost])

        for ch in range(c):
            mask[:, :, ch] = base_mask * min(channel_weights[ch],
                                              self.max_strength)

        # Clip to valid range
        mask = np.clip(mask, self.min_strength, self.max_strength)

        return mask

    def apply_mask(self, image: np.ndarray,
                   perturbation: np.ndarray) -> np.ndarray:
        """
        Apply perceptual mask to a perturbation.

        Scales the perturbation at each pixel based on the local
        perceptual sensitivity.

        Parameters
        ----------
        image : np.ndarray
            Original image for computing the mask.
        perturbation : np.ndarray
            Raw perturbation to be masked.

        Returns
        -------
        np.ndarray
            Masked perturbation (same shape as input).
        """
        mask = self.compute_mask(image)
        return perturbation * mask
