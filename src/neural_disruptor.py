"""
Neural Feature Space Disruption (NFSD)
=======================================

Generates perturbations optimized to maximally disrupt the internal
feature representations that AI models use.

Theory:
-------
AI image models (CNNs, ViTs, VAEs, diffusion models) all extract
hierarchical feature representations:
  - Low-level: edges, textures, colors
  - Mid-level: parts, patterns, shapes
  - High-level: objects, scenes, semantics

Without requiring access to specific model weights, we can create
"universal" disruptions by targeting statistical properties that
ALL neural networks rely on:
  1. Local spatial correlations (what conv filters detect)
  2. Channel-wise statistics (batch norm depends on these)
  3. Patch-level coherence (ViT patch embeddings)
  4. Frequency-domain energy distribution (learned filters)

This module implements model-free adversarial perturbation generation
that disrupts these universal statistical properties.
"""

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import convolve2d
from typing import Tuple, Optional


class SpatialCorrelationDisruptor:
    """
    Disrupts local spatial correlations that convolutional filters rely on.

    Neural network conv layers learn filters that detect specific local
    patterns. By adding noise that breaks these local correlations, we
    cause the filters to produce incorrect activations.

    Strategy: Generate structured noise whose autocorrelation is the
    NEGATIVE of natural image autocorrelation (anti-correlated noise).
    """

    def __init__(self, kernel_sizes: list = [3, 5, 7],
                 strength: float = 0.03):
        self.kernel_sizes = kernel_sizes
        self.strength = strength

    def generate(self, image: np.ndarray, seed: int = 50) -> np.ndarray:
        """
        Generate anti-correlated spatial noise.

        The noise is designed to cancel out local correlation patterns
        that conv layers expect to find.
        """
        rng = np.random.RandomState(seed)
        h, w, c = image.shape
        perturbation = np.zeros_like(image)

        for ch in range(c):
            channel = image[:, :, ch]
            noise_sum = np.zeros((h, w))

            for ks in self.kernel_sizes:
                # Estimate local correlation using sliding window
                kernel = np.ones((ks, ks)) / (ks * ks)
                local_mean = convolve2d(channel, kernel, mode='same',
                                        boundary='symm')

                # Anti-correlation noise: push away from local mean
                deviation = channel - local_mean
                # Flip the deviation direction and add structured noise
                anti_corr = -deviation * rng.uniform(0.5, 1.5,
                                                      size=(h, w))
                noise_sum += anti_corr / len(self.kernel_sizes)

            perturbation[:, :, ch] = noise_sum * self.strength

        return perturbation


class ChannelStatisticsDisruptor:
    """
    Disrupts channel-wise statistics that batch normalization relies on.

    Most modern neural networks use BatchNorm (or LayerNorm, GroupNorm).
    These normalize features using channel-wise mean and variance.
    By carefully perturbing the image to change these statistics when
    processed by standard preprocessing, we cause the normalization
    to produce unexpected outputs.

    Strategy: Add channel-specific bias and variance modifications
    that are imperceptible but shift the channel statistics enough
    to cause misprocessing.
    """

    def __init__(self, strength: float = 0.02):
        self.strength = strength

    def generate(self, image: np.ndarray, seed: int = 51) -> np.ndarray:
        """
        Generate channel-statistics-disrupting perturbation.
        """
        rng = np.random.RandomState(seed)
        h, w, c = image.shape
        perturbation = np.zeros_like(image)

        for ch in range(c):
            channel = image[:, :, ch]
            ch_mean = channel.mean()
            ch_std = channel.std() + 1e-8

            # Shift mean slightly (affects BN centering)
            mean_shift = rng.randn() * self.strength * 0.5
            # Modify variance (affects BN scaling)
            var_mod = rng.randn(h, w) * self.strength
            # Weight by distance from mean (larger effect on outliers)
            z_scores = (channel - ch_mean) / ch_std
            perturbation[:, :, ch] = mean_shift + var_mod * np.abs(z_scores)

        return perturbation


class PatchCoherenceDisruptor:
    """
    Disrupts patch-level coherence targeted at Vision Transformer (ViT)
    architectures.

    ViTs process images by splitting them into patches (typically 16x16
    or 14x14), embedding each patch, and computing attention between
    patches. We disrupt this by:
      1. Adding inter-patch boundary artifacts that confuse patch embedding
      2. Creating subtle phase shifts between adjacent patches
      3. Breaking the smooth gradient across patch boundaries

    This makes the self-attention mechanism produce noisy attention maps.
    """

    def __init__(self, patch_sizes: list = [14, 16, 32],
                 strength: float = 0.025):
        self.patch_sizes = patch_sizes
        self.strength = strength

    def generate(self, image: np.ndarray, seed: int = 52) -> np.ndarray:
        """
        Generate patch-coherence-disrupting perturbation.
        """
        rng = np.random.RandomState(seed)
        h, w, c = image.shape
        perturbation = np.zeros_like(image)

        for ps in self.patch_sizes:
            for ch in range(c):
                # Create per-patch random offset
                n_patches_h = (h + ps - 1) // ps
                n_patches_w = (w + ps - 1) // ps

                # Random offset per patch
                offsets = rng.randn(n_patches_h, n_patches_w) * self.strength

                for pi in range(n_patches_h):
                    for pj in range(n_patches_w):
                        y1 = pi * ps
                        y2 = min((pi + 1) * ps, h)
                        x1 = pj * ps
                        x2 = min((pj + 1) * ps, w)

                        # Add patch-specific offset
                        patch_noise = np.ones((y2-y1, x2-x1)) * offsets[pi, pj]

                        # Add boundary emphasis (stronger at edges)
                        boundary = np.zeros((y2-y1, x2-x1))
                        bw = max(1, ps // 8)  # Boundary width
                        boundary[:bw, :] = 1.0
                        boundary[-bw:, :] = 1.0
                        boundary[:, :bw] = 1.0
                        boundary[:, -bw:] = 1.0
                        boundary_noise = rng.randn(y2-y1, x2-x1) * \
                            self.strength * 0.5
                        patch_noise += boundary * boundary_noise

                        perturbation[y1:y2, x1:x2, ch] += \
                            patch_noise / len(self.patch_sizes)

        return perturbation


class NeuralFeatureDisruptor:
    """
    Combines all neural feature disruption strategies into a unified
    perturbation that targets multiple architectural paradigms simultaneously.
    """

    def __init__(self, strength: float = 0.03):
        self.strength = strength
        self.spatial = SpatialCorrelationDisruptor(strength=strength)
        self.channel = ChannelStatisticsDisruptor(strength=strength * 0.7)
        self.patch = PatchCoherenceDisruptor(strength=strength * 0.8)

    def generate(self, image: np.ndarray, seed: int = 50) -> np.ndarray:
        """
        Generate unified neural feature disruption perturbation.

        Returns
        -------
        np.ndarray
            Combined perturbation (not yet applied to image).
        """
        spatial_pert = self.spatial.generate(image, seed=seed)
        channel_pert = self.channel.generate(image, seed=seed + 10)
        patch_pert = self.patch.generate(image, seed=seed + 20)

        # Weighted combination
        combined = (0.4 * spatial_pert +
                    0.3 * channel_pert +
                    0.3 * patch_pert)

        return combined
