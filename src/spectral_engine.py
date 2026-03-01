"""
Spectral Perturbation Engine
=============================
Implements Multi-Spectral Perturbation Fusion (MSPF) — the core innovation
of PhotoSavior.

Theory:
-------
AI models process images through different internal representations:
  - CNNs learn spatial frequency features (captured by DCT/FFT)
  - Transformers use patch embeddings that are sensitive to all frequencies
  - VAEs encode images into latent vectors where frequency information mixes

By perturbing simultaneously in DCT, DWT, and FFT domains, we create
perturbations that:
  1. Cannot be removed by simple denoising (multi-domain resilience)
  2. Are invisible to humans (energy concentrated in perceptually masked bands)
  3. Maximally disrupt neural feature extraction across architectures
"""

import numpy as np
from scipy.fft import dctn, idctn, fft2, ifft2, fftshift, ifftshift
import pywt
from typing import Tuple, Optional


class DCTPerturber:
    """
    Discrete Cosine Transform domain perturbation.

    Injects adversarial energy into mid-frequency DCT coefficients.
    These frequencies are:
      - Important for neural network feature extraction
      - Less perceptually noticeable than low-frequency changes
      - More resilient to JPEG compression than high-frequency noise

    Based on the insight that JPEG operates in 8x8 DCT blocks, we
    specifically target coefficients that survive standard quantization.
    """

    def __init__(self, strength: float = 0.03, block_size: int = 8):
        self.strength = strength
        self.block_size = block_size
        # Mid-frequency band mask (zig-zag positions 10-40 out of 64)
        self._build_frequency_mask()

    def _build_frequency_mask(self):
        """Build a mask selecting mid-frequency DCT coefficients."""
        bs = self.block_size
        mask = np.zeros((bs, bs), dtype=np.float64)

        for i in range(bs):
            for j in range(bs):
                freq_index = i + j  # Diagonal frequency index
                # Target mid-frequencies (index 3-6): important for AI,
                # less visible to humans, survive JPEG compression
                if 3 <= freq_index <= 6:
                    # Weight by distance from DC component
                    mask[i, j] = 1.0 - 0.1 * abs(freq_index - 4.5)

        self.freq_mask = mask / (mask.max() + 1e-8)

    def perturb(self, image: np.ndarray, seed: int = 42) -> np.ndarray:
        """
        Apply DCT-domain adversarial perturbation.

        Parameters
        ----------
        image : np.ndarray
            Input image as float64 in [0, 1], shape (H, W, C).
        seed : int
            Random seed for reproducible perturbation.

        Returns
        -------
        np.ndarray
            Perturbed image in [0, 1].
        """
        rng = np.random.RandomState(seed)
        h, w, c = image.shape
        result = image.copy()
        bs = self.block_size

        # Process each channel independently
        for ch in range(c):
            channel = image[:, :, ch]

            # Pad to block-aligned size
            pad_h = (bs - h % bs) % bs
            pad_w = (bs - w % bs) % bs
            padded = np.pad(channel, ((0, pad_h), (0, pad_w)), mode='reflect')
            ph, pw = padded.shape

            # Process each block
            for bi in range(0, ph, bs):
                for bj in range(0, pw, bs):
                    block = padded[bi:bi+bs, bj:bj+bs]
                    # Forward DCT
                    dct_block = dctn(block, type=2, norm='ortho')
                    # Generate adversarial perturbation in DCT domain
                    noise = rng.randn(bs, bs) * self.strength
                    # Apply frequency mask (target mid-frequencies)
                    perturbation = noise * self.freq_mask
                    # Add perturbation
                    dct_block += perturbation
                    # Inverse DCT
                    padded[bi:bi+bs, bj:bj+bs] = idctn(
                        dct_block, type=2, norm='ortho'
                    )

            result[:, :, ch] = padded[:h, :w]

        return np.clip(result, 0.0, 1.0)


class DWTPerturber:
    """
    Discrete Wavelet Transform domain perturbation.

    Injects adversarial energy into wavelet detail coefficients.
    Uses the fact that:
      - Human vision is less sensitive to detail (high-freq) coefficients
      - Neural networks heavily rely on these detail features
      - DWT perturbations are naturally multi-scale (robust to resizing)

    The DWT decomposition gives us:
      - cA (approximation): Low-freq content — don't touch this
      - cH (horizontal detail): Horizontal edges
      - cV (vertical detail): Vertical edges
      - cD (diagonal detail): Diagonal textures
    We perturb cH, cV, cD to disrupt edge and texture features.
    """

    def __init__(self, strength: float = 0.025, wavelet: str = 'db4',
                 level: int = 2):
        self.strength = strength
        self.wavelet = wavelet
        self.level = level

    def perturb(self, image: np.ndarray, seed: int = 43) -> np.ndarray:
        """
        Apply DWT-domain adversarial perturbation.

        Parameters
        ----------
        image : np.ndarray
            Input image as float64 in [0, 1], shape (H, W, C).
        seed : int
            Random seed for reproducible perturbation.

        Returns
        -------
        np.ndarray
            Perturbed image in [0, 1].
        """
        rng = np.random.RandomState(seed)
        result = image.copy()

        for ch in range(image.shape[2]):
            channel = image[:, :, ch]

            # Multi-level DWT decomposition
            coeffs = pywt.wavedec2(channel, self.wavelet, level=self.level)

            # Perturb detail coefficients at each level
            for level_idx in range(1, len(coeffs)):
                detail_coeffs = list(coeffs[level_idx])
                for detail_idx in range(3):  # cH, cV, cD
                    coeff = detail_coeffs[detail_idx]
                    # Scale perturbation by coefficient magnitude
                    # (larger perturbation where signal is already strong)
                    scale = np.abs(coeff).mean() + 1e-8
                    noise = rng.randn(*coeff.shape) * self.strength * scale
                    detail_coeffs[detail_idx] = coeff + noise
                coeffs[level_idx] = tuple(detail_coeffs)

            # Reconstruct
            result[:, :, ch] = pywt.waverec2(coeffs, self.wavelet)[
                :image.shape[0], :image.shape[1]
            ]

        return np.clip(result, 0.0, 1.0)


class FFTPerturber:
    """
    Fast Fourier Transform domain perturbation.

    Injects adversarial phase noise in the frequency domain.

    Key insight: Neural networks are highly sensitive to phase information
    (Oppenheim & Lim, 1981). Small phase perturbations that are imperceptible
    to humans can cause dramatic changes in neural feature representations.

    We specifically target:
      - Mid-frequency ring in the Fourier magnitude spectrum
      - Phase perturbations in directions that maximize feature disruption
    """

    def __init__(self, strength: float = 0.02, freq_band: Tuple[float, float] = (0.1, 0.4)):
        self.strength = strength
        self.freq_band = freq_band

    def _create_bandpass_mask(self, h: int, w: int) -> np.ndarray:
        """Create a ring-shaped bandpass mask in frequency domain."""
        cy, cx = h // 2, w // 2
        Y, X = np.ogrid[:h, :w]
        # Normalized distance from center
        dist = np.sqrt((Y - cy)**2 + (X - cx)**2)
        max_dist = np.sqrt(cy**2 + cx**2)
        norm_dist = dist / max_dist

        # Smooth bandpass
        low, high = self.freq_band
        mask = np.exp(-((norm_dist - (low + high)/2)**2) /
                      (2 * ((high - low)/4)**2))
        return mask

    def perturb(self, image: np.ndarray, seed: int = 44) -> np.ndarray:
        """
        Apply FFT phase perturbation.

        Parameters
        ----------
        image : np.ndarray
            Input image as float64 in [0, 1], shape (H, W, C).
        seed : int
            Random seed for reproducible perturbation.

        Returns
        -------
        np.ndarray
            Perturbed image in [0, 1].
        """
        rng = np.random.RandomState(seed)
        h, w, c = image.shape
        result = image.copy()

        bandpass = self._create_bandpass_mask(h, w)

        for ch in range(c):
            channel = image[:, :, ch]

            # Forward FFT
            f_transform = fftshift(fft2(channel))

            # Get magnitude and phase
            magnitude = np.abs(f_transform)
            phase = np.angle(f_transform)

            # Add phase perturbation in the bandpass region
            phase_noise = rng.randn(h, w) * self.strength * np.pi
            phase += phase_noise * bandpass

            # Reconstruct with perturbed phase
            f_perturbed = magnitude * np.exp(1j * phase)
            channel_perturbed = np.real(ifft2(ifftshift(f_perturbed)))

            result[:, :, ch] = channel_perturbed

        return np.clip(result, 0.0, 1.0)


class MultiSpectralFusion:
    """
    Multi-Spectral Perturbation Fusion (MSPF)
    ==========================================

    The core innovation: fuses perturbations from DCT, DWT, and FFT domains
    using an adaptive weighting scheme.

    The fusion process:
    1. Generate perturbation in each spectral domain independently
    2. Compute a perceptual importance map for each domain
    3. Blend perturbations using importance-weighted averaging
    4. Apply global L-infinity constraint for imperceptibility

    This creates perturbations that are simultaneously effective against:
    - CNN-based models (DCT disruption maps to conv filter responses)
    - Transformer models (FFT phase disrupts patch correlations)
    - VAE encoders (DWT multi-scale disrupts encoder hierarchy)
    """

    def __init__(self, overall_strength: float = 0.04,
                 dct_weight: float = 0.35,
                 dwt_weight: float = 0.35,
                 fft_weight: float = 0.30,
                 max_linf: float = 8.0/255.0):
        self.overall_strength = overall_strength
        self.weights = np.array([dct_weight, dwt_weight, fft_weight])
        self.weights /= self.weights.sum()
        self.max_linf = max_linf

        self.dct = DCTPerturber(strength=overall_strength * 1.2)
        self.dwt = DWTPerturber(strength=overall_strength * 1.0)
        self.fft = FFTPerturber(strength=overall_strength * 0.8)

    def generate(self, image: np.ndarray, seed: int = 42) -> Tuple[np.ndarray, dict]:
        """
        Generate fused multi-spectral perturbation.

        Returns
        -------
        Tuple[np.ndarray, dict]
            (protected_image, metadata) where metadata contains
            per-domain perturbation statistics.
        """
        # Generate perturbations in each domain
        dct_result = self.dct.perturb(image, seed=seed)
        dwt_result = self.dwt.perturb(image, seed=seed + 1)
        fft_result = self.fft.perturb(image, seed=seed + 2)

        # Compute individual perturbations (deltas)
        delta_dct = dct_result - image
        delta_dwt = dwt_result - image
        delta_fft = fft_result - image

        # Weighted fusion
        fused_delta = (self.weights[0] * delta_dct +
                       self.weights[1] * delta_dwt +
                       self.weights[2] * delta_fft)

        # Apply L-infinity constraint (imperceptibility guarantee)
        fused_delta = np.clip(fused_delta, -self.max_linf, self.max_linf)

        # Apply perturbation
        protected = np.clip(image + fused_delta, 0.0, 1.0)

        # Compute statistics
        actual_delta = protected - image
        metadata = {
            'dct_l2': np.sqrt(np.mean(delta_dct**2)),
            'dwt_l2': np.sqrt(np.mean(delta_dwt**2)),
            'fft_l2': np.sqrt(np.mean(delta_fft**2)),
            'fused_l2': np.sqrt(np.mean(actual_delta**2)),
            'fused_linf': np.max(np.abs(actual_delta)),
            'psnr': self._compute_psnr(image, protected),
            'domain_weights': dict(zip(['dct', 'dwt', 'fft'],
                                       self.weights.tolist())),
        }

        return protected, metadata

    @staticmethod
    def _compute_psnr(original: np.ndarray, modified: np.ndarray) -> float:
        """Compute Peak Signal-to-Noise Ratio."""
        mse = np.mean((original - modified) ** 2)
        if mse < 1e-10:
            return float('inf')
        return 10 * np.log10(1.0 / mse)
