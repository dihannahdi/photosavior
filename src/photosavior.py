"""
PhotoSavior - Main Protection Engine
=====================================

Integrates all protection layers into a unified, easy-to-use system:

  Layer 1: Multi-Spectral Perturbation Fusion (MSPF)
           → Disrupts AI feature extraction across DCT/DWT/FFT domains

  Layer 2: Texture-Adaptive Perturbation Masking (TAPM)
           → Hides perturbations in textured areas for imperceptibility

  Layer 3: Neural Feature Space Disruption (NFSD)
           → Corrupts internal representations of neural networks

  Layer 4: Forensic Watermark Embedding (FWE)
           → Embeds survivable cryptographic proof of protection

  Processing Pipeline:
  1. Analyze image texture map
  2. Generate multi-spectral perturbation
  3. Generate neural feature disruption
  4. Apply texture-adaptive masking to combined perturbation
  5. Embed forensic watermark
  6. Quality assurance checks (PSNR, SSIM)
"""

import numpy as np
from PIL import Image
import os
from typing import Tuple, Optional, Dict, Union
from pathlib import Path

from .spectral_engine import MultiSpectralFusion
from .texture_mask import PerceptualMask
from .neural_disruptor import NeuralFeatureDisruptor
from .forensic_watermark import ForensicWatermark

# Try to load CLIP adversarial engine (requires torch + transformers)
try:
    from .clip_adversarial import CLIPAdversarialShield
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False


class ProtectionLevel:
    """Protection strength presets."""
    LIGHT = 1       # Minimal perturbation, best visual quality
    MODERATE = 2    # Balanced protection and quality
    STRONG = 3      # Strong protection, slight quality trade-off
    MAXIMUM = 4     # Maximum protection, noticeable perturbation


# Configuration for each protection level
LEVEL_CONFIGS = {
    ProtectionLevel.LIGHT: {
        'spectral_strength': 0.02,
        'neural_strength': 0.015,
        'max_linf': 4.0 / 255.0,
        'mask_min': 0.4,
        'mask_max': 1.0,
        'wm_q_step': 0.12,
        'clip_strength': 'subtle',
    },
    ProtectionLevel.MODERATE: {
        'spectral_strength': 0.035,
        'neural_strength': 0.025,
        'max_linf': 8.0 / 255.0,
        'mask_min': 0.3,
        'mask_max': 1.0,
        'wm_q_step': 0.15,
        'clip_strength': 'moderate',
    },
    ProtectionLevel.STRONG: {
        'spectral_strength': 0.05,
        'neural_strength': 0.035,
        'max_linf': 12.0 / 255.0,
        'mask_min': 0.25,
        'mask_max': 1.0,
        'wm_q_step': 0.15,
        'clip_strength': 'strong',
    },
    ProtectionLevel.MAXIMUM: {
        'spectral_strength': 0.07,
        'neural_strength': 0.05,
        'max_linf': 16.0 / 255.0,
        'mask_min': 0.2,
        'mask_max': 1.0,
        'wm_q_step': 0.20,
        'clip_strength': 'maximum',
    },
}


class PhotoSavior:
    """
    Main PhotoSavior protection engine.

    Usage::

        savior = PhotoSavior(protection_level=ProtectionLevel.STRONG)
        protected, report = savior.protect("input.jpg", "output.jpg")
        print(report)
    """

    def __init__(self, protection_level: int = ProtectionLevel.STRONG,
                 seed: int = 42, use_clip: bool = True):
        self.protection_level = protection_level
        self.seed = seed
        self.use_clip = use_clip and CLIP_AVAILABLE

        config = LEVEL_CONFIGS[protection_level]

        # CLIP adversarial attack — the primary, real protection
        if self.use_clip:
            self.clip_shield = CLIPAdversarialShield(
                strength=config['clip_strength']
            )
        else:
            self.clip_shield = None

        # Legacy layers (used as supplementary or fallback)
        self.spectral = MultiSpectralFusion(
            overall_strength=config['spectral_strength'],
            max_linf=config['max_linf']
        )
        self.mask = PerceptualMask(
            min_strength=config['mask_min'],
            max_strength=config['mask_max']
        )
        self.neural = NeuralFeatureDisruptor(
            strength=config['neural_strength']
        )
        self.watermark = ForensicWatermark(
            quantization_step=config['wm_q_step']
        )

    def load_image(self, path: str) -> np.ndarray:
        """Load image and convert to float64 [0, 1] RGB."""
        img = Image.open(path).convert('RGB')
        return np.array(img, dtype=np.float64) / 255.0

    def save_image(self, image: np.ndarray, path: str,
                   quality: int = 95) -> None:
        """Save float64 [0, 1] image to file."""
        img_uint8 = (np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)
        img = Image.fromarray(img_uint8, 'RGB')

        # Save with appropriate format
        ext = Path(path).suffix.lower()
        if ext in ['.jpg', '.jpeg']:
            img.save(path, 'JPEG', quality=quality)
        elif ext == '.png':
            img.save(path, 'PNG')
        else:
            img.save(path)

    def protect(self, input_path: str,
                output_path: Optional[str] = None,
                save_quality: int = 95) -> Tuple[np.ndarray, Dict]:
        """
        Protect an image with all MSAS layers.

        Parameters
        ----------
        input_path : str
            Path to the input image.
        output_path : str, optional
            Path to save the protected image.
        save_quality : int
            JPEG quality for saving (default: 95).

        Returns
        -------
        Tuple[np.ndarray, Dict]
            (protected_image, protection_report)
        """
        # Load image
        original = self.load_image(input_path)
        h, w, c = original.shape

        report = {
            'input_file': input_path,
            'image_size': f'{w}x{h}',
            'protection_level': self.protection_level,
            'clip_attack': self.use_clip,
            'layers': {},
        }

        # ═══════════════════════════════════════════
        # PRIMARY: CLIP Adversarial Attack (PGD)
        # This is the REAL protection — gradient-based
        # attack against a real vision model (CLIP)
        # ═══════════════════════════════════════════
        if self.use_clip and self.clip_shield is not None:
            protected, clip_report = self.clip_shield.protect(
                original, verbose=True
            )
            report['layers']['clip_adversarial'] = clip_report
        else:
            # Fallback: use legacy spectral/neural layers
            spectral_protected, spectral_meta = self.spectral.generate(
                original, seed=self.seed
            )
            spectral_perturbation = spectral_protected - original
            report['layers']['spectral'] = spectral_meta

            neural_perturbation = self.neural.generate(
                original, seed=self.seed + 100
            )
            report['layers']['neural'] = {
                'l2_norm': float(np.sqrt(np.mean(neural_perturbation**2))),
                'linf_norm': float(np.max(np.abs(neural_perturbation))),
            }

            combined_perturbation = 0.6 * spectral_perturbation + \
                                    0.4 * neural_perturbation

            masked_perturbation = self.mask.apply_mask(
                original, combined_perturbation
            )

            config = LEVEL_CONFIGS[self.protection_level]
            max_linf = config['max_linf']
            masked_perturbation = np.clip(
                masked_perturbation, -max_linf, max_linf
            )

            protected = np.clip(original + masked_perturbation, 0.0, 1.0)

            report['layers']['masking'] = {
                'masked_l2': float(np.sqrt(np.mean(masked_perturbation**2))),
                'masked_linf': float(np.max(np.abs(masked_perturbation))),
            }

        # ═══════════════════════════════════════════
        # LAYER 4: Forensic Watermark Embedding
        # ═══════════════════════════════════════════
        protected, wm_meta = self.watermark.embed(
            protected, protection_level=self.protection_level
        )
        report['layers']['watermark'] = wm_meta

        # ═══════════════════════════════════════════
        # Quality Assurance
        # ═══════════════════════════════════════════
        final_delta = protected - original
        report['quality'] = {
            'psnr_db': float(self._compute_psnr(original, protected)),
            'ssim': float(self._compute_ssim(original, protected)),
            'final_l2': float(np.sqrt(np.mean(final_delta**2))),
            'final_linf': float(np.max(np.abs(final_delta))),
            'max_pixel_change': float(np.max(np.abs(final_delta)) * 255),
        }

        # Save if output path provided
        if output_path:
            os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
            self.save_image(protected, output_path, quality=save_quality)
            report['output_file'] = output_path

        return protected, report

    def verify_protection(self, image_path: str) -> Dict:
        """
        Verify if an image has PhotoSavior protection.

        Returns extraction info including watermark validity.
        """
        image = self.load_image(image_path)
        payload, info = self.watermark.extract(image)
        return info

    def detect_tampering(self, original_path: str,
                          suspect_path: str) -> Dict:
        """
        Detect if a suspect image was AI-modified from a protected original.. 
        """
        original = self.load_image(original_path)
        suspect = self.load_image(suspect_path)

        # Watermark-based verification
        wm_result = self.watermark.verify(original, suspect)

        # Pixel-level analysis
        if original.shape == suspect.shape:
            delta = suspect - original
            pixel_analysis = {
                'mse': float(np.mean(delta**2)),
                'psnr': float(self._compute_psnr(original, suspect)),
                'ssim': float(self._compute_ssim(original, suspect)),
                'max_change': float(np.max(np.abs(delta)) * 255),
                'changed_pixels_pct': float(
                    np.mean(np.max(np.abs(delta), axis=2) > 0.01) * 100
                ),
            }
        else:
            pixel_analysis = {
                'size_changed': True,
                'original_size': original.shape[:2],
                'suspect_size': suspect.shape[:2],
            }

        return {
            'watermark_analysis': wm_result,
            'pixel_analysis': pixel_analysis,
        }

    @staticmethod
    def _compute_psnr(original: np.ndarray, modified: np.ndarray) -> float:
        mse = np.mean((original - modified) ** 2)
        if mse < 1e-10:
            return float('inf')
        return 10 * np.log10(1.0 / mse)

    @staticmethod
    def _compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
        """
        Compute Structural Similarity Index (simplified).

        Based on Wang et al., 2004: "Image quality assessment:
        from error visibility to structural similarity"
        """
        C1 = (0.01) ** 2  # Stabilization constant
        C2 = (0.03) ** 2

        ssim_vals = []
        for ch in range(img1.shape[2]):
            mu1 = img1[:, :, ch].mean()
            mu2 = img2[:, :, ch].mean()
            sigma1_sq = img1[:, :, ch].var()
            sigma2_sq = img2[:, :, ch].var()
            sigma12 = np.mean(
                (img1[:, :, ch] - mu1) * (img2[:, :, ch] - mu2)
            )

            numerator = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
            denominator = (mu1**2 + mu2**2 + C1) * \
                          (sigma1_sq + sigma2_sq + C2)
            ssim_vals.append(numerator / denominator)

        return float(np.mean(ssim_vals))
