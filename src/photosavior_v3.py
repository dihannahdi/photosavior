"""
PhotoSavior v3 — Phantom Spectral Encoding Engine
====================================================

The unified engine that integrates ALL novel contributions:

1. Multi-Model Ensemble Adversarial Attack (MEAA)
   → Simultaneous PGD against CLIP + DINOv2 + SigLIP
   → Creates "universal" adversarial images

2. Differentiable JPEG-Robust Optimization (DJRO)  
   → Perturbations survive lossy compression
   → Differentiable DCT/quantization in the PGD loop

3. Psychovisual Frequency Shaping (PFS)
   → CSF-model-based perturbation hiding
   → Luminance adaptation + texture masking

4. PhotoSavior Format (.psf)
   → Purpose-built AI-resistant image container
   → HMAC integrity verification + metadata

Architecture:
=============

    Input Image (RGB)
         │
         ▼
    ┌────────────────────────┐
    │  Psychovisual Analysis │ ← Compute perturbation budget map
    │  (CSF + Texture + Lum) │
    └────────────┬───────────┘
                 │
         ▼
    ┌────────────────────────┐
    │  Ensemble PGD Attack   │ ← Multi-model optimization loop
    │  ┌──────────┐          │   ┌──────────────────────┐
    │  │ CLIP     │←─ grad ──│───│ Differentiable JPEG  │
    │  │ DINOv2   │←─ grad ──│───│ Differentiable Resize│
    │  │ SigLIP   │←─ grad ──│───│ Gaussian Blur/Noise  │
    │  └──────────┘          │   └──────────────────────┘
    │       │                │
    │  Adaptive Weighting    │
    │  Momentum + Cos Anneal │
    │  PV Constraint Proj.   │
    └────────────┬───────────┘
                 │
         ▼
    ┌────────────────────────┐
    │  Output Generation     │
    │  - PNG (standard)      │
    │  - PSF (novel format)  │
    │  - Metrics report      │
    └────────────────────────┘

Usage
=====

Basic::

    from photosavior.src.photosavior_v3 import PhotoSaviorV3

    engine = PhotoSaviorV3(strength='moderate')
    result = engine.protect('photo.jpg')
    result.save('protected.png')
    result.save('protected.psf')  # Novel format

Advanced::

    engine = PhotoSaviorV3(
        strength='strong',
        models=['clip', 'dinov2', 'siglip'],
        jpeg_robustness=True,
        psychovisual=True,
    )
    result = engine.protect(image_numpy)
    print(result.metrics)

Author: PhotoSavior Research Team
"""

import numpy as np
from PIL import Image
from pathlib import Path
from typing import Optional, Dict, List, Union
import time
import json


class ProtectionResult:
    """
    Container for protection results.
    
    Properties:
        image: protected image as numpy array (H, W, 3) float [0, 1]
        image_uint8: protected image as uint8
        delta: perturbation applied
        metrics: attack metrics dict
        original: original image (if stored)
    """
    
    def __init__(self, protected_image: np.ndarray,
                 delta: np.ndarray,
                 metrics: Dict,
                 original_image: Optional[np.ndarray] = None,
                 protection_level: str = 'moderate'):
        self.image = protected_image
        self.delta = delta
        self.metrics = metrics
        self.original = original_image
        self.protection_level = protection_level
    
    @property
    def image_uint8(self) -> np.ndarray:
        """Protected image as uint8."""
        return (self.image * 255).clip(0, 255).astype(np.uint8)
    
    @property
    def psnr(self) -> float:
        """Peak Signal-to-Noise Ratio (dB)."""
        return self.metrics.get('image_quality', {}).get('psnr_db', 0.0)
    
    @property
    def displacement(self) -> Dict[str, float]:
        """Per-model feature displacement."""
        per_model = self.metrics.get('per_model', {})
        return {k: v.get('feature_displacement', 0) 
                for k, v in per_model.items()}
    
    def save(self, path: str, **kwargs):
        """
        Save protected image. Format detected from extension.
        
        Supported: .png, .jpg, .psf, .bmp, .tiff
        """
        path = Path(path)
        ext = path.suffix.lower()
        
        if ext == '.psf':
            self._save_psf(str(path), **kwargs)
        else:
            self._save_standard(str(path), **kwargs)
    
    def _save_standard(self, path: str, quality: int = 95):
        """Save as standard image format (PNG, JPEG, etc.)."""
        img = Image.fromarray(self.image_uint8)
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        if path_obj.suffix.lower() in ('.jpg', '.jpeg'):
            img.save(path, quality=quality)
        else:
            img.save(path)
    
    def _save_psf(self, path: str, **kwargs):
        """Save in PhotoSavior Format."""
        from .psf_codec import save_psf
        save_psf(
            protected_image=self.image,
            output_path=path,
            protection_level=self.protection_level,
            original_image=self.original,
            metrics=self.metrics,
        )
    
    def summary(self) -> str:
        """Human-readable summary of protection results."""
        lines = [
            "PhotoSavior v3 — Protection Report",
            "=" * 40,
            f"Protection Level: {self.protection_level}",
            f"Image Quality (PSNR): {self.psnr:.1f} dB",
            f"Max Perturbation (L∞): {self.metrics.get('image_quality', {}).get('linf_255', 0):.1f}/255",
            "",
            "Feature Displacement by Model:",
        ]
        for model, disp in self.displacement.items():
            lines.append(f"  {model}: {disp:.1%}")
        
        elapsed = self.metrics.get('elapsed_seconds', 0)
        lines.append(f"\nProcessing Time: {elapsed:.1f}s")
        
        attack_cfg = self.metrics.get('attack_config', {})
        lines.append(f"JPEG Robustness: {attack_cfg.get('jpeg_robustness', False)}")
        lines.append(f"Psychovisual Shaping: {attack_cfg.get('psychovisual', False)}")
        
        return "\n".join(lines)


class PhotoSaviorV3:
    """
    PhotoSavior v3 — Phantom Spectral Encoding Engine.
    
    The main entry point for AI-resistant image protection with
    all novel contributions integrated.
    
    Parameters
    ----------
    strength : str
        Protection strength: 'subtle', 'moderate', 'strong', 'maximum'
    models : list of str, optional
        Which AI models to attack. Default depends on strength.
        Options: 'clip', 'dinov2', 'siglip'
    jpeg_robustness : bool
        Enable differentiable JPEG robustness optimization
    jpeg_quality : int
        Target JPEG quality for robustness (lower = more robust)
    psychovisual : bool
        Enable HVS-based psychovisual frequency shaping
    legacy_layers : bool
        Also apply legacy spectral/neural layers (v1/v2 compatibility)
    """
    
    VERSION = "3.0.0"
    CODENAME = "Phantom Spectral Encoding"
    
    def __init__(self,
                 strength: str = 'moderate',
                 models: Optional[List[str]] = None,
                 jpeg_robustness: bool = True,
                 jpeg_quality: int = 75,
                 psychovisual: bool = True,
                 legacy_layers: bool = False):
        
        self.strength = strength
        self.models = models
        self.jpeg_robustness = jpeg_robustness
        self.jpeg_quality = jpeg_quality
        self.psychovisual = psychovisual
        self.legacy_layers = legacy_layers
        
        # Lazy-loaded components
        self._shield = None
        self._legacy_engine = None
    
    def _get_shield(self):
        """Lazy-load the ensemble shield."""
        if self._shield is None:
            from .ensemble_attack import PhantomSpectralShield
            kwargs = {
                'strength': self.strength,
                'jpeg_robustness': self.jpeg_robustness,
                'psychovisual': self.psychovisual,
            }
            if self.models is not None:
                kwargs['models'] = self.models
            self._shield = PhantomSpectralShield(**kwargs)
        return self._shield
    
    def _get_legacy_engine(self):
        """Lazy-load legacy protection layers."""
        if self._legacy_engine is None and self.legacy_layers:
            try:
                from .photosavior import PhotoSavior
                level_map = {
                    'subtle': 1, 'moderate': 2, 
                    'strong': 3, 'maximum': 4
                }
                self._legacy_engine = PhotoSavior(
                    protection_level=level_map.get(self.strength, 2),
                    use_clip=False,  # We handle CLIP in ensemble
                )
            except ImportError:
                self._legacy_engine = None
        return self._legacy_engine
    
    def protect(self, 
                image: Union[str, np.ndarray, Image.Image],
                verbose: bool = True) -> ProtectionResult:
        """
        Protect an image with Phantom Spectral Encoding.
        
        Args:
            image: path to image, numpy array, or PIL Image
            verbose: print progress information
        Returns:
            ProtectionResult with protected image and metrics
        """
        start_time = time.time()
        
        if verbose:
            print(f"\n{'#'*60}")
            print(f"  PhotoSavior v{self.VERSION} — {self.CODENAME}")
            print(f"  Strength: {self.strength}")
            print(f"{'#'*60}")
        
        # 1. Load image
        image_np, original_path = self._load_image(image)
        h, w = image_np.shape[:2]
        
        if verbose:
            print(f"  Image: {w}×{h} ({w*h/1e6:.1f} megapixels)")
        
        # 2. Apply Phantom Spectral Encoding (main novel engine)
        shield = self._get_shield()
        result = shield.protect(image_np, verbose=verbose)
        
        protected_image = result['protected_image']
        delta = result['delta']
        metrics = result['metrics']
        
        # 3. Apply legacy layers if requested
        if self.legacy_layers:
            legacy = self._get_legacy_engine()
            if legacy is not None:
                if verbose:
                    print("\n  Applying legacy spectral/neural layers...")
                # Convert to format expected by legacy engine
                protected_uint8 = (protected_image * 255).clip(0, 255).astype(np.uint8)
                from PIL import Image as PILImage
                pil_img = PILImage.fromarray(protected_uint8)
                
                legacy_result = legacy.protect(pil_img)
                legacy_protected = np.array(legacy_result['protected_image']).astype(np.float64) / 255.0
                
                # Blend: 80% PSE + 20% legacy
                protected_image = 0.8 * protected_image + 0.2 * legacy_protected
                protected_image = np.clip(protected_image, 0.0, 1.0)
                
                # Update delta
                delta = protected_image - image_np
                
                metrics['legacy_layers'] = legacy_result.get('report', {}).get('layers', {})
        
        # 4. Final quality check
        psnr = 10 * np.log10(1.0 / (np.mean(delta ** 2) + 1e-10))
        metrics['image_quality']['psnr_db'] = psnr
        metrics['image_quality']['linf'] = np.max(np.abs(delta))
        metrics['image_quality']['linf_255'] = np.max(np.abs(delta)) * 255
        
        total_time = time.time() - start_time
        metrics['total_elapsed_seconds'] = total_time
        
        if verbose:
            print(f"\n  Total processing time: {total_time:.1f}s")
            print(f"  Final PSNR: {psnr:.1f} dB")
        
        return ProtectionResult(
            protected_image=protected_image,
            delta=delta,
            metrics=metrics,
            original_image=image_np,
            protection_level=self.strength,
        )
    
    def _load_image(self, image) -> tuple:
        """Load image from various input types."""
        original_path = None
        
        if isinstance(image, str) or isinstance(image, Path):
            original_path = str(image)
            pil_img = Image.open(image).convert('RGB')
            image_np = np.array(pil_img).astype(np.float64) / 255.0
        elif isinstance(image, Image.Image):
            image_np = np.array(image.convert('RGB')).astype(np.float64) / 255.0
        elif isinstance(image, np.ndarray):
            image_np = image.copy()
            if image_np.dtype == np.uint8:
                image_np = image_np.astype(np.float64) / 255.0
            elif image_np.max() > 1.0:
                image_np = image_np / 255.0
            if image_np.ndim == 2:
                image_np = np.stack([image_np] * 3, axis=-1)
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")
        
        return image_np, original_path
    
    def protect_batch(self,
                      image_paths: List[str],
                      output_dir: str,
                      output_format: str = 'png',
                      verbose: bool = True) -> List[Dict]:
        """
        Protect multiple images.
        
        Args:
            image_paths: list of input image paths
            output_dir: directory for output files
            output_format: 'png', 'jpg', or 'psf'
            verbose: print progress
        Returns:
            list of metrics dicts
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = []
        for i, path in enumerate(image_paths):
            if verbose:
                print(f"\n[{i+1}/{len(image_paths)}] Processing: {path}")
            
            result = self.protect(path, verbose=verbose)
            
            # Generate output path
            stem = Path(path).stem
            out_path = output_dir / f"{stem}_protected.{output_format}"
            result.save(str(out_path))
            
            results.append({
                'input': path,
                'output': str(out_path),
                'metrics': result.metrics,
            })
        
        return results


# ============================================================
# Convenience Functions
# ============================================================

def protect_image(image, strength='moderate', **kwargs) -> ProtectionResult:
    """
    One-line image protection.
    
    Args:
        image: path, numpy array, or PIL Image
        strength: 'subtle', 'moderate', 'strong', 'maximum'
        **kwargs: passed to PhotoSaviorV3
    Returns:
        ProtectionResult
    """
    engine = PhotoSaviorV3(strength=strength, **kwargs)
    return engine.protect(image)


def verify_protection(psf_path: str) -> Dict:
    """
    Verify a .psf file's integrity.
    
    Args:
        psf_path: path to .psf file
    Returns:
        verification result dict
    """
    from .psf_codec import verify_psf
    return verify_psf(psf_path)
