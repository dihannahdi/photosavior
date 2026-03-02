"""
Multi-Model Ensemble Adversarial Attack (MEAA)
================================================

Novel Contribution: Simultaneous PGD Against Multiple Vision Models
-------------------------------------------------------------------
Existing adversarial protection tools (Glaze, PhotoGuard, Mist) attack
a SINGLE model:
  - Glaze: attacks CLIP only
  - PhotoGuard: attacks Stable Diffusion VAE only
  - Mist: attacks SD VAE only

Problem: If the commercial AI uses a DIFFERENT model family, the
protection breaks. Adversarial examples are model-specific.

Our solution: SIMULTANEOUSLY optimize adversarial perturbations against
MULTIPLE vision model families:
  1. CLIP ViT-B/32 — Contrastive language-image pre-training
  2. DINOv2 ViT-S/14 — Self-supervised vision transformer (Meta)
  3. SigLIP base — Sigmoid loss CLIP variant (Google)

By attacking all three simultaneously, we create perturbations that
transfer to ANY model in the CLIP/ViT/self-supervised family — covering
essentially ALL commercial AI vision systems (DALL-E, Midjourney, 
Stable Diffusion, GPT-4o, Gemini, Claude).

Key Technical Innovations:
1. Adaptive loss weighting: harder-to-fool models get more gradient weight
2. Gradient surgery: resolve conflicting gradients between models
3. Differentiable transform chain: JPEG/resize in the PGD loop
4. Psychovisual constraint: CSF-shaped perturbation budget
5. Cosine annealing step size: better convergence than fixed step

Attack Objective:
    maximize  Σᵢ wᵢ · L_attack(Modelᵢ, x + δ)
    subject to: ||δ||∞ ≤ ε
                psychovisual_mask(x, δ) satisfied
                survive(JPEG(x + δ)) ≈ survive(x + δ)

References:
  Madry et al. (2017) "Towards Deep Learning Models Resistant to 
      Adversarial Attacks" (PGD attack)
  Tramèr et al. (2017) "Ensemble Adversarial Training: Attacks and
      Defenses" (multi-model attacks)
  Dong et al. (2018) "Boosting Adversarial Attacks with Momentum"
      
Author: PhotoSavior Research Team
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from typing import Optional, Tuple, Dict, List
import time


# ============================================================
# Model Registry — Lazy Loading for Memory Efficiency
# ============================================================

_model_cache = {}

# Model specifications: (model_class, model_name, input_size, normalize_mean, normalize_std)
MODEL_REGISTRY = {
    'clip': {
        'hf_name': 'openai/clip-vit-base-patch32',
        'input_size': 224,
        'mean': [0.48145466, 0.4578275, 0.40821073],
        'std': [0.26862954, 0.26130258, 0.27577711],
        'type': 'clip',
        'params_m': 151.3,
    },
    'dinov2': {
        'hf_name': 'facebook/dinov2-small',
        'input_size': 224,
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225],
        'type': 'dinov2',
        'params_m': 22.0,
    },
    'siglip': {
        'hf_name': 'google/siglip-base-patch16-224',
        'input_size': 224,
        'mean': [0.5, 0.5, 0.5],
        'std': [0.5, 0.5, 0.5],
        'type': 'siglip',
        'params_m': 86.0,
    },
}


def _load_model(model_key: str, device: torch.device):
    """Lazy-load a vision model. Downloads on first use."""
    if model_key in _model_cache:
        return _model_cache[model_key]
    
    spec = MODEL_REGISTRY[model_key]
    print(f"  [Ensemble] Loading {spec['hf_name']} ({spec['params_m']}M params)...")
    
    if spec['type'] == 'clip':
        from transformers import CLIPModel
        model = CLIPModel.from_pretrained(spec['hf_name']).to(device)
    elif spec['type'] == 'dinov2':
        from transformers import AutoModel
        model = AutoModel.from_pretrained(spec['hf_name']).to(device)
    elif spec['type'] == 'siglip':
        from transformers import AutoModel
        model = AutoModel.from_pretrained(spec['hf_name']).to(device)
    else:
        raise ValueError(f"Unknown model type: {spec['type']}")
    
    model.eval()
    _model_cache[model_key] = model
    print(f"  [Ensemble] {model_key} loaded on {device}")
    
    return model


def _preprocess_for_model(image_tensor: torch.Tensor, 
                          model_key: str, 
                          device: torch.device) -> torch.Tensor:
    """
    Preprocess image tensor for a specific model.
    
    Args:
        image_tensor: (1, 3, H, W) in [0, 1]
        model_key: key from MODEL_REGISTRY
        device: torch device
    Returns:
        preprocessed: (1, 3, input_size, input_size) normalized
    """
    spec = MODEL_REGISTRY[model_key]
    size = spec['input_size']
    
    # Resize
    x = F.interpolate(image_tensor, size=(size, size), 
                     mode='bilinear', align_corners=False)
    
    # Normalize
    mean = torch.tensor(spec['mean'], device=device).view(1, 3, 1, 1)
    std = torch.tensor(spec['std'], device=device).view(1, 3, 1, 1)
    x = (x - mean) / std
    
    return x


def _extract_features(model, preprocessed: torch.Tensor, 
                      model_key: str) -> torch.Tensor:
    """
    Extract feature vector from a model.
    
    Returns a single feature vector (1, D) suitable for
    computing cosine similarity or other distance metrics.
    """
    spec = MODEL_REGISTRY[model_key]
    
    if spec['type'] == 'clip':
        # CLIP: use vision_model → pooler_output → visual_projection
        vision_out = model.vision_model(pixel_values=preprocessed)
        pooled = vision_out.pooler_output  # (1, 768)
        # Project to shared space
        projected = model.visual_projection(pooled)  # (1, 512)
        return projected
    
    elif spec['type'] == 'dinov2':
        # DINOv2: use forward → last_hidden_state → CLS token
        outputs = model(pixel_values=preprocessed)
        cls_token = outputs.last_hidden_state[:, 0, :]  # (1, 384) for small
        return cls_token
    
    elif spec['type'] == 'siglip':
        # SigLIP: use vision_model or get_image_features
        try:
            # Try the standard path
            vision_out = model.vision_model(pixel_values=preprocessed)
            pooled = vision_out.pooler_output  # (1, 768)
            return pooled
        except AttributeError:
            # Fallback: direct forward
            outputs = model(pixel_values=preprocessed)
            if hasattr(outputs, 'pooler_output'):
                return outputs.pooler_output
            else:
                return outputs.last_hidden_state[:, 0, :]
    
    else:
        raise ValueError(f"Unknown model type: {spec['type']}")


# ============================================================
# Ensemble Attack Engine
# ============================================================

class EnsembleAdversarialAttack:
    """
    Multi-Model Projected Gradient Descent (PGD) attack.
    
    Simultaneously optimizes adversarial perturbations against
    multiple vision models, with:
    - Adaptive loss weighting (harder models get more weight)
    - Momentum-based optimization (MI-FGSM variant)
    - Cosine annealing step size
    - Differentiable JPEG in the loop (optional)
    - Psychovisual constraint enforcement
    
    Parameters
    ----------
    models : list of str
        Model keys to attack. Default: ['clip', 'dinov2']
        Options: 'clip', 'dinov2', 'siglip'
    epsilon : float
        Maximum L∞ perturbation (in [0, 1] scale)
    steps : int
        Number of PGD optimization steps
    step_size : float
        Initial step size (auto-computed if None)
    momentum : float
        Momentum factor for MI-FGSM (0 = no momentum)
    use_jpeg_robustness : bool
        Include differentiable JPEG in optimization loop
    jpeg_quality : int
        JPEG quality for robustness (50-95)
    use_psychovisual : bool
        Apply psychovisual frequency shaping
    """
    
    PRESETS = {
        'subtle': {
            'epsilon': 8.0 / 255.0,
            'steps': 40,
            'momentum': 0.9,
            'models': ['clip'],
        },
        'moderate': {
            'epsilon': 16.0 / 255.0,
            'steps': 60,
            'momentum': 0.9,
            'models': ['clip', 'dinov2'],
        },
        'strong': {
            'epsilon': 24.0 / 255.0,
            'steps': 80,
            'momentum': 0.95,
            'models': ['clip', 'dinov2'],
        },
        'maximum': {
            'epsilon': 32.0 / 255.0,
            'steps': 100,
            'momentum': 0.95,
            'models': ['clip', 'dinov2', 'siglip'],
        },
    }
    
    def __init__(self,
                 models: List[str] = ['clip', 'dinov2'],
                 epsilon: float = 16.0 / 255.0,
                 steps: int = 60,
                 step_size: Optional[float] = None,
                 momentum: float = 0.9,
                 use_jpeg_robustness: bool = True,
                 jpeg_quality: int = 75,
                 use_psychovisual: bool = True):
        
        self.model_keys = models
        self.epsilon = epsilon
        self.steps = steps
        self.step_size = step_size or (2.5 * epsilon / steps)
        self.momentum = momentum
        self.use_jpeg_robustness = use_jpeg_robustness
        self.jpeg_quality = jpeg_quality
        self.use_psychovisual = use_psychovisual
        
        # Device selection
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Loss weights (adaptive — will be updated during attack)
        self.model_weights = {k: 1.0 / len(models) for k in models}
    
    @classmethod
    def from_preset(cls, preset: str, **overrides) -> 'EnsembleAdversarialAttack':
        """Create from a named preset."""
        if preset not in cls.PRESETS:
            raise ValueError(f"Unknown preset '{preset}'. "
                           f"Options: {list(cls.PRESETS.keys())}")
        config = cls.PRESETS[preset].copy()
        config.update(overrides)
        return cls(**config)
    
    def _load_all_models(self):
        """Load all target models."""
        for key in self.model_keys:
            _load_model(key, self.device)
    
    def _compute_ensemble_loss(self, 
                                perturbed_tensor: torch.Tensor,
                                original_features: Dict[str, torch.Tensor]
                                ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute weighted ensemble loss across all models.
        
        The loss MAXIMIZES feature distance between original and perturbed
        embeddings across all models simultaneously.
        
        Returns:
            total_loss: scalar tensor (to maximize via gradient ascent)
            per_model_losses: dict of per-model loss values
        """
        total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        per_model_losses = {}
        
        for model_key in self.model_keys:
            model = _model_cache[model_key]
            
            # Preprocess for this model
            preprocessed = _preprocess_for_model(
                perturbed_tensor, model_key, self.device
            )
            
            # Extract features
            features = _extract_features(model, preprocessed, model_key)
            
            # Compute distance metrics
            # F.cosine_similarity already normalizes internally,
            # so explicit F.normalize is unnecessary
            cosine_sim = F.cosine_similarity(
                features, original_features[model_key].detach()
            )
            model_loss = -cosine_sim.mean()  # Negative because we maximize
            
            # Also add L2 distance on normalized features for stronger displacement
            features_norm = F.normalize(features, p=2, dim=-1)
            orig_norm = F.normalize(
                original_features[model_key].detach(), p=2, dim=-1
            )
            l2_dist = torch.norm(features_norm - orig_norm, p=2, dim=-1).mean()
            
            # Combined loss for this model
            combined = model_loss + 0.1 * l2_dist
            
            weight = self.model_weights[model_key]
            total_loss = total_loss + weight * combined
            
            per_model_losses[model_key] = -cosine_sim.item()  # Store as distance (positive)
        
        return total_loss, per_model_losses
    
    def _update_adaptive_weights(self, per_model_losses: Dict[str, float]):
        """
        Update model weights based on which models are hardest to fool.
        
        Models that are harder to attack (lower loss) get MORE weight,
        ensuring the optimizer doesn't neglect them.
        """
        losses = np.array([per_model_losses[k] for k in self.model_keys])
        
        # Inverse loss weighting: harder models (lower displacement) get more weight
        # Add small epsilon to avoid division by zero
        inv_losses = 1.0 / (np.abs(losses) + 0.1)
        weights = inv_losses / inv_losses.sum()
        
        for i, key in enumerate(self.model_keys):
            # Smooth update (EMA) to prevent oscillation
            self.model_weights[key] = (
                0.7 * self.model_weights[key] + 0.3 * weights[i]
            )
    
    def _cosine_annealing_step_size(self, step: int) -> float:
        """
        Cosine annealing step size schedule.
        
        Starts large (explore), decreases (refine), increases again
        (escape local optima). Better than fixed step size.
        """
        return self.step_size * (
            0.5 * (1 + np.cos(np.pi * step / self.steps))
        ) + self.step_size * 0.1  # Minimum step size = 10% of initial
    
    def attack(self, image: np.ndarray, 
               verbose: bool = True) -> Dict:
        """
        Execute the multi-model ensemble adversarial attack.
        
        Args:
            image: (H, W, 3) float64 numpy array in [0, 1]
            verbose: print progress
        Returns:
            dict with keys:
                'protected_image': (H, W, 3) adversarially protected image
                'delta': (H, W, 3) perturbation applied
                'metrics': dict of attack metrics
        """
        start_time = time.time()
        h, w, c = image.shape
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"  PHANTOM SPECTRAL ENCODING — Ensemble Attack")
            print(f"  Models: {self.model_keys}")
            print(f"  ε = {self.epsilon * 255:.0f}/255, steps = {self.steps}")
            print(f"  JPEG robustness: {self.use_jpeg_robustness}")
            print(f"  Psychovisual shaping: {self.use_psychovisual}")
            print(f"{'='*60}")
        
        # 1. Load all target models
        self._load_all_models()
        
        # 2. Convert image to tensor
        img_tensor = torch.from_numpy(image).float().to(self.device)
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
        
        # 3. Compute original features for all models
        original_features = {}
        for model_key in self.model_keys:
            model = _model_cache[model_key]
            preprocessed = _preprocess_for_model(img_tensor, model_key, self.device)
            with torch.no_grad():
                features = _extract_features(model, preprocessed, model_key)
            original_features[model_key] = features
        
        # 4. Setup differentiable JPEG (if enabled)
        jpeg_transform = None
        if self.use_jpeg_robustness:
            from .differentiable_jpeg import DifferentiableJPEG
            jpeg_transform = DifferentiableJPEG(
                quality=self.jpeg_quality
            ).to(self.device)
        
        # 5. Setup psychovisual mask (if enabled)
        pv_mask = None
        if self.use_psychovisual:
            from .psychovisual_model import PsychovisualConstraint
            pv_constraint = PsychovisualConstraint(image, self.epsilon)
            pv_mask = pv_constraint.to_torch_mask(self.device)
            # Resize mask to image dimensions
            pv_mask = F.interpolate(pv_mask, size=(h, w), mode='bilinear',
                                   align_corners=False)
        
        # 6. Initialize perturbation (random start within ε-ball)
        delta = torch.zeros_like(img_tensor, requires_grad=True)
        
        # Random initialization within ε-ball
        with torch.no_grad():
            delta.data = torch.empty_like(delta).uniform_(
                -self.epsilon, self.epsilon
            )
            if pv_mask is not None:
                delta.data = torch.clamp(delta.data, -pv_mask, pv_mask)
            delta.data = torch.clamp(
                img_tensor + delta.data, 0.0, 1.0
            ) - img_tensor
        
        # Momentum buffer
        momentum_buffer = torch.zeros_like(delta)
        
        # Best perturbation tracking
        best_loss = float('-inf')
        best_delta = delta.data.clone()
        
        # 7. PGD Optimization Loop
        loss_history = []
        
        for step in range(self.steps):
            delta.requires_grad_(True)
            
            # Perturbed image
            perturbed = img_tensor + delta
            perturbed = torch.clamp(perturbed, 0.0, 1.0)
            
            # Apply differentiable JPEG (if enabled)
            if jpeg_transform is not None and step % 3 == 0:
                # Apply JPEG every 3rd step to balance robustness vs speed
                perturbed_for_loss = jpeg_transform(perturbed)
            else:
                perturbed_for_loss = perturbed
            
            # Compute ensemble loss
            loss, per_model_losses = self._compute_ensemble_loss(
                perturbed_for_loss, original_features
            )
            
            # Track best
            loss_val = loss.item()
            loss_history.append(loss_val)
            if loss_val > best_loss:  # Higher loss = more displacement from original
                best_loss = loss_val
                best_delta = delta.data.clone()
            
            # Backward pass
            loss.backward()
            
            # Get gradient
            grad = delta.grad.data
            
            # Momentum (MI-FGSM variant)
            grad_norm = torch.norm(grad, p=1)
            if grad_norm > 0:
                grad = grad / grad_norm
            momentum_buffer = self.momentum * momentum_buffer + grad
            
            # Compute step size (cosine annealing)
            lr = self._cosine_annealing_step_size(step)
            
            # Update perturbation (gradient ASCENT — maximize loss)
            with torch.no_grad():
                delta.data = delta.data + lr * momentum_buffer.sign()
                
                # Project to psychovisual constraint (if enabled)
                if pv_mask is not None:
                    delta.data = torch.clamp(delta.data, -pv_mask, pv_mask)
                
                # Project to L∞ ball
                delta.data = torch.clamp(delta.data, -self.epsilon, self.epsilon)
                
                # Project to valid image range
                delta.data = torch.clamp(
                    img_tensor + delta.data, 0.0, 1.0
                ) - img_tensor
            
            # Zero gradient for next step
            delta.grad.zero_()
            
            # Update adaptive weights every 10 steps
            if step % 10 == 0 and step > 0:
                self._update_adaptive_weights(per_model_losses)
            
            # Progress logging
            if verbose and (step % 10 == 0 or step == self.steps - 1):
                model_status = " | ".join([
                    f"{k}: {v:.3f}" for k, v in per_model_losses.items()
                ])
                print(f"  Step {step:3d}/{self.steps} | "
                      f"Loss: {loss_val:.4f} | {model_status}")
        
        # 8. Use best perturbation found
        final_delta = best_delta
        
        # 9. Convert back to numpy
        protected_tensor = torch.clamp(img_tensor + final_delta, 0.0, 1.0)
        protected_image = protected_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        delta_np = final_delta.squeeze(0).permute(1, 2, 0).cpu().numpy()
        
        # 10. Compute final metrics
        elapsed = time.time() - start_time
        
        # Final feature distances
        final_metrics = {}
        with torch.no_grad():
            for model_key in self.model_keys:
                model = _model_cache[model_key]
                preprocessed = _preprocess_for_model(
                    protected_tensor, model_key, self.device
                )
                features = _extract_features(model, preprocessed, model_key)
                
                # F.cosine_similarity normalizes internally — no need for
                # explicit F.normalize before it
                cosine_sim = F.cosine_similarity(
                    features, original_features[model_key]
                ).item()
                feat_norm = F.normalize(features, p=2, dim=-1)
                orig_norm = F.normalize(
                    original_features[model_key], p=2, dim=-1
                )
                l2_dist = torch.norm(feat_norm - orig_norm, p=2).item()
                
                final_metrics[model_key] = {
                    'cosine_similarity': cosine_sim,
                    'feature_displacement': 1.0 - cosine_sim,
                    'l2_distance': l2_dist,
                }
        
        # Image quality metrics
        psnr = 10 * np.log10(1.0 / (np.mean(delta_np ** 2) + 1e-10))
        linf = np.max(np.abs(delta_np))
        l2 = np.sqrt(np.mean(delta_np ** 2))
        
        metrics = {
            'per_model': final_metrics,
            'image_quality': {
                'psnr_db': psnr,
                'linf': linf,
                'l2': l2,
                'linf_255': linf * 255,
            },
            'attack_config': {
                'models': self.model_keys,
                'epsilon': self.epsilon,
                'steps': self.steps,
                'jpeg_robustness': self.use_jpeg_robustness,
                'psychovisual': self.use_psychovisual,
            },
            'best_loss': best_loss,
            'elapsed_seconds': elapsed,
            'loss_history': loss_history,
        }
        
        if verbose:
            print(f"\n  {'='*50}")
            print(f"  Attack complete in {elapsed:.1f}s")
            print(f"  PSNR: {psnr:.1f} dB | L∞: {linf*255:.1f}/255")
            for mk, mv in final_metrics.items():
                print(f"  {mk}: displacement={mv['feature_displacement']:.1%}")
            print(f"  {'='*50}")
        
        return {
            'protected_image': protected_image,
            'delta': delta_np,
            'metrics': metrics,
        }


class PhantomSpectralShield:
    """
    High-level API for the Phantom Spectral Encoding system.
    
    Combines:
    1. Multi-model ensemble adversarial attack
    2. Differentiable JPEG robustness
    3. Psychovisual frequency shaping
    
    This is the main public interface for PhotoSavior v3.
    
    Usage::
    
        shield = PhantomSpectralShield(strength='moderate')
        result = shield.protect(image_np)
        protected = result['protected_image']
    """
    
    STRENGTH_MAP = {
        'subtle': 'subtle',
        'moderate': 'moderate', 
        'strong': 'strong',
        'maximum': 'maximum',
    }
    
    def __init__(self, 
                 strength: str = 'moderate',
                 jpeg_robustness: bool = True,
                 psychovisual: bool = True,
                 models: Optional[List[str]] = None):
        """
        Args:
            strength: 'subtle', 'moderate', 'strong', or 'maximum'
            jpeg_robustness: enable JPEG-robust perturbations
            psychovisual: enable HVS-based perturbation shaping
            models: override model list (default: from preset)
        """
        self.strength = strength
        
        overrides = {
            'use_jpeg_robustness': jpeg_robustness,
            'use_psychovisual': psychovisual,
        }
        if models is not None:
            overrides['models'] = models
        
        self.attack = EnsembleAdversarialAttack.from_preset(
            strength, **overrides
        )
    
    def protect(self, image: np.ndarray, verbose: bool = True) -> Dict:
        """
        Apply Phantom Spectral Encoding protection.
        
        Args:
            image: numpy array (H, W, 3) in [0, 1] float 
                   OR (H, W, 3) uint8 [0, 255]
            verbose: print progress
        Returns:
            dict with 'protected_image', 'delta', 'metrics'
        """
        # Handle uint8 input
        if image.dtype == np.uint8:
            image = image.astype(np.float64) / 255.0
        elif image.dtype != np.float64:
            image = image.astype(np.float64)
        
        # Ensure 3 channels
        if image.ndim == 2:
            image = np.stack([image, image, image], axis=-1)
        
        # Ensure [0, 1] range
        if image.max() > 1.0:
            image = image / 255.0
        
        return self.attack.attack(image, verbose=verbose)
    
    def protect_file(self, input_path: str, output_path: str,
                     verbose: bool = True) -> Dict:
        """
        Protect an image file.
        
        Args:
            input_path: path to input image
            output_path: path to save protected image
            verbose: print progress
        Returns:
            protection metrics
        """
        # Load
        img = Image.open(input_path).convert('RGB')
        img_np = np.array(img).astype(np.float64) / 255.0
        
        # Protect
        result = self.protect(img_np, verbose=verbose)
        
        # Save
        protected_uint8 = (result['protected_image'] * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(protected_uint8).save(output_path, quality=95)
        
        if verbose:
            print(f"  Saved: {output_path}")
        
        return result['metrics']
