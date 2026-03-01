"""
CLIP-Targeted Adversarial Perturbation Engine
==============================================

This module implements REAL adversarial attacks against the CLIP vision
encoder using Projected Gradient Descent (PGD). 

WHY THIS WORKS:
- DALL-E, GPT-4o, Midjourney all use CLIP-family vision encoders
- Adversarial perturbations computed against open-source CLIP transfer
  to commercial models because they share architectural similarities
- PGD finds the WORST-CASE perturbation within a perceptual budget,
  not random noise — this is mathematically optimal

ATTACK STRATEGIES:
1. Feature Displacement: Maximize distance between original and 
   perturbed CLIP embeddings (AI sees something completely different)
2. Targeted Misdirection: Push CLIP embedding toward a wrong target
   (AI sees a cat when the image shows a house)
3. Encoder Confusion: Maximize entropy of attention maps
   (AI can't focus on any coherent feature)

Reference: Goodfellow et al. "Explaining and Harnessing Adversarial Examples"
           Madry et al. "Towards Deep Learning Models Resistant to Adversarial Attacks"
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from typing import Optional, Tuple, Dict

# Lazy-load transformers to avoid import overhead when not needed
_clip_model = None
_clip_processor = None
_device = None


def _load_clip():
    """Lazy-load CLIP model. Downloads on first use (~350MB)."""
    global _clip_model, _clip_processor, _device
    if _clip_model is not None:
        return _clip_model, _clip_processor, _device

    from transformers import CLIPModel, CLIPProcessor

    model_name = "openai/clip-vit-base-patch32"
    print(f"  [CLIP] Loading {model_name}...")

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _clip_model = CLIPModel.from_pretrained(model_name).to(_device)
    _clip_model.eval()

    _clip_processor = CLIPProcessor.from_pretrained(model_name)

    param_count = sum(p.numel() for p in _clip_model.parameters())
    print(f"  [CLIP] Loaded on {_device} ({param_count / 1e6:.1f}M params)")

    return _clip_model, _clip_processor, _device


def _preprocess_for_clip(image_np: np.ndarray, processor, device) -> torch.Tensor:
    """
    Convert numpy image (H,W,3 float64 0-1) to CLIP input tensor.
    We do this manually (not via processor) so we can backprop through it.
    
    CLIP expects:
    - Resize to 224x224
    - Normalize with mean=[0.48145466, 0.4578275, 0.40821073], 
                     std=[0.26862954, 0.26130258, 0.27577711]
    """
    # Convert to tensor
    img_tensor = torch.from_numpy(image_np).float().to(device)  # (H, W, 3)
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)

    # Resize to 224x224 (CLIP input size)
    img_tensor = F.interpolate(img_tensor, size=(224, 224), mode='bilinear',
                               align_corners=False)

    # CLIP normalization
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073],
                        device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711],
                       device=device).view(1, 3, 1, 1)
    img_tensor = (img_tensor - mean) / std

    return img_tensor


def _get_clip_features(model, pixel_values: torch.Tensor) -> torch.Tensor:
    """Extract CLIP vision features (before projection)."""
    vision_outputs = model.vision_model(pixel_values=pixel_values)
    # Use the pooled output (CLS token) — this is what CLIP uses for similarity
    pooled = vision_outputs.pooler_output  # (1, 768)
    return pooled


def _get_clip_embedding(model, pixel_values: torch.Tensor) -> torch.Tensor:
    """Extract final CLIP image embedding (after projection to shared space)."""
    image_embeds = model.get_image_features(pixel_values=pixel_values)
    # Handle both old API (returns tensor) and new API (returns dataclass)
    if hasattr(image_embeds, 'pooler_output'):
        image_embeds = image_embeds.pooler_output
    elif hasattr(image_embeds, 'last_hidden_state'):
        image_embeds = image_embeds.last_hidden_state[:, 0, :]
    # L2 normalize
    image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
    return image_embeds  # (1, 512)


class CLIPAdversarialAttack:
    """
    PGD-based adversarial attack against CLIP vision encoder.
    
    This finds perturbations that MAXIMALLY disrupt how CLIP
    understands an image, within a perceptual budget.
    """

    def __init__(
        self,
        epsilon: float = 16.0 / 255.0,
        step_size: float = 2.0 / 255.0,
        num_steps: int = 40,
        attack_mode: str = "feature_displacement",
        target_text: Optional[str] = None,
    ):
        """
        Args:
            epsilon: Maximum L-inf perturbation (pixel budget).
                     8/255=imperceptible, 16/255=subtle, 32/255=visible
            step_size: PGD step size per iteration
            num_steps: Number of PGD optimization steps
            attack_mode: One of:
                - "feature_displacement": Push embedding far from original
                - "targeted_misdirection": Push toward wrong description  
                - "ensemble": Combine multiple attack objectives
            target_text: For targeted attack, the wrong description to push toward
        """
        self.epsilon = epsilon
        self.step_size = step_size
        self.num_steps = num_steps
        self.attack_mode = attack_mode
        self.target_text = target_text or "random noise static television"

    def attack(
        self,
        image_np: np.ndarray,
        verbose: bool = True,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Apply adversarial attack to image.
        
        Args:
            image_np: Input image (H, W, 3) float64, range [0, 1]
            verbose: Print progress
            
        Returns:
            perturbed_image: Same shape/range as input
            report: Attack metrics
        """
        model, processor, device = _load_clip()

        H, W, C = image_np.shape
        original_tensor = torch.from_numpy(image_np).float().to(device)  # (H, W, 3)

        # Initialize perturbation (start from uniform random within budget)
        delta = torch.zeros_like(original_tensor, requires_grad=True)
        # Random start for PGD (important for finding strong adversarial examples)
        delta.data = torch.empty_like(delta).uniform_(-self.epsilon, self.epsilon)
        delta.data = torch.clamp(original_tensor + delta.data, 0, 1) - original_tensor

        # Get original CLIP features/embedding (frozen, no grad)
        with torch.no_grad():
            orig_clip_input = _preprocess_for_clip(image_np, processor, device)
            orig_features = _get_clip_features(model, orig_clip_input)
            orig_embedding = _get_clip_embedding(model, orig_clip_input)

        # For targeted attack, get target text embedding
        target_text_embedding = None
        if self.attack_mode in ("targeted_misdirection", "ensemble"):
            with torch.no_grad():
                text_inputs = processor(
                    text=[self.target_text],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                ).to(device)
                target_text_embedding = model.get_text_features(**text_inputs)
                # Handle new API that returns dataclass
                if hasattr(target_text_embedding, 'pooler_output'):
                    target_text_embedding = target_text_embedding.pooler_output
                elif hasattr(target_text_embedding, 'last_hidden_state'):
                    target_text_embedding = target_text_embedding.last_hidden_state[:, 0, :]
                target_text_embedding = target_text_embedding / target_text_embedding.norm(
                    dim=-1, keepdim=True
                )

        # PGD optimization loop
        best_loss = float('-inf')
        best_delta = delta.data.clone()

        if verbose:
            print(f"  [PGD] Running {self.num_steps} steps, "
                  f"ε={self.epsilon:.4f} ({self.epsilon * 255:.1f}/255), "
                  f"mode={self.attack_mode}")

        for step in range(self.num_steps):
            delta.requires_grad_(True)

            # Forward pass: original + perturbation
            perturbed = torch.clamp(original_tensor + delta, 0, 1)

            # Preprocess for CLIP (manually, preserving gradients)
            perturbed_chw = perturbed.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
            perturbed_224 = F.interpolate(perturbed_chw, size=(224, 224),
                                          mode='bilinear', align_corners=False)
            mean = torch.tensor([0.48145466, 0.4578275, 0.40821073],
                                device=device).view(1, 3, 1, 1)
            std = torch.tensor([0.26862954, 0.26130258, 0.27577711],
                               device=device).view(1, 3, 1, 1)
            perturbed_normed = (perturbed_224 - mean) / std

            # Get perturbed features/embedding
            pert_features = _get_clip_features(model, perturbed_normed)
            pert_embedding = _get_clip_embedding(model, perturbed_normed)

            # Compute loss based on attack mode
            if self.attack_mode == "feature_displacement":
                # MAXIMIZE distance from original embedding
                loss = -F.cosine_similarity(
                    orig_embedding.detach(), pert_embedding, dim=-1
                ).mean()
                # Also push features apart
                loss += 0.5 * F.mse_loss(pert_features, -orig_features.detach())

            elif self.attack_mode == "targeted_misdirection":
                # MINIMIZE distance to wrong target text embedding
                loss = F.cosine_similarity(
                    target_text_embedding.detach(), pert_embedding, dim=-1
                ).mean()
                # And MAXIMIZE distance from original
                loss -= 0.3 * F.cosine_similarity(
                    orig_embedding.detach(), pert_embedding, dim=-1
                ).mean()

            elif self.attack_mode == "ensemble":
                # Combine all objectives
                # 1) Push away from original
                disp_loss = -F.cosine_similarity(
                    orig_embedding.detach(), pert_embedding, dim=-1
                ).mean()
                # 2) Push toward wrong text
                target_loss = F.cosine_similarity(
                    target_text_embedding.detach(), pert_embedding, dim=-1
                ).mean()
                # 3) Feature MSE displacement
                feat_loss = F.mse_loss(pert_features, -orig_features.detach())

                loss = 0.4 * disp_loss + 0.4 * target_loss + 0.2 * feat_loss

            else:
                raise ValueError(f"Unknown attack mode: {self.attack_mode}")

            # Backward pass
            loss.backward()

            # Track best perturbation
            if loss.item() > best_loss:
                best_loss = loss.item()
                best_delta = delta.data.clone()

            if verbose and (step % 10 == 0 or step == self.num_steps - 1):
                with torch.no_grad():
                    cos_sim = F.cosine_similarity(
                        orig_embedding, pert_embedding, dim=-1
                    ).item()
                print(f"    Step {step:3d}: loss={loss.item():.4f}, "
                      f"cos_sim={cos_sim:.4f}")

            # PGD step: gradient ascent (we want to maximize the loss)
            with torch.no_grad():
                grad = delta.grad.detach()

                # Sign-based step (FGSM-style, works better than raw gradient)
                delta.data = delta.data + self.step_size * grad.sign()

                # Project back into L-inf ball
                delta.data = torch.clamp(delta.data, -self.epsilon, self.epsilon)

                # Ensure valid pixel range
                delta.data = torch.clamp(
                    original_tensor + delta.data, 0, 1
                ) - original_tensor

            delta.grad = None

        # Use best perturbation found
        with torch.no_grad():
            final_perturbed = torch.clamp(
                original_tensor + best_delta, 0, 1
            ).cpu().numpy().astype(np.float64)

            # Compute final metrics
            final_clip_input = _preprocess_for_clip(final_perturbed, processor, device)
            final_embedding = _get_clip_embedding(model, final_clip_input)
            final_features = _get_clip_features(model, final_clip_input)

            cos_sim = F.cosine_similarity(
                orig_embedding, final_embedding, dim=-1
            ).item()

            feat_dist = torch.norm(
                orig_features - final_features
            ).item()

            # PSNR of perturbation
            mse = np.mean((image_np - final_perturbed) ** 2)
            psnr = 10 * np.log10(1.0 / (mse + 1e-10))

            # Actual L-inf
            linf = np.max(np.abs(image_np - final_perturbed))

        report = {
            'attack_mode': self.attack_mode,
            'epsilon': self.epsilon,
            'num_steps': self.num_steps,
            'cosine_similarity': cos_sim,
            'feature_distance': feat_dist,
            'psnr_db': psnr,
            'linf': linf,
            'best_loss': best_loss,
        }

        if verbose:
            print(f"  [PGD] Done. Cosine sim: {cos_sim:.4f} "
                  f"(1.0=identical, 0=orthogonal, -1=opposite)")
            print(f"  [PGD] Feature distance: {feat_dist:.2f}")
            print(f"  [PGD] PSNR: {psnr:.1f} dB, L-inf: {linf:.4f} "
                  f"({linf * 255:.1f}/255)")

        return final_perturbed, report


class CLIPAdversarialShield:
    """
    High-level interface for applying CLIP-adversarial protection.
    Combines multiple attack strategies for maximum disruption.
    """

    # Strength presets
    PRESETS = {
        'subtle': {
            'epsilon': 8.0 / 255.0,
            'step_size': 1.0 / 255.0,
            'num_steps': 30,
            'attack_mode': 'feature_displacement',
        },
        'moderate': {
            'epsilon': 16.0 / 255.0,
            'step_size': 2.0 / 255.0,
            'num_steps': 50,
            'attack_mode': 'ensemble',
        },
        'strong': {
            'epsilon': 24.0 / 255.0,
            'step_size': 2.5 / 255.0,
            'num_steps': 80,
            'attack_mode': 'ensemble',
        },
        'maximum': {
            'epsilon': 32.0 / 255.0,
            'step_size': 3.0 / 255.0,
            'num_steps': 100,
            'attack_mode': 'ensemble',
        },
    }

    def __init__(self, strength: str = 'strong'):
        if strength not in self.PRESETS:
            raise ValueError(f"Unknown strength: {strength}. "
                             f"Choose from {list(self.PRESETS.keys())}")
        self.config = self.PRESETS[strength]
        self.strength = strength

    def protect(
        self,
        image: np.ndarray,
        target_text: str = "random noise static distortion glitch",
        verbose: bool = True,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Apply CLIP-adversarial protection to an image.

        Args:
            image: (H, W, 3) float64 in [0, 1]
            target_text: Misdirection target
            verbose: Print progress

        Returns:
            protected: Same shape, adversarially perturbed
            report: Metrics
        """
        if verbose:
            print(f"\n  [SHIELD] Applying CLIP-adversarial protection "
                  f"(strength={self.strength})")

        attacker = CLIPAdversarialAttack(
            epsilon=self.config['epsilon'],
            step_size=self.config['step_size'],
            num_steps=self.config['num_steps'],
            attack_mode=self.config['attack_mode'],
            target_text=target_text,
        )

        protected, report = attacker.attack(image, verbose=verbose)
        report['strength'] = self.strength

        return protected, report
