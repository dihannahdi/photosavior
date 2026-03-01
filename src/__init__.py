"""
PhotoSavior - Multi-Spectral Adversarial Shield (MSAS)
======================================================

A novel image protection system that injects imperceptible adversarial
perturbations to prevent AI models from modifying photos.

Academic Foundation:
--------------------
This system builds upon and extends concepts from:
  - Glaze (Shan et al., 2023): Style cloaking via adversarial perturbations
  - PhotoGuard (Salman et al., 2023): Encoder/diffusion attack perturbations
  - AdvPaint (Jeon et al., 2025): Attention disruption in diffusion models
  - FGSM/PGD (Goodfellow et al., 2015; Madry et al., 2018): Adversarial ML

Novel Contributions of PhotoSavior (MSAS):
------------------------------------------
  1. Multi-Spectral Perturbation Fusion (MSPF): Simultaneously perturbs DCT,
     DWT, and FFT domains, creating perturbations that are resilient across
     different neural network preprocessing pipelines.

  2. Texture-Adaptive Perturbation Masking (TAPM): Uses local variance and
     edge detection to adaptively allocate perturbation budget to high-texture
     regions where human perception is least sensitive.

  3. Neural Feature Space Disruption (NFSD): Targets the latent space
     representations that AI models use internally, maximizing disruption
     of feature correlations.

  4. Forensic Watermark Embedding (FWE): Embeds a cryptographic hash
     in the DWT domain that survives JPEG compression, resizing, and
     mild cropping — proving ownership and detecting tampering.

  5. Cross-Architecture Transferability: The multi-spectral approach
     ensures perturbations transfer across VAE, ViT, CNN, and
     diffusion-based architectures.

Author: PhotoSavior Research Team
License: MIT
"""

__version__ = "1.0.0"
__author__ = "PhotoSavior Research"
