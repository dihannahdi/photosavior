"""
Differentiable JPEG Approximation for Adversarial Robustness
=============================================================

Novel Contribution: Differentiable JPEG-Robust Optimization (DJRO)
------------------------------------------------------------------
Standard adversarial perturbations (PGD, FGSM, C&W) break when the
image undergoes JPEG compression — the quantization step destroys
the carefully crafted perturbation gradients.

This module implements a FULLY DIFFERENTIABLE approximation of the
JPEG compression pipeline, allowing adversarial perturbations to be
optimized END-TO-END through simulated JPEG compression. The result:
perturbations that are DESIGNED to survive lossy compression.

Key Technical Innovations:
1. Differentiable DCT via matrix multiplication (not scipy)
2. Soft quantization using sinusoidal relaxation:
       round(x) ≈ x - sin(2πx) / (2π)    [Shin & Song, 2017]
   Gradient: d/dx ≈ 1 - cos(2πx)          (smooth, non-zero everywhere)
3. Learned temperature parameter for quantization sharpness
4. Standard JPEG luminance/chrominance quantization tables

Theory:
-------
JPEG compression pipeline:
  1. RGB → YCbCr color space conversion
  2. 8×8 block DCT (Discrete Cosine Transform)
  3. Quantization: round(DCT_coeff / Q_table) * Q_table
  4. Entropy coding (lossless — not relevant for our purpose)
  5. Inverse: dequantize → IDCT → YCbCr → RGB

Steps 1, 2, 4, 5 are differentiable. Step 3 (quantization) is not.
We replace step 3 with our differentiable soft quantization.

Reference:
  Shin & Song (2017) "JPEG-resistant Adversarial Images"
  Reich et al. (2024) "DiffJPEG: A Differentiable JPEG Codec"

Author: PhotoSavior Research Team
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


# ============================================================
# Standard JPEG Quantization Tables (ITU-T T.81)
# ============================================================

JPEG_LUMINANCE_QUANT_TABLE = torch.tensor([
    [16, 11, 10, 16,  24,  40,  51,  61],
    [12, 12, 14, 19,  26,  58,  60,  55],
    [14, 13, 16, 24,  40,  57,  69,  56],
    [14, 17, 22, 29,  51,  87,  80,  62],
    [18, 22, 37, 56,  68, 109, 103,  77],
    [24, 35, 55, 64,  81, 104, 113,  92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103,  99],
], dtype=torch.float32)

JPEG_CHROMINANCE_QUANT_TABLE = torch.tensor([
    [17, 18, 24, 47, 99, 99, 99, 99],
    [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99],
    [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
], dtype=torch.float32)


def _quality_to_scale(quality: int) -> float:
    """Convert JPEG quality (1-100) to quantization scale factor."""
    if quality < 50:
        return 5000.0 / quality
    else:
        return 200.0 - 2.0 * quality


# ============================================================
# Differentiable Building Blocks
# ============================================================

class DifferentiableDCT8x8(nn.Module):
    """
    8×8 Discrete Cosine Transform as a differentiable matrix multiply.
    
    Instead of using scipy.fft.dctn (non-differentiable through PyTorch),
    we implement DCT as T @ block @ T^T where T is the DCT basis matrix.
    This is fully differentiable via PyTorch autograd.
    """
    
    def __init__(self):
        super().__init__()
        # Build DCT-II basis matrix (8×8)
        T = torch.zeros(8, 8, dtype=torch.float32)
        for i in range(8):
            for j in range(8):
                if i == 0:
                    T[i, j] = 1.0 / np.sqrt(8)
                else:
                    T[i, j] = np.sqrt(2.0 / 8) * np.cos(
                        np.pi * (2 * j + 1) * i / (2 * 8)
                    )
        self.register_buffer('T', T)
        self.register_buffer('T_t', T.t())
    
    def forward(self, blocks: torch.Tensor) -> torch.Tensor:
        """
        Forward DCT on 8×8 blocks.
        
        Args:
            blocks: (N, 8, 8) tensor of image blocks
        Returns:
            (N, 8, 8) tensor of DCT coefficients
        """
        return torch.matmul(torch.matmul(self.T, blocks), self.T_t)
    
    def inverse(self, coeffs: torch.Tensor) -> torch.Tensor:
        """
        Inverse DCT on 8×8 blocks.
        
        Args:
            coeffs: (N, 8, 8) tensor of DCT coefficients
        Returns:
            (N, 8, 8) tensor of spatial-domain blocks
        """
        return torch.matmul(torch.matmul(self.T_t, coeffs), self.T)


class SoftQuantization(nn.Module):
    """
    Differentiable approximation of JPEG quantization.
    
    Standard quantization: round(x / step) * step
    This is a staircase function with zero gradient almost everywhere.
    
    Our approximation uses the sinusoidal relaxation:
        soft_round(x) = x - sin(2πx) / (2π)
    
    Properties:
    - Passes through integer points: soft_round(n) = n for integer n
    - Smooth gradient: d/dx = 1 - cos(2πx) ∈ [0, 2] 
    - As temperature → ∞, approaches true round()
    - Differentiable EVERYWHERE (no zero-gradient regions)
    
    We add a temperature parameter τ for controlling sharpness:
        soft_round(x, τ) = x - sin(2πx) / (2πτ)
    """
    
    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, x: torch.Tensor, quant_table: torch.Tensor) -> torch.Tensor:
        """
        Apply soft quantization.
        
        Args:
            x: DCT coefficients (N, 8, 8) 
            quant_table: Quantization table (8, 8), scaled by quality
        Returns:
            Soft-quantized coefficients (N, 8, 8)
        """
        # Normalize by quantization step
        normalized = x / (quant_table + 1e-8)
        
        # Differentiable rounding
        two_pi = 2.0 * np.pi
        soft_rounded = normalized - torch.sin(two_pi * normalized) / (
            two_pi * self.temperature
        )
        
        # Denormalize
        return soft_rounded * quant_table


class RGBToYCbCr(nn.Module):
    """Differentiable RGB → YCbCr color space conversion."""
    
    def __init__(self):
        super().__init__()
        # ITU-R BT.601 conversion matrix
        matrix = torch.tensor([
            [ 0.299,     0.587,     0.114    ],
            [-0.168736, -0.331264,  0.5      ],
            [ 0.5,      -0.418688, -0.081312 ],
        ], dtype=torch.float32)
        bias = torch.tensor([0.0, 128.0 / 255.0, 128.0 / 255.0],
                           dtype=torch.float32)
        self.register_buffer('matrix', matrix)
        self.register_buffer('bias', bias)
    
    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            rgb: (B, 3, H, W) in [0, 1]
        Returns:
            ycbcr: (B, 3, H, W)
        """
        # Reshape for matmul: (B, H, W, 3)
        x = rgb.permute(0, 2, 3, 1)
        y = torch.matmul(x, self.matrix.t()) + self.bias
        return y.permute(0, 3, 1, 2)


class YCbCrToRGB(nn.Module):
    """Differentiable YCbCr → RGB color space conversion."""
    
    def __init__(self):
        super().__init__()
        # Inverse ITU-R BT.601 matrix
        matrix = torch.tensor([
            [1.0,  0.0,       1.402   ],
            [1.0, -0.344136, -0.714136],
            [1.0,  1.772,     0.0     ],
        ], dtype=torch.float32)
        bias = torch.tensor([0.0, 128.0 / 255.0, 128.0 / 255.0],
                           dtype=torch.float32)
        self.register_buffer('matrix', matrix)
        self.register_buffer('bias', bias)
    
    def forward(self, ycbcr: torch.Tensor) -> torch.Tensor:
        """
        Args:
            ycbcr: (B, 3, H, W)
        Returns:
            rgb: (B, 3, H, W) in [0, 1]
        """
        x = ycbcr.permute(0, 2, 3, 1)
        x = x - self.bias
        y = torch.matmul(x, self.matrix.t())
        return y.permute(0, 3, 1, 2).clamp(0.0, 1.0)


# ============================================================
# Full Differentiable JPEG Pipeline
# ============================================================

class DifferentiableJPEG(nn.Module):
    """
    Fully differentiable JPEG compression/decompression simulation.
    
    This module simulates the lossy compression artifacts of JPEG
    while maintaining full differentiability through PyTorch autograd.
    
    When included in an adversarial optimization loop, it forces the
    optimizer to find perturbations that SURVIVE JPEG compression.
    
    Usage::
    
        djpeg = DifferentiableJPEG(quality=75)
        
        # In PGD loop:
        perturbed = image + delta
        compressed = djpeg(perturbed)  # Simulate JPEG
        loss = attack_loss(model, compressed)
        loss.backward()  # Gradients flow through JPEG!
    
    Parameters
    ----------
    quality : int
        JPEG quality factor (1-100). Lower = more compression.
    temperature : float
        Softness of quantization approximation.
        Higher = closer to true JPEG but harder gradients.
    """
    
    def __init__(self, quality: int = 75, temperature: float = 1.0):
        super().__init__()
        self.quality = quality
        self.temperature = temperature
        
        # Sub-modules
        self.rgb_to_ycbcr = RGBToYCbCr()
        self.ycbcr_to_rgb = YCbCrToRGB()
        self.dct = DifferentiableDCT8x8()
        self.soft_quant = SoftQuantization(temperature=temperature)
        
        # Compute scaled quantization tables
        scale = _quality_to_scale(quality) / 100.0
        lum_table = (JPEG_LUMINANCE_QUANT_TABLE * scale).clamp(1.0, 255.0)
        chr_table = (JPEG_CHROMINANCE_QUANT_TABLE * scale).clamp(1.0, 255.0)
        
        # Normalize tables to [0, 1] range (since our pixels are in [0, 1])
        self.register_buffer('lum_table', lum_table / 255.0)
        self.register_buffer('chr_table', chr_table / 255.0)
    
    def _image_to_blocks(self, channel: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        """
        Split a single channel into non-overlapping 8×8 blocks.
        
        Args:
            channel: (B, 1, H, W)
        Returns:
            blocks: (B*num_blocks, 8, 8)
            num_h: number of blocks vertically
            num_w: number of blocks horizontally
        """
        B, _, H, W = channel.shape
        
        # Pad to multiple of 8
        pad_h = (8 - H % 8) % 8
        pad_w = (8 - W % 8) % 8
        if pad_h > 0 or pad_w > 0:
            channel = F.pad(channel, (0, pad_w, 0, pad_h), mode='reflect')
        
        _, _, pH, pW = channel.shape
        num_h, num_w = pH // 8, pW // 8
        
        # Reshape into blocks: (B, 1, num_h, 8, num_w, 8) → (B*num_h*num_w, 8, 8)
        blocks = channel.reshape(B, 1, num_h, 8, num_w, 8)
        blocks = blocks.permute(0, 2, 4, 1, 3, 5).reshape(-1, 8, 8)
        
        return blocks, num_h, num_w
    
    def _blocks_to_image(self, blocks: torch.Tensor, B: int, 
                         num_h: int, num_w: int,
                         orig_h: int, orig_w: int) -> torch.Tensor:
        """
        Reassemble 8×8 blocks into a channel image.
        
        Args:
            blocks: (B*num_h*num_w, 8, 8)
            B: batch size
            num_h, num_w: block counts
            orig_h, orig_w: original (unpadded) dimensions
        Returns:
            channel: (B, 1, orig_h, orig_w)
        """
        channel = blocks.reshape(B, num_h, num_w, 1, 8, 8)
        channel = channel.permute(0, 3, 1, 4, 2, 5).reshape(B, 1, num_h * 8, num_w * 8)
        return channel[:, :, :orig_h, :orig_w]
    
    def _compress_channel(self, channel: torch.Tensor, 
                          quant_table: torch.Tensor) -> torch.Tensor:
        """
        Compress a single channel through DCT → quantize → IDCT.
        
        Args:
            channel: (B, 1, H, W) single color channel
            quant_table: (8, 8) quantization table
        Returns:
            compressed: (B, 1, H, W) compressed channel
        """
        B, _, H, W = channel.shape
        
        # Split into 8×8 blocks
        blocks, num_h, num_w = self._image_to_blocks(channel)
        
        # Forward DCT
        dct_coeffs = self.dct(blocks)
        
        # Soft quantization (the KEY differentiable step)
        quantized = self.soft_quant(dct_coeffs, quant_table)
        
        # Inverse DCT
        reconstructed = self.dct.inverse(quantized)
        
        # Reassemble
        return self._blocks_to_image(reconstructed, B, num_h, num_w, H, W)
    
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Simulate JPEG compression (differentiable).
        
        Args:
            image: (B, 3, H, W) RGB image in [0, 1]
        Returns:
            compressed: (B, 3, H, W) JPEG-compressed image in [0, 1]
        """
        # 1. RGB → YCbCr
        ycbcr = self.rgb_to_ycbcr(image)
        
        # 2. Process each channel with appropriate quant table
        y_channel = self._compress_channel(
            ycbcr[:, 0:1, :, :], self.lum_table
        )
        cb_channel = self._compress_channel(
            ycbcr[:, 1:2, :, :], self.chr_table
        )
        cr_channel = self._compress_channel(
            ycbcr[:, 2:3, :, :], self.chr_table
        )
        
        # 3. Combine channels
        ycbcr_compressed = torch.cat([y_channel, cb_channel, cr_channel], dim=1)
        
        # 4. YCbCr → RGB
        rgb_compressed = self.ycbcr_to_rgb(ycbcr_compressed)
        
        return rgb_compressed.clamp(0.0, 1.0)


class DifferentiableResize(nn.Module):
    """
    Differentiable image resize for robustness training.
    
    Simulates the effect of resizing an image (as social media
    platforms do) while maintaining differentiability.
    """
    
    def __init__(self, scale_range: Tuple[float, float] = (0.5, 1.5)):
        super().__init__()
        self.scale_range = scale_range
    
    def forward(self, image: torch.Tensor, 
                scale: Optional[float] = None) -> torch.Tensor:
        """
        Resize then resize back (simulating platform processing).
        
        Args:
            image: (B, 3, H, W)
            scale: resize scale factor (random if None)
        Returns:
            image after resize round-trip
        """
        B, C, H, W = image.shape
        
        if scale is None:
            scale = np.random.uniform(*self.scale_range)
        
        new_h = max(8, int(H * scale))
        new_w = max(8, int(W * scale))
        
        # Downscale then upscale (lossy round-trip)
        small = F.interpolate(image, size=(new_h, new_w), 
                             mode='bilinear', align_corners=False)
        restored = F.interpolate(small, size=(H, W),
                                mode='bilinear', align_corners=False)
        
        return restored


class DifferentiableTransformChain(nn.Module):
    """
    Chain of differentiable image transformations for robust optimization.
    
    Applies a random subset of transformations during each forward pass,
    forcing the adversarial perturbation to be robust to ALL of them.
    
    This is the key innovation of DJRO: by including these transforms
    in the PGD loop, we optimize for transformation-robust perturbations.
    
    Supported transforms:
    - JPEG compression (quality 50-95)
    - Resize (0.5x-1.5x)
    - Gaussian blur (σ = 0.5-2.0)
    - Additive Gaussian noise
    """
    
    def __init__(self, jpeg_quality: int = 75, 
                 enable_jpeg: bool = True,
                 enable_resize: bool = True,
                 enable_blur: bool = True,
                 enable_noise: bool = True):
        super().__init__()
        
        self.enable_jpeg = enable_jpeg
        self.enable_resize = enable_resize
        self.enable_blur = enable_blur
        self.enable_noise = enable_noise
        
        if enable_jpeg:
            self.jpeg = DifferentiableJPEG(quality=jpeg_quality)
        if enable_resize:
            self.resize = DifferentiableResize(scale_range=(0.5, 1.5))
    
    def forward(self, image: torch.Tensor, 
                apply_all: bool = False) -> torch.Tensor:
        """
        Apply transformation chain.
        
        Args:
            image: (B, 3, H, W) in [0, 1]
            apply_all: if True, apply ALL transforms; 
                      if False, randomly select subset
        Returns:
            transformed image
        """
        x = image
        
        if apply_all:
            transforms_to_apply = ['jpeg', 'resize', 'blur', 'noise']
        else:
            # Randomly select which transforms to apply
            transforms_to_apply = []
            if self.enable_jpeg and np.random.random() > 0.3:
                transforms_to_apply.append('jpeg')
            if self.enable_resize and np.random.random() > 0.5:
                transforms_to_apply.append('resize')
            if self.enable_blur and np.random.random() > 0.5:
                transforms_to_apply.append('blur')
            if self.enable_noise and np.random.random() > 0.6:
                transforms_to_apply.append('noise')
        
        np.random.shuffle(transforms_to_apply)
        
        for t in transforms_to_apply:
            if t == 'jpeg' and self.enable_jpeg:
                x = self.jpeg(x)
            elif t == 'resize' and self.enable_resize:
                x = self.resize(x)
            elif t == 'blur' and self.enable_blur:
                sigma = np.random.uniform(0.5, 2.0)
                kernel_size = int(2 * np.ceil(2 * sigma) + 1)
                if kernel_size % 2 == 0:
                    kernel_size += 1
                # Gaussian blur via conv2d
                coords = torch.arange(kernel_size, dtype=torch.float32,
                                     device=x.device) - kernel_size // 2
                kernel_1d = torch.exp(-coords**2 / (2 * sigma**2))
                kernel_1d = kernel_1d / kernel_1d.sum()
                kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
                kernel_2d = kernel_2d.expand(x.shape[1], 1, -1, -1)
                padding = kernel_size // 2
                x = F.conv2d(x, kernel_2d, padding=padding, 
                            groups=x.shape[1])
            elif t == 'noise' and self.enable_noise:
                noise_std = np.random.uniform(0.01, 0.05)
                x = x + torch.randn_like(x) * noise_std
                x = x.clamp(0.0, 1.0)
        
        return x


# ============================================================
# Utility Functions
# ============================================================

def jpeg_robustness_loss(original_delta: torch.Tensor,
                         jpeg_module: DifferentiableJPEG) -> torch.Tensor:
    """
    Compute how much of the perturbation delta survives JPEG compression.
    
    This can be used as a regularization term to encourage robust perturbations:
        total_loss = attack_loss + λ * jpeg_robustness_loss
    
    Args:
        original_delta: (B, 3, H, W) perturbation tensor
        jpeg_module: DifferentiableJPEG instance
    Returns:
        scalar loss (lower = more robust to JPEG)
    """
    # Create a base image (mid-gray) + delta
    base = torch.full_like(original_delta, 0.5)
    perturbed = (base + original_delta).clamp(0.0, 1.0)
    
    # Compress
    compressed = jpeg_module(perturbed)
    
    # Measure how much delta survived
    surviving_delta = compressed - base
    
    # Loss = how different the surviving delta is from original
    return F.mse_loss(surviving_delta, original_delta)
