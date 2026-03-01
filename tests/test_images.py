"""
Test Image Generator
====================

Creates synthetic test images with known properties for
rigorous testing of the PhotoSavior protection system.
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os


def create_test_image_natural(width: int = 512, height: int = 512,
                                seed: int = 42) -> np.ndarray:
    """
    Create a synthetic "natural-looking" test image with:
    - Smooth gradients (sky-like regions)
    - Textured areas (ground-like regions)
    - Sharp edges (object boundaries)
    - Fine details (texture patterns)

    This tests all aspects of the perceptual masking system.
    """
    rng = np.random.RandomState(seed)
    img = np.zeros((height, width, 3), dtype=np.float64)

    # Sky gradient (smooth region - perturbations most visible here)
    sky_height = height // 3
    for y in range(sky_height):
        t = y / sky_height
        img[y, :, 0] = 0.3 + 0.3 * t  # R
        img[y, :, 1] = 0.5 + 0.2 * t  # G
        img[y, :, 2] = 0.8 - 0.1 * t  # B

    # Ground with texture (textured region - perturbations hidden here)
    for y in range(sky_height, height):
        t = (y - sky_height) / (height - sky_height)
        base_r = 0.3 + 0.2 * t
        base_g = 0.5 - 0.2 * t
        base_b = 0.2 + 0.1 * t

        # Add grass-like texture
        texture = rng.randn(width) * 0.05
        img[y, :, 0] = base_r + texture * 0.5
        img[y, :, 1] = base_g + texture
        img[y, :, 2] = base_b + texture * 0.3

    # Add a "sun" (circular gradient)
    cy, cx = height // 5, width * 3 // 4
    Y, X = np.ogrid[:height, :width]
    dist = np.sqrt((Y - cy)**2 + (X - cx)**2)
    sun_mask = np.exp(-dist**2 / (2 * 40**2))
    img[:, :, 0] += sun_mask * 0.5
    img[:, :, 1] += sun_mask * 0.4
    img[:, :, 2] += sun_mask * 0.1

    # Add a "building" (sharp rectangular edges)
    bx1, bx2 = width // 4, width // 4 + width // 6
    by1, by2 = sky_height - height // 5, sky_height + height // 8
    img[by1:by2, bx1:bx2, 0] = 0.4
    img[by1:by2, bx1:bx2, 1] = 0.35
    img[by1:by2, bx1:bx2, 2] = 0.3

    # Windows on building (fine detail)
    for wy in range(by1 + 10, by2 - 10, 20):
        for wx in range(bx1 + 10, bx2 - 10, 15):
            ws = 8
            img[wy:wy+ws, wx:wx+ws, 0] = 0.8
            img[wy:wy+ws, wx:wx+ws, 1] = 0.9
            img[wy:wy+ws, wx:wx+ws, 2] = 0.5

    # Add a "tree" (complex organic shape)
    tx, ty = width * 2 // 3, sky_height
    # Trunk
    img[ty:ty+60, tx-5:tx+5, :] = np.array([0.3, 0.2, 0.1])
    # Canopy (irregular shape using random circles)
    for _ in range(20):
        cx_t = tx + rng.randint(-30, 30)
        cy_t = ty - rng.randint(20, 70)
        r = rng.randint(10, 25)
        Y2, X2 = np.ogrid[:height, :width]
        tree_mask = ((Y2 - cy_t)**2 + (X2 - cx_t)**2) < r**2
        green_val = 0.2 + rng.random() * 0.3
        img[tree_mask, 1] = green_val
        img[tree_mask, 0] = green_val * 0.3
        img[tree_mask, 2] = green_val * 0.1

    # Add fine noise texture everywhere (simulates sensor noise)
    noise = rng.randn(height, width, 3) * 0.01
    img += noise

    return np.clip(img, 0.0, 1.0)


def create_test_image_portrait(width: int = 512, height: int = 512,
                                 seed: int = 43) -> np.ndarray:
    """
    Create a simplified portrait-like test image.
    This simulates the type of image an adversary might try to
    AI-modify (face swap, style transfer, etc.).
    """
    rng = np.random.RandomState(seed)
    img = np.ones((height, width, 3), dtype=np.float64) * 0.9

    # Background gradient
    for y in range(height):
        t = y / height
        img[y, :, 0] = 0.85 - 0.1 * t
        img[y, :, 1] = 0.85 - 0.15 * t
        img[y, :, 2] = 0.9 - 0.05 * t

    # Face (oval)
    cx, cy = width // 2, height // 2 - 30
    Y, X = np.ogrid[:height, :width]
    face_mask = ((X - cx)**2 / 80**2 + (Y - cy)**2 / 100**2) < 1

    # Skin tone
    skin = np.array([0.85, 0.72, 0.6])
    img[face_mask] = skin + rng.randn(face_mask.sum(), 3) * 0.02

    # Eyes (two small circles)
    for ex in [cx - 25, cx + 25]:
        ey = cy - 15
        eye_mask = ((X - ex)**2 + (Y - ey)**2) < 8**2
        img[eye_mask] = np.array([0.2, 0.15, 0.1])
        # Iris
        iris_mask = ((X - ex)**2 + (Y - ey)**2) < 4**2
        img[iris_mask] = np.array([0.3, 0.5, 0.3])

    # Nose and mouth (simple lines)
    nose_x = cx
    for ny in range(cy - 5, cy + 20):
        if 0 <= ny < height:
            img[ny, nose_x-1:nose_x+1, :] *= 0.9

    # Mouth
    mouth_y = cy + 35
    mouth_w = 20
    for mx in range(cx - mouth_w, cx + mouth_w):
        if 0 <= mx < width and 0 <= mouth_y < height:
            img[mouth_y, mx, :] = np.array([0.7, 0.3, 0.3])

    # Hair
    hair_mask = ((X - cx)**2 / 90**2 + (Y - (cy - 80))**2 / 50**2) < 1
    hair_mask = hair_mask & (Y < cy - 30)
    img[hair_mask] = np.array([0.15, 0.1, 0.05]) + rng.randn(hair_mask.sum(), 3) * 0.03

    # Add subtle skin texture
    texture = rng.randn(height, width) * 0.008
    for ch in range(3):
        img[:, :, ch] += texture * (0.5 if ch == 1 else 1.0)

    return np.clip(img, 0.0, 1.0)


def create_test_image_geometric(width: int = 512, height: int = 512) -> np.ndarray:
    """
    Create a geometric test pattern with known frequency content.
    Useful for validating spectral perturbation behavior.
    """
    img = np.zeros((height, width, 3), dtype=np.float64)
    X, Y = np.meshgrid(np.linspace(0, 1, width),
                        np.linspace(0, 1, height))

    # Varying frequency sinusoids
    img[:, :, 0] = 0.5 + 0.3 * np.sin(2 * np.pi * 8 * X) * \
                    np.cos(2 * np.pi * 4 * Y)
    img[:, :, 1] = 0.5 + 0.3 * np.sin(2 * np.pi * 12 * X + np.pi/4) * \
                    np.cos(2 * np.pi * 6 * Y)
    img[:, :, 2] = 0.5 + 0.3 * np.sin(2 * np.pi * 4 * X) * \
                    np.cos(2 * np.pi * 10 * Y + np.pi/3)

    # Checkerboard overlay
    checker_size = 32
    checker = ((X * width // checker_size).astype(int) +
               (Y * height // checker_size).astype(int)) % 2
    img += checker[:, :, np.newaxis] * 0.05

    return np.clip(img, 0.0, 1.0)


def save_test_images(output_dir: str = "samples") -> dict:
    """Generate and save all test images."""
    os.makedirs(output_dir, exist_ok=True)

    images = {
        'natural': create_test_image_natural(),
        'portrait': create_test_image_portrait(),
        'geometric': create_test_image_geometric(),
    }

    paths = {}
    for name, img in images.items():
        path = os.path.join(output_dir, f'test_{name}.png')
        img_uint8 = (img * 255).astype(np.uint8)
        Image.fromarray(img_uint8, 'RGB').save(path)
        paths[name] = path
        print(f"  Created: {path} ({img.shape[1]}x{img.shape[0]})")

    return paths


if __name__ == '__main__':
    save_test_images()
