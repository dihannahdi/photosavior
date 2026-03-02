"""
Visual Proof Generator for Phantom Spectral Encoding
=====================================================

Generates comparison images and metrics tables that demonstrate
the effectiveness of all 4 novel contributions.

Produces:
  1. Before/after image comparisons
  2. Perturbation heatmaps (PFS vs uniform)
  3. JPEG survival chart data
  4. Multi-model displacement comparison
  5. Summary report with all metrics

Output: results/ directory with PNG images and report.txt
"""

import sys
import os
import io
import time
import json
import numpy as np
import warnings

warnings.filterwarnings('ignore')
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image, ImageDraw, ImageFont
from src.ensemble_attack import (EnsembleAdversarialAttack, _load_model,
                                 _preprocess_for_model, _extract_features,
                                 _model_cache)


def create_demo_image(h=256, w=256, seed=42):
    """Create a demo image with rich content."""
    rng = np.random.RandomState(seed)
    img = np.zeros((h, w, 3), dtype=np.float64)

    # Color gradient background
    for y in range(h):
        for x in range(w):
            img[y, x, 0] = 0.3 + 0.4 * (x / w)  # Red gradient
            img[y, x, 1] = 0.2 + 0.5 * (y / h)  # Green gradient
            img[y, x, 2] = 0.4 + 0.3 * ((x + y) / (w + h))  # Blue gradient

    # Add texture (gaussian noise in patches)
    img[30:100, 30:100] += rng.randn(70, 70, 3) * 0.08
    img[150:220, 150:220] += rng.randn(70, 70, 3) * 0.08

    # Bright and dark regions
    img[60:90, 160:220, :] = 0.95
    img[160:190, 30:90, :] = 0.05

    # Fine edges
    img[120:122, :, :] = 0.9
    img[:, 128:130, :] = 0.9

    # Circular shape
    cy, cx, r = 180, 180, 25
    Y, X = np.ogrid[:h, :w]
    mask = (X - cx)**2 + (Y - cy)**2 <= r**2
    img[mask, 0] = 0.8
    img[mask, 1] = 0.3
    img[mask, 2] = 0.3

    return np.clip(img, 0, 1)


def measure_displacement(original_np, protected_np, model_key='clip'):
    """Measure feature displacement for a given model."""
    import torch
    import torch.nn.functional as F

    device = torch.device('cpu')
    _load_model(model_key, device)
    model = _model_cache[model_key]

    def get_feat(img_np):
        t = torch.from_numpy(img_np).float().permute(2, 0, 1).unsqueeze(0)
        pre = _preprocess_for_model(t, model_key, device)
        with torch.no_grad():
            return _extract_features(model, pre, model_key)

    f_orig = get_feat(original_np)
    f_prot = get_feat(protected_np)
    cos = F.cosine_similarity(
        F.normalize(f_orig, p=2, dim=-1),
        F.normalize(f_prot, p=2, dim=-1)
    ).item()
    return 1.0 - cos


def jpeg_compress(image_np, quality=75):
    """Real PIL JPEG compression."""
    u8 = (image_np * 255).clip(0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(u8).save(buf, format='JPEG', quality=quality)
    buf.seek(0)
    return np.array(Image.open(buf)).astype(np.float64) / 255.0


def to_uint8(img_np):
    return (img_np * 255).clip(0, 255).astype(np.uint8)


def amplify_delta(delta, scale=10):
    """Amplify perturbation for visualization."""
    amp = np.abs(delta) * scale
    amp = amp / max(amp.max(), 1e-8)
    return (amp * 255).clip(0, 255).astype(np.uint8)


def generate_all_proofs(output_dir='results'):
    """Generate all visual proof artifacts."""
    os.makedirs(output_dir, exist_ok=True)
    report = []
    report.append("=" * 70)
    report.append("  PHANTOM SPECTRAL ENCODING - Visual Proof Report")
    report.append("=" * 70)
    report.append(f"  Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")

    img = create_demo_image(224, 224)

    # ================================================================
    # PROOF 1: Multi-Model Attack Effectiveness
    # ================================================================
    print("\n[1/5] Multi-Model Attack...")

    results = {}
    configs = {
        'clip_only': {'models': ['clip'], 'label': 'CLIP Only'},
        'clip_dinov2': {'models': ['clip', 'dinov2'], 'label': 'CLIP + DINOv2'},
        'all_three': {'models': ['clip', 'dinov2', 'siglip'], 'label': 'All 3 Models'},
    }

    for key, cfg in configs.items():
        attack = EnsembleAdversarialAttack(
            models=cfg['models'],
            epsilon=16.0 / 255,
            steps=25,
            use_jpeg_robustness=False,
            use_psychovisual=False,
        )
        result = attack.attack(img, verbose=False)
        results[key] = result

        # Save protected image
        Image.fromarray(to_uint8(result['protected_image'])).save(
            os.path.join(output_dir, f'protected_{key}.png'))

    # Save original
    Image.fromarray(to_uint8(img)).save(
        os.path.join(output_dir, 'original.png'))

    report.append("PROOF 1: Multi-Model Attack Effectiveness")
    report.append("-" * 50)
    report.append(f"{'Config':<20} {'CLIP':>8} {'DINOv2':>8} {'SigLIP':>8} {'PSNR':>8}")
    report.append("-" * 50)

    for key, result in results.items():
        metrics = result['metrics']
        pm = metrics['per_model']
        clip_d = pm.get('clip', {}).get('feature_displacement', 0)
        dino_d = pm.get('dinov2', {}).get('feature_displacement', 0)
        sig_d = pm.get('siglip', {}).get('feature_displacement', 0)
        psnr = metrics['image_quality']['psnr_db']
        label = configs[key]['label']
        report.append(f"{label:<20} {clip_d:>7.1%} {dino_d:>7.1%} {sig_d:>7.1%} {psnr:>6.1f}dB")

    # Cross-model transfer: measure DINOv2 displacement of CLIP-only attack
    clip_only_dino = measure_displacement(img, results['clip_only']['protected_image'], 'dinov2')
    clip_only_sig = measure_displacement(img, results['clip_only']['protected_image'], 'siglip')

    report.append("")
    report.append("Cross-Model Transfer (CLIP-only attack measured on other models):")
    report.append(f"  DINOv2 transfer: {clip_only_dino:.1%}")
    report.append(f"  SigLIP transfer: {clip_only_sig:.1%}")
    report.append("")

    # ================================================================
    # PROOF 2: JPEG Survival with DJRO
    # ================================================================
    print("[2/5] JPEG Survival...")

    attack_djro = EnsembleAdversarialAttack(
        models=['clip'], epsilon=16.0/255, steps=30,
        use_jpeg_robustness=True, jpeg_quality=75, use_psychovisual=False)
    result_djro = attack_djro.attack(img, verbose=False)

    attack_naive = EnsembleAdversarialAttack(
        models=['clip'], epsilon=16.0/255, steps=30,
        use_jpeg_robustness=False, use_psychovisual=False)
    result_naive = attack_naive.attack(img, verbose=False)

    report.append("PROOF 2: JPEG Survival (DJRO)")
    report.append("-" * 60)
    report.append(f"{'Quality':<8} {'Naive':>12} {'Naive%':>8} {'DJRO':>12} {'DJRO%':>8}")
    report.append("-" * 60)

    pre_djro = result_djro['metrics']['per_model']['clip']['feature_displacement']
    pre_naive = result_naive['metrics']['per_model']['clip']['feature_displacement']
    report.append(f"{'Pre-JPEG':<8} {pre_naive:>11.1%} {'100%':>8} {pre_djro:>11.1%} {'100%':>8}")

    for q in [95, 85, 75, 60, 50]:
        j_djro = jpeg_compress(result_djro['protected_image'], q)
        j_naive = jpeg_compress(result_naive['protected_image'], q)
        d_djro = measure_displacement(img, j_djro)
        d_naive = measure_displacement(img, j_naive)
        s_djro = d_djro / max(pre_djro, 1e-6) * 100
        s_naive = d_naive / max(pre_naive, 1e-6) * 100
        report.append(f"Q{q:<7} {d_naive:>11.1%} {s_naive:>7.0f}% {d_djro:>11.1%} {s_djro:>7.0f}%")

        if q == 75:
            Image.fromarray(to_uint8(j_djro)).save(
                os.path.join(output_dir, 'jpeg_q75_djro.png'))
            Image.fromarray(to_uint8(j_naive)).save(
                os.path.join(output_dir, 'jpeg_q75_naive.png'))

    report.append("")

    # ================================================================
    # PROOF 3: Psychovisual Frequency Shaping
    # ================================================================
    print("[3/5] Psychovisual Shaping...")

    attack_pfs = EnsembleAdversarialAttack(
        models=['clip'], epsilon=16.0/255, steps=25,
        use_jpeg_robustness=False, use_psychovisual=True)
    result_pfs = attack_pfs.attack(img, verbose=False)

    attack_nopfs = EnsembleAdversarialAttack(
        models=['clip'], epsilon=16.0/255, steps=25,
        use_jpeg_robustness=False, use_psychovisual=False)
    result_nopfs = attack_nopfs.attack(img, verbose=False)

    from skimage.metrics import structural_similarity as ssim

    ssim_pfs = ssim(img, result_pfs['protected_image'], channel_axis=2, data_range=1.0)
    ssim_nopfs = ssim(img, result_nopfs['protected_image'], channel_axis=2, data_range=1.0)
    psnr_pfs = result_pfs['metrics']['image_quality']['psnr_db']
    psnr_nopfs = result_nopfs['metrics']['image_quality']['psnr_db']
    disp_pfs = result_pfs['metrics']['per_model']['clip']['feature_displacement']
    disp_nopfs = result_nopfs['metrics']['per_model']['clip']['feature_displacement']

    report.append("PROOF 3: Psychovisual Frequency Shaping (PFS)")
    report.append("-" * 50)
    report.append(f"{'Metric':<20} {'Without PFS':>15} {'With PFS':>15} {'Delta':>10}")
    report.append("-" * 50)
    report.append(f"{'PSNR (dB)':<20} {psnr_nopfs:>14.1f} {psnr_pfs:>14.1f} {psnr_pfs-psnr_nopfs:>+9.1f}")
    report.append(f"{'SSIM':<20} {ssim_nopfs:>14.4f} {ssim_pfs:>14.4f} {ssim_pfs-ssim_nopfs:>+9.4f}")
    report.append(f"{'Displacement':<20} {disp_nopfs:>14.1%} {disp_pfs:>14.1%} {disp_pfs-disp_nopfs:>+8.1%}")
    report.append("")
    report.append(f"PFS BENEFIT: +{psnr_pfs-psnr_nopfs:.1f} dB PSNR, +{ssim_pfs-ssim_nopfs:.4f} SSIM")
    report.append(f"  while maintaining {disp_pfs:.1%} displacement (vs {disp_nopfs:.1%})")
    report.append("")

    # Save perturbation heatmaps
    delta_pfs = amplify_delta(result_pfs['delta'], scale=15)
    delta_nopfs = amplify_delta(result_nopfs['delta'], scale=15)
    Image.fromarray(delta_pfs).save(os.path.join(output_dir, 'delta_with_pfs.png'))
    Image.fromarray(delta_nopfs).save(os.path.join(output_dir, 'delta_without_pfs.png'))
    Image.fromarray(to_uint8(result_pfs['protected_image'])).save(
        os.path.join(output_dir, 'protected_with_pfs.png'))
    Image.fromarray(to_uint8(result_nopfs['protected_image'])).save(
        os.path.join(output_dir, 'protected_without_pfs.png'))

    # ================================================================
    # PROOF 4: PSF Format
    # ================================================================
    print("[4/5] PSF Format...")

    from src.psf_codec import save_psf, load_psf, verify_psf

    # Use the triple-model result
    best_result = results['all_three']
    psf_path = os.path.join(output_dir, 'protected.psf')
    save_info = save_psf(
        protected_image=best_result['protected_image'],
        output_path=psf_path,
        protection_level='moderate',
        original_image=img,
        metrics=best_result['metrics'],
    )

    loaded = load_psf(psf_path, verify=True)

    report.append("PROOF 4: PSF Format Integrity")
    report.append("-" * 50)
    report.append(f"File size: {save_info['file_size_bytes']:,} bytes")
    report.append(f"Compression ratio: {save_info['compression_ratio']:.1f}x")
    report.append(f"Pixel data: {save_info['pixel_data_raw']:,} -> {save_info['pixel_data_compressed']:,} bytes")
    report.append(f"Integrity check: {'PASSED' if loaded['integrity_valid'] else 'FAILED'}")
    report.append(f"Protection level: {loaded['protection_level']}")
    report.append(f"Lossless round-trip: {np.array_equal(loaded['image'], to_uint8(best_result['protected_image']))}")
    report.append("")

    # ================================================================
    # PROOF 5: Full Pipeline Metrics Summary
    # ================================================================
    print("[5/5] Summary metrics...")

    # Full pipeline test with all features
    from src.photosavior_v3 import PhotoSaviorV3

    engine = PhotoSaviorV3(
        strength='moderate',
        models=['clip', 'dinov2'],
        jpeg_robustness=True,
        psychovisual=True,
    )
    full_result = engine.protect(img, verbose=False)

    report.append("PROOF 5: Full Pipeline (All Features)")
    report.append("-" * 50)
    report.append(f"Strength: moderate")
    report.append(f"Models: CLIP + DINOv2")
    report.append(f"JPEG Robustness: ON")
    report.append(f"Psychovisual Shaping: ON")
    report.append(f"PSNR: {full_result.psnr:.1f} dB")
    for m, d in full_result.displacement.items():
        report.append(f"{m} displacement: {d:.1%}")

    # JPEG survival of full pipeline
    full_jpeg = jpeg_compress(full_result.image, 75)
    for model_key in ['clip', 'dinov2']:
        post_disp = measure_displacement(img, full_jpeg, model_key)
        report.append(f"{model_key} after Q75 JPEG: {post_disp:.1%}")

    report.append(f"Processing time: {full_result.metrics.get('total_elapsed_seconds', 0):.1f}s")
    report.append("")

    # Save full pipeline result
    Image.fromarray(full_result.image_uint8).save(
        os.path.join(output_dir, 'full_pipeline_result.png'))

    # ================================================================
    # Write report
    # ================================================================
    report.append("=" * 70)
    report.append("  ALL PROOFS GENERATED SUCCESSFULLY")
    report.append("=" * 70)

    report_text = "\n".join(report)
    report_path = os.path.join(output_dir, 'proof_report.txt')
    with open(report_path, 'w') as f:
        f.write(report_text)

    print(f"\n{report_text}")
    print(f"\nAll artifacts saved to: {output_dir}/")
    print(f"Report saved to: {report_path}")

    # List generated files
    files = sorted(os.listdir(output_dir))
    print(f"\nGenerated files ({len(files)}):")
    for f in files:
        size = os.path.getsize(os.path.join(output_dir, f))
        print(f"  {f:<40} {size:>10,} bytes")


if __name__ == '__main__':
    output_dir = sys.argv[1] if len(sys.argv) > 1 else 'results'
    generate_all_proofs(output_dir)
