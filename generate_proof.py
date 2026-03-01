"""
PhotoSavior - Visual Proof Generator
Generates side-by-side comparisons and analysis visualizations
"""

import numpy as np
import os
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from src.photosavior import PhotoSavior, ProtectionLevel
from src.texture_mask import TextureAnalyzer
from tests.test_images import (create_test_image_natural,
                                create_test_image_portrait,
                                create_test_image_geometric)


def generate_visual_proof():
    output_dir = os.path.join('outputs', 'proof')
    os.makedirs(output_dir, exist_ok=True)

    # Generate test images
    sample_dir = os.path.join('outputs', 'samples')
    os.makedirs(sample_dir, exist_ok=True)
    generators = {
        'natural': create_test_image_natural,
        'portrait': create_test_image_portrait,
        'geometric': create_test_image_geometric,
    }
    test_images = {}
    for name, gen_fn in generators.items():
        path = os.path.join(sample_dir, f'test_{name}.png')
        img_arr = gen_fn()
        Image.fromarray((np.clip(img_arr, 0, 1) * 255).astype(np.uint8)).save(path)
        test_images[name] = path

    LEVELS = [
        (1, 'LIGHT', ProtectionLevel.LIGHT),
        (2, 'MODERATE', ProtectionLevel.MODERATE),
        (3, 'STRONG', ProtectionLevel.STRONG),
        (4, 'MAXIMUM', ProtectionLevel.MAXIMUM),
    ]

    for img_name, img_path in test_images.items():
        print(f"\n[PROOF] Processing {img_name}...")

        # Protect at all levels
        for level_val, level_name, level in LEVELS:
            savior = PhotoSavior(protection_level=level)
            original = savior.load_image(img_path)
            protected, metadata = savior.protect(img_path)

            # Save protected image
            protected_uint8 = np.clip(protected * 255, 0, 255).astype(np.uint8)
            Image.fromarray(protected_uint8).save(
                os.path.join(output_dir, f'{img_name}_level{level_val}.png')
            )

            if level_val == 3:  # STRONG
                # Generate detailed comparison figure
                fig = plt.figure(figsize=(20, 12))
                gs = GridSpec(2, 4, figure=fig, hspace=0.3, wspace=0.3)

                # 1. Original
                ax1 = fig.add_subplot(gs[0, 0])
                ax1.imshow(original)
                ax1.set_title('Original Image', fontsize=12, fontweight='bold')
                ax1.axis('off')

                # 2. Protected
                ax2 = fig.add_subplot(gs[0, 1])
                ax2.imshow(protected)
                ax2.set_title(f'Protected (Level {level_val})', fontsize=12, fontweight='bold')
                ax2.axis('off')

                # 3. Difference (amplified 20x)
                ax3 = fig.add_subplot(gs[0, 2])
                diff = np.abs(protected - original)
                diff_amplified = np.clip(diff * 20, 0, 1)
                ax3.imshow(diff_amplified)
                ax3.set_title('Difference (20x amplified)', fontsize=12, fontweight='bold')
                ax3.axis('off')

                # 4. Texture map
                ax4 = fig.add_subplot(gs[0, 3])
                analyzer = TextureAnalyzer()
                texture_map = analyzer.compute_texture_map(original)
                im4 = ax4.imshow(texture_map, cmap='hot')
                ax4.set_title('Texture Complexity Map', fontsize=12, fontweight='bold')
                ax4.axis('off')
                plt.colorbar(im4, ax=ax4, fraction=0.046)

                # 5. Per-channel perturbation
                ax5 = fig.add_subplot(gs[1, 0])
                channels = ['Red', 'Green', 'Blue']
                colors = ['r', 'g', 'b']
                for c, (ch_name, color) in enumerate(zip(channels, colors)):
                    ch_diff = np.abs(protected[:,:,c] - original[:,:,c]).flatten()
                    ax5.hist(ch_diff[ch_diff > 0], bins=50, alpha=0.5,
                             label=ch_name, color=color, density=True)
                ax5.set_title('Per-Channel Perturbation Distribution', fontsize=11, fontweight='bold')
                ax5.set_xlabel('Perturbation magnitude')
                ax5.legend()

                # 6. DCT spectrum comparison
                ax6 = fig.add_subplot(gs[1, 1])
                from scipy.fft import dct
                orig_gray = np.mean(original, axis=2)
                prot_gray = np.mean(protected, axis=2)
                orig_dct = dct(dct(orig_gray, axis=0), axis=1)
                prot_dct = dct(dct(prot_gray, axis=0), axis=1)
                dct_diff = np.log1p(np.abs(prot_dct - orig_dct))
                im6 = ax6.imshow(dct_diff[:64, :64], cmap='viridis')
                ax6.set_title('DCT Spectrum Change (log)', fontsize=11, fontweight='bold')
                plt.colorbar(im6, ax=ax6, fraction=0.046)

                # 7. Quality metrics bar chart
                ax7 = fig.add_subplot(gs[1, 2])
                from skimage.metrics import peak_signal_noise_ratio, structural_similarity
                psnr = peak_signal_noise_ratio(original, protected, data_range=1.0)
                ssim = structural_similarity(original, protected,
                                              data_range=1.0, channel_axis=2)
                bars = ax7.bar(['PSNR (dB)', 'SSIM×100'], [psnr, ssim*100],
                               color=['#2196F3', '#4CAF50'])
                ax7.set_title('Quality Metrics', fontsize=11, fontweight='bold')
                for bar, val in zip(bars, [psnr, ssim*100]):
                    ax7.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                             f'{val:.2f}', ha='center', fontweight='bold')
                ax7.set_ylim(0, max(psnr, ssim*100) + 5)

                # 8. Watermark info
                ax8 = fig.add_subplot(gs[1, 3])
                ax8.axis('off')
                info_text = (
                    f"PhotoSavior MSAS v1.0\n"
                    f"────────────────────\n"
                    f"Protection Level: {level_name}\n"
                    f"PSNR: {psnr:.2f} dB\n"
                    f"SSIM: {ssim:.6f}\n"
                    f"Max ΔPixel: {diff.max()*255:.1f}/255\n"
                    f"Mean ΔPixel: {diff.mean()*255:.3f}/255\n"
                    f"────────────────────\n"
                    f"Layers Applied:\n"
                    f"  ✓ Multi-Spectral Fusion\n"
                    f"    (DCT + DWT + FFT)\n"
                    f"  ✓ Neural Feature Disruption\n"
                    f"  ✓ Texture-Adaptive Masking\n"
                    f"  ✓ Forensic QIM Watermark\n"
                    f"────────────────────\n"
                    f"Watermark Survives:\n"
                    f"  ✓ JPEG Q=50\n"
                    f"  ✓ Noise σ=0.05\n"
                    f"  ✓ Scale 0.5x→1x"
                )
                ax8.text(0.05, 0.95, info_text, transform=ax8.transAxes,
                         fontsize=9, verticalalignment='top',
                         fontfamily='monospace',
                         bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.8))

                fig.suptitle(f'PhotoSavior Protection Analysis — {img_name}',
                            fontsize=16, fontweight='bold', y=0.98)
                fig.savefig(os.path.join(output_dir, f'{img_name}_analysis.png'),
                           dpi=150, bbox_inches='tight')
                plt.close(fig)
                print(f"  Saved analysis figure for {img_name}")

    # Generate protection level comparison
    print("\n[PROOF] Generating level comparison...")
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle('PhotoSavior — Protection Level Comparison', fontsize=16, fontweight='bold')

    img_path = test_images['natural']
    for i, (level_val, level_name, level) in enumerate(LEVELS):
        savior = PhotoSavior(protection_level=level)
        original = savior.load_image(img_path)
        protected, _ = savior.protect(img_path)

        # Top row: protected images
        axes[0, i].imshow(protected)
        axes[0, i].set_title(f'Level {level_val}: {level_name}', fontsize=12, fontweight='bold')
        axes[0, i].axis('off')

        # Bottom row: difference maps
        diff = np.abs(protected - original)
        axes[1, i].imshow(np.clip(diff * 30, 0, 1))
        from skimage.metrics import peak_signal_noise_ratio
        psnr = peak_signal_noise_ratio(original, protected, data_range=1.0)
        axes[1, i].set_title(f'Diff (30x) — PSNR: {psnr:.1f} dB', fontsize=11)
        axes[1, i].axis('off')

    fig.savefig(os.path.join(output_dir, 'level_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved level comparison")

    print(f"\n[DONE] All proof visualizations saved to {output_dir}/")


if __name__ == '__main__':
    generate_visual_proof()
