"""
PhotoSavior — Manual Testing Guide for Real AI Services
=========================================================

This script creates protected test images and provides step-by-step
instructions for testing them against real generative AI services
(ChatGPT/DALL-E, Grok, Midjourney, Stable Diffusion, etc.)

It generates:
  1. Original test images 
  2. Protected versions at all 4 levels
  3. Side-by-side comparison cards you can upload
  4. A testing checklist with specific prompts to try

USAGE:
  python manual_test_guide.py

Then follow the printed instructions to test with each AI service.
"""

import os
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.photosavior import PhotoSavior, ProtectionLevel


def create_test_photo(width=768, height=768):
    """
    Create a realistic-looking test photo with a face-like subject,
    background, and varied texture — the kind of image people actually
    want to protect from AI modification.
    """
    img = np.zeros((height, width, 3), dtype=np.float64)
    
    # Sky gradient background
    for y in range(height):
        t = y / height
        img[y, :, 0] = 0.3 + 0.4 * (1 - t)  # R
        img[y, :, 1] = 0.5 + 0.3 * (1 - t)  # G
        img[y, :, 2] = 0.7 + 0.2 * (1 - t)  # B

    # Ground / landscape
    ground_start = int(height * 0.55)
    for y in range(ground_start, height):
        t = (y - ground_start) / (height - ground_start)
        img[y, :, 0] = 0.3 + 0.15 * t
        img[y, :, 1] = 0.5 - 0.2 * t
        img[y, :, 2] = 0.2 + 0.1 * t
    
    # Add some "clouds"
    rng = np.random.RandomState(123)
    for _ in range(8):
        cx, cy = rng.randint(50, width-50), rng.randint(30, int(height*0.4))
        rx, ry = rng.randint(40, 120), rng.randint(15, 40)
        for y in range(max(0, cy-ry), min(height, cy+ry)):
            for x in range(max(0, cx-rx), min(width, cx+rx)):
                d = ((x-cx)/rx)**2 + ((y-cy)/ry)**2
                if d < 1:
                    alpha = (1 - d) * 0.5
                    img[y, x] = img[y, x] * (1-alpha) + np.array([0.95, 0.95, 0.97]) * alpha

    # Simple circular "portrait" subject
    cx, cy = width // 2, int(height * 0.35)
    face_r = 80
    for y in range(max(0, cy - face_r - 10), min(height, cy + face_r + 10)):
        for x in range(max(0, cx - face_r - 10), min(width, cx + face_r + 10)):
            d = np.sqrt((x-cx)**2 + (y-cy)**2)
            if d < face_r:
                # Skin-like tone
                img[y, x] = np.array([0.85, 0.72, 0.58])
            elif d < face_r + 5:
                alpha = (face_r + 5 - d) / 5
                img[y, x] = img[y, x] * (1-alpha) + np.array([0.85, 0.72, 0.58]) * alpha

    # Body/torso
    body_top = cy + face_r
    body_bottom = min(height, body_top + 200)
    for y in range(body_top, body_bottom):
        half_w = 90 + int((y - body_top) * 0.3)
        for x in range(max(0, cx - half_w), min(width, cx + half_w)):
            img[y, x] = np.array([0.2, 0.3, 0.5])  # Blue shirt

    # Add texture/noise for realism
    noise = rng.randn(height, width, 3) * 0.015
    img = np.clip(img + noise, 0, 1)

    return img


def main():
    output_dir = os.path.join('outputs', 'manual_test')
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("  PHOTOSAVIOR — MANUAL AI SERVICE TESTING GUIDE")
    print("=" * 70)

    # ─────────────────────────────────────────────
    # Step 1: Create test images
    # ─────────────────────────────────────────────
    print("\n[1/3] Creating test images...")
    
    test_photo = create_test_photo()
    test_photo_uint8 = (test_photo * 255).astype(np.uint8)
    
    # Save original
    orig_path = os.path.join(output_dir, '01_ORIGINAL.png')
    Image.fromarray(test_photo_uint8).save(orig_path)
    print(f"  Saved: {orig_path}")

    # Also let user supply their own photo
    print("\n  TIP: You can also test with your OWN photo!")
    print("  Just place it as: outputs/manual_test/my_photo.png")
    
    own_photo_path = os.path.join(output_dir, 'my_photo.png')
    photos_to_protect = [('synthetic', orig_path)]
    if os.path.exists(own_photo_path):
        photos_to_protect.append(('personal', own_photo_path))
        print(f"  Found your photo: {own_photo_path}")

    # ─────────────────────────────────────────────
    # Step 2: Generate protected versions
    # ─────────────────────────────────────────────
    print("\n[2/3] Generating protected versions at all levels...")
    
    levels = [
        (ProtectionLevel.LIGHT, 'LIGHT'),
        (ProtectionLevel.MODERATE, 'MODERATE'),
        (ProtectionLevel.STRONG, 'STRONG'),
        (ProtectionLevel.MAXIMUM, 'MAXIMUM'),
    ]

    for photo_name, photo_path in photos_to_protect:
        for level, level_name in levels:
            savior = PhotoSavior(protection_level=level)
            protected, report = savior.protect(photo_path)
            
            out_name = f'02_PROTECTED_{level_name}.png'
            if photo_name != 'synthetic':
                out_name = f'02_PROTECTED_{level_name}_{photo_name}.png'
            
            out_path = os.path.join(output_dir, out_name)
            savior.save_image(protected, out_path)
            
            psnr = report['quality']['psnr_db']
            ssim = report['quality']['ssim']
            print(f"  Saved: {out_name}  (PSNR={psnr:.1f} dB, SSIM={ssim:.4f})")

    # ─────────────────────────────────────────────
    # Step 3: Print testing instructions
    # ─────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  TESTING INSTRUCTIONS")
    print("=" * 70)

    print("""
╔══════════════════════════════════════════════════════════════════════╗
║  HOW TO TEST AGAINST REAL AI SERVICES                              ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                    ║
║  The test files are in: outputs/manual_test/                       ║
║                                                                    ║
║  For each test, upload BOTH the original AND protected version     ║
║  and compare how the AI handles each one.                          ║
║                                                                    ║
╚══════════════════════════════════════════════════════════════════════╝

───────────────────────────────────────────────────
TEST A: ChatGPT / DALL-E (Image Edit)
───────────────────────────────────────────────────

1. Go to: chat.openai.com
2. Upload: 01_ORIGINAL.png
3. Ask: "Edit this image to make the person look older" 
   or: "Change the background to a beach"
   or: "Apply a Van Gogh style to this image"
4. Save the result as: RESULT_original_chatgpt.png

5. Now upload: 02_PROTECTED_STRONG.png
6. Give the EXACT same prompt
7. Save as: RESULT_protected_chatgpt.png

8. COMPARE:
   - Does the protected version produce worse/stranger edits?
   - Does the AI refuse or struggle with the protected version?
   - Are there more artifacts in the protected result?


───────────────────────────────────────────────────
TEST B: Grok (xAI Image Analysis/Edit) 
───────────────────────────────────────────────────

1. Go to: grok.x.ai or use via X app
2. Upload: 01_ORIGINAL.png
3. Ask: "Modify this photo to change the lighting"
   or: "Recreate this image in anime style"
4. Save the result

5. Upload: 02_PROTECTED_STRONG.png  
6. Same prompt
7. Save and COMPARE


───────────────────────────────────────────────────
TEST C: Midjourney (Image Variation / Describe)
───────────────────────────────────────────────────

1. In Midjourney Discord, use /describe with 01_ORIGINAL.png
2. Note the descriptions generated
3. Use /describe with 02_PROTECTED_STRONG.png
4. COMPARE: Are descriptions different/worse?

5. Also try /blend with the protected image
6. COMPARE quality vs blending with original


───────────────────────────────────────────────────
TEST D: Stable Diffusion (img2img)
───────────────────────────────────────────────────

If you have local Stable Diffusion (Automatic1111 / ComfyUI):

1. Load 01_ORIGINAL.png in img2img
2. Settings: denoising=0.5, steps=30, sampler=Euler
3. Prompt: "portrait of a person, oil painting style"
4. Generate and save

5. Load 02_PROTECTED_STRONG.png with SAME settings
6. Generate and save
7. COMPARE: Protected should produce more distorted output


───────────────────────────────────────────────────
TEST E: Online AI Photo Editors
───────────────────────────────────────────────────

Test with free online AI editors:
- Canva AI (canva.com) — Magic Edit feature
- Fotor AI (fotor.com) — AI face editing
- Pixlr AI (pixlr.com) — AI background removal
- Remove.bg — Background removal

For each:
1. Upload original, apply AI edit, save result
2. Upload protected, apply same edit, save result
3. COMPARE quality and accuracy


───────────────────────────────────────────────────
WHAT TO LOOK FOR (Success Criteria)
───────────────────────────────────────────────────

PROTECTION IS WORKING if you observe ANY of these:

  ✓ AI produces more artifacts on the protected image
  ✓ Style transfer looks worse/different on protected version
  ✓ AI mis-identifies objects or features in protected image
  ✓ Inpainting/editing produces visible glitches
  ✓ Background removal has more errors on protected version
  ✓ Generated variations are less faithful to protected original
  ✓ AI refuses to process the protected image
  ✓ Color accuracy is worse on protected edits

PROTECTION IS NOT WORKING if:

  ✗ AI produces identical quality edits on both versions
  ✗ No visible difference in output quality
  ✗ Style transfer works equally well on both

NOTE: The level of disruption depends on:
  - Protection level (MAXIMUM > STRONG > MODERATE > LIGHT)
  - The specific AI model's architecture
  - The type of editing operation
  - The AI service's preprocessing pipeline

For best results, test with STRONG or MAXIMUM protection level.
""")

    # ─────────────────────────────────────────────
    # Create a comparison card for easy sharing
    # ─────────────────────────────────────────────
    print("[3/3] Creating comparison card...")

    savior = PhotoSavior(protection_level=ProtectionLevel.STRONG)
    original = savior.load_image(orig_path)
    protected, _ = savior.protect(orig_path)

    orig_pil = Image.fromarray((original * 255).astype(np.uint8))
    prot_pil = Image.fromarray((np.clip(protected, 0, 1) * 255).astype(np.uint8))

    # Resize for card
    card_img_size = 384
    orig_pil = orig_pil.resize((card_img_size, card_img_size))
    prot_pil = prot_pil.resize((card_img_size, card_img_size))

    # Difference visualization
    diff = np.abs(protected - original)
    diff_amplified = np.clip(diff * 30, 0, 1)
    diff_pil = Image.fromarray((diff_amplified * 255).astype(np.uint8)).resize(
        (card_img_size, card_img_size))

    # Create comparison card
    card_width = card_img_size * 3 + 40
    card_height = card_img_size + 80
    card = Image.new('RGB', (card_width, card_height), (30, 30, 30))
    draw = ImageDraw.Draw(card)

    # Paste images
    card.paste(orig_pil, (10, 60))
    card.paste(prot_pil, (card_img_size + 20, 60))
    card.paste(diff_pil, (card_img_size * 2 + 30, 60))

    # Labels
    try:
        font = ImageFont.truetype("arial.ttf", 18)
        font_title = ImageFont.truetype("arial.ttf", 24)
    except Exception:
        font = ImageFont.load_default()
        font_title = font

    draw.text((card_width//2 - 200, 5), "PhotoSavior Protection Comparison",
              fill=(255, 255, 255), font=font_title)
    draw.text((10 + card_img_size//2 - 30, 38), "ORIGINAL",
              fill=(100, 255, 100), font=font)
    draw.text((card_img_size + 20 + card_img_size//2 - 50, 38), "PROTECTED",
              fill=(100, 200, 255), font=font)
    draw.text((card_img_size*2 + 30 + card_img_size//2 - 60, 38), "DIFFERENCE (30x)",
              fill=(255, 200, 100), font=font)

    card_path = os.path.join(output_dir, '03_COMPARISON_CARD.png')
    card.save(card_path)
    print(f"  Saved: {card_path}")

    print(f"\n  All test files saved to: {output_dir}/")
    print(f"  Files generated:")
    for f in sorted(os.listdir(output_dir)):
        if f.endswith('.png'):
            size = os.path.getsize(os.path.join(output_dir, f))
            print(f"    {f}  ({size//1024} KB)")

    print("\n" + "=" * 70)
    print("  READY FOR TESTING!")
    print("  Upload these images to ChatGPT, Grok, etc. and compare results")
    print("=" * 70)


if __name__ == '__main__':
    main()
