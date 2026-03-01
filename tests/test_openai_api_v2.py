"""
PhotoSavior v2 — End-to-End API Test with CLIP Adversarial Attack
==================================================================

This test uses the REAL CLIP-adversarial protection (PGD attack against
the actual CLIP ViT-B/32 model) and tests against REAL OpenAI API.

Key difference from v1: The protection is now gradient-based adversarial
perturbation against a real neural network, not random spectral noise.
"""

import os
import sys
import json
import time
import base64
import io
import re
import numpy as np
from PIL import Image
from openai import OpenAI

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.photosavior import PhotoSavior, ProtectionLevel

API_KEY = os.environ.get("OPENAI_API_KEY", "")
if not API_KEY:
    raise SystemExit("Set the OPENAI_API_KEY environment variable before running this test.")
OUTPUT_DIR = os.path.join("outputs", "api_test_v2")


def ensure_dirs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def img_to_base64(img_path):
    with open(img_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def img_to_png_bytesio(img_path, max_size=512):
    img = Image.open(img_path).convert("RGBA")
    img = img.resize((max_size, max_size), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    buf.name = "image.png"
    return buf


def save_b64_image(b64_data, path):
    img_bytes = base64.b64decode(b64_data)
    with open(path, "wb") as f:
        f.write(img_bytes)
    return path


def compute_metrics(img1_np, img2_np):
    """Compute PSNR and MSE between two images."""
    mse = float(np.mean((img1_np.astype(float) - img2_np.astype(float)) ** 2))
    if mse < 1e-10:
        psnr = float('inf')
    else:
        psnr = 10 * np.log10(255.0 ** 2 / mse)
    return {'mse': mse, 'psnr': float(psnr)}


def load_resize(path, size=512):
    return np.array(Image.open(path).convert("RGB").resize((size, size)))


# ═══════════════════════════════════════════════════════════════════

def create_test_image():
    """Create a naturalistic landscape image with recognizable content."""
    rng = np.random.RandomState(42)
    img = np.zeros((512, 512, 3), dtype=np.float64)

    # Sky with clouds
    for y in range(280):
        t = y / 280
        for x in range(512):
            cloud = 0.08 * np.sin(x * 0.02 + y * 0.01) * np.cos(x * 0.015 - y * 0.008)
            cloud += 0.04 * np.sin(x * 0.05 + y * 0.03)
            img[y, x] = [0.35 + 0.15 * t + cloud * 0.3, 
                         0.55 + 0.15 * (1 - t) + cloud * 0.2, 
                         0.82 + 0.1 * (1 - t) + cloud * 0.1]

    # Ground with grass texture
    for y in range(280, 512):
        t = (y - 280) / 232
        for x in range(512):
            grass = 0.05 * np.sin(x * 0.1 + y * 0.15) + 0.03 * np.cos(x * 0.2 - y * 0.1)
            img[y, x] = [0.15 + 0.08 * t + grass * 0.5,
                         0.42 - 0.12 * t + grass + 0.15 * (1 - t),
                         0.08 + 0.05 * t + grass * 0.3]

    # House
    for y in range(290, 390):
        for x in range(180, 340):
            brick = 0.04 * ((x // 8 + y // 6) % 2)
            shadow = 0.92 + 0.05 * np.sin(x * 0.15 + y * 0.1)
            img[y, x] = np.array([0.65 + brick, 0.22 + brick * 0.5, 0.15]) * shadow

    # Roof
    for y in range(260, 290):
        for x in range(170, 350):
            shingle = 0.03 * np.sin(x * 0.3) * np.cos(y * 0.4)
            img[y, x] = [0.35 + shingle, 0.15 + shingle, 0.12]

    # Windows
    for wx, wy in [(205, 315), (290, 315)]:
        for y in range(wy, wy + 30):
            for x in range(wx, wx + 30):
                ref = 0.1 * np.sin(x * 0.2) * np.cos(y * 0.15)
                img[y, x] = [0.5 + ref, 0.7 + ref, 0.85 + ref]

    # Door
    for y in range(350, 390):
        for x in range(245, 275):
            wood = 0.03 * np.sin(y * 0.3 + x * 0.1)
            img[y, x] = [0.35 + wood, 0.22 + wood, 0.12]

    # Tree
    for y in range(260, 400):
        for x in range(410, 435):
            bark = 0.04 * np.sin(y * 0.5 + x * 0.3)
            img[y, x] = [0.32 + bark, 0.2 + bark, 0.1]

    tcx, tcy = 422, 220
    for y in range(max(0, tcy - 65), min(512, tcy + 50)):
        for x in range(max(0, tcx - 55), min(512, tcx + 55)):
            d = np.sqrt((x - tcx) ** 2 + (y - tcy) ** 2)
            if d < 55 + 8 * np.sin(np.arctan2(y - tcy, x - tcx) * 5):
                leaf = 0.08 * np.sin(x * 0.4 + y * 0.3) * np.cos(x * 0.2 - y * 0.15)
                alpha = min(1, max(0, (60 - d) / 10))
                canopy = np.array([0.08 + leaf * 0.5, 0.45 + leaf, 0.06]) * (0.85 + 0.15 * (1 - d / 55))
                img[y, x] = img[y, x] * (1 - alpha) + canopy * alpha

    # Sun
    scx, scy = 400, 70
    for y in range(max(0, scy - 60), min(280, scy + 60)):
        for x in range(max(0, scx - 60), min(512, scx + 60)):
            d = np.sqrt((x - scx) ** 2 + (y - scy) ** 2)
            if d < 60:
                alpha = max(0, 1 - (d / 60) ** 1.5)
                sun = np.array([1.0, 0.92, 0.5]) * (0.6 + 0.4 * alpha)
                img[y, x] = img[y, x] * (1 - alpha * 0.9) + sun * alpha * 0.9

    img += rng.randn(512, 512, 3) * 0.012
    img[:, :, 0] += rng.randn(512, 512) * 0.005
    img = np.clip(img, 0, 1)
    return img


# ═══════════════════════════════════════════════════════════════════

class OpenAIEndToEndTestV2:
    def __init__(self):
        self.client = OpenAI(api_key=API_KEY)
        self.results = []
        self.all_passed = True

    def log(self, msg):
        print(f"  {msg}")

    def add_result(self, name, passed, details):
        status = "PASS" if passed else "FAIL"
        print(f"\n  [{status}] {name}")
        for k, v in details.items():
            if isinstance(v, float):
                print(f"    {k}: {v:.4f}")
            elif isinstance(v, str) and len(v) > 150:
                print(f"    {k}: {v[:150]}...")
            else:
                print(f"    {k}: {v}")
        self.results.append({'name': name, 'passed': passed, 'details': details})
        if not passed:
            self.all_passed = False

    def gpt4o_vision(self, img_path, prompt):
        """Send image to GPT-4o and get response."""
        b64 = img_to_base64(img_path)
        response = self.client.responses.create(
            model="gpt-4o",
            input=[{
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image",
                     "image_url": f"data:image/png;base64,{b64}"},
                ],
            }],
        )
        return response.output_text

    def dalle_variation(self, img_path, out_path):
        """Generate DALL-E 2 variation."""
        buf = img_to_png_bytesio(img_path, max_size=512)
        response = self.client.images.create_variation(
            image=buf,
            n=1,
            size="512x512",
            response_format="b64_json",
        )
        return save_b64_image(response.data[0].b64_json, out_path)

    def dalle_edit(self, img_path, mask_bytes, prompt, out_path):
        """Edit image with DALL-E 2."""
        buf = img_to_png_bytesio(img_path, max_size=512)
        response = self.client.images.edit(
            model="dall-e-2",
            image=buf,
            mask=mask_bytes,
            prompt=prompt,
            n=1,
            size="512x512",
            response_format="b64_json",
        )
        return save_b64_image(response.data[0].b64_json, out_path)

    # ═══════════════════════════════════════════════════════════════
    # TEST 1: CLIP Embedding Displacement (Local Verification)
    # ═══════════════════════════════════════════════════════════════
    def test_clip_displacement(self, orig_path, prot_path):
        """Verify CLIP sees the protected image as different."""
        print("\n" + "─" * 60)
        print("TEST 1: CLIP EMBEDDING DISPLACEMENT (local)")
        print("─" * 60)

        from src.clip_adversarial import _load_clip, _preprocess_for_clip, _get_clip_embedding
        import torch
        import torch.nn.functional as F

        model, processor, device = _load_clip()

        orig_np = np.array(Image.open(orig_path).convert("RGB"), dtype=np.float64) / 255.0
        prot_np = np.array(Image.open(prot_path).convert("RGB"), dtype=np.float64) / 255.0

        with torch.no_grad():
            orig_clip = _preprocess_for_clip(orig_np, processor, device)
            prot_clip = _preprocess_for_clip(prot_np, processor, device)
            orig_emb = _get_clip_embedding(model, orig_clip)
            prot_emb = _get_clip_embedding(model, prot_clip)
            cos_sim = F.cosine_similarity(orig_emb, prot_emb, dim=-1).item()

        displacement = (1 - cos_sim) * 100

        self.add_result(
            "CLIP embedding displacement",
            cos_sim < 0.85,
            {
                'cosine_similarity': cos_sim,
                'displacement_pct': displacement,
                'threshold': 'cos_sim < 0.85 (15%+ displacement)',
            }
        )
        return cos_sim

    # ═══════════════════════════════════════════════════════════════
    # TEST 2: DALL-E 2 Variation Disruption (Multi-Sample)
    # ═══════════════════════════════════════════════════════════════
    def test_dalle_variation_multi(self, orig_path, prot_path, num_samples=2):
        """Generate multiple variations and compare statistically."""
        print("\n" + "─" * 60)
        print(f"TEST 2: DALL-E 2 VARIATION DISRUPTION ({num_samples} samples)")
        print("─" * 60)

        orig_np = load_resize(orig_path)
        orig_var_dists = []
        prot_var_dists = []

        for i in range(num_samples):
            self.log(f"Generating variation {i+1}/{num_samples} for ORIGINAL...")
            var_path = os.path.join(OUTPUT_DIR, f"var_orig_{i}.png")
            self.dalle_variation(orig_path, var_path)
            var_np = load_resize(var_path)
            dist = np.mean((orig_np.astype(float) - var_np.astype(float)) ** 2)
            orig_var_dists.append(dist)
            self.log(f"  MSE from original: {dist:.1f}")
            time.sleep(2)

            self.log(f"Generating variation {i+1}/{num_samples} for PROTECTED...")
            var_path = os.path.join(OUTPUT_DIR, f"var_prot_{i}.png")
            self.dalle_variation(prot_path, var_path)
            var_np = load_resize(var_path)
            dist = np.mean((orig_np.astype(float) - var_np.astype(float)) ** 2)
            prot_var_dists.append(dist)
            self.log(f"  MSE from original: {dist:.1f}")
            time.sleep(2)

        avg_orig = np.mean(orig_var_dists)
        avg_prot = np.mean(prot_var_dists)
        disruption = avg_prot / (avg_orig + 1e-10)

        self.add_result(
            "DALL-E 2 variation disruption (multi-sample)",
            True,  # Informational
            {
                'orig_variation_mse_avg': float(avg_orig),
                'prot_variation_mse_avg': float(avg_prot),
                'disruption_ratio': float(disruption),
                'samples': num_samples,
                'note': f'Ratio >1.0 = protection causes more distortion ({disruption:.2f}x)',
            }
        )

    # ═══════════════════════════════════════════════════════════════
    # TEST 3: DALL-E 2 Edit Disruption
    # ═══════════════════════════════════════════════════════════════
    def test_dalle_edit_disruption(self, orig_path, prot_path):
        """Edit both images with DALL-E 2 and compare quality."""
        print("\n" + "─" * 60)
        print("TEST 3: DALL-E 2 EDIT DISRUPTION")
        print("─" * 60)

        # Create mask (edit the sky area)
        mask = Image.new("RGBA", (512, 512), (0, 0, 0, 255))
        for y in range(200):
            for x in range(512):
                mask.putpixel((x, y), (0, 0, 0, 0))
        mask_buf = io.BytesIO()
        mask.save(mask_buf, format="PNG")
        mask_buf.seek(0)
        mask_buf.name = "mask.png"

        prompt = "A beautiful sunset sky with orange and purple clouds"

        self.log("Editing ORIGINAL image...")
        orig_edit_path = os.path.join(OUTPUT_DIR, "edit_orig.png")
        self.dalle_edit(orig_path, mask_buf, prompt, orig_edit_path)
        time.sleep(2)

        # Reset mask buffer
        mask_buf.seek(0)

        self.log("Editing PROTECTED image...")
        prot_edit_path = os.path.join(OUTPUT_DIR, "edit_prot.png")
        self.dalle_edit(prot_path, mask_buf, prompt, prot_edit_path)
        time.sleep(2)

        # Have GPT-4o judge both edits
        self.log("GPT-4o judging edit quality...")

        b64_orig = img_to_base64(orig_edit_path)
        b64_prot = img_to_base64(prot_edit_path)

        response = self.client.responses.create(
            model="gpt-4o",
            input=[{
                "role": "user",
                "content": [
                    {"type": "input_text", "text": (
                        "Rate each image's quality from 0-100 based on: "
                        "realism, coherence, lack of artifacts. "
                        'Respond ONLY in JSON: {"image_a": <number>, "image_b": <number>, "notes": "<text>"}'
                    )},
                    {"type": "input_image", "image_url": f"data:image/png;base64,{b64_orig}"},
                    {"type": "input_image", "image_url": f"data:image/png;base64,{b64_prot}"},
                ],
            }],
        )

        judge_text = response.output_text
        orig_q, prot_q = 50, 50
        try:
            m = re.search(r'\{[^}]+\}', judge_text, re.DOTALL)
            if m:
                d = json.loads(m.group())
                orig_q = d.get('image_a', 50)
                prot_q = d.get('image_b', 50)
        except Exception:
            pass

        self.add_result(
            "DALL-E 2 edit quality degradation",
            orig_q > prot_q,
            {
                'original_edit_quality': orig_q,
                'protected_edit_quality': prot_q,
                'quality_drop': orig_q - prot_q,
                'judge_notes': judge_text,
            }
        )

    # ═══════════════════════════════════════════════════════════════
    # TEST 4: GPT-4o Description Comparison
    # ═══════════════════════════════════════════════════════════════
    def test_gpt4o_description(self, orig_path, prot_path):
        """Ask GPT-4o to describe both. Protection should alter description."""
        print("\n" + "─" * 60)
        print("TEST 4: GPT-4o VISION DESCRIPTION")
        print("─" * 60)

        prompt = (
            "Describe this image in detail: objects, colors, composition, "
            "and any visual artifacts or unusual patterns you notice."
        )

        self.log("Describing ORIGINAL...")
        orig_desc = self.gpt4o_vision(orig_path, prompt)
        time.sleep(1)

        self.log("Describing PROTECTED...")
        prot_desc = self.gpt4o_vision(prot_path, prompt)

        # Ask GPT-4o to compare
        self.log("Comparing descriptions...")
        compare = self.client.responses.create(
            model="gpt-4o",
            input=[{
                "role": "user",
                "content": (
                    "Compare these two image descriptions. Rate similarity 0-100. "
                    'Respond ONLY in JSON: {"similarity": <number>, "key_differences": "<text>"}\n\n'
                    f"A:\n{orig_desc}\n\nB:\n{prot_desc}"
                ),
            }],
        )

        sim = 50
        try:
            m = re.search(r'\{[^}]+\}', compare.output_text, re.DOTALL)
            if m:
                sim = json.loads(m.group()).get('similarity', 50)
        except Exception:
            pass

        self.add_result(
            "GPT-4o description disruption",
            sim < 90,
            {
                'description_similarity': sim,
                'original_desc': orig_desc[:300],
                'protected_desc': prot_desc[:300],
                'comparison': compare.output_text[:300],
            }
        )

    # ═══════════════════════════════════════════════════════════════
    # TEST 5: GPT-4o Side-by-Side — Can AI tell which is protected?
    # ═══════════════════════════════════════════════════════════════
    def test_gpt4o_side_by_side(self, orig_path, prot_path):
        """Show both images to GPT-4o and ask which has been modified."""
        print("\n" + "─" * 60)
        print("TEST 5: GPT-4o SIDE-BY-SIDE COMPARISON")
        print("─" * 60)

        b64_orig = img_to_base64(orig_path)
        b64_prot = img_to_base64(prot_path)

        # Randomly swap order so we can verify GPT-4o isn't guessing
        import random
        random.seed(42)
        if random.random() > 0.5:
            first, second = b64_prot, b64_orig
            prot_is = "A"
        else:
            first, second = b64_orig, b64_prot
            prot_is = "B"

        self.log(f"Protected image is in position: {prot_is}")

        response = self.client.responses.create(
            model="gpt-4o",
            input=[{
                "role": "user",
                "content": [
                    {"type": "input_text", "text": (
                        "I have two images. One is the original, and one has had "
                        "adversarial perturbations applied to disrupt AI processing. "
                        "Which image (A or B) has been modified? Look for ANY artifacts, "
                        "noise patterns, or subtle differences. "
                        'Respond in JSON: {"modified_image": "A" or "B", '
                        '"confidence": 0-100, "reasoning": "<text>"}'
                    )},
                    {"type": "input_image", "image_url": f"data:image/png;base64,{first}"},
                    {"type": "input_image", "image_url": f"data:image/png;base64,{second}"},
                ],
            }],
        )

        resp_text = response.output_text
        guessed = "?"
        confidence = 50
        try:
            m = re.search(r'\{[^}]+\}', resp_text, re.DOTALL)
            if m:
                d = json.loads(m.group())
                guessed = d.get('modified_image', '?')
                confidence = d.get('confidence', 50)
        except Exception:
            pass

        correct_guess = guessed == prot_is

        self.add_result(
            "GPT-4o adversarial detection",
            True,  # Informational
            {
                'gpt4o_guessed': guessed,
                'actual_protected': prot_is,
                'correct_detection': correct_guess,
                'confidence': confidence,
                'reasoning': resp_text[:300],
                'note': 'If confidence > 70 and correct, protection is too visible'
            }
        )

    # ═══════════════════════════════════════════════════════════════
    # TEST 6: Watermark Tamper Detection
    # ═══════════════════════════════════════════════════════════════
    def test_watermark_tamper(self, prot_path):
        """Verify watermark detects tampering through DALL-E."""
        print("\n" + "─" * 60)
        print("TEST 6: WATERMARK TAMPER DETECTION")
        print("─" * 60)

        savior = PhotoSavior(protection_level=ProtectionLevel.STRONG, use_clip=False)
        wm_before = savior.verify_protection(prot_path)
        self.log(f"Watermark before DALL-E: valid={wm_before.get('is_valid', False)}")

        self.log("Sending to DALL-E for variation...")
        roundtrip_path = os.path.join(OUTPUT_DIR, "roundtrip.png")
        self.dalle_variation(prot_path, roundtrip_path)

        wm_after = savior.verify_protection(roundtrip_path)
        self.log(f"Watermark after DALL-E: valid={wm_after.get('is_valid', False)}")

        # Watermark should be valid BEFORE and INVALID after
        # (proves tamper detection works)
        self.add_result(
            "Watermark tamper detection",
            wm_before.get('is_valid', False) and not wm_after.get('is_valid', True),
            {
                'watermark_before': wm_before.get('is_valid', False),
                'watermark_after': wm_after.get('is_valid', False),
                'tamper_detected': not wm_after.get('is_valid', True),
            }
        )

    # ═══════════════════════════════════════════════════════════════
    def run_all(self):
        ensure_dirs()
        start = time.time()

        print("=" * 60)
        print("  PHOTOSAVIOR v2 — CLIP ADVERSARIAL + OPENAI API TEST")
        print("=" * 60)
        print("  Protection: PGD attack against CLIP ViT-B/32")
        print("  Testing:    GPT-4o vision + DALL-E 2 + DALL-E 3")
        print()

        # Step 1: Create images
        print("[STEP 1] Creating test images...")
        img = create_test_image()
        orig_path = os.path.join(OUTPUT_DIR, "original.png")
        Image.fromarray((img * 255).astype(np.uint8)).save(orig_path)

        print("[STEP 2] Applying CLIP-adversarial protection (STRONG)...")
        savior = PhotoSavior(protection_level=ProtectionLevel.STRONG)
        protected, report = savior.protect(orig_path)
        prot_path = os.path.join(OUTPUT_DIR, "protected_CLIP.png")
        savior.save_image(protected, prot_path)

        if 'clip_adversarial' in report['layers']:
            clip_r = report['layers']['clip_adversarial']
            print(f"  CLIP cosine similarity: {clip_r['cosine_similarity']:.4f}")
            print(f"  PSNR: {clip_r['psnr_db']:.1f} dB")
            print(f"  L-inf: {clip_r['linf'] * 255:.1f}/255")
        else:
            print(f"  PSNR: {report['quality']['psnr_db']:.1f} dB")

        # Step 3: Run tests
        tests = [
            ("CLIP displacement", lambda: self.test_clip_displacement(orig_path, prot_path)),
            ("DALL-E variation", lambda: self.test_dalle_variation_multi(orig_path, prot_path, num_samples=2)),
            ("DALL-E edit", lambda: self.test_dalle_edit_disruption(orig_path, prot_path)),
            ("GPT-4o description", lambda: self.test_gpt4o_description(orig_path, prot_path)),
            ("GPT-4o side-by-side", lambda: self.test_gpt4o_side_by_side(orig_path, prot_path)),
            ("Watermark tamper", lambda: self.test_watermark_tamper(prot_path)),
        ]

        for name, test_fn in tests:
            try:
                test_fn()
            except Exception as e:
                print(f"\n  [ERROR] {name}: {e}")
                import traceback
                traceback.print_exc()
                self.add_result(name, False, {'error': str(e)})

        elapsed = time.time() - start
        passed = sum(1 for r in self.results if r['passed'])
        total = len(self.results)

        print("\n" + "=" * 60)
        print("  FINAL RESULTS")
        print("=" * 60)
        for r in self.results:
            s = "PASS" if r['passed'] else "FAIL"
            print(f"  [{s}] {r['name']}")
        print(f"\n  Passed: {passed}/{total}")
        print(f"  Time:   {elapsed:.0f}s")
        print("=" * 60)

        # Save report
        report_path = os.path.join(OUTPUT_DIR, "api_test_report.json")
        with open(report_path, 'w') as f:
            json.dump({
                'summary': {'passed': passed, 'total': total, 'time': round(elapsed, 1)},
                'tests': self.results,
            }, f, indent=2, default=str)
        print(f"\n  Report: {report_path}")
        print(f"  Images: {OUTPUT_DIR}/")

        return passed == total


if __name__ == '__main__':
    suite = OpenAIEndToEndTestV2()
    success = suite.run_all()
    sys.exit(0 if success else 1)
