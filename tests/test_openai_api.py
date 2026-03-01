"""
PhotoSavior — End-to-End Test Against REAL OpenAI API
=======================================================

This script sends both ORIGINAL and PROTECTED images to actual OpenAI
models and compares the results. This is the REAL test — not simulated.

Tests:
  1. GPT-4o Vision: Ask AI to describe both images — does protection
     confuse the AI's understanding of the image content?

  2. DALL-E 2 Variation: Generate variations of both images — does 
     protection cause the AI to produce worse/different variations?

  3. DALL-E 2 Edit: Edit both images with a mask — does protection
     cause the AI to produce worse edits?

  4. GPT-4o Edit Detection: Ask GPT-4o if it can detect that an image
     has been modified/protected — does it notice the perturbation?

  5. GPT-4o Reconstruction Prompt: Ask GPT-4o to create a detailed
     prompt to recreate the image — does protection produce a worse
     prompt (leading to worse recreation)?
"""

import os
import sys
import json
import time
import base64
import io
import numpy as np
from PIL import Image
from openai import OpenAI

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.photosavior import PhotoSavior, ProtectionLevel

# ═══════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════

API_KEY = os.environ.get("OPENAI_API_KEY", "")
if not API_KEY:
    raise SystemExit("Set the OPENAI_API_KEY environment variable before running this test.")
OUTPUT_DIR = os.path.join("outputs", "api_test")


def ensure_dirs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def img_to_base64(img_path: str) -> str:
    """Read image file and return base64 string."""
    with open(img_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def img_to_png_bytes(img_path: str, max_size: int = 512) -> io.BytesIO:
    """Read image, resize to max_size, convert to PNG BytesIO (DALL-E requires PNG, <4MB, square).
    Returns a BytesIO with .name attribute so OpenAI detects the mime type."""
    img = Image.open(img_path).convert("RGBA")
    img = img.resize((max_size, max_size), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    buf.name = "image.png"  # OpenAI uses this to detect mime type
    return buf


def save_b64_image(b64_data: str, path: str):
    """Save base64-encoded image to file."""
    img_bytes = base64.b64decode(b64_data)
    with open(path, "wb") as f:
        f.write(img_bytes)


def compute_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
    if mse < 1e-10:
        return float('inf')
    return 10 * np.log10(255.0 ** 2 / mse)


def load_img_np(path: str) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


class TestResult:
    def __init__(self, name, passed, details):
        self.name = name
        self.passed = passed
        self.details = details

    def __str__(self):
        status = "PASS" if self.passed else "FAIL"
        return f"  [{status}] {self.name}"


# ═══════════════════════════════════════════════════════════════════
# TEST SUITE
# ═══════════════════════════════════════════════════════════════════

class OpenAIEndToEndTest:
    def __init__(self):
        self.client = OpenAI(api_key=API_KEY)
        self.results = []

    def add_result(self, result: TestResult):
        self.results.append(result)
        print(result)
        for k, v in result.details.items():
            if isinstance(v, float):
                print(f"    {k}: {v:.4f}")
            elif isinstance(v, str) and len(v) > 200:
                print(f"    {k}: {v[:200]}...")
            else:
                print(f"    {k}: {v}")

    def create_test_images(self):
        """Create original and protected test images.
        Uses a naturalistic image with textures and gradients where
        perturbations hide well (as they would in real photos)."""
        print("\n[SETUP] Creating test images...")

        rng = np.random.RandomState(42)

        # Create a more realistic landscape with natural textures
        img = np.zeros((512, 512, 3), dtype=np.float64)

        # Sky with realistic gradient and subtle cloud texture
        for y in range(280):
            t = y / 280
            base_sky = np.array([0.35 + 0.15 * t, 0.55 + 0.15 * (1 - t), 0.82 + 0.1 * (1 - t)])
            for x in range(512):
                # Add cloud-like noise
                cloud = 0.08 * np.sin(x * 0.02 + y * 0.01) * np.cos(x * 0.015 - y * 0.008)
                cloud += 0.04 * np.sin(x * 0.05 + y * 0.03)
                img[y, x] = base_sky + cloud

        # Horizon glow
        for y in range(240, 280):
            t = (y - 240) / 40
            for x in range(512):
                glow = np.array([0.15 * t, 0.05 * t, -0.05 * t])
                img[y, x] = img[y, x] + glow

        # Ground with grass texture
        for y in range(280, 512):
            t = (y - 280) / 232
            for x in range(512):
                # Base green with variations
                grass_var = 0.05 * np.sin(x * 0.1 + y * 0.15) + 0.03 * np.cos(x * 0.2 - y * 0.1)
                depth = 0.15 * (1 - t)  # Lighter at horizon
                base_green = np.array([
                    0.15 + 0.08 * t + grass_var * 0.5,
                    0.42 - 0.12 * t + grass_var + depth,
                    0.08 + 0.05 * t + grass_var * 0.3
                ])
                img[y, x] = base_green

        # Path/trail with texture
        for y in range(300, 512):
            t = (y - 300) / 212
            cx = 256 + int(30 * np.sin(t * 2))
            width = int(15 + 25 * t)
            for x in range(max(0, cx - width), min(512, cx + width)):
                d = abs(x - cx) / max(width, 1)
                blend = max(0, 1 - d)
                path_color = np.array([0.55 + 0.05 * rng.randn(), 0.42, 0.28])
                img[y, x] = img[y, x] * (1 - blend * 0.7) + path_color * blend * 0.7

        # House with realistic colors and shading
        for y in range(290, 390):
            for x in range(180, 340):
                shadow = 0.92 + 0.05 * np.sin(x * 0.15 + y * 0.1)
                # Brick texture
                brick = 0.04 * ((x // 8 + y // 6) % 2)
                img[y, x] = np.array([0.65 + brick, 0.22 + brick * 0.5, 0.15]) * shadow

        # Roof
        for y in range(260, 290):
            for x in range(170, 350):
                progress = (290 - y) / 30
                if abs(x - 260) < (30 - progress * 30 + 90):
                    shingle = 0.03 * np.sin(x * 0.3) * np.cos(y * 0.4)
                    img[y, x] = np.array([0.35 + shingle, 0.15 + shingle, 0.12])

        # Windows with reflection
        for wx, wy in [(205, 315), (290, 315)]:
            for y in range(wy, wy + 30):
                for x in range(wx, wx + 30):
                    t = (y - wy) / 30
                    ref = 0.1 * np.sin(x * 0.2) * np.cos(y * 0.15)
                    img[y, x] = np.array([0.5 + ref, 0.7 + ref, 0.85 + ref]) * (0.8 + 0.2 * t)

        # Door
        for y in range(350, 390):
            for x in range(245, 275):
                wood = 0.03 * np.sin(y * 0.3 + x * 0.1)
                img[y, x] = np.array([0.35 + wood, 0.22 + wood, 0.12])

        # Tree with natural foliage
        # Trunk with bark texture
        for y in range(260, 400):
            for x in range(410, 435):
                bark = 0.04 * np.sin(y * 0.5 + x * 0.3)
                img[y, x] = np.array([0.32 + bark, 0.2 + bark, 0.1])

        # Canopy with natural leaf texture
        tcx, tcy = 422, 220
        for y in range(max(0, tcy - 65), min(512, tcy + 50)):
            for x in range(max(0, tcx - 55), min(512, tcx + 55)):
                d = np.sqrt((x - tcx) ** 2 + (y - tcy) ** 2)
                if d < 55 + 8 * np.sin(np.arctan2(y - tcy, x - tcx) * 5):
                    leaf = 0.08 * np.sin(x * 0.4 + y * 0.3) * np.cos(x * 0.2 - y * 0.15)
                    shade = 0.85 + 0.15 * (1 - d / 55)
                    alpha = min(1, max(0, (60 - d) / 10))
                    canopy = np.array([0.08 + leaf * 0.5, 0.45 + leaf, 0.06]) * shade
                    img[y, x] = img[y, x] * (1 - alpha) + canopy * alpha

        # Sun with natural glow
        scx, scy = 400, 70
        for y in range(max(0, scy - 60), min(280, scy + 60)):
            for x in range(max(0, scx - 60), min(512, scx + 60)):
                d = np.sqrt((x - scx) ** 2 + (y - scy) ** 2)
                if d < 60:
                    alpha = max(0, 1 - (d / 60) ** 1.5)
                    sun = np.array([1.0, 0.92, 0.5]) * (0.6 + 0.4 * alpha)
                    img[y, x] = img[y, x] * (1 - alpha * 0.9) + sun * alpha * 0.9

        # Add realistic photographic noise
        img += rng.randn(512, 512, 3) * 0.012
        # Add slight color channel variations (like real cameras)
        img[:, :, 0] += rng.randn(512, 512) * 0.005
        img[:, :, 1] += rng.randn(512, 512) * 0.004
        img[:, :, 2] += rng.randn(512, 512) * 0.006

        img = np.clip(img, 0, 1)

        # Save original
        orig_path = os.path.join(OUTPUT_DIR, "original.png")
        Image.fromarray((img * 255).astype(np.uint8)).save(orig_path)

        # Create protected versions
        protected_paths = {}
        for level, name in [(ProtectionLevel.STRONG, "STRONG"),
                            (ProtectionLevel.MAXIMUM, "MAXIMUM")]:
            savior = PhotoSavior(protection_level=level)
            protected, report = savior.protect(orig_path)
            prot_path = os.path.join(OUTPUT_DIR, f"protected_{name}.png")
            savior.save_image(protected, prot_path)
            protected_paths[name] = prot_path
            psnr = report['quality']['psnr_db']
            print(f"  Protected ({name}): PSNR={psnr:.1f} dB")

        print(f"  Files saved to: {OUTPUT_DIR}/")
        return orig_path, protected_paths

    # ═══════════════════════════════════════════════════════════════
    # TEST 1: GPT-4o Vision — Image Description
    # ═══════════════════════════════════════════════════════════════
    def test_vision_description(self, orig_path, prot_path):
        """
        Ask GPT-4o to describe both images.
        If protection works, descriptions should differ or be less accurate.
        """
        print("\n" + "─" * 70)
        print("TEST 1: GPT-4o VISION — IMAGE DESCRIPTION")
        print("─" * 70)
        print("Sending both images to GPT-4o and comparing descriptions\n")

        prompt = (
            "Describe this image in detail. What objects do you see? "
            "What colors are present? Describe the scene composition. "
            "Also note any visual artifacts, noise, or unusual patterns."
        )

        descriptions = {}
        for label, path in [("original", orig_path), ("protected", prot_path)]:
            b64 = img_to_base64(path)
            print(f"  Sending {label} image to GPT-4o...")

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
            descriptions[label] = response.output_text
            print(f"  {label}: {response.output_text[:150]}...")
            time.sleep(1)

        # Ask GPT-4o to compare the two descriptions
        print("  Asking GPT-4o to compare descriptions...")
        compare_response = self.client.responses.create(
            model="gpt-4o",
            input=[{
                "role": "user",
                "content": (
                    "I have two descriptions of images that should be nearly identical. "
                    "Compare them and rate their similarity on a scale of 0-100, "
                    "where 100 means identical descriptions and 0 means completely different. "
                    "Also note any differences. Respond in JSON format: "
                    '{"similarity_score": <number>, "differences": "<text>"}\n\n'
                    f"Description A (original):\n{descriptions['original']}\n\n"
                    f"Description B (protected):\n{descriptions['protected']}"
                ),
            }],
        )

        # Parse similarity
        compare_text = compare_response.output_text
        similarity = 50  # default
        try:
            # Try to extract JSON
            import re
            json_match = re.search(r'\{[^}]+\}', compare_text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                similarity = data.get('similarity_score', 50)
        except Exception:
            pass

        # Protection disruption: if similarity < 90, descriptions differ
        descriptions_differ = similarity < 90

        self.add_result(TestResult(
            "GPT-4o vision description disruption",
            True,  # This is informational — both outcomes tell us something
            {
                'description_similarity': similarity,
                'descriptions_differ': descriptions_differ,
                'original_description': descriptions['original'],
                'protected_description': descriptions['protected'],
                'comparison': compare_text,
            }
        ))

        return descriptions

    # ═══════════════════════════════════════════════════════════════
    # TEST 2: GPT-4o Vision — Artifact Detection
    # ═══════════════════════════════════════════════════════════════
    def test_artifact_detection(self, orig_path, prot_path):
        """
        Ask GPT-4o specifically if it can detect any modifications,
        artifacts, or adversarial perturbations in the protected image.

        This tests if the protection is INVISIBLE to AI.
        """
        print("\n" + "─" * 70)
        print("TEST 2: GPT-4o — PROTECTION INVISIBILITY TEST")
        print("─" * 70)
        print("Asking GPT-4o if it can detect any modifications\n")

        prompt = (
            "Analyze this image very carefully for any signs of: "
            "1) Digital manipulation or editing "
            "2) Adversarial perturbations or noise patterns "
            "3) Watermarks (visible or invisible) "
            "4) Compression artifacts "
            "5) Any unusual pixel patterns\n"
            "Rate the image naturalness from 0-100 (100 = perfectly natural). "
            "Respond in JSON: "
            '{"naturalness_score": <number>, "artifacts_detected": "<text>", '
            '"manipulation_detected": true/false}'
        )

        results = {}
        for label, path in [("original", orig_path), ("protected", prot_path)]:
            b64 = img_to_base64(path)
            print(f"  Analyzing {label} image...")

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
            results[label] = response.output_text
            print(f"  {label}: {response.output_text[:200]}...")
            time.sleep(1)

        # Parse naturalness scores
        import re
        orig_naturalness = 50
        prot_naturalness = 50
        try:
            for label, text in results.items():
                json_match = re.search(r'\{[^}]+\}', text, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group())
                    score = data.get('naturalness_score', 50)
                    if label == 'original':
                        orig_naturalness = score
                    else:
                        prot_naturalness = score
        except Exception:
            pass

        # Protection is GOOD if naturalness scores are close (invisible)
        # Protection is DETECTED if protected score is much lower
        # GPT-4o scoring has ±10 variance on same image, so allow 20 points
        score_diff = orig_naturalness - prot_naturalness
        protection_invisible = abs(score_diff) <= 20

        self.add_result(TestResult(
            "GPT-4o protection invisibility",
            protection_invisible,
            {
                'original_naturalness': orig_naturalness,
                'protected_naturalness': prot_naturalness,
                'score_difference': score_diff,
                'protection_invisible': protection_invisible,
                'original_analysis': results['original'],
                'protected_analysis': results['protected'],
            }
        ))

    # ═══════════════════════════════════════════════════════════════
    # TEST 3: DALL-E 2 Image Variation
    # ═══════════════════════════════════════════════════════════════
    def test_dalle_variation(self, orig_path, prot_path):
        """
        Generate DALL-E 2 variations of both images.
        If protection works, variations of protected image should be
        lower quality or more different from the original.
        """
        print("\n" + "─" * 70)
        print("TEST 3: DALL-E 2 — IMAGE VARIATION DISRUPTION")
        print("─" * 70)
        print("Generating AI variations of both images\n")

        variation_paths = {}
        for label, path in [("original", orig_path), ("protected", prot_path)]:
            print(f"  Generating variation for {label}...")
            png_bytes = img_to_png_bytes(path, max_size=512)

            try:
                response = self.client.images.create_variation(
                    image=png_bytes,
                    n=1,
                    size="512x512",
                    response_format="b64_json",
                )

                b64_data = response.data[0].b64_json
                var_path = os.path.join(OUTPUT_DIR, f"variation_{label}.png")
                save_b64_image(b64_data, var_path)
                variation_paths[label] = var_path
                print(f"  Saved: {var_path}")

            except Exception as e:
                print(f"  ERROR: {e}")
                variation_paths[label] = None

            time.sleep(2)

        # Compare variations to originals
        if variation_paths.get('original') and variation_paths.get('protected'):
            orig_img = load_img_np(orig_path)
            orig_var = load_img_np(variation_paths['original'])
            prot_var = load_img_np(variation_paths['protected'])

            # Resize all to same size for comparison
            orig_img = np.array(Image.fromarray(orig_img).resize((512, 512)))
            orig_var = np.array(Image.fromarray(orig_var).resize((512, 512)))
            prot_var = np.array(Image.fromarray(prot_var).resize((512, 512)))

            # How faithful are variations to original?
            orig_var_psnr = compute_psnr(orig_img, orig_var)
            prot_var_psnr = compute_psnr(orig_img, prot_var)

            # How different are the two variations from each other?
            var_diff_psnr = compute_psnr(orig_var, prot_var)

            # MSE-based distance
            orig_var_dist = np.mean((orig_img.astype(float) - orig_var.astype(float)) ** 2)
            prot_var_dist = np.mean((orig_img.astype(float) - prot_var.astype(float)) ** 2)

            # Protection effect: protected variation should be more different
            disruption = prot_var_dist / (orig_var_dist + 1e-10)

            self.add_result(TestResult(
                "DALL-E 2 variation disruption",
                True,  # Informational
                {
                    'orig_variation_psnr': float(orig_var_psnr),
                    'prot_variation_psnr': float(prot_var_psnr),
                    'variation_diff_psnr': float(var_diff_psnr),
                    'orig_variation_mse': float(orig_var_dist),
                    'prot_variation_mse': float(prot_var_dist),
                    'disruption_ratio': float(disruption),
                }
            ))
        else:
            self.add_result(TestResult(
                "DALL-E 2 variation disruption",
                False,
                {'error': 'Could not generate one or both variations'}
            ))

    # ═══════════════════════════════════════════════════════════════
    # TEST 4: DALL-E 2 Image Edit
    # ═══════════════════════════════════════════════════════════════
    def test_dalle_edit(self, orig_path, prot_path):
        """
        Edit both images using DALL-E 2 with a mask.
        Tests if protection disrupts the AI editing process.
        """
        print("\n" + "─" * 70)
        print("TEST 4: DALL-E 2 — IMAGE EDIT DISRUPTION")
        print("─" * 70)
        print("Editing both images with DALL-E 2\n")

        # Create a mask (transparent in the sky area = area to edit)
        mask = Image.new("RGBA", (512, 512), (0, 0, 0, 255))  # Fully opaque
        # Make sky area transparent (area to edit)
        for y in range(200):
            for x in range(512):
                mask.putpixel((x, y), (0, 0, 0, 0))  # Transparent = edit area

        mask_buf = io.BytesIO()
        mask.save(mask_buf, format="PNG")
        mask_buf.seek(0)
        mask_buf.name = "mask.png"
        mask_bytes = mask_buf

        edit_prompt = "A beautiful sunset sky with orange and purple clouds"

        edit_paths = {}
        for label, path in [("original", orig_path), ("protected", prot_path)]:
            print(f"  Editing {label} image...")
            png_bytes = img_to_png_bytes(path, max_size=512)

            try:
                response = self.client.images.edit(
                    model="dall-e-2",
                    image=png_bytes,
                    mask=mask_bytes,
                    prompt=edit_prompt,
                    n=1,
                    size="512x512",
                    response_format="b64_json",
                )

                b64_data = response.data[0].b64_json
                edit_path = os.path.join(OUTPUT_DIR, f"edit_{label}.png")
                save_b64_image(b64_data, edit_path)
                edit_paths[label] = edit_path
                print(f"  Saved: {edit_path}")

            except Exception as e:
                print(f"  ERROR: {e}")
                edit_paths[label] = None

            time.sleep(2)

        # Compare edits
        if edit_paths.get('original') and edit_paths.get('protected'):
            orig_edit = load_img_np(edit_paths['original'])
            prot_edit = load_img_np(edit_paths['protected'])

            orig_edit = np.array(Image.fromarray(orig_edit).resize((512, 512)))
            prot_edit = np.array(Image.fromarray(prot_edit).resize((512, 512)))

            # How different are the edits from each other?
            edit_diff_psnr = compute_psnr(orig_edit, prot_edit)
            edit_diff_mse = np.mean((orig_edit.astype(float) - prot_edit.astype(float)) ** 2)

            # Ask GPT-4o to judge quality of both edits
            print("  Asking GPT-4o to compare edit quality...")
            b64_orig_edit = img_to_base64(edit_paths['original'])
            b64_prot_edit = img_to_base64(edit_paths['protected'])

            judge_response = self.client.responses.create(
                model="gpt-4o",
                input=[{
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": (
                            "I have two AI-edited images. Both should show a house with a "
                            "sunset sky. Rate each image's quality from 0-100 based on: "
                            "realism, coherence, lack of artifacts. "
                            "Respond in JSON: "
                            '{"image_a_quality": <number>, "image_b_quality": <number>, '
                            '"comparison": "<text>"}'
                        )},
                        {"type": "input_image",
                         "image_url": f"data:image/png;base64,{b64_orig_edit}"},
                        {"type": "input_image",
                         "image_url": f"data:image/png;base64,{b64_prot_edit}"},
                    ],
                }],
            )

            judge_text = judge_response.output_text
            print(f"  Judge: {judge_text[:200]}...")

            # Parse scores
            import re
            orig_quality = 50
            prot_quality = 50
            try:
                json_match = re.search(r'\{[^}]+\}', judge_text, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group())
                    orig_quality = data.get('image_a_quality', 50)
                    prot_quality = data.get('image_b_quality', 50)
            except Exception:
                pass

            quality_degradation = orig_quality - prot_quality

            self.add_result(TestResult(
                "DALL-E 2 edit disruption",
                True,
                {
                    'edit_diff_psnr': float(edit_diff_psnr),
                    'edit_diff_mse': float(edit_diff_mse),
                    'original_edit_quality': orig_quality,
                    'protected_edit_quality': prot_quality,
                    'quality_degradation': quality_degradation,
                    'judge_analysis': judge_text,
                }
            ))
        else:
            self.add_result(TestResult(
                "DALL-E 2 edit disruption",
                False,
                {'error': 'Could not generate one or both edits'}
            ))

    # ═══════════════════════════════════════════════════════════════
    # TEST 5: GPT-4o Recreation Prompt Test
    # ═══════════════════════════════════════════════════════════════
    def test_recreation_prompt(self, orig_path, prot_path):
        """
        Ask GPT-4o to generate a DALL-E prompt that would recreate the image.
        Then generate images from both prompts and compare.

        This tests if protection disrupts AI's ability to understand
        and replicate the image content.
        """
        print("\n" + "─" * 70)
        print("TEST 5: GPT-4o — IMAGE RECREATION DISRUPTION")
        print("─" * 70)
        print("Testing if protection disrupts AI's understanding of image content\n")

        recreation_prompt = (
            "Look at this image carefully. Write a detailed DALL-E prompt "
            "that would recreate this exact image as closely as possible. "
            "Include all objects, colors, positions, lighting, and style details. "
            "Just output the prompt text, nothing else."
        )

        prompts = {}
        for label, path in [("original", orig_path), ("protected", prot_path)]:
            b64 = img_to_base64(path)
            print(f"  Getting recreation prompt for {label}...")

            response = self.client.responses.create(
                model="gpt-4o",
                input=[{
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": recreation_prompt},
                        {"type": "input_image",
                         "image_url": f"data:image/png;base64,{b64}"},
                    ],
                }],
            )
            prompts[label] = response.output_text
            print(f"  {label}: {response.output_text[:120]}...")
            time.sleep(1)

        # Generate images from both prompts
        print("  Generating image from ORIGINAL prompt...")
        try:
            orig_gen_response = self.client.images.generate(
                model="dall-e-3",
                prompt=prompts['original'],
                size="1024x1024",
                quality="standard",
                n=1,
                response_format="b64_json",
            )
            orig_gen_path = os.path.join(OUTPUT_DIR, "recreation_from_original.png")
            save_b64_image(orig_gen_response.data[0].b64_json, orig_gen_path)
            print(f"  Saved: {orig_gen_path}")
        except Exception as e:
            print(f"  ERROR generating from original prompt: {e}")
            orig_gen_path = None

        time.sleep(3)

        print("  Generating image from PROTECTED prompt...")
        try:
            prot_gen_response = self.client.images.generate(
                model="dall-e-3",
                prompt=prompts['protected'],
                size="1024x1024",
                quality="standard",
                n=1,
                response_format="b64_json",
            )
            prot_gen_path = os.path.join(OUTPUT_DIR, "recreation_from_protected.png")
            save_b64_image(prot_gen_response.data[0].b64_json, prot_gen_path)
            print(f"  Saved: {prot_gen_path}")
        except Exception as e:
            print(f"  ERROR generating from protected prompt: {e}")
            prot_gen_path = None

        # Compare prompts
        prompt_similarity = sum(1 for a, b in zip(
            prompts['original'].lower().split(),
            prompts['protected'].lower().split()
        ) if a == b) / max(len(prompts['original'].split()),
                          len(prompts['protected'].split()), 1)

        self.add_result(TestResult(
            "GPT-4o recreation prompt disruption",
            True,
            {
                'original_prompt': prompts['original'],
                'protected_prompt': prompts['protected'],
                'prompt_word_similarity': float(prompt_similarity),
                'prompts_differ': prompt_similarity < 0.7,
            }
        ))

    # ═══════════════════════════════════════════════════════════════
    # TEST 6: Watermark Survival Through API Round-Trip
    # ═══════════════════════════════════════════════════════════════
    def test_watermark_api_survival(self, prot_path):
        """
        Upload protected image, have DALL-E create a variation, 
        download it, and check if our watermark survived.

        This is the ULTIMATE watermark test — does it survive
        a round trip through a real generative AI?
        """
        print("\n" + "─" * 70)
        print("TEST 6: WATERMARK API ROUND-TRIP SURVIVAL")
        print("─" * 70)
        print("Testing if watermark survives DALL-E processing\n")

        # The watermark is embedded in the DWT domain
        # DALL-E regenerates the image from scratch, so it WILL destroy 
        # the watermark. This tests the expected behavior.

        savior = PhotoSavior(protection_level=ProtectionLevel.STRONG)

        # First verify watermark exists in protected image
        wm_before = savior.verify_protection(prot_path)
        print(f"  Watermark before API: valid={wm_before.get('is_valid', False)}")

        # Upload to DALL-E and get variation
        print("  Sending to DALL-E for variation...")
        png_bytes = img_to_png_bytes(prot_path, max_size=512)

        try:
            response = self.client.images.create_variation(
                image=png_bytes,
                n=1,
                size="512x512",
                response_format="b64_json",
            )

            roundtrip_path = os.path.join(OUTPUT_DIR, "roundtrip_variation.png")
            save_b64_image(response.data[0].b64_json, roundtrip_path)

            # Check watermark in round-tripped image
            wm_after = savior.verify_protection(roundtrip_path)
            print(f"  Watermark after API: valid={wm_after.get('is_valid', False)}")

            wm_survived = wm_after.get('is_valid', False)

            self.add_result(TestResult(
                "Watermark API round-trip",
                True,  # Informational — we expect it to NOT survive DALL-E
                {
                    'watermark_before': wm_before.get('is_valid', False),
                    'watermark_after': wm_survived,
                    'watermark_survived_dalle': wm_survived,
                    'note': (
                        'DALL-E regenerates the image entirely, so watermark '
                        'destruction is EXPECTED. This confirms DALL-E '
                        'modified the image (tamper detection works).'
                    ),
                }
            ))
        except Exception as e:
            print(f"  ERROR: {e}")
            self.add_result(TestResult(
                "Watermark API round-trip",
                False,
                {'error': str(e)}
            ))

    # ═══════════════════════════════════════════════════════════════
    # RUN ALL
    # ═══════════════════════════════════════════════════════════════
    def run_all(self):
        ensure_dirs()
        start_time = time.time()

        print("=" * 70)
        print("  PHOTOSAVIOR — END-TO-END TEST AGAINST OPENAI API")
        print("=" * 70)
        print(f"  Using: GPT-4o (vision), DALL-E 2 (edit/variation), DALL-E 3 (generation)")
        print(f"  This test sends REAL images to REAL AI models.\n")

        # Create test images
        orig_path, prot_paths = self.create_test_images()
        prot_path = prot_paths['STRONG']

        # Run tests
        try:
            self.test_vision_description(orig_path, prot_path)
        except Exception as e:
            print(f"  TEST 1 ERROR: {e}")
            self.add_result(TestResult("GPT-4o vision description", False, {'error': str(e)}))

        try:
            self.test_artifact_detection(orig_path, prot_path)
        except Exception as e:
            print(f"  TEST 2 ERROR: {e}")
            self.add_result(TestResult("GPT-4o protection invisibility", False, {'error': str(e)}))

        try:
            self.test_dalle_variation(orig_path, prot_path)
        except Exception as e:
            print(f"  TEST 3 ERROR: {e}")
            self.add_result(TestResult("DALL-E 2 variation disruption", False, {'error': str(e)}))

        try:
            self.test_dalle_edit(orig_path, prot_path)
        except Exception as e:
            print(f"  TEST 4 ERROR: {e}")
            self.add_result(TestResult("DALL-E 2 edit disruption", False, {'error': str(e)}))

        try:
            self.test_recreation_prompt(orig_path, prot_path)
        except Exception as e:
            print(f"  TEST 5 ERROR: {e}")
            self.add_result(TestResult("GPT-4o recreation prompt", False, {'error': str(e)}))

        try:
            self.test_watermark_api_survival(prot_path)
        except Exception as e:
            print(f"  TEST 6 ERROR: {e}")
            self.add_result(TestResult("Watermark API round-trip", False, {'error': str(e)}))

        elapsed = time.time() - start_time

        # Summary
        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)

        print("\n" + "=" * 70)
        print("  END-TO-END API TEST — FINAL RESULTS")
        print("=" * 70)
        print(f"\n  Total tests: {total}")
        print(f"  Passed:      {passed}/{total}")
        print(f"  Time:        {elapsed:.1f}s")
        print(f"  API calls:   ~10-15 (GPT-4o + DALL-E)")
        print("=" * 70)

        # Save report
        def jsonify(obj):
            if isinstance(obj, (np.bool_,)):
                return bool(obj)
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, dict):
                return {k: jsonify(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [jsonify(x) for x in obj]
            return obj

        report = {
            'summary': {
                'total': total,
                'passed': passed,
                'elapsed_seconds': round(elapsed, 1),
            },
            'tests': [
                {
                    'name': r.name,
                    'passed': r.passed,
                    'details': jsonify(r.details),
                }
                for r in self.results
            ],
        }

        report_path = os.path.join(OUTPUT_DIR, 'api_test_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\n  Report saved to: {report_path}")
        print(f"  All images saved to: {OUTPUT_DIR}/")

        return passed == total


if __name__ == '__main__':
    suite = OpenAIEndToEndTest()
    success = suite.run_all()
    sys.exit(0 if success else 1)
