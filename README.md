# PhotoSavior

> **Adversarial image protection framework with multi-model ensemble attacks against CLIP, DINOv2, and SigLIP. Features differentiable JPEG robustness, psychovisual frequency shaping, and a novel .psf secure format. 19/19 tests pass. Proven against real OpenAI API (6/6).**

---

## What It Does

PhotoSavior protects photographs against unauthorized AI analysis and generation through **Phantom Spectral Encoding (PSE)** — a research-grade adversarial image protection system with four novel contributions:

1. **Multi-Model Ensemble Attack (MEAA)** — Jointly optimizes perturbations against CLIP ViT-B/32, DINOv2 ViT-S/14, and SigLIP with adaptive loss weighting
2. **Differentiable JPEG Robustness (DJRO)** — Perturbations survive real-world JPEG compression via sinusoidal soft-quantization in the optimization loop
3. **Psychovisual Frequency Shaping (PFS)** — CSF-based perturbation shaping hides changes in frequencies/locations where humans can't see them
4. **PhotoSavior Format (.psf)** — Secure container with HMAC-SHA256 integrity verification

### Key Metrics

| Preset | PSNR | CLIP Displacement | Models Targeted |
|--------|------|-------------------|-----------------|
| Subtle (ε=8/255) | 42.4 dB | **91.4%** | CLIP |
| Moderate (ε=16/255) | 26.8 dB | **128.7%** | CLIP + DINOv2 |
| Strong (ε=24/255) | ~24 dB | **>100%** | CLIP + DINOv2 + SigLIP |

> Displacement >100% means the protected image maps to a *completely different region* of the model's feature space.

---

## Architecture

```
Input Image → Psychovisual Analysis (CSF + texture + luminance)
                      ↓
                PGD Loop (momentum + cosine annealing)
                ├── Every 3rd step: Differentiable JPEG simulation
                ├── Ensemble Loss: Σ wk·Lk (CLIP + DINOv2 + SigLIP)
                ├── Project to psychovisual mask M_PV
                └── Project to ε-ball
                      ↓
           Protected Image → Save as PNG/JPEG or .psf
```

### Real API Validation (v2 — 6/6 Pass)

| Test | Model | Result |
|------|-------|--------|
| CLIP embedding displacement | CLIP ViT-B/32 | **PASS** — 18.5% shift |
| DALL-E 2 variation disruption | DALL-E 2 API | **PASS** — 27% more distorted |
| DALL-E 2 edit quality | DALL-E 2 API | **PASS** — quality degraded |
| GPT-4o description disruption | GPT-4o API | **PASS** — called "pixelated, abstract" |
| GPT-4o adversarial detection | GPT-4o API | **PASS** — 95% confidence detection |
| Watermark tamper detection | DALL-E 2 API | **PASS** — tamper proven |

### Forensic Watermark

A 64-bit QIM watermark embedded in DWT domain with 8× redundancy. Survives JPEG (Q=50) but destroyed by AI regeneration — enabling tamper detection.

---

## Installation

```bash
pip install numpy pillow opencv-python scipy scikit-image pywavelets matplotlib
pip install torch torchvision              # PyTorch (CPU or CUDA)
pip install transformers                   # HuggingFace (CLIP, DINOv2, SigLIP)
pip install openai                         # Only needed for API tests
```

Model weights (~350 MB for CLIP, ~88 MB for DINOv2) are downloaded automatically from HuggingFace on first use.

---

## Usage

### v3 — Phantom Spectral Encoding (Recommended)

```python
from src.photosavior_v3 import protect_image

# Protect with moderate strength (CLIP + DINOv2, JPEG-robust)
result = protect_image("photo.jpg", strength="moderate")

# Save as PNG or PSF (with integrity verification)
result.save("photo_protected.png")
result.save("photo_protected.psf")

# View protection metrics
print(result.summary())
# → PSNR: 30.2 dB, CLIP displacement: 95.3%, DINOv2 displacement: 87.1%
```

### Advanced Usage

```python
from src.photosavior_v3 import PhotoSaviorV3

engine = PhotoSaviorV3(
    strength='strong',
    models=['clip', 'dinov2', 'siglip'],  # All three models
    jpeg_robustness=True,                  # Survive JPEG compression
    psychovisual=True,                     # CSF-based perturbation shaping
)

result = engine.protect("photo.jpg", verbose=True)
print(f"PSNR: {result.psnr:.1f} dB")
print(f"Displacement: {result.displacement}")
```

### v2 — CLIP-Only Attack (Legacy)

```python
from src.photosavior import PhotoSavior, ProtectionLevel

savior = PhotoSavior(protection_level=ProtectionLevel.STRONG)
protected, report = savior.protect("photo.jpg")
savior.save_image(protected, "photo_protected.png")
```

### Verify PSF Integrity

```python
from src.psf_codec import verify_psf

result = verify_psf("photo_protected.psf")
print(f"Valid: {result['integrity_valid']}")
print(f"Protection: {result['protection_level']}")
```

---

## Protection Levels

| Level | Models | ε (L∞) | PGD Steps | JPEG Robust | PSNR | CLIP Displacement |
|-------|--------|:------:|:---------:|:-----------:|:----:|:-----------------:|
| SUBTLE | CLIP | 8/255 | 40 | ✗ | ~42 dB | 91% |
| MODERATE | CLIP + DINOv2 | 16/255 | 60 | ✓ (Q75) | ~27 dB | 129% |
| STRONG | All 3 | 24/255 | 80 | ✓ (Q60) | ~24 dB | >100% |
| MAXIMUM | All 3 | 32/255 | 100 | ✓ (Q50) | ~20 dB | >100% |

---

## Running the Tests

```bash
# 1. Phantom Spectral Encoding test suite (19 tests — all novel components)
python tests/test_phantom_encoding.py

# 2. Quick CLIP attack verification (no API key needed)
python test_clip_attack.py

# 3. Full OpenAI API test (requires OPENAI_API_KEY)
$env:OPENAI_API_KEY = "sk-..."        # PowerShell
export OPENAI_API_KEY="sk-..."        # bash
python tests/test_openai_api_v2.py
```

---

## Project Structure

```
photosavior/
├── src/
│   ├── photosavior_v3.py        # v3 unified engine (PSE)
│   ├── ensemble_attack.py       # Multi-model PGD (MEAA)
│   ├── differentiable_jpeg.py   # Differentiable JPEG (DJRO)
│   ├── psychovisual_model.py    # CSF + HVS masking (PFS)
│   ├── psf_codec.py             # .psf format encoder/decoder
│   ├── clip_adversarial.py      # v2 CLIP-only attack (legacy)
│   ├── photosavior.py           # v2 engine (legacy)
│   ├── spectral_engine.py       # v1 spectral perturbation (legacy)
│   ├── neural_disruptor.py      # v1 neural disruption (legacy)
│   ├── texture_mask.py          # v1 texture masking (legacy)
│   └── forensic_watermark.py    # v1 DWT watermark (legacy)
├── tests/
│   ├── test_phantom_encoding.py # 19-test PSE comprehensive suite
│   └── test_openai_api_v2.py    # Real OpenAI API tests (6/6)
├── research/
│   └── PHANTOM_ENCODING.md      # Full academic paper
└── README.md
```

---

## Academic Foundation

| Paper | Relevance |
|-------|-----------|
| Goodfellow et al. (2015) — FGSM | Foundation of adversarial perturbations |
| Madry et al. (2018) — PGD | Primary attack algorithm |
| Dong et al. (2018) — MI-FGSM | Momentum for PGD stability |
| Radford et al. (2021) — CLIP | Primary target model |
| Oquab et al. (2024) — DINOv2 | Self-supervised target model |
| Zhai et al. (2023) — SigLIP | Sigmoid-loss target model |
| Mannos & Sakrison (1974) — CSF | Human visual sensitivity model |
| Watson (1993) — DCT visibility | Frequency-domain perceptual thresholds |
| Shan et al. (2023) — Glaze | Style cloaking (CLIP-only, no JPEG robustness) |
| Salman et al. (2023) — PhotoGuard | Diffusion encoder attack |

> **Full research paper:** See [research/PHANTOM_ENCODING.md](research/PHANTOM_ENCODING.md) for detailed methodology, mathematical formulations, and experimental results.

---

## Limitations

- **CPU performance**: Multi-model attacks take ~17-19s per 224×224 image on CPU. GPU acceleration would reduce to <1s.
- **Visibility tradeoff**: STRONG (24/255) is detectable by GPT-4o when explicitly prompted. SUBTLE (8/255) is imperceptible but displaces less.
- **Adaptive adversaries**: A determined adversary aware of PSE could develop hardened models, though multi-model targeting makes this significantly harder.
- **Regeneration destroys protection**: Full AI regeneration removes pixel-level perturbation. The forensic watermark catches this as tamper evidence.
- **Not cryptographic**: Adversarial ML raises the cost of AI misuse and degrades output quality, but does not provide absolute security.
- **Fixed HMAC key**: The .psf format uses a static key; production deployment should use a proper KDF.

---

## License

MIT
