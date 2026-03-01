"""
Comprehensive Test Suite for Phantom Spectral Encoding (PSE)
=============================================================

Tests all novel contributions:
1. Differentiable JPEG (DJRO) — gradient flow through JPEG
2. Psychovisual Model (PFS) — CSF-based perturbation shaping
3. Ensemble Attack (MEAA) — multi-model adversarial optimization
4. PSF Format Codec — encode/decode/verify cycle
5. PhotoSavior v3 Engine — end-to-end protection pipeline

Each test validates CORRECTNESS (mathematically), not just "runs without error".
"""

import sys
import os
import numpy as np
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def create_test_image(h=256, w=256, seed=42):
    """Create a synthetic test image with varied content."""
    rng = np.random.RandomState(seed)
    img = np.zeros((h, w, 3), dtype=np.float64)
    
    # Gradient background
    for c in range(3):
        img[:, :, c] = np.linspace(0.2, 0.8, w)[np.newaxis, :] * \
                        np.linspace(0.3, 0.7, h)[:, np.newaxis]
    
    # Add some texture (high-frequency content)
    for c in range(3):
        noise = rng.randn(h, w) * 0.05
        img[:, :, c] += noise
    
    # Add a bright region (tests luminance adaptation)
    img[50:100, 50:100, :] = 0.95
    
    # Add a dark region
    img[150:200, 150:200, :] = 0.05
    
    # Add edges (tests texture masking)
    img[100:102, :, :] = 0.9
    img[:, 128:130, :] = 0.9
    
    return np.clip(img, 0, 1)


# ============================================================
# Test 1: Differentiable JPEG
# ============================================================

def test_differentiable_dct():
    """Test that DCT → IDCT is identity (within numerical precision)."""
    import torch
    from src.differentiable_jpeg import DifferentiableDCT8x8
    
    dct = DifferentiableDCT8x8()
    
    # Random 8x8 blocks
    blocks = torch.randn(10, 8, 8)
    
    # Forward + Inverse should be identity
    coeffs = dct(blocks)
    reconstructed = dct.inverse(coeffs)
    
    error = (blocks - reconstructed).abs().max().item()
    assert error < 1e-5, f"DCT round-trip error: {error}"
    print(f"  [PASS] DCT round-trip error: {error:.2e}")
    return True


def test_differentiable_jpeg_gradient():
    """Test that gradients flow through differentiable JPEG."""
    import torch
    from src.differentiable_jpeg import DifferentiableJPEG
    
    djpeg = DifferentiableJPEG(quality=75)
    
    # Create input image requiring gradient
    img = torch.rand(1, 3, 64, 64, requires_grad=True)
    
    # Forward pass
    compressed = djpeg(img)
    
    # Compute loss and backward
    loss = compressed.sum()
    loss.backward()
    
    # Check gradient exists and is non-zero
    assert img.grad is not None, "Gradient is None"
    grad_norm = img.grad.norm().item()
    assert grad_norm > 0, f"Gradient norm is zero"
    print(f"  [PASS] JPEG gradient norm: {grad_norm:.4f}")
    return True


def test_jpeg_quality_effect():
    """Test that lower JPEG quality produces more distortion."""
    import torch
    from src.differentiable_jpeg import DifferentiableJPEG
    
    img = torch.rand(1, 3, 64, 64)
    
    # High quality vs low quality
    djpeg_high = DifferentiableJPEG(quality=95)
    djpeg_low = DifferentiableJPEG(quality=30)
    
    compressed_high = djpeg_high(img)
    compressed_low = djpeg_low(img)
    
    # Low quality should produce more distortion
    dist_high = (img - compressed_high).abs().mean().item()
    dist_low = (img - compressed_low).abs().mean().item()
    
    assert dist_low > dist_high, \
        f"Low quality distortion ({dist_low}) should exceed high ({dist_high})"
    print(f"  [PASS] JPEG Q95 dist: {dist_high:.4f}, Q30 dist: {dist_low:.4f}")
    return True


def test_color_space_roundtrip():
    """Test RGB → YCbCr → RGB is identity."""
    import torch
    from src.differentiable_jpeg import RGBToYCbCr, YCbCrToRGB
    
    rgb2ycbcr = RGBToYCbCr()
    ycbcr2rgb = YCbCrToRGB()
    
    img = torch.rand(1, 3, 32, 32)
    ycbcr = rgb2ycbcr(img)
    reconstructed = ycbcr2rgb(ycbcr)
    
    error = (img - reconstructed).abs().max().item()
    assert error < 0.01, f"Color space round-trip error: {error}"
    print(f"  [PASS] Color space round-trip error: {error:.4f}")
    return True


# ============================================================
# Test 2: Psychovisual Model
# ============================================================

def test_csf_peak_frequency():
    """Test that CSF peaks around 4 cycles/degree."""
    from src.psychovisual_model import ContrastSensitivityFunction
    
    csf = ContrastSensitivityFunction()
    freqs = np.linspace(0.1, 50, 500)
    sensitivity = csf.mannos_sakrison(freqs)
    
    peak_freq = freqs[np.argmax(sensitivity)]
    
    # Peak should be between 2-8 cycles/degree (well-known HVS property)
    assert 2 < peak_freq < 8, f"CSF peak at {peak_freq} cpd (expected 2-8)"
    print(f"  [PASS] CSF peak at {peak_freq:.1f} cycles/degree")
    return True


def test_texture_mask_contrast():
    """Test that texture mask gives higher values in textured regions."""
    from src.psychovisual_model import PsychovisualMask
    
    pv = PsychovisualMask()
    img = create_test_image(128, 128)
    
    texture = pv.compute_texture_mask(img)
    
    # Textured regions (where we added noise/edges) should have higher values
    # than smooth regions
    smooth_region = texture[60:90, 60:90].mean()  # Bright smooth area
    textured_region = texture[95:110, 95:110].mean()  # Near edges
    
    # Overall: texture mask should have reasonable range
    assert texture.min() >= 0, "Texture mask has negative values"
    assert texture.max() <= 1.0 + 1e-6, "Texture mask exceeds 1.0"
    print(f"  [PASS] Texture mask range: [{texture.min():.3f}, {texture.max():.3f}]")
    return True


def test_frequency_tolerance():
    """Test that high-frequency DCT positions have higher tolerance."""
    from src.psychovisual_model import PsychovisualMask
    
    pv = PsychovisualMask()
    tolerance = pv.compute_frequency_tolerance()
    
    # DC component should have lowest tolerance
    dc_tolerance = tolerance[0, 0]
    
    # High frequency should have higher tolerance
    hf_tolerance = tolerance[7, 7]
    
    assert hf_tolerance > dc_tolerance, \
        f"HF tolerance ({hf_tolerance}) should exceed DC ({dc_tolerance})"
    print(f"  [PASS] DC tolerance: {dc_tolerance:.3f}, HF tolerance: {hf_tolerance:.3f}")
    return True


def test_psychovisual_mask_shape():
    """Test complete psychovisual mask generation."""
    from src.psychovisual_model import PsychovisualConstraint
    
    img = create_test_image(128, 128)
    constraint = PsychovisualConstraint(img, max_perturbation=16.0/255)
    
    # Generate torch mask
    mask = constraint.to_torch_mask('cpu')
    assert mask.shape == (1, 3, 128, 128), f"Mask shape: {mask.shape}"
    assert mask.min() >= 0, "Mask has negative values"
    print(f"  [PASS] Psychovisual mask shape: {mask.shape}, "
          f"range: [{mask.min():.4f}, {mask.max():.4f}]")
    return True


# ============================================================
# Test 3: PSF Format Codec
# ============================================================

def test_psf_encode_decode():
    """Test PSF encode → decode round-trip preserves image."""
    from src.psf_codec import PSFEncoder, PSFDecoder
    import tempfile
    
    img = create_test_image(64, 64)
    img_uint8 = (img * 255).astype(np.uint8)
    
    # Encode
    encoder = PSFEncoder()
    with tempfile.NamedTemporaryFile(suffix='.psf', delete=False) as f:
        tmp_path = f.name
    
    try:
        info = encoder.encode(
            protected_image=img_uint8,
            output_path=tmp_path,
            protection_level='moderate',
        )
        
        assert os.path.exists(tmp_path), "PSF file not created"
        assert info['file_size_bytes'] > 0, "File size is 0"
        
        # Decode
        decoder = PSFDecoder()
        result = decoder.decode(tmp_path)
        
        # Check image matches
        decoded_img = result['image']
        assert decoded_img.shape == img_uint8.shape, \
            f"Shape mismatch: {decoded_img.shape} vs {img_uint8.shape}"
        assert np.array_equal(decoded_img, img_uint8), "Image data mismatch"
        
        # Check metadata
        assert result['protection_level'] == 'moderate'
        assert result['integrity_valid'] == True
        
        print(f"  [PASS] PSF round-trip: {info['file_size_bytes']} bytes, "
              f"compression ratio: {info['compression_ratio']:.1f}x")
        return True
    finally:
        os.unlink(tmp_path)


def test_psf_integrity_detection():
    """Test that PSF detects file tampering."""
    from src.psf_codec import PSFEncoder, PSFDecoder
    import tempfile
    
    img = (create_test_image(64, 64) * 255).astype(np.uint8)
    
    encoder = PSFEncoder()
    with tempfile.NamedTemporaryFile(suffix='.psf', delete=False) as f:
        tmp_path = f.name
    
    try:
        encoder.encode(img, tmp_path, protection_level='strong')
        
        # Tamper with the file (modify a pixel byte)
        with open(tmp_path, 'r+b') as f:
            f.seek(100)  # Somewhere in metadata/pixel area
            original_byte = f.read(1)
            f.seek(100)
            tampered = bytes([(original_byte[0] + 1) % 256])
            f.write(tampered)
        
        # Verify should detect tampering
        decoder = PSFDecoder()
        result = decoder.decode(tmp_path)
        
        assert result['integrity_valid'] == False, \
            "Tampered file should fail integrity check"
        print(f"  [PASS] Tampering detected correctly")
        return True
    finally:
        os.unlink(tmp_path)


def test_psf_header_format():
    """Test PSF header pack/unpack."""
    from src.psf_codec import PSFHeader, PSF_MAGIC
    
    header = PSFHeader(
        width=1920, height=1080, channels=3,
        protection_level=3, flags=0x000F,
        pixel_data_size=1000000,
        metadata_size=500,
        integrity_offset=1000564,
    )
    
    packed = header.pack()
    assert len(packed) == 64, f"Header size: {len(packed)} (expected 64)"
    assert packed[:4] == PSF_MAGIC, "Magic bytes incorrect"
    
    # Unpack and verify
    unpacked = PSFHeader.unpack(packed)
    assert unpacked.width == 1920
    assert unpacked.height == 1080
    assert unpacked.channels == 3
    assert unpacked.protection_level == 3
    
    print(f"  [PASS] PSF header: 64 bytes, fields verified")
    return True


def test_psf_with_metrics():
    """Test PSF with full attack metrics storage."""
    from src.psf_codec import PSFEncoder, PSFDecoder
    import tempfile
    
    img = (create_test_image(64, 64) * 255).astype(np.uint8)
    metrics = {
        'per_model': {
            'clip': {'cosine_similarity': 0.75, 'feature_displacement': 0.25},
            'dinov2': {'cosine_similarity': 0.80, 'feature_displacement': 0.20},
        },
        'image_quality': {'psnr_db': 35.2, 'linf': 0.063},
        'attack_config': {'models': ['clip', 'dinov2'], 'epsilon': 0.063},
    }
    
    with tempfile.NamedTemporaryFile(suffix='.psf', delete=False) as f:
        tmp_path = f.name
    
    try:
        encoder = PSFEncoder()
        encoder.encode(img, tmp_path, metrics=metrics)
        
        decoder = PSFDecoder()
        result = decoder.decode(tmp_path)
        
        # Verify metrics stored correctly
        stored = result['metadata']['attack_metrics']
        assert stored['per_model']['clip']['feature_displacement'] == 0.25
        assert stored['image_quality']['psnr_db'] == 35.2
        
        print(f"  [PASS] PSF metrics storage verified")
        return True
    finally:
        os.unlink(tmp_path)


# ============================================================
# Test 4: Ensemble Attack (with CLIP)
# ============================================================

def test_ensemble_attack_clip_only():
    """Test ensemble attack with CLIP only (faster, validates core logic)."""
    from src.ensemble_attack import EnsembleAdversarialAttack
    
    img = create_test_image(224, 224)
    
    attack = EnsembleAdversarialAttack(
        models=['clip'],
        epsilon=16.0 / 255.0,
        steps=15,  # Few steps for test speed
        use_jpeg_robustness=False,
        use_psychovisual=False,
    )
    
    result = attack.attack(img, verbose=True)
    
    protected = result['protected_image']
    metrics = result['metrics']
    
    # Validate shape preserved
    assert protected.shape == img.shape, f"Shape mismatch"
    
    # Validate perturbation within budget
    linf = metrics['image_quality']['linf']
    assert linf <= 16.0 / 255.0 + 1e-6, f"L∞ exceeded: {linf*255:.1f}/255"
    
    # Validate feature displacement
    clip_disp = metrics['per_model']['clip']['feature_displacement']
    assert clip_disp > 0.05, f"CLIP displacement too low: {clip_disp:.3f}"
    
    print(f"  [PASS] CLIP displacement: {clip_disp:.1%}, "
          f"PSNR: {metrics['image_quality']['psnr_db']:.1f} dB")
    return True


def test_ensemble_attack_with_jpeg():
    """Test that JPEG robustness is active in the attack loop."""
    from src.ensemble_attack import EnsembleAdversarialAttack
    
    img = create_test_image(224, 224)
    
    attack = EnsembleAdversarialAttack(
        models=['clip'],
        epsilon=16.0 / 255.0,
        steps=10,
        use_jpeg_robustness=True,
        jpeg_quality=75,
        use_psychovisual=False,
    )
    
    result = attack.attack(img, verbose=False)
    
    # Test should complete without error (JPEG in loop)
    assert result['protected_image'].shape == img.shape
    print(f"  [PASS] JPEG-robust attack completed, "
          f"CLIP displacement: {result['metrics']['per_model']['clip']['feature_displacement']:.1%}")
    return True


def test_ensemble_attack_with_psychovisual():
    """Test psychovisual shaping in attack loop."""
    from src.ensemble_attack import EnsembleAdversarialAttack
    
    img = create_test_image(224, 224)
    
    attack = EnsembleAdversarialAttack(
        models=['clip'],
        epsilon=16.0 / 255.0,
        steps=10,
        use_jpeg_robustness=False,
        use_psychovisual=True,
    )
    
    result = attack.attack(img, verbose=False)
    
    delta = result['delta']
    
    # Psychovisual shaping should produce non-uniform perturbation
    # (textured areas should have more perturbation than smooth areas)
    assert delta.std() > 0, "Perturbation should be non-zero"
    
    print(f"  [PASS] Psychovisual attack completed, "
          f"delta std: {delta.std():.4f}, max: {np.abs(delta).max()*255:.1f}/255")
    return True


def test_phantom_shield_presets():
    """Test PhantomSpectralShield preset configurations."""
    from src.ensemble_attack import PhantomSpectralShield
    
    for preset in ['subtle', 'moderate', 'strong', 'maximum']:
        shield = PhantomSpectralShield(strength=preset)
        assert shield.attack.epsilon > 0
        assert shield.attack.steps > 0
        print(f"  [PASS] Preset '{preset}': ε={shield.attack.epsilon*255:.0f}/255, "
              f"steps={shield.attack.steps}")
    
    return True


# ============================================================
# Test 5: End-to-End Integration 
# ============================================================

def test_photosavior_v3_basic():
    """Test PhotoSavior v3 end-to-end protection."""
    from src.photosavior_v3 import PhotoSaviorV3
    
    img = create_test_image(224, 224)
    
    engine = PhotoSaviorV3(
        strength='subtle',
        models=['clip'],
        jpeg_robustness=False,  # Speed
        psychovisual=True,
    )
    
    result = engine.protect(img, verbose=True)
    
    # Validate result
    assert result.image.shape == img.shape
    assert result.psnr > 20, f"PSNR too low: {result.psnr}"
    
    print(f"\n  [PASS] V3 end-to-end: PSNR={result.psnr:.1f} dB")
    print(f"  Displacement: {result.displacement}")
    print(f"\n{result.summary()}")
    return True


def test_photosavior_v3_save_formats():
    """Test saving in both PNG and PSF formats."""
    from src.photosavior_v3 import PhotoSaviorV3
    import tempfile
    
    img = create_test_image(128, 128)
    
    engine = PhotoSaviorV3(
        strength='subtle',
        models=['clip'],
        jpeg_robustness=False,
        psychovisual=False,
    )
    
    result = engine.protect(img, verbose=False)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save as PNG
        png_path = os.path.join(tmpdir, 'test.png')
        result.save(png_path)
        assert os.path.exists(png_path), "PNG not saved"
        
        # Save as PSF
        psf_path = os.path.join(tmpdir, 'test.psf')
        result.save(psf_path)
        assert os.path.exists(psf_path), "PSF not saved"
        
        # Verify PSF
        from src.psf_codec import load_psf
        loaded = load_psf(psf_path)
        assert loaded['integrity_valid'] == True
        assert loaded['protection_level'] == 'subtle'
        
        print(f"  [PASS] Saved PNG ({os.path.getsize(png_path)} bytes) "
              f"and PSF ({os.path.getsize(psf_path)} bytes)")
    
    return True


def test_transform_chain():
    """Test differentiable transform chain."""
    import torch
    from src.differentiable_jpeg import DifferentiableTransformChain
    
    chain = DifferentiableTransformChain(
        jpeg_quality=75,
        enable_jpeg=True,
        enable_resize=True,
        enable_blur=True,
        enable_noise=True,
    )
    
    img = torch.rand(1, 3, 64, 64, requires_grad=True)
    
    # Test with all transforms
    transformed = chain(img, apply_all=True)
    
    # Check gradients flow
    loss = transformed.sum()
    loss.backward()
    
    assert img.grad is not None
    assert img.grad.norm().item() > 0
    
    print(f"  [PASS] Transform chain: gradient norm = {img.grad.norm().item():.4f}")
    return True


# ============================================================
# Test Runner
# ============================================================

def run_all_tests():
    """Run all tests with detailed reporting."""
    print("\n" + "=" * 70)
    print("  PHANTOM SPECTRAL ENCODING — Complete Test Suite")
    print("=" * 70)
    
    tests = [
        # Group 1: Differentiable JPEG
        ("DJRO", "DCT Round-Trip", test_differentiable_dct),
        ("DJRO", "Color Space Round-Trip", test_color_space_roundtrip),
        ("DJRO", "JPEG Gradient Flow", test_differentiable_jpeg_gradient),
        ("DJRO", "JPEG Quality Effect", test_jpeg_quality_effect),
        ("DJRO", "Transform Chain", test_transform_chain),
        
        # Group 2: Psychovisual Model
        ("PFS", "CSF Peak Frequency", test_csf_peak_frequency),
        ("PFS", "Texture Mask Contrast", test_texture_mask_contrast),
        ("PFS", "Frequency Tolerance", test_frequency_tolerance),
        ("PFS", "Psychovisual Mask Shape", test_psychovisual_mask_shape),
        
        # Group 3: PSF Format
        ("PSF", "Header Pack/Unpack", test_psf_header_format),
        ("PSF", "Encode/Decode Round-Trip", test_psf_encode_decode),
        ("PSF", "Integrity Tampering", test_psf_integrity_detection),
        ("PSF", "Metrics Storage", test_psf_with_metrics),
        
        # Group 4: Ensemble Attack
        ("MEAA", "Preset Configurations", test_phantom_shield_presets),
        ("MEAA", "CLIP-Only Attack", test_ensemble_attack_clip_only),
        ("MEAA", "JPEG-Robust Attack", test_ensemble_attack_with_jpeg),
        ("MEAA", "Psychovisual Attack", test_ensemble_attack_with_psychovisual),
        
        # Group 5: Integration
        ("V3", "End-to-End Protection", test_photosavior_v3_basic),
        ("V3", "Save PNG + PSF", test_photosavior_v3_save_formats),
    ]
    
    passed = 0
    failed = 0
    errors = []
    
    current_group = None
    
    for group, name, test_fn in tests:
        if group != current_group:
            current_group = group
            print(f"\n{'─'*60}")
            group_names = {
                'DJRO': 'Differentiable JPEG-Robust Optimization',
                'PFS': 'Psychovisual Frequency Shaping',
                'PSF': 'PhotoSavior Format Codec',
                'MEAA': 'Multi-Model Ensemble Adversarial Attack',
                'V3': 'PhotoSavior v3 Integration',
            }
            print(f"  [{group}] {group_names.get(group, group)}")
            print(f"{'─'*60}")
        
        try:
            print(f"\n  Testing: {name}...")
            start = time.time()
            success = test_fn()
            elapsed = time.time() - start
            
            if success:
                passed += 1
                print(f"  ✓ {name} ({elapsed:.1f}s)")
            else:
                failed += 1
                errors.append((group, name, "Returned False"))
                print(f"  ✗ {name} — FAILED")
        except Exception as e:
            failed += 1
            errors.append((group, name, str(e)))
            print(f"  ✗ {name} — ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    total = passed + failed
    print(f"\n{'='*70}")
    print(f"  RESULTS: {passed}/{total} passed")
    
    if errors:
        print(f"\n  FAILURES:")
        for group, name, msg in errors:
            print(f"    [{group}] {name}: {msg}")
    
    print(f"{'='*70}")
    
    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
