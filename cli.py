#!/usr/bin/env python3
"""
PhotoSavior CLI — Phantom Spectral Encoding Command Line Interface
===================================================================

Protect your images against AI with a single command.

Usage:
  python -m photosavior protect photo.jpg
  python -m photosavior protect photo.jpg -o protected.png --strength moderate
  python -m photosavior protect photo.jpg -o protected.psf --format psf
  python -m photosavior protect *.jpg -o output/ --strength strong
  python -m photosavior verify protected.psf
  python -m photosavior info protected.psf

Strength levels:
  subtle   - Low perturbation (8/255), single model, fast
  moderate - Medium perturbation (16/255), dual model (default)
  strong   - High perturbation (24/255), dual model
  maximum  - Maximum perturbation (32/255), triple model
"""

import argparse
import sys
import os
import time
import glob
import numpy as np
from pathlib import Path


def cmd_protect(args):
    """Protect image(s) against AI."""
    from src.photosavior_v3 import PhotoSaviorV3
    from PIL import Image

    # Expand glob patterns
    input_files = []
    for pattern in args.input:
        expanded = glob.glob(pattern)
        if expanded:
            input_files.extend(expanded)
        elif os.path.exists(pattern):
            input_files.append(pattern)
        else:
            print(f"Warning: '{pattern}' not found, skipping")

    if not input_files:
        print("Error: No input files found")
        return 1

    # Determine models from strength (or override)
    models = None
    if args.models:
        models = [m.strip() for m in args.models.split(',')]

    # Create engine
    engine = PhotoSaviorV3(
        strength=args.strength,
        models=models,
        jpeg_robustness=not args.no_jpeg,
        psychovisual=not args.no_psychovisual,
    )

    print(f"\nPhotoSavior v{engine.VERSION} - {engine.CODENAME}")
    print(f"Strength: {args.strength}")
    print(f"JPEG robustness: {not args.no_jpeg}")
    print(f"Psychovisual shaping: {not args.no_psychovisual}")
    print(f"Files to process: {len(input_files)}")
    print()

    total_start = time.time()
    results = []

    for i, input_path in enumerate(input_files):
        print(f"[{i+1}/{len(input_files)}] Processing: {input_path}")

        try:
            result = engine.protect(input_path, verbose=args.verbose)

            # Determine output path
            if args.output:
                if len(input_files) == 1 and not os.path.isdir(args.output):
                    out_path = args.output
                else:
                    out_dir = args.output
                    os.makedirs(out_dir, exist_ok=True)
                    stem = Path(input_path).stem
                    ext = f'.{args.format}' if args.format else '.png'
                    out_path = os.path.join(out_dir, f"{stem}_protected{ext}")
            else:
                stem = Path(input_path).stem
                parent = Path(input_path).parent
                ext = f'.{args.format}' if args.format else '.png'
                out_path = str(parent / f"{stem}_protected{ext}")

            result.save(out_path)

            # Print summary
            print(f"  Saved: {out_path}")
            print(f"  PSNR: {result.psnr:.1f} dB")
            for model, disp in result.displacement.items():
                print(f"  {model} displacement: {disp:.1%}")
            print()

            results.append({
                'input': input_path,
                'output': out_path,
                'psnr': result.psnr,
                'displacement': result.displacement,
            })

        except Exception as e:
            print(f"  Error: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()

    total_time = time.time() - total_start
    print(f"Completed {len(results)}/{len(input_files)} files in {total_time:.1f}s")

    return 0 if len(results) == len(input_files) else 1


def cmd_verify(args):
    """Verify a PSF file's integrity."""
    from src.psf_codec import verify_psf

    for path in args.input:
        if not path.lower().endswith('.psf'):
            print(f"Warning: {path} is not a .psf file")
            continue

        if not os.path.exists(path):
            print(f"Error: {path} not found")
            continue

        result = verify_psf(path)

        status = "VALID" if result['valid'] else "TAMPERED"
        level = result['protection_level']

        print(f"{path}: {status} (protection: {level})")

        if result.get('metadata'):
            meta = result['metadata']
            if 'attack_metrics' in meta:
                metrics = meta['attack_metrics']
                if 'image_quality' in metrics:
                    psnr = metrics['image_quality'].get('psnr_db', 'N/A')
                    print(f"  PSNR: {psnr}")
                if 'per_model' in metrics:
                    for model, data in metrics['per_model'].items():
                        disp = data.get('feature_displacement', 'N/A')
                        print(f"  {model} displacement: {disp}")

    return 0


def cmd_info(args):
    """Show detailed information about a PSF file."""
    from src.psf_codec import load_psf
    import json

    for path in args.input:
        if not os.path.exists(path):
            print(f"Error: {path} not found")
            continue

        result = load_psf(path, verify=True)

        print(f"\nFile: {path}")
        print(f"  Protection Level: {result['protection_level']}")
        print(f"  Integrity: {'Valid' if result['integrity_valid'] else 'TAMPERED'}")
        print(f"  Image Size: {result['header'].width}x{result['header'].height}")
        print(f"  Channels: {result['header'].channels}")

        if result.get('original_hash'):
            print(f"  Original Hash: {result['original_hash'][:16]}...")
        if result.get('protected_hash'):
            print(f"  Protected Hash: {result['protected_hash'][:16]}...")

        if result.get('metadata'):
            print(f"\n  Metadata:")
            print(f"  {json.dumps(result['metadata'], indent=4)}")

    return 0


def cmd_convert(args):
    """Convert PSF to standard image format."""
    from src.psf_codec import PSFDecoder
    from PIL import Image

    decoder = PSFDecoder()

    for path in args.input:
        if not os.path.exists(path):
            print(f"Error: {path} not found")
            continue

        if args.output:
            out_path = args.output
        else:
            out_path = str(Path(path).with_suffix(f'.{args.format}'))

        if args.format == 'png':
            decoder.to_png(path, out_path)
        else:
            img = decoder.to_pil(path)
            img.save(out_path, quality=args.quality)

        print(f"Converted: {path} -> {out_path}")

    return 0


def main():
    parser = argparse.ArgumentParser(
        prog='photosavior',
        description='PhotoSavior - AI-Resistant Image Protection with '
                    'Phantom Spectral Encoding',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli.py protect photo.jpg                        # Basic protection
  python cli.py protect photo.jpg -s strong              # Strong protection
  python cli.py protect photo.jpg -o protected.psf       # Save as PSF
  python cli.py protect *.jpg -o output/ -s moderate     # Batch protection
  python cli.py verify protected.psf                     # Verify integrity
  python cli.py info protected.psf                       # Show file info
  python cli.py convert protected.psf -f png             # Convert to PNG
        """,
    )

    subparsers = parser.add_subparsers(dest='command', help='Command')

    # Protect command
    p_protect = subparsers.add_parser('protect', help='Protect image(s)')
    p_protect.add_argument('input', nargs='+', help='Input image path(s)')
    p_protect.add_argument('-o', '--output', help='Output path or directory')
    p_protect.add_argument('-s', '--strength', default='moderate',
                           choices=['subtle', 'moderate', 'strong', 'maximum'],
                           help='Protection strength (default: moderate)')
    p_protect.add_argument('-f', '--format', default='png',
                           choices=['png', 'jpg', 'psf', 'bmp', 'tiff'],
                           help='Output format (default: png)')
    p_protect.add_argument('-m', '--models',
                           help='Comma-separated model list (clip,dinov2,siglip)')
    p_protect.add_argument('--no-jpeg', action='store_true',
                           help='Disable JPEG robustness optimization')
    p_protect.add_argument('--no-psychovisual', action='store_true',
                           help='Disable psychovisual frequency shaping')
    p_protect.add_argument('-v', '--verbose', action='store_true',
                           help='Verbose output')

    # Verify command
    p_verify = subparsers.add_parser('verify', help='Verify PSF file integrity')
    p_verify.add_argument('input', nargs='+', help='PSF file path(s)')

    # Info command
    p_info = subparsers.add_parser('info', help='Show PSF file information')
    p_info.add_argument('input', nargs='+', help='PSF file path(s)')

    # Convert command
    p_convert = subparsers.add_parser('convert', help='Convert PSF to image')
    p_convert.add_argument('input', nargs='+', help='PSF file path(s)')
    p_convert.add_argument('-o', '--output', help='Output path')
    p_convert.add_argument('-f', '--format', default='png',
                           choices=['png', 'jpg', 'bmp'],
                           help='Output format (default: png)')
    p_convert.add_argument('-q', '--quality', type=int, default=95,
                           help='JPEG quality (default: 95)')

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 1

    commands = {
        'protect': cmd_protect,
        'verify': cmd_verify,
        'info': cmd_info,
        'convert': cmd_convert,
    }

    return commands[args.command](args)


if __name__ == '__main__':
    sys.exit(main())
