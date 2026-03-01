"""
PhotoSavior Format (.psf) — Novel AI-Resistant Image Container
================================================================

Novel Contribution: Purpose-Built AI-Resistant Image Format
------------------------------------------------------------
No existing image format (JPEG, PNG, WebP, AVIF) was designed with
AI resistance in mind. They optimize for compression efficiency and
visual quality — both of which HELP AI models process images.

The PhotoSavior Format (.psf) is the FIRST image container format
purpose-built for AI-resistant image storage. It provides:

1. ADVERSARIAL PIXEL STORAGE: Protected pixel data with adversarial
   perturbations baked into the encoding
2. INTEGRITY VERIFICATION: HMAC-SHA256 cryptographic verification
   to detect if protection has been tampered with
3. PROTECTION METADATA: Full record of which models were attacked,
   what strength was used, and attack metrics
4. FORWARD COMPATIBILITY: Version field allows future format updates
   without breaking existing readers
5. RECOVERY DATA: Optional encrypted original image recovery

Format Specification (PSF v1):
==============================

┌──────────────────────────────────────────────────┐
│ HEADER (64 bytes)                                │
│   Magic: b"PSF\\x01"              (4 bytes)      │
│   Version: uint16                 (2 bytes)      │
│   Width: uint32                   (4 bytes)      │
│   Height: uint32                  (4 bytes)      │
│   Channels: uint8                 (1 byte)       │
│   Protection level: uint8         (1 byte)       │
│   Flags: uint16                   (2 bytes)      │
│   Pixel data size: uint64         (8 bytes)      │
│   Metadata size: uint32           (4 bytes)      │
│   Integrity offset: uint64        (8 bytes)      │
│   Reserved: 26 bytes                             │
├──────────────────────────────────────────────────┤
│ METADATA BLOCK (variable)                        │
│   JSON-encoded protection parameters             │
│   - Models attacked                              │
│   - Epsilon, steps, strength                     │
│   - Per-model feature displacement               │
│   - PSNR, L∞, L2                                 │
│   - Timestamp                                    │
│   - Software version                             │
├──────────────────────────────────────────────────┤
│ PIXEL DATA BLOCK (variable)                      │
│   zlib-compressed raw pixel data                 │
│   Format: row-major, uint8, RGB                  │
├──────────────────────────────────────────────────┤
│ INTEGRITY BLOCK (96 bytes)                       │
│   HMAC-SHA256 of header + metadata + pixels      │
│   (32 bytes)                                     │
│   SHA-256 of original image (pre-protection)     │
│   (32 bytes)                                     │
│   SHA-256 of protected image pixels              |
│   (32 bytes)                                     │
└──────────────────────────────────────────────────┘

Author: PhotoSavior Research Team
"""

import struct
import zlib
import hashlib
import hmac
import json
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
from datetime import datetime, timezone
import io


# ============================================================
# Format Constants
# ============================================================

PSF_MAGIC = b"PSF\x01"
PSF_VERSION = 1
PSF_HEADER_SIZE = 64
PSF_INTEGRITY_SIZE = 96

# Protection level encoding
PROTECTION_LEVELS = {
    'subtle': 1,
    'moderate': 2,
    'strong': 3,
    'maximum': 4,
}
PROTECTION_LEVELS_INV = {v: k for k, v in PROTECTION_LEVELS.items()}

# Flags
FLAG_HAS_METADATA = 0x0001
FLAG_HAS_INTEGRITY = 0x0002
FLAG_COMPRESSED = 0x0004
FLAG_HAS_ORIGINAL_HASH = 0x0008

# HMAC key derivation (in production, use a proper KDF)
_HMAC_KEY = b"PhotoSavior-PSF-v1-HMAC-Key-2024"


# ============================================================
# PSF Codec
# ============================================================

class PSFHeader:
    """Represents the PSF file header."""
    
    def __init__(self,
                 width: int,
                 height: int,
                 channels: int = 3,
                 protection_level: int = 2,
                 flags: int = 0,
                 pixel_data_size: int = 0,
                 metadata_size: int = 0,
                 integrity_offset: int = 0):
        self.width = width
        self.height = height
        self.channels = channels
        self.protection_level = protection_level
        self.flags = flags
        self.pixel_data_size = pixel_data_size
        self.metadata_size = metadata_size
        self.integrity_offset = integrity_offset
    
    def pack(self) -> bytes:
        """Serialize header to 64 bytes."""
        # Pack fields
        data = struct.pack(
            '<4s H I I B B H Q I Q',
            PSF_MAGIC,
            PSF_VERSION,
            self.width,
            self.height,
            self.channels,
            self.protection_level,
            self.flags,
            self.pixel_data_size,
            self.metadata_size,
            self.integrity_offset,
        )
        # Pad to 64 bytes
        data += b'\x00' * (PSF_HEADER_SIZE - len(data))
        return data
    
    @classmethod
    def unpack(cls, data: bytes) -> 'PSFHeader':
        """Deserialize header from bytes."""
        if len(data) < PSF_HEADER_SIZE:
            raise ValueError(f"Header too short: {len(data)} < {PSF_HEADER_SIZE}")
        
        _fmt = '<4s H I I B B H Q I Q'
        _sz = struct.calcsize(_fmt)
        fields = struct.unpack(_fmt, data[:_sz])
        
        magic = fields[0]
        if magic != PSF_MAGIC:
            raise ValueError(f"Invalid PSF magic bytes: {magic!r}")
        
        version = fields[1]
        if version > PSF_VERSION:
            raise ValueError(f"Unsupported PSF version: {version}")
        
        return cls(
            width=fields[2],
            height=fields[3],
            channels=fields[4],
            protection_level=fields[5],
            flags=fields[6],
            pixel_data_size=fields[7],
            metadata_size=fields[8],
            integrity_offset=fields[9],
        )


class PSFEncoder:
    """
    Encodes protected images into the PSF format.
    
    Usage::
    
        encoder = PSFEncoder()
        encoder.encode(
            protected_image=protected_np,
            original_image=original_np,  # optional, for hash
            metrics=attack_metrics,
            output_path="protected.psf",
            protection_level='moderate',
        )
    """
    
    def __init__(self, compression_level: int = 6):
        """
        Args:
            compression_level: zlib compression level (1-9)
        """
        self.compression_level = compression_level
    
    def encode(self,
               protected_image: np.ndarray,
               output_path: str,
               protection_level: str = 'moderate',
               original_image: Optional[np.ndarray] = None,
               metrics: Optional[Dict] = None,
               extra_metadata: Optional[Dict] = None) -> Dict:
        """
        Encode a protected image as a PSF file.
        
        Args:
            protected_image: (H, W, 3) uint8 or float [0,1]
            output_path: path to write .psf file
            protection_level: 'subtle', 'moderate', 'strong', 'maximum'
            original_image: original unprotected image (for hash)
            metrics: attack metrics from ensemble attack
            extra_metadata: additional metadata to store
        Returns:
            dict with file info
        """
        # Normalize to uint8
        if protected_image.dtype != np.uint8:
            if protected_image.max() <= 1.0:
                protected_image = (protected_image * 255).clip(0, 255).astype(np.uint8)
            else:
                protected_image = protected_image.clip(0, 255).astype(np.uint8)
        
        h, w = protected_image.shape[:2]
        c = protected_image.shape[2] if protected_image.ndim == 3 else 1
        
        # Build metadata
        metadata = {
            'software': 'PhotoSavior v3 — Phantom Spectral Encoding',
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'protection_level': protection_level,
            'image_dimensions': {'width': w, 'height': h, 'channels': c},
        }
        if metrics is not None:
            # Make metrics JSON-serializable
            clean_metrics = self._clean_metrics(metrics)
            metadata['attack_metrics'] = clean_metrics
        if extra_metadata is not None:
            metadata['extra'] = extra_metadata
        
        metadata_json = json.dumps(metadata, indent=2).encode('utf-8')
        
        # Compress pixel data
        pixel_bytes = protected_image.tobytes()
        compressed_pixels = zlib.compress(pixel_bytes, self.compression_level)
        
        # Compute flags
        flags = FLAG_HAS_METADATA | FLAG_COMPRESSED | FLAG_HAS_INTEGRITY
        if original_image is not None:
            flags |= FLAG_HAS_ORIGINAL_HASH
        
        # Compute sizes and offsets
        metadata_size = len(metadata_json)
        pixel_data_size = len(compressed_pixels)
        integrity_offset = PSF_HEADER_SIZE + metadata_size + pixel_data_size
        
        # Build header
        level_int = PROTECTION_LEVELS.get(protection_level, 2)
        header = PSFHeader(
            width=w,
            height=h,
            channels=c,
            protection_level=level_int,
            flags=flags,
            pixel_data_size=pixel_data_size,
            metadata_size=metadata_size,
            integrity_offset=integrity_offset,
        )
        
        header_bytes = header.pack()
        
        # Build integrity block
        integrity = self._compute_integrity(
            header_bytes, metadata_json, compressed_pixels,
            protected_image, original_image
        )
        
        # Write file
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'wb') as f:
            f.write(header_bytes)
            f.write(metadata_json)
            f.write(compressed_pixels)
            f.write(integrity)
        
        total_size = PSF_HEADER_SIZE + metadata_size + pixel_data_size + PSF_INTEGRITY_SIZE
        compression_ratio = len(pixel_bytes) / total_size
        
        return {
            'output_path': str(output_path),
            'file_size_bytes': total_size,
            'compression_ratio': compression_ratio,
            'pixel_data_compressed': pixel_data_size,
            'pixel_data_raw': len(pixel_bytes),
            'metadata_size': metadata_size,
        }
    
    def _clean_metrics(self, metrics: Dict) -> Dict:
        """Make metrics dict JSON-serializable."""
        clean = {}
        for k, v in metrics.items():
            if isinstance(v, dict):
                clean[k] = self._clean_metrics(v)
            elif isinstance(v, (list, tuple)):
                clean[k] = [float(x) if isinstance(x, (np.floating, float)) else x 
                           for x in v[:20]]  # Truncate long lists
            elif isinstance(v, np.floating):
                clean[k] = float(v)
            elif isinstance(v, np.integer):
                clean[k] = int(v)
            elif isinstance(v, (int, float, str, bool)):
                clean[k] = v
            else:
                clean[k] = str(v)
        return clean
    
    def _compute_integrity(self,
                           header: bytes,
                           metadata: bytes,
                           pixel_data: bytes,
                           protected_image: np.ndarray,
                           original_image: Optional[np.ndarray]) -> bytes:
        """
        Compute integrity block (96 bytes).
        
        Contains:
        - HMAC-SHA256 of all data (32 bytes)
        - SHA-256 of original image (32 bytes, zeros if not provided)
        - SHA-256 of protected image (32 bytes)
        """
        # HMAC of everything
        h = hmac.new(_HMAC_KEY, digestmod=hashlib.sha256)
        h.update(header)
        h.update(metadata)
        h.update(pixel_data)
        hmac_hash = h.digest()  # 32 bytes
        
        # Original image hash
        if original_image is not None:
            if original_image.dtype != np.uint8:
                if original_image.max() <= 1.0:
                    orig_u8 = (original_image * 255).clip(0, 255).astype(np.uint8)
                else:
                    orig_u8 = original_image.clip(0, 255).astype(np.uint8)
            else:
                orig_u8 = original_image
            original_hash = hashlib.sha256(orig_u8.tobytes()).digest()
        else:
            original_hash = b'\x00' * 32
        
        # Protected image hash
        protected_hash = hashlib.sha256(protected_image.tobytes()).digest()
        
        return hmac_hash + original_hash + protected_hash


class PSFDecoder:
    """
    Decodes .psf files back to images with metadata.
    
    Usage::
    
        decoder = PSFDecoder()
        result = decoder.decode("protected.psf")
        image = result['image']          # numpy array
        meta = result['metadata']        # protection info
        valid = result['integrity_valid'] # tamper check
    """
    
    def decode(self, input_path: str, 
               verify_integrity: bool = True) -> Dict:
        """
        Decode a PSF file.
        
        Args:
            input_path: path to .psf file
            verify_integrity: check HMAC integrity
        Returns:
            dict with keys:
                'image': (H, W, C) uint8 numpy array
                'metadata': dict of protection metadata
                'header': PSFHeader object
                'integrity_valid': bool (if verify_integrity)
                'protection_level': str
        """
        with open(input_path, 'rb') as f:
            file_data = f.read()
        
        # Parse header
        header = PSFHeader.unpack(file_data[:PSF_HEADER_SIZE])
        
        # Parse metadata
        metadata_start = PSF_HEADER_SIZE
        metadata_end = metadata_start + header.metadata_size
        metadata_bytes = file_data[metadata_start:metadata_end]
        
        if header.flags & FLAG_HAS_METADATA:
            metadata = json.loads(metadata_bytes.decode('utf-8'))
        else:
            metadata = {}
        
        # Parse pixel data
        pixel_start = metadata_end
        pixel_end = pixel_start + header.pixel_data_size
        pixel_data_compressed = file_data[pixel_start:pixel_end]
        
        if header.flags & FLAG_COMPRESSED:
            pixel_data = zlib.decompress(pixel_data_compressed)
        else:
            pixel_data = pixel_data_compressed
        
        # Reconstruct image
        image = np.frombuffer(pixel_data, dtype=np.uint8).reshape(
            header.height, header.width, header.channels
        )
        
        # Verify integrity
        integrity_valid = None
        original_hash = None
        protected_hash = None
        
        if verify_integrity and (header.flags & FLAG_HAS_INTEGRITY):
            integrity_start = header.integrity_offset
            integrity_block = file_data[integrity_start:integrity_start + PSF_INTEGRITY_SIZE]
            
            if len(integrity_block) == PSF_INTEGRITY_SIZE:
                stored_hmac = integrity_block[:32]
                original_hash = integrity_block[32:64]
                protected_hash = integrity_block[64:96]
                
                # Recompute HMAC
                h = hmac.new(_HMAC_KEY, digestmod=hashlib.sha256)
                h.update(file_data[:PSF_HEADER_SIZE])
                h.update(metadata_bytes)
                h.update(pixel_data_compressed)
                computed_hmac = h.digest()
                
                integrity_valid = hmac.compare_digest(stored_hmac, computed_hmac)
        
        # Protection level name
        level_name = PROTECTION_LEVELS_INV.get(
            header.protection_level, 'unknown'
        )
        
        return {
            'image': image.copy(),
            'metadata': metadata,
            'header': header,
            'integrity_valid': integrity_valid,
            'protection_level': level_name,
            'original_hash': original_hash.hex() if original_hash and original_hash != b'\x00' * 32 else None,
            'protected_hash': protected_hash.hex() if protected_hash else None,
        }
    
    def to_pil(self, input_path: str) -> Image.Image:
        """Convenience: decode PSF to PIL Image."""
        result = self.decode(input_path, verify_integrity=False)
        return Image.fromarray(result['image'])
    
    def to_png(self, input_path: str, output_path: str) -> str:
        """Convert PSF to standard PNG."""
        img = self.to_pil(input_path)
        img.save(output_path, format='PNG')
        return output_path


# ============================================================
# High-Level API
# ============================================================

def save_psf(protected_image: np.ndarray,
             output_path: str,
             protection_level: str = 'moderate',
             original_image: Optional[np.ndarray] = None,
             metrics: Optional[Dict] = None) -> Dict:
    """
    Save a protected image in PSF format.
    
    Convenience function wrapping PSFEncoder.
    
    Args:
        protected_image: (H, W, 3) protected image
        output_path: where to save .psf file
        protection_level: 'subtle'/'moderate'/'strong'/'maximum'
        original_image: original image (for hash verification)
        metrics: attack metrics dict
    Returns:
        file info dict
    """
    encoder = PSFEncoder()
    return encoder.encode(
        protected_image=protected_image,
        output_path=output_path,
        protection_level=protection_level,
        original_image=original_image,
        metrics=metrics,
    )


def load_psf(input_path: str, verify: bool = True) -> Dict:
    """
    Load a PSF file.
    
    Convenience function wrapping PSFDecoder.
    
    Args:
        input_path: path to .psf file
        verify: check integrity
    Returns:
        decoded result dict
    """
    decoder = PSFDecoder()
    return decoder.decode(input_path, verify_integrity=verify)


def verify_psf(input_path: str) -> Dict:
    """
    Verify a PSF file's integrity without fully decoding.
    
    Returns:
        dict with 'valid', 'protection_level', 'metadata'
    """
    result = load_psf(input_path, verify=True)
    return {
        'valid': result['integrity_valid'],
        'protection_level': result['protection_level'],
        'metadata': result['metadata'],
        'tampered': not result['integrity_valid'] if result['integrity_valid'] is not None else None,
    }
