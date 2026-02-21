"""
Pillow-based Universal Image Compressor Plugin
compressors/pillow_compressor.py

Supports: PNG, WebP, TIFF, AVIF
Uses pure Python Pillow library for maximum compatibility.
"""

from pathlib import Path
from typing import Optional, Dict
import time
import sys

from PIL import Image

sys.path.append(str(Path(__file__).parent.parent))
from main import ImageCompressor, CompressionMetrics, CompressionLevel, CompressorFactory
from image_size_calculator import ImageSizeCalculator


class PillowCompressorBase(ImageCompressor):
    """
    Base class for Pillow-based compressors.
    Provides common functionality for all Pillow formats.
    """
    
    # Class-level defaults (will be overridden by subclasses)
    _format_name = None
    _file_extension = None
    
    def __init__(self, lib_path: Optional[Path] = None):
        # Don't set instance variables here - subclasses should set them
        # before calling super().__init__()
        super().__init__(lib_path)
    
    def _validate_dependencies(self) -> None:
        """Check if Pillow and required plugins are available"""
        try:
            import PIL
            # Check specific format support
            if self._format_name:
                Image.init()
                if self._format_name not in Image.SAVE:
                    raise RuntimeError(
                        f"Pillow does not support saving {self._format_name} format. "
                        f"Install required plugin: pip install pillow-{self._format_name.lower()}-plugin"
                    )
        except ImportError as e:
            raise RuntimeError(f"Pillow is not installed: {e}")
    
    def _get_compression_params(self, level: CompressionLevel) -> Dict:
        """
        Get format-specific compression parameters.
        Override in subclasses for format-specific settings.
        """
        return {}
    
    def compress(self, 
                 input_path: Path, 
                 output_path: Path,
                 level: CompressionLevel = CompressionLevel.BALANCED) -> CompressionMetrics:
        """Compress image using Pillow"""
        
        try:
            original_size = ImageSizeCalculator.calculate_uncompressed_size(input_path)
            
            # Load image
            img = Image.open(input_path)
            
            # Convert mode if necessary
            img = self._prepare_image(img)
            
            start_time = time.perf_counter()
            
            # Get compression parameters
            save_params = self._get_compression_params(level)
            
            # Ensure output path has correct extension
            if not str(output_path).lower().endswith(self._file_extension.lower()):
                output_path = output_path.with_suffix(self._file_extension)
            
            # Save compressed image
            img.save(output_path, format=self._format_name, **save_params)
            
            compression_time = time.perf_counter() - start_time
            compressed_size = output_path.stat().st_size
            
            # Test decompression
            temp_decomp = output_path.parent / f"temp_decomp_{output_path.stem}.png"
            try:
                decompression_time = self.decompress(output_path, temp_decomp)
            finally:
                if temp_decomp.exists():
                    temp_decomp.unlink()
            
            return CompressionMetrics(
                original_size=original_size,
                compressed_size=compressed_size,
                compression_ratio=original_size / compressed_size if compressed_size > 0 else 0,
                compression_time=compression_time,
                decompression_time=decompression_time,
                success=True
            )
            
        except Exception as e:
            return CompressionMetrics(
                original_size=0,
                compressed_size=0,
                compression_ratio=0,
                compression_time=0,
                decompression_time=0,
                success=False,
                error_message=str(e)
            )
    
    def decompress(self, input_path: Path, output_path: Path) -> float:
        """Decompress image by loading and re-saving"""
        start_time = time.perf_counter()
        
        img = Image.open(input_path)
        img.load()  # ✅ Force loading pixels into memory
        img.save(output_path, format='PNG')
        
        return time.perf_counter() - start_time
    
    def _prepare_image(self, img: Image.Image) -> Image.Image:
        """
        Prepare image for compression (convert mode if needed).
        Override in subclasses for format-specific requirements.
        """
        return img
    
    @property
    def name(self) -> str:
        return f"Pillow-{self._format_name}"
    
    @property
    def extension(self) -> str:
        return self._file_extension


# ============================================================================
# PNG COMPRESSOR
# ============================================================================

class PillowPNGCompressor(PillowCompressorBase):
    """PNG lossless compressor using Pillow"""
    
    def __init__(self, lib_path: Optional[Path] = None):
        # Set format info BEFORE calling super().__init__()
        self._format_name = "PNG"
        self._file_extension = ".png"
        super().__init__(lib_path)
    
    def _get_compression_params(self, level: CompressionLevel) -> Dict:
        """PNG compression parameters"""
        # Pillow's PNG compress_level: 0 (no compression) to 9 (best compression)
        level_map = {
            CompressionLevel.FASTEST: 1,
            CompressionLevel.BALANCED: 6,
            CompressionLevel.BEST: 9,
        }
        
        compress_level = level_map.get(level, 6)
        
        return {
            'compress_level': compress_level,
            'optimize': level in [CompressionLevel.BEST]
        }


# ============================================================================
# WEBP COMPRESSOR
# ============================================================================

class PillowWebPCompressor(PillowCompressorBase):
    """WebP lossless compressor using Pillow"""
    
    def __init__(self, lib_path: Optional[Path] = None):
        # Set format info BEFORE calling super().__init__()
        self._format_name = "WEBP"
        self._file_extension = ".webp"
        super().__init__(lib_path)
    
    def _get_compression_params(self, level: CompressionLevel) -> Dict:
        """WebP lossless compression parameters"""
        # quality: 0-100 for lossless (higher = better but slower)
        # method: 0-6 (0=fast, 6=slowest but best compression)
        level_map = {
            CompressionLevel.FASTEST: {'quality': 75, 'method': 0},
            CompressionLevel.BALANCED: {'quality': 90, 'method': 4},
            CompressionLevel.BEST: {'quality': 100, 'method': 6},
        }
        
        params = level_map.get(level, level_map[CompressionLevel.BALANCED])
        
        return {
            'lossless': True,
            'quality': params['quality'],
            'method': params['method']
        }
    
    def _prepare_image(self, img: Image.Image) -> Image.Image:
        """Keep RGBA intact - lossless WebP supports alpha natively"""
        return img


# ============================================================================
# TIFF COMPRESSOR
# ============================================================================

class PillowTIFFCompressor(PillowCompressorBase):
    """TIFF lossless compressor using Pillow"""
    
    def __init__(self, lib_path: Optional[Path] = None):
        # Set format info BEFORE calling super().__init__()
        self._format_name = "TIFF"
        self._file_extension = ".tiff"
        super().__init__(lib_path)
    
    def _get_compression_params(self, level: CompressionLevel) -> Dict:
        """TIFF compression parameters"""
        # TIFF supports various compression algorithms
        # lzw: good balance
        # tiff_deflate: better compression (similar to PNG)
        # packbits: faster but less compression
        
        level_map = {
            CompressionLevel.FASTEST: 'packbits',
            CompressionLevel.BALANCED: 'lzw',
            CompressionLevel.BEST: 'tiff_deflate',
        }
        
        compression = level_map.get(level, 'tiff_deflate')
        
        return {
            'compression': compression
        }


# ============================================================================
# REGISTRATION
# ============================================================================

# Register all Pillow-based compressors
CompressorFactory.register("pillow-png", PillowPNGCompressor)
CompressorFactory.register("pillow-webp", PillowWebPCompressor)
CompressorFactory.register("pillow-tiff", PillowTIFFCompressor)