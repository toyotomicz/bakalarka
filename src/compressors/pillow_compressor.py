"""
Pillow-Based Universal image compressor plugin.

Pure-Python compressors for PNG, WebP, and TIFF using the Pillow (PIL) library.
These serve as portable fallbacks when native CLI tools are unavailable, and as
reference implementations for benchmarking.

Registered compressor keys:
"pillow-png"  → PillowPNGCompressor   (lossless, .png)
"pillow-webp" → PillowWebPCompressor  (lossless, .webp)
"pillow-tiff" → PillowTIFFCompressor  (lossless, .tiff)
"""

import sys
import time
from pathlib import Path
from typing import Dict, Optional

from PIL import Image

sys.path.append(str(Path(__file__).parent.parent))
from main import CompressionLevel, CompressionMetrics, CompressorFactory, ImageCompressor
from utils.image_size_calculator import ImageSizeCalculator


# Base class

class PillowCompressorBase(ImageCompressor):
    """
    Shared implementation for all Pillow-based compressors.

    Subclasses must set the two class-level attributes and override
    _get_compression_params() to provide format-specific save kwargs.
    They may also override _prepare_image() when the format has restrictions
    on colour modes (e.g. WebP lossless requires RGB or RGBA).

    Class attributes:
        _format_name: Pillow format string used in Image.save(), e.g. "PNG".
            Must be overridden by every subclass before calling super().__init__().
        _file_extension: Output file extension including the dot, e.g. ".png".
            Must be overridden by every subclass before calling super().__init__().
    """

    _format_name:    str = ""
    _file_extension: str = ""

    def __init__(self, lib_path: Optional[Path] = None):
        super().__init__(lib_path)

    def _validate_dependencies(self) -> None:
        """
        Verify that Pillow is installed and supports saving the target format.

        Raises:
            RuntimeError: If Pillow is not installed or does not support the
                format specified by _format_name.
        """
        try:
            Image.init()
            if self._format_name and self._format_name not in Image.SAVE:
                raise RuntimeError(
                    f"Pillow cannot save '{self._format_name}' format. "
                    f"Try: pip install pillow"
                )
        except ImportError as exc:
            raise RuntimeError(f"Pillow is not installed: {exc}") from exc

    def _get_compression_params(self, level: CompressionLevel) -> Dict:
        """
        Return format-specific keyword arguments for Image.save().

        Override in subclasses to provide concrete compression settings.

        Args:
            level: Compression effort level.

        Returns:
            Dict of kwargs forwarded to Image.save().
        """
        return {}

    def _prepare_image(self, img: Image.Image) -> Image.Image:
        """
        Convert the image colour mode if the target format requires it.

        The default implementation converts exotic modes (palette, CMYK, ...)
        to RGB while preserving RGB and RGBA unchanged.  Override in subclasses
        with stricter format requirements.

        Args:
            img: Source Pillow Image.

        Returns:
            Pillow Image in a mode accepted by the target format.
        """
        if img.mode not in ("RGB", "RGBA", "L", "LA"):
            img = img.convert("RGB")
        return img

    # ImageCompressor interface

    @property
    def name(self) -> str:
        """Human-readable compressor name shown in benchmark reports."""
        return f"Pillow-{self._format_name}"

    @property
    def extension(self) -> str:
        """Output file extension including the leading dot."""
        return self._file_extension

    def compress(
        self,
        input_path: Path,
        output_path: Path,
        level: CompressionLevel = CompressionLevel.BALANCED,
    ) -> CompressionMetrics:
        """
        Compress an image using Pillow and measure compression / decompression time.

        The uncompressed size is calculated from raw pixel data so that all
        compressors report a comparable baseline regardless of source format.

        Args:
            input_path: Source image file (any Pillow-readable format).
            output_path: Destination compressed file.
            level: Compression effort level.

        Returns:
            CompressionMetrics with timing and size data.
        """
        try:
            original_size = ImageSizeCalculator.calculate_uncompressed_size(input_path)

            img = Image.open(input_path)
            img.load()  # Force full decode before .info can be dropped
            img = self._prepare_image(img)


            save_params = self._get_compression_params(level)

            # Guarantee the correct extension regardless of what the caller passed.
            if output_path.suffix.lower() != self._file_extension.lower():
                output_path = output_path.with_suffix(self._file_extension)

            start_time = time.perf_counter()
            img.save(output_path, format=self._format_name, **save_params)
            compression_time = time.perf_counter() - start_time

            compressed_size = output_path.stat().st_size

            # Measure decompression time via a temporary decode round-trip.
            temp_decomp = output_path.parent / f"temp_decomp_{output_path.stem}.png"
            try:
                decompression_time = self.decompress(output_path, temp_decomp)
            finally:
                if temp_decomp.exists():
                    temp_decomp.unlink()

            return CompressionMetrics(
                original_size=original_size,
                compressed_size=compressed_size,
                compression_ratio=(
                    original_size / compressed_size if compressed_size > 0 else 0
                ),
                compression_time=compression_time,
                decompression_time=decompression_time,
                success=True,
            )

        except Exception as exc:
            return CompressionMetrics(
                original_size=0,
                compressed_size=0,
                compression_ratio=0,
                compression_time=0,
                decompression_time=0,
                success=False,
                error_message=str(exc),
            )

    def decompress(self, input_path: Path, output_path: Path) -> float:
        """
        Decode the compressed file and re-save as PNG, measuring wall-clock time.

        img.load() forces the full pixel decode (not just header parsing),
        giving a realistic decompression benchmark.

        Args:
            input_path: Compressed source file.
            output_path: Destination PNG file.

        Returns:
            Wall-clock decompression time in seconds.
        """
        start_time = time.perf_counter()

        img = Image.open(input_path)
        img.load()          # Force full pixel decode into memory
        img.save(output_path, format="PNG")

        return time.perf_counter() - start_time


# PNG compressor

class PillowPNGCompressor(PillowCompressorBase):
    """
    PNG lossless compressor using Pillow's built-in deflate encoder.

    compress_level maps to zlib's 0–9 scale (0 = store, 9 = best deflate).
    The 'optimize' flag makes Pillow try multiple filter heuristics when
    BEST compression is requested, at the cost of extra encoding time.
    """

    def __init__(self, lib_path: Optional[Path] = None):
        # Class-level attributes must be set before super().__init__() so that
        # _validate_dependencies() can check Image.SAVE for the correct key.
        self._format_name    = "PNG"
        self._file_extension = ".png"
        super().__init__(lib_path)

    def _get_compression_params(self, level: CompressionLevel) -> Dict:
        """
        Return PNG-specific save parameters for the given level.

        Args:
            level: Compression effort level.

        Returns:
            Dict with 'compress_level' (int) and 'optimize' (bool).
        """
        level_map = {
            CompressionLevel.FASTEST:  {"compress_level": 1, "optimize": False},
            CompressionLevel.BALANCED: {"compress_level": 6, "optimize": False},
            CompressionLevel.BEST:     {"compress_level": 9, "optimize": True},
        }
        return level_map.get(level, level_map[CompressionLevel.BALANCED])


# WebP compressor

class PillowWebPCompressor(PillowCompressorBase):
    """
    WebP lossless compressor using Pillow's libwebp encoder.

    In lossless mode, 'quality' controls the compression effort (not lossy
    quality): higher values produce smaller files at the cost of CPU time.
    'method' (0–6) controls the encoding algorithm effort level.
    """

    def __init__(self, lib_path: Optional[Path] = None):
        self._format_name    = "WEBP"
        self._file_extension = ".webp"
        super().__init__(lib_path)

    def _get_compression_params(self, level: CompressionLevel) -> Dict:
        """Return WebP lossless save parameters for the given level.

        Args:
            level: Compression effort level.

        Returns:
            Dict with 'lossless', 'quality', and 'method' keys.
        """
        level_map = {
            CompressionLevel.FASTEST:  {"quality": 0,   "method": 0},
            CompressionLevel.BALANCED: {"quality": 50,  "method": 4},
            CompressionLevel.BEST:     {"quality": 100, "method": 6},
        }
        params = level_map.get(level, level_map[CompressionLevel.BALANCED])
        return {
            "lossless": True,           # Ensure lossless mode regardless of quality value
            "quality":  params["quality"],
            "method":   params["method"],
        }

    def _prepare_image(self, img: Image.Image) -> Image.Image:
        """
        Prepare the image for WebP encoding.

        WebP lossless natively supports RGBA, so alpha is preserved.
        Only non-standard modes are converted.

        Args:
            img: Source Pillow Image.

        Returns:
            Pillow Image in RGB or RGBA mode.
        """
        if img.mode not in ("RGB", "RGBA", "L", "LA"):
            img = img.convert("RGB")
        return img


# TIFF compressor

class PillowTIFFCompressor(PillowCompressorBase):
    """
    TIFF lossless compressor using Pillow's libtiff encoder.

    TIFF supports multiple lossless compression schemes:
    - packbits    : Simple run-length encoding; very fast, moderate ratio.
    - lzw         : LZW dictionary coding; good ratio, moderate speed.
    - tiff_deflate: Deflate (same algorithm as PNG); best ratio, slowest.
    """

    def __init__(self, lib_path: Optional[Path] = None):
        self._format_name    = "TIFF"
        self._file_extension = ".tiff"
        super().__init__(lib_path)

    def _get_compression_params(self, level: CompressionLevel) -> Dict:
        """
        Return TIFF-specific save parameters for the given level.

        Args:
            level: Compression effort level.

        Returns:
            Dict with a 'compression' key selecting the TIFF codec.
        """
        level_map = {
            CompressionLevel.FASTEST:  "packbits",
            CompressionLevel.BALANCED: "lzw",
            CompressionLevel.BEST:     "tiff_deflate",
        }
        return {"compression": level_map.get(level, "lzw")}


# Registration

CompressorFactory.register("pillow-png",  PillowPNGCompressor)
CompressorFactory.register("pillow-webp", PillowWebPCompressor)
CompressorFactory.register("pillow-tiff", PillowTIFFCompressor)