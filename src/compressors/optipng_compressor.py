"""
OptiPNG Lossless PNG compressor plugin.

Wraps the OptiPNG command-line tool to optimise PNG files losslessly.
OptiPNG recompresses the deflate stream and tries multiple filter strategies
to find the smallest representation without altering pixel data.

Binary location: libs/png/optipng[.exe]
OptiPNG documentation: https://optipng.sourceforge.net/
"""

import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

from PIL import Image

sys.path.append(str(Path(__file__).parent.parent))
from main import CompressionLevel, CompressionMetrics, CompressorFactory, ImageCompressor
from utils.subprocess_utils import run_with_affinity
from utils.image_size_calculator import ImageSizeCalculator


class OptiPNGCompressor(ImageCompressor):
    """
    PNG lossless compressor backed by the OptiPNG CLI tool.

    OptiPNG optimises an existing PNG file in-place, so the source image is
    first saved as PNG by Pillow (handling format conversion if needed), then
    the binary is invoked on that copy.

    Compression levels map to OptiPNG's -oN flag (0 = fastest, 7 = best).

    Attributes:
        _bin_dir: Directory containing the optipng binary; set by
            _validate_dependencies().
    """

    def __init__(self, lib_path: Optional[Path] = None):
        # Must be set before super().__init__() calls _validate_dependencies().
        self._bin_dir: Optional[Path] = None
        super().__init__(lib_path)

    def _validate_dependencies(self) -> None:
        """
        Locate the OptiPNG binary inside libs/png/.

        Raises:
            RuntimeError: If the directory or binary is not found.
        """
        base_dir    = Path(__file__).parent.parent
        bin_dir     = base_dir / "libs" / "png"
        binary_name = _binary_name("optipng")
        binary_path = bin_dir / binary_name

        if not bin_dir.exists():
            raise RuntimeError(f"OptiPNG directory not found: {bin_dir}")
        if not binary_path.exists():
            raise RuntimeError(f"OptiPNG binary not found: {binary_path}")

        self._bin_dir = bin_dir

    # ImageCompressor interface

    @property
    def name(self) -> str:
        """Human-readable compressor name shown in benchmark reports."""
        return "OptiPNG"

    @property
    def extension(self) -> str:
        """Output file extension including the leading dot."""
        return ".png"

    def compress(
        self,
        input_path: Path,
        output_path: Path,
        level: CompressionLevel = CompressionLevel.BALANCED,
    ) -> CompressionMetrics:
        """
        Compress an image to an optimised PNG file using OptiPNG.

        Because OptiPNG only accepts an existing PNG as input, the source image
        is first re-saved as PNG by Pillow, then OptiPNG re-optimises the
        deflate stream in-place on the same file.

        Args:
            input_path: Source image file (any Pillow-readable format).
            output_path: Destination PNG file (written by Pillow, then optimised
                in-place by OptiPNG).
            level: Compression level controlling the -oN flag.

        Returns:
            CompressionMetrics with timing and size data.
        """
        try:
            original_size = ImageSizeCalculator.calculate_uncompressed_size(input_path)

            # The runner already prepared a normalised PNG (mode-safe, metadata
            # state correct) via _prepare_input().  Just copy it to output_path
            # so OptiPNG can optimise it in-place.
            img = Image.open(input_path)
            img.load()
            img.save(output_path, format="PNG")

            # Time only the OptiPNG pass, not the Pillow write above.
            start_time = time.perf_counter()
            self._run_optipng(output_path, level)
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
                compression_ratio=original_size / compressed_size,
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

    def _run_optipng(self, target_path: Path, level: CompressionLevel) -> None:
        """
        Invoke OptiPNG on target_path, which is optimised in-place.

        Args:
            target_path: PNG file to optimise; modified in-place by OptiPNG.
            level: Compression level controlling the -oN flag.

        Raises:
            RuntimeError: If OptiPNG exits with a non-zero return code.
        """
        # OptiPNG -oN: 0 = fastest / least trials, 7 = most trials / smallest output.
        level_map = {
            CompressionLevel.FASTEST:  0,
            CompressionLevel.BALANCED: 4,
            CompressionLevel.BEST:     7,
        }
        o_level = level_map.get(level, 4)

        binary = self._bin_dir / _binary_name("optipng")
        cmd = [
            str(binary),
            f"-o{o_level}",  # optimisation level (0–7)
            "-quiet",        # suppress progress output to stderr
            # Note: -strip all is intentionally omitted; metadata state is
            # controlled by BenchmarkConfig.strip_metadata in the runner.
            str(target_path), # PNG file to optimise in-place
        ]

        result = run_with_affinity(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"OptiPNG failed (exit {result.returncode}): {result.stderr}")

    def decompress(self, input_path: Path, output_path: Path) -> float:
        """
        Decode a PNG file and save it as PNG, measuring wall-clock time.

        img.load() forces the full pixel decode (not just header parsing),
        giving a realistic decompression benchmark.

        Args:
            input_path: Compressed PNG source file.
            output_path: Destination PNG file.

        Returns:
            Wall-clock decompression time in seconds.
        """
        start_time = time.perf_counter()

        img = Image.open(input_path)
        img.load()          # Force full pixel decode into memory
        img.save(output_path, format="PNG")

        return time.perf_counter() - start_time


# Helpers

def _binary_name(base: str) -> str:
    """Return the platform-specific binary filename.

    Args:
        base: Binary base name without extension, e.g. 'optipng'.

    Returns:
        Filename string with .exe suffix.
    """
    return f"{base}.exe"


# Register so CompressorFactory.create("optipng") works.
CompressorFactory.register("optipng", OptiPNGCompressor)