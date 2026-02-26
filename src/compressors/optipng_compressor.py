"""
OptiPNG Lossless PNG Compressor Plugin
compressors/optipng_compressor.py

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
from image_size_calculator import ImageSizeCalculator


class OptiPNGCompressor(ImageCompressor):
    """
    PNG lossless compressor backed by the OptiPNG CLI tool.

    OptiPNG optimises an existing PNG file in-place, so the input is first
    copied to the output path and then the binary is invoked on that copy.
    Compression levels map to OptiPNG's -oN flag (0 = fastest, 7 = best).
    """

    def __init__(self, lib_path: Optional[Path] = None):
        # Set to the directory containing the binary; filled by _validate_dependencies().
        self._bin_dir: Optional[Path] = None
        super().__init__(lib_path)

    def _validate_dependencies(self) -> None:
        """Locate the OptiPNG binary inside libs/png/."""
        base_dir    = Path(__file__).parent.parent
        bin_dir     = base_dir / "libs" / "png"
        binary_name = _binary_name("optipng")
        binary_path = bin_dir / binary_name

        if not bin_dir.exists():
            raise RuntimeError(f"OptiPNG directory not found: {bin_dir}")
        if not binary_path.exists():
            raise RuntimeError(f"OptiPNG binary not found: {binary_path}")

        self._bin_dir = bin_dir

    # -- ImageCompressor interface --

    @property
    def name(self) -> str:
        return "OptiPNG"

    @property
    def extension(self) -> str:
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
        is first re-saved as PNG by Pillow (preserving all pixel data), then
        OptiPNG re-optimises the deflate stream in-place.
        """
        try:
            original_size = ImageSizeCalculator.calculate_uncompressed_size(input_path)

            # Ensure output_path contains a valid PNG before calling OptiPNG.
            # Pillow handles conversion from any supported source format.
            img = Image.open(input_path)
            if img.mode not in ("RGB", "RGBA", "L", "LA"):
                img = img.convert("RGB")
            img.save(output_path, format="PNG")

            start_time = time.perf_counter()
            self._run_optipng(output_path, level)
            compression_time = time.perf_counter() - start_time

            compressed_size = output_path.stat().st_size

            # Measure decompression time via a temporary decode.
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
        Invoke OptiPNG on *target_path* (optimises the file in-place).

        Args:
            target_path: PNG file to optimise; modified in-place.
            level:       Compression level controlling the -oN flag.

        Raises:
            RuntimeError: If OptiPNG exits with a non-zero return code.
        """
        # OptiPNG -oN: 0 = fastest / least work, 7 = most trials / smallest output.
        level_map = {
            CompressionLevel.FASTEST:  0,
            CompressionLevel.BALANCED: 4,
            CompressionLevel.BEST:     7,
        }
        o_level = level_map.get(level, 4)

        binary = self._bin_dir / _binary_name("optipng")
        cmd = [
            str(binary),
            f"-o{o_level}",   # optimisation level (0–7)
            "-strip", "all",  # strip all metadata chunks for smaller files
            "-quiet",         # suppress progress output
            str(target_path), # file to optimise in-place
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"OptiPNG failed (exit {result.returncode}): {result.stderr}")

    def decompress(self, input_path: Path, output_path: Path) -> float:
        """
        Decode a PNG file and save it as PNG, measuring wall-clock time.

        img.load() forces the full pixel decode (not just header parsing),
        giving a realistic decompression benchmark.
        """
        start_time = time.perf_counter()

        img = Image.open(input_path)
        img.load()          # Force full pixel decode into memory
        img.save(output_path, format="PNG")

        return time.perf_counter() - start_time


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _binary_name(base: str) -> str:
    """Return the Windows binary name (always appends .exe)."""
    return f"{base}.exe"


# Register so CompressorFactory.create("optipng") works.
CompressorFactory.register("optipng", OptiPNGCompressor)