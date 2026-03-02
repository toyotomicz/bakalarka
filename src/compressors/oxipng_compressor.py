"""
OxiPNG Lossless PNG Compressor Plugin
compressors/oxipng_compressor.py

Wraps the OxiPNG command-line tool for multi-threaded, lossless PNG optimisation.
OxiPNG is a Rust rewrite of OptiPNG that is significantly faster thanks to
parallelism, while producing comparable (often smaller) output files.

Binary location: libs/oxipng/oxipng[.exe]
OxiPNG documentation: https://github.com/shssoichiro/oxipng
"""

import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

from PIL import Image

sys.path.append(str(Path(__file__).parent.parent))
from main import CompressionLevel, CompressionMetrics, CompressorFactory, ImageCompressor
from utils.subprocess_utils import run_with_affinity
from image_size_calculator import ImageSizeCalculator


class OxiPNGCompressor(ImageCompressor):
    """
    PNG lossless compressor backed by the OxiPNG CLI tool.

    Unlike OptiPNG, OxiPNG supports an explicit --out flag so the input file
    is never modified; Pillow first converts the source to a temporary PNG,
    then OxiPNG writes the optimised result to output_path.
    Compression levels map to OxiPNG's -oN flag (1 = fastest, 6 = best).
    """

    def __init__(self, lib_path: Optional[Path] = None):
        # Set to the directory containing the binary; filled by _validate_dependencies().
        self._bin_dir: Optional[Path] = None
        super().__init__(lib_path)

    def _validate_dependencies(self) -> None:
        """Locate the OxiPNG binary inside libs/oxipng/."""
        base_dir    = Path(__file__).parent.parent
        bin_dir     = base_dir / "libs" / "oxipng"
        binary_name = _binary_name("oxipng")
        binary_path = bin_dir / binary_name

        if not bin_dir.exists():
            raise RuntimeError(f"OxiPNG directory not found: {bin_dir}")
        if not binary_path.exists():
            raise RuntimeError(f"OxiPNG binary not found: {binary_path}")

        self._bin_dir = bin_dir

    # -- ImageCompressor interface --

    @property
    def name(self) -> str:
        return "OxiPNG"

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
        Compress an image to an optimised PNG file using OxiPNG.

        Pillow first saves the image as a standard PNG (handling format
        conversion if needed), then OxiPNG re-optimises the deflate stream
        and writes the result to output_path via --out.
        """
        try:
            original_size = ImageSizeCalculator.calculate_uncompressed_size(input_path)

            # Pillow converts the source to a well-formed PNG that OxiPNG can process.
            img = Image.open(input_path)
            if img.mode not in ("RGB", "RGBA", "L", "LA"):
                img = img.convert("RGB")

            # Write a temporary intermediate PNG; OxiPNG reads this and
            # writes the optimised result to output_path.
            temp_input = output_path.parent / f"temp_input_{output_path.stem}.png"
            try:
                img.save(temp_input, format="PNG")

                start_time = time.perf_counter()
                self._run_oxipng(temp_input, output_path, level)
                compression_time = time.perf_counter() - start_time
            finally:
                if temp_input.exists():
                    temp_input.unlink()

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

    def _run_oxipng(
        self,
        input_path: Path,
        output_path: Path,
        level: CompressionLevel,
    ) -> None:
        """
        Invoke OxiPNG to produce an optimised PNG at output_path.

        Args:
            input_path:  Source PNG file (not modified).
            output_path: Destination for the optimised PNG.
            level:       Compression level controlling the -oN flag.

        Raises:
            RuntimeError: If OxiPNG exits with a non-zero return code.
        """
        # OxiPNG -oN: 1 = fastest / least work, 6 = most trials / smallest output.
        level_map = {
            CompressionLevel.FASTEST:  1,
            CompressionLevel.BALANCED: 3,
            CompressionLevel.BEST:     6,
        }
        o_level = level_map.get(level, 3)

        binary = self._bin_dir / _binary_name("oxipng")
        cmd = [
            str(binary),
            "-o", str(o_level),           # optimisation level (1–6)
            "--strip", "all",             # strip all metadata chunks
            "-q",                         # quiet: suppress progress output
            "--out", str(output_path),    # explicit output path (input unchanged)
            str(input_path),              # source file
        ]

        result = run_with_affinity(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"OxiPNG failed (exit {result.returncode}): {result.stderr}"
            )

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


# Register so CompressorFactory.create("oxipng") works.
CompressorFactory.register("oxipng", OxiPNGCompressor)