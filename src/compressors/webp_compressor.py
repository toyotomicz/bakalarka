"""
WebP Lossless Compressor Plugin
compressors/webp_compressor.py

Wraps the official WebP command-line tools (cwebp / dwebp) for lossless
image compression. Using the native binaries gives access to encoding options
that are not exposed through Pillow's WebP backend.

Binary location: libs/webp/cwebp[.exe] and dwebp[.exe]
WebP documentation: https://developers.google.com/speed/webp/docs/cwebp
"""

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

sys.path.append(str(Path(__file__).parent.parent))
from main import CompressionLevel, CompressionMetrics, CompressorFactory, ImageCompressor
from utils.subprocess_utils import run_with_affinity
from image_size_calculator import ImageSizeCalculator


class WebPCompressor(ImageCompressor):
    """
    WebP lossless compressor backed by the cwebp / dwebp CLI tools.

    Compression is always performed in lossless mode (-lossless).
    The three CompressionLevel values map to combinations of cwebp's
    -z (preset level), -q (quality/effort), and -m (method) flags.
    """

    def __init__(self, lib_path: Optional[Path] = None):
        # Set to the directory containing the binaries; filled by _validate_dependencies().
        self._bin_dir: Optional[Path] = None
        super().__init__(lib_path)

    def _validate_dependencies(self) -> None:
        """Locate cwebp and dwebp binaries inside libs/webp/."""
        base_dir = Path(__file__).parent.parent
        bin_dir  = base_dir / "libs" / "webp"

        cwebp_path = bin_dir / _binary_name("cwebp")
        dwebp_path = bin_dir / _binary_name("dwebp")

        if not bin_dir.exists():
            raise RuntimeError(f"WebP tools directory not found: {bin_dir}")
        if not cwebp_path.exists():
            raise RuntimeError(f"cwebp binary not found: {cwebp_path}")
        if not dwebp_path.exists():
            raise RuntimeError(f"dwebp binary not found: {dwebp_path}")

        self._bin_dir = bin_dir

    # -- ImageCompressor interface --

    @property
    def name(self) -> str:
        return "WebP-Lossless"

    @property
    def extension(self) -> str:
        return ".webp"

    def compress(
        self,
        input_path: Path,
        output_path: Path,
        level: CompressionLevel = CompressionLevel.BALANCED,
    ) -> CompressionMetrics:
        """
        Compress an image to lossless WebP format using cwebp.

        cwebp can read PNG, BMP, TIFF, and raw YUV directly; Pillow is not
        needed for the encoding step. Decompression is performed by dwebp.
        """
        try:
            original_size = ImageSizeCalculator.calculate_uncompressed_size(input_path)

            start_time = time.perf_counter()
            self._run_cwebp(input_path, output_path, level)
            compression_time = time.perf_counter() - start_time

            compressed_size = output_path.stat().st_size

            # Measure decompression time; dwebp writes a PNG to temp_decomp.
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

    def _run_cwebp(
        self,
        input_path: Path,
        output_path: Path,
        level: CompressionLevel,
    ) -> None:
        """
        Invoke cwebp to produce a lossless WebP file.

        Flag meanings:
          -lossless : Enable lossless compression mode.
          -exact    : Preserve RGB values in fully-transparent pixels.
          -q N      : In lossless mode, controls compression effort (0–100);
                      higher = smaller output but slower encoding.
          -z N      : Lossless compression preset (0–9); overrides some -q/-m values.
          -m N      : Encoding method (0–6); higher = more CPU, better ratio.
          -alpha_q  : Quality for the alpha channel (100 = lossless alpha).

        Args:
            input_path:  Source image file (PNG / BMP / TIFF / …).
            output_path: Destination .webp file.
            level:       Compression level controlling effort flags.

        Raises:
            RuntimeError: If cwebp exits with a non-zero return code.
        """
        # Lossless WebP preset levels; higher = more compression effort.
        level_map = {
            CompressionLevel.FASTEST:  {"z": 0, "q": 75,  "m": 4},
            CompressionLevel.BALANCED: {"z": 6, "q": 100, "m": 6},
            CompressionLevel.BEST:     {"z": 9, "q": 100, "m": 6},
        }
        params = level_map.get(level, level_map[CompressionLevel.BALANCED])

        binary = self._bin_dir / _binary_name("cwebp")
        cmd = [
            str(binary),
            "-lossless",                  # lossless compression mode
            "-exact",                     # preserve RGB in transparent pixels
            "-q",      str(params["q"]),  # compression effort (0–100)
            "-z",      str(params["z"]),  # preset level (0–9)
            "-m",      str(params["m"]),  # encoding method (0–6)
            "-alpha_q", "100",            # lossless alpha channel
            "-metadata", "none",          # strip all metadata (EXIF, ICC, XMP)
            str(input_path),
            "-o", str(output_path),
        ]

        result = run_with_affinity(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"cwebp failed (exit {result.returncode}): {result.stderr}"
            )

        # Flush the output file to disk so the OS write-back cache does not hide
        # the actual I/O from the system monitor's io_counters() measurements.
        # Without this, repeated runs on the same output path can show 0 write bytes
        # because Windows defers the physical write.
        try:
            with open(output_path, "r+b") as fh:
                fh.flush()
                os.fsync(fh.fileno())
        except OSError:
            pass  # Non-fatal, metrics may be under-reported but compression succeeded.

    def decompress(self, input_path: Path, output_path: Path) -> float:
        """
        Decode a WebP file to PNG using dwebp, measuring wall-clock time.

        dwebp is used for decompression (rather than Pillow) to keep the
        benchmark symmetric with the native cwebp encoder.
        """
        start_time = time.perf_counter()

        binary = self._bin_dir / _binary_name("dwebp")
        cmd    = [str(binary), str(input_path), "-o", str(output_path)]

        result = run_with_affinity(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"dwebp failed (exit {result.returncode}): {result.stderr}"
            )

        return time.perf_counter() - start_time


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _binary_name(base: str) -> str:
    """Return the Windows binary name (always appends .exe)."""
    return f"{base}.exe"


# Register so CompressorFactory.create("webp") works.
CompressorFactory.register("webp", WebPCompressor)