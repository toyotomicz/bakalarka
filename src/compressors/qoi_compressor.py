"""
QOI (Quite OK Image) Lossless compressor plugin.

QOI is a simple, fast lossless image format designed for real-time use cases.
It encodes each pixel as a small number of bytes using a short list of
run-length, delta, and index-lookup operations - no compression-level dial exists.

Dependency: pip install qoi
QOI specification: https://qoiformat.org/
"""

import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

sys.path.append(str(Path(__file__).parent.parent))
from main import CompressionLevel, CompressionMetrics, CompressorFactory, ImageCompressor
from utils.image_size_calculator import ImageSizeCalculator


class QOICompressor(ImageCompressor):
    """
    QOI (Quite OK Image) lossless compressor.

    QOI has a fixed encoding algorithm with no compression-level knob.
    The CompressionLevel argument is accepted for interface compatibility
    but has no effect on the output.

    Supports RGB and RGBA images; other modes are converted to RGB first.

    Attributes:
        _qoi: Imported qoi module handle, or None before initialisation.
    """

    def __init__(self, lib_path: Optional[Path] = None):
        # Must be set before super().__init__() calls _validate_dependencies().
        self._qoi = None
        super().__init__(lib_path)

    def _validate_dependencies(self) -> None:
        """
        Check that the 'qoi' Python package is available.

        Raises:
            RuntimeError: If the qoi package is not installed.
        """
        try:
            import qoi as qoi_module
            self._qoi = qoi_module
        except ImportError as exc:
            raise RuntimeError(
                "The 'qoi' package is not installed. "
                "Install it with: pip install qoi"
            ) from exc

    # ImageCompressor interface

    @property
    def name(self) -> str:
        """Human-readable compressor name shown in benchmark reports."""
        return "QOI"

    @property
    def extension(self) -> str:
        """Output file extension including the leading dot."""
        return ".qoi"

    def compress(
        self,
        input_path: Path,
        output_path: Path,
        level: CompressionLevel = CompressionLevel.BALANCED,
    ) -> CompressionMetrics:
        """
        Encode an image to QOI format.

        Note:
            QOI has no compression levels.  The ``level`` parameter is ignored
            but kept for interface compatibility with other compressors.

        Args:
            input_path: Source image file (any Pillow-readable format).
            output_path: Destination .qoi file.
            level: Accepted for interface compatibility; has no effect.

        Returns:
            CompressionMetrics with timing and size data.
        """
        try:
            original_size = ImageSizeCalculator.calculate_uncompressed_size(input_path)

            # QOI only accepts RGB or RGBA pixel data.
            img = Image.open(input_path)
            img.load()  # Force full decode before .info can be dropped

            if img.mode not in ("RGB", "RGBA"):
                img = img.convert("RGB")


            # Convert to a contiguous uint8 NumPy array - required by the qoi encoder.
            image_data = np.ascontiguousarray(np.array(img), dtype=np.uint8)

            start_time = time.perf_counter()

            # Encode pixels to a raw QOI byte string and write to disk.
            qoi_bytes = self._qoi.encode(image_data)
            output_path.write_bytes(qoi_bytes)

            compression_time = time.perf_counter() - start_time
            compressed_size  = output_path.stat().st_size

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
        Decode a QOI file back to pixels and save as PNG.

        The full byte read + decode + PNG save is included in the timing so the
        measurement reflects realistic I/O + decode latency, consistent with
        other compressors.

        Args:
            input_path: Compressed .qoi source file.
            output_path: Destination PNG file.

        Returns:
            Wall-clock decompression time in seconds.
        """
        start_time = time.perf_counter()

        qoi_bytes   = input_path.read_bytes()
        image_array = self._qoi.decode(qoi_bytes)   # returns a uint8 ndarray (H, W, C)
        img         = Image.fromarray(image_array)
        img.save(output_path, format="PNG")

        return time.perf_counter() - start_time


# Register so CompressorFactory.create("qoi") works.
CompressorFactory.register("qoi", QOICompressor)