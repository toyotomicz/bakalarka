"""
Universal Compressor Template – Standardised Version
compressors/compressor_template_standardized.py

Use this file as the starting point for every new compressor plugin.
Following this template guarantees that all compressors:

  1. Report a comparable baseline via ImageSizeCalculator (not file size).
  2. Preserve alpha channels (RGBA) unless the target format forbids it.
  3. Measure decompression time via a temp-file round-trip.
  4. Return consistent CompressionMetrics on both success and failure.
  5. Clean up temporary files even when exceptions are raised.
  6. Register themselves with CompressorFactory under a short key.

Quick-start checklist
---------------------
  [ ] Copy this file to compressors/myformat_compressor.py
  [ ] Replace "StandardFormat" / ".std" with the real name and extension
  [ ] Implement _validate_dependencies()  – check CLI binaries / Python libs
  [ ] Implement the actual encoding inside compress()
  [ ] Implement decompress()              – decode + save as PNG
  [ ] Update CompressorFactory.register() at the bottom
  [ ] Delete all TODO comments and this docstring block
"""

import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

sys.path.append(str(Path(__file__).parent.parent))
from main import CompressionLevel, CompressionMetrics, CompressorFactory, ImageCompressor
from image_size_calculator import ImageSizeCalculator


class StandardizedCompressor(ImageCompressor):
    """
    Template for a standardised compressor implementation.

    Replace this class name and all TODO sections before using in production.
    """

    def __init__(self, lib_path: Optional[Path] = None):
        super().__init__(lib_path)

    def _validate_dependencies(self) -> None:
        """
        Check that all required external tools or Python packages are available.

        Raise RuntimeError with a descriptive message if anything is missing.
        This method is called by super().__init__(), so keep it side-effect-free
        apart from setting instance attributes needed for compress() / decompress().
        """
        # TODO: check CLI binary path or `import mylib`
        pass

    # -- ImageCompressor interface --

    @property
    def name(self) -> str:
        """Human-readable compressor name shown in benchmark reports."""
        return "StandardFormat"   # TODO: replace with real name

    @property
    def extension(self) -> str:
        """Output file extension including the leading dot."""
        return ".std"             # TODO: replace with real extension

    def compress(
        self,
        input_path: Path,
        output_path: Path,
        level: CompressionLevel = CompressionLevel.BALANCED,
    ) -> CompressionMetrics:
        """
        Compress an image and return timing / size metrics.

        IMPORTANT: Always use ImageSizeCalculator for original_size so that all
        compressors report the same uncompressed baseline regardless of source
        format, metadata, or file-system block size.
        """
        try:
            # ----------------------------------------------------------------
            # 1. Measure the uncompressed pixel size (width × height × channels).
            # ----------------------------------------------------------------
            original_size = ImageSizeCalculator.calculate_uncompressed_size(input_path)

            # ----------------------------------------------------------------
            # 2. Load and prepare the image.
            # ----------------------------------------------------------------
            img = Image.open(input_path)
            img.load()  # Force full decode; also needed before .info can be dropped

            # Preserve RGBA where possible; only convert truly exotic modes.
            if img.mode not in ("RGB", "RGBA", "L", "LA"):
                img = img.convert("RGB")

            # Drop any residual metadata (EXIF, ICC, XMP) that survived the
            # strip step or came from a non-stripped source.  Rebuild from raw
            # pixel data into a fresh Image with an empty .info dict.
            clean = Image.new(img.mode, img.size)
            clean.putdata(img.getdata())
            img = clean

            # Convert to a NumPy array if the encoding library needs raw pixels.
            image_data = np.array(img)

            # ----------------------------------------------------------------
            # 3. Encode and write the output file (timed).
            # ----------------------------------------------------------------
            start_time = time.perf_counter()

            # TODO: replace the placeholder save with the actual encoder call.
            img.save(output_path)

            compression_time = time.perf_counter() - start_time

            # ----------------------------------------------------------------
            # 4. Collect output size.
            # ----------------------------------------------------------------
            compressed_size = output_path.stat().st_size

            # ----------------------------------------------------------------
            # 5. Measure decompression via a temp file (always clean up).
            # ----------------------------------------------------------------
            temp_decomp = output_path.parent / f"temp_decomp_{output_path.stem}.png"
            try:
                decompression_time = self.decompress(output_path, temp_decomp)
            finally:
                if temp_decomp.exists():
                    temp_decomp.unlink()

            # ----------------------------------------------------------------
            # 6. Return metrics.
            # ----------------------------------------------------------------
            return CompressionMetrics(
                original_size=original_size,
                compressed_size=compressed_size,
                compression_ratio=original_size / compressed_size,
                compression_time=compression_time,
                decompression_time=decompression_time,
                success=True,
            )

        except Exception as exc:
            # Return a zero-filled failure record rather than propagating the
            # exception so that benchmark loops can continue with other images.
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
        Decode the compressed file and save the result as PNG.

        Returns the wall-clock decode time in seconds. The PNG save is included
        in the measurement so that all compressors report a comparable number
        (some use native CLI tools, others use Pillow).

        img.load() forces the full pixel decode into memory before timing stops;
        without it Pillow may defer decoding until the first pixel access.
        """
        start_time = time.perf_counter()

        # TODO: replace with the actual decoder if a CLI tool is used.
        img = Image.open(input_path)
        img.load()          # Force full pixel decode into memory
        img.save(output_path, format="PNG")

        return time.perf_counter() - start_time


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

# Uncomment and update the key when this template is used for a real compressor.
# CompressorFactory.register("myformat", StandardizedCompressor)