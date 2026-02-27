"""
Image Verification Module
utils/verification.py

Validates lossless compression by pixel-level comparison of the original image
and the decompressed output.

Design note: This module intentionally does NOT import from main.py.
The CompressorFactory dependency is injected as a parameter, which prevents
circular imports and makes the module easy to unit-test in isolation.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, TYPE_CHECKING

import numpy as np
from PIL import Image

if TYPE_CHECKING:
    # Imported for type checkers only; never executed at runtime.
    from main import CompressorFactory as CompressorFactoryType

logger = logging.getLogger(__name__)


@dataclass
class VerificationResult:
    """Pixel-level comparison result between an original and a decompressed image."""

    is_lossless:      bool
    max_difference:   float   # maximum absolute per-channel pixel difference
    different_pixels: int     # number of pixels where any channel differs
    total_pixels:     int
    error_message:    Optional[str] = None

    @property
    def accuracy_percent(self) -> float:
        """Percentage of pixels that are bit-exactly identical."""
        if self.total_pixels == 0:
            return 0.0
        return ((self.total_pixels - self.different_pixels) / self.total_pixels) * 100.0

    @property
    def identical_pixels(self) -> int:
        return self.total_pixels - self.different_pixels


class ImageVerifier:
    """
    Verifies that a compression round-trip is truly lossless.

    Usage:
        result = ImageVerifier.verify_lossless(
            original_path    = Path("photo.png"),
            compressed_path  = Path("photo.jls"),
            compressor_factory = CompressorFactory,   # injected from outside
            temp_dir         = Path("/tmp"),
        )
    """

    @staticmethod
    def verify_lossless(
        original_path: Path,
        compressed_path: Path,
        compressor_factory: Optional["CompressorFactoryType"] = None,
        temp_dir: Optional[Path] = None,
    ) -> VerificationResult:
        """
        Compare the original image and the decompressed output pixel by pixel.

        Args:
            original_path:      Path to the original (stripped) image — must be the
                                 same file that was fed to the compressor.
            compressed_path:    Path to the compressed output file.
            compressor_factory: CompressorFactory used to decompress formats that
                                 Pillow cannot open directly.  Pass None to rely on
                                 Pillow alone.
            temp_dir:           Directory for the temporary decompressed PNG.
                                 Defaults to the directory of compressed_path.

        Returns:
            VerificationResult with detailed per-pixel statistics.
        """
        try:
            img_original = Image.open(original_path)

            img_compressed, temp_path = ImageVerifier._open_compressed(
                compressed_path=compressed_path,
                compressor_factory=compressor_factory,
                temp_dir=temp_dir or compressed_path.parent,
            )

            if img_compressed is None:
                return VerificationResult(
                    is_lossless=False,
                    max_difference=0.0,
                    different_pixels=0,
                    total_pixels=0,
                    error_message=(
                        f"Cannot open or decompress '{compressed_path.suffix}' file."
                    ),
                )

            try:
                return ImageVerifier._compare(img_original, img_compressed)
            finally:
                if temp_path is not None:
                    try:
                        temp_path.unlink()
                    except OSError as exc:
                        logger.debug("Could not delete temp file %s: %s", temp_path, exc)

        except Exception as exc:
            logger.exception("verify_lossless failed for %s", compressed_path)
            return VerificationResult(
                is_lossless=False,
                max_difference=0.0,
                different_pixels=0,
                total_pixels=0,
                error_message=str(exc),
            )

    @staticmethod
    def create_difference_map(
        original_path: Path,
        compressed_path: Path,
        compressor_factory: Optional["CompressorFactoryType"] = None,
        temp_dir: Optional[Path] = None,
    ) -> Optional[np.ndarray]:
        """
        Return a boolean mask (H × W) that is True wherever the two images differ.

        Returns None on any error (mismatched sizes, decoding failure, etc.).
        """
        try:
            img_original = Image.open(original_path)

            img_compressed, temp_path = ImageVerifier._open_compressed(
                compressed_path=compressed_path,
                compressor_factory=compressor_factory,
                temp_dir=temp_dir or compressed_path.parent,
            )

            if img_compressed is None:
                return None

            try:
                if img_original.size != img_compressed.size:
                    return None

                img_compressed = img_compressed.convert(img_original.mode)
                arr_orig = np.array(img_original, dtype=np.float32)
                arr_comp = np.array(img_compressed, dtype=np.float32)
                diff = np.abs(arr_orig - arr_comp)

                # Collapse the channel axis for colour images; keep scalar for greyscale.
                return np.any(diff > 0, axis=-1) if diff.ndim == 3 else diff > 0

            finally:
                if temp_path is not None:
                    try:
                        temp_path.unlink()
                    except OSError as exc:
                        logger.debug("Could not delete temp file %s: %s", temp_path, exc)

        except Exception:
            logger.exception("create_difference_map failed for %s", compressed_path)
            return None

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _open_compressed(
        compressed_path: Path,
        compressor_factory: Optional["CompressorFactoryType"],
        temp_dir: Path,
    ) -> tuple[Optional[Image.Image], Optional[Path]]:
        """
        Try to open a compressed file as a PIL Image.

        Strategy:
          1. Ask Pillow to open the file directly (handles PNG, WEBP, TIFF, …).
          2. If Pillow fails and a compressor_factory is provided, decompress to a
             temporary PNG and open that instead.

        Returns:
            (PIL Image or None, temp file path to delete or None)
        """
        # First attempt: Pillow direct open.
        try:
            return Image.open(compressed_path), None
        except Exception:
            pass

        # Second attempt: decompress via CompressorFactory.
        if compressor_factory is None:
            logger.debug(
                "Pillow cannot open '%s' and no compressor_factory was provided.",
                compressed_path,
            )
            return None, None

        temp_output = ImageVerifier._decompress_via_factory(
            compressed_path=compressed_path,
            compressor_factory=compressor_factory,
            temp_dir=temp_dir,
        )

        if temp_output is None:
            return None, None

        try:
            return Image.open(temp_output), temp_output
        except Exception as exc:
            logger.debug("Cannot open decompressed temp file %s: %s", temp_output, exc)
            try:
                temp_output.unlink()
            except OSError:
                pass
            return None, None

    @staticmethod
    def _decompress_via_factory(
        compressed_path: Path,
        compressor_factory: "CompressorFactoryType",
        temp_dir: Path,
    ) -> Optional[Path]:
        """
        Find the registered compressor whose extension matches compressed_path and
        decompress the file to a temporary PNG in temp_dir.

        Returns the path to the temporary PNG, or None if no suitable compressor
        is found or decompression fails.
        """
        extension = compressed_path.suffix.lower()

        for comp_name in compressor_factory.list_available():
            try:
                compressor = compressor_factory.create(comp_name)
                if compressor.extension.lower() != extension:
                    continue

                temp_output = temp_dir / f"_verify_{compressed_path.stem}.png"
                compressor.decompress(compressed_path, temp_output)
                logger.debug("Decompressed via %s → %s", comp_name, temp_output)
                return temp_output

            except Exception as exc:
                logger.debug(
                    "Compressor '%s' failed to decompress %s: %s",
                    comp_name, compressed_path, exc,
                )
                continue

        logger.warning("No compressor found for extension '%s'.", extension)
        return None

    @staticmethod
    def _compare(
        img_original: Image.Image,
        img_compressed: Image.Image,
    ) -> VerificationResult:
        """
        Perform a pixel-level comparison of two PIL Images.

        Colour modes are unified before comparison; dimension mismatches are
        reported as a failure rather than raising an exception.
        """
        # Unify colour modes so the array shapes are comparable.
        if img_original.mode != img_compressed.mode:
            img_compressed = img_compressed.convert(img_original.mode)

        if img_original.size != img_compressed.size:
            total = img_original.size[0] * img_original.size[1]
            return VerificationResult(
                is_lossless=False,
                max_difference=float("inf"),
                different_pixels=total,
                total_pixels=total,
                error_message=(
                    f"Dimension mismatch: "
                    f"{img_original.size} vs {img_compressed.size}"
                ),
            )

        arr_orig = np.array(img_original, dtype=np.float32)
        arr_comp = np.array(img_compressed, dtype=np.float32)

        # Add a channel dimension to greyscale arrays so the channel-collapse
        # logic below works uniformly for both greyscale and colour images.
        if arr_orig.ndim == 2:
            arr_orig = arr_orig[:, :, np.newaxis]
        if arr_comp.ndim == 2:
            arr_comp = arr_comp[:, :, np.newaxis]

        diff = np.abs(arr_orig - arr_comp)
        max_diff        = float(np.max(diff))
        different_pixels = int(np.sum(np.any(diff > 0, axis=-1)))
        total_pixels    = arr_orig.shape[0] * arr_orig.shape[1]

        return VerificationResult(
            is_lossless=(max_diff == 0.0),
            max_difference=max_diff,
            different_pixels=different_pixels,
            total_pixels=total_pixels,
        )