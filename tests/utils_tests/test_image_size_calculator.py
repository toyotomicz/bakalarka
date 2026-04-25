"""
Tests for utils/image_size_calculator.py.

Covers ImageInfo computed properties, ImageSizeCalculator.get_image_info(),
calculate_uncompressed_size() and get_compression_baseline().

Location: tests/utils_tests/test_image_size_calculator.py
"""

from pathlib import Path
from unittest.mock import patch

import pytest
from PIL import Image

from utils.image_size_calculator import ImageInfo, ImageSizeCalculator


# Helpers (for generating test PNG files with various modes and sizes)

def _save_png(tmp_path: Path, mode: str, size=(4, 4), color=None) -> Path:
    """
    Saves a test PNG file and returns its path.

    Args:
        tmp_path: Directory in which to save the file.
        mode: Pillow colour mode (e.g. 'RGB', 'RGBA', 'L').
        size: Image dimensions as (width, height).
        color: Fill colour; None = default for the given mode.

    Returns:
        Path to the saved PNG file.
    """
    path = tmp_path / f"img_{mode}.png"
    img = Image.new(mode, size, color)
    img.save(path, format="PNG")
    return path


# ImageInfo computed properties

class TestImageInfo:
    """Verifies the computed properties of ImageInfo."""

    def _make_info(self, width=10, height=20, channels=3, bpp=8) -> ImageInfo:
        raw = width * height * channels * bpp // 8
        return ImageInfo(
            width=width,
            height=height,
            mode="RGB",
            channels=channels,
            bits_per_channel=bpp,
            bits_per_pixel=channels * bpp,
            uncompressed_size_bytes=raw,
            file_size_bytes=raw // 2,
        )

    def test_resolution_is_width_times_height(self):
        info = self._make_info(width=10, height=20)
        assert info.resolution == 200

    def test_megapixels_correct(self):
        info = self._make_info(width=1000, height=1000)
        assert info.megapixels == pytest.approx(1.0)

    def test_megapixels_small_image(self):
        info = self._make_info(width=4, height=4)
        assert info.megapixels == pytest.approx(16 / 1_000_000)


# ImageSizeCalculator.get_image_info()

class TestGetImageInfo:
    """Verifies geometry and bit-depth calculations for various colour modes."""

    def test_rgb_4x4_correct_size(self, tmp_path):
        path = _save_png(tmp_path, "RGB", size=(4, 4))
        info = ImageSizeCalculator.get_image_info(path)
        assert info is not None
        # 4 x 4 x 3 channels x 8 bits / 8 = 48 bytes
        assert info.uncompressed_size_bytes == 48
        assert info.channels == 3
        assert info.bits_per_channel == 8

    def test_rgba_4x4_correct_size(self, tmp_path):
        path = _save_png(tmp_path, "RGBA", size=(4, 4))
        info = ImageSizeCalculator.get_image_info(path)
        assert info is not None
        # 4 x 4 x 4 channels x 8 bits / 8 = 64 bytes
        assert info.uncompressed_size_bytes == 64
        assert info.channels == 4

    def test_grayscale_l_mode(self, tmp_path):
        path = _save_png(tmp_path, "L", size=(4, 4))
        info = ImageSizeCalculator.get_image_info(path)
        assert info is not None
        # 4 x 4 x 1 channel x 8 bits / 8 = 16 bytes
        assert info.uncompressed_size_bytes == 16
        assert info.channels == 1

    def test_returns_correct_width_and_height(self, tmp_path):
        path = _save_png(tmp_path, "RGB", size=(10, 7))
        info = ImageSizeCalculator.get_image_info(path)
        assert info.width == 10
        assert info.height == 7

    def test_returns_correct_mode_string(self, tmp_path):
        path = _save_png(tmp_path, "RGBA", size=(4, 4))
        info = ImageSizeCalculator.get_image_info(path)
        assert info.mode == "RGBA"

    def test_file_size_bytes_matches_actual_file(self, tmp_path):
        path = _save_png(tmp_path, "RGB", size=(4, 4))
        info = ImageSizeCalculator.get_image_info(path)
        assert info.file_size_bytes == path.stat().st_size

    def test_returns_none_for_missing_file(self, tmp_path):
        missing = tmp_path / "nonexistent.png"
        info = ImageSizeCalculator.get_image_info(missing)
        assert info is None

    def test_returns_none_for_corrupt_file(self, tmp_path):
        corrupt = tmp_path / "corrupt.png"
        corrupt.write_bytes(b"\x00\x01\x02\x03")
        info = ImageSizeCalculator.get_image_info(corrupt)
        assert info is None

    def test_palette_mode_treated_as_one_channel(self, tmp_path):
        """Palette (P) mode must be interpreted as 1 channel, 8 bits."""
        path = tmp_path / "palette.png"
        img = Image.new("P", (4, 4))
        img.save(path, format="PNG")
        info = ImageSizeCalculator.get_image_info(path)
        assert info is not None
        assert info.channels == 1

    def test_bits_per_pixel_is_channels_times_bits_per_channel(self, tmp_path):
        path = _save_png(tmp_path, "RGB", size=(4, 4))
        info = ImageSizeCalculator.get_image_info(path)
        assert info.bits_per_pixel == info.channels * info.bits_per_channel

    def test_uncompressed_size_equals_width_times_height_times_bytes_per_pixel(self, tmp_path):
        """Direct verification of the formula: width x height x (bpp / 8)."""
        path = _save_png(tmp_path, "RGB", size=(8, 6))
        info = ImageSizeCalculator.get_image_info(path)
        expected = 8 * 6 * 3  # RGB = 3 bytes per pixel
        assert info.uncompressed_size_bytes == expected


# ImageSizeCalculator.calculate_uncompressed_size()

class TestCalculateUncompressedSize:
    """Verifies that calculate_uncompressed_size() returns the correct value or a fallback."""

    def test_returns_positive_int_for_valid_file(self, tmp_path):
        path = _save_png(tmp_path, "RGB", size=(4, 4))
        size = ImageSizeCalculator.calculate_uncompressed_size(path)
        assert isinstance(size, int)
        assert size > 0

    def test_returns_correct_value_for_rgb(self, tmp_path):
        path = _save_png(tmp_path, "RGB", size=(4, 4))
        size = ImageSizeCalculator.calculate_uncompressed_size(path)
        assert size == 48  # 4 x 4 x 3

    def test_returns_correct_value_for_rgba(self, tmp_path):
        path = _save_png(tmp_path, "RGBA", size=(4, 4))
        size = ImageSizeCalculator.calculate_uncompressed_size(path)
        assert size == 64  # 4 x 4 x 4

    def test_fallback_to_file_size_when_info_is_none(self, tmp_path):
        """
        If get_image_info() returns None (e.g. corruption), the actual
        on-disk file size must be returned instead of 0 or an exception.
        """
        path = _save_png(tmp_path, "RGB", size=(4, 4))
        with patch.object(ImageSizeCalculator, "get_image_info", return_value=None):
            result = ImageSizeCalculator.calculate_uncompressed_size(path)
        assert result == path.stat().st_size

    def test_rgba_is_larger_than_rgb_for_same_dimensions(self, tmp_path):
        rgb_path = tmp_path / "rgb.png"
        rgba_path = tmp_path / "rgba.png"
        Image.new("RGB", (8, 8)).save(rgb_path)
        Image.new("RGBA", (8, 8)).save(rgba_path)

        rgb_size  = ImageSizeCalculator.calculate_uncompressed_size(rgb_path)
        rgba_size = ImageSizeCalculator.calculate_uncompressed_size(rgba_path)
        assert rgba_size > rgb_size


# ImageSizeCalculator.get_compression_baseline()

class TestGetCompressionBaseline:
    """Verifies the content and structure of the dict returned by get_compression_baseline()."""

    def test_returns_dict_with_required_keys(self, tmp_path):
        path = _save_png(tmp_path, "RGB", size=(4, 4))
        result = ImageSizeCalculator.get_compression_baseline(path)
        for key in ("uncompressed_size", "file_size", "baseline_ratio", "format"):
            assert key in result, f"Missing key: {key}"

    def test_baseline_ratio_is_positive_for_valid_image(self, tmp_path):
        path = _save_png(tmp_path, "RGB", size=(4, 4))
        result = ImageSizeCalculator.get_compression_baseline(path)
        assert result["baseline_ratio"] > 0

    def test_format_matches_file_suffix(self, tmp_path):
        path = _save_png(tmp_path, "RGB", size=(4, 4))
        result = ImageSizeCalculator.get_compression_baseline(path)
        assert result["format"] == ".png"

    def test_width_and_height_present_for_valid_image(self, tmp_path):
        path = _save_png(tmp_path, "RGB", size=(6, 8))
        result = ImageSizeCalculator.get_compression_baseline(path)
        assert result["width"] == 6
        assert result["height"] == 8

    def test_returns_zero_ratio_for_missing_file(self, tmp_path):
        missing = tmp_path / "gone.png"
        result = ImageSizeCalculator.get_compression_baseline(missing)
        assert result["baseline_ratio"] == 0
        assert result["uncompressed_size"] == 0