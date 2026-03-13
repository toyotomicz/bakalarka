"""
Tests for utils/verification.py

Covers:
  - VerificationResult : accuracy_percent, identical_pixels
  - ImageVerifier      : _compare(), verify_lossless(), create_difference_map()
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from utils.verification import ImageVerifier, VerificationResult


# ============================================================================
# Helpers
# ============================================================================

def _make_rgb_image(width: int = 4, height: int = 4, color=(100, 150, 200)) -> Image.Image:
    """Create a solid-colour RGB image of the given dimensions."""
    return Image.new("RGB", (width, height), color)


def _save_temp_png(img: Image.Image, directory: Path) -> Path:
    path = directory / "temp_image.png"
    img.save(path)
    return path


# ============================================================================
# VerificationResult
# ============================================================================

class TestVerificationResult:
    def test_accuracy_100_percent_when_lossless(self):
        result = VerificationResult(
            is_lossless=True,
            max_difference=0.0,
            different_pixels=0,
            total_pixels=1000,
        )
        assert result.accuracy_percent == pytest.approx(100.0)

    def test_accuracy_zero_when_all_pixels_differ(self):
        result = VerificationResult(
            is_lossless=False,
            max_difference=255.0,
            different_pixels=1000,
            total_pixels=1000,
        )
        assert result.accuracy_percent == pytest.approx(0.0)

    def test_accuracy_50_percent(self):
        result = VerificationResult(
            is_lossless=False,
            max_difference=1.0,
            different_pixels=500,
            total_pixels=1000,
        )
        assert result.accuracy_percent == pytest.approx(50.0)

    def test_accuracy_zero_when_no_pixels(self):
        result = VerificationResult(
            is_lossless=False,
            max_difference=0.0,
            different_pixels=0,
            total_pixels=0,
        )
        assert result.accuracy_percent == 0.0

    def test_identical_pixels(self):
        result = VerificationResult(
            is_lossless=False,
            max_difference=1.0,
            different_pixels=300,
            total_pixels=1000,
        )
        assert result.identical_pixels == 700


# ============================================================================
# ImageVerifier._compare()
# ============================================================================

class TestImageVerifierCompare:
    def test_identical_images_are_lossless(self):
        img = _make_rgb_image(color=(128, 64, 32))
        result = ImageVerifier._compare(img, img.copy())

        assert result.is_lossless is True
        assert result.max_difference == 0.0
        assert result.different_pixels == 0
        assert result.total_pixels == 16  # 4×4

    def test_different_images_not_lossless(self):
        img_a = _make_rgb_image(color=(0, 0, 0))
        img_b = _make_rgb_image(color=(255, 255, 255))
        result = ImageVerifier._compare(img_a, img_b)

        assert result.is_lossless is False
        assert result.max_difference == pytest.approx(255.0)
        assert result.different_pixels == 16

    def test_single_pixel_difference(self):
        img_a = _make_rgb_image(color=(100, 100, 100))
        img_b = img_a.copy()

        pixels = img_b.load()
        pixels[0, 0] = (101, 100, 100)

        result = ImageVerifier._compare(img_a, img_b)

        assert result.is_lossless is False
        assert result.different_pixels == 1
        assert result.max_difference == pytest.approx(1.0)

    def test_dimension_mismatch_returns_failure(self):
        img_a = Image.new("RGB", (4, 4), (0, 0, 0))
        img_b = Image.new("RGB", (8, 8), (0, 0, 0))
        result = ImageVerifier._compare(img_a, img_b)

        assert result.is_lossless is False
        assert result.error_message is not None
        assert "mismatch" in result.error_message.lower()

    def test_grayscale_images(self):
        img_a = Image.new("L", (4, 4), 128)
        img_b = Image.new("L", (4, 4), 128)
        result = ImageVerifier._compare(img_a, img_b)

        assert result.is_lossless is True

    def test_mode_conversion_before_compare(self):
        """Images with different modes must be unified before comparison."""
        img_rgb  = Image.new("RGB",  (4, 4), (128, 128, 128))
        img_rgba = Image.new("RGBA", (4, 4), (128, 128, 128, 255))
        # Must not raise
        result = ImageVerifier._compare(img_rgb, img_rgba)
        assert isinstance(result, VerificationResult)


# ============================================================================
# ImageVerifier.verify_lossless() – integration via temporary files
# ============================================================================

class TestVerifyLossless:
    def test_identical_png_files_lossless(self, tmp_path):
        img = _make_rgb_image()
        path = _save_temp_png(img, tmp_path)

        result = ImageVerifier.verify_lossless(path, path)

        assert result.is_lossless is True
        assert result.error_message is None

    def test_different_png_files_not_lossless(self, tmp_path):
        img_a = Image.new("RGB", (4, 4), (0, 0, 0))
        img_b = Image.new("RGB", (4, 4), (255, 255, 255))

        path_a = tmp_path / "a.png"
        path_b = tmp_path / "b.png"
        img_a.save(path_a)
        img_b.save(path_b)

        result = ImageVerifier.verify_lossless(path_a, path_b)

        assert result.is_lossless is False
        assert result.different_pixels == 16

    def test_missing_original_returns_error_result(self, tmp_path):
        img = _make_rgb_image()
        compressed = _save_temp_png(img, tmp_path)
        missing = tmp_path / "nonexistent.png"

        result = ImageVerifier.verify_lossless(missing, compressed)

        assert result.is_lossless is False
        assert result.error_message is not None

    def test_unreadable_compressed_without_factory_returns_error(self, tmp_path):
        img = _make_rgb_image()
        original = _save_temp_png(img, tmp_path)
        fake_compressed = tmp_path / "fake.jls"
        fake_compressed.write_bytes(b"\x00\x01\x02")  # invalid content

        result = ImageVerifier.verify_lossless(
            original, fake_compressed, compressor_factory=None
        )

        assert result.is_lossless is False


# ============================================================================
# ImageVerifier.create_difference_map()
# ============================================================================

class TestCreateDifferenceMap:
    def test_identical_images_all_false(self, tmp_path):
        img = _make_rgb_image()
        path = _save_temp_png(img, tmp_path)

        diff_map = ImageVerifier.create_difference_map(path, path)

        assert diff_map is not None
        assert diff_map.shape == (4, 4)
        assert not diff_map.any()

    def test_completely_different_images_all_true(self, tmp_path):
        img_a = Image.new("RGB", (4, 4), (0, 0, 0))
        img_b = Image.new("RGB", (4, 4), (255, 255, 255))

        path_a = tmp_path / "a.png"
        path_b = tmp_path / "b.png"
        img_a.save(path_a)
        img_b.save(path_b)

        diff_map = ImageVerifier.create_difference_map(path_a, path_b)

        assert diff_map is not None
        assert diff_map.all()

    def test_single_pixel_diff(self, tmp_path):
        img_a = Image.new("RGB", (4, 4), (100, 100, 100))
        img_b = img_a.copy()
        px = img_b.load()
        px[1, 2] = (101, 100, 100)

        path_a = tmp_path / "a.png"
        path_b = tmp_path / "b.png"
        img_a.save(path_a)
        img_b.save(path_b)

        diff_map = ImageVerifier.create_difference_map(path_a, path_b)

        assert diff_map is not None
        assert diff_map.sum() == 1

    def test_missing_file_returns_none(self, tmp_path):
        img = _make_rgb_image()
        path_ok = _save_temp_png(img, tmp_path)
        missing = tmp_path / "gone.png"

        result = ImageVerifier.create_difference_map(path_ok, missing)

        assert result is None
