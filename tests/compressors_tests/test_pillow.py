"""
Unit tests for PillowPNGCompressor, PillowWebPCompressor, PillowTIFFCompressor.

Dependencies are mocked; no real Pillow I/O beyond small in-memory images.
Stubs for main and image_size_calculator are provided by conftest.py.
"""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from PIL import Image

# Stubs registered in conftest.py
CompressionLevel = sys.modules["main"].CompressionLevel

from compressors.pillow_compressor import (  # noqa: E402
    PillowPNGCompressor,
    PillowWebPCompressor,
    PillowTIFFCompressor,
    PillowCompressorBase,
)


# ---------------------------------------------------------------------------
# Helper image factories
# ---------------------------------------------------------------------------

def _make_rgb_image(width: int = 4, height: int = 4) -> Image.Image:
    return Image.new("RGB", (width, height), color=(128, 64, 32))


def _make_rgba_image(width: int = 4, height: int = 4) -> Image.Image:
    return Image.new("RGBA", (width, height), color=(128, 64, 32, 200))


def _make_palette_image() -> Image.Image:
    return Image.new("P", (4, 4))


# ---------------------------------------------------------------------------
# PillowCompressorBase – shared logic tested through concrete subclass
# ---------------------------------------------------------------------------

class TestPillowCompressorBase:

    def test_name_contains_format(self):
        assert "PNG" in PillowPNGCompressor().name

    def test_extension_correct_for_each_subclass(self):
        assert PillowPNGCompressor().extension == ".png"
        assert PillowWebPCompressor().extension == ".webp"
        assert PillowTIFFCompressor().extension == ".tiff"

    def test_prepare_image_preserves_rgb(self):
        result = PillowPNGCompressor()._prepare_image(_make_rgb_image())
        assert result.mode == "RGB"

    def test_prepare_image_preserves_rgba(self):
        result = PillowPNGCompressor()._prepare_image(_make_rgba_image())
        assert result.mode == "RGBA"

    def test_prepare_image_converts_palette_to_rgb(self):
        result = PillowPNGCompressor()._prepare_image(_make_palette_image())
        assert result.mode == "RGB"

    def test_compress_returns_failure_metrics_on_error(self, tmp_path):
        """When Image.open fails, compress must return metrics with success=False."""
        c = PillowPNGCompressor()
        metrics = c.compress(tmp_path / "missing.png", tmp_path / "out.png")

        assert metrics.success is False
        assert metrics.error_message != ""
        assert metrics.compressed_size == 0
        assert metrics.original_size == 0

    def test_compress_corrects_output_extension(self, tmp_path):
        """compress() must fix a wrong output file extension."""
        c = PillowPNGCompressor()
        src = tmp_path / "src.png"
        _make_rgb_image().save(src, format="PNG")

        with patch.object(c, "decompress", return_value=0.001):
            metrics = c.compress(src, tmp_path / "out.jpg")  # intentionally wrong extension

        assert (tmp_path / "out.png").exists()
        assert metrics.success is True

    def test_compress_returns_correct_metrics(self, tmp_path):
        c = PillowPNGCompressor()
        src = tmp_path / "src.png"
        out = tmp_path / "out.png"
        _make_rgb_image().save(src, format="PNG")

        with patch.object(c, "decompress", return_value=0.005):
            metrics = c.compress(src, out)

        assert metrics.success is True
        assert metrics.original_size > 0
        assert metrics.compressed_size > 0
        assert metrics.compression_ratio > 0
        assert metrics.compression_time >= 0
        assert metrics.decompression_time == 0.005

    def test_decompress_returns_float(self, tmp_path):
        c = PillowPNGCompressor()
        src = tmp_path / "src.png"
        out = tmp_path / "out.png"
        _make_rgb_image().save(src, format="PNG")

        result = c.decompress(src, out)

        assert isinstance(result, float)
        assert result >= 0
        assert out.exists()

    def test_temp_decomp_file_cleaned_up_on_success(self, tmp_path):
        c = PillowPNGCompressor()
        src = tmp_path / "src.png"
        out = tmp_path / "out.png"
        _make_rgb_image().save(src, format="PNG")

        c.compress(src, out)

        assert list(tmp_path.glob("temp_decomp_*.png")) == []

    def test_temp_decomp_file_cleaned_up_on_decompress_error(self, tmp_path):
        """Temp file must be removed even when decompress raises."""
        c = PillowPNGCompressor()
        src = tmp_path / "src.png"
        out = tmp_path / "out.png"
        _make_rgb_image().save(src, format="PNG")

        with patch.object(c, "decompress", side_effect=RuntimeError("failure")):
            metrics = c.compress(src, out)

        assert metrics.success is False
        assert list(tmp_path.glob("temp_decomp_*.png")) == []


# ---------------------------------------------------------------------------
# PillowPNGCompressor
# ---------------------------------------------------------------------------

class TestPillowPNGCompressor:

    def test_compression_params_fastest(self):
        params = PillowPNGCompressor()._get_compression_params(CompressionLevel.FASTEST)
        assert params["compress_level"] == 1
        assert params["optimize"] is False

    def test_compression_params_balanced(self):
        params = PillowPNGCompressor()._get_compression_params(CompressionLevel.BALANCED)
        assert params["compress_level"] == 6
        assert params["optimize"] is False

    def test_compression_params_best(self):
        params = PillowPNGCompressor()._get_compression_params(CompressionLevel.BEST)
        assert params["compress_level"] == 9
        assert params["optimize"] is True

    def test_compress_level_increases_with_quality(self):
        c = PillowPNGCompressor()
        fastest = c._get_compression_params(CompressionLevel.FASTEST)["compress_level"]
        balanced = c._get_compression_params(CompressionLevel.BALANCED)["compress_level"]
        best = c._get_compression_params(CompressionLevel.BEST)["compress_level"]
        assert fastest < balanced < best

    def test_output_is_valid_png(self, tmp_path):
        c = PillowPNGCompressor()
        src = tmp_path / "src.png"
        out = tmp_path / "out.png"
        _make_rgb_image().save(src, format="PNG")

        with patch.object(c, "decompress", return_value=0.0):
            c.compress(src, out)

        assert Image.open(out).format == "PNG"

    def test_compress_rgba_preserves_transparency(self, tmp_path):
        c = PillowPNGCompressor()
        src = tmp_path / "src.png"
        out = tmp_path / "out.png"
        _make_rgba_image().save(src, format="PNG")

        with patch.object(c, "decompress", return_value=0.0):
            metrics = c.compress(src, out)

        assert metrics.success is True
        assert Image.open(out).mode == "RGBA"


# ---------------------------------------------------------------------------
# PillowWebPCompressor
# ---------------------------------------------------------------------------

class TestPillowWebPCompressor:

    def test_lossless_always_true(self):
        c = PillowWebPCompressor()
        for level in (CompressionLevel.FASTEST, CompressionLevel.BALANCED, CompressionLevel.BEST):
            assert c._get_compression_params(level)["lossless"] is True, (
                f"lossless must be True for level={level}"
            )

    def test_method_within_valid_range(self):
        c = PillowWebPCompressor()
        for level in (CompressionLevel.FASTEST, CompressionLevel.BALANCED, CompressionLevel.BEST):
            assert 0 <= c._get_compression_params(level)["method"] <= 6

    def test_quality_within_valid_range(self):
        c = PillowWebPCompressor()
        for level in (CompressionLevel.FASTEST, CompressionLevel.BALANCED, CompressionLevel.BEST):
            assert 0 <= c._get_compression_params(level)["quality"] <= 100

    def test_best_has_highest_quality(self):
        c = PillowWebPCompressor()
        assert (
            c._get_compression_params(CompressionLevel.BEST)["quality"]
            >= c._get_compression_params(CompressionLevel.FASTEST)["quality"]
        )

    def test_prepare_image_preserves_rgba(self):
        result = PillowWebPCompressor()._prepare_image(_make_rgba_image())
        assert result.mode == "RGBA"

    def test_output_is_valid_webp(self, tmp_path):
        c = PillowWebPCompressor()
        src = tmp_path / "src.png"
        out = tmp_path / "out.webp"
        _make_rgb_image().save(src, format="PNG")

        with patch.object(c, "decompress", return_value=0.0):
            metrics = c.compress(src, out)

        assert metrics.success is True
        assert Image.open(out).format == "WEBP"


# ---------------------------------------------------------------------------
# PillowTIFFCompressor
# ---------------------------------------------------------------------------

class TestPillowTIFFCompressor:

    def test_fastest_uses_packbits(self):
        params = PillowTIFFCompressor()._get_compression_params(CompressionLevel.FASTEST)
        assert params["compression"] == "packbits"

    def test_balanced_uses_lzw(self):
        params = PillowTIFFCompressor()._get_compression_params(CompressionLevel.BALANCED)
        assert params["compression"] == "lzw"

    def test_best_uses_tiff_deflate(self):
        params = PillowTIFFCompressor()._get_compression_params(CompressionLevel.BEST)
        assert params["compression"] == "tiff_deflate"

    def test_unknown_level_falls_back_to_lzw(self):
        """Undefined levels must fall back to lzw, not raise an exception."""
        params = PillowTIFFCompressor()._get_compression_params("unknown_level")
        assert params["compression"] == "lzw"