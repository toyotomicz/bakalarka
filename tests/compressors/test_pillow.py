"""
Unit testy pro PillowPNGCompressor, PillowWebPCompressor, PillowTIFFCompressor.

Závislosti jsou mockovány – žádné reálné soubory ani Pillow I/O se nepoužívají.
"""

import sys
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest
from PIL import Image

# ---------------------------------------------------------------------------
# Stub pro main.py (není dostupný v testovacím prostředí)
# ---------------------------------------------------------------------------

import types

main_stub = types.ModuleType("main")


class CompressionLevel:
    FASTEST = "fastest"
    BALANCED = "balanced"
    BEST = "best"


class CompressionMetrics:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class ImageCompressor:
    def __init__(self, lib_path=None):
        self._validate_dependencies()

    def _validate_dependencies(self):
        pass

    @property
    def name(self):
        raise NotImplementedError

    @property
    def extension(self):
        raise NotImplementedError

    def compress(self, input_path, output_path, level=None):
        raise NotImplementedError

    def decompress(self, input_path, output_path):
        raise NotImplementedError


class CompressorFactory:
    _registry = {}

    @classmethod
    def register(cls, key, klass):
        cls._registry[key] = klass

    @classmethod
    def create(cls, key):
        return cls._registry[key]()


main_stub.CompressionLevel = CompressionLevel
main_stub.CompressionMetrics = CompressionMetrics
main_stub.ImageCompressor = ImageCompressor
main_stub.CompressorFactory = CompressorFactory
sys.modules["main"] = main_stub

image_size_stub = types.ModuleType("image_size_calculator")


class ImageSizeCalculator:
    @staticmethod
    def calculate_uncompressed_size(path):
        return 1_000_000  # 1 MB default stub


image_size_stub.ImageSizeCalculator = ImageSizeCalculator
sys.modules["image_size_calculator"] = image_size_stub

# Teď lze importovat modul
from compressors.pillow_compressor import (  # noqa: E402
    PillowPNGCompressor,
    PillowWebPCompressor,
    PillowTIFFCompressor,
    PillowCompressorBase,
)


# ---------------------------------------------------------------------------
# Pomocné funkce
# ---------------------------------------------------------------------------

def _make_rgb_image(width: int = 4, height: int = 4) -> Image.Image:
    return Image.new("RGB", (width, height), color=(128, 64, 32))


def _make_rgba_image(width: int = 4, height: int = 4) -> Image.Image:
    return Image.new("RGBA", (width, height), color=(128, 64, 32, 200))


def _make_palette_image() -> Image.Image:
    img = Image.new("P", (4, 4))
    return img


# ---------------------------------------------------------------------------
# PillowCompressorBase – sdílená logika
# ---------------------------------------------------------------------------

class TestPillowCompressorBase:
    """Testy sdílené logiky base třídy přes konkrétní podtřídu."""

    def test_name_obsahuje_format(self):
        c = PillowPNGCompressor()
        assert "PNG" in c.name

    def test_extension_vraci_spravnou_priponu(self):
        assert PillowPNGCompressor().extension == ".png"
        assert PillowWebPCompressor().extension == ".webp"
        assert PillowTIFFCompressor().extension == ".tiff"

    def test_prepare_image_zachova_rgb(self):
        c = PillowPNGCompressor()
        img = _make_rgb_image()
        result = c._prepare_image(img)
        assert result.mode == "RGB"

    def test_prepare_image_zachova_rgba(self):
        c = PillowPNGCompressor()
        img = _make_rgba_image()
        result = c._prepare_image(img)
        assert result.mode == "RGBA"

    def test_prepare_image_konvertuje_palette_na_rgb(self):
        c = PillowPNGCompressor()
        img = _make_palette_image()
        result = c._prepare_image(img)
        assert result.mode == "RGB"

    def test_compress_vraci_metrics_pri_chybe(self, tmp_path):
        """Když Image.open selže, compress musí vrátit metrics s success=False."""
        c = PillowPNGCompressor()
        neexistujici = tmp_path / "neexistuje.png"
        output = tmp_path / "out.png"

        metrics = c.compress(neexistujici, output)

        assert metrics.success is False
        assert metrics.error_message != ""
        assert metrics.compressed_size == 0
        assert metrics.original_size == 0

    def test_compress_opravuje_priponu_vystupu(self, tmp_path):
        """compress() musí opravit špatnou příponu výstupního souboru."""
        c = PillowPNGCompressor()
        src = tmp_path / "src.png"
        _make_rgb_image().save(src, format="PNG")
        out_wrong = tmp_path / "out.jpg"  # záměrně špatná přípona

        with patch.object(c, "decompress", return_value=0.001):
            metrics = c.compress(src, out_wrong)

        # Soubor musí existovat s .png příponou
        assert (tmp_path / "out.png").exists()
        assert metrics.success is True

    def test_compress_vrati_spravne_metriky(self, tmp_path):
        src = tmp_path / "src.png"
        _make_rgb_image().save(src, format="PNG")
        out = tmp_path / "out.png"

        with patch.object(c := PillowPNGCompressor(), "decompress", return_value=0.005):
            metrics = c.compress(src, out)

        assert metrics.success is True
        assert metrics.original_size == 1_000_000  # ze stubu
        assert metrics.compressed_size > 0
        assert metrics.compression_ratio > 0
        assert metrics.compression_time >= 0
        assert metrics.decompression_time == 0.005

    def test_decompress_vraci_float(self, tmp_path):
        c = PillowPNGCompressor()
        src = tmp_path / "src.png"
        out = tmp_path / "out.png"
        _make_rgb_image().save(src, format="PNG")

        result = c.decompress(src, out)

        assert isinstance(result, float)
        assert result >= 0
        assert out.exists()

    def test_compress_cleanup_temp_souboru_pri_uspechu(self, tmp_path):
        """Dočasný decomp soubor musí být smazán i při úspěchu."""
        c = PillowPNGCompressor()
        src = tmp_path / "src.png"
        _make_rgb_image().save(src, format="PNG")
        out = tmp_path / "out.png"

        c.compress(src, out)

        temp_files = list(tmp_path.glob("temp_decomp_*.png"))
        assert len(temp_files) == 0

    def test_compress_cleanup_temp_souboru_pri_chybe_decompressu(self, tmp_path):
        """Dočasný soubor musí být smazán i když decompress vyhodí výjimku."""
        c = PillowPNGCompressor()
        src = tmp_path / "src.png"
        _make_rgb_image().save(src, format="PNG")
        out = tmp_path / "out.png"

        with patch.object(c, "decompress", side_effect=RuntimeError("selhání")):
            metrics = c.compress(src, out)

        assert metrics.success is False
        temp_files = list(tmp_path.glob("temp_decomp_*.png"))
        assert len(temp_files) == 0


# ---------------------------------------------------------------------------
# PillowPNGCompressor
# ---------------------------------------------------------------------------

class TestPillowPNGCompressor:

    def test_compression_params_fastest(self):
        c = PillowPNGCompressor()
        params = c._get_compression_params(CompressionLevel.FASTEST)
        assert params["compress_level"] == 1
        assert params["optimize"] is False

    def test_compression_params_balanced(self):
        c = PillowPNGCompressor()
        params = c._get_compression_params(CompressionLevel.BALANCED)
        assert params["compress_level"] == 6
        assert params["optimize"] is False

    def test_compression_params_best(self):
        c = PillowPNGCompressor()
        params = c._get_compression_params(CompressionLevel.BEST)
        assert params["compress_level"] == 9
        assert params["optimize"] is True

    def test_vysledny_soubor_je_validni_png(self, tmp_path):
        c = PillowPNGCompressor()
        src = tmp_path / "src.png"
        out = tmp_path / "out.png"
        _make_rgb_image().save(src, format="PNG")

        with patch.object(c, "decompress", return_value=0.0):
            c.compress(src, out)

        img = Image.open(out)
        assert img.format == "PNG"

    def test_komprese_rgba_zachova_pruhlednost(self, tmp_path):
        c = PillowPNGCompressor()
        src = tmp_path / "src.png"
        out = tmp_path / "out.png"
        _make_rgba_image().save(src, format="PNG")

        with patch.object(c, "decompress", return_value=0.0):
            metrics = c.compress(src, out)

        assert metrics.success is True
        img = Image.open(out)
        assert img.mode == "RGBA"


# ---------------------------------------------------------------------------
# PillowWebPCompressor
# ---------------------------------------------------------------------------

class TestPillowWebPCompressor:

    def test_compression_params_lossless_vzdy_true(self):
        c = PillowWebPCompressor()
        for level in (CompressionLevel.FASTEST, CompressionLevel.BALANCED, CompressionLevel.BEST):
            params = c._get_compression_params(level)
            assert params["lossless"] is True, f"lossless musí být True pro level={level}"

    def test_compression_params_method_rozsah(self):
        c = PillowWebPCompressor()
        for level in (CompressionLevel.FASTEST, CompressionLevel.BALANCED, CompressionLevel.BEST):
            params = c._get_compression_params(level)
            assert 0 <= params["method"] <= 6

    def test_compression_params_quality_rozsah(self):
        c = PillowWebPCompressor()
        for level in (CompressionLevel.FASTEST, CompressionLevel.BALANCED, CompressionLevel.BEST):
            params = c._get_compression_params(level)
            assert 0 <= params["quality"] <= 100

    def test_best_ma_nejvyssi_quality(self):
        c = PillowWebPCompressor()
        fastest = c._get_compression_params(CompressionLevel.FASTEST)["quality"]
        best = c._get_compression_params(CompressionLevel.BEST)["quality"]
        assert best >= fastest

    def test_prepare_image_zachova_rgba(self):
        c = PillowWebPCompressor()
        img = _make_rgba_image()
        result = c._prepare_image(img)
        assert result.mode == "RGBA"


# ---------------------------------------------------------------------------
# PillowTIFFCompressor
# ---------------------------------------------------------------------------

class TestPillowTIFFCompressor:

    def test_compression_params_fastest_packbits(self):
        c = PillowTIFFCompressor()
        params = c._get_compression_params(CompressionLevel.FASTEST)
        assert params["compression"] == "packbits"

    def test_compression_params_balanced_lzw(self):
        c = PillowTIFFCompressor()
        params = c._get_compression_params(CompressionLevel.BALANCED)
        assert params["compression"] == "lzw"

    def test_compression_params_best_tiff_deflate(self):
        c = PillowTIFFCompressor()
        params = c._get_compression_params(CompressionLevel.BEST)
        assert params["compression"] == "tiff_deflate"

    def test_neznamy_level_fallback_lzw(self):
        c = PillowTIFFCompressor()
        params = c._get_compression_params("neznamy_level")
        assert params["compression"] == "lzw"