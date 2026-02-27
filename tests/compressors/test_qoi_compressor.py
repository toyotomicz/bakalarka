"""
Unit testy pro QOICompressor.

qoi Python balíček a souborové I/O jsou mockovány.
"""

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import numpy as np
import pytest
from PIL import Image

# ---------------------------------------------------------------------------
# Stub main + image_size_calculator (pokud ještě nejsou v sys.modules)
# ---------------------------------------------------------------------------

if "main" not in sys.modules:
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

    class CompressorFactory:
        _registry = {}

        @classmethod
        def register(cls, key, klass):
            cls._registry[key] = klass

    main_stub.CompressionLevel = CompressionLevel
    main_stub.CompressionMetrics = CompressionMetrics
    main_stub.ImageCompressor = ImageCompressor
    main_stub.CompressorFactory = CompressorFactory
    sys.modules["main"] = main_stub
else:
    from main import CompressionLevel, CompressionMetrics

if "image_size_calculator" not in sys.modules:
    iscalc_stub = types.ModuleType("image_size_calculator")

    class ImageSizeCalculator:
        @staticmethod
        def calculate_uncompressed_size(path):
            return 1_000_000

    iscalc_stub.ImageSizeCalculator = ImageSizeCalculator
    sys.modules["image_size_calculator"] = iscalc_stub


# ---------------------------------------------------------------------------
# Fixture: mock qoi modul
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_qoi_module():
    """Vrátí mock `qoi` modul s encode/decode metodami."""
    qoi_mock = MagicMock()
    qoi_mock.encode.return_value = b"\x71\x6f\x69\x66" + b"\x00" * 100  # fake QOI bytes
    qoi_mock.decode.return_value = np.zeros((4, 4, 3), dtype=np.uint8)
    return qoi_mock


@pytest.fixture()
def compressor(mock_qoi_module):
    """QOICompressor s mockovaným qoi modulem."""
    with patch.dict("sys.modules", {"qoi": mock_qoi_module}):
        from compressors.qoi_compressor import QOICompressor
        c = QOICompressor()
        c._qoi = mock_qoi_module
        return c


# ---------------------------------------------------------------------------
# Testy: _validate_dependencies
# ---------------------------------------------------------------------------

class TestQOIValidateDependencies:

    def test_vyhodi_pri_chybejicim_balicku(self):
        with patch.dict("sys.modules", {"qoi": None}):
            # Znovu importovat, aby se spustil __init__
            import importlib
            # Odstraníme cache modulu, aby se znovu inicializoval
            if "compressors.qoi_compressor" in sys.modules:
                del sys.modules["compressors.qoi_compressor"]

            from compressors.qoi_compressor import QOICompressor

            c = object.__new__(QOICompressor)
            c._qoi = None
            with pytest.raises(RuntimeError, match="qoi"):
                c._validate_dependencies()

    def test_nastavi_self_qoi(self, mock_qoi_module):
        with patch.dict("sys.modules", {"qoi": mock_qoi_module}):
            if "compressors.qoi_compressor" in sys.modules:
                del sys.modules["compressors.qoi_compressor"]
            from compressors.qoi_compressor import QOICompressor

            c = object.__new__(QOICompressor)
            c._qoi = None
            c._validate_dependencies()
            assert c._qoi is mock_qoi_module


# ---------------------------------------------------------------------------
# Testy: vlastnosti
# ---------------------------------------------------------------------------

class TestQOIProperties:

    def test_name(self, compressor):
        assert compressor.name == "QOI"

    def test_extension(self, compressor):
        assert compressor.extension == ".qoi"


# ---------------------------------------------------------------------------
# Testy: compress()
# ---------------------------------------------------------------------------

class TestQOICompress:

    def test_compress_vola_encode_s_uint8_array(self, compressor, tmp_path):
        src = tmp_path / "src.png"
        Image.new("RGB", (4, 4)).save(src, format="PNG")
        out = tmp_path / "out.qoi"

        with patch.object(compressor, "decompress", return_value=0.001):
            metrics = compressor.compress(src, out)

        assert compressor._qoi.encode.called
        args = compressor._qoi.encode.call_args[0][0]
        assert args.dtype == np.uint8

    def test_compress_zapise_bytes_do_souboru(self, compressor, tmp_path):
        src = tmp_path / "src.png"
        Image.new("RGB", (4, 4)).save(src, format="PNG")
        out = tmp_path / "out.qoi"

        with patch.object(compressor, "decompress", return_value=0.001):
            compressor.compress(src, out)

        assert out.exists()
        assert out.read_bytes() == compressor._qoi.encode.return_value

    def test_compress_vraci_uspesne_metriky(self, compressor, tmp_path):
        src = tmp_path / "src.png"
        Image.new("RGB", (4, 4)).save(src, format="PNG")
        out = tmp_path / "out.qoi"

        with patch.object(compressor, "decompress", return_value=0.002):
            metrics = compressor.compress(src, out)

        assert metrics.success is True
        assert metrics.original_size == 1_000_000
        assert metrics.compressed_size > 0
        assert metrics.compression_ratio > 0
        assert metrics.decompression_time == 0.002

    def test_compress_konvertuje_nekompatibilni_mode(self, compressor, tmp_path):
        """Palette (P) mode musí být konvertován na RGB před encodingem."""
        src = tmp_path / "src.png"
        img = Image.new("P", (4, 4))
        img.save(src, format="PNG")
        out = tmp_path / "out.qoi"

        with patch.object(compressor, "decompress", return_value=0.0):
            metrics = compressor.compress(src, out)

        assert metrics.success is True
        # Ověříme, že encode byl volán s 3kanálovým polem (RGB)
        args = compressor._qoi.encode.call_args[0][0]
        assert args.ndim == 3
        assert args.shape[2] == 3

    def test_compress_vrati_failure_pri_chybe_encode(self, compressor, tmp_path):
        compressor._qoi.encode.side_effect = RuntimeError("encode selhal")
        src = tmp_path / "src.png"
        Image.new("RGB", (4, 4)).save(src, format="PNG")
        out = tmp_path / "out.qoi"

        metrics = compressor.compress(src, out)

        assert metrics.success is False
        assert "encode selhal" in metrics.error_message

    def test_compress_vrati_failure_pro_neexistujici_soubor(self, compressor, tmp_path):
        out = tmp_path / "out.qoi"
        metrics = compressor.compress(tmp_path / "neexistuje.png", out)
        assert metrics.success is False

    def test_compress_smaze_temp_soubor(self, compressor, tmp_path):
        src = tmp_path / "src.png"
        Image.new("RGB", (4, 4)).save(src, format="PNG")
        out = tmp_path / "out.qoi"

        with patch.object(compressor, "decompress", return_value=0.0):
            compressor.compress(src, out)

        temp_files = list(tmp_path.glob("temp_decomp_*.png"))
        assert len(temp_files) == 0

    def test_compress_smaze_temp_soubor_i_pri_chybe_decompress(self, compressor, tmp_path):
        src = tmp_path / "src.png"
        Image.new("RGB", (4, 4)).save(src, format="PNG")
        out = tmp_path / "out.qoi"

        with patch.object(compressor, "decompress", side_effect=RuntimeError("chyba")):
            metrics = compressor.compress(src, out)

        assert metrics.success is False
        temp_files = list(tmp_path.glob("temp_decomp_*.png"))
        assert len(temp_files) == 0

    def test_compress_ratio_neni_nula_kdyz_compressed_size_nula(self, compressor, tmp_path):
        """Pokud compressed_size == 0, ratio musí být 0 (ne ZeroDivisionError)."""
        compressor._qoi.encode.return_value = b""  # prázdné bytes → compressed_size=0
        src = tmp_path / "src.png"
        Image.new("RGB", (4, 4)).save(src, format="PNG")
        out = tmp_path / "out.qoi"

        with patch.object(compressor, "decompress", return_value=0.0):
            metrics = compressor.compress(src, out)

        # success=False protože compressed_size==0 vede k dělení nulou
        # NEBO success=True s ratio=0 — obojí je akceptovatelné, nesmí vybuchnout
        assert metrics.compression_ratio == 0 or metrics.success is False


# ---------------------------------------------------------------------------
# Testy: decompress()
# ---------------------------------------------------------------------------

class TestQOIDecompress:

    def test_decompress_vola_decode_s_bytes(self, compressor, tmp_path):
        qoi_data = b"\x71\x6f\x69\x66" + b"\x00" * 50
        src = tmp_path / "test.qoi"
        src.write_bytes(qoi_data)
        out = tmp_path / "out.png"

        compressor.decompress(src, out)

        compressor._qoi.decode.assert_called_once_with(qoi_data)

    def test_decompress_vraci_float(self, compressor, tmp_path):
        src = tmp_path / "test.qoi"
        src.write_bytes(b"\x00" * 50)
        out = tmp_path / "out.png"

        result = compressor.decompress(src, out)

        assert isinstance(result, float)
        assert result >= 0

    def test_decompress_vytvori_png(self, compressor, tmp_path):
        src = tmp_path / "test.qoi"
        src.write_bytes(b"\x00" * 50)
        out = tmp_path / "out.png"

        compressor.decompress(src, out)

        assert out.exists()
        img = Image.open(out)
        assert img.format == "PNG"
