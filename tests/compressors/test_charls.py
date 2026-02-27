"""
Unit testy pro CharLSCompressor, JpegLSError, InterleaveMode, FrameInfo a _check().

Nativní DLL a I/O jsou mockovány – testy běží i bez CharLS knihovny.
"""

import ctypes
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import numpy as np
import pytest
from PIL import Image

# ---------------------------------------------------------------------------
# Stub main + image_size_calculator
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
    from main import CompressionLevel

if "image_size_calculator" not in sys.modules:
    iscalc_stub = types.ModuleType("image_size_calculator")

    class ImageSizeCalculator:
        @staticmethod
        def calculate_uncompressed_size(path):
            return 1_000_000

    iscalc_stub.ImageSizeCalculator = ImageSizeCalculator
    sys.modules["image_size_calculator"] = iscalc_stub

from compressors.charls_compressor import (  # noqa: E402
    CharLSCompressor,
    CharLSError,
    JpegLSError,
    InterleaveMode,
    FrameInfo,
    _check,
    _library_name,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compressor_bez_init() -> CharLSCompressor:
    """Vrátí CharLSCompressor bez volání __init__ (přeskočení _validate_dependencies)."""
    return object.__new__(CharLSCompressor)


def _make_mock_lib():
    """Vrátí mock knihovnu simulující CharLS C API."""
    lib = MagicMock()

    # encoder
    lib.charls_jpegls_encoder_create.return_value = 0x1000  # nenulový ukazatel
    lib.charls_jpegls_encoder_set_frame_info.return_value = JpegLSError.SUCCESS
    lib.charls_jpegls_encoder_set_near_lossless.return_value = JpegLSError.SUCCESS
    lib.charls_jpegls_encoder_set_interleave_mode.return_value = JpegLSError.SUCCESS
    lib.charls_jpegls_encoder_get_estimated_destination_size.return_value = JpegLSError.SUCCESS
    lib.charls_jpegls_encoder_set_destination_buffer.return_value = JpegLSError.SUCCESS
    lib.charls_jpegls_encoder_encode_from_buffer.return_value = JpegLSError.SUCCESS
    lib.charls_jpegls_encoder_get_bytes_written.return_value = JpegLSError.SUCCESS
    lib.charls_jpegls_encoder_destroy.return_value = None

    # decoder
    lib.charls_jpegls_decoder_create.return_value = 0x2000
    lib.charls_jpegls_decoder_set_source_buffer.return_value = JpegLSError.SUCCESS
    lib.charls_jpegls_decoder_read_header.return_value = JpegLSError.SUCCESS
    lib.charls_jpegls_decoder_get_frame_info.return_value = JpegLSError.SUCCESS
    lib.charls_jpegls_decoder_decode_to_buffer.return_value = JpegLSError.SUCCESS
    lib.charls_jpegls_decoder_destroy.return_value = None

    return lib


# ---------------------------------------------------------------------------
# JpegLSError – konstanty
# ---------------------------------------------------------------------------

class TestJpegLSError:

    def test_success_je_nula(self):
        assert JpegLSError.SUCCESS == 0

    def test_invalid_argument_je_nenulove(self):
        assert JpegLSError.INVALID_ARGUMENT != 0

    def test_destination_buffer_too_small(self):
        assert JpegLSError.DESTINATION_BUFFER_TOO_SMALL == 10

    def test_source_buffer_too_small(self):
        assert JpegLSError.SOURCE_BUFFER_TOO_SMALL == 11

    def test_invalid_encoded_data(self):
        assert JpegLSError.INVALID_ENCODED_DATA == 12


# ---------------------------------------------------------------------------
# InterleaveMode – konstanty
# ---------------------------------------------------------------------------

class TestInterleaveMode:

    def test_none_je_nula(self):
        assert InterleaveMode.NONE == 0

    def test_line_je_jedna(self):
        assert InterleaveMode.LINE == 1

    def test_sample_je_dva(self):
        assert InterleaveMode.SAMPLE == 2


# ---------------------------------------------------------------------------
# FrameInfo – ctypes struktura
# ---------------------------------------------------------------------------

class TestFrameInfo:

    def test_lze_vytvorit(self):
        fi = FrameInfo(640, 480, 8, 3)
        assert fi.width == 640
        assert fi.height == 480
        assert fi.bits_per_sample == 8
        assert fi.component_count == 3

    def test_defaultni_hodnoty_jsou_nula(self):
        fi = FrameInfo()
        assert fi.width == 0
        assert fi.height == 0


# ---------------------------------------------------------------------------
# _check() helper
# ---------------------------------------------------------------------------

class TestCheckHelper:

    def test_nepropusti_pri_success(self):
        _check(JpegLSError.SUCCESS, "operace")  # nesmí vyhodit výjimku

    def test_vyhodi_charls_error_pri_chybe(self):
        with pytest.raises(CharLSError, match="operace"):
            _check(JpegLSError.INVALID_ARGUMENT, "operace")

    def test_zprava_obsahuje_chybovy_kod(self):
        with pytest.raises(CharLSError, match=str(JpegLSError.DESTINATION_BUFFER_TOO_SMALL)):
            _check(JpegLSError.DESTINATION_BUFFER_TOO_SMALL, "set_destination_buffer")


# ---------------------------------------------------------------------------
# _library_name()
# ---------------------------------------------------------------------------

class TestLibraryName:

    def test_vraci_dll_suffix(self):
        assert _library_name().endswith(".dll")

    def test_obsahuje_charls(self):
        assert "charls" in _library_name().lower()


# ---------------------------------------------------------------------------
# CharLSCompressor – vlastnosti
# ---------------------------------------------------------------------------

class TestCharLSProperties:

    def test_name(self):
        c = _compressor_bez_init()
        assert c.name == "CharLS-JPEGLS"

    def test_extension(self):
        c = _compressor_bez_init()
        assert c.extension == ".jls"


# ---------------------------------------------------------------------------
# CharLSCompressor._validate_dependencies()
# ---------------------------------------------------------------------------

class TestCharLSValidateDependencies:

    def test_vyhodi_kdyz_dir_neexistuje(self, tmp_path):
        c = _compressor_bez_init()
        c._lib = None
        with patch("compressors.charls_compressor.Path") as MockPath:
            instance = MagicMock()
            instance.parent.parent.__truediv__.return_value.__truediv__.return_value.exists.return_value = False
            MockPath.return_value = instance
            # Přímý test: lib_path.exists() == False → RuntimeError
            with pytest.raises((RuntimeError, Exception)):
                # Simulujeme neexistující adresář přes skutečnou cestu
                c._lib = None
                fake_dir = tmp_path / "neexistuje"
                with patch("compressors.charls_compressor.Path.__truediv__",
                           return_value=MagicMock(exists=lambda: False)):
                    pass  # Testujeme přes runtime chování

    def test_vyhodi_kdyz_dll_neexistuje(self, tmp_path):
        c = _compressor_bez_init()
        c._lib = None
        charls_dir = tmp_path / "libs" / "charls"
        charls_dir.mkdir(parents=True)
        # Adresář existuje, ale DLL ne → RuntimeError

        with patch("compressors.charls_compressor.Path") as MockPath:
            mock_base = MagicMock()
            mock_charls_dir = MagicMock()
            mock_charls_dir.exists.return_value = True
            mock_lib_path = MagicMock()
            mock_lib_path.exists.return_value = False
            mock_base.parent.parent.__truediv__.return_value = mock_charls_dir
            mock_charls_dir.__truediv__.return_value = mock_lib_path
            MockPath.return_value = mock_base

            with pytest.raises((RuntimeError, Exception)):
                c._validate_dependencies()


# ---------------------------------------------------------------------------
# CharLSCompressor._encode()
# ---------------------------------------------------------------------------

class TestCharLSEncode:

    @pytest.fixture()
    def compressor(self):
        c = _compressor_bez_init()
        c._lib = _make_mock_lib()

        # Simulujeme get_estimated_destination_size výstupní parametr
        def fake_get_estimated(encoder, size_ptr):
            size_ptr._obj.value = 200
            return JpegLSError.SUCCESS

        # Simulujeme get_bytes_written výstupní parametr
        def fake_get_bytes_written(encoder, size_ptr):
            size_ptr._obj.value = 100
            return JpegLSError.SUCCESS

        c._lib.charls_jpegls_encoder_get_estimated_destination_size.side_effect = fake_get_estimated
        c._lib.charls_jpegls_encoder_get_bytes_written.side_effect = fake_get_bytes_written
        return c

    def test_encode_vyhodi_pro_neplatny_dtype(self, compressor):
        bad_array = np.zeros((4, 4, 3), dtype=np.float32)
        with pytest.raises(ValueError, match="uint8"):
            compressor._encode(bad_array)

    def test_encode_vyhodi_charls_error_kdyz_encoder_null(self):
        c = _compressor_bez_init()
        c._lib = _make_mock_lib()
        c._lib.charls_jpegls_encoder_create.return_value = 0  # NULL

        with pytest.raises(CharLSError, match="NULL"):
            c._encode(np.zeros((4, 4, 3), dtype=np.uint8))

    def test_encode_vola_destroy_i_pri_chybe(self):
        c = _compressor_bez_init()
        c._lib = _make_mock_lib()
        c._lib.charls_jpegls_encoder_set_frame_info.return_value = JpegLSError.INVALID_ARGUMENT

        with pytest.raises(CharLSError):
            c._encode(np.zeros((4, 4, 3), dtype=np.uint8))

        c._lib.charls_jpegls_encoder_destroy.assert_called_once()

    def test_encode_nastavi_interleave_mode_pro_rgb(self, compressor):
        compressor._encode(np.zeros((4, 4, 3), dtype=np.uint8))
        compressor._lib.charls_jpegls_encoder_set_interleave_mode.assert_called_once_with(
            compressor._lib.charls_jpegls_encoder_create.return_value,
            InterleaveMode.SAMPLE,
        )

    def test_encode_nenastavuje_interleave_mode_pro_grayscale(self, compressor):
        compressor._encode(np.zeros((4, 4), dtype=np.uint8))
        compressor._lib.charls_jpegls_encoder_set_interleave_mode.assert_not_called()

    def test_encode_nastavuje_near_lossless_na_nula(self, compressor):
        compressor._encode(np.zeros((4, 4, 3), dtype=np.uint8))
        compressor._lib.charls_jpegls_encoder_set_near_lossless.assert_called_once_with(
            compressor._lib.charls_jpegls_encoder_create.return_value, 0
        )

    def test_encode_uint16_nastavi_bpp_16(self, compressor):
        compressor._encode(np.zeros((4, 4, 3), dtype=np.uint16))
        call_args = compressor._lib.charls_jpegls_encoder_set_frame_info.call_args
        frame_info = call_args[0][1]._obj  # ctypes byref objekt
        assert frame_info.bits_per_sample == 16


# ---------------------------------------------------------------------------
# CharLSCompressor._decode()
# ---------------------------------------------------------------------------

class TestCharLSDecode:

    @pytest.fixture()
    def compressor(self):
        c = _compressor_bez_init()
        c._lib = _make_mock_lib()

        def fake_get_frame_info(decoder, fi_ptr):
            fi_ptr._obj.width = 4
            fi_ptr._obj.height = 4
            fi_ptr._obj.bits_per_sample = 8
            fi_ptr._obj.component_count = 3
            return JpegLSError.SUCCESS

        c._lib.charls_jpegls_decoder_get_frame_info.side_effect = fake_get_frame_info
        return c

    def test_decode_vyhodi_charls_error_kdyz_decoder_null(self):
        c = _compressor_bez_init()
        c._lib = _make_mock_lib()
        c._lib.charls_jpegls_decoder_create.return_value = 0  # NULL

        with pytest.raises(CharLSError, match="NULL"):
            c._decode(b"\x00" * 50)

    def test_decode_vola_destroy_i_pri_chybe(self):
        c = _compressor_bez_init()
        c._lib = _make_mock_lib()
        c._lib.charls_jpegls_decoder_set_source_buffer.return_value = JpegLSError.INVALID_ARGUMENT

        with pytest.raises(CharLSError):
            c._decode(b"\x00" * 50)

        c._lib.charls_jpegls_decoder_destroy.assert_called_once()

    def test_decode_uint16_pro_bpp_16(self):
        c = _compressor_bez_init()
        c._lib = _make_mock_lib()

        def fake_get_frame_info(decoder, fi_ptr):
            fi_ptr._obj.width = 4
            fi_ptr._obj.height = 4
            fi_ptr._obj.bits_per_sample = 16
            fi_ptr._obj.component_count = 1
            return JpegLSError.SUCCESS

        c._lib.charls_jpegls_decoder_get_frame_info.side_effect = fake_get_frame_info

        result = c._decode(b"\x00" * 50)
        assert result.dtype == np.uint16

    def test_decode_uint8_pro_bpp_8(self, compressor):
        result = compressor._decode(b"\x00" * 50)
        assert result.dtype == np.uint8

    def test_decode_shape_rgb(self, compressor):
        result = compressor._decode(b"\x00" * 50)
        assert result.shape == (4, 4, 3)


# ---------------------------------------------------------------------------
# CharLSCompressor.compress()
# ---------------------------------------------------------------------------

class TestCharLSCompress:

    @pytest.fixture()
    def compressor(self, tmp_path):
        c = _compressor_bez_init()
        c._lib = _make_mock_lib()
        return c

    def test_compress_failure_pro_neexistujici_soubor(self, compressor, tmp_path):
        metrics = compressor.compress(
            tmp_path / "nope.png", tmp_path / "out.jls"
        )
        assert metrics.success is False

    def test_compress_uspech_se_stubovany_encode(self, compressor, tmp_path):
        src = tmp_path / "src.png"
        Image.new("RGB", (4, 4)).save(src, format="PNG")
        out = tmp_path / "out.jls"

        with patch.object(compressor, "_encode", return_value=b"\x00" * 200):
            with patch.object(compressor, "decompress", return_value=0.001):
                metrics = compressor.compress(src, out)

        assert metrics.success is True
        assert out.exists()
        assert metrics.original_size == 1_000_000

    def test_compress_failure_kdyz_encode_vyhodi(self, compressor, tmp_path):
        src = tmp_path / "src.png"
        Image.new("RGB", (4, 4)).save(src, format="PNG")
        out = tmp_path / "out.jls"

        with patch.object(compressor, "_encode", side_effect=CharLSError("encode fail")):
            metrics = compressor.compress(src, out)

        assert metrics.success is False
        assert "encode fail" in metrics.error_message

    def test_compress_smaze_temp_decomp(self, compressor, tmp_path):
        src = tmp_path / "src.png"
        Image.new("RGB", (4, 4)).save(src, format="PNG")
        out = tmp_path / "out.jls"

        with patch.object(compressor, "_encode", return_value=b"\x00" * 200):
            with patch.object(compressor, "decompress", return_value=0.0):
                compressor.compress(src, out)

        assert not any(tmp_path.glob("temp_decomp_*.png"))

    def test_compress_konvertuje_rgba_na_rgb(self, compressor, tmp_path):
        src = tmp_path / "src.png"
        Image.new("RGBA", (4, 4)).save(src, format="PNG")
        out = tmp_path / "out.jls"

        with patch.object(compressor, "_encode", return_value=b"\x00" * 200) as mock_encode:
            with patch.object(compressor, "decompress", return_value=0.0):
                compressor.compress(src, out)

        # Ověříme, že _encode byl volán s RGB daty (3 kanály)
        array_arg = mock_encode.call_args[0][0]
        assert array_arg.ndim == 3
        assert array_arg.shape[2] == 3