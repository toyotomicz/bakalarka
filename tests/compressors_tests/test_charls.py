"""
Unit tests for CharLSCompressor, CharLSError, JpegLSError, InterleaveMode, FrameInfo, and _check().

The native DLL and all I/O are mocked; tests run without a real CharLS library.
"""

import sys
import ctypes
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from compressors.charls_compressor import (  # noqa: E402
    CharLSCompressor,
    CharLSError,
    JpegLSError,
    InterleaveMode,
    FrameInfo,
    _check,
    _library_name,
)


# Helpers

def _compressor_no_init() -> CharLSCompressor:
    """
    Instantiate CharLSCompressor without calling __init__.

    Skips ``_validate_dependencies``, allowing tests to inject a mock library
    directly via ``c._lib``.

    Returns:
        A bare CharLSCompressor instance with no attributes set.
    """
    return object.__new__(CharLSCompressor)


def _make_mock_lib() -> MagicMock:
    """
    Build a mock object that simulates the CharLS C API.

    All encoder and decoder entry points return success codes by default.
    Output-parameter functions use ``side_effect`` to write through the
    ctypes pointer so the compressor can read back sizes and frame metadata.

    Returns:
        A configured MagicMock representing the loaded CharLS shared library.
    """
    lib = MagicMock()

    lib.charls_jpegls_encoder_create.return_value = 0x1000  # non-NULL pointer
    lib.charls_jpegls_encoder_set_frame_info.return_value = JpegLSError.SUCCESS
    lib.charls_jpegls_encoder_set_near_lossless.return_value = JpegLSError.SUCCESS
    lib.charls_jpegls_encoder_set_interleave_mode.return_value = JpegLSError.SUCCESS
    lib.charls_jpegls_encoder_set_destination_buffer.return_value = JpegLSError.SUCCESS
    lib.charls_jpegls_encoder_encode_from_buffer.return_value = JpegLSError.SUCCESS
    lib.charls_jpegls_encoder_destroy.return_value = None

    def fake_get_estimated(encoder, size_ptr):
        size_ptr._obj.value = 200
        return JpegLSError.SUCCESS

    def fake_get_bytes_written(encoder, size_ptr):
        size_ptr._obj.value = 100
        return JpegLSError.SUCCESS

    lib.charls_jpegls_encoder_get_estimated_destination_size.side_effect = fake_get_estimated
    lib.charls_jpegls_encoder_get_bytes_written.side_effect = fake_get_bytes_written

    lib.charls_jpegls_decoder_create.return_value = 0x2000
    lib.charls_jpegls_decoder_set_source_buffer.return_value = JpegLSError.SUCCESS
    lib.charls_jpegls_decoder_read_header.return_value = JpegLSError.SUCCESS
    lib.charls_jpegls_decoder_decode_to_buffer.return_value = JpegLSError.SUCCESS
    lib.charls_jpegls_decoder_destroy.return_value = None

    def fake_get_frame_info(decoder, fi_ptr):
        fi_ptr._obj.width = 4
        fi_ptr._obj.height = 4
        fi_ptr._obj.bits_per_sample = 8
        fi_ptr._obj.component_count = 3
        return JpegLSError.SUCCESS

    lib.charls_jpegls_decoder_get_frame_info.side_effect = fake_get_frame_info

    return lib


# JpegLSError

class TestJpegLSError:
    """Verify that JpegLSError integer constants match the CharLS specification."""

    def test_success_is_zero(self):
        assert JpegLSError.SUCCESS == 0

    def test_invalid_argument_is_nonzero(self):
        assert JpegLSError.INVALID_ARGUMENT != 0

    def test_destination_buffer_too_small(self):
        assert JpegLSError.DESTINATION_BUFFER_TOO_SMALL == 10

    def test_source_buffer_too_small(self):
        assert JpegLSError.SOURCE_BUFFER_TOO_SMALL == 11

    def test_invalid_encoded_data(self):
        assert JpegLSError.INVALID_ENCODED_DATA == 12


# InterleaveMode

class TestInterleaveMode:
    """Verify that InterleaveMode constants follow the JPEG-LS standard ordering."""

    def test_none_is_zero(self):
        assert InterleaveMode.NONE == 0

    def test_line_is_one(self):
        assert InterleaveMode.LINE == 1

    def test_sample_is_two(self):
        assert InterleaveMode.SAMPLE == 2


# FrameInfo

class TestFrameInfo:
    """Verify that FrameInfo stores field values correctly as a ctypes structure."""

    def test_can_be_created(self):
        fi = FrameInfo(640, 480, 8, 3)
        assert fi.width == 640
        assert fi.height == 480
        assert fi.bits_per_sample == 8
        assert fi.component_count == 3

    def test_default_values_are_zero(self):
        fi = FrameInfo()
        assert fi.width == 0
        assert fi.height == 0


# _check()

class TestCheckHelper:
    """Verify that _check() raises CharLSError on failure and is silent on success."""

    def test_does_not_raise_on_success(self):
        _check(JpegLSError.SUCCESS, "operation")

    def test_raises_charls_error_on_failure(self):
        with pytest.raises(CharLSError, match="operation"):
            _check(JpegLSError.INVALID_ARGUMENT, "operation")

    def test_message_contains_error_code(self):
        with pytest.raises(CharLSError, match=str(JpegLSError.DESTINATION_BUFFER_TOO_SMALL)):
            _check(JpegLSError.DESTINATION_BUFFER_TOO_SMALL, "set_destination_buffer")


# _library_name()

class TestLibraryName:
    """Verify that _library_name() returns a platform-appropriate shared library name."""

    def test_returns_platform_specific_suffix(self):
        import platform
        name = _library_name()
        expected = {
            "Windows": ".dll",
            "Linux": ".so",
            "Darwin": ".dylib",
        }.get(platform.system())
        if expected:
            assert name.endswith(expected), (
                f"Expected suffix '{expected}' on {platform.system()}, got '{name}'"
            )

    def test_name_contains_charls(self):
        assert "charls" in _library_name().lower()


# CharLSCompressor properties

class TestCharLSProperties:
    """Verify the name and file extension reported by CharLSCompressor."""

    def test_name(self):
        c = _compressor_no_init()
        assert c.name == "CharLS-JPEGLS"

    def test_extension(self):
        c = _compressor_no_init()
        assert c.extension == ".jls"


# CharLSCompressor._validate_dependencies()

class TestCharLSValidateDependencies:
    """Verify that _validate_dependencies() raises when required files are absent."""

    def test_raises_when_lib_dir_missing(self, tmp_path):
        c = _compressor_no_init()
        c._lib = None

        with patch("compressors.charls_compressor.Path") as MockPath:
            mock_base = MagicMock()
            mock_charls_dir = MagicMock()
            mock_charls_dir.exists.return_value = False
            mock_base.parent.parent.__truediv__.return_value = mock_charls_dir
            MockPath.return_value = mock_base

            with pytest.raises((RuntimeError, Exception)):
                c._validate_dependencies()

    def test_raises_when_dll_missing(self, tmp_path):
        c = _compressor_no_init()
        c._lib = None

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

    def test_succeeds_when_all_present(self, tmp_path):
        """Validation should set _lib when the directory and DLL both exist."""
        charls_dir = tmp_path / "libs" / "charls"
        charls_dir.mkdir(parents=True)
        (charls_dir / _library_name()).touch()

        c = _compressor_no_init()
        c._lib = None

        with patch("compressors.charls_compressor.Path") as MockPath:
            mock_file = MagicMock()
            mock_file.parent.parent = tmp_path
            MockPath.return_value = mock_file

            with patch("compressors.charls_compressor.ctypes.CDLL", return_value=MagicMock()):
                c._validate_dependencies()

        assert c._lib is not None


# CharLSCompressor._encode()

class TestCharLSEncode:
    """Verify the JPEG-LS encoding path against the mock C library."""

    @pytest.fixture()
    def compressor(self):
        """
        Return a CharLSCompressor with a fully mocked C library.

        Returns:
            CharLSCompressor instance whose _lib attribute is a mock.
        """
        c = _compressor_no_init()
        c._lib = _make_mock_lib()
        return c

    def test_raises_for_invalid_dtype(self, compressor):
        bad_array = np.zeros((4, 4, 3), dtype=np.float32)
        with pytest.raises(ValueError, match="uint8"):
            compressor._encode(bad_array)

    def test_raises_charls_error_when_encoder_null(self):
        c = _compressor_no_init()
        c._lib = _make_mock_lib()
        c._lib.charls_jpegls_encoder_create.return_value = 0  # NULL

        with pytest.raises(CharLSError, match="NULL"):
            c._encode(np.zeros((4, 4, 3), dtype=np.uint8))

    def test_destroy_called_even_on_error(self):
        """The encoder handle must be released even when a subsequent call fails."""
        c = _compressor_no_init()
        c._lib = _make_mock_lib()
        c._lib.charls_jpegls_encoder_set_frame_info.return_value = JpegLSError.INVALID_ARGUMENT

        with pytest.raises(CharLSError):
            c._encode(np.zeros((4, 4, 3), dtype=np.uint8))

        c._lib.charls_jpegls_encoder_destroy.assert_called_once()

    def test_sets_interleave_mode_sample_for_rgb(self, compressor):
        compressor._encode(np.zeros((4, 4, 3), dtype=np.uint8))
        compressor._lib.charls_jpegls_encoder_set_interleave_mode.assert_called_once_with(
            compressor._lib.charls_jpegls_encoder_create.return_value,
            InterleaveMode.SAMPLE,
        )

    def test_does_not_set_interleave_mode_for_grayscale(self, compressor):
        compressor._encode(np.zeros((4, 4), dtype=np.uint8))
        compressor._lib.charls_jpegls_encoder_set_interleave_mode.assert_not_called()

    def test_sets_near_lossless_to_zero(self, compressor):
        compressor._encode(np.zeros((4, 4, 3), dtype=np.uint8))
        compressor._lib.charls_jpegls_encoder_set_near_lossless.assert_called_once_with(
            compressor._lib.charls_jpegls_encoder_create.return_value, 0
        )

    def test_uint16_sets_bpp_16(self, compressor):
        """FrameInfo passed to the encoder must report 16 bpp for uint16 input."""
        captured: list[FrameInfo] = []

        def capture_frame_info(encoder, fi_byref):
            captured.append(fi_byref._obj)
            return JpegLSError.SUCCESS

        compressor._lib.charls_jpegls_encoder_set_frame_info.side_effect = capture_frame_info

        compressor._encode(np.zeros((4, 4, 3), dtype=np.uint16))

        assert len(captured) == 1
        assert captured[0].bits_per_sample == 16


# CharLSCompressor._decode()

class TestCharLSDecode:
    """Verify the JPEG-LS decoding path against the mock C library."""

    @pytest.fixture()
    def compressor(self):
        """
        Return a CharLSCompressor with a fully mocked C library.

        Returns:
            CharLSCompressor instance whose _lib attribute is a mock.
        """
        c = _compressor_no_init()
        c._lib = _make_mock_lib()
        return c

    def test_raises_charls_error_when_decoder_null(self):
        c = _compressor_no_init()
        c._lib = _make_mock_lib()
        c._lib.charls_jpegls_decoder_create.return_value = 0  # NULL

        with pytest.raises(CharLSError, match="NULL"):
            c._decode(b"\x00" * 50)

    def test_destroy_called_even_on_error(self):
        """The decoder handle must be released even when a subsequent call fails."""
        c = _compressor_no_init()
        c._lib = _make_mock_lib()
        c._lib.charls_jpegls_decoder_set_source_buffer.return_value = JpegLSError.INVALID_ARGUMENT

        with pytest.raises(CharLSError):
            c._decode(b"\x00" * 50)

        c._lib.charls_jpegls_decoder_destroy.assert_called_once()

    def test_returns_uint16_array_for_bpp_16(self):
        c = _compressor_no_init()
        c._lib = _make_mock_lib()

        def fake_get_frame_info_16(decoder, fi_ptr):
            fi_ptr._obj.width = 4
            fi_ptr._obj.height = 4
            fi_ptr._obj.bits_per_sample = 16
            fi_ptr._obj.component_count = 1
            return JpegLSError.SUCCESS

        c._lib.charls_jpegls_decoder_get_frame_info.side_effect = fake_get_frame_info_16

        result = c._decode(b"\x00" * 50)
        assert result.dtype == np.uint16

    def test_returns_uint8_array_for_bpp_8(self, compressor):
        result = compressor._decode(b"\x00" * 50)
        assert result.dtype == np.uint8

    def test_rgb_output_shape(self, compressor):
        result = compressor._decode(b"\x00" * 50)
        assert result.shape == (4, 4, 3)


# CharLSCompressor.compress()

class TestCharLSCompress:
    """Verify the public compress() API including error handling and temp-file cleanup."""

    @pytest.fixture()
    def compressor(self):
        """
        Return a CharLSCompressor with a fully mocked C library.

        Returns:
            CharLSCompressor instance whose _lib attribute is a mock.
        """
        c = _compressor_no_init()
        c._lib = _make_mock_lib()
        return c

    def test_failure_for_missing_input(self, compressor, tmp_path):
        metrics = compressor.compress(tmp_path / "nope.png", tmp_path / "out.jls")
        assert metrics.success is False

    def test_success_with_stubbed_encode(self, compressor, tmp_path):
        src = tmp_path / "src.png"
        Image.new("RGB", (4, 4)).save(src, format="PNG")
        out = tmp_path / "out.jls"

        with patch.object(compressor, "_encode", return_value=b"\x00" * 200):
            with patch.object(compressor, "decompress", return_value=0.001):
                metrics = compressor.compress(src, out)

        assert metrics.success is True
        assert out.exists()
        assert metrics.original_size > 0

    def test_failure_when_encode_raises(self, compressor, tmp_path):
        src = tmp_path / "src.png"
        Image.new("RGB", (4, 4)).save(src, format="PNG")
        out = tmp_path / "out.jls"

        with patch.object(compressor, "_encode", side_effect=CharLSError("encode fail")):
            metrics = compressor.compress(src, out)

        assert metrics.success is False
        assert "encode fail" in metrics.error_message

    def test_temp_decomp_file_cleaned_up(self, compressor, tmp_path):
        src = tmp_path / "src.png"
        Image.new("RGB", (4, 4)).save(src, format="PNG")
        out = tmp_path / "out.jls"

        with patch.object(compressor, "_encode", return_value=b"\x00" * 200):
            with patch.object(compressor, "decompress", return_value=0.0):
                compressor.compress(src, out)

        assert not any(tmp_path.glob("temp_decomp_*.png"))

    def test_rgba_input_converted_to_rgb(self, compressor, tmp_path):
        """RGBA source images must be flattened to RGB before JPEG-LS encoding."""
        src = tmp_path / "src.png"
        Image.new("RGBA", (4, 4)).save(src, format="PNG")
        out = tmp_path / "out.jls"

        with patch.object(compressor, "_encode", return_value=b"\x00" * 200) as mock_encode:
            with patch.object(compressor, "decompress", return_value=0.0):
                compressor.compress(src, out)

        array_arg = mock_encode.call_args[0][0]
        assert array_arg.ndim == 3
        assert array_arg.shape[2] == 3


# CharLSCompressor.decompress()

class TestCharLSDecompress:
    """Verify the public decompress() API including output file creation."""

    @pytest.fixture()
    def compressor(self):
        """
        Return a CharLSCompressor with a fully mocked C library.

        Returns:
            CharLSCompressor instance whose _lib attribute is a mock.
        """
        c = _compressor_no_init()
        c._lib = _make_mock_lib()
        return c

    def test_returns_float(self, compressor, tmp_path):
        src = tmp_path / "src.jls"
        out = tmp_path / "out.png"
        src.write_bytes(b"\x00" * 50)

        with patch.object(compressor, "_decode", return_value=np.zeros((4, 4, 3), dtype=np.uint8)):
            result = compressor.decompress(src, out)

        assert isinstance(result, float)
        assert result >= 0

    def test_output_png_is_created(self, compressor, tmp_path):
        src = tmp_path / "src.jls"
        out = tmp_path / "out.png"
        src.write_bytes(b"\x00" * 50)

        with patch.object(compressor, "_decode", return_value=np.zeros((4, 4, 3), dtype=np.uint8)):
            compressor.decompress(src, out)

        assert out.exists()
        assert Image.open(out).format == "PNG"

    def test_decode_called_with_file_bytes(self, compressor, tmp_path):
        data = b"\xAB\xCD" * 25
        src = tmp_path / "src.jls"
        out = tmp_path / "out.png"
        src.write_bytes(data)

        with patch.object(
            compressor, "_decode", return_value=np.zeros((4, 4, 3), dtype=np.uint8)
        ) as mock_decode:
            compressor.decompress(src, out)

        mock_decode.assert_called_once_with(data)