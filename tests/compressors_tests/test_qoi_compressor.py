"""
Unit tests for QOICompressor.

The qoi Python package and file I/O are mocked.
Stubs for main and image_size_calculator are provided by conftest.py.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

CompressionLevel = sys.modules["main"].CompressionLevel


# Fixtures

@pytest.fixture()
def mock_qoi_module() -> MagicMock:
    """
    Return a mock ``qoi`` module with encode/decode methods.

    Returns:
        MagicMock whose encode() returns fake QOI bytes and decode() returns
        a zeroed 4x4 RGB array.
    """
    qoi_mock = MagicMock()
    qoi_mock.encode.return_value = b"\x71\x6f\x69\x66" + b"\x00" * 100
    qoi_mock.decode.return_value = np.zeros((4, 4, 3), dtype=np.uint8)
    return qoi_mock


@pytest.fixture()
def compressor(mock_qoi_module) -> "QOICompressor":
    """
    Return a QOICompressor with a mocked qoi module.

    The module is patched in sys.modules so that the module-level
    ``import qoi`` inside the compressor picks up the mock.

    Args:
        mock_qoi_module: Pre-configured mock for the qoi package.

    Returns:
        QOICompressor instance whose _qoi attribute is the mock module.
    """
    with patch.dict("sys.modules", {"qoi": mock_qoi_module}):
        if "compressors.qoi_compressor" in sys.modules:
            del sys.modules["compressors.qoi_compressor"]
        from compressors.qoi_compressor import QOICompressor
        c = QOICompressor()
    c._qoi = mock_qoi_module
    return c


# _validate_dependencies()

class TestQOIValidateDependencies:
    """Verify that _validate_dependencies() raises when the qoi package is absent."""

    def test_raises_when_package_missing(self):
        """_validate_dependencies() must raise RuntimeError when qoi is not importable.

        The qoi module is removed from sys.modules so that the internal
        ``import qoi`` actually fails with ImportError.
        """
        from compressors.qoi_compressor import QOICompressor
        c = object.__new__(QOICompressor)
        c._qoi = None

        with patch.dict("sys.modules", {"qoi": None}):
            with pytest.raises(RuntimeError, match="qoi"):
                c._validate_dependencies()

    def test_sets_self_qoi(self, mock_qoi_module):
        """_validate_dependencies() must assign the imported module to self._qoi."""
        with patch.dict("sys.modules", {"qoi": mock_qoi_module}):
            if "compressors.qoi_compressor" in sys.modules:
                del sys.modules["compressors.qoi_compressor"]
            from compressors.qoi_compressor import QOICompressor

            c = object.__new__(QOICompressor)
            c._qoi = None
            c._validate_dependencies()
            assert c._qoi is mock_qoi_module


# Properties

class TestQOIProperties:
    """Verify the name and file extension reported by QOICompressor."""

    def test_name(self, compressor):
        assert compressor.name == "QOI"

    def test_extension(self, compressor):
        assert compressor.extension == ".qoi"


# compress()

class TestQOICompress:
    """Verify compress() encoding behaviour, metrics, and cleanup."""

    def test_encode_called_with_uint8_array_of_correct_shape(self, compressor, tmp_path):
        src = tmp_path / "src.png"
        Image.new("RGB", (4, 4)).save(src, format="PNG")

        with patch.object(compressor, "decompress", return_value=0.001):
            compressor.compress(src, tmp_path / "out.qoi")

        assert compressor._qoi.encode.called
        arr = compressor._qoi.encode.call_args[0][0]
        assert arr.dtype == np.uint8
        assert arr.ndim == 3
        assert arr.shape[2] in (3, 4), "encode must receive an RGB or RGBA array"

    def test_encoded_bytes_written_to_output_file(self, compressor, tmp_path):
        src = tmp_path / "src.png"
        out = tmp_path / "out.qoi"
        Image.new("RGB", (4, 4)).save(src, format="PNG")

        with patch.object(compressor, "decompress", return_value=0.001):
            compressor.compress(src, out)

        assert out.read_bytes() == compressor._qoi.encode.return_value

    def test_success_metrics(self, compressor, tmp_path):
        src = tmp_path / "src.png"
        Image.new("RGB", (4, 4)).save(src, format="PNG")

        with patch.object(compressor, "decompress", return_value=0.002):
            metrics = compressor.compress(src, tmp_path / "out.qoi")

        assert metrics.success is True
        assert metrics.original_size > 0
        assert metrics.compressed_size > 0
        assert metrics.compression_ratio > 0
        assert metrics.decompression_time == 0.002

    def test_palette_mode_converted_to_rgb(self, compressor, tmp_path):
        """Palette (P) mode must be converted to RGB before QOI encoding."""
        src = tmp_path / "src.png"
        Image.new("P", (4, 4)).save(src, format="PNG")

        with patch.object(compressor, "decompress", return_value=0.0):
            metrics = compressor.compress(src, tmp_path / "out.qoi")

        assert metrics.success is True
        arr = compressor._qoi.encode.call_args[0][0]
        assert arr.ndim == 3
        assert arr.shape[2] == 3

    def test_rgba_source_encodes_as_four_channel(self, compressor, tmp_path):
        """RGBA images must be encoded with 4 channels because QOI supports RGBA natively."""
        src = tmp_path / "src.png"
        Image.new("RGBA", (4, 4), color=(10, 20, 30, 200)).save(src, format="PNG")

        with patch.object(compressor, "decompress", return_value=0.0):
            metrics = compressor.compress(src, tmp_path / "out.qoi")

        assert metrics.success is True
        arr = compressor._qoi.encode.call_args[0][0]
        assert arr.shape[2] == 4

    def test_failure_when_encode_raises(self, compressor, tmp_path):
        compressor._qoi.encode.side_effect = RuntimeError("encode failed")
        src = tmp_path / "src.png"
        Image.new("RGB", (4, 4)).save(src, format="PNG")

        metrics = compressor.compress(src, tmp_path / "out.qoi")

        assert metrics.success is False
        assert "encode failed" in metrics.error_message

    def test_failure_for_missing_input(self, compressor, tmp_path):
        metrics = compressor.compress(tmp_path / "missing.png", tmp_path / "out.qoi")
        assert metrics.success is False

    def test_temp_file_cleaned_up_on_success(self, compressor, tmp_path):
        src = tmp_path / "src.png"
        Image.new("RGB", (4, 4)).save(src, format="PNG")

        with patch.object(compressor, "decompress", return_value=0.0):
            compressor.compress(src, tmp_path / "out.qoi")

        assert list(tmp_path.glob("temp_decomp_*.png")) == []

    def test_temp_file_cleaned_up_on_decompress_error(self, compressor, tmp_path):
        src = tmp_path / "src.png"
        Image.new("RGB", (4, 4)).save(src, format="PNG")

        with patch.object(compressor, "decompress", side_effect=RuntimeError("failure")):
            metrics = compressor.compress(src, tmp_path / "out.qoi")

        assert metrics.success is False
        assert list(tmp_path.glob("temp_decomp_*.png")) == []

    def test_zero_compressed_size_yields_zero_ratio_without_exception(self, compressor, tmp_path):
        """If encode() returns empty bytes, compression_ratio must be 0, not ZeroDivisionError."""
        compressor._qoi.encode.return_value = b""
        src = tmp_path / "src.png"
        Image.new("RGB", (4, 4)).save(src, format="PNG")

        with patch.object(compressor, "decompress", return_value=0.0):
            metrics = compressor.compress(src, tmp_path / "out.qoi")

        assert metrics.compression_ratio == 0


# decompress()

class TestQOIDecompress:
    """Verify decompress() reads the source file and produces a valid PNG output."""

    def test_decode_called_with_file_bytes(self, compressor, tmp_path):
        data = b"\x71\x6f\x69\x66" + b"\x00" * 50
        src = tmp_path / "test.qoi"
        src.write_bytes(data)

        compressor.decompress(src, tmp_path / "out.png")

        compressor._qoi.decode.assert_called_once_with(data)

    def test_returns_float(self, compressor, tmp_path):
        src = tmp_path / "test.qoi"
        src.write_bytes(b"\x00" * 50)

        result = compressor.decompress(src, tmp_path / "out.png")

        assert isinstance(result, float)
        assert result >= 0

    def test_output_png_created_and_valid(self, compressor, tmp_path):
        src = tmp_path / "test.qoi"
        out = tmp_path / "out.png"
        src.write_bytes(b"\x00" * 50)

        compressor.decompress(src, out)

        assert out.exists()
        assert Image.open(out).format == "PNG"