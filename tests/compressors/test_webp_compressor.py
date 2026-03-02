"""
Unit tests for WebPCompressor.

cwebp/dwebp subprocess calls are mocked.
Stubs for main and image_size_calculator are provided by conftest.py.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

CompressionLevel = sys.modules["main"].CompressionLevel

from compressors.webp_compressor import WebPCompressor  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ok() -> MagicMock:
    r = MagicMock()
    r.returncode = 0
    r.stderr = ""
    return r


def _fail(code: int = 1, stderr: str = "webp error") -> MagicMock:
    r = MagicMock()
    r.returncode = code
    r.stderr = stderr
    return r


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def fake_bin_dir(tmp_path) -> Path:
    bin_dir = tmp_path / "libs" / "webp"
    bin_dir.mkdir(parents=True)
    (bin_dir / "cwebp.exe").touch()
    (bin_dir / "dwebp.exe").touch()
    return bin_dir


@pytest.fixture()
def compressor(fake_bin_dir) -> WebPCompressor:
    c = object.__new__(WebPCompressor)
    c._bin_dir = fake_bin_dir
    return c


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------

class TestWebPProperties:

    def test_name(self, compressor):
        assert compressor.name == "WebP-Lossless"

    def test_extension(self, compressor):
        assert compressor.extension == ".webp"


# ---------------------------------------------------------------------------
# _validate_dependencies
# ---------------------------------------------------------------------------

class TestWebPValidateDependencies:

    def test_raises_when_bin_dir_missing(self, tmp_path):
        # tmp_path / "libs" / "webp" does NOT exist → should raise
        c = object.__new__(WebPCompressor)
        c._bin_dir = None

        with patch("compressors.webp_compressor.Path") as MockPath:
            mock_file = MagicMock()
            mock_file.parent.parent = tmp_path
            MockPath.return_value = mock_file

            with pytest.raises((RuntimeError, Exception)):
                c._validate_dependencies()

    def test_raises_when_cwebp_missing(self, tmp_path):
        # bin_dir exists but cwebp.exe is absent
        bin_dir = tmp_path / "libs" / "webp"
        bin_dir.mkdir(parents=True)
        (bin_dir / "dwebp.exe").touch()

        c = object.__new__(WebPCompressor)
        c._bin_dir = None

        with patch("compressors.webp_compressor.Path") as MockPath:
            mock_file = MagicMock()
            mock_file.parent.parent = tmp_path
            MockPath.return_value = mock_file

            with pytest.raises((RuntimeError, Exception)):
                c._validate_dependencies()

    def test_raises_when_dwebp_missing(self, tmp_path):
        # bin_dir exists but dwebp.exe is absent
        bin_dir = tmp_path / "libs" / "webp"
        bin_dir.mkdir(parents=True)
        (bin_dir / "cwebp.exe").touch()

        c = object.__new__(WebPCompressor)
        c._bin_dir = None

        with patch("compressors.webp_compressor.Path") as MockPath:
            mock_file = MagicMock()
            mock_file.parent.parent = tmp_path
            MockPath.return_value = mock_file

            with pytest.raises((RuntimeError, Exception)):
                c._validate_dependencies()


# ---------------------------------------------------------------------------
# _run_cwebp
# ---------------------------------------------------------------------------

class TestWebPRunCwebp:

    @pytest.mark.parametrize("level,expected_z", [
        (CompressionLevel.FASTEST, "0"),
        (CompressionLevel.BALANCED, "6"),
        (CompressionLevel.BEST, "9"),
    ])
    def test_z_flag_for_level(self, compressor, level, expected_z):
        with patch("subprocess.run", return_value=_ok()) as mock_run:
            compressor._run_cwebp(Path("in.png"), Path("out.webp"), level)

        cmd = mock_run.call_args[0][0]
        assert cmd[cmd.index("-z") + 1] == expected_z

    def test_z_levels_are_monotonically_ordered(self, compressor):
        levels = [CompressionLevel.FASTEST, CompressionLevel.BALANCED, CompressionLevel.BEST]
        z_values = []
        for level in levels:
            with patch("subprocess.run", return_value=_ok()) as mock_run:
                compressor._run_cwebp(Path("in.png"), Path("out.webp"), level)
            cmd = mock_run.call_args[0][0]
            z_values.append(int(cmd[cmd.index("-z") + 1]))
        assert z_values == sorted(z_values)

    def test_command_contains_lossless_flag(self, compressor):
        with patch("subprocess.run", return_value=_ok()) as mock_run:
            compressor._run_cwebp(Path("in.png"), Path("out.webp"), CompressionLevel.BALANCED)
        assert "-lossless" in mock_run.call_args[0][0]

    def test_command_contains_exact_flag(self, compressor):
        with patch("subprocess.run", return_value=_ok()) as mock_run:
            compressor._run_cwebp(Path("in.png"), Path("out.webp"), CompressionLevel.BALANCED)
        assert "-exact" in mock_run.call_args[0][0]

    def test_command_contains_alpha_q_100(self, compressor):
        with patch("subprocess.run", return_value=_ok()) as mock_run:
            compressor._run_cwebp(Path("in.png"), Path("out.webp"), CompressionLevel.BALANCED)

        cmd = mock_run.call_args[0][0]
        assert "-alpha_q" in cmd
        assert cmd[cmd.index("-alpha_q") + 1] == "100"

    def test_command_contains_output_flag(self, compressor, tmp_path):
        out = tmp_path / "result.webp"
        with patch("subprocess.run", return_value=_ok()) as mock_run:
            compressor._run_cwebp(Path("in.png"), out, CompressionLevel.BALANCED)

        cmd = mock_run.call_args[0][0]
        assert cmd[cmd.index("-o") + 1] == str(out)

    def test_raises_runtime_error_on_nonzero_return_code(self, compressor):
        with patch("subprocess.run", return_value=_fail(1, "cwebp failed")):
            with pytest.raises(RuntimeError, match="cwebp failed"):
                compressor._run_cwebp(Path("in.png"), Path("out.webp"), CompressionLevel.BALANCED)

    def test_balanced_and_best_use_q_100(self, compressor):
        for level in (CompressionLevel.BALANCED, CompressionLevel.BEST):
            with patch("subprocess.run", return_value=_ok()) as mock_run:
                compressor._run_cwebp(Path("in.png"), Path("out.webp"), level)
            cmd = mock_run.call_args[0][0]
            assert cmd[cmd.index("-q") + 1] == "100", f"-q must be 100 for level={level}"

    def test_m_flag_within_valid_range(self, compressor):
        for level in (CompressionLevel.FASTEST, CompressionLevel.BALANCED, CompressionLevel.BEST):
            with patch("subprocess.run", return_value=_ok()) as mock_run:
                compressor._run_cwebp(Path("in.png"), Path("out.webp"), level)
            cmd = mock_run.call_args[0][0]
            assert 0 <= int(cmd[cmd.index("-m") + 1]) <= 6


# ---------------------------------------------------------------------------
# compress()
# ---------------------------------------------------------------------------

class TestWebPCompress:

    def test_success(self, compressor, tmp_path):
        src = tmp_path / "src.png"
        Image.new("RGB", (4, 4)).save(src, format="PNG")
        out = tmp_path / "out.webp"

        def fake_run(cmd, **kwargs):
            Path(cmd[cmd.index("-o") + 1]).write_bytes(b"RIFF\x00\x00\x00\x00WEBP")
            return _ok()

        with patch("subprocess.run", side_effect=fake_run):
            with patch.object(compressor, "decompress", return_value=0.003):
                metrics = compressor.compress(src, out)

        assert metrics.success is True
        assert metrics.original_size == 1_000_000

    def test_failure_on_cwebp_error(self, compressor, tmp_path):
        src = tmp_path / "src.png"
        Image.new("RGB", (4, 4)).save(src, format="PNG")

        with patch("subprocess.run", return_value=_fail(1, "cwebp error")):
            metrics = compressor.compress(src, tmp_path / "out.webp")

        assert metrics.success is False
        assert "cwebp failed" in metrics.error_message

    def test_failure_for_missing_input(self, compressor, tmp_path):
        metrics = compressor.compress(tmp_path / "nope.png", tmp_path / "out.webp")
        assert metrics.success is False

    def test_temp_decomp_file_cleaned_up(self, compressor, tmp_path):
        src = tmp_path / "src.png"
        Image.new("RGB", (4, 4)).save(src, format="PNG")
        out = tmp_path / "out.webp"

        def fake_run(cmd, **kwargs):
            if "-o" in cmd:
                Path(cmd[cmd.index("-o") + 1]).write_bytes(b"RIFF\x00\x00\x00\x00WEBP")
            return _ok()

        with patch("subprocess.run", side_effect=fake_run):
            with patch.object(compressor, "decompress", return_value=0.0):
                compressor.compress(src, out)

        assert not any(tmp_path.glob("temp_decomp_*.png"))


# ---------------------------------------------------------------------------
# decompress()
# ---------------------------------------------------------------------------

class TestWebPDecompress:

    def test_dwebp_binary_is_invoked(self, compressor, tmp_path):
        src = tmp_path / "src.webp"
        src.write_bytes(b"RIFF\x00\x00\x00\x00WEBP")
        out = tmp_path / "out.png"

        def fake_run(cmd, **kwargs):
            Path(cmd[cmd.index("-o") + 1]).write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 20)
            return _ok()

        with patch("subprocess.run", side_effect=fake_run) as mock_run:
            compressor.decompress(src, out)

        assert "dwebp" in mock_run.call_args[0][0][0]

    def test_returns_float(self, compressor, tmp_path):
        src = tmp_path / "src.webp"
        src.write_bytes(b"RIFF\x00\x00\x00\x00WEBP")
        out = tmp_path / "out.png"

        def fake_run(cmd, **kwargs):
            Path(cmd[cmd.index("-o") + 1]).write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 20)
            return _ok()

        with patch("subprocess.run", side_effect=fake_run):
            result = compressor.decompress(src, out)

        assert isinstance(result, float)
        assert result >= 0

    def test_raises_on_dwebp_error(self, compressor, tmp_path):
        src = tmp_path / "src.webp"
        src.write_bytes(b"\x00")

        with patch("subprocess.run", return_value=_fail(1, "dwebp failed")):
            with pytest.raises(RuntimeError, match="dwebp failed"):
                compressor.decompress(src, tmp_path / "out.png")