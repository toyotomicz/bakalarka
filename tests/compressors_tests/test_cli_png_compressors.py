"""
Unit tests for OptiPNGCompressor and OxiPNGCompressor.

Subprocess calls and the filesystem are mocked; no real binaries or PNG
files are required.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

# Stubs are registered in conftest.py, just pull CompressionLevel for use in
# parametrize decorators (which are evaluated at collection time).
CompressionLevel = sys.modules["main"].CompressionLevel


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def fake_bin_dir(tmp_path):
    """Temporary directory tree that looks like libs/png/ and libs/oxipng/."""
    png_dir = tmp_path / "libs" / "png"
    png_dir.mkdir(parents=True)
    (png_dir / "optipng.exe").touch()

    oxipng_dir = tmp_path / "libs" / "oxipng"
    oxipng_dir.mkdir(parents=True)
    (oxipng_dir / "oxipng.exe").touch()

    return tmp_path


@pytest.fixture()
def optipng(fake_bin_dir):
    from compressors.optipng_compressor import OptiPNGCompressor
    c = object.__new__(OptiPNGCompressor)
    c._bin_dir = fake_bin_dir / "libs" / "png"
    return c


@pytest.fixture()
def oxipng(fake_bin_dir):
    from compressors.oxipng_compressor import OxiPNGCompressor
    c = object.__new__(OxiPNGCompressor)
    c._bin_dir = fake_bin_dir / "libs" / "oxipng"
    return c


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _subprocess_ok() -> MagicMock:
    r = MagicMock()
    r.returncode = 0
    r.stderr = ""
    return r


def _subprocess_fail(code: int = 1, stderr: str = "error") -> MagicMock:
    r = MagicMock()
    r.returncode = code
    r.stderr = stderr
    return r


def _create_png(path: Path) -> None:
    Image.new("RGB", (4, 4), color=(10, 20, 30)).save(path, format="PNG")


# ===========================================================================
# OptiPNG
# ===========================================================================

class TestOptiPNGProperties:

    def test_name(self, optipng):
        assert optipng.name == "OptiPNG"

    def test_extension(self, optipng):
        assert optipng.extension == ".png"


class TestOptiPNGValidateDependencies:

    def test_raises_when_bin_dir_missing(self, tmp_path):
        from compressors.optipng_compressor import OptiPNGCompressor
        c = object.__new__(OptiPNGCompressor)
        c._bin_dir = None

        with patch("compressors.optipng_compressor.Path") as MockPath:
            mock_base = MagicMock()
            mock_bin_dir = MagicMock()
            mock_bin_dir.exists.return_value = False
            mock_base.parent.parent.__truediv__.return_value.__truediv__.return_value = mock_bin_dir
            MockPath.return_value = mock_base

            with pytest.raises((RuntimeError, Exception)):
                c._validate_dependencies()

    def test_succeeds_when_binary_exists(self, fake_bin_dir):
        from compressors.optipng_compressor import OptiPNGCompressor
        c = object.__new__(OptiPNGCompressor)
        c._bin_dir = None

        with patch("compressors.optipng_compressor.Path") as MockPath:
            mock_base = MagicMock()
            mock_bin_dir = MagicMock()
            mock_bin_dir.exists.return_value = True
            mock_optipng_exe = MagicMock()
            mock_optipng_exe.exists.return_value = True
            mock_bin_dir.__truediv__.return_value = mock_optipng_exe
            mock_base.parent.parent.__truediv__.return_value.__truediv__.return_value = mock_bin_dir
            MockPath.return_value = mock_base

            # Should not raise
            c._validate_dependencies()

        assert c._bin_dir is not None


class TestOptiPNGRunOptipng:

    @pytest.mark.parametrize("level,expected_o", [
        (CompressionLevel.FASTEST, "0"),
        (CompressionLevel.BALANCED, "4"),
        (CompressionLevel.BEST, "7"),
    ])
    def test_level_maps_to_o_flag(self, optipng, level, expected_o):
        with patch("compressors.optipng_compressor.run_with_affinity", return_value=_subprocess_ok()) as mock_run:
            optipng._run_optipng(Path("dummy.png"), level)

        cmd = mock_run.call_args[0][0]
        assert f"-o{expected_o}" in cmd

    def test_compression_levels_are_monotonically_ordered(self, optipng):
        """FASTEST < BALANCED < BEST in terms of the -oN flag value."""
        levels = [CompressionLevel.FASTEST, CompressionLevel.BALANCED, CompressionLevel.BEST]
        o_values = []
        for level in levels:
            with patch("compressors.optipng_compressor.run_with_affinity", return_value=_subprocess_ok()) as mock_run:
                optipng._run_optipng(Path("dummy.png"), level)
            cmd = mock_run.call_args[0][0]
            o_flag = next(a for a in cmd if a.startswith("-o") and a[2:].isdigit())
            o_values.append(int(o_flag[2:]))

        assert o_values == sorted(o_values), "Optimization levels must be strictly ordered"

    def test_command_does_not_contain_strip(self, optipng):
        """Strip metadata is controlled by BenchmarkConfig, not the compressor."""
        with patch("compressors.optipng_compressor.run_with_affinity", return_value=_subprocess_ok()) as mock_run:
            optipng._run_optipng(Path("dummy.png"), CompressionLevel.BALANCED)

        cmd = mock_run.call_args[0][0]
        assert "-strip" not in cmd

    def test_command_contains_quiet(self, optipng):
        with patch("compressors.optipng_compressor.run_with_affinity", return_value=_subprocess_ok()) as mock_run:
            optipng._run_optipng(Path("dummy.png"), CompressionLevel.BALANCED)

        cmd = mock_run.call_args[0][0]
        assert "-quiet" in cmd

    def test_raises_runtime_error_on_nonzero_return_code(self, optipng):
        with patch("compressors.optipng_compressor.run_with_affinity", return_value=_subprocess_fail(1, "optipng error")):
            with pytest.raises(RuntimeError, match="OptiPNG failed"):
                optipng._run_optipng(Path("dummy.png"), CompressionLevel.BALANCED)

    def test_command_contains_target_path(self, optipng, tmp_path):
        target = tmp_path / "test.png"
        with patch("compressors.optipng_compressor.run_with_affinity", return_value=_subprocess_ok()) as mock_run:
            optipng._run_optipng(target, CompressionLevel.BALANCED)

        cmd = mock_run.call_args[0][0]
        assert str(target) in cmd


class TestOptiPNGCompress:

    def test_success(self, optipng, tmp_path):
        src = tmp_path / "src.png"
        out = tmp_path / "out.png"
        _create_png(src)

        with patch("compressors.optipng_compressor.run_with_affinity", return_value=_subprocess_ok()):
            with patch.object(optipng, "decompress", return_value=0.003) as mock_decompress:
                metrics = optipng.compress(src, out)

        assert metrics.success is True
        assert metrics.original_size > 0
        assert metrics.compressed_size > 0
        assert mock_decompress.called

    def test_failure_on_subprocess_error(self, optipng, tmp_path):
        src = tmp_path / "src.png"
        out = tmp_path / "out.png"
        _create_png(src)

        with patch("compressors.optipng_compressor.run_with_affinity", return_value=_subprocess_fail()):
            metrics = optipng.compress(src, out)

        assert metrics.success is False
        assert "OptiPNG failed" in metrics.error_message

    def test_temp_decomp_file_cleaned_up(self, optipng, tmp_path):
        src = tmp_path / "src.png"
        out = tmp_path / "out.png"
        _create_png(src)

        with patch("compressors.optipng_compressor.run_with_affinity", return_value=_subprocess_ok()):
            with patch.object(optipng, "decompress", return_value=0.0):
                optipng.compress(src, out)

        assert not any(tmp_path.glob("temp_decomp_*.png"))

    def test_failure_for_missing_input(self, optipng, tmp_path):
        metrics = optipng.compress(tmp_path / "nope.png", tmp_path / "out.png")
        assert metrics.success is False


class TestOptiPNGDecompress:

    def test_returns_float(self, optipng, tmp_path):
        src = tmp_path / "src.png"
        out = tmp_path / "out.png"
        _create_png(src)

        result = optipng.decompress(src, out)

        assert isinstance(result, float)
        assert result >= 0
        assert out.exists()


# ===========================================================================
# OxiPNG
# ===========================================================================

class TestOxiPNGProperties:

    def test_name(self, oxipng):
        assert oxipng.name == "OxiPNG"

    def test_extension(self, oxipng):
        assert oxipng.extension == ".png"


class TestOxiPNGRunOxipng:

    @pytest.mark.parametrize("level,expected_o", [
        (CompressionLevel.FASTEST, "1"),
        (CompressionLevel.BALANCED, "3"),
        (CompressionLevel.BEST, "6"),
    ])
    def test_level_maps_to_o_flag(self, oxipng, level, expected_o):
        with patch("compressors.oxipng_compressor.run_with_affinity", return_value=_subprocess_ok()) as mock_run:
            oxipng._run_oxipng(Path("in.png"), Path("out.png"), level)

        cmd = mock_run.call_args[0][0]
        assert "-o" in cmd
        o_idx = cmd.index("-o")
        assert cmd[o_idx + 1] == expected_o

    def test_compression_levels_are_monotonically_ordered(self, oxipng):
        levels = [CompressionLevel.FASTEST, CompressionLevel.BALANCED, CompressionLevel.BEST]
        o_values = []
        for level in levels:
            with patch("compressors.oxipng_compressor.run_with_affinity", return_value=_subprocess_ok()) as mock_run:
                oxipng._run_oxipng(Path("in.png"), Path("out.png"), level)
            cmd = mock_run.call_args[0][0]
            o_idx = cmd.index("-o")
            o_values.append(int(cmd[o_idx + 1]))

        assert o_values == sorted(o_values), "Optimization levels must be strictly ordered"

    def test_command_does_not_contain_strip(self, oxipng):
        """Strip metadata is controlled by BenchmarkConfig, not the compressor."""
        with patch("compressors.oxipng_compressor.run_with_affinity", return_value=_subprocess_ok()) as mock_run:
            oxipng._run_oxipng(Path("in.png"), Path("out.png"), CompressionLevel.BALANCED)

        cmd = mock_run.call_args[0][0]
        assert "--strip" not in cmd

    def test_command_contains_quiet(self, oxipng):
        with patch("compressors.oxipng_compressor.run_with_affinity", return_value=_subprocess_ok()) as mock_run:
            oxipng._run_oxipng(Path("in.png"), Path("out.png"), CompressionLevel.BALANCED)

        cmd = mock_run.call_args[0][0]
        assert "-q" in cmd

    def test_command_contains_out_flag(self, oxipng, tmp_path):
        out = tmp_path / "result.png"
        with patch("compressors.oxipng_compressor.run_with_affinity", return_value=_subprocess_ok()) as mock_run:
            oxipng._run_oxipng(Path("in.png"), out, CompressionLevel.BALANCED)

        cmd = mock_run.call_args[0][0]
        assert "--out" in cmd
        assert cmd[cmd.index("--out") + 1] == str(out)

    def test_input_file_is_last_argument(self, oxipng, tmp_path):
        """OxiPNG writes to --out; the source file must be the final positional arg."""
        in_png = tmp_path / "input.png"
        out_png = tmp_path / "output.png"

        with patch("compressors.oxipng_compressor.run_with_affinity", return_value=_subprocess_ok()) as mock_run:
            oxipng._run_oxipng(in_png, out_png, CompressionLevel.BALANCED)

        cmd = mock_run.call_args[0][0]
        assert cmd[-1] == str(in_png)

    def test_raises_runtime_error_on_failure(self, oxipng):
        with patch("compressors.oxipng_compressor.run_with_affinity", return_value=_subprocess_fail(2, "oxipng error")):
            with pytest.raises(RuntimeError, match="OxiPNG failed"):
                oxipng._run_oxipng(Path("in.png"), Path("out.png"), CompressionLevel.BALANCED)


class TestOxiPNGCompress:

    def test_success(self, oxipng, tmp_path):
        src = tmp_path / "src.png"
        out = tmp_path / "out.png"
        _create_png(src)

        def fake_run(cmd, **kwargs):
            out_idx = cmd.index("--out")
            Path(cmd[out_idx + 1]).write_bytes(
                Path(cmd[-1]).read_bytes() if Path(cmd[-1]).exists()
                else b"\x89PNG\r\n\x1a\n" + b"\x00" * 50
            )
            return _subprocess_ok()

        with patch("compressors.oxipng_compressor.run_with_affinity", side_effect=fake_run):
            with patch.object(oxipng, "decompress", return_value=0.004):
                metrics = oxipng.compress(src, out)

        assert metrics.success is True

    def test_temp_input_file_cleaned_up(self, oxipng, tmp_path):
        src = tmp_path / "src.png"
        out = tmp_path / "out.png"
        _create_png(src)

        def fake_run(cmd, **kwargs):
            out_idx = cmd.index("--out")
            Path(cmd[out_idx + 1]).write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 20)
            return _subprocess_ok()

        with patch("compressors.oxipng_compressor.run_with_affinity", side_effect=fake_run):
            with patch.object(oxipng, "decompress", return_value=0.0):
                oxipng.compress(src, out)

        assert not any(tmp_path.glob("temp_input_*.png"))

    def test_failure_on_subprocess_error(self, oxipng, tmp_path):
        src = tmp_path / "src.png"
        out = tmp_path / "out.png"
        _create_png(src)

        with patch("compressors.oxipng_compressor.run_with_affinity", return_value=_subprocess_fail(1, "oxipng error")):
            metrics = oxipng.compress(src, out)

        assert metrics.success is False
        assert "OxiPNG failed" in metrics.error_message

    def test_failure_for_missing_input(self, oxipng, tmp_path):
        metrics = oxipng.compress(tmp_path / "nope.png", tmp_path / "out.png")
        assert metrics.success is False


class TestOxiPNGDecompress:

    def test_returns_float(self, oxipng, tmp_path):
        src = tmp_path / "src.png"
        out = tmp_path / "out.png"
        _create_png(src)

        result = oxipng.decompress(src, out)

        assert isinstance(result, float)
        assert result >= 0
        assert out.exists()