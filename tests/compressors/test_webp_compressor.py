"""
Unit testy pro WebPCompressor.

cwebp/dwebp subprocess volání jsou mockovány.
"""

import subprocess
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

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

from compressors.webp_compressor import WebPCompressor  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ok():
    r = MagicMock()
    r.returncode = 0
    r.stderr = ""
    return r


def _fail(code=1, stderr="webp error"):
    r = MagicMock()
    r.returncode = code
    r.stderr = stderr
    return r


@pytest.fixture()
def fake_bin_dir(tmp_path):
    bin_dir = tmp_path / "libs" / "webp"
    bin_dir.mkdir(parents=True)
    (bin_dir / "cwebp.exe").touch()
    (bin_dir / "dwebp.exe").touch()
    return bin_dir


@pytest.fixture()
def compressor(fake_bin_dir):
    c = object.__new__(WebPCompressor)
    c._bin_dir = fake_bin_dir
    return c


# ---------------------------------------------------------------------------
# Vlastnosti
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

    def test_vyhodi_kdyz_bin_dir_neexistuje(self, tmp_path):
        c = object.__new__(WebPCompressor)
        c._bin_dir = None
        # Patchujeme Path tak, aby bin_dir neexistoval
        with patch("compressors.webp_compressor.Path") as MockPath:
            mock_base = MagicMock()
            mock_bin_dir = MagicMock()
            mock_bin_dir.exists.return_value = False
            mock_base.parent.parent.__truediv__.return_value = mock_bin_dir
            MockPath.return_value = mock_base

            with pytest.raises((RuntimeError, Exception)):
                c._validate_dependencies()

    def test_vyhodi_kdyz_cwebp_chybi(self, tmp_path):
        c = object.__new__(WebPCompressor)
        c._bin_dir = None
        bin_dir = tmp_path / "libs" / "webp"
        bin_dir.mkdir(parents=True)
        (bin_dir / "dwebp.exe").touch()  # jen dwebp, cwebp chybí

        with patch("compressors.webp_compressor.Path") as MockPath:
            mock_base = MagicMock()
            mock_bin_dir = MagicMock()
            mock_bin_dir.exists.return_value = True

            def side_effect_div(name):
                m = MagicMock()
                m.exists.return_value = ("dwebp" in name)
                return m

            mock_bin_dir.__truediv__ = side_effect_div
            mock_base.parent.parent.__truediv__.return_value = mock_bin_dir
            MockPath.return_value = mock_base

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
    def test_z_flag_pro_level(self, compressor, level, expected_z):
        with patch("subprocess.run", return_value=_ok()) as mock_run:
            compressor._run_cwebp(Path("in.png"), Path("out.webp"), level)

        cmd = mock_run.call_args[0][0]
        z_idx = cmd.index("-z")
        assert cmd[z_idx + 1] == expected_z

    def test_prikaz_obsahuje_lossless_flag(self, compressor):
        with patch("subprocess.run", return_value=_ok()) as mock_run:
            compressor._run_cwebp(Path("in.png"), Path("out.webp"), CompressionLevel.BALANCED)

        cmd = mock_run.call_args[0][0]
        assert "-lossless" in cmd

    def test_prikaz_obsahuje_exact_flag(self, compressor):
        with patch("subprocess.run", return_value=_ok()) as mock_run:
            compressor._run_cwebp(Path("in.png"), Path("out.webp"), CompressionLevel.BALANCED)

        cmd = mock_run.call_args[0][0]
        assert "-exact" in cmd

    def test_prikaz_obsahuje_alpha_q_100(self, compressor):
        with patch("subprocess.run", return_value=_ok()) as mock_run:
            compressor._run_cwebp(Path("in.png"), Path("out.webp"), CompressionLevel.BALANCED)

        cmd = mock_run.call_args[0][0]
        assert "-alpha_q" in cmd
        aq_idx = cmd.index("-alpha_q")
        assert cmd[aq_idx + 1] == "100"

    def test_prikaz_obsahuje_output_flag(self, compressor, tmp_path):
        out = tmp_path / "result.webp"
        with patch("subprocess.run", return_value=_ok()) as mock_run:
            compressor._run_cwebp(Path("in.png"), out, CompressionLevel.BALANCED)

        cmd = mock_run.call_args[0][0]
        assert "-o" in cmd
        o_idx = cmd.index("-o")
        assert cmd[o_idx + 1] == str(out)

    def test_vyhodi_runtime_error_pri_nenulove_navratove_hodnote(self, compressor):
        with patch("subprocess.run", return_value=_fail(1, "cwebp selhalo")):
            with pytest.raises(RuntimeError, match="cwebp failed"):
                compressor._run_cwebp(Path("in.png"), Path("out.webp"), CompressionLevel.BALANCED)

    def test_balanced_a_best_maji_q_100(self, compressor):
        for level in (CompressionLevel.BALANCED, CompressionLevel.BEST):
            with patch("subprocess.run", return_value=_ok()) as mock_run:
                compressor._run_cwebp(Path("in.png"), Path("out.webp"), level)

            cmd = mock_run.call_args[0][0]
            q_idx = cmd.index("-q")
            assert cmd[q_idx + 1] == "100", f"q musí být 100 pro {level}"

    def test_m_flag_je_v_rozsahu_0_az_6(self, compressor):
        for level in (CompressionLevel.FASTEST, CompressionLevel.BALANCED, CompressionLevel.BEST):
            with patch("subprocess.run", return_value=_ok()) as mock_run:
                compressor._run_cwebp(Path("in.png"), Path("out.webp"), level)

            cmd = mock_run.call_args[0][0]
            m_idx = cmd.index("-m")
            assert 0 <= int(cmd[m_idx + 1]) <= 6


# ---------------------------------------------------------------------------
# compress()
# ---------------------------------------------------------------------------

class TestWebPCompress:

    def test_compress_uspech(self, compressor, tmp_path):
        src = tmp_path / "src.png"
        Image.new("RGB", (4, 4)).save(src, format="PNG")
        out = tmp_path / "out.webp"

        def fake_run(cmd, **kwargs):
            # Vytvoříme výstupní soubor
            o_idx = cmd.index("-o")
            Path(cmd[o_idx + 1]).write_bytes(b"RIFF\x00\x00\x00\x00WEBP")
            return _ok()

        with patch("subprocess.run", side_effect=fake_run):
            with patch.object(compressor, "decompress", return_value=0.003):
                metrics = compressor.compress(src, out)

        assert metrics.success is True
        assert metrics.original_size == 1_000_000

    def test_compress_failure_pri_cwebp_chybe(self, compressor, tmp_path):
        src = tmp_path / "src.png"
        Image.new("RGB", (4, 4)).save(src, format="PNG")
        out = tmp_path / "out.webp"

        with patch("subprocess.run", return_value=_fail(1, "cwebp error")):
            metrics = compressor.compress(src, out)

        assert metrics.success is False
        assert "cwebp failed" in metrics.error_message

    def test_compress_failure_pro_neexistujici_soubor(self, compressor, tmp_path):
        metrics = compressor.compress(tmp_path / "nope.png", tmp_path / "out.webp")
        assert metrics.success is False

    def test_compress_smaze_temp_decomp(self, compressor, tmp_path):
        src = tmp_path / "src.png"
        Image.new("RGB", (4, 4)).save(src, format="PNG")
        out = tmp_path / "out.webp"

        def fake_run(cmd, **kwargs):
            if "-o" in cmd:
                o_idx = cmd.index("-o")
                Path(cmd[o_idx + 1]).write_bytes(b"RIFF\x00\x00\x00\x00WEBP")
            return _ok()

        with patch("subprocess.run", side_effect=fake_run):
            with patch.object(compressor, "decompress", return_value=0.0):
                compressor.compress(src, out)

        assert not any(tmp_path.glob("temp_decomp_*.png"))


# ---------------------------------------------------------------------------
# decompress()
# ---------------------------------------------------------------------------

class TestWebPDecompress:

    def test_decompress_vola_dwebp(self, compressor, tmp_path):
        src = tmp_path / "src.webp"
        src.write_bytes(b"RIFF\x00\x00\x00\x00WEBP")
        out = tmp_path / "out.png"

        def fake_run(cmd, **kwargs):
            Path(cmd[cmd.index("-o") + 1]).write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 20)
            return _ok()

        with patch("subprocess.run", side_effect=fake_run) as mock_run:
            result = compressor.decompress(src, out)

        cmd = mock_run.call_args[0][0]
        assert "dwebp" in cmd[0]
        assert isinstance(result, float)
        assert result >= 0

    def test_decompress_vyhodi_pri_dwebp_chybe(self, compressor, tmp_path):
        src = tmp_path / "src.webp"
        src.write_bytes(b"\x00")
        out = tmp_path / "out.png"

        with patch("subprocess.run", return_value=_fail(1, "dwebp selhalo")):
            with pytest.raises(RuntimeError, match="dwebp failed"):
                compressor.decompress(src, out)
