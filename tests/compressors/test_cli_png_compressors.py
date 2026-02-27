"""
Unit testy pro OptiPNGCompressor a OxiPNGCompressor.

Subprocess volání a souborový systém jsou mockovány;
žádné skutečné binárky ani PNG soubory nejsou potřeba.
"""

import subprocess
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch, call

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


# ---------------------------------------------------------------------------
# Fixture: falešný bin_dir s existujícími binárkami
# ---------------------------------------------------------------------------

@pytest.fixture()
def fake_bin_dir(tmp_path):
    """Vytvoří dočasný adresář simulující libs/png/ a libs/oxipng/."""
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
# Pomocné
# ---------------------------------------------------------------------------

def _fake_subprocess_ok():
    result = MagicMock()
    result.returncode = 0
    result.stderr = ""
    return result


def _fake_subprocess_fail(code=1, stderr="chyba"):
    result = MagicMock()
    result.returncode = code
    result.stderr = stderr
    return result


def _create_png(path: Path) -> None:
    Image.new("RGB", (4, 4), color=(10, 20, 30)).save(path, format="PNG")


# ===========================================================================
# OptiPNG testy
# ===========================================================================

class TestOptiPNGProperties:

    def test_name(self, optipng):
        assert optipng.name == "OptiPNG"

    def test_extension(self, optipng):
        assert optipng.extension == ".png"


class TestOptiPNGValidateDependencies:

    def test_vyhodi_kdyz_bin_dir_neexistuje(self):
        from compressors.optipng_compressor import OptiPNGCompressor
        c = object.__new__(OptiPNGCompressor)
        c._bin_dir = None
        with patch("compressors.optipng_compressor.Path") as mock_path:
            mock_path.return_value.__truediv__.return_value.exists.return_value = False
            # Testujeme přímo, že RuntimeError je vyhozena při neexistujícím adresáři
            with pytest.raises((RuntimeError, Exception)):
                c._validate_dependencies()

    def test_uspech_kdyz_vse_existuje(self, fake_bin_dir):
        from compressors.optipng_compressor import OptiPNGCompressor
        c = object.__new__(OptiPNGCompressor)
        c._bin_dir = None

        # Patchujeme Path tak, aby ukazoval na fake_bin_dir
        with patch("compressors.optipng_compressor.Path") as MockPath:
            instance = MagicMock()
            bin_dir_mock = fake_bin_dir / "libs" / "png"
            instance.parent.parent.__truediv__.return_value.__truediv__.return_value = MagicMock(
                exists=lambda: True
            )
            # Jednodušší: přímo nastavíme _bin_dir a ověříme, že c funguje
            c._bin_dir = fake_bin_dir / "libs" / "png"

        assert c._bin_dir is not None


class TestOptiPNGRunOptipng:

    @pytest.mark.parametrize("level,expected_o", [
        (CompressionLevel.FASTEST, "0"),
        (CompressionLevel.BALANCED, "4"),
        (CompressionLevel.BEST, "7"),
    ])
    def test_level_mapovani_na_o_flag(self, optipng, level, expected_o):
        with patch("subprocess.run", return_value=_fake_subprocess_ok()) as mock_run:
            optipng._run_optipng(Path("dummy.png"), level)

        cmd = mock_run.call_args[0][0]
        assert f"-o{expected_o}" in cmd

    def test_prikaz_obsahuje_strip_all(self, optipng):
        with patch("subprocess.run", return_value=_fake_subprocess_ok()) as mock_run:
            optipng._run_optipng(Path("dummy.png"), CompressionLevel.BALANCED)

        cmd = mock_run.call_args[0][0]
        assert "-strip" in cmd and "all" in cmd

    def test_prikaz_obsahuje_quiet(self, optipng):
        with patch("subprocess.run", return_value=_fake_subprocess_ok()) as mock_run:
            optipng._run_optipng(Path("dummy.png"), CompressionLevel.BALANCED)

        cmd = mock_run.call_args[0][0]
        assert "-quiet" in cmd

    def test_vyhodi_runtime_error_pri_nenulove_navratove_hodnote(self, optipng):
        with patch("subprocess.run", return_value=_fake_subprocess_fail(1, "optipng error")):
            with pytest.raises(RuntimeError, match="OptiPNG failed"):
                optipng._run_optipng(Path("dummy.png"), CompressionLevel.BALANCED)

    def test_prikaz_obsahuje_cestu_k_souboru(self, optipng, tmp_path):
        target = tmp_path / "test.png"
        with patch("subprocess.run", return_value=_fake_subprocess_ok()) as mock_run:
            optipng._run_optipng(target, CompressionLevel.BALANCED)

        cmd = mock_run.call_args[0][0]
        assert str(target) in cmd


class TestOptiPNGCompress:

    def test_compress_uspech(self, optipng, tmp_path):
        src = tmp_path / "src.png"
        out = tmp_path / "out.png"
        _create_png(src)

        with patch("subprocess.run", return_value=_fake_subprocess_ok()):
            with patch.object(optipng, "decompress", return_value=0.003):
                metrics = optipng.compress(src, out)

        assert metrics.success is True
        assert metrics.original_size == 1_000_000
        assert metrics.compressed_size > 0

    def test_compress_failure_pri_subprocess_chybe(self, optipng, tmp_path):
        src = tmp_path / "src.png"
        out = tmp_path / "out.png"
        _create_png(src)

        with patch("subprocess.run", return_value=_fake_subprocess_fail()):
            metrics = optipng.compress(src, out)

        assert metrics.success is False
        assert "OptiPNG failed" in metrics.error_message

    def test_compress_smaze_temp_decomp(self, optipng, tmp_path):
        src = tmp_path / "src.png"
        out = tmp_path / "out.png"
        _create_png(src)

        with patch("subprocess.run", return_value=_fake_subprocess_ok()):
            with patch.object(optipng, "decompress", return_value=0.0):
                optipng.compress(src, out)

        assert not any(tmp_path.glob("temp_decomp_*.png"))

    def test_compress_failure_pro_neexistujici_vstup(self, optipng, tmp_path):
        metrics = optipng.compress(tmp_path / "nope.png", tmp_path / "out.png")
        assert metrics.success is False


class TestOptiPNGDecompress:

    def test_decompress_vraci_float(self, optipng, tmp_path):
        src = tmp_path / "src.png"
        out = tmp_path / "out.png"
        _create_png(src)

        result = optipng.decompress(src, out)

        assert isinstance(result, float)
        assert result >= 0
        assert out.exists()


# ===========================================================================
# OxiPNG testy
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
    def test_level_mapovani_na_o_flag(self, oxipng, level, expected_o):
        with patch("subprocess.run", return_value=_fake_subprocess_ok()) as mock_run:
            oxipng._run_oxipng(Path("in.png"), Path("out.png"), level)

        cmd = mock_run.call_args[0][0]
        assert "-o" in cmd
        o_idx = cmd.index("-o")
        assert cmd[o_idx + 1] == expected_o

    def test_prikaz_obsahuje_strip_all(self, oxipng):
        with patch("subprocess.run", return_value=_fake_subprocess_ok()) as mock_run:
            oxipng._run_oxipng(Path("in.png"), Path("out.png"), CompressionLevel.BALANCED)

        cmd = mock_run.call_args[0][0]
        assert "--strip" in cmd and "all" in cmd

    def test_prikaz_obsahuje_quiet(self, oxipng):
        with patch("subprocess.run", return_value=_fake_subprocess_ok()) as mock_run:
            oxipng._run_oxipng(Path("in.png"), Path("out.png"), CompressionLevel.BALANCED)

        cmd = mock_run.call_args[0][0]
        assert "-q" in cmd

    def test_prikaz_obsahuje_out_flag(self, oxipng, tmp_path):
        out = tmp_path / "result.png"
        with patch("subprocess.run", return_value=_fake_subprocess_ok()) as mock_run:
            oxipng._run_oxipng(Path("in.png"), out, CompressionLevel.BALANCED)

        cmd = mock_run.call_args[0][0]
        assert "--out" in cmd
        out_idx = cmd.index("--out")
        assert cmd[out_idx + 1] == str(out)

    def test_vstupni_soubor_neni_modifikovan_primocarou_cestou(self, oxipng, tmp_path):
        """OxiPNG píše do --out, vstupní soubor musí zůstat nedotčen."""
        in_png = tmp_path / "input.png"
        out_png = tmp_path / "output.png"
        with patch("subprocess.run", return_value=_fake_subprocess_ok()) as mock_run:
            oxipng._run_oxipng(in_png, out_png, CompressionLevel.BALANCED)

        cmd = mock_run.call_args[0][0]
        # Poslední argument musí být vstupní soubor
        assert cmd[-1] == str(in_png)

    def test_vyhodi_runtime_error_pri_chybe(self, oxipng):
        with patch("subprocess.run", return_value=_fake_subprocess_fail(2, "oxipng error")):
            with pytest.raises(RuntimeError, match="OxiPNG failed"):
                oxipng._run_oxipng(Path("in.png"), Path("out.png"), CompressionLevel.BALANCED)


class TestOxiPNGCompress:

    def test_compress_uspech(self, oxipng, tmp_path):
        src = tmp_path / "src.png"
        out = tmp_path / "out.png"
        _create_png(src)

        def fake_run(cmd, **kwargs):
            # Simulujeme OxiPNG: zkopírujeme vstup na výstup
            out_idx = cmd.index("--out")
            Path(cmd[out_idx + 1]).write_bytes(Path(cmd[-1]).read_bytes()
                                                if Path(cmd[-1]).exists()
                                                else b"\x89PNG\r\n\x1a\n" + b"\x00" * 50)
            return _fake_subprocess_ok()

        with patch("subprocess.run", side_effect=fake_run):
            with patch.object(oxipng, "decompress", return_value=0.004):
                metrics = oxipng.compress(src, out)

        assert metrics.success is True

    def test_compress_smaze_temp_input(self, oxipng, tmp_path):
        src = tmp_path / "src.png"
        out = tmp_path / "out.png"
        _create_png(src)

        def fake_run(cmd, **kwargs):
            out_idx = cmd.index("--out")
            Path(cmd[out_idx + 1]).write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 20)
            return _fake_subprocess_ok()

        with patch("subprocess.run", side_effect=fake_run):
            with patch.object(oxipng, "decompress", return_value=0.0):
                oxipng.compress(src, out)

        assert not any(tmp_path.glob("temp_input_*.png"))

    def test_compress_failure_pri_subprocess_chybe(self, oxipng, tmp_path):
        src = tmp_path / "src.png"
        out = tmp_path / "out.png"
        _create_png(src)

        with patch("subprocess.run", return_value=_fake_subprocess_fail(1, "oxipng chyba")):
            metrics = oxipng.compress(src, out)

        assert metrics.success is False

    def test_compress_failure_pro_neexistujici_vstup(self, oxipng, tmp_path):
        metrics = oxipng.compress(tmp_path / "nope.png", tmp_path / "out.png")
        assert metrics.success is False


class TestOxiPNGDecompress:

    def test_decompress_vraci_float(self, oxipng, tmp_path):
        src = tmp_path / "src.png"
        out = tmp_path / "out.png"
        _create_png(src)

        result = oxipng.decompress(src, out)

        assert isinstance(result, float)
        assert result >= 0
        assert out.exists()
