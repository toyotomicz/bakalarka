"""
Unit testy pro projekt Image Compression Benchmark
Pokrývá: main.py, image_size_calculator.py, verification.py
"""

import io
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Helpers – vytvoření testovacích obrázků v paměti / na disku
# ---------------------------------------------------------------------------

def make_rgb_image(width=64, height=64, color=(100, 150, 200)) -> Image.Image:
    img = Image.new("RGB", (width, height), color)
    return img


def make_rgba_image(width=64, height=64, color=(100, 150, 200, 128)) -> Image.Image:
    img = Image.new("RGBA", (width, height), color)
    return img


def make_grayscale_image(width=64, height=64, value=128) -> Image.Image:
    img = Image.new("L", (width, height), value)
    return img


def save_png(img: Image.Image, path: Path) -> Path:
    img.save(path, format="PNG")
    return path


# ---------------------------------------------------------------------------
# Fixtures – dočasný adresář sdílený uvnitř test třídy
# ---------------------------------------------------------------------------

class TempDirMixin(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()


# ===========================================================================
# 1. Testy CompressionMetrics
# ===========================================================================

class TestCompressionMetrics(unittest.TestCase):

    def _make(self, original=1_000_000, compressed=400_000,
              comp_time=1.0, decomp_time=0.5, success=True, err=None):
        from main import CompressionMetrics
        return CompressionMetrics(
            original_size=original,
            compressed_size=compressed,
            compression_ratio=original / compressed if compressed else 0,
            compression_time=comp_time,
            decompression_time=decomp_time,
            success=success,
            error_message=err,
        )

    def test_space_saving_percent_normal(self):
        m = self._make(1_000_000, 600_000)
        self.assertAlmostEqual(m.space_saving_percent, 40.0, places=5)

    def test_space_saving_percent_zero_original(self):
        m = self._make(0, 0)
        self.assertEqual(m.space_saving_percent, 0.0)

    def test_compression_speed_mbps(self):
        m = self._make(original=2 * 1024 * 1024, comp_time=2.0)
        self.assertAlmostEqual(m.compression_speed_mbps, 1.0, places=5)

    def test_compression_speed_zero_time(self):
        m = self._make(comp_time=0.0)
        self.assertEqual(m.compression_speed_mbps, 0.0)

    def test_decompression_speed_mbps(self):
        m = self._make(original=4 * 1024 * 1024, decomp_time=2.0)
        self.assertAlmostEqual(m.decompression_speed_mbps, 2.0, places=5)

    def test_decompression_speed_zero_time(self):
        m = self._make(decomp_time=0.0)
        self.assertEqual(m.decompression_speed_mbps, 0.0)

    def test_error_message_stored(self):
        m = self._make(success=False, err="test error")
        self.assertFalse(m.success)
        self.assertEqual(m.error_message, "test error")

    def test_compression_ratio_correct(self):
        m = self._make(original=1000, compressed=250)
        self.assertAlmostEqual(m.compression_ratio, 4.0, places=5)


# ===========================================================================
# 2. Testy CompressionLevel enum
# ===========================================================================

class TestCompressionLevel(unittest.TestCase):

    def test_values_ordered(self):
        from main import CompressionLevel
        self.assertLess(
            CompressionLevel.FASTEST.value,
            CompressionLevel.BALANCED.value,
        )
        self.assertLess(
            CompressionLevel.BALANCED.value,
            CompressionLevel.BEST.value,
        )

    def test_all_levels_exist(self):
        from main import CompressionLevel
        names = {l.name for l in CompressionLevel}
        self.assertIn("FASTEST", names)
        self.assertIn("BALANCED", names)
        self.assertIn("BEST", names)


# ===========================================================================
# 3. Testy CompressorFactory
# ===========================================================================

class TestCompressorFactory(unittest.TestCase):

    def setUp(self):
        from main import CompressorFactory
        # Ulož původní stav a začni čistý
        self._original = dict(CompressorFactory._compressors)
        CompressorFactory._compressors.clear()

    def tearDown(self):
        from main import CompressorFactory
        CompressorFactory._compressors.clear()
        CompressorFactory._compressors.update(self._original)

    def _make_dummy_class(self, name="Dummy", ext=".dummy"):
        from main import ImageCompressor, CompressionMetrics, CompressionLevel
        class DummyCompressor(ImageCompressor):
            def _validate_dependencies(self): pass
            @property
            def name(self): return name
            @property
            def extension(self): return ext
            def compress(self, i, o, level=CompressionLevel.BALANCED):
                return CompressionMetrics(0,0,0,0,0,True)
            def decompress(self, i, o): return 0.0
        return DummyCompressor

    def test_register_and_list(self):
        from main import CompressorFactory
        DummyCls = self._make_dummy_class()
        CompressorFactory.register("dummy", DummyCls)
        self.assertIn("dummy", CompressorFactory.list_available())

    def test_create_registered(self):
        from main import CompressorFactory
        DummyCls = self._make_dummy_class()
        CompressorFactory.register("dummy", DummyCls)
        instance = CompressorFactory.create("dummy")
        self.assertEqual(instance.name, "Dummy")

    def test_create_unknown_raises(self):
        from main import CompressorFactory
        with self.assertRaises(ValueError):
            CompressorFactory.create("nonexistent_codec")

    def test_get_by_extension_found(self):
        from main import CompressorFactory
        DummyCls = self._make_dummy_class(ext=".xyz")
        CompressorFactory.register("dummy_xyz", DummyCls)
        result = CompressorFactory.get_by_extension(".xyz")
        self.assertIsNotNone(result)

    def test_get_by_extension_not_found(self):
        from main import CompressorFactory
        result = CompressorFactory.get_by_extension(".zzz_nonexistent")
        self.assertIsNone(result)

    def test_register_overwrite(self):
        from main import CompressorFactory
        A = self._make_dummy_class("A")
        B = self._make_dummy_class("B")
        CompressorFactory.register("codec", A)
        CompressorFactory.register("codec", B)
        instance = CompressorFactory.create("codec")
        self.assertEqual(instance.name, "B")

    def test_list_available_empty(self):
        from main import CompressorFactory
        self.assertEqual(CompressorFactory.list_available(), [])


# ===========================================================================
# 4. Testy BenchmarkResult
# ===========================================================================

class TestBenchmarkResult(unittest.TestCase):

    def test_basic_fields(self):
        from main import BenchmarkResult, CompressionMetrics
        m = CompressionMetrics(100, 50, 2.0, 0.1, 0.05, True)
        r = BenchmarkResult(
            image_path=Path("test.png"),
            format_name="PNG",
            metrics=m,
        )
        self.assertEqual(r.format_name, "PNG")
        self.assertEqual(r.metrics.compressed_size, 50)
        self.assertIsNone(r.system_metrics)
        self.assertEqual(r.source_file_size, 0)

    def test_metadata_default(self):
        from main import BenchmarkResult, CompressionMetrics
        m = CompressionMetrics(0, 0, 0, 0, 0, False)
        r = BenchmarkResult(Path("x"), "fmt", m)
        self.assertIsInstance(r.metadata, dict)

    def test_system_metrics_assignable(self):
        from main import BenchmarkResult, CompressionMetrics
        m = CompressionMetrics(0, 0, 0, 0, 0, True)
        r = BenchmarkResult(Path("x"), "fmt", m, system_metrics=object())
        self.assertIsNotNone(r.system_metrics)


# ===========================================================================
# 5. Testy ImageSizeCalculator
# ===========================================================================

class TestImageSizeCalculator(TempDirMixin):

    def _save(self, img, name="test.png"):
        p = self.tmp / name
        img.save(p, format="PNG")
        return p

    # --- calculate_uncompressed_size ---

    def test_rgb_size(self):
        from image_size_calculator import ImageSizeCalculator
        img = make_rgb_image(100, 200)
        path = self._save(img)
        size = ImageSizeCalculator.calculate_uncompressed_size(path)
        self.assertEqual(size, 100 * 200 * 3)

    def test_rgba_size(self):
        from image_size_calculator import ImageSizeCalculator
        img = make_rgba_image(50, 50)
        path = self._save(img)
        size = ImageSizeCalculator.calculate_uncompressed_size(path)
        self.assertEqual(size, 50 * 50 * 4)

    def test_grayscale_size(self):
        from image_size_calculator import ImageSizeCalculator
        img = make_grayscale_image(80, 60)
        path = self._save(img)
        size = ImageSizeCalculator.calculate_uncompressed_size(path)
        self.assertEqual(size, 80 * 60 * 1)

    def test_nonexistent_file_fallback(self):
        """Pro neexistující soubor by měl padnout nebo vrátit 0."""
        from image_size_calculator import ImageSizeCalculator
        path = self.tmp / "ghost.png"
        # calculate_uncompressed_size volá get_image_info → vrátí None → fallback na stat()
        # Pokud soubor neexistuje, stat() vyhodí FileNotFoundError
        with self.assertRaises((FileNotFoundError, OSError)):
            ImageSizeCalculator.calculate_uncompressed_size(path)

    # --- get_image_info ---

    def test_image_info_fields(self):
        from image_size_calculator import ImageSizeCalculator
        img = make_rgb_image(32, 16)
        path = self._save(img)
        info = ImageSizeCalculator.get_image_info(path)
        self.assertIsNotNone(info)
        self.assertEqual(info.width, 32)
        self.assertEqual(info.height, 16)
        self.assertEqual(info.mode, "RGB")
        self.assertEqual(info.channels, 3)
        self.assertEqual(info.bits_per_channel, 8)
        self.assertEqual(info.bits_per_pixel, 24)

    def test_image_info_resolution(self):
        from image_size_calculator import ImageSizeCalculator
        img = make_rgb_image(10, 20)
        path = self._save(img)
        info = ImageSizeCalculator.get_image_info(path)
        self.assertEqual(info.resolution, 200)

    def test_image_info_megapixels(self):
        from image_size_calculator import ImageSizeCalculator
        img = make_rgb_image(1000, 1000)
        path = self._save(img)
        info = ImageSizeCalculator.get_image_info(path)
        self.assertAlmostEqual(info.megapixels, 1.0, places=5)

    def test_image_info_file_size(self):
        from image_size_calculator import ImageSizeCalculator
        img = make_rgb_image(64, 64)
        path = self._save(img)
        info = ImageSizeCalculator.get_image_info(path)
        self.assertEqual(info.file_size_bytes, path.stat().st_size)

    def test_get_image_info_nonexistent(self):
        from image_size_calculator import ImageSizeCalculator
        path = self.tmp / "nope.png"
        info = ImageSizeCalculator.get_image_info(path)
        self.assertIsNone(info)

    # --- get_compression_baseline ---

    def test_compression_baseline_keys(self):
        from image_size_calculator import ImageSizeCalculator
        img = make_rgb_image(64, 64)
        path = self._save(img)
        baseline = ImageSizeCalculator.get_compression_baseline(path)
        for key in ("uncompressed_size", "file_size", "baseline_ratio", "format"):
            self.assertIn(key, baseline)

    def test_compression_baseline_ratio_positive(self):
        from image_size_calculator import ImageSizeCalculator
        img = make_rgb_image(64, 64)
        path = self._save(img)
        baseline = ImageSizeCalculator.get_compression_baseline(path)
        self.assertGreater(baseline["baseline_ratio"], 0)

    def test_compression_baseline_format(self):
        from image_size_calculator import ImageSizeCalculator
        img = make_rgb_image(32, 32)
        path = self._save(img, "sample.png")
        baseline = ImageSizeCalculator.get_compression_baseline(path)
        self.assertEqual(baseline["format"], ".png")

    # --- Různé barevné módy ---

    def test_l_mode(self):
        from image_size_calculator import ImageSizeCalculator
        img = Image.new("L", (40, 30), 100)
        path = self.tmp / "gray.png"
        img.save(path)
        info = ImageSizeCalculator.get_image_info(path)
        self.assertEqual(info.channels, 1)
        self.assertEqual(info.bits_per_channel, 8)

    def test_uncompressed_size_equals_width_x_height_x_channels(self):
        from image_size_calculator import ImageSizeCalculator
        for mode, channels, w, h in [
            ("RGB", 3, 50, 50),
            ("RGBA", 4, 30, 40),
            ("L", 1, 100, 80),
        ]:
            img = Image.new(mode, (w, h))
            path = self.tmp / f"test_{mode}.png"
            img.save(path)
            size = ImageSizeCalculator.calculate_uncompressed_size(path)
            self.assertEqual(size, w * h * channels,
                             msg=f"Failed for mode={mode}")


# ===========================================================================
# 6. Testy VerificationResult
# ===========================================================================

class TestVerificationResult(unittest.TestCase):

    def _make(self, is_lossless=True, max_diff=0.0, diff_px=0, total=1000):
        from utils.verification import VerificationResult
        return VerificationResult(
            is_lossless=is_lossless,
            max_difference=max_diff,
            different_pixels=diff_px,
            total_pixels=total,
        )

    def test_accuracy_percent_perfect(self):
        r = self._make(diff_px=0, total=1000)
        self.assertAlmostEqual(r.accuracy_percent, 100.0)

    def test_accuracy_percent_half(self):
        r = self._make(diff_px=500, total=1000)
        self.assertAlmostEqual(r.accuracy_percent, 50.0)

    def test_accuracy_percent_zero_total(self):
        r = self._make(diff_px=0, total=0)
        self.assertEqual(r.accuracy_percent, 0.0)

    def test_identical_pixels(self):
        r = self._make(diff_px=300, total=1000)
        self.assertEqual(r.identical_pixels, 700)

    def test_lossless_flag(self):
        r = self._make(is_lossless=True, max_diff=0.0)
        self.assertTrue(r.is_lossless)

    def test_lossy_flag(self):
        r = self._make(is_lossless=False, max_diff=5.0, diff_px=10)
        self.assertFalse(r.is_lossless)

    def test_error_message_default_none(self):
        r = self._make()
        self.assertIsNone(r.error_message)


# ===========================================================================
# 7. Testy ImageVerifier._compare
# ===========================================================================

class TestImageVerifierCompare(unittest.TestCase):
    """Testuje statickou interní metodu _compare přímo."""

    def _compare(self, img1, img2):
        from utils.verification import ImageVerifier
        return ImageVerifier._compare(img1, img2)

    def test_identical_rgb_images(self):
        img = make_rgb_image(32, 32, (100, 100, 100))
        result = self._compare(img, img.copy())
        self.assertTrue(result.is_lossless)
        self.assertEqual(result.max_difference, 0.0)
        self.assertEqual(result.different_pixels, 0)

    def test_different_rgb_images(self):
        img1 = make_rgb_image(32, 32, (0, 0, 0))
        img2 = make_rgb_image(32, 32, (255, 255, 255))
        result = self._compare(img1, img2)
        self.assertFalse(result.is_lossless)
        self.assertEqual(result.max_difference, 255.0)
        self.assertEqual(result.different_pixels, 32 * 32)

    def test_single_pixel_difference(self):
        img1 = make_rgb_image(10, 10, (100, 100, 100))
        img2 = img1.copy()
        pixels = img2.load()
        pixels[5, 5] = (101, 100, 100)  # jeden pixel se liší o 1
        result = self._compare(img1, img2)
        self.assertFalse(result.is_lossless)
        self.assertEqual(result.different_pixels, 1)
        self.assertEqual(result.max_difference, 1.0)

    def test_size_mismatch(self):
        img1 = make_rgb_image(10, 10)
        img2 = make_rgb_image(20, 20)
        result = self._compare(img1, img2)
        self.assertFalse(result.is_lossless)
        self.assertIn("Nesoulad", result.error_message)

    def test_mode_conversion(self):
        """Komparátor musí zvládnout různé barevné módy převodem."""
        img_rgb = make_rgb_image(16, 16, (128, 128, 128))
        img_rgba = make_rgba_image(16, 16, (128, 128, 128, 255))
        # Měl by proběhnout konverzí, ne selhat
        result = self._compare(img_rgb, img_rgba)
        self.assertIsNotNone(result)

    def test_identical_grayscale(self):
        img = make_grayscale_image(16, 16, 200)
        result = self._compare(img, img.copy())
        self.assertTrue(result.is_lossless)

    def test_total_pixels_correct(self):
        img = make_rgb_image(8, 12)
        result = self._compare(img, img.copy())
        self.assertEqual(result.total_pixels, 8 * 12)


# ===========================================================================
# 8. Testy ImageVerifier.verify_lossless (end-to-end s reálnými soubory)
# ===========================================================================

class TestImageVerifierLossless(TempDirMixin):

    def test_verify_png_is_lossless(self):
        """PNG uložený a znovu otevřený musí být pixel-perfect."""
        from utils.verification import ImageVerifier
        img = make_rgb_image(32, 32)
        original_path = save_png(img, self.tmp / "original.png")
        compressed_path = save_png(img, self.tmp / "compressed.png")

        result = ImageVerifier.verify_lossless(original_path, compressed_path)
        self.assertTrue(result.is_lossless)
        self.assertEqual(result.max_difference, 0.0)

    def test_detect_pixel_change(self):
        """Upravený obrázek musí být detekován jako lossy."""
        from utils.verification import ImageVerifier
        img1 = make_rgb_image(32, 32, (0, 0, 0))
        img2 = make_rgb_image(32, 32, (10, 0, 0))

        original_path = save_png(img1, self.tmp / "orig.png")
        compressed_path = save_png(img2, self.tmp / "comp.png")

        result = ImageVerifier.verify_lossless(original_path, compressed_path)
        self.assertFalse(result.is_lossless)
        self.assertGreater(result.different_pixels, 0)

    def test_missing_original_returns_error(self):
        """Chybějící soubor → is_lossless=False a error_message."""
        from utils.verification import ImageVerifier
        ghost = self.tmp / "ghost.png"
        existing = save_png(make_rgb_image(), self.tmp / "existing.png")

        result = ImageVerifier.verify_lossless(ghost, existing)
        self.assertFalse(result.is_lossless)
        self.assertIsNotNone(result.error_message)

    def test_missing_compressed_returns_error(self):
        from utils.verification import ImageVerifier
        existing = save_png(make_rgb_image(), self.tmp / "existing.png")
        ghost = self.tmp / "ghost.png"

        result = ImageVerifier.verify_lossless(existing, ghost)
        self.assertFalse(result.is_lossless)

    def test_rgba_lossless(self):
        from utils.verification import ImageVerifier
        img = make_rgba_image(16, 16)
        path_a = save_png(img, self.tmp / "a.png")
        path_b = save_png(img, self.tmp / "b.png")

        result = ImageVerifier.verify_lossless(path_a, path_b)
        self.assertTrue(result.is_lossless)


# ===========================================================================
# 9. Testy create_difference_map
# ===========================================================================

class TestCreateDifferenceMap(TempDirMixin):

    def test_returns_none_for_missing_file(self):
        from utils.verification import ImageVerifier
        result = ImageVerifier.create_difference_map(
            self.tmp / "no.png",
            self.tmp / "no2.png",
        )
        self.assertIsNone(result)

    def test_identical_images_all_false(self):
        from utils.verification import ImageVerifier
        img = make_rgb_image(16, 16)
        p1 = save_png(img, self.tmp / "a.png")
        p2 = save_png(img, self.tmp / "b.png")

        diff_map = ImageVerifier.create_difference_map(p1, p2)
        self.assertIsNotNone(diff_map)
        self.assertFalse(np.any(diff_map))

    def test_different_images_some_true(self):
        from utils.verification import ImageVerifier
        img1 = make_rgb_image(16, 16, (0, 0, 0))
        img2 = make_rgb_image(16, 16, (255, 0, 0))
        p1 = save_png(img1, self.tmp / "a.png")
        p2 = save_png(img2, self.tmp / "b.png")

        diff_map = ImageVerifier.create_difference_map(p1, p2)
        self.assertIsNotNone(diff_map)
        self.assertTrue(np.any(diff_map))

    def test_output_shape(self):
        from utils.verification import ImageVerifier
        img = make_rgb_image(10, 20)
        p1 = save_png(img, self.tmp / "a.png")
        p2 = save_png(img, self.tmp / "b.png")

        diff_map = ImageVerifier.create_difference_map(p1, p2)
        self.assertEqual(diff_map.shape, (20, 10))


# ===========================================================================
# 10. Testy AbstractCompressor protokolu (přes dummy implementaci)
# ===========================================================================

class TestAbstractCompressorProtocol(TempDirMixin):
    """Ověří, že dummy kompresor respektuje rozhraní."""

    def _make_compressor(self):
        from main import ImageCompressor, CompressionMetrics, CompressionLevel
        class EchoCompressor(ImageCompressor):
            def _validate_dependencies(self): pass
            @property
            def name(self): return "Echo"
            @property
            def extension(self): return ".png"
            def compress(self, i, o, level=CompressionLevel.BALANCED):
                Image.open(i).save(o, format="PNG")
                size = i.stat().st_size
                return CompressionMetrics(size, o.stat().st_size, 1.0, 0.001, 0.001, True)
            def decompress(self, i, o):
                Image.open(i).save(o, format="PNG")
                return 0.001
        return EchoCompressor()

    def test_get_info_returns_dict(self):
        c = self._make_compressor()
        info = c.get_info()
        self.assertIn("name", info)
        self.assertIn("extension", info)
        self.assertIn("lib_path", info)

    def test_compress_returns_metrics(self):
        from main import CompressionMetrics, CompressionLevel
        c = self._make_compressor()
        img = make_rgb_image(32, 32)
        inp = save_png(img, self.tmp / "in.png")
        out = self.tmp / "out.png"
        metrics = c.compress(inp, out)
        self.assertIsInstance(metrics, CompressionMetrics)
        self.assertTrue(metrics.success)

    def test_decompress_returns_float(self):
        c = self._make_compressor()
        img = make_rgb_image(32, 32)
        inp = save_png(img, self.tmp / "in.png")
        out = self.tmp / "out.png"
        t = c.decompress(inp, out)
        self.assertIsInstance(t, float)
        self.assertGreaterEqual(t, 0.0)

    def test_name_and_extension(self):
        c = self._make_compressor()
        self.assertEqual(c.name, "Echo")
        self.assertEqual(c.extension, ".png")


# ===========================================================================
# 11. Testy PluginLoader
# ===========================================================================

class TestPluginLoader(TempDirMixin):

    def test_load_from_empty_directory(self):
        """Prázdný adresář nesmí způsobit výjimku."""
        from main import PluginLoader
        PluginLoader.load_plugins_from_directory(self.tmp)  # nic by nemělo selhat

    def test_load_creates_missing_directory(self):
        """Neexistující adresář by měl být vytvořen."""
        from main import PluginLoader
        new_dir = self.tmp / "plugins_new"
        self.assertFalse(new_dir.exists())
        PluginLoader.load_plugins_from_directory(new_dir)
        self.assertTrue(new_dir.exists())

    def test_load_valid_plugin(self):
        """Validní plugin soubor musí být nahrán bez chyby."""
        from main import PluginLoader, CompressorFactory
        original = dict(CompressorFactory._compressors)

        plugin_code = '''
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from main import ImageCompressor, CompressionMetrics, CompressionLevel, CompressorFactory
class _TestPlugin(ImageCompressor):
    def _validate_dependencies(self): pass
    @property
    def name(self): return "TestPlugin"
    @property
    def extension(self): return ".tp"
    def compress(self, i, o, level=CompressionLevel.BALANCED):
        return CompressionMetrics(0,0,0,0,0,True)
    def decompress(self, i, o): return 0.0
CompressorFactory.register("test_plugin", _TestPlugin)
'''
        plugin_file = self.tmp / "myplugin_compressor.py"
        plugin_file.write_text(plugin_code)

        try:
            PluginLoader.load_plugins_from_directory(self.tmp)
            self.assertIn("test_plugin", CompressorFactory.list_available())
        finally:
            CompressorFactory._compressors.clear()
            CompressorFactory._compressors.update(original)

    def test_broken_plugin_does_not_crash_loader(self):
        """Syntakticky chybný plugin nesmí způsobit pád celého loaderu."""
        from main import PluginLoader
        broken = self.tmp / "broken_compressor.py"
        broken.write_text("this is not valid python !!!")
        # Nesmí vyhodit výjimku:
        try:
            PluginLoader.load_plugins_from_directory(self.tmp)
        except Exception as e:
            self.fail(f"PluginLoader raised an exception on broken plugin: {e}")


# ===========================================================================
# 12. Testy ImageFinder (z benchmark_shared)
# ===========================================================================

class TestImageFinder(TempDirMixin):

    def _create_files(self, names):
        for name in names:
            (self.tmp / name).write_bytes(b"fake")

    def test_finds_png_files(self):
        from benchmark_shared import ImageFinder
        self._create_files(["a.png", "b.png", "c.txt"])
        result = ImageFinder.find_images(self.tmp, patterns=["*.png"], recursive=False)
        names = [p.name for p in result]
        self.assertIn("a.png", names)
        self.assertIn("b.png", names)
        self.assertNotIn("c.txt", names)

    def test_finds_multiple_formats(self):
        from benchmark_shared import ImageFinder
        self._create_files(["img.jpg", "img.bmp", "img.png"])
        result = ImageFinder.find_images(
            self.tmp,
            patterns=["*.jpg", "*.bmp", "*.png"],
            recursive=False,
        )
        self.assertEqual(len(result), 3)

    def test_recursive_search(self):
        from benchmark_shared import ImageFinder
        sub = self.tmp / "sub"
        sub.mkdir()
        (sub / "deep.png").write_bytes(b"fake")
        result = ImageFinder.find_images(self.tmp, patterns=["*.png"], recursive=True)
        names = [p.name for p in result]
        self.assertIn("deep.png", names)

    def test_non_recursive_misses_subdirs(self):
        from benchmark_shared import ImageFinder
        sub = self.tmp / "sub"
        sub.mkdir()
        (sub / "hidden.png").write_bytes(b"fake")
        result = ImageFinder.find_images(self.tmp, patterns=["*.png"], recursive=False)
        names = [p.name for p in result]
        self.assertNotIn("hidden.png", names)

    def test_empty_directory(self):
        from benchmark_shared import ImageFinder
        result = ImageFinder.find_images(self.tmp)
        self.assertEqual(result, [])

    def test_result_is_sorted(self):
        from benchmark_shared import ImageFinder
        self._create_files(["z.png", "a.png", "m.png"])
        result = ImageFinder.find_images(self.tmp, patterns=["*.png"], recursive=False)
        names = [p.name for p in result]
        self.assertEqual(names, sorted(names))

    def test_default_patterns_cover_common_formats(self):
        from benchmark_shared import ImageFinder
        self._create_files(["x.png", "x.jpg", "x.bmp", "x.tiff"])
        result = ImageFinder.find_images(self.tmp, recursive=False)
        self.assertGreaterEqual(len(result), 4)


# ===========================================================================
# 13. Testy BenchmarkConfig
# ===========================================================================

class TestBenchmarkConfig(unittest.TestCase):

    def _make_config(self, **kwargs):
        from benchmark_shared import BenchmarkConfig
        from main import CompressionLevel
        defaults = dict(
            dataset_dir=Path("/tmp"),
            output_dir=Path("/tmp"),
            libs_dir=Path("/tmp"),
            compressor_names=["dummy"],
            image_paths=[Path("img.png")],
            compression_levels=[CompressionLevel.BALANCED],
        )
        defaults.update(kwargs)
        return BenchmarkConfig(**defaults)

    def test_defaults(self):
        cfg = self._make_config()
        self.assertTrue(cfg.verify_lossless)
        self.assertTrue(cfg.strip_metadata)
        self.assertEqual(cfg.num_iterations, 1)
        self.assertEqual(cfg.warmup_iterations, 1)
        self.assertTrue(cfg.monitor_resources)
        self.assertFalse(cfg.isolate_process)

    def test_custom_values(self):
        cfg = self._make_config(num_iterations=5, warmup_iterations=2, verify_lossless=False)
        self.assertEqual(cfg.num_iterations, 5)
        self.assertEqual(cfg.warmup_iterations, 2)
        self.assertFalse(cfg.verify_lossless)


# ===========================================================================
# 14. Edge-case testy pro ImageSizeCalculator – okrajové vstupy
# ===========================================================================

class TestImageSizeCalculatorEdgeCases(TempDirMixin):

    def test_1x1_pixel(self):
        from image_size_calculator import ImageSizeCalculator
        img = Image.new("RGB", (1, 1), (255, 0, 0))
        path = self.tmp / "tiny.png"
        img.save(path)
        size = ImageSizeCalculator.calculate_uncompressed_size(path)
        self.assertEqual(size, 3)  # 1×1×3 bytes

    def test_large_image_size(self):
        from image_size_calculator import ImageSizeCalculator
        img = Image.new("RGB", (1000, 1000), (0, 128, 255))
        path = self.tmp / "big.png"
        img.save(path)
        size = ImageSizeCalculator.calculate_uncompressed_size(path)
        self.assertEqual(size, 1000 * 1000 * 3)

    def test_palette_image(self):
        """Palette (P) mód – 8 bitů na pixel."""
        from image_size_calculator import ImageSizeCalculator
        img = Image.new("P", (32, 32))
        path = self.tmp / "palette.png"
        img.save(path)
        info = ImageSizeCalculator.get_image_info(path)
        self.assertIsNotNone(info)
        # P má 1 kanál, 8 bitů
        self.assertEqual(info.channels, 1)
        self.assertEqual(info.bits_per_channel, 8)


# ===========================================================================
# Spuštění
# ===========================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)