"""Shared fixtures and module stubs for compressor unit tests."""

import sys
import types
from pathlib import Path

import pytest
from PIL import Image


def _build_main_stub() -> types.ModuleType:
    """
    Build a minimal stub for the ``main`` module.

    Returns:
        A synthetic ``main`` module containing ``CompressionLevel``,
        ``CompressionMetrics``, ``BenchmarkResult``, ``ImageCompressor``,
        and ``CompressorFactory``.
    """
    stub = types.ModuleType("main")

    class CompressionLevel:
        FASTEST = "fastest"
        BALANCED = "balanced"
        BEST = "best"

    class CompressionMetrics:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    class BenchmarkResult:
        def __init__(
            self,
            image_path=None,
            format_name="",
            metrics=None,
            metadata=None,
            source_file_size=0,
            system_metrics=None,
        ):
            self.image_path = image_path
            self.format_name = format_name
            self.metrics = metrics
            self.metadata = metadata if metadata is not None else {}
            self.source_file_size = source_file_size
            self.system_metrics = system_metrics

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
        _registry: dict = {}

        @classmethod
        def register(cls, key, klass):
            cls._registry[key] = klass

        @classmethod
        def create(cls, key):
            return cls._registry[key]()

    class PluginLoader:
        @staticmethod
        def load_plugins_from_directory(plugin_dir):
            pass

    stub.CompressionLevel = CompressionLevel
    stub.CompressionMetrics = CompressionMetrics
    stub.BenchmarkResult = BenchmarkResult
    stub.ImageCompressor = ImageCompressor
    stub.CompressorFactory = CompressorFactory
    stub.PluginLoader = PluginLoader
    return stub


def _build_image_size_stub() -> types.ModuleType:
    """
    Build a minimal stub for the ``image_size_calculator`` module.

    Returns:
        A synthetic module whose ``ImageSizeCalculator.calculate_uncompressed_size``
        always returns 1 MB so tests are not affected by real file I/O.
    """
    stub = types.ModuleType("image_size_calculator")

    class ImageSizeCalculator:
        @staticmethod
        def calculate_uncompressed_size(path):
            return 1_000_000  # 1 MB

    stub.ImageSizeCalculator = ImageSizeCalculator
    return stub


# Register stubs only when the real modules cannot be imported.
# This prevents the stub from shadowing the real main.py when core_tests
# are collected in the same pytest session.
try:
    import main as _real_main  # noqa: F401
except ImportError:
    sys.modules["main"] = _build_main_stub()

try:
    import image_size_calculator as _real_isc  # noqa: F401
except ImportError:
    sys.modules["image_size_calculator"] = _build_image_size_stub()

# Re-export CompressionLevel so test files can reference it directly without importing from main.
CompressionLevel = sys.modules["main"].CompressionLevel


# Shared image-file fixtures

@pytest.fixture()
def rgb_png(tmp_path) -> Path:
    """Create a small 4x4 RGB PNG file.

    Args:
        tmp_path: Pytest-provided temporary directory.

    Returns:
        Path to the generated PNG file.
    """
    p = tmp_path / "rgb.png"
    Image.new("RGB", (4, 4), color=(100, 150, 200)).save(p, format="PNG")
    return p


@pytest.fixture()
def rgba_png(tmp_path) -> Path:
    """Create a small 4x4 RGBA PNG file.

    Args:
        tmp_path: Pytest-provided temporary directory.

    Returns:
        Path to the generated PNG file.
    """
    p = tmp_path / "rgba.png"
    Image.new("RGBA", (4, 4), color=(100, 150, 200, 128)).save(p, format="PNG")
    return p


@pytest.fixture()
def grayscale_png(tmp_path) -> Path:
    """Create a small 4x4 grayscale PNG file.

    Args:
        tmp_path: Pytest-provided temporary directory.

    Returns:
        Path to the generated PNG file.
    """
    p = tmp_path / "gray.png"
    Image.new("L", (4, 4), color=128).save(p, format="PNG")
    return p