"""
Shared fixtures and module stubs for compressor unit tests.
"""

import sys
import types
from pathlib import Path

import pytest
from PIL import Image


# ---------------------------------------------------------------------------
# Module stubs, registered once before any test file is imported.
# All test files rely on these.
# ---------------------------------------------------------------------------

def _build_main_stub() -> types.ModuleType:
    stub = types.ModuleType("main")

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

    stub.CompressionLevel = CompressionLevel
    stub.CompressionMetrics = CompressionMetrics
    stub.ImageCompressor = ImageCompressor
    stub.CompressorFactory = CompressorFactory
    return stub


def _build_image_size_stub() -> types.ModuleType:
    stub = types.ModuleType("image_size_calculator")

    class ImageSizeCalculator:
        @staticmethod
        def calculate_uncompressed_size(path):
            return 1_000_000  # 1 MB

    stub.ImageSizeCalculator = ImageSizeCalculator
    return stub


# Register stubs unconditionally so every test file sees the same objects.
# If the real modules exist in the environment, they take precedence only if
# already present – but in the test environment they never are.
if "main" not in sys.modules:
    sys.modules["main"] = _build_main_stub()

if "image_size_calculator" not in sys.modules:
    sys.modules["image_size_calculator"] = _build_image_size_stub()


# Re-export CompressionLevel so test files can import it from conftest
# via `from conftest import CompressionLevel` if needed.  Most tests simply
# reference sys.modules["main"].CompressionLevel, but the explicit attribute
# below keeps things readable.
CompressionLevel = sys.modules["main"].CompressionLevel


# ---------------------------------------------------------------------------
# Shared image-file fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def rgb_png(tmp_path) -> Path:
    """Small 4×4 RGB PNG file."""
    p = tmp_path / "rgb.png"
    Image.new("RGB", (4, 4), color=(100, 150, 200)).save(p, format="PNG")
    return p


@pytest.fixture()
def rgba_png(tmp_path) -> Path:
    """Small 4×4 RGBA PNG file."""
    p = tmp_path / "rgba.png"
    Image.new("RGBA", (4, 4), color=(100, 150, 200, 128)).save(p, format="PNG")
    return p


@pytest.fixture()
def grayscale_png(tmp_path) -> Path:
    """Small 4×4 grayscale PNG file."""
    p = tmp_path / "gray.png"
    Image.new("L", (4, 4), color=128).save(p, format="PNG")
    return p