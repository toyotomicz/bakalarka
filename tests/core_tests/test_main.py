"""
Unit tests for main.py.

Covers CompressionMetrics computed properties, BenchmarkResult construction,
CompressionLevel enum values, CompressorFactory registry/factory/query methods,
ImageCompressor abstract contract, and PluginLoader file discovery.
"""

import sys
import types
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

from main import (
    BenchmarkResult,
    CompressionLevel,
    CompressionMetrics,
    CompressorFactory,
    ImageCompressor,
    PluginLoader,
)


# Helpers

def _make_metrics(
    original_size: int = 1_000_000,
    compressed_size: int = 500_000,
    compression_ratio: float = 2.0,
    compression_time: float = 1.0,
    decompression_time: float = 0.5,
    success: bool = True,
    error_message: Optional[str] = None,
) -> CompressionMetrics:
    """
    Create a CompressionMetrics instance with sensible defaults.

    Args:
        original_size: Uncompressed size in bytes.
        compressed_size: Compressed output size in bytes.
        compression_ratio: Ratio of original to compressed size.
        compression_time: Wall-clock compression duration in seconds.
        decompression_time: Wall-clock decompression duration in seconds.
        success: Whether compression succeeded.
        error_message: Optional error detail when success is False.

    Returns:
        A populated CompressionMetrics instance.
    """
    return CompressionMetrics(
        original_size=original_size,
        compressed_size=compressed_size,
        compression_ratio=compression_ratio,
        compression_time=compression_time,
        decompression_time=decompression_time,
        success=success,
        error_message=error_message,
    )


def _make_concrete_compressor(name: str = "Test", extension: str = ".tst") -> type:
    """
    Build a minimal concrete ImageCompressor subclass for testing.

    Args:
        name: Value returned by the name property.
        extension: Value returned by the extension property.

    Returns:
        A concrete subclass of ImageCompressor that registers nothing and
        whose compress()/decompress() return trivial values.
    """
    class _Concrete(ImageCompressor):
        def _validate_dependencies(self): pass

        def compress(self, input_path, output_path, level=CompressionLevel.BALANCED):
            return _make_metrics()

        def decompress(self, input_path, output_path):
            return 0.0

        @property
        def name(self):
            return name

        @property
        def extension(self):
            return extension

    return _Concrete


# CompressionMetrics - computed properties

class TestCompressionMetricsProperties:
    """Verify all computed properties of CompressionMetrics."""

    def test_space_saving_percent_normal(self):
        m = _make_metrics(original_size=1_000_000, compressed_size=600_000)
        assert m.space_saving_percent == pytest.approx(40.0)

    def test_space_saving_percent_zero_original(self):
        m = _make_metrics(original_size=0, compressed_size=0)
        assert m.space_saving_percent == 0.0

    def test_space_saving_percent_no_saving(self):
        """When compressed_size equals original_size, savings must be 0%."""
        m = _make_metrics(original_size=500, compressed_size=500)
        assert m.space_saving_percent == pytest.approx(0.0)

    def test_compression_speed_mbps(self):
        """1 MB compressed in 0.5 s should yield 2.0 MB/s."""
        m = _make_metrics(original_size=1_048_576, compression_time=0.5)
        assert m.compression_speed_mbps == pytest.approx(2.0)

    def test_compression_speed_mbps_zero_time(self):
        m = _make_metrics(compression_time=0.0)
        assert m.compression_speed_mbps == 0.0

    def test_decompression_speed_mbps(self):
        """1 MB decompressed in 0.25 s should yield 4.0 MB/s."""
        m = _make_metrics(original_size=1_048_576, decompression_time=0.25)
        assert m.decompression_speed_mbps == pytest.approx(4.0)

    def test_decompression_speed_mbps_zero_time(self):
        m = _make_metrics(decompression_time=0.0)
        assert m.decompression_speed_mbps == 0.0

    def test_space_saving_percent_expansion(self):
        """When the output is larger than the input, savings must be negative."""
        m = _make_metrics(original_size=100, compressed_size=200)
        assert m.space_saving_percent < 0.0


# CompressionMetrics - field values

class TestCompressionMetricsFields:
    """Verify that CompressionMetrics stores field values correctly."""

    def test_success_true(self):
        assert _make_metrics(success=True).success is True

    def test_success_false_with_message(self):
        m = _make_metrics(success=False, error_message="disk full")
        assert m.success is False
        assert m.error_message == "disk full"

    def test_error_message_defaults_to_none(self):
        assert _make_metrics().error_message is None


# BenchmarkResult

class TestBenchmarkResult:
    """Verify construction and optional fields of BenchmarkResult."""

    def test_basic_construction(self, tmp_path):
        p = tmp_path / "img.png"
        m = _make_metrics()
        result = BenchmarkResult(image_path=p, format_name="TestFmt", metrics=m)

        assert result.image_path == p
        assert result.format_name == "TestFmt"
        assert result.metrics is m

    def test_metadata_defaults_to_empty_dict(self, tmp_path):
        result = BenchmarkResult(
            image_path=tmp_path / "img.png",
            format_name="X",
            metrics=_make_metrics(),
        )
        assert result.metadata == {}

    def test_system_metrics_defaults_to_none(self, tmp_path):
        result = BenchmarkResult(
            image_path=tmp_path / "img.png",
            format_name="X",
            metrics=_make_metrics(),
        )
        assert result.system_metrics is None

    def test_source_file_size_defaults_to_zero(self, tmp_path):
        result = BenchmarkResult(
            image_path=tmp_path / "img.png",
            format_name="X",
            metrics=_make_metrics(),
        )
        assert result.source_file_size == 0

    def test_metadata_stored(self, tmp_path):
        result = BenchmarkResult(
            image_path=tmp_path / "img.png",
            format_name="X",
            metrics=_make_metrics(),
            metadata={"compression_level": "BEST"},
        )
        assert result.metadata["compression_level"] == "BEST"


# CompressionLevel

class TestCompressionLevel:
    """Verify CompressionLevel enum values and ordering."""

    def test_fastest_value(self):
        assert CompressionLevel.FASTEST.value == 1

    def test_balanced_value(self):
        assert CompressionLevel.BALANCED.value == 5

    def test_best_value(self):
        assert CompressionLevel.BEST.value == 9

    def test_levels_are_monotonically_ordered(self):
        assert (
            CompressionLevel.FASTEST.value
            < CompressionLevel.BALANCED.value
            < CompressionLevel.BEST.value
        )

    def test_all_three_levels_exist(self):
        names = {level.name for level in CompressionLevel}
        assert {"FASTEST", "BALANCED", "BEST"}.issubset(names)


# CompressorFactory - register / create / list

class TestCompressorFactoryRegisterAndCreate:
    """Verify register(), create(), and list_available() on CompressorFactory."""

    def setup_method(self):
        """Snapshot and clear the registry before each test."""
        self._saved = dict(CompressorFactory._compressors)
        CompressorFactory._compressors.clear()

    def teardown_method(self):
        """Restore the original registry after each test."""
        CompressorFactory._compressors.clear()
        CompressorFactory._compressors.update(self._saved)

    def test_register_and_list(self):
        Cls = _make_concrete_compressor("Foo", ".foo")
        CompressorFactory.register("foo", Cls)
        assert "foo" in CompressorFactory.list_available()

    def test_create_returns_correct_instance(self):
        Cls = _make_concrete_compressor("Bar", ".bar")
        CompressorFactory.register("bar", Cls)
        instance = CompressorFactory.create("bar")
        assert isinstance(instance, Cls)

    def test_create_raises_for_unknown_key(self):
        with pytest.raises(ValueError, match="unknown_key"):
            CompressorFactory.create("unknown_key")

    def test_list_available_returns_list_of_strings(self):
        Cls = _make_concrete_compressor()
        CompressorFactory.register("z", Cls)
        result = CompressorFactory.list_available()
        assert isinstance(result, list)
        assert all(isinstance(k, str) for k in result)

    def test_register_overwrites_existing_key(self):
        """Registering the same key twice replaces the old mapping."""
        Cls1 = _make_concrete_compressor("First", ".f1")
        Cls2 = _make_concrete_compressor("Second", ".f2")
        CompressorFactory.register("dup", Cls1)
        CompressorFactory.register("dup", Cls2)
        assert isinstance(CompressorFactory.create("dup"), Cls2)

    def test_create_with_lib_path_forwarded(self, tmp_path):
        """lib_path passed to create() must reach the compressor constructor."""
        received = []

        class _TrackLib(ImageCompressor):
            def __init__(self, lib_path=None):
                received.append(lib_path)

            def _validate_dependencies(self): pass
            def compress(self, i, o, level=CompressionLevel.BALANCED): return _make_metrics()
            def decompress(self, i, o): return 0.0

            @property
            def name(self): return "TrackLib"

            @property
            def extension(self): return ".tl"

        CompressorFactory.register("tracklib", _TrackLib)
        p = tmp_path / "lib.dll"
        CompressorFactory.create("tracklib", lib_path=p)
        assert received[0] == p


# CompressorFactory - get_by_extension

class TestCompressorFactoryGetByExtension:
    """Verify get_by_extension() matches by extension and handles edge cases."""

    def setup_method(self):
        self._saved = dict(CompressorFactory._compressors)
        CompressorFactory._compressors.clear()

    def teardown_method(self):
        CompressorFactory._compressors.clear()
        CompressorFactory._compressors.update(self._saved)

    def test_returns_correct_class(self):
        Cls = _make_concrete_compressor("Ext", ".xyz")
        CompressorFactory.register("exttest", Cls)
        result = CompressorFactory.get_by_extension(".xyz")
        assert result is Cls

    def test_returns_none_for_unknown_extension(self):
        result = CompressorFactory.get_by_extension(".doesnotexist")
        assert result is None

    def test_skips_class_whose_init_raises(self):
        """A compressor that raises during instantiation must be skipped silently."""
        class _Broken(ImageCompressor):
            def __init__(self, lib_path=None):
                raise RuntimeError("missing DLL")

            def _validate_dependencies(self): pass
            def compress(self, i, o, level=CompressionLevel.BALANCED): return _make_metrics()
            def decompress(self, i, o): return 0.0

            @property
            def name(self): return "Broken"

            @property
            def extension(self): return ".broken"

        CompressorFactory.register("broken", _Broken)
        # Must not raise, must return None because the only candidate errored
        result = CompressorFactory.get_by_extension(".broken")
        assert result is None


# ImageCompressor - abstract contract

class TestImageCompressorAbstractContract:
    """Verify that ImageCompressor cannot be instantiated without all abstract members."""

    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            ImageCompressor()  # type: ignore[abstract]

    def test_concrete_subclass_is_instantiable(self):
        Cls = _make_concrete_compressor()
        instance = Cls()
        assert isinstance(instance, ImageCompressor)

    def test_get_info_returns_dict_with_required_keys(self):
        Cls = _make_concrete_compressor("MyComp", ".mc")
        instance = Cls()
        info = instance.get_info()
        assert "name" in info
        assert "extension" in info
        assert "lib_path" in info

    def test_get_info_name_matches_property(self):
        Cls = _make_concrete_compressor("MyComp", ".mc")
        instance = Cls()
        assert instance.get_info()["name"] == "MyComp"

    def test_get_info_extension_matches_property(self):
        Cls = _make_concrete_compressor("X", ".qx")
        instance = Cls()
        assert instance.get_info()["extension"] == ".qx"

    def test_get_info_lib_path_none_when_not_set(self):
        Cls = _make_concrete_compressor()
        instance = Cls()
        assert instance.get_info()["lib_path"] is None

    def test_validate_dependencies_called_on_init(self):
        """_validate_dependencies() must be invoked automatically by __init__."""
        called = []

        class _Track(ImageCompressor):
            def _validate_dependencies(self):
                called.append(True)

            def compress(self, i, o, level=CompressionLevel.BALANCED): return _make_metrics()
            def decompress(self, i, o): return 0.0

            @property
            def name(self): return "Track"

            @property
            def extension(self): return ".tr"

        _Track()
        assert called == [True]

    def test_missing_compress_raises_type_error(self):
        """Omitting compress() must prevent instantiation."""
        with pytest.raises(TypeError):
            class _NoCompress(ImageCompressor):
                def _validate_dependencies(self): pass
                def decompress(self, i, o): return 0.0

                @property
                def name(self): return "X"

                @property
                def extension(self): return ".x"

            _NoCompress()


# PluginLoader

class TestPluginLoader:
    """Verify PluginLoader file discovery and error isolation."""

    def test_creates_directory_if_missing(self, tmp_path):
        missing = tmp_path / "plugins"
        PluginLoader.load_plugins_from_directory(missing)
        assert missing.exists()

    def test_loads_valid_plugin_file(self, tmp_path):
        """A valid *_compressor.py module must be importable by the loader."""
        plugin = tmp_path / "dummy_compressor.py"
        plugin.write_text("LOADED = True\n")

        module_name = "plugin_dummy_compressor"
        # Clean up any previous import of this name
        sys.modules.pop(module_name, None)

        PluginLoader.load_plugins_from_directory(tmp_path)

        assert module_name in sys.modules
        assert sys.modules[module_name].LOADED is True

    def test_ignores_non_compressor_files(self, tmp_path):
        """Files not matching *_compressor.py must not be imported."""
        (tmp_path / "helper.py").write_text("SHOULD_NOT_LOAD = True\n")
        PluginLoader.load_plugins_from_directory(tmp_path)
        assert "plugin_helper" not in sys.modules

    def test_bad_plugin_does_not_prevent_others_loading(self, tmp_path):
        """A plugin that raises on import must not stop subsequent plugins from loading."""
        bad = tmp_path / "bad_compressor.py"
        bad.write_text("raise RuntimeError('broken plugin')\n")

        good = tmp_path / "good_compressor.py"
        good.write_text("GOOD = True\n")

        sys.modules.pop("plugin_bad_compressor", None)
        sys.modules.pop("plugin_good_compressor", None)

        PluginLoader.load_plugins_from_directory(tmp_path)

        assert "plugin_good_compressor" in sys.modules
        assert sys.modules["plugin_good_compressor"].GOOD is True

    def test_empty_directory_is_noop(self, tmp_path):
        """An empty plugins directory must not raise any exception."""
        PluginLoader.load_plugins_from_directory(tmp_path)

    def test_plugins_loaded_in_sorted_order(self, tmp_path):
        """Plugins must be loaded in alphabetical order so registration is deterministic."""
        order = []

        for letter in ("c", "a", "b"):
            (tmp_path / f"{letter}_compressor.py").write_text(
                f"import sys; sys.modules.setdefault('_load_order', []).append('{letter}')\n"
            )
            sys.modules.pop(f"plugin_{letter}_compressor", None)

        sys.modules.pop("_load_order", None)

        PluginLoader.load_plugins_from_directory(tmp_path)

        loaded = sys.modules.get("_load_order", [])
        assert loaded == sorted(loaded), "Plugins must be loaded in sorted (alphabetical) order"