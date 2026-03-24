"""
Core data structures, abstract base classes, and the plugin system.

This module is the single source of truth for:
    - CompressionMetrics  : timing and size data for one compression run.
    - BenchmarkResult     : wraps metrics with image path, format name, and optional
                            system resource data.
    - CompressionLevel    : enum used by every compressor plugin.
    - ImageCompressor     : abstract base class every plugin must implement.
    - CompressorFactory   : registry and factory for compressor instances.
    - PluginLoader        : dynamically imports *_compressor.py files at startup.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Type
import importlib.util
import sys

if TYPE_CHECKING:
    # Imported for type hints only, it avoids a circular dependency at runtime.
    from utils.system_metrics import SystemMetrics


# Data structures

@dataclass
class CompressionMetrics:
    """
    Timing and size data produced by a single compress() call.

    Attributes:
        original_size: Uncompressed pixel data size in bytes.
        compressed_size: Size of the output file on disk in bytes.
        compression_ratio: Ratio of original_size to compressed_size.
        compression_time: Wall-clock compression duration in seconds.
        decompression_time: Wall-clock decompression duration in seconds.
        success: True when compression completed without error.
        error_message: Human-readable error detail when success is False.
    """

    original_size: int
    compressed_size: int
    compression_ratio: float
    compression_time: float
    decompression_time: float
    success: bool
    error_message: Optional[str] = None

    @property
    def space_saving_percent(self) -> float:
        """Percentage of space saved relative to the uncompressed input."""
        if self.original_size == 0:
            return 0.0
        return (1.0 - self.compressed_size / self.original_size) * 100.0

    @property
    def compression_speed_mbps(self) -> float:
        """Compression throughput in MB/s."""
        if self.compression_time == 0:
            return 0.0
        return (self.original_size / (1024 * 1024)) / self.compression_time

    @property
    def decompression_speed_mbps(self) -> float:
        """Decompression throughput in MB/s."""
        if self.decompression_time == 0:
            return 0.0
        return (self.original_size / (1024 * 1024)) / self.decompression_time


@dataclass
class BenchmarkResult:
    """
    Result of a single benchmark run.

    Attributes:
        image_path: Path to the source image file that was compressed.
        format_name: Human-readable compressor name, e.g. 'CharLS-JPEGLS'.
        metrics: Timing, size, and success data for this run.
        metadata: Arbitrary key-value annotations (e.g. compression_level name).
        source_file_size: On-disk size of the original file including metadata.
            Equals metrics.original_size when strip_metadata is False.
        system_metrics: CPU / RAM / I/O data collected during compression.
            The concrete type is utils.system_metrics.SystemMetrics; typed as
            Optional[object] here to avoid importing that module from main.py
            and creating a circular dependency at import time.
    """

    image_path: Path
    format_name: str
    metrics: CompressionMetrics
    metadata: Dict = field(default_factory=dict)
    source_file_size: int = 0
    system_metrics: Optional[object] = None  # utils.system_metrics.SystemMetrics


class CompressionLevel(Enum):
    """
    Compression effort levels, from fastest encode to smallest output.

    Values match typical zlib / libpng level conventions (1 = fastest, 9 = best)
    so that plugins can pass the enum's value directly to underlying libraries.
    """

    FASTEST  = 1
    BALANCED = 5
    BEST     = 9


# Abstract base class

class ImageCompressor(ABC):
    """
    Abstract base for all compressor plugins.

    Every plugin in the compressors/ directory must:
        1) Subclass ImageCompressor.
        2) Implement all abstract methods and properties.
        3) Register itself at module level via: CompressorFactory.register("my_key", MyCompressor)

    The constructor calls _validate_dependencies() automatically. 
    Subclasses must therefore set any instance attributes they need (e.g. binary paths)
    *before* calling super().__init__(), or override __init__ entirely and call
    _validate_dependencies() themselves at the end.

    Args:
        lib_path: Optional path to a native library or CLI binary required by
            this compressor. None means the compressor locates its dependency
            via PATH or a hard-coded default.
    """

    def __init__(self, lib_path: Optional[Path] = None):
        self.lib_path = lib_path
        self._validate_dependencies()

    @abstractmethod
    def _validate_dependencies(self) -> None:
        """
        Verify that all external dependencies are present and accessible.

        Check for DLLs, CLI binaries, Python packages, or any other
        requirement. Called automatically by __init__.

        Raises:
            RuntimeError: If a required dependency is missing, with a message
                that names the missing item and suggests how to install it.
        """

    @abstractmethod
    def compress(
        self,
        input_path: Path,
        output_path: Path,
        level: CompressionLevel = CompressionLevel.BALANCED,
    ) -> CompressionMetrics:
        """
        Compress input_path to output_path and return timing / size metrics.

        Args:
            input_path: Source image file (metadata-stripped PNG when
                BenchmarkConfig.strip_metadata is True).
            output_path: Destination for the compressed output file.
            level: Compression effort level.

        Returns:
            CompressionMetrics with timing, sizes, and success flag.
        """

    @abstractmethod
    def decompress(self, input_path: Path, output_path: Path) -> float:
        """
        Decompress input_path to output_path and return elapsed time.

        Args:
            input_path: Compressed source file.
            output_path: Destination for the decoded output (always PNG).

        Returns:
            Wall clock decompression time in seconds.
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Human-readable compressor name used in reports.

        Example: 'CharLS-JPEGLS'
        """

    @property
    @abstractmethod
    def extension(self) -> str:
        """
        Output file extension including the leading dot.

        Example: '.jls'
        """

    def get_info(self) -> Dict:
        """
        Return a simple dict describing this compressor (used for logging).

        Returns:
            Dict with keys 'name', 'extension', and 'lib_path'.
        """
        return {
            "name":      self.name,
            "extension": self.extension,
            "lib_path":  str(self.lib_path) if self.lib_path else None,
        }


# Factory and plugin loader

class CompressorFactory:
    """
    Registry and factory for ImageCompressor subclasses.

    Plugins self-register at import time:
        CompressorFactory.register("charls", CharLSCompressor)

    The GUI / CLI then creates instances on demand:
        compressor = CompressorFactory.create("charls")
    """

    # Class-level registry: factory_key -> compressor class
    _compressors: Dict[str, Type[ImageCompressor]] = {}

    @classmethod
    def register(cls, name: str, compressor_class: Type[ImageCompressor]) -> None:
        """
        Register a compressor class under the given factory key.

        Args:
            name: Short identifier used in config files and CLI flags,
                e.g. 'charls' or 'webp'.
            compressor_class: The ImageCompressor subclass to register.
        """
        cls._compressors[name] = compressor_class

    @classmethod
    def create(cls, name: str, lib_path: Optional[Path] = None) -> ImageCompressor:
        """
        Instantiate and return a registered compressor.

        Args:
            name: Factory key previously passed to register().
            lib_path: Optional library / binary path forwarded to the compressor's constructor.

        Returns:
            A fully initialized ImageCompressor instance.

        Raises:
            ValueError: If name has not been registered.
        """
        if name not in cls._compressors:
            raise ValueError(
                f"Unknown compressor: '{name}'. "
                f"Available: {list(cls._compressors)}"
            )
        return cls._compressors[name](lib_path)

    @classmethod
    def list_available(cls) -> List[str]:
        """
        Return all registered compressor factory keys.

        Returns:
            List of string keys in registration order.
        """
        return list(cls._compressors.keys())

    @classmethod
    def get_by_extension(cls, extension: str) -> Optional[Type[ImageCompressor]]:
        """
        Return the compressor class that produces files with the given extension.

        Temporarily instantiates each registered class to read its ``extension`` property.  
        Returns None if no match is found.

        Args:
            extension: File extension to search for, including the leading dot
                (e.g. '.jls', '.png').

        Returns:
            The matching ImageCompressor subclass, or None.

        Note:
            This method instantiates every compressor in the registry, which
            triggers _validate_dependencies() for each one. 
            Prefer caching the result when calling this in a tight loop.
        """
        for compressor_class in cls._compressors.values():
            try:
                instance = compressor_class(lib_path=None)
                if instance.extension == extension:
                    return compressor_class
            except Exception:
                # Compressor failed to initialise (missing dependency etc.)
                # skip it and keep searching
                continue
        return None


class PluginLoader:
    """Dynamically loads compressor plugins from a directory at startup."""

    @staticmethod
    def load_plugins_from_directory(plugin_dir: Path) -> None:
        """
        Import all *_compressor.py files found in plugin_dir.

        Each plugin is responsible for calling CompressorFactory.register()
        at module level so it is available immediately after import.
        The directory is created (empty) if it does not yet exist.

        Args:
            plugin_dir: Directory to scan for plugin files.
        """
        if not plugin_dir.exists():
            plugin_dir.mkdir(parents=True)
            return

        for plugin_file in sorted(plugin_dir.glob("*_compressor.py")):
            PluginLoader._load_plugin_module(plugin_file)

    @staticmethod
    def _load_plugin_module(plugin_path: Path) -> None:
        """
        Import a single plugin file as an isolated module.

        Errors are intentionally swallowed so that one broken plugin does not
        prevent the rest from loading.  The factory simply won't list a
        compressor whose plugin failed to import.

        Args:
            plugin_path: Absolute path to the *_compressor.py file.
        """
        module_name = f"plugin_{plugin_path.stem}"
        try:
            spec = importlib.util.spec_from_file_location(module_name, plugin_path)
            if spec is None or spec.loader is None:
                return

            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)  # type: ignore[union-attr]

        except Exception:
            pass


# Entry point

def main() -> None:
    """Launch the GUI application."""
    try:
        from gui import main as gui_main
        gui_main()
    except ImportError as exc:
        print(f"Error: Could not import GUI module: {exc}")
        print("Make sure gui.py is in the same directory as main.py")
        sys.exit(1)


if __name__ == "__main__":
    main()