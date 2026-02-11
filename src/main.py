from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Type, TYPE_CHECKING
import importlib.util
import sys
from enum import Enum

if TYPE_CHECKING:
    from utils.system_metrics import SystemMetrics


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class CompressionMetrics:
    """Metrics for a single compression/decompression run."""
    original_size: int          # bytes — velikost vstupu předaného kompresoru
    compressed_size: int        # bytes
    compression_ratio: float    # original / compressed
    compression_time: float     # seconds
    decompression_time: float   # seconds
    success: bool
    error_message: Optional[str] = None

    @property
    def space_saving_percent(self) -> float:
        """Percentage of space saved compared to the compressed input."""
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

    Obsahuje volitelně:
    - source_file_size: velikost originálního souboru NA DISKU (s metadaty)
    - system_metrics:   CPU/RAM/I/O metriky nasbírané během komprese
    """
    image_path: Path
    format_name: str
    metrics: CompressionMetrics
    metadata: Dict = field(default_factory=dict)

    # Velikost originálního souboru na disku (před strippováním metadat).
    # Pokud strip_metadata=False, shoduje se s metrics.original_size.
    source_file_size: int = 0

    # System metrics jsou sem přesunuty z benchmark_shared, kde způsobovaly
    # kolizi @dataclass dědičnosti (BenchmarkResult(BenchmarkResult)).
    # Typ je Optional[Any] aby main.py nezávisel na system_metrics modulu —
    # skutečný typ je utils.system_metrics.SystemMetrics.
    system_metrics: Optional[object] = None


class CompressionLevel(Enum):
    """Compression levels from fastest to smallest output."""
    FASTEST = 1
    BALANCED = 5
    BEST = 9


# ============================================================================
# ABSTRACT BASE CLASS
# ============================================================================

class ImageCompressor(ABC):
    """
    Abstraktní základ pro všechny kompresory.
    Každý plugin v adresáři compressors/ musí tuto třídu implementovat
    a zaregistrovat se přes CompressorFactory.register().
    """

    def __init__(self, lib_path: Optional[Path] = None):
        self.lib_path = lib_path
        self._validate_dependencies()

    @abstractmethod
    def _validate_dependencies(self) -> None:
        """Ověří dostupnost externích závislostí (knihoven, binárních nástrojů)."""
        pass

    @abstractmethod
    def compress(
        self,
        input_path: Path,
        output_path: Path,
        level: CompressionLevel = CompressionLevel.BALANCED,
    ) -> CompressionMetrics:
        """
        Zkomprimuje soubor a vrátí metriky.

        Args:
            input_path:  Cesta ke vstupnímu souboru (po případném strippování metadat).
            output_path: Cesta pro výstupní komprimovaný soubor.
            level:       Úroveň komprese.

        Returns:
            CompressionMetrics — včetně časů a poměru komprese.
        """
        pass

    @abstractmethod
    def decompress(self, input_path: Path, output_path: Path) -> float:
        """
        Dekomprimuje soubor.

        Args:
            input_path:  Cesta ke komprimovanému souboru.
            output_path: Cesta pro dekomprimovaný výstup.

        Returns:
            Čas dekomprese v sekundách.
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Identifikátor kompresoru, např. 'PNG', 'WebP', 'JPEG-LS'."""
        pass

    @property
    @abstractmethod
    def extension(self) -> str:
        """Přípona výstupního souboru, např. '.png', '.webp', '.jls'."""
        pass

    def get_info(self) -> Dict:
        return {
            "name": self.name,
            "extension": self.extension,
            "lib_path": str(self.lib_path) if self.lib_path else None,
        }


# ============================================================================
# FACTORY & PLUGIN LOADER
# ============================================================================

class CompressorFactory:
    """
    Registr kompresů a továrna na jejich instance.

    Pluginy se registrují voláním:
        CompressorFactory.register("MyCodec", MyCodecCompressor)
    """

    _compressors: Dict[str, Type[ImageCompressor]] = {}

    @classmethod
    def register(cls, name: str, compressor_class: Type[ImageCompressor]) -> None:
        cls._compressors[name] = compressor_class

    @classmethod
    def create(cls, name: str, lib_path: Optional[Path] = None) -> ImageCompressor:
        if name not in cls._compressors:
            raise ValueError(f"Neznámý kompresor: '{name}'. "
                             f"Dostupné: {list(cls._compressors)}")
        return cls._compressors[name](lib_path)

    @classmethod
    def list_available(cls) -> List[str]:
        return list(cls._compressors.keys())

    @classmethod
    def get_by_extension(cls, extension: str) -> Optional[Type[ImageCompressor]]:
        """
        Vrátí třídu kompresoru podle přípony souboru.
        Používá verification.py místo lazy importu z main.
        """
        for compressor_class in cls._compressors.values():
            # Vytvoříme dočasnou instanci jen kvůli atributu extension.
            # Třídy by neměly mít side-effecty v __init__ před validate_dependencies,
            # ale pro jistotu obalíme try/except.
            try:
                instance = compressor_class(lib_path=None)
                if instance.extension == extension:
                    return compressor_class
            except Exception:
                continue
        return None


class PluginLoader:
    """Dynamicky načítá kompresory z adresáře s pluginy."""

    @staticmethod
    def load_plugins_from_directory(plugin_dir: Path) -> None:
        """
        Načte všechny soubory *_compressor.py z plugin_dir.

        Každý plugin je zodpovědný za zavolání
            CompressorFactory.register(...)
        při načtení modulu.
        """
        if not plugin_dir.exists():
            plugin_dir.mkdir(parents=True)
            return

        plugin_files = sorted(plugin_dir.glob("*_compressor.py"))
        if not plugin_files:
            return

        for plugin_file in plugin_files:
            PluginLoader._load_plugin_module(plugin_file)

    @staticmethod
    def _load_plugin_module(plugin_path: Path) -> None:
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


# ============================================================================
# MAIN (GUI entry point)
# ============================================================================

def main() -> None:
    """Launch GUI application"""
    try:
        from gui import main as gui_main
        gui_main()
    except ImportError as e:
        print(f"Error: Could not import GUI module: {e}")
        print("Make sure gui.py is in the same directory as main.py")
        sys.exit(1)


if __name__ == "__main__":
    main()