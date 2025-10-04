from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Type
import time
import subprocess
import json
from enum import Enum
import importlib.util
import sys


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class CompressionMetrics:
    """Metrics for compression/decompression results"""
    original_size: int  # bytes
    compressed_size: int  # bytes
    compression_ratio: float  # original/compressed
    compression_time: float  # seconds
    decompression_time: float  # seconds
    success: bool
    error_message: Optional[str] = None
    
    @property
    def space_saving_percent(self) -> float:
        """Percentage of space saved"""
        return (1 - (self.compressed_size / self.original_size)) * 100
    
    @property
    def compression_speed_mbps(self) -> float:
        """Speed of compression in MB/s"""
        if self.compression_time == 0:
            return 0
        return (self.original_size / (1024 * 1024)) / self.compression_time
    
    @property
    def decompression_speed_mbps(self) -> float:
        """Speed of decompression in MB/s"""
        if self.decompression_time == 0:
            return 0
        return (self.original_size / (1024 * 1024)) / self.decompression_time


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run"""
    image_path: Path
    format_name: str
    metrics: CompressionMetrics
    metadata: Dict | None = None


class CompressionLevel(Enum):
    """Compression levels from fastest to best quality"""
    FASTEST = 1
    FAST = 3
    BALANCED = 5
    GOOD = 7
    BEST = 9


# ============================================================================
# ABSTRACT BASE CLASS FOR COMPRESSORS
# ============================================================================

class ImageCompressor(ABC):
    """
    Abstract base class for all image compressors.
    Defines a standardized API.
    """
    
    def __init__(self, lib_path: Optional[Path] = None):
        """
        Args:
            lib_path: Path to the C/C++ library if needed
        """
        self.lib_path = lib_path
        self._validate_dependencies()
    
    @abstractmethod
    def _validate_dependencies(self) -> None:
        """Check if required dependencies are available"""
        pass
    
    @abstractmethod
    def compress(self, 
                input_path: Path, 
                output_path: Path,
                level: CompressionLevel = CompressionLevel.BALANCED) -> CompressionMetrics:
        """
        Compresses an image and returns metrics.
        
        Args:
            input_path: Path to the input image
            output_path: Path where to save the compressed output
            level: Compression level
            
        Returns:
            CompressionMetrics with measurement results
        """
        pass
    
    @abstractmethod
    def decompress(self, input_path: Path, output_path: Path) -> float:
        """
        Decompresses an image.
        
        Args:
            input_path: Path to the compressed file
            output_path: Path where to save the decompressed output
            
        Returns:
            Decompression time in seconds
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the compressor (e.g., 'PNG', 'WebP')"""
        pass
    
    @property
    @abstractmethod
    def extension(self) -> str:
        """File extension used by this compressor (e.g., '.png', '.webp')"""
        pass
    
    def get_info(self) -> Dict:
        """Returns basic info about the compressor"""
        return {
            "name": self.name,
            "extension": self.extension,
            "lib_path": str(self.lib_path) if self.lib_path else None
        }


# ============================================================================
# FACTORY PATTERN FOR COMPRESSORS
# ============================================================================

class CompressorFactory:
    """Factory for creating compressor instances"""
    
    _compressors: Dict[str, Type[ImageCompressor]] = {}
    
    @classmethod
    def register(cls, name: str, compressor_class: Type[ImageCompressor]):
        """
        Registers a new compressor type.
        
        Args:
            name: Compressor identifier
            compressor_class: Compressor class
        """
        cls._compressors[name] = compressor_class
    
    @classmethod
    def create(cls, name: str, lib_path: Optional[Path] = None) -> ImageCompressor:
        """
        Creates an instance of the compressor.
        
        Args:
            name: Compressor identifier
            lib_path: Path to the C/C++ library
            
        Returns:
            Instance ImageCompressor
        """
        if name not in cls._compressors:
            raise ValueError(f"Undefined compressor: {name}")
        
        return cls._compressors[name](lib_path)
    
    @classmethod
    def list_available(cls) -> List[str]:
        """Returns a list of available compressor names"""
        return list(cls._compressors.keys())


# ============================================================================
# PLUGIN LOADER
# ============================================================================

class PluginLoader:
    """Dynamically loads compressor plugins from a directory"""
    
    @staticmethod
    def load_plugins_from_directory(plugin_dir: Path):
        """
        Loads all plugin modules from the folder.
        
        Args:
            plugin_dir: Path to the folder with plugins
        """
        if not plugin_dir.exists():
            plugin_dir.mkdir(parents=True)
            print(f"Folder for plugins created: {plugin_dir}")
            return
        
        plugin_files = list(plugin_dir.glob("*_compressor.py"))
        if not plugin_files:
            print(f"No plugin files (*_compressor.py) found in {plugin_dir}")
            return
            
        for plugin_file in plugin_files:
            PluginLoader._load_plugin_module(plugin_file)
    
    @staticmethod
    def _load_plugin_module(plugin_path: Path):
        """Loads a single plugin module"""
        try:
            module_name = f"compressor_{plugin_path.stem}"
            spec = importlib.util.spec_from_file_location(
                module_name, 
                plugin_path
            )
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)
                print(f"Plugin loaded: {plugin_path.name}")
        except Exception as e:
            print(f"Error when loading {plugin_path.name}: {e}")
            import traceback
            traceback.print_exc()


# ============================================================================
# BENCHMARK ORCHESTRATOR
# ============================================================================

class BenchmarkOrchestrator:
    """Orchestrates the benchmarking process"""
    
    def __init__(self, 
                dataset_dir: Path,
                output_dir: Path,
                libs_dir: Path):
        """
        Args:
            dataset_dir: Folder with image datasets
            output_dir: Folder where to save results
            libs_dir: Folder with C/C++ libraries
        """
        self.dataset_dir = Path(dataset_dir)
        self.output_dir = Path(output_dir)
        self.libs_dir = Path(libs_dir)
        
        # Create folders if they do not exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.libs_dir.mkdir(parents=True, exist_ok=True)
        
        self.results: List[BenchmarkResult] = []
    
    def run_benchmark(self,
                    compressor_names: List[str],
                    image_patterns: List[str] = ["*.png", "*.jpg", "*.bmp"],
                    compression_levels: List[CompressionLevel] = None) -> List[BenchmarkResult]:
        """
        Runs a benchmark for selected compressors.
        
        Args:
            compressor_names: List of compressor names to test
            image_patterns: Glob patterns for image search
            compression_levels: Compression levels to test
            
        Returns:
            List of results
        """
        if compression_levels is None:
            compression_levels = [CompressionLevel.BALANCED]
        
        # Search for images
        images = self._find_images(image_patterns)
        print(f"{len(images)} images found in dataset.")
        
        if not images:
            print("\n⚠️  No images found in dataset directory!")
            print(f"   Add images to: {self.dataset_dir}")
            return []
        
        # For every compressor
        for comp_name in compressor_names:
            print(f"\n{'='*60}")
            print(f"Testing: {comp_name}")
            print(f"{'='*60}")
            
            try:
                # Create compressor instance
                lib_path = self._find_lib_for_compressor(comp_name)
                compressor = CompressorFactory.create(comp_name, lib_path)
                
                # For every compression level
                for level in compression_levels:
                    print(f"\n  Level: {level.name}")
                    
                    # For every image
                    for img_path in images:
                        result = self._benchmark_single(
                            compressor, 
                            img_path, 
                            level
                        )
                        self.results.append(result)
                        self._print_result(result)
                        
            except Exception as e:
                print(f"❌ Error while testing {comp_name}: {e}")
                import traceback
                traceback.print_exc()
        
        return self.results
    
    def _find_images(self, patterns: List[str]) -> List[Path]:
        """Searches for images in the dataset directory"""
        images: list[Path] = []
        for pattern in patterns:
            images.extend(self.dataset_dir.rglob(pattern))
        return sorted(images)
    
    def _find_lib_for_compressor(self, compressor_name: str) -> Optional[Path]:
        """Finds the C/C++ library for the compressor if needed"""
        # Searches according to convention: libs/libpng.so, libs/libwebp.so, etc.
        lib_patterns = [
            f"lib{compressor_name.lower()}.*",
            f"{compressor_name.lower()}.*"
        ]
        
        for pattern in lib_patterns:
            libs = list(self.libs_dir.glob(pattern))
            if libs:
                return libs[0]
        
        return None
    
    def _benchmark_single(self,
                        compressor: ImageCompressor,
                        image_path: Path,
                        level: CompressionLevel) -> BenchmarkResult:
        """Benchmarks a single image with a given compressor and level"""
        # Create output directory for this format
        format_dir = self.output_dir / compressor.name
        format_dir.mkdir(exist_ok=True)
        
        # Path for compressed file
        compressed_path = format_dir / f"{image_path.stem}{compressor.extension}"
        
        # Compress and measure
        metrics = compressor.compress(image_path, compressed_path, level)
        
        return BenchmarkResult(
            image_path=image_path,
            format_name=compressor.name,
            metrics=metrics,
            metadata={"compression_level": level.name}
        )
    
    def _print_result(self, result: BenchmarkResult):
        """Prints a single benchmark result"""
        m = result.metrics
        if m.success:
            print(f"    {result.image_path.name}: "
                  f"{m.space_saving_percent:.1f}% saved, "
                  f"C:{m.compression_speed_mbps:.1f} MB/s, "
                  f"D:{m.decompression_speed_mbps:.1f} MB/s")
        else:
            print(f"    ❌ Error - {result.image_path.name}: {m.error_message}")
    
    def export_results(self, output_file: Path):
        """Exports results to a JSON file"""
        data = []
        for result in self.results:
            m = result.metrics
            data.append({
                "image": str(result.image_path),
                "format": result.format_name,
                "original_size": m.original_size,
                "compressed_size": m.compressed_size,
                "compression_ratio": m.compression_ratio,
                "space_saving_percent": m.space_saving_percent,
                "compression_time": m.compression_time,
                "decompression_time": m.decompression_time,
                "compression_speed_mbps": m.compression_speed_mbps,
                "decompression_speed_mbps": m.decompression_speed_mbps,
                "success": m.success,
                "metadata": result.metadata
            })
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"\nResults saved to: {output_file}")


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    # Paths
    PROJECT_ROOT = Path(__file__).parent
    PLUGINS_DIR = PROJECT_ROOT / "compressors"
    DATASET_DIR = PROJECT_ROOT / "image_datasets"
    OUTPUT_DIR = PROJECT_ROOT / "benchmark_results"
    LIBS_DIR = PROJECT_ROOT / "libs"
    
    print("Lossless Image Compression Benchmark")
    print("="*60)
    
    # Load plugins
    print("\nLoading plugins...")
    PluginLoader.load_plugins_from_directory(PLUGINS_DIR)
    
    # List available compressors
    available = CompressorFactory.list_available()
    print(f"Registered: {', '.join(available) if available else 'none'}")
    
    if not available:
        print("\n⚠️  No compressors found!")
        print("   Create plugins in the 'compressors/' folder according to the template.")
        print(f"\n   Expected format: *_compressor.py in {PLUGINS_DIR}")
        return
    
    print(f"\n✓ Available compressors: {', '.join(available)}")
    
    # Create orchestrator
    orchestrator = BenchmarkOrchestrator(
        dataset_dir=DATASET_DIR,
        output_dir=OUTPUT_DIR,
        libs_dir=LIBS_DIR
    )
    
    # Run benchmark
    print("\nStarting benchmark...")
    results = orchestrator.run_benchmark(
        compressor_names=available,
        compression_levels=[
            CompressionLevel.FAST,
            CompressionLevel.BALANCED,
            CompressionLevel.BEST
        ]
    )
    
    if results:
        # Export results
        orchestrator.export_results(OUTPUT_DIR / "results.json")
        print("\nDone!")
    else:
        print("\nNo results generated")


if __name__ == "__main__":
    # Ensure main is in sys.modules for plugin imports
    if "__main__" in sys.modules and "main" not in sys.modules:
        sys.modules["main"] = sys.modules["__main__"]
    main()
