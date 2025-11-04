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
        if self.original_size == 0:
            return 0.0
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
            print(f"📁 Plugin folder created: {plugin_dir}")
            return
        
        plugin_files = list(plugin_dir.glob("*_compressor.py"))
        if not plugin_files:
            print(f"⚠️  No plugin files (*_compressor.py) found in {plugin_dir}")
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
                print(f"  ✓ {plugin_path.name}")
        except Exception as e:
            print(f"  ✗ {plugin_path.name}: {e}")


# ============================================================================
# BENCHMARK ORCHESTRATOR (Deprecated - use BenchmarkRunner from benchmark_shared)
# ============================================================================

class BenchmarkOrchestrator:
    """Legacy orchestrator - kept for backward compatibility"""
    
    def __init__(self, 
                dataset_dir: Path,
                output_dir: Path,
                libs_dir: Path):
        self.dataset_dir = Path(dataset_dir)
        self.output_dir = Path(output_dir)
        self.libs_dir = Path(libs_dir)
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.libs_dir.mkdir(parents=True, exist_ok=True)
        
        self.results: List[BenchmarkResult] = []
    
    def _find_lib_for_compressor(self, compressor_name: str) -> Optional[Path]:
        """Finds the C/C++ library for the compressor"""
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
        """Benchmarks a single image"""
        format_dir = self.output_dir / compressor.name
        format_dir.mkdir(exist_ok=True)
        
        compressed_path = format_dir / f"{image_path.stem}{compressor.extension}"
        metrics = compressor.compress(image_path, compressed_path, level)
        
        return BenchmarkResult(
            image_path=image_path,
            format_name=compressor.name,
            metrics=metrics,
            metadata={"compression_level": level.name}
        )
    
    def export_results(self, output_file: Path):
        """Exports results to JSON"""
        data = []
        for result in self.results:
            m = result.metrics
            if not m.success:
                continue
                
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
        
        total_original = sum(r.metrics.original_size for r in self.results if r.metrics.success)
        total_compressed = sum(r.metrics.compressed_size for r in self.results if r.metrics.success)
        
        summary = {
            "total_images": len(self.results),
            "successful": sum(1 for r in self.results if r.metrics.success),
            "failed": sum(1 for r in self.results if not r.metrics.success),
            "total_original_size": total_original,
            "total_compressed_size": total_compressed,
            "overall_compression_ratio": total_original / total_compressed if total_compressed > 0 else 0,
            "results": data
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Results saved to: {output_file}")


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    # Import shared utilities
    from benchmark_shared import (
        BenchmarkConfig,
        BenchmarkRunner,
        BenchmarkSummarizer,
        ImageFinder
    )
    
    # Paths
    PROJECT_ROOT = Path(__file__).parent
    PLUGINS_DIR = PROJECT_ROOT / "compressors"
    DATASET_DIR = PROJECT_ROOT / "image_datasets"
    OUTPUT_DIR = PROJECT_ROOT / "benchmark_results"
    LIBS_DIR = PROJECT_ROOT / "libs"
    
    print("╔" + "═"*68 + "╗")
    print("║" + " "*15 + "Lossless Image Compression Benchmark" + " "*17 + "║")
    print("╚" + "═"*68 + "╝")
    
    # Load plugins
    print("\n📦 Loading plugins...")
    PluginLoader.load_plugins_from_directory(PLUGINS_DIR)
    
    available = CompressorFactory.list_available()
    
    if not available:
        print("\n⚠️  No compressors found!")
        print("   Create plugins in the 'compressors/' folder.")
        print(f"\n   Expected format: *_compressor.py in {PLUGINS_DIR}")
        return
    
    print(f"\n✅ Available compressors: {', '.join(available)}")
    
    # Find images
    images = ImageFinder.find_images(DATASET_DIR)
    print(f"\n🖼️  Found {len(images)} images in dataset")
    
    if not images:
        print(f"\n⚠️  No images found in dataset directory!")
        print(f"   Add images to: {DATASET_DIR}")
        return
    
    # Configuration
    NUM_ITERATIONS = 3  # Number of times to run each test (for averaging)
    WARMUP_ITERATIONS = 1  # Number of warmup runs (smoke test)
    
    print(f"\n⚙️  Configuration:")
    print(f"   Iterations per test: {NUM_ITERATIONS}")
    print(f"   Warmup iterations: {WARMUP_ITERATIONS}")
    
    # Create benchmark configuration
    config = BenchmarkConfig(
        dataset_dir=DATASET_DIR,
        output_dir=OUTPUT_DIR,
        libs_dir=LIBS_DIR,
        compressor_names=available,
        image_paths=images,
        compression_levels=[
            CompressionLevel.FAST,
            CompressionLevel.BALANCED,
            CompressionLevel.BEST
        ],
        verify_lossless=True  # Enable lossless verification
    )
    
    # Run benchmark
    print("\n🚀 Starting benchmark...")
    runner = BenchmarkRunner(config)
    results, verification_results = runner.run(progress_callback=print)
    
    if results:
        # Print summaries
        BenchmarkSummarizer.print_compression_summary(results, print)
        BenchmarkSummarizer.print_verification_summary(verification_results, print)
        
        # Export results using legacy orchestrator for JSON format
        orchestrator = BenchmarkOrchestrator(DATASET_DIR, OUTPUT_DIR, LIBS_DIR)
        orchestrator.results = results
        orchestrator.export_results(OUTPUT_DIR / "results.json")
        
        print("\n✅ Benchmark completed successfully!")
    else:
        print("\n⚠️  No results generated")


if __name__ == "__main__":
    # Ensure main is in sys.modules for plugin imports
    if "__main__" in sys.modules and "main" not in sys.modules:
        sys.modules["main"] = sys.modules["__main__"]
    main()