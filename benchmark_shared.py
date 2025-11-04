"""
Shared benchmark functions used by both CLI and GUI
benchmark_shared.py
"""

from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass

from main import (
    ImageCompressor,
    CompressionMetrics,
    CompressionLevel,
    BenchmarkResult,
    CompressorFactory
)
from utils.verification import ImageVerifier, VerificationResult


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark execution"""
    dataset_dir: Path
    output_dir: Path
    libs_dir: Path
    compressor_names: List[str]
    image_paths: List[Path]
    compression_levels: List[CompressionLevel]
    verify_lossless: bool = True
    num_iterations: int = 1  # Number of times to repeat each test
    warmup_iterations: int = 1  # Number of warmup runs before actual testing


class BenchmarkRunner:
    """Shared benchmark execution logic for CLI and GUI"""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.results: List[BenchmarkResult] = []
        self.verification_results: Dict[tuple, VerificationResult] = {}
        self.should_stop = False
    
    def stop(self):
        """Signal the benchmark to stop"""
        self.should_stop = True
    
    def run(self, progress_callback=None) -> tuple[List[BenchmarkResult], Dict[tuple, VerificationResult]]:
        """
        Execute benchmark with optional progress callback
        
        Args:
            progress_callback: Function called with status messages (optional)
            
        Returns:
            Tuple of (results, verification_results)
        """
        self.results = []
        self.verification_results = {}
        self.should_stop = False
        
        def log(message: str):
            if progress_callback:
                progress_callback(message)
        
        log(f"\n{'='*70}")
        log(f"Starting Benchmark")
        log(f"{'='*70}")
        log(f"Images: {len(self.config.image_paths)}")
        log(f"Compressors: {', '.join(self.config.compressor_names)}")
        log(f"Levels: {', '.join(l.name for l in self.config.compression_levels)}")
        log(f"Verification: {'Enabled' if self.config.verify_lossless else 'Disabled'}")
        
        # For each compressor
        for comp_name in self.config.compressor_names:
            if self.should_stop:
                break
            
            log(f"\n{'='*70}")
            log(f"Testing: {comp_name}")
            log(f"{'='*70}")
            
            try:
                # Create compressor instance
                lib_path = self._find_lib_for_compressor(comp_name)
                compressor = CompressorFactory.create(comp_name, lib_path)
                
                # For each compression level
                for level in self.config.compression_levels:
                    if self.should_stop:
                        break
                    
                    log(f"\n  Compression Level: {level.name}")
                    log(f"  {'-'*66}")
                    
                    # For each image
                    for img_path in self.config.image_paths:
                        if self.should_stop:
                            break
                        
                        # Warmup runs (smoke test)
                        if self.config.warmup_iterations > 0:
                            log(f"    🔥 Warming up: {img_path.name} ({self.config.warmup_iterations}x)")
                            for _ in range(self.config.warmup_iterations):
                                if self.should_stop:
                                    break
                                self._benchmark_single(compressor, img_path, level)
                        
                        # Actual benchmark runs
                        iteration_results = []
                        for iteration in range(self.config.num_iterations):
                            if self.should_stop:
                                break
                            
                            if self.config.num_iterations > 1:
                                log(f"    📊 Run {iteration + 1}/{self.config.num_iterations}: {img_path.name}")
                            
                            result = self._benchmark_single(compressor, img_path, level)
                            iteration_results.append(result)
                        
                        if not iteration_results:
                            continue
                        
                        # Calculate average metrics
                        avg_result = self._average_results(iteration_results)
                        self.results.append(avg_result)
                        
                        # Log result
                        self._log_result(avg_result, log, self.config.num_iterations)
                        
                        # Verify lossless if enabled (only once, on last iteration)
                        if self.config.verify_lossless and avg_result.metrics.success:
                            self._verify_result(compressor, img_path, avg_result, log)
                
            except Exception as e:
                log(f"  Error: {str(e)}")
        
        if self.should_stop:
            log("\nBenchmark stopped by user.")
        
        return self.results, self.verification_results
    
    def _find_lib_for_compressor(self, compressor_name: str) -> Optional[Path]:
        """Find C/C++ library for compressor"""
        lib_patterns = [
            f"lib{compressor_name.lower()}.*",
            f"{compressor_name.lower()}.*"
        ]
        
        for pattern in lib_patterns:
            libs = list(self.config.libs_dir.glob(pattern))
            if libs:
                return libs[0]
        
        return None
    
    def _benchmark_single(
        self,
        compressor: ImageCompressor,
        image_path: Path,
        level: CompressionLevel
    ) -> BenchmarkResult:
        """Benchmark a single image"""
        # Create output directory
        format_dir = self.config.output_dir / compressor.name
        format_dir.mkdir(exist_ok=True, parents=True)
        
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
    
    def _average_results(self, results: List[BenchmarkResult]) -> BenchmarkResult:
        """
        Calculate average metrics from multiple benchmark runs
        
        Args:
            results: List of benchmark results from iterations
            
        Returns:
            Single result with averaged metrics
        """
        if not results:
            raise ValueError("No results to average")
        
        if len(results) == 1:
            return results[0]
        
        # Use first result as template
        first = results[0]
        
        # Calculate averages
        avg_comp_time = sum(r.metrics.compression_time for r in results) / len(results)
        avg_decomp_time = sum(r.metrics.decompression_time for r in results) / len(results)
        
        # All other metrics should be the same, take from first result
        avg_metrics = CompressionMetrics(
            original_size=first.metrics.original_size,
            compressed_size=first.metrics.compressed_size,
            compression_ratio=first.metrics.compression_ratio,
            compression_time=avg_comp_time,
            decompression_time=avg_decomp_time,
            success=first.metrics.success,
            error_message=first.metrics.error_message
        )
        
        return BenchmarkResult(
            image_path=first.image_path,
            format_name=first.format_name,
            metrics=avg_metrics,
            metadata={
                **first.metadata,
                "iterations": len(results),
                "warmup_iterations": self.config.warmup_iterations
            }
        )
    
    def _log_result(self, result: BenchmarkResult, log_callback, num_iterations: int = 1):
        """Log benchmark result"""
        m = result.metrics
        if m.success:
            iterations_info = f" (avg of {num_iterations} runs)" if num_iterations > 1 else ""
            log_callback(f"    {result.image_path.name}{iterations_info}")
            log_callback(f"       Size: {m.original_size:,} B → {m.compressed_size:,} B")
            log_callback(f"       Savings: {m.space_saving_percent:.1f}% | Ratio: {m.compression_ratio:.2f}x")
            log_callback(f"       Compression: {m.compression_time:.3f}s ({m.compression_speed_mbps:.1f} MB/s)")
            log_callback(f"       Decompression: {m.decompression_time:.3f}s ({m.decompression_speed_mbps:.1f} MB/s)")
        else:
            log_callback(f"    {result.image_path.name}: FAILED - {m.error_message}")
    
    def _verify_result(
        self,
        compressor: ImageCompressor,
        img_path: Path,
        result: BenchmarkResult,
        log_callback
    ):
        """Verify lossless compression"""
        format_dir = self.config.output_dir / compressor.name
        compressed_path = format_dir / f"{img_path.stem}{compressor.extension}"
        
        if compressed_path.exists():
            verification = ImageVerifier.verify_lossless(img_path, compressed_path)
            
            key = (img_path.name, compressor.name)
            self.verification_results[key] = verification
            
            if verification.is_lossless:
                log_callback(f"       Verification: ✓ LOSSLESS (100.0000% accurate)")
            else:
                log_callback(f"       Verification: ✗ LOSSY")
                log_callback(f"          Max difference: {verification.max_difference:.2f}")
                log_callback(f"          Different pixels: {verification.different_pixels:,} / {verification.total_pixels:,}")
                log_callback(f"          Accuracy: {verification.accuracy_percent:.4f}%")


class BenchmarkSummarizer:
    """Generate summaries and reports from benchmark results"""
    
    @staticmethod
    def print_compression_summary(
        results: List[BenchmarkResult],
        log_callback
    ):
        """Print compression performance summary"""
        log_callback(f"\n{'='*70}")
        log_callback("COMPRESSION SUMMARY")
        log_callback(f"{'='*70}")
        
        # Group by format
        by_format = {}
        for result in results:
            if result.metrics.success:
                if result.format_name not in by_format:
                    by_format[result.format_name] = []
                by_format[result.format_name].append(result.metrics)
        
        # Print statistics for each format
        for format_name, metrics_list in sorted(by_format.items()):
            avg_ratio = sum(m.compression_ratio for m in metrics_list) / len(metrics_list)
            avg_savings = sum(m.space_saving_percent for m in metrics_list) / len(metrics_list)
            avg_comp_time = sum(m.compression_time for m in metrics_list) / len(metrics_list)
            avg_decomp_time = sum(m.decompression_time for m in metrics_list) / len(metrics_list)
            avg_comp_speed = sum(m.compression_speed_mbps for m in metrics_list) / len(metrics_list)
            avg_decomp_speed = sum(m.decompression_speed_mbps for m in metrics_list) / len(metrics_list)
            
            log_callback(f"\n{format_name}")
            log_callback(f"   Compression Ratio: {avg_ratio:.2f}x")
            log_callback(f"   Space Savings: {avg_savings:.1f}%")
            log_callback(f"   Avg Compression Time: {avg_comp_time:.3f}s ({avg_comp_speed:.1f} MB/s)")
            log_callback(f"   Avg Decompression Time: {avg_decomp_time:.3f}s ({avg_decomp_speed:.1f} MB/s)")
    
    @staticmethod
    def print_verification_summary(
        verification_results: Dict[tuple, VerificationResult],
        log_callback
    ):
        """Print lossless verification summary"""
        if not verification_results:
            return
        
        log_callback(f"\n{'='*70}")
        log_callback("LOSSLESS VERIFICATION SUMMARY")
        log_callback(f"{'='*70}")
        
        lossless_count = sum(
            1 for v in verification_results.values()
            if v.is_lossless
        )
        total_count = len(verification_results)
        lossy_count = total_count - lossless_count
        
        log_callback(f"\nTotal Tests: {total_count}")
        log_callback(f"✓ Truly Lossless: {lossless_count} ({100*lossless_count/total_count:.1f}%)")
        log_callback(f"✗ Lossy: {lossy_count} ({100*lossy_count/total_count:.1f}%)")
        
        if lossy_count > 0:
            log_callback("\nLossy compressions detected:")
            for key, verification in verification_results.items():
                if not verification.is_lossless:
                    img_name, comp_name = key
                    log_callback(f"   • {img_name} with {comp_name}")
                    log_callback(f"      Max diff: {verification.max_difference:.2f}, "
                               f"Different pixels: {verification.different_pixels:,}")
        else:
            log_callback("\n✓ All compressions are truly lossless!")


class ImageFinder:
    """Utility for finding images in directories"""
    
    DEFAULT_PATTERNS = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff", "*.tif"]
    
    @staticmethod
    def find_images(
        directory: Path,
        patterns: List[str] = None,
        recursive: bool = True
    ) -> List[Path]:
        """
        Find images in directory
        
        Args:
            directory: Directory to search
            patterns: Glob patterns (default: common image formats)
            recursive: Search recursively if True
            
        Returns:
            Sorted list of image paths
        """
        if patterns is None:
            patterns = ImageFinder.DEFAULT_PATTERNS
        
        images = []
        for pattern in patterns:
            if recursive:
                images.extend(directory.rglob(pattern))
            else:
                images.extend(directory.glob(pattern))
        
        return sorted(set(images))