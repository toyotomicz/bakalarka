"""
Shared benchmark functions used by both CLI and GUI
benchmark_shared.py
"""

from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from PIL import Image
import tempfile
import json

from main import (
    ImageCompressor,
    CompressionMetrics,
    CompressionLevel,
    BenchmarkResult,
    CompressorFactory
)
from utils.verification import ImageVerifier, VerificationResult
from utils.system_metrics import (
    SystemMonitor,
    SystemMetrics,
    ScenarioAnalyzer,
    ProcessIsolator
)

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
    strip_metadata: bool = True
    num_iterations: int = 1  # Number of times to repeat each test
    warmup_iterations: int = 1  # Number of warmup runs before actual testing
    monitor_resources: bool = True  # Monitor CPU/RAM during tests
    isolate_process: bool = False  # Use process isolation for more accurate results


class BenchmarkRunner:
    """Shared benchmark execution logic for CLI and GUI"""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.results: List[BenchmarkResult] = []
        self.verification_results: Dict[tuple, VerificationResult] = {}
        self.should_stop = False
        self.isolator = ProcessIsolator() if config.isolate_process else None
    
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
        
        # Setup process isolation if requested
        if self.isolator:
            log("🔒 Isolating process for accurate measurements...")
            self.isolator.isolate(high_priority=True)
        
        log(f"\n{'='*70}")
        log(f"Starting Benchmark")
        log(f"{'='*70}")
        log(f"Images: {len(self.config.image_paths)}")
        log(f"Compressors: {', '.join(self.config.compressor_names)}")
        log(f"Levels: {', '.join(l.name for l in self.config.compression_levels)}")
        log(f"Verification: {'Enabled' if self.config.verify_lossless else 'Disabled'}")
        log(f"Strip Metadata: {'Enabled' if self.config.strip_metadata else 'Disabled'}")
        log(f"Resource Monitoring: {'Enabled' if self.config.monitor_resources else 'Disabled'}")
        log(f"Process Isolation: {'Enabled' if self.config.isolate_process else 'Disabled'}")
        
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
                                self._benchmark_single(compressor, img_path, level, monitor=False)
                        
                        # Actual benchmark runs
                        iteration_results = []
                        for iteration in range(self.config.num_iterations):
                            if self.should_stop:
                                break
                            
                            if self.config.num_iterations > 1:
                                log(f"    📊 Run {iteration + 1}/{self.config.num_iterations}: {img_path.name}")
                            
                            result = self._benchmark_single(
                                compressor, 
                                img_path, 
                                level,
                                monitor=self.config.monitor_resources
                            )
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
        
        # Restore process settings if isolated
        if self.isolator:
            self.isolator.restore()
            log("\n🔓 Process isolation restored")
        
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
    
    def _strip_metadata(self, image_path: Path) -> tuple:
        """
        Create a metadata-free copy of the image in a temp file.

        Returns:
            (actual_path, temp_file_object_or_None)
            Caller must delete temp_file.name when done.
        """
        try:
            with Image.open(image_path) as img:
                data = list(img.getdata())
                clean_img = Image.new(img.mode, img.size)
                clean_img.putdata(data)

                temp_file = tempfile.NamedTemporaryFile(
                    suffix=image_path.suffix,
                    delete=False
                )
                temp_path = Path(temp_file.name)
                temp_file.close()

                save_kwargs = {}
                if img.format in {"JPEG", "WEBP"}:
                    save_kwargs["exif"] = b""

                clean_img.save(temp_path, format=img.format, **save_kwargs)
                return temp_path, temp_file

        except Exception:
            return image_path, None

    def _benchmark_single(
        self,
        compressor: ImageCompressor,
        image_path: Path,
        level: CompressionLevel,
        monitor: bool = True
    ) -> BenchmarkResult:
        """Benchmark a single image with optional resource monitoring"""
        
        # Create output directory
        format_dir = self.config.output_dir / compressor.name
        format_dir.mkdir(exist_ok=True, parents=True)

        temp_file = None
        if self.config.strip_metadata:
            actual_image_path, temp_file = self._strip_metadata(image_path)
        else:
            actual_image_path = image_path

        
        # Start system monitoring if enabled
        sys_monitor = None
        if monitor:
            sys_monitor = SystemMonitor(
                sampling_interval=0.05,
                adaptive=True,
                force_pre_post=True
            )
            file_size = actual_image_path.stat().st_size
            sys_monitor.start(file_size_bytes=file_size)
        
        result = None
        try:
            # Path for compressed file
            compressed_path = format_dir / f"{image_path.stem}{compressor.extension}"
            
            # Compress and measure
            metrics = compressor.compress(actual_image_path, compressed_path, level)
            
            result = BenchmarkResult(
                image_path=image_path,
                format_name=compressor.name,
                metrics=metrics,
                metadata={"compression_level": level.name},
                system_metrics=None  # assigned after monitor.stop() below
            )
            return result
        finally:
            # Stop monitor unconditionally — even if compress() raised an exception.
            # This prevents the background sampling thread from running forever.
            if sys_monitor:
                system_metrics = sys_monitor.stop()
                if result is not None:
                    result.system_metrics = system_metrics
            # Clean up temporary file if created
            if temp_file and actual_image_path != image_path:
                try:
                    actual_image_path.unlink()
                except OSError as e:
                    # Temp soubor nebyl smazán - nevadí, ale zalogujme pro diagnostiku
                    import logging
                    logging.debug(f"Could not delete temp file {actual_image_path}: {e}")
    
    def _average_results(self, results: List[BenchmarkResult]) -> BenchmarkResult:
        """
        Calculate average metrics from multiple successful benchmark runs.

        Args:
            results: List of benchmark results from measured iterations (warmup excluded)

        Returns:
            Single BenchmarkResult with averaged metrics

        Raises:
            ValueError: if no successful results are available
        """

        if not results:
            raise ValueError("No benchmark results provided")

        # Keep only successful results
        successful_results = [r for r in results if r.metrics.success]

        if not successful_results:
            raise ValueError("All benchmark iterations failed; nothing to average")

        # If only one successful result, return it directly
        if len(successful_results) == 1:
            return successful_results[0]

        first = successful_results[0]
        count = len(successful_results)

        # --- Average compression metrics ---
        avg_comp_time = sum(r.metrics.compression_time for r in successful_results) / count
        avg_decomp_time = sum(r.metrics.decompression_time for r in successful_results) / count

        # --- Average system metrics (if present) ---
        system_results = [r.system_metrics for r in successful_results if r.system_metrics]
        avg_system_metrics = None

        if system_results:
            avg_system_metrics = SystemMetrics(
                avg_cpu_percent=sum(s.avg_cpu_percent for s in system_results) / len(system_results),
                max_cpu_percent=max(s.max_cpu_percent for s in system_results),
                avg_process_cpu=sum(s.avg_process_cpu for s in system_results) / len(system_results),
                max_process_cpu=max(s.max_process_cpu for s in system_results),
                avg_ram_mb=sum(s.avg_ram_mb for s in system_results) / len(system_results),
                peak_ram_mb=max(s.peak_ram_mb for s in system_results),
                total_io_read_mb=sum(s.total_io_read_mb for s in system_results) / len(system_results),
                total_io_write_mb=sum(s.total_io_write_mb for s in system_results) / len(system_results),
                duration_seconds=sum(s.duration_seconds for s in system_results) / len(system_results),
                sample_count=sum(s.sample_count for s in system_results) // len(system_results),
            )

        # --- Construct averaged compression metrics ---
        avg_metrics = CompressionMetrics(
            original_size=first.metrics.original_size,
            compressed_size=first.metrics.compressed_size,
            compression_ratio=first.metrics.compression_ratio,
            compression_time=avg_comp_time,
            decompression_time=avg_decomp_time,
            success=True,
            error_message=None,
        )

        # --- Metadata (truthful, audit-friendly) ---
        metadata = {
            **(first.metadata or {}),
            "iterations_requested": len(results),
            "iterations_successful": count,
            "warmup_iterations": self.config.warmup_iterations,
        }

        return BenchmarkResult(
            image_path=first.image_path,
            format_name=first.format_name,
            metrics=avg_metrics,
            metadata=metadata,
            system_metrics=avg_system_metrics,
        )

    
    def _log_result(self, result: BenchmarkResult, log_callback, num_iterations: int = 1):
        m = result.metrics
        if not m.success:
            log_callback(f"    {result.image_path.name}: FAILED - {m.error_message}")
            return

        iterations_info = f" (avg of {num_iterations} runs)" if num_iterations > 1 else ""
        log_callback(f"    {result.image_path.name}{iterations_info}")
        log_callback(f"       Size: {m.original_size:,} B → {m.compressed_size:,} B")
        log_callback(f"       Savings: {m.space_saving_percent:.1f}% | Ratio: {m.compression_ratio:.2f}x")
        log_callback(f"       Compression: {m.compression_time:.3f}s ({m.compression_speed_mbps:.1f} MB/s)")
        log_callback(f"       Decompression: {m.decompression_time:.3f}s ({m.decompression_speed_mbps:.1f} MB/s)")

        if result.system_metrics:
            sm = result.system_metrics
            log_callback(f"       CPU: Avg {sm.avg_process_cpu:.1f}% | Peak {sm.max_process_cpu:.1f}%")
            log_callback(f"       RAM: Avg {sm.avg_ram_mb:.1f} MB | Peak {sm.peak_ram_mb:.1f} MB")
            log_callback(f"       I/O: Read {sm.total_io_read_mb:.2f} MB | Write {sm.total_io_write_mb:.2f} MB")
            if not sm.is_reliable:
                log_callback(f"       ⚠️  Quality: {sm.measurement_quality} ({sm.sample_count} samples)")

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
        
        if not compressed_path.exists():
            return
        
        # CRITICAL: verify against the SAME stripped input used for compression
        temp_file = None
        if self.config.strip_metadata:
            verification_input, temp_file = self._strip_metadata(img_path)
            if verification_input == img_path:
                # stripping failed
                log_callback("       Verification: ✗ FAILED - Could not create stripped input")
                return
        else:
            verification_input = img_path
        
        try:
            # Now verify using the correct input
            verification = ImageVerifier.verify_lossless(
                verification_input, 
                compressed_path,
                compressor_factory=CompressorFactory,
                temp_dir=format_dir
            )
            
            key = (img_path.name, compressor.name)
            self.verification_results[key] = verification
            
            if verification.is_lossless:
                log_callback(f"       Verification: ✓ LOSSLESS (100.0000% accurate)")
            else:
                log_callback(f"       Verification: ✗ LOSSY")
                if verification.error_message:
                    log_callback(f"          Error: {verification.error_message}")
                else:
                    log_callback(f"          Max difference: {verification.max_difference:.2f}")
                    log_callback(f"          Different pixels: {verification.different_pixels:,} / {verification.total_pixels:,}")
                    log_callback(f"          Accuracy: {verification.accuracy_percent:.4f}%")
        
        finally:
            # Clean up temp verification input if created
            if temp_file and verification_input != img_path:
                try:
                    verification_input.unlink()
                except OSError:
                    pass


class BenchmarkSummarizer:
    """Generate summaries and reports from benchmark results"""
    
    @staticmethod
    def generate_unique_filename(config: BenchmarkConfig) -> str:
        """
        Generuje unikátní název souboru podle parametrů benchmarku
        
        Args:
            config: Konfigurace benchmarku
            
        Returns:
            Název souboru ve formátu: results_YYYYMMDD_HHMMSS_params.json
        """
        # Časová značka
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Kompresory (max 3, pak "+" pokud jich je víc)
        if len(config.compressor_names) <= 3:
            compressors_str = "-".join(config.compressor_names)
        else:
            compressors_str = "-".join(config.compressor_names[:3]) + f"-plus{len(config.compressor_names) - 3}"
        
        # Úrovně komprese
        levels_str = "-".join([level.name for level in config.compression_levels])
        
        # Počet iterací
        iterations_str = f"iter{config.num_iterations}"
        if config.warmup_iterations > 0:
            iterations_str += f"w{config.warmup_iterations}"
        
        # Dodatečné parametry
        params = []
        if config.verify_lossless:
            params.append("verify")
        if config.strip_metadata:
            params.append("strip")
        if config.monitor_resources:
            params.append("monitor")
        if config.isolate_process:
            params.append("isolate")
        
        params_str = "-".join(params) if params else "basic"
        
        # Počet obrázků
        images_count = f"{len(config.image_paths)}img"
        
        # Sestavení názvu
        filename_parts = [
            "results",
            timestamp,
            compressors_str,
            levels_str,
            iterations_str,
            images_count,
            params_str
        ]
        
        filename = "_".join(filename_parts) + ".json"
        
        # Zkrácení pokud je příliš dlouhé (max 200 znaků pro jistotu)
        if len(filename) > 200:
            # Fallback na kratší verzi
            filename = (
                f"results_{timestamp}_"
                f"{len(config.compressor_names)}comp_"
                f"{len(config.compression_levels)}lvl_"
                f"{images_count}_"
                f"{params_str}.json"
            )
        
        return filename
    
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
                by_format[result.format_name].append(result)
        
        # Print statistics for each format
        for format_name, result_list in sorted(by_format.items()):
            metrics_list = [r.metrics for r in result_list]
            
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
            
            # System metrics if available
            system_results = [r.system_metrics for r in result_list if r.system_metrics]
            if system_results:
                avg_cpu = sum(s.avg_process_cpu for s in system_results) / len(system_results)
                peak_cpu = max(s.max_process_cpu for s in system_results)
                avg_ram = sum(s.avg_ram_mb for s in system_results) / len(system_results)
                peak_ram = max(s.peak_ram_mb for s in system_results)
                avg_io = sum(s.io_total_mb for s in system_results) / len(system_results)
                
                log_callback(f"   Avg CPU Usage: {avg_cpu:.1f}% (peak {peak_cpu:.1f}%)")
                log_callback(f"   Avg RAM Usage: {avg_ram:.1f} MB (peak {peak_ram:.1f} MB)")
                log_callback(f"   Avg I/O Total: {avg_io:.2f} MB")
    
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
    
    @staticmethod
    def print_scenario_analysis(
        results: List[BenchmarkResult],
        log_callback
    ):
        """Print best/worst case scenario analysis"""
        if not results:
            return
        
        # Analyze different metrics
        metrics_to_analyze = [
            ("compression_ratio", "Compression Ratio"),
            ("ram_usage", "RAM Usage"),
            ("cpu_usage", "CPU Usage"),
            ("io_total", "I/O Total")
        ]
        
        for metric_key, metric_name in metrics_to_analyze:
            scenarios = ScenarioAnalyzer.identify_scenarios(results, metric_key)
            
            if scenarios["best"] and scenarios["worst"]:
                log_callback(f"\n{'='*70}")
                log_callback(f"SCENARIO ANALYSIS: {metric_name}")
                log_callback(f"{'='*70}")
                ScenarioAnalyzer.print_scenario_comparison(scenarios, log_callback)
    
    @staticmethod
    def export_results_json(
        results: List[BenchmarkResult],
        verification_results: Dict[tuple, VerificationResult],
        output_dir: Path,  # ZMĚNĚNO: z output_file na output_dir
        config: BenchmarkConfig
    ) -> Path:  # Přidán return type
        """
        Export all results including system metrics to JSON with unique filename
        
        Args:
            results: List of benchmark results
            verification_results: Verification results dict
            output_dir: Output directory path (NOT a file!)
            config: Benchmark configuration
            
        Returns:
            Path to created JSON file
        """
        # Vygenerovat unikátní název souboru
        filename = BenchmarkSummarizer.generate_unique_filename(config)
        output_file = output_dir / filename
        
        # Zajistit že adresář existuje
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert results to serializable format
        results_data = []
        for result in results:
            m = result.metrics
            
            result_dict = {
                "image": str(result.image_path.name),
                "image_full_path": str(result.image_path),
                "format": result.format_name,
                "compression": {
                    "original_size": m.original_size,
                    "compressed_size": m.compressed_size,
                    "compression_ratio": m.compression_ratio,
                    "space_saving_percent": m.space_saving_percent,
                    "compression_time": m.compression_time,
                    "decompression_time": m.decompression_time,
                    "compression_speed_mbps": m.compression_speed_mbps,
                    "decompression_speed_mbps": m.decompression_speed_mbps,
                    "success": m.success,
                    "error_message": m.error_message
                },
                "metadata": result.metadata
            }
            
            # Add system metrics if available
            if result.system_metrics:
                sm = result.system_metrics
                result_dict["system_metrics"] = {
                    "cpu": {
                        "avg_percent": sm.avg_cpu_percent,
                        "max_percent": sm.max_cpu_percent,
                        "avg_process_percent": sm.avg_process_cpu,
                        "max_process_percent": sm.max_process_cpu
                    },
                    "memory": {
                        "avg_mb": sm.avg_ram_mb,
                        "max_mb": sm.peak_ram_mb,
                        "ram_efficiency_mb_per_sec": sm.ram_efficiency_mb_per_sec
                    },
                    "io": {
                        "total_read_mb": sm.total_io_read_mb,
                        "total_write_mb": sm.total_io_write_mb,
                        "total_mb": sm.io_total_mb
                    },
                    "timing": {
                        "duration_seconds": sm.duration_seconds,
                        "sample_count": sm.sample_count
                    }
                }
            
            results_data.append(result_dict)
        
        # Convert verification results with proper type conversion
        verification_data = []
        for key, verification in verification_results.items():
            img_name, comp_name = key
            verification_data.append({
                "image": img_name,
                "compressor": comp_name,
                "is_lossless": bool(verification.is_lossless),
                "max_difference": float(verification.max_difference) if verification.max_difference is not None else None,
                "different_pixels": int(verification.different_pixels) if verification.different_pixels is not None else None,
                "total_pixels": int(verification.total_pixels) if verification.total_pixels is not None else None,
                "accuracy_percent": float(verification.accuracy_percent) if verification.accuracy_percent is not None else None,
                "error_message": str(verification.error_message) if verification.error_message else None
            })
        
        # Calculate summaries
        successful_results = [r for r in results if r.metrics.success]
        total_original = sum(r.metrics.original_size for r in successful_results)
        total_compressed = sum(r.metrics.compressed_size for r in successful_results)
        
        # Scenario analysis
        scenarios_data = {}
        if config.monitor_resources:
            for metric_key, metric_name in [("compression_ratio", "compression_ratio"),
                                        ("ram_usage", "ram_usage"),
                                        ("cpu_usage", "cpu_usage"),
                                        ("io_total", "io_total")]:
                scenarios = ScenarioAnalyzer.identify_scenarios(successful_results, metric_key)
                if scenarios["best"] and scenarios["worst"]:
                    scenarios_data[metric_key] = {
                        "best": {
                            "file": scenarios["best"].file_path.name,
                            "file_size_mb": scenarios["best"].file_size_mb,
                            "compression_ratio": scenarios["best"].compression_ratio,
                            "peak_ram_mb": scenarios["best"].system_metrics.peak_ram_mb,
                            "avg_cpu_percent": scenarios["best"].system_metrics.avg_process_cpu,
                            "io_total_mb": scenarios["best"].system_metrics.io_total_mb,
                            "ram_per_mb": scenarios["best"].ram_per_mb,
                            "cpu_efficiency": scenarios["best"].cpu_efficiency
                        },
                        "worst": {
                            "file": scenarios["worst"].file_path.name,
                            "file_size_mb": scenarios["worst"].file_size_mb,
                            "compression_ratio": scenarios["worst"].compression_ratio,
                            "peak_ram_mb": scenarios["worst"].system_metrics.peak_ram_mb,
                            "avg_cpu_percent": scenarios["worst"].system_metrics.avg_process_cpu,
                            "io_total_mb": scenarios["worst"].system_metrics.io_total_mb,
                            "ram_per_mb": scenarios["worst"].ram_per_mb,
                            "cpu_efficiency": scenarios["worst"].cpu_efficiency
                        }
                    }
        
        # Complete output structure with metadata
        output_data = {
            "benchmark_info": {
                "filename": filename,
                "timestamp": datetime.now().isoformat(),
                "generated_by": "Image Compression Benchmark Tool v1.0"
            },
            "benchmark_config": {
                "num_iterations": config.num_iterations,
                "warmup_iterations": config.warmup_iterations,
                "verify_lossless": config.verify_lossless,
                "strip_metadata": config.strip_metadata,
                "monitor_resources": config.monitor_resources,
                "isolate_process": config.isolate_process,
                "compressors": config.compressor_names,
                "compression_levels": [level.name for level in config.compression_levels]
            },
            "summary": {
                "total_images": len(results),
                "successful": len(successful_results),
                "failed": len(results) - len(successful_results),
                "total_original_size": total_original,
                "total_compressed_size": total_compressed,
                "overall_compression_ratio": total_original / total_compressed if total_compressed > 0 else 0,
                "lossless_verified": len([v for v in verification_results.values() if v.is_lossless]),
                "lossy_detected": len([v for v in verification_results.values() if not v.is_lossless])
            },
            "scenarios": scenarios_data,
            "results": results_data,
            "verification": verification_data
        }
        
        # Write to file with error handling
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Successfully saved results to: {output_file.name}")
            
        except Exception as e:
            print(f"❌ Error writing JSON: {e}")
            import traceback
            traceback.print_exc()
            
            # Fallback: zkusit jednodušší název
            fallback_file = output_dir / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            print(f"⚠️  Trying fallback filename: {fallback_file.name}")
            
            with open(fallback_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            output_file = fallback_file
        
        return output_file


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