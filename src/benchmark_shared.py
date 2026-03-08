"""
Shared benchmark logic used by both the CLI and the GUI.
benchmark_shared.py

Classes:
  BenchmarkConfig     : Immutable configuration for one benchmark session.
  BenchmarkRunner     : Executes the configured benchmark and collects results.
  BenchmarkSummarizer : Formats and exports results to the console and JSON.
  ImageFinder         : Utility for locating image files in directories.
"""

import json
import logging
import tempfile
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from PIL import Image

from main import (
    BenchmarkResult,
    CompressionLevel,
    CompressionMetrics,
    CompressorFactory,
    ImageCompressor,
)
from utils.verification import ImageVerifier, VerificationResult
from utils.cpu_affinity import IsolationConfig, ProcessIsolator
from utils.system_metrics import (
    ScenarioAnalyzer,
    SystemMetrics,
    SystemMonitor,
)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class BenchmarkConfig:
    """
    All parameters that control a single benchmark session.

    Passed to BenchmarkRunner.run() and embedded verbatim in the JSON report
    so that any result file is fully self-describing.

    isolation:
        Controls both high-priority scheduling and CPU core pinning.
        Build an IsolationConfig(high_priority=True, cpu_core=1) and pass it here.
        Use IsolationConfig() (all defaults) or None for no isolation.
    """

    dataset_dir:         Path
    output_dir:          Path
    libs_dir:            Path
    compressor_names:    List[str]            # factory keys, e.g. ["charls", "webp"]
    image_paths:         List[Path]
    compression_levels:  List[CompressionLevel]
    verify_lossless:     bool          = True
    strip_metadata:      bool          = True
    num_iterations:      int           = 1    # measurement iterations per image
    warmup_iterations:   int           = 1    # warm-up runs excluded from averages
    trim_top_n:          int           = 0    # drop N slowest runs before averaging
    monitor_resources:   bool            = True  # collect CPU / RAM / I/O metrics
    isolation:           IsolationConfig  = field(default_factory=IsolationConfig)  # high priority and / or core pin


# ============================================================================
# Runner
# ============================================================================

class BenchmarkRunner:
    """
    Executes the benchmark described by a BenchmarkConfig.

    CPU affinity and priority handling
    ----------------------------------
    Isolation settings come entirely from config.isolation (an IsolationConfig).
    ProcessIsolator reads them in isolate() and appends human-readable notes to
    IsolationState.isolation_notes, which are forwarded to the log callback.
    """

    def __init__(self, config: BenchmarkConfig):
        self.config               = config
        self.results:              List[BenchmarkResult]               = []
        self.verification_results: Dict[tuple, VerificationResult]     = {}
        self.should_stop          = False
        self.isolator = ProcessIsolator(config.isolation)

    def stop(self) -> None:
        """Signal the runner to stop after the current iteration."""
        self.should_stop = True

    def run(
        self,
        progress_callback=None,
    ) -> tuple[List[BenchmarkResult], Dict[tuple, VerificationResult]]:
        self.results               = []
        self.verification_results  = {}
        self.should_stop           = False

        def log(message: str) -> None:
            if progress_callback:
                progress_callback(message)

        # ---- Apply isolation (priority + optional CPU affinity) ----
        if self.config.isolation.enabled:
            log("Isolating process for accurate measurements...")
            state = self.isolator.isolate()
            for note in state.isolation_notes:
                log(f"  {note}")

        log(f"\n{'=' * 70}")
        log("Starting Benchmark")
        log(f"{'=' * 70}")
        log(f"Images:               {len(self.config.image_paths)}")
        log(f"Compressors:          {', '.join(self.config.compressor_names)}")
        log(f"Levels:               {', '.join(l.name for l in self.config.compression_levels)}")
        log(f"Verification:         {'Enabled' if self.config.verify_lossless else 'Disabled'}")
        log(f"Strip Metadata:       {'Enabled' if self.config.strip_metadata else 'Disabled'}")
        log(f"Resource Monitoring:  {'Enabled' if self.config.monitor_resources else 'Disabled'}")
        log(f"Process Isolation:    {'Enabled' if self.config.isolation.high_priority else 'Disabled'}")
        if self.config.isolation.cpu_core is not None:
            log(f"CPU Affinity Core:    {self.config.isolation.cpu_core}")
        else:
            log("CPU Affinity Core:    Disabled (all cores)")

        for comp_name in self.config.compressor_names:
            if self.should_stop:
                break

            log(f"\n{'=' * 70}")
            log(f"Testing: {comp_name}")
            log(f"{'=' * 70}")

            try:
                lib_path   = self._find_lib_for_compressor(comp_name)
                compressor = CompressorFactory.create(comp_name, lib_path)

                for level in self.config.compression_levels:
                    if self.should_stop:
                        break

                    log(f"\n  Compression Level: {level.name}")
                    log(f"  {'-' * 66}")

                    for img_path in self.config.image_paths:
                        if self.should_stop:
                            break

                        if self.config.warmup_iterations > 0:
                            log(f"    Warming up: {img_path.name} ({self.config.warmup_iterations}x)")
                            for _ in range(self.config.warmup_iterations):
                                if self.should_stop:
                                    break
                                self._benchmark_single(compressor, img_path, level, monitor=False)

                        iteration_results: List[BenchmarkResult] = []
                        for iteration in range(self.config.num_iterations):
                            if self.should_stop:
                                break

                            if self.config.num_iterations > 1:
                                log(f"    Run {iteration + 1}/{self.config.num_iterations}: {img_path.name}")

                            result = self._benchmark_single(
                                compressor,
                                img_path,
                                level,
                                monitor=self.config.monitor_resources,
                            )
                            iteration_results.append(result)

                        if not iteration_results:
                            continue

                        avg_result = self._average_results(iteration_results)
                        self.results.append(avg_result)
                        self._log_result(avg_result, log, self.config.num_iterations)

                        if self.config.verify_lossless and avg_result.metrics.success:
                            self._verify_result(compressor, img_path, avg_result, log)

            except Exception as exc:
                log(f"  Error: {exc}")

        # ---- Restore isolation ----
        if self.config.isolation.enabled:
            self.isolator.restore()
            log("\nProcess isolation restored.")

        if self.should_stop:
            log("\nBenchmark stopped by user.")

        return self.results, self.verification_results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _find_lib_for_compressor(self, compressor_name: str) -> Optional[Path]:
        patterns = [
            f"lib{compressor_name.lower()}.*",
            f"{compressor_name.lower()}.*",
        ]
        for pattern in patterns:
            matches = list(self.config.libs_dir.glob(pattern))
            if matches:
                return matches[0]
        return None

    def _strip_metadata(self, image_path: Path) -> tuple:
        """
        Strip all metadata from image_path and write a clean PNG to a temp file.

        Output is always PNG regardless of the original format because:
          - PNG is lossless and supports all bit depths / modes used here.
          - Every compressor plugin accepts PNG as input.
          - A neutral PNG baseline eliminates format-specific metadata (EXIF,
            XMP, ICC, JFIF APP markers, WebP EXIF chunks, etc.) completely.
          - ImageSizeCalculator always computes size from raw pixel data
            (width × height × bpp), so the PNG container overhead is irrelevant.

        The clean image is built by copying only raw pixel data into a fresh
        Image object with no ``info`` dict, guaranteeing that no metadata
        survives the round-trip.

        Returns:
            (temp_path, True)   on success — caller must delete temp_path when done.
            (image_path, False) on failure — original path, nothing to delete.
        """
        try:
            with Image.open(image_path) as img:
                # Ensure the full pixel data is loaded before the file is closed.
                img.load()

                # Normalise mode: Pillow can open exotic modes (P, PA, CMYK, …)
                # that PNG supports but that some compressors do not handle.
                # Preserve L (grayscale), LA, RGB, RGBA; convert everything else.
                mode = img.mode
                if mode not in ("L", "LA", "RGB", "RGBA"):
                    mode = "RGBA" if "A" in img.getbands() else "RGB"
                    img = img.convert(mode)

                # Build a brand-new Image with no .info dict to strip all metadata.
                clean = Image.new(mode, img.size)
                clean.putdata(img.getdata())

            # Always write as PNG — lossless, universally accepted, no metadata.
            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            temp_path = Path(tmp.name)
            tmp.close()
            clean.save(temp_path, format="PNG")
            return temp_path, True

        except Exception:
            return image_path, False

    def _benchmark_single(
        self,
        compressor: ImageCompressor,
        image_path: Path,
        level: CompressionLevel,
        monitor: bool = True,
    ) -> BenchmarkResult:
        format_dir = self.config.output_dir / compressor.name
        format_dir.mkdir(parents=True, exist_ok=True)

        temp_file = None
        if self.config.strip_metadata:
            actual_path, stripped = self._strip_metadata(image_path)
            temp_file = actual_path if stripped else None
        else:
            actual_path = image_path

        sys_monitor = None
        if monitor:
            sys_monitor = SystemMonitor(
                sampling_interval=0.05,
                adaptive=True,
                force_pre_post=True,
            )
            pinned = (
                1 if self.config.isolation.cpu_core is not None else None
            )
            sys_monitor.start(
                file_size_bytes=actual_path.stat().st_size,
                pinned_core_count=pinned,
            )

        result: Optional[BenchmarkResult] = None
        try:
            compressed_path = format_dir / f"{image_path.stem}{compressor.extension}"
            metrics = compressor.compress(actual_path, compressed_path, level)

            result = BenchmarkResult(
                image_path=image_path,
                format_name=compressor.name,
                metrics=metrics,
                metadata={"compression_level": level.name},
                system_metrics=None,
            )
            return result

        finally:
            if sys_monitor:
                system_metrics = sys_monitor.stop()
                if result is not None:
                    result.system_metrics = system_metrics

            if temp_file is not None:
                try:
                    temp_file.unlink()
                except OSError as exc:
                    logging.debug(f"Could not delete temp file {temp_file}: {exc}")

    def _average_results(self, results: List[BenchmarkResult]) -> BenchmarkResult:
        if not results:
            raise ValueError("No benchmark results provided.")

        successful = [r for r in results if r.metrics.success]
        if not successful:
            raise ValueError("All benchmark iterations failed; nothing to average.")

        if len(successful) == 1:
            return successful[0]

        # Drop the N slowest runs (by compression_time) before averaging.
        trim_n = getattr(self.config, "trim_top_n", 0)
        if trim_n > 0 and len(successful) > trim_n:
            successful = sorted(successful, key=lambda r: r.metrics.compression_time)
            successful = successful[:len(successful) - trim_n]

        first = successful[0]
        count = len(successful)

        avg_comp_time   = sum(r.metrics.compression_time   for r in successful) / count
        avg_decomp_time = sum(r.metrics.decompression_time for r in successful) / count

        system_samples = [r.system_metrics for r in successful if r.system_metrics]
        avg_system = None
        if system_samples:
            n = len(system_samples)
            avg_system = SystemMetrics(
                avg_cpu_percent    = sum(s.avg_cpu_percent    for s in system_samples) / n,
                max_cpu_percent    = max(s.max_cpu_percent    for s in system_samples),
                avg_process_cpu    = sum(s.avg_process_cpu    for s in system_samples) / n,
                max_process_cpu    = max(s.max_process_cpu    for s in system_samples),
                avg_ram_mb         = sum(s.avg_ram_mb         for s in system_samples) / n,
                peak_ram_mb        = max(s.peak_ram_mb        for s in system_samples),
                ram_baseline_mb    = system_samples[0].ram_baseline_mb,
                total_io_read_mb   = sum(s.total_io_read_mb   for s in system_samples) / n,
                total_io_write_mb  = sum(s.total_io_write_mb  for s in system_samples) / n,
                duration_seconds   = sum(s.duration_seconds   for s in system_samples) / n,
                sample_count       = sum(s.sample_count       for s in system_samples) // n,
                logical_core_count = system_samples[0].logical_core_count,
                is_reliable        = all(s.is_reliable for s in system_samples),
                reliability_score  = sum(s.reliability_score for s in system_samples) / n,
                measurement_quality = (
                    "good" if all(s.measurement_quality == "good" for s in system_samples)
                    else "fair" if any(s.measurement_quality != "poor" for s in system_samples)
                    else "poor"
                ),
            )

        avg_metrics = CompressionMetrics(
            original_size      = first.metrics.original_size,
            compressed_size    = first.metrics.compressed_size,
            compression_ratio  = first.metrics.compression_ratio,
            compression_time   = avg_comp_time,
            decompression_time = avg_decomp_time,
            success            = True,
            error_message      = None,
        )

        metadata = {
            **(first.metadata or {}),
            "iterations_requested":  len(results),
            "iterations_successful": count,
            "warmup_iterations":     self.config.warmup_iterations,
            "trim_top_n":            trim_n,
        }

        return BenchmarkResult(
            image_path   = first.image_path,
            format_name  = first.format_name,
            metrics      = avg_metrics,
            metadata     = metadata,
            system_metrics = avg_system,
        )

    def _log_result(
        self,
        result: BenchmarkResult,
        log_callback,
        num_iterations: int = 1,
    ) -> None:
        m = result.metrics

        if not m.success:
            log_callback(f"    {result.image_path.name}: FAILED — {m.error_message}")
            return

        avg_label = f" (avg of {num_iterations} runs)" if num_iterations > 1 else ""
        log_callback(f"    {result.image_path.name}{avg_label}")
        log_callback(f"       Size:          {m.original_size:,} B → {m.compressed_size:,} B")
        log_callback(f"       Savings:       {m.space_saving_percent:.1f}% | Ratio: {m.compression_ratio:.2f}x")
        log_callback(f"       Compression:   {m.compression_time:.3f}s ({m.compression_speed_mbps:.1f} MB/s)")
        log_callback(f"       Decompression: {m.decompression_time:.3f}s ({m.decompression_speed_mbps:.1f} MB/s)")

        if result.system_metrics:
            sm = result.system_metrics
            log_callback(
                f"       CPU: Avg {sm.cpu_percent_normalized:.1f}% "
                f"| Peak {sm.peak_cpu_percent_normalized:.1f}%"
            )
            log_callback(
                f"       RAM: Avg {sm.avg_ram_mb:.1f} MB | Peak {sm.peak_ram_mb:.1f} MB"
                f" | Net peak {sm.net_peak_ram_mb:.1f} MB"
            )
            log_callback(
                f"       I/O: Read {sm.total_io_read_mb:.2f} MB "
                f"| Write {sm.total_io_write_mb:.2f} MB "
                f"| Total {sm.io_total_mb:.2f} MB"
            )
            if sm.measurement_quality == "none":
                log_callback("       Warning: No samples collected — metrics unavailable")
            elif not sm.is_reliable:
                log_callback(f"       Warning: {sm.measurement_quality} quality ({sm.sample_count} samples)")

    def _verify_result(
        self,
        compressor: ImageCompressor,
        img_path: Path,
        result: BenchmarkResult,
        log_callback,
    ) -> None:
        format_dir      = self.config.output_dir / compressor.name
        compressed_path = format_dir / f"{img_path.stem}{compressor.extension}"

        if not compressed_path.exists():
            return

        temp_file = None
        if self.config.strip_metadata:
            verification_input, stripped = self._strip_metadata(img_path)
            temp_file = verification_input if stripped else None
            if not stripped:
                log_callback("       Verification: FAILED — could not create stripped input")
                return
        else:
            verification_input = img_path

        try:
            verification = ImageVerifier.verify_lossless(
                verification_input,
                compressed_path,
                compressor_factory=CompressorFactory,
                temp_dir=format_dir,
            )

            key = (img_path.name, compressor.name)
            self.verification_results[key] = verification

            if verification.is_lossless:
                log_callback("       Verification: LOSSLESS (100.0000% accurate)")
            else:
                log_callback("       Verification: LOSSY")
                if verification.error_message:
                    log_callback(f"          Error: {verification.error_message}")
                else:
                    log_callback(f"          Max difference:    {verification.max_difference:.2f}")
                    log_callback(f"          Different pixels:  {verification.different_pixels:,} / {verification.total_pixels:,}")
                    log_callback(f"          Accuracy:          {verification.accuracy_percent:.4f}%")

        finally:
            if temp_file is not None:
                try:
                    temp_file.unlink()
                except OSError:
                    pass


# ============================================================================
# Summarizer
# ============================================================================

class BenchmarkSummarizer:
    """Format, print, and export benchmark results."""

    @staticmethod
    def generate_unique_filename(config: BenchmarkConfig) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if len(config.compressor_names) <= 3:
            compressors_str = "-".join(config.compressor_names)
        else:
            compressors_str = (
                "-".join(config.compressor_names[:3])
                + f"-plus{len(config.compressor_names) - 3}"
            )

        levels_str     = "-".join(level.name for level in config.compression_levels)
        iterations_str = f"iter{config.num_iterations}"
        if config.warmup_iterations > 0:
            iterations_str += f"w{config.warmup_iterations}"
        images_count = f"{len(config.image_paths)}img"

        flags = []
        if config.verify_lossless:   flags.append("verify")
        if config.strip_metadata:    flags.append("strip")
        if config.monitor_resources: flags.append("monitor")
        if config.isolation.high_priority: flags.append("isolate")
        if config.isolation.cpu_core is not None:
            flags.append(f"core{config.isolation.cpu_core}")
        flags_str = "-".join(flags) if flags else "basic"

        filename = "_".join([
            "results", timestamp, compressors_str,
            levels_str, iterations_str, images_count, flags_str,
        ]) + ".json"

        if len(filename) > 200:
            filename = (
                f"results_{timestamp}_"
                f"{len(config.compressor_names)}comp_"
                f"{len(config.compression_levels)}lvl_"
                f"{images_count}_{flags_str}.json"
            )

        return filename

    @staticmethod
    def print_compression_summary(
        results: List[BenchmarkResult],
        log_callback,
    ) -> None:
        log_callback(f"\n{'=' * 70}")
        log_callback("COMPRESSION SUMMARY")
        log_callback(f"{'=' * 70}")

        by_format: Dict[str, List[BenchmarkResult]] = {}
        for result in results:
            if result.metrics.success:
                by_format.setdefault(result.format_name, []).append(result)

        for format_name, result_list in sorted(by_format.items()):
            metrics_list = [r.metrics for r in result_list]
            n = len(metrics_list)

            avg_ratio        = sum(m.compression_ratio       for m in metrics_list) / n
            avg_savings      = sum(m.space_saving_percent     for m in metrics_list) / n
            avg_comp_time    = sum(m.compression_time         for m in metrics_list) / n
            avg_decomp_time  = sum(m.decompression_time       for m in metrics_list) / n
            avg_comp_speed   = sum(m.compression_speed_mbps   for m in metrics_list) / n
            avg_decomp_speed = sum(m.decompression_speed_mbps for m in metrics_list) / n

            log_callback(f"\n{format_name}")
            log_callback(f"   Compression Ratio:        {avg_ratio:.2f}x")
            log_callback(f"   Space Savings:            {avg_savings:.1f}%")
            log_callback(f"   Avg Compression Time:     {avg_comp_time:.3f}s ({avg_comp_speed:.1f} MB/s)")
            log_callback(f"   Avg Decompression Time:   {avg_decomp_time:.3f}s ({avg_decomp_speed:.1f} MB/s)")

            system_samples = [r.system_metrics for r in result_list if r.system_metrics]
            if system_samples:
                ns = len(system_samples)
                avg_cpu_norm  = sum(s.cpu_percent_normalized      for s in system_samples) / ns
                peak_cpu_norm = max(s.peak_cpu_percent_normalized  for s in system_samples)
                avg_ram       = sum(s.avg_ram_mb      for s in system_samples) / ns
                peak_ram      = max(s.peak_ram_mb     for s in system_samples)
                net_peak_ram  = max(s.net_peak_ram_mb for s in system_samples)
                avg_io        = sum(s.io_total_mb     for s in system_samples) / ns

                log_callback(
                    f"   Avg CPU Usage:            {avg_cpu_norm:.1f}%"
                    f" | Peak {peak_cpu_norm:.1f}%"
                )
                log_callback(
                    f"   Avg RAM Usage:            {avg_ram:.1f} MB"
                    f" (peak {peak_ram:.1f} MB, net peak {net_peak_ram:.1f} MB)"
                )
                log_callback(f"   Avg I/O Total:            {avg_io:.2f} MB")

    @staticmethod
    def print_verification_summary(
        verification_results: Dict[tuple, VerificationResult],
        log_callback,
    ) -> None:
        if not verification_results:
            return

        log_callback(f"\n{'=' * 70}")
        log_callback("LOSSLESS VERIFICATION SUMMARY")
        log_callback(f"{'=' * 70}")

        total    = len(verification_results)
        lossless = sum(1 for v in verification_results.values() if v.is_lossless)
        lossy    = total - lossless

        log_callback(f"\nTotal Tests:      {total}")
        log_callback(f"  Lossless:       {lossless} ({100 * lossless / total:.1f}%)")
        log_callback(f"  Lossy:          {lossy}    ({100 * lossy    / total:.1f}%)")

        if lossy > 0:
            log_callback("\nLossy compressions detected:")
            for (img_name, comp_name), v in verification_results.items():
                if not v.is_lossless:
                    log_callback(f"   {img_name} with {comp_name}")
                    log_callback(
                        f"      Max diff: {v.max_difference:.2f}, "
                        f"Different pixels: {v.different_pixels:,}"
                    )
        else:
            log_callback("\nAll compressions are truly lossless.")

    @staticmethod
    def print_scenario_analysis(
        results: List[BenchmarkResult],
        log_callback,
    ) -> None:
        if not results:
            return

        metrics_to_analyze = [
            ("compression_ratio", "Compression Ratio"),
            ("ram_usage",         "RAM Usage"),
            ("cpu_usage",         "CPU Usage"),
            ("io_total",          "I/O Total"),
        ]

        for metric_key, metric_name in metrics_to_analyze:
            scenarios = ScenarioAnalyzer.identify_scenarios(results, metric_key)
            if scenarios["best"] and scenarios["worst"]:
                log_callback(f"\n{'=' * 70}")
                log_callback(f"SCENARIO ANALYSIS: {metric_name}")
                log_callback(f"{'=' * 70}")
                ScenarioAnalyzer.print_scenario_comparison(scenarios, log_callback)

    @staticmethod
    def export_results_json(
        results: List[BenchmarkResult],
        verification_results: Dict[tuple, VerificationResult],
        output_dir: Path,
        config: BenchmarkConfig,
    ) -> Path:
        filename    = BenchmarkSummarizer.generate_unique_filename(config)
        output_file = output_dir / filename
        output_dir.mkdir(parents=True, exist_ok=True)

        results_data = []
        for result in results:
            m = result.metrics
            entry: Dict = {
                "image":           str(result.image_path.name),
                "image_full_path": str(result.image_path),
                "format":          result.format_name,
                "compression": {
                    "original_size":            m.original_size,
                    "compressed_size":          m.compressed_size,
                    "compression_ratio":        m.compression_ratio,
                    "space_saving_percent":     m.space_saving_percent,
                    "compression_time":         m.compression_time,
                    "decompression_time":       m.decompression_time,
                    "compression_speed_mbps":   m.compression_speed_mbps,
                    "decompression_speed_mbps": m.decompression_speed_mbps,
                    "success":                  m.success,
                    "error_message":            m.error_message,
                },
                "metadata": result.metadata,
            }

            if result.system_metrics:
                sm = result.system_metrics
                entry["system_metrics"] = {
                    "cpu": {
                        "avg_percent":         sm.avg_cpu_percent,
                        "max_percent":         sm.max_cpu_percent,
                        "avg_process_percent": sm.avg_process_cpu,
                        "max_process_percent": sm.max_process_cpu,
                    },
                    "memory": {
                        "avg_mb":                    sm.avg_ram_mb,
                        "peak_mb":                   sm.peak_ram_mb,
                        "net_peak_mb":               sm.net_peak_ram_mb,
                        "baseline_mb":               sm.ram_baseline_mb,
                        "ram_efficiency_mb_per_sec": sm.ram_efficiency_mb_per_sec,
                    },
                    "io": {
                        "total_read_mb":  sm.total_io_read_mb,
                        "total_write_mb": sm.total_io_write_mb,
                        "total_mb":       sm.io_total_mb,
                    },
                    "timing": {
                        "duration_seconds": sm.duration_seconds,
                        "sample_count":     sm.sample_count,
                    },
                }
            results_data.append(entry)

        verification_data = [
            {
                "image":            img_name,
                "compressor":       comp_name,
                "is_lossless":      bool(v.is_lossless),
                "max_difference":   float(v.max_difference)   if v.max_difference   is not None else None,
                "different_pixels": int(v.different_pixels)   if v.different_pixels is not None else None,
                "total_pixels":     int(v.total_pixels)       if v.total_pixels     is not None else None,
                "accuracy_percent": float(v.accuracy_percent) if v.accuracy_percent is not None else None,
                "error_message":    str(v.error_message)      if v.error_message    else None,
            }
            for (img_name, comp_name), v in verification_results.items()
        ]

        successful = [r for r in results if r.metrics.success]
        total_original   = sum(r.metrics.original_size   for r in successful)
        total_compressed = sum(r.metrics.compressed_size for r in successful)

        scenarios_data: Dict = {}
        if config.monitor_resources:
            for metric_key, _ in [
                ("compression_ratio", "compression_ratio"),
                ("ram_usage",         "ram_usage"),
                ("cpu_usage",         "cpu_usage"),
                ("io_total",          "io_total"),
            ]:
                scenarios = ScenarioAnalyzer.identify_scenarios(successful, metric_key)
                if scenarios["best"] and scenarios["worst"]:
                    scenarios_data[metric_key] = {
                        side: {
                            "file":              s.file_path.name,
                            "file_size_mb":      s.file_size_mb,
                            "compression_ratio": s.compression_ratio,
                            "peak_ram_mb":       s.system_metrics.peak_ram_mb,
                            "avg_cpu_percent":   s.system_metrics.avg_process_cpu,
                            "io_total_mb":       s.system_metrics.io_total_mb,
                            "ram_per_mb":        s.ram_per_mb,
                            "cpu_efficiency":    s.cpu_efficiency,
                        }
                        for side, s in scenarios.items()
                        if s is not None
                    }

        output_data = {
            "benchmark_info": {
                "filename":     filename,
                "timestamp":    datetime.now().isoformat(),
                "generated_by": "Image Compression Benchmark Tool v1.0",
            },
            "benchmark_config": {
                "num_iterations":     config.num_iterations,
                "warmup_iterations":  config.warmup_iterations,
                "verify_lossless":    config.verify_lossless,
                "strip_metadata":     config.strip_metadata,
                "monitor_resources":  config.monitor_resources,
                "high_priority":      config.isolation.high_priority,
                "cpu_affinity_core":  config.isolation.cpu_core,
                "compressors":        config.compressor_names,
                "compression_levels": [level.name for level in config.compression_levels],
            },
            "summary": {
                "total_images":            len(results),
                "successful":              len(successful),
                "failed":                  len(results) - len(successful),
                "total_original_size":     total_original,
                "total_compressed_size":   total_compressed,
                "overall_compression_ratio": (
                    total_original / total_compressed if total_compressed > 0 else 0
                ),
                "lossless_verified": sum(1 for v in verification_results.values() if v.is_lossless),
                "lossy_detected":    sum(1 for v in verification_results.values() if not v.is_lossless),
            },
            "scenarios":    scenarios_data,
            "results":      results_data,
            "verification": verification_data,
        }

        try:
            with open(output_file, "w", encoding="utf-8") as fh:
                json.dump(output_data, fh, indent=2, ensure_ascii=False)
            print(f"Results saved to: {output_file.name}")

        except Exception as exc:
            print(f"Error writing JSON: {exc}")
            fallback = output_dir / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            print(f"Trying fallback filename: {fallback.name}")
            with open(fallback, "w", encoding="utf-8") as fh:
                json.dump(output_data, fh, indent=2, ensure_ascii=False)
            output_file = fallback

        return output_file


# ============================================================================
# Image finder
# ============================================================================

class ImageFinder:
    """Utility for locating image files in a directory tree."""

    DEFAULT_PATTERNS = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff", "*.tif"]

    @staticmethod
    def find_images(
        directory: Path,
        patterns: Optional[List[str]] = None,
        recursive: bool = True,
    ) -> List[Path]:
        if patterns is None:
            patterns = ImageFinder.DEFAULT_PATTERNS

        images: List[Path] = []
        for pattern in patterns:
            if recursive:
                images.extend(directory.rglob(pattern))
            else:
                images.extend(directory.glob(pattern))

        return sorted(set(images))