"""Shared benchmark logic used by both the CLI and the GUI.

Classes:
  BenchmarkConfig     : Immutable configuration for one benchmark session.
  BenchmarkRunner     : Executes the configured benchmark and collects results.
  BenchmarkSummarizer : Formats and exports results to the console and JSON.
  ImageFinder         : Utility for locating image files in directories.
"""

import json
import logging
import tempfile
from collections import defaultdict
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


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkConfig:
    """All parameters that control a single benchmark session.

    Passed to BenchmarkRunner.run() and embedded verbatim in every JSON report
    so that any result file is fully self-describing.

    Attributes:
        dataset_dir: Root directory of the image dataset (informational only;
            actual images are listed in image_paths).
        output_dir: Directory where compressed files and reports are written.
        libs_dir: Directory that contains native libraries / CLI binaries for
            compressors that need them (e.g. libcharls.dll, oxipng.exe).
        compressor_names: Factory keys for the compressors to benchmark,
            e.g. ['charls', 'webp'].
        image_paths: Ordered list of source images to compress.
        compression_levels: Compression levels to test for each compressor /
            image combination.
        verify_lossless: When True, decompress each result and compare it
            pixel-by-pixel against the original to confirm losslessness.
        strip_metadata: When True, strip EXIF/XMP/ICC data before compressing
            so that metadata does not influence the size measurement.
        num_iterations: Number of measurement runs per image / level.
            Results are averaged after optional trimming.
        warmup_iterations: Number of warm-up runs executed before measurement
            starts.  These runs are excluded from the average.
        trim_top_n: Drop the N slowest measurement runs before averaging.
            Useful for removing OS scheduling outliers.
        monitor_resources: When True, collect per-run CPU, RAM, and I/O metrics
            via SystemMonitor.
        isolation: Controls high-priority scheduling and optional CPU core
            pinning.  Build an IsolationConfig(high_priority=True, cpu_core=1)
            and pass it here; use IsolationConfig() (all defaults) or None for
            no isolation.
    """

    dataset_dir:        Path
    output_dir:         Path
    libs_dir:           Path
    compressor_names:   List[str]            # factory keys, e.g. ["charls", "webp"]
    image_paths:        List[Path]
    compression_levels: List[CompressionLevel]
    verify_lossless:    bool           = True
    strip_metadata:     bool           = True
    num_iterations:     int            = 1   # measurement iterations per image
    warmup_iterations:  int            = 1   # warm-up runs excluded from averages
    trim_top_n:         int            = 0   # drop N slowest runs before averaging
    monitor_resources:  bool           = True
    isolation:          IsolationConfig = field(default_factory=IsolationConfig)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class BenchmarkRunner:
    """Executes the benchmark described by a BenchmarkConfig.

    CPU affinity and priority handling:
        Isolation settings come entirely from config.isolation (an
        IsolationConfig).  ProcessIsolator reads them in isolate() and appends
        human-readable notes to IsolationState.isolation_notes, which are
        forwarded to the progress callback.

    Threading:
        run() is designed to be called from a background thread (as the GUI
        does).  All progress messages are delivered via the progress_callback
        so the caller can marshal them onto the UI thread if needed.

    Attributes:
        config: The BenchmarkConfig passed at construction time.
        results: Accumulated BenchmarkResult objects after run() completes.
        verification_results: Lossless-check outcomes keyed by
            (image_name, compressor_name).
        should_stop: Set to True by stop() to request early termination.
    """

    def __init__(self, config: BenchmarkConfig):
        self.config               = config
        self.results:              List[BenchmarkResult]           = []
        self.verification_results: Dict[tuple, VerificationResult] = {}
        self.should_stop          = False
        self.isolator             = ProcessIsolator(config.isolation)

    def stop(self) -> None:
        """Signal the runner to abort after the current iteration completes."""
        self.should_stop = True

    def run(
        self,
        progress_callback=None,
    ) -> tuple[List[BenchmarkResult], Dict[tuple, VerificationResult]]:
        """Execute the full benchmark and return collected results.

        Iterates over every combination of compressor × level × image.  For
        each combination, warm-up runs are executed first (not recorded), then
        num_iterations measurement runs whose timing is averaged.

        Args:
            progress_callback: Optional callable(str) invoked with each log
                line.  Safe to call from a background thread — the callback is
                responsible for thread-safe forwarding to the UI.

        Returns:
            A two-tuple of:
              - List of averaged BenchmarkResult objects (one per
                compressor × level × image combination).
              - Dict mapping (image_name, compressor_name) to VerificationResult
                (only populated when config.verify_lossless is True).
        """
        self.results               = []
        self.verification_results  = {}
        self.should_stop           = False

        def log(message: str) -> None:
            if progress_callback:
                progress_callback(message)

        # Apply process isolation (priority boost + optional CPU core pinning).
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

                        # Warm-up runs: prime OS file cache and JIT paths without
                        # recording any timing data.
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

        # Restore process priority / affinity after the benchmark finishes so
        # that the host system returns to its normal scheduling behaviour.
        if self.config.isolation.enabled:
            self.isolator.restore()
            log("\nProcess isolation restored.")

        if self.should_stop:
            log("\nBenchmark stopped by user.")

        return self.results, self.verification_results

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    def _find_lib_for_compressor(self, compressor_name: str) -> Optional[Path]:
        """Search libs_dir for a native library matching the compressor name.

        Tries 'lib<name>.*' first (POSIX convention), then '<name>.*' (Windows).

        Args:
            compressor_name: Factory key of the compressor, e.g. 'charls'.

        Returns:
            Path to the first matching file, or None if nothing is found.
        """
        patterns = [
            f"lib{compressor_name.lower()}.*",  # e.g. libcharls.so / .dll
            f"{compressor_name.lower()}.*",     # e.g. oxipng.exe
        ]
        for pattern in patterns:
            matches = list(self.config.libs_dir.glob(pattern))
            if matches:
                return matches[0]
        return None

    def _prepare_input(self, image_path: Path, strip: bool) -> tuple:
        """Convert a source image to a normalised temp PNG ready for compression.

        Always writes a temp PNG so every compressor plugin receives a consistent
        input format (PNG, compressor-safe colour mode) regardless of the source
        file type.  The only difference between strip=True and strip=False is
        whether the metadata (.info dict) is carried into the output file.

        Output is always PNG because:
          - PNG is lossless and supports all bit depths / modes used here.
          - Every compressor plugin accepts PNG as input.
          - ImageSizeCalculator always computes size from raw pixel data
            (width × height × bpp), so the PNG container overhead is irrelevant.

        Args:
            image_path: Source image file, any Pillow-readable format.
            strip: When True, pixel data is rebuilt into a fresh Image with an
                empty .info dict so all EXIF / XMP / ICC metadata is discarded.
                When False, the image is saved with its original .info dict
                intact so metadata is preserved in the output.

        Returns:
            (temp_path, True)   on success — caller must delete temp_path when done.
            (image_path, False) on failure — original path returned, nothing to delete.
        """
        try:
            with Image.open(image_path) as img:
                img.load()  # Force full decode before the file handle closes.

                # Normalise to a compressor-safe colour mode.
                # Pillow can open exotic modes (P, PA, CMYK, …) that some plugins
                # do not handle.  Keep L, LA, RGB, RGBA; convert everything else.
                mode = img.mode
                if mode not in ("L", "LA", "RGB", "RGBA"):
                    mode = "RGBA" if "A" in img.getbands() else "RGB"
                    img = img.convert(mode)

                if strip:
                    # Rebuild from raw pixel data into a fresh Image with an empty
                    # .info dict — this guarantees no metadata survives.
                    out_img = Image.new(mode, img.size)
                    out_img.putdata(img.getdata())
                else:
                    # Preserve the original .info dict (EXIF, ICC, XMP, …).
                    # img.copy() duplicates both pixel data and the .info dict.
                    out_img = img.copy()

            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            temp_path = Path(tmp.name)
            tmp.close()
            # pnginfo=None lets Pillow carry the .info dict through when strip=False.
            out_img.save(temp_path, format="PNG")
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
        """Run a single compress + (optional) monitor cycle for one image.

        The SystemMonitor is started before compression and stopped in the
        finally block so that metrics are always captured even when compress()
        raises an exception.  If the compressor raises, result is None and
        system_metrics is simply discarded.

        The temporary stripped PNG (when strip_metadata is True) is always
        deleted in the finally block to avoid leaking temp files on error.

        Args:
            compressor: Initialised compressor plugin to use.
            image_path: Original (pre-strip) source image path.
            level: Compression level to pass to the plugin.
            monitor: When False, system monitoring is skipped (used for warm-up
                runs to avoid polluting resource measurements).

        Returns:
            BenchmarkResult for this single run.
        """
        format_dir = self.config.output_dir / compressor.name
        format_dir.mkdir(parents=True, exist_ok=True)

        # Always prepare a normalised temp PNG so every plugin receives a
        # consistent input format.  _prepare_input() handles mode normalisation
        # and optionally strips metadata depending on config.strip_metadata.
        actual_path, prepared = self._prepare_input(image_path, strip=self.config.strip_metadata)
        temp_file = actual_path if prepared else None

        sys_monitor = None
        if monitor:
            sys_monitor = SystemMonitor(
                sampling_interval=0.05,
                adaptive=True,
                force_pre_post=True,
            )
            # Pass the pinned core count so the monitor can normalise CPU% correctly:
            # a process pinned to 1 core has a ceiling of 100 %, not N×100 %.
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
            # Always stop the monitor and attach metrics to the result.
            # If compress() raised, result is None and metrics are discarded.
            if sys_monitor:
                system_metrics = sys_monitor.stop()
                if result is not None:
                    result.system_metrics = system_metrics

            # Always clean up the temp PNG regardless of whether compression succeeded.
            if temp_file is not None:
                try:
                    temp_file.unlink()
                except OSError as exc:
                    logging.debug(f"Could not delete temp file {temp_file}: {exc}")

    def _average_results(self, results: List[BenchmarkResult]) -> BenchmarkResult:
        """Average timing and system metrics over multiple iteration results.

        Size fields (original_size, compressed_size, compression_ratio) are
        taken from the first successful result because they are deterministic —
        every iteration produces exactly the same output file for the same
        compressor / level / input triple.  Averaging them would be meaningless.

        RAM baseline is also taken from the first sample because it represents
        the system state *before* compression started — not something that
        should be averaged across runs.

        Args:
            results: List of BenchmarkResult objects from successive iterations.
                May contain a mix of successful and failed runs.

        Returns:
            A single BenchmarkResult whose metrics reflect the average of all
            successful iterations after optional trim_top_n trimming.

        Raises:
            ValueError: If results is empty or all iterations failed.
        """
        if not results:
            raise ValueError("No benchmark results provided.")

        successful = [r for r in results if r.metrics.success]
        if not successful:
            raise ValueError("All benchmark iterations failed; nothing to average.")

        if len(successful) == 1:
            return successful[0]

        # Drop the N slowest runs (by compression_time) before computing averages.
        # This mitigates OS scheduling spikes that inflate measured times.
        trim_n = getattr(self.config, "trim_top_n", 0)
        if trim_n > 0 and len(successful) > trim_n:
            successful = sorted(successful, key=lambda r: r.metrics.compression_time)
            successful = successful[:len(successful) - trim_n]  # drop tail (slowest)

        first = successful[0]
        count = len(successful)

        avg_comp_time   = sum(r.metrics.compression_time   for r in successful) / count
        avg_decomp_time = sum(r.metrics.decompression_time for r in successful) / count

        # Average system metrics across all successful iterations that have them.
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
                # ram_baseline_mb is the pre-compression system RAM level; it should
                # NOT be averaged — take the first sample as the representative value.
                ram_baseline_mb    = system_samples[0].ram_baseline_mb,
                total_io_read_mb   = sum(s.total_io_read_mb   for s in system_samples) / n,
                total_io_write_mb  = sum(s.total_io_write_mb  for s in system_samples) / n,
                duration_seconds   = sum(s.duration_seconds   for s in system_samples) / n,
                sample_count       = sum(s.sample_count       for s in system_samples) // n,
                logical_core_count = system_samples[0].logical_core_count,
                is_reliable        = all(s.is_reliable for s in system_samples),
                reliability_score  = sum(s.reliability_score for s in system_samples) / n,
                # Measurement quality is degraded to the worst observed level:
                # good only when all samples are good, poor when all are poor,
                # fair otherwise.
                measurement_quality = (
                    "good" if all(s.measurement_quality == "good" for s in system_samples)
                    else "fair" if any(s.measurement_quality != "poor" for s in system_samples)
                    else "poor"
                ),
            )

        # Size and ratio are deterministic across iterations — use first result.
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
            image_path     = first.image_path,
            format_name    = first.format_name,
            metrics        = avg_metrics,
            metadata       = metadata,
            system_metrics = avg_system,
        )

    def _log_result(
        self,
        result: BenchmarkResult,
        log_callback,
        num_iterations: int = 1,
    ) -> None:
        """Write a single benchmark result to the progress log.

        Args:
            result: The (possibly averaged) result to display.
            log_callback: Callable(str) that accepts one log line at a time.
            num_iterations: Number of iterations that were averaged; used to
                annotate the output line when greater than 1.
        """
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
        """Verify that a compressed file is truly lossless against the original.

        Strips metadata from the original (matching the pre-compression step),
        decompresses the output, and performs a pixel-exact comparison.
        Results are stored in self.verification_results.

        Args:
            compressor: The plugin used to produce the compressed file.
            img_path: Original (pre-strip) image path.
            result: The benchmark result associated with this image (used for
                format_dir resolution only; not mutated).
            log_callback: Callable(str) for progress messages.
        """
        format_dir      = self.config.output_dir / compressor.name
        compressed_path = format_dir / f"{img_path.stem}{compressor.extension}"

        if not compressed_path.exists():
            return

        # Prepare the verification input using the same strip setting that was
        # used during compression, so pixel data is identical to what the
        # compressor received.
        verification_input, prepared = self._prepare_input(img_path, strip=self.config.strip_metadata)
        temp_file = verification_input if prepared else None
        if not prepared:
            log_callback("       Verification: FAILED — could not prepare input")
            return

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


# ---------------------------------------------------------------------------
# Summarizer
# ---------------------------------------------------------------------------


class BenchmarkSummarizer:
    """Format, print, and export benchmark results.

    All methods are static — this class acts as a namespace for output-related
    utilities and is never instantiated.
    """

    @staticmethod
    def generate_unique_filename(
        config: BenchmarkConfig,
        level: Optional[CompressionLevel] = None,
    ) -> str:
        """Generate a unique filename for a benchmark result JSON file.

        Encodes the most important run parameters (timestamp, compressors,
        level, iteration count, image count, and feature flags) into the
        filename so that the file is self-identifying without opening it.

        Args:
            config: The BenchmarkConfig for this session.
            level: When provided, only this level's name is embedded in the
                filename (used when saving per-level files).  When None, all
                levels from the config are joined (fallback for combined files).

        Returns:
            A filename string ending in '.json', at most 200 characters long.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Limit compressor list length so filenames stay human-readable.
        if len(config.compressor_names) <= 3:
            compressors_str = "-".join(config.compressor_names)
        else:
            compressors_str = (
                "-".join(config.compressor_names[:3])
                + f"-plus{len(config.compressor_names) - 3}"
            )

        # Use only the supplied level name, or join all levels from the config.
        if level is not None:
            levels_str = level.name
        else:
            levels_str = "-".join(lv.name for lv in config.compression_levels)

        iterations_str = f"iter{config.num_iterations}"
        if config.warmup_iterations > 0:
            iterations_str += f"w{config.warmup_iterations}"
        images_count = f"{len(config.image_paths)}img"

        # Build a short feature-flag suffix so two runs with different settings
        # on the same dataset do not accidentally overwrite each other.
        flags = []
        if config.verify_lossless:         flags.append("verify")
        if config.strip_metadata:          flags.append("strip")
        if config.monitor_resources:       flags.append("monitor")
        if config.isolation.high_priority: flags.append("isolate")
        if config.isolation.cpu_core is not None:
            flags.append(f"core{config.isolation.cpu_core}")
        flags_str = "-".join(flags) if flags else "basic"

        filename = "_".join([
            "results", timestamp, compressors_str,
            levels_str, iterations_str, images_count, flags_str,
        ]) + ".json"

        # Fall back to a shorter name if the full one exceeds 200 characters.
        # Most filesystems allow 255 bytes, but 200 leaves room for path prefixes.
        if len(filename) > 200:
            filename = (
                f"results_{timestamp}_"
                f"{len(config.compressor_names)}comp_"
                f"{levels_str}_"
                f"{images_count}_{flags_str}.json"
            )

        return filename

    @staticmethod
    def print_compression_summary(
        results: List[BenchmarkResult],
        log_callback,
    ) -> None:
        """Print per-format averaged metrics to the log.

        Groups all successful results by format_name and emits one block of
        summary statistics per compressor.

        Args:
            results: Full list of BenchmarkResult objects from the run.
            log_callback: Callable(str) for progress / log output.
        """
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
        """Print a losslessness check summary to the log.

        Args:
            verification_results: Dict keyed by (image_name, compressor_name)
                with VerificationResult values.  Empty dict is a no-op.
            log_callback: Callable(str) for progress / log output.
        """
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
        """Print best / worst case scenario comparisons to the log.

        Analyses four metrics: compression ratio, RAM usage, CPU usage, and
        total I/O.  Skips metrics for which ScenarioAnalyzer finds no contrast.

        Args:
            results: Full list of BenchmarkResult objects from the run.
            log_callback: Callable(str) for progress / log output.
        """
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
    ) -> List[Path]:
        """Save benchmark results to separate JSON files — one file per compression level.

        Results are split by the 'compression_level' key stored in each
        BenchmarkResult's metadata dict.  Verification results are matched to
        their level via the same metadata on the corresponding benchmark result.

        Args:
            results: Full list of BenchmarkResult objects from the run.
            verification_results: Dict keyed by (image_name, compressor_name).
            output_dir: Directory where JSON files will be written.
            config: The BenchmarkConfig used for this session (embedded in each file).

        Returns:
            List of Path objects for the files that were successfully written,
            in level iteration order.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # --- Group results by compression level ---
        results_by_level: Dict[str, List[BenchmarkResult]] = defaultdict(list)
        for result in results:
            level_name = (result.metadata or {}).get("compression_level", "UNKNOWN")
            results_by_level[level_name].append(result)

        # Build a fast (image_name, compressor_name) -> level_name lookup so that
        # verification results can be routed to the correct per-level bucket.
        result_level_lookup: Dict[tuple, str] = {
            (str(r.image_path.name), r.format_name): (r.metadata or {}).get("compression_level", "UNKNOWN")
            for r in results
        }

        # --- Group verification results by compression level ---
        verif_by_level: Dict[str, Dict[tuple, VerificationResult]] = defaultdict(dict)
        for (img_name, comp_name), v in verification_results.items():
            level_name = result_level_lookup.get((img_name, comp_name), "UNKNOWN")
            verif_by_level[level_name][(img_name, comp_name)] = v

        saved_paths: List[Path] = []

        for level_name, level_results in results_by_level.items():
            # Resolve the level name back to a CompressionLevel enum value so
            # generate_unique_filename can embed it in the filename correctly.
            matching_level = next(
                (lv for lv in config.compression_levels if lv.name == level_name),
                None,
            )

            filename    = BenchmarkSummarizer.generate_unique_filename(config, level=matching_level)
            output_file = output_dir / filename

            level_verif = verif_by_level.get(level_name, {})

            successful       = [r for r in level_results if r.metrics.success]
            total_original   = sum(r.metrics.original_size   for r in successful)
            total_compressed = sum(r.metrics.compressed_size for r in successful)

            # --- Build the per-result detail list for this level ---
            results_data = []
            for result in level_results:
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

            # --- Build the verification list for this level ---
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
                for (img_name, comp_name), v in level_verif.items()
            ]

            # --- Build scenario analysis for this level ---
            scenarios_data: Dict = {}
            if config.monitor_resources:
                for metric_key in ("compression_ratio", "ram_usage", "cpu_usage", "io_total"):
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
                    "compression_levels": [level_name],  # one file per level
                },
                "summary": {
                    "total_images":              len(level_results),
                    "successful":                len(successful),
                    "failed":                    len(level_results) - len(successful),
                    "total_original_size":       total_original,
                    "total_compressed_size":     total_compressed,
                    "overall_compression_ratio": (
                        total_original / total_compressed if total_compressed > 0 else 0
                    ),
                    "lossless_verified": sum(1 for v in level_verif.values() if v.is_lossless),
                    "lossy_detected":    sum(1 for v in level_verif.values() if not v.is_lossless),
                },
                "scenarios":    scenarios_data,
                "results":      results_data,
                "verification": verification_data,
            }

            try:
                with open(output_file, "w", encoding="utf-8") as fh:
                    json.dump(output_data, fh, indent=2, ensure_ascii=False)
                print(f"Results saved to: {output_file.name}")
                saved_paths.append(output_file)

            except Exception as exc:
                # Primary filename failed — fall back to a minimal safe name
                # to avoid losing the data completely.
                print(f"Error writing JSON for level {level_name}: {exc}")
                fallback = output_dir / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{level_name}.json"
                with open(fallback, "w", encoding="utf-8") as fh:
                    json.dump(output_data, fh, indent=2, ensure_ascii=False)
                saved_paths.append(fallback)

        return saved_paths


# ---------------------------------------------------------------------------
# Image finder
# ---------------------------------------------------------------------------


class ImageFinder:
    """Utility for locating image files in a directory tree."""

    DEFAULT_PATTERNS = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff", "*.tif"]

    @staticmethod
    def find_images(
        directory: Path,
        patterns: Optional[List[str]] = None,
        recursive: bool = True,
    ) -> List[Path]:
        """Find image files in a directory, sorted and deduplicated.

        Args:
            directory: Root directory to search.
            patterns: Glob patterns to match.  Defaults to DEFAULT_PATTERNS
                when None.
            recursive: When True, descend into subdirectories (rglob).
                When False, only the immediate children are checked (glob).

        Returns:
            Sorted, deduplicated list of matching Path objects.
        """
        if patterns is None:
            patterns = ImageFinder.DEFAULT_PATTERNS

        images: List[Path] = []
        for pattern in patterns:
            if recursive:
                images.extend(directory.rglob(pattern))
            else:
                images.extend(directory.glob(pattern))

        # Use a set to remove duplicates that can arise when multiple patterns
        # match the same file (e.g. *.tif and *.tiff on some filesystems).
        return sorted(set(images))