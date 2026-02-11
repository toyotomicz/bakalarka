"""
System Resource Monitoring Module
Monitors CPU, RAM, and I/O metrics during compression benchmarks.

Classes:
- SystemMonitor   – adaptive background-thread sampling of CPU / RAM / I/O
- ProcessIsolator – process isolation for fair benchmarking (priority, CPU affinity)
- ScenarioAnalyzer / ScenarioMetrics – best / worst case analysis
"""

import gc
import logging
import os
import platform
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import psutil

logger = logging.getLogger(__name__)


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class SystemSnapshot:
    """Single point-in-time system resource sample."""
    timestamp: float
    cpu_percent: float           # total system CPU (%)
    process_cpu_percent: float   # this process CPU, cumulative across cores (%)
    ram_used_mb: float           # RSS memory of this process (MB)
    ram_available_mb: float      # free system RAM (MB)
    io_read_mb: float            # cumulative process reads (MB)
    io_write_mb: float           # cumulative process writes (MB)


@dataclass
class IsolationState:
    """Process state saved before isolation — used to restore after the benchmark."""
    affinity: Optional[List[int]] = None
    nice: Optional[int] = None
    isolated: bool = False
    isolation_notes: List[str] = field(default_factory=list)


@dataclass
class SystemMetrics:
    """Aggregated resource metrics for one benchmark run."""

    # CPU
    avg_cpu_percent: float    # average total system CPU (%)
    max_cpu_percent: float    # peak total system CPU (%)
    avg_process_cpu: float    # average process CPU, cumulative per-core (%)
    max_process_cpu: float    # peak process CPU

    # Memory
    # NOTE: max_ram_mb was removed — it was always identical to peak_ram_mb.
    #       Only peak_ram_mb is kept. Any code that referenced max_ram_mb
    #       should be updated to use peak_ram_mb instead.
    avg_ram_mb: float         # average RSS (MB)
    peak_ram_mb: float        # maximum RSS observed during the run (MB)

    # I/O — delta from start to end of the monitored operation
    total_io_read_mb: float
    total_io_write_mb: float

    # Timing and sample count
    duration_seconds: float
    sample_count: int

    # Measurement quality
    is_reliable: bool = True
    reliability_score: float = 1.0
    measurement_quality: str = "good"

    @property
    def io_total_mb(self) -> float:
        return self.total_io_read_mb + self.total_io_write_mb

    @property
    def ram_efficiency_mb_per_sec(self) -> float:
        """Average RAM usage per second of operation."""
        if self.duration_seconds == 0:
            return 0.0
        return self.avg_ram_mb / self.duration_seconds


# ============================================================================
# SYSTEM MONITOR
# ============================================================================

class SystemMonitor:
    """
    Measures CPU, RAM, and I/O in a background thread during compression.

    Key fixes vs. the original version:
    - cpu_percent(interval=None) called in __init__ so the first sample is
      not always 0.0 (psutil needs one warm-up call to initialise its counter).
    - Duplicate max_ram_mb / peak_ram_mb fields collapsed into peak_ram_mb only.
    - Bare except replaced with explicit (psutil.Error, OSError) + debug logging.
    - time.perf_counter() used instead of time.time() for short-duration accuracy.
    """

    # Adaptive sampling intervals
    ULTRA_FAST_INTERVAL = 0.001   # 1 ms  — files < 10 KB
    FAST_INTERVAL       = 0.010   # 10 ms — files < 100 KB
    NORMAL_INTERVAL     = 0.050   # 50 ms — files < 1 MB
    SLOW_INTERVAL       = 0.100   # 100 ms — files >= 1 MB

    MIN_SAMPLES_RELIABLE  = 5
    MIN_SAMPLES_FAIR      = 3
    MIN_DURATION_RELIABLE = 0.010  # 10 ms

    def __init__(
        self,
        sampling_interval: float = 0.05,
        adaptive: bool = True,
        force_pre_post: bool = True,
    ) -> None:
        self.base_sampling_interval = sampling_interval
        self.sampling_interval = sampling_interval
        self.adaptive = adaptive
        self.force_pre_post = force_pre_post

        self.process = psutil.Process(os.getpid())

        # Initialise psutil's internal CPU counter so the first real sample
        # returns a meaningful value instead of 0.0.
        self.process.cpu_percent(interval=None)
        psutil.cpu_percent(interval=None)

        self.snapshots: List[SystemSnapshot] = []
        self.is_monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None

        self.baseline_io_read  = 0.0
        self.baseline_io_write = 0.0
        self.start_time = 0.0
        self.end_time   = 0.0

        self.pre_snapshot:  Optional[SystemSnapshot] = None
        self.post_snapshot: Optional[SystemSnapshot] = None

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self, file_size_bytes: Optional[int] = None) -> None:
        """Start monitoring. Safe to call when already running (no-op)."""
        if self.is_monitoring:
            return

        if file_size_bytes is not None and self.adaptive:
            self._set_adaptive_sampling(file_size_bytes)

        self.snapshots = []
        self.is_monitoring = True
        self.start_time = time.perf_counter()

        if self.force_pre_post:
            self.pre_snapshot = self._take_snapshot()

        try:
            io = self.process.io_counters()
            self.baseline_io_read  = io.read_bytes  / (1024 * 1024)
            self.baseline_io_write = io.write_bytes / (1024 * 1024)
        except (psutil.Error, OSError):
            self.baseline_io_read  = 0.0
            self.baseline_io_write = 0.0

        self.monitor_thread = threading.Thread(
            target=self._monitor_loop, daemon=True, name="SystemMonitor"
        )
        self.monitor_thread.start()

    def stop(self) -> SystemMetrics:
        """Stop monitoring and return aggregated metrics."""
        self.is_monitoring = False
        self.end_time = time.perf_counter()

        if self.force_pre_post:
            self.post_snapshot = self._take_snapshot()

        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)

        return self._calculate_metrics()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _monitor_loop(self) -> None:
        while self.is_monitoring:
            try:
                self.snapshots.append(self._take_snapshot())
            except (psutil.Error, OSError) as e:
                logger.debug("SystemMonitor snapshot error: %s", e)
            time.sleep(self.sampling_interval)

    def _take_snapshot(self) -> SystemSnapshot:
        cpu_system  = psutil.cpu_percent(interval=0)
        cpu_process = self.process.cpu_percent(interval=0)
        mem  = self.process.memory_info()
        vmem = psutil.virtual_memory()

        try:
            io = self.process.io_counters()
            io_read  = io.read_bytes  / (1024 * 1024)
            io_write = io.write_bytes / (1024 * 1024)
        except (psutil.Error, OSError):
            io_read  = 0.0
            io_write = 0.0

        return SystemSnapshot(
            timestamp=time.perf_counter(),
            cpu_percent=cpu_system,
            process_cpu_percent=cpu_process,
            ram_used_mb=mem.rss / (1024 * 1024),
            ram_available_mb=vmem.available / (1024 * 1024),
            io_read_mb=io_read,
            io_write_mb=io_write,
        )

    def _set_adaptive_sampling(self, file_size_bytes: int) -> None:
        est = self._estimate_duration(file_size_bytes)
        if est < 0.01:
            self.sampling_interval = self.ULTRA_FAST_INTERVAL
        elif est < 0.1:
            self.sampling_interval = self.FAST_INTERVAL
        elif est < 1.0:
            self.sampling_interval = self.NORMAL_INTERVAL
        else:
            self.sampling_interval = self.SLOW_INTERVAL

    @staticmethod
    def _estimate_duration(file_size_bytes: int) -> float:
        if file_size_bytes < 10_000:
            return 0.001
        if file_size_bytes < 100_000:
            return 0.01
        if file_size_bytes < 1_000_000:
            return 0.1
        return file_size_bytes / (50 * 1024 * 1024)

    def _assess_quality(self, duration: float, sample_count: int) -> tuple:
        if duration < 0.001:
            return False, 0.0, "unreliable"

        duration_score = min(duration / self.MIN_DURATION_RELIABLE, 1.0)

        if sample_count == 0:
            sample_score = 0.0
        elif sample_count < self.MIN_SAMPLES_FAIR:
            sample_score = sample_count / self.MIN_SAMPLES_FAIR * 0.5
        elif sample_count < self.MIN_SAMPLES_RELIABLE:
            sample_score = 0.5 + (
                (sample_count - self.MIN_SAMPLES_FAIR)
                / (self.MIN_SAMPLES_RELIABLE - self.MIN_SAMPLES_FAIR)
                * 0.3
            )
        else:
            sample_score = 0.8 + min(sample_count / 20, 0.2)

        score = duration_score * 0.4 + sample_score * 0.6

        if score >= 0.8:
            return True,  score, "good"
        if score >= 0.5:
            return True,  score, "fair"
        if score >= 0.3:
            return False, score, "poor"
        return False, score, "unreliable"

    def _calculate_metrics(self) -> SystemMetrics:
        duration = self.end_time - self.start_time

        all_snapshots = list(self.snapshots)
        if self.pre_snapshot:
            all_snapshots.insert(0, self.pre_snapshot)
        if self.post_snapshot:
            all_snapshots.append(self.post_snapshot)

        if not all_snapshots:
            is_rel, score, quality = self._assess_quality(duration, 0)
            return SystemMetrics(
                avg_cpu_percent=0.0, max_cpu_percent=0.0,
                avg_process_cpu=0.0, max_process_cpu=0.0,
                avg_ram_mb=0.0, peak_ram_mb=0.0,
                total_io_read_mb=0.0, total_io_write_mb=0.0,
                duration_seconds=duration, sample_count=0,
                is_reliable=is_rel, reliability_score=score,
                measurement_quality=quality,
            )

        cpu_sys  = [s.cpu_percent         for s in all_snapshots]
        cpu_proc = [s.process_cpu_percent  for s in all_snapshots]
        ram_vals = [s.ram_used_mb          for s in all_snapshots]

        ref = self.post_snapshot or all_snapshots[-1]
        io_read_delta  = max(0.0, ref.io_read_mb  - self.baseline_io_read)
        io_write_delta = max(0.0, ref.io_write_mb - self.baseline_io_write)

        is_rel, score, quality = self._assess_quality(duration, len(all_snapshots))

        return SystemMetrics(
            avg_cpu_percent=sum(cpu_sys)  / len(cpu_sys),
            max_cpu_percent=max(cpu_sys),
            avg_process_cpu=sum(cpu_proc) / len(cpu_proc),
            max_process_cpu=max(cpu_proc),
            avg_ram_mb=sum(ram_vals) / len(ram_vals),
            peak_ram_mb=max(ram_vals),
            total_io_read_mb=io_read_delta,
            total_io_write_mb=io_write_delta,
            duration_seconds=duration,
            sample_count=len(all_snapshots),
            is_reliable=is_rel,
            reliability_score=score,
            measurement_quality=quality,
        )


# ============================================================================
# PROCESS ISOLATOR
# ============================================================================

class ProcessIsolator:
    """
    Isolates the benchmark process for fairer measurements.

    What it does:
    1. Raises process priority (nice / Windows priority class).
    2. Optionally pins the process to specific CPU cores (affinity).
    3. Runs GC and a short busy-wait to stabilise CPU frequency before
       the first measurement.
    4. Restores original settings after the benchmark.

    Fixes vs. the original version:
    - cpu_affinity() is not available on macOS → gracefully skipped with a note.
    - nice(-10) without sudo fails on Linux → caught, falls back to nice(-5),
      then logs a clear message if that also fails.
    - Windows HIGH_PRIORITY_CLASS handled separately from Unix nice values.
    - Pre-benchmark warmup added: GC collection + CPU frequency stabilisation.
    - isolate() returns IsolationState with per-step notes for diagnostics.
    """

    def __init__(self) -> None:
        self.process = psutil.Process(os.getpid())
        self._state = IsolationState()

    # ── Public API ────────────────────────────────────────────────────────────

    def isolate(
        self,
        cpu_cores: Optional[List[int]] = None,
        high_priority: bool = True,
        warmup: bool = True,
    ) -> IsolationState:
        """
        Isolate the process for benchmarking.

        Args:
            cpu_cores:     Cores to pin to (None = leave OS scheduler unchanged).
                           Example: [2, 3] pins to cores 2 and 3.
            high_priority: Raise process priority.
            warmup:        Run GC and a 50 ms busy-wait to wake the CPU from
                           a low-power state before the first measurement.

        Returns:
            IsolationState describing what was and was not applied.
        """
        self._state = IsolationState()
        self._save_original_state()

        if cpu_cores is not None:
            self._set_affinity(cpu_cores)
        else:
            self._state.isolation_notes.append(
                "CPU affinity: not set (cpu_cores=None, OS scheduler unchanged)"
            )

        if high_priority:
            self._set_high_priority()

        if warmup:
            self._warmup()

        self._state.isolated = True
        logger.info(
            "ProcessIsolator: isolation complete. %s",
            "; ".join(self._state.isolation_notes),
        )
        return self._state

    def restore(self) -> None:
        """Restore the original process priority and CPU affinity."""
        if not self._state.isolated:
            return

        restored = []

        if self._state.affinity is not None:
            try:
                self.process.cpu_affinity(self._state.affinity)
                restored.append("CPU affinity")
            except (psutil.Error, OSError, AttributeError) as e:
                logger.debug("Could not restore CPU affinity: %s", e)

        if self._state.nice is not None:
            try:
                self.process.nice(self._state.nice)
                restored.append("process priority")
            except (psutil.Error, OSError, PermissionError) as e:
                logger.debug("Could not restore process priority: %s", e)

        self._state.isolated = False
        logger.info(
            "ProcessIsolator: restored (%s).",
            ", ".join(restored) if restored else "nothing",
        )

    @property
    def state(self) -> IsolationState:
        return self._state

    # ── Diagnostics ───────────────────────────────────────────────────────────

    @staticmethod
    def get_available_cores() -> List[int]:
        """Return the logical CPU cores available to this process."""
        try:
            return list(psutil.Process(os.getpid()).cpu_affinity())
        except (AttributeError, psutil.Error, OSError):
            return list(range(psutil.cpu_count(logical=True) or 1))

    @staticmethod
    def recommend_benchmark_cores() -> List[int]:
        """
        Suggest cores suitable for benchmarking.

        Skips core 0 (typically busiest with OS interrupts) when more than
        two cores are available.  Example: [2, 3] on a quad-core CPU.
        """
        cores = ProcessIsolator.get_available_cores()
        if len(cores) <= 2:
            return cores
        return cores[1:]

    @staticmethod
    def is_high_priority_available() -> bool:
        """
        Check whether high priority can be set without elevated privileges.

        - Windows: HIGH_PRIORITY_CLASS does not require admin.
        - Linux / macOS: nice < 0 requires CAP_SYS_NICE or root.
        """
        if os.name == "nt":
            return True
        try:
            proc = psutil.Process(os.getpid())
            original = proc.nice()
            proc.nice(-1)
            proc.nice(original)
            return True
        except (PermissionError, psutil.AccessDenied):
            return False

    # ── Internal methods ──────────────────────────────────────────────────────

    def _save_original_state(self) -> None:
        try:
            self._state.affinity = self.process.cpu_affinity()
        except (AttributeError, psutil.Error, OSError):
            # cpu_affinity() is unavailable on macOS and some VMs
            self._state.affinity = None

        try:
            self._state.nice = self.process.nice()
        except (psutil.Error, OSError):
            self._state.nice = None

    def _set_affinity(self, cpu_cores: List[int]) -> None:
        try:
            available = self.get_available_cores()
            valid = [c for c in cpu_cores if c in available]

            if not valid:
                self._state.isolation_notes.append(
                    f"CPU affinity: requested cores {cpu_cores} are not available "
                    f"(available: {available}). Affinity not set."
                )
                return

            if len(valid) < len(cpu_cores):
                self._state.isolation_notes.append(
                    f"CPU affinity: some requested cores unavailable, "
                    f"using {valid}."
                )

            self.process.cpu_affinity(valid)
            self._state.isolation_notes.append(
                f"CPU affinity: pinned to cores {valid}"
            )

        except (AttributeError, psutil.Error, OSError) as e:
            self._state.isolation_notes.append(
                f"CPU affinity: cannot set ({e}). Platform: {platform.system()}."
            )

    def _set_high_priority(self) -> None:
        try:
            if os.name == "nt":
                # HIGH_PRIORITY_CLASS does not require admin on Windows.
                # REALTIME_PRIORITY_CLASS could destabilise the system.
                self.process.nice(psutil.HIGH_PRIORITY_CLASS)
                self._state.isolation_notes.append(
                    "Priority: HIGH_PRIORITY_CLASS (Windows)"
                )
            else:
                # Linux / macOS: -10 gives noticeably higher priority
                # without requiring root on most systems.
                self.process.nice(-10)
                self._state.isolation_notes.append("Priority: nice=-10 (Unix)")

        except (PermissionError, psutil.AccessDenied) as e:
            # No permission for -10; try -5 which usually works without sudo
            try:
                self.process.nice(-5)
                self._state.isolation_notes.append(
                    f"Priority: nice=-5 (fallback; no permission for -10: {e})"
                )
            except (PermissionError, psutil.AccessDenied, OSError):
                self._state.isolation_notes.append(
                    "Priority: could not be raised (insufficient privileges). "
                    "On Linux: run with sudo or grant CAP_SYS_NICE. "
                    "On Windows: this is unexpected — check UAC settings."
                )
        except (psutil.Error, OSError) as e:
            self._state.isolation_notes.append(
                f"Priority: unexpected error: {e}"
            )

    @staticmethod
    def _warmup() -> None:
        """
        Short warmup before the first measurement:
        1. Force a full GC cycle so the collector does not interrupt timing.
        2. 50 ms busy-wait to wake the CPU from a low-power / throttled state
           (prevents frequency-scaling artefacts on the very first run).
        """
        gc.collect()
        deadline = time.perf_counter() + 0.05
        _x = 0
        while time.perf_counter() < deadline:
            _x += 1


# ============================================================================
# SCENARIO ANALYSIS
# ============================================================================

@dataclass
class ScenarioMetrics:
    """Metrics for a best / worst case scenario."""
    scenario_type: str        # "best" or "worst"
    file_path: Path
    file_size_mb: float
    compression_ratio: float
    system_metrics: SystemMetrics

    @property
    def ram_per_mb(self) -> float:
        """Peak RAM per MB of input file."""
        if self.file_size_mb == 0:
            return 0.0
        return self.system_metrics.peak_ram_mb / self.file_size_mb

    @property
    def cpu_efficiency(self) -> float:
        """CPU seconds consumed per MB of input file."""
        if self.file_size_mb == 0:
            return 0.0
        cpu_time = (
            self.system_metrics.avg_process_cpu / 100.0
        ) * self.system_metrics.duration_seconds
        return cpu_time / self.file_size_mb


class ScenarioAnalyzer:
    """Identifies the best and worst result from a set of benchmark results."""

    SUPPORTED_METRICS = {
        "compression_ratio": ("Compression Ratio", True),   # (label, higher_is_better)
        "ram_usage":         ("RAM Usage",          False),
        "cpu_usage":         ("CPU Usage",          False),
        "io_total":          ("I/O Total",          False),
    }

    @staticmethod
    def identify_scenarios(
        results: List,
        metric: str = "compression_ratio",
    ) -> dict:
        if metric not in ScenarioAnalyzer.SUPPORTED_METRICS:
            logger.warning("ScenarioAnalyzer: unknown metric '%s'", metric)
            return {"best": None, "worst": None}

        valid = [
            r for r in results
            if hasattr(r, "system_metrics")
            and r.system_metrics is not None
            and r.metrics.success
        ]
        if len(valid) < 2:
            return {"best": None, "worst": None}

        _, higher_is_better = ScenarioAnalyzer.SUPPORTED_METRICS[metric]

        def sort_key(r):
            if metric == "compression_ratio":
                return r.metrics.compression_ratio
            if metric == "ram_usage":
                return r.system_metrics.peak_ram_mb
            if metric == "cpu_usage":
                return r.system_metrics.avg_process_cpu
            return r.system_metrics.io_total_mb

        ranked = sorted(valid, key=sort_key, reverse=higher_is_better)

        def to_scenario(r, stype: str) -> ScenarioMetrics:
            return ScenarioMetrics(
                scenario_type=stype,
                file_path=r.image_path,
                file_size_mb=r.metrics.original_size / (1024 * 1024),
                compression_ratio=r.metrics.compression_ratio,
                system_metrics=r.system_metrics,
            )

        return {
            "best":  to_scenario(ranked[0],  "best"),
            "worst": to_scenario(ranked[-1], "worst"),
        }

    @staticmethod
    def print_scenario_comparison(scenarios: dict, log_callback) -> None:
        best  = scenarios.get("best")
        worst = scenarios.get("worst")
        if not best or not worst:
            return

        log_callback(f"\n{'='*70}")
        log_callback("BEST vs WORST CASE ANALYSIS")
        log_callback(f"{'='*70}")

        for label, s in [("BEST CASE", best), ("WORST CASE", worst)]:
            log_callback(f"\n{label}:")
            log_callback(f"  File:              {s.file_path.name}")
            log_callback(f"  Input size:        {s.file_size_mb:.2f} MB")
            log_callback(f"  Compression ratio: {s.compression_ratio:.2f}x")
            log_callback(f"  Peak RAM:          {s.system_metrics.peak_ram_mb:.1f} MB")
            log_callback(f"  Avg CPU:           {s.system_metrics.avg_process_cpu:.1f}%")
            log_callback(f"  Total I/O:         {s.system_metrics.io_total_mb:.2f} MB")

        if best.system_metrics.peak_ram_mb > 0:
            ram_ratio = worst.system_metrics.peak_ram_mb / best.system_metrics.peak_ram_mb
            log_callback(f"\n  RAM variation:  {ram_ratio:.2f}x")

        if best.system_metrics.avg_process_cpu > 0:
            cpu_ratio = (
                worst.system_metrics.avg_process_cpu
                / best.system_metrics.avg_process_cpu
            )
            log_callback(f"  CPU variation:  {cpu_ratio:.2f}x")