"""
System Resource Monitoring Module
Monitors CPU, RAM, and I/O metrics during compression benchmarks.

Classes:
- SystemMonitor   – adaptive background-thread sampling of CPU / RAM / I/O
- ProcessIsolator – process isolation for fair benchmarking (priority, CPU affinity)
- ScenarioAnalyzer / ScenarioMetrics – best / worst case analysis

IMPORTANT NOTES:
- CPU values can exceed 100% on multi-core systems (e.g., 200% = 2 cores fully utilized)
- RAM measurements include the entire Python process baseline (~300MB for GUI apps)
- I/O measurements may be affected by OS caching (reads can appear lower than file size)
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

    # CPU (can exceed 100% on multi-core systems)
    avg_cpu_percent: float    # average total system CPU (%)
    max_cpu_percent: float    # peak total system CPU (%)
    avg_process_cpu: float    # average process CPU, cumulative per-core (%)
    max_process_cpu: float    # peak process CPU (can exceed 100% on multi-core)

    # Memory (includes Python baseline + GUI overhead)
    avg_ram_mb: float         # average RSS (MB)
    peak_ram_mb: float        # maximum RSS observed during the run (MB)

    # I/O — delta from start to end (may be affected by caching)
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
    
    @property
    def cpu_percent_normalized(self) -> float:
        """Returns CPU usage normalized to 0-100% (dividing by core count)."""
        core_count = psutil.cpu_count(logical=True) or 1
        return min(100.0, self.avg_process_cpu / core_count)
    
    @property
    def peak_cpu_percent_normalized(self) -> float:
        """Returns peak CPU usage normalized to 0-100% (dividing by core count)."""
        core_count = psutil.cpu_count(logical=True) or 1
        return min(100.0, self.max_process_cpu / core_count)


# ============================================================================
# SYSTEM MONITOR
# ============================================================================

class SystemMonitor:
    """
    Measures CPU, RAM, and I/O in a background thread during compression.

    Key fixes vs. the original version:
    - cpu_percent(interval=None) called in __init__ so the first sample is
      not always 0.0 (psutil needs one warm-up call to initialize its counter).
    - Duplicate max_ram_mb / peak_ram_mb fields collapsed into peak_ram_mb only.
    - Bare except replaced with explicit (psutil.Error, OSError) + debug logging.
    - time.perf_counter() used instead of time.time() for short-duration accuracy.
    
    IMPORTANT CPU MEASUREMENT NOTE:
    - psutil.cpu_percent() returns cumulative CPU across ALL cores
    - On a 4-core system, one fully loaded single-threaded process shows ~100%
    - A multi-threaded process using all cores can show 400%
    - When we sum parent + child processes, values can exceed 100%
    - This is CORRECT behavior, not a bug!
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

        # Initialize psutil's internal CPU counter so the first real sample
        # returns a meaningful value instead of 0.0.
        # Also warm up cpu_percent() for all current children (best-effort).
        self.process.cpu_percent(interval=None)
        psutil.cpu_percent(interval=None)
        for child in self.process.children(recursive=True):
            try:
                child.cpu_percent(interval=None)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

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

        # Note: We don't set baseline_io here anymore because subprocesses
        # don't exist yet. Instead, we'll use the first snapshot's I/O values
        # as baseline in _calculate_metrics()
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
        cpu_system = psutil.cpu_percent(interval=0)

        # Sum CPU, RAM, and I/O of this process + all child processes.
        # Subprocess-based compressors (cwebp, optipng, ...) run as separate PIDs
        # and are invisible to self.process metrics alone — this fixes the
        # "0% CPU" and "0 I/O" readings for WebP-Lossless and OptiPNG.
        # 
        # IMPORTANT: CPU values can exceed 100% on multi-core systems!
        # E.g., on 4-core CPU: 143% means ~1.4 cores utilized
        
        # Collect all processes (parent + children)
        all_processes = [self.process]
        children = []
        try:
            children = self.process.children(recursive=True)
            all_processes.extend(children)
        except (psutil.Error, OSError):
            pass
        
        # Debug log if children found
        if children and logger.isEnabledFor(logging.DEBUG):
            child_pids = [str(c.pid) for c in children]
            logger.debug(
                "Found %d child process(es): PIDs %s",
                len(children), ", ".join(child_pids)
            )
        
        # Aggregate CPU
        cpu_process = 0.0
        for proc in all_processes:
            try:
                cpu_process += proc.cpu_percent(interval=0)
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.Error):
                pass
        
        # Aggregate RAM (sum RSS of all processes)
        ram_used_mb = 0.0
        for proc in all_processes:
            try:
                mem_info = proc.memory_info()
                ram_used_mb += mem_info.rss / (1024 * 1024)
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.Error):
                pass
        
        # Aggregate I/O (sum read/write bytes of all processes)
        io_read_mb = 0.0
        io_write_mb = 0.0
        for proc in all_processes:
            try:
                io = proc.io_counters()
                io_read_mb += io.read_bytes / (1024 * 1024)
                io_write_mb += io.write_bytes / (1024 * 1024)
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.Error, AttributeError):
                # AttributeError: io_counters() not available on some platforms
                pass
        
        vmem = psutil.virtual_memory()

        return SystemSnapshot(
            timestamp=time.perf_counter(),
            cpu_percent=cpu_system,
            process_cpu_percent=cpu_process,
            ram_used_mb=ram_used_mb,
            ram_available_mb=vmem.available / (1024 * 1024),
            io_read_mb=io_read_mb,
            io_write_mb=io_write_mb,
        )

    def _set_adaptive_sampling(self, file_size_bytes: int) -> None:
        """
        Adjusts sampling interval based on file size:
        - Smaller files need faster sampling (down to 1 ms) to capture brief operations.
        - Larger files can use slower sampling (up to 100 ms) to reduce overhead.
        """
        kb = file_size_bytes / 1024
        if kb < 10:
            self.sampling_interval = self.ULTRA_FAST_INTERVAL
        elif kb < 100:
            self.sampling_interval = self.FAST_INTERVAL
        elif kb < 1024:
            self.sampling_interval = self.NORMAL_INTERVAL
        else:
            self.sampling_interval = self.SLOW_INTERVAL

        logger.debug(
            "Adaptive sampling: file_size=%.1f KB → interval=%.3f s",
            kb, self.sampling_interval
        )

    def _calculate_metrics(self) -> SystemMetrics:
        """
        Aggregates snapshots into summary statistics.
        
        I/O delta is calculated from first to last snapshot to properly handle
        subprocesses that are created during monitoring (e.g., cwebp, optipng).
        
        Note: Returns baseline Python process RAM (~200-300MB), not just compression overhead.
        """
        duration = self.end_time - self.start_time

        all_snaps = []
        if self.pre_snapshot:
            all_snaps.append(self.pre_snapshot)
        all_snaps.extend(self.snapshots)
        if self.post_snapshot:
            all_snaps.append(self.post_snapshot)

        if not all_snaps:
            return self._empty_metrics(duration)

        cpu_vals     = [s.cpu_percent for s in all_snaps]
        cpu_proc_vals = [s.process_cpu_percent for s in all_snaps]
        ram_vals     = [s.ram_used_mb for s in all_snaps]

        # I/O delta: Use first snapshot as baseline, last snapshot as end
        # This properly handles subprocesses created during monitoring
        io_read_delta  = 0.0
        io_write_delta = 0.0
        if len(all_snaps) >= 2:
            first_snap = all_snaps[0]
            last_snap = all_snaps[-1]
            io_read_delta  = max(0.0, last_snap.io_read_mb  - first_snap.io_read_mb)
            io_write_delta = max(0.0, last_snap.io_write_mb - first_snap.io_write_mb)
        elif len(all_snaps) == 1:
            # Only one snapshot - use it as delta (rare case)
            io_read_delta  = all_snaps[0].io_read_mb
            io_write_delta = all_snaps[0].io_write_mb

        sample_count = len(all_snaps)
        is_reliable = (
            sample_count >= self.MIN_SAMPLES_RELIABLE
            and duration >= self.MIN_DURATION_RELIABLE
        )
        reliability_score = min(1.0, sample_count / self.MIN_SAMPLES_RELIABLE)

        if sample_count >= self.MIN_SAMPLES_RELIABLE:
            quality = "good"
        elif sample_count >= self.MIN_SAMPLES_FAIR:
            quality = "fair"
        else:
            quality = "poor"

        return SystemMetrics(
            avg_cpu_percent=sum(cpu_vals) / len(cpu_vals),
            max_cpu_percent=max(cpu_vals),
            avg_process_cpu=sum(cpu_proc_vals) / len(cpu_proc_vals),
            max_process_cpu=max(cpu_proc_vals),
            avg_ram_mb=sum(ram_vals) / len(ram_vals),
            peak_ram_mb=max(ram_vals),
            total_io_read_mb=io_read_delta,
            total_io_write_mb=io_write_delta,
            duration_seconds=duration,
            sample_count=sample_count,
            is_reliable=is_reliable,
            reliability_score=reliability_score,
            measurement_quality=quality,
        )

    @staticmethod
    def _empty_metrics(duration: float) -> SystemMetrics:
        """Returns zero metrics when no samples were collected."""
        return SystemMetrics(
            avg_cpu_percent=0.0,
            max_cpu_percent=0.0,
            avg_process_cpu=0.0,
            max_process_cpu=0.0,
            avg_ram_mb=0.0,
            peak_ram_mb=0.0,
            total_io_read_mb=0.0,
            total_io_write_mb=0.0,
            duration_seconds=duration,
            sample_count=0,
            is_reliable=False,
            reliability_score=0.0,
            measurement_quality="none",
        )


# ============================================================================
# PROCESS ISOLATION
# ============================================================================

class ProcessIsolator:
    """
    Isolates the benchmark process for more consistent measurements:
    - Sets high process priority (Windows: HIGH_PRIORITY_CLASS, Unix: nice=-10)
    - Pins to specific CPU cores (optional, platform-dependent)
    - Performs warmup to stabilize CPU frequency scaling
    
    WARNING: Requires elevated privileges on some platforms.
    On Linux, may need CAP_SYS_NICE or sudo for priority changes.
    """

    def __init__(self, enable: bool = True) -> None:
        self.enable = enable
        self.process = psutil.Process(os.getpid())
        self._state = IsolationState()

    def isolate(self, cpu_cores: Optional[List[int]] = None) -> IsolationState:
        """
        Isolate process for benchmarking.
        
        Args:
            cpu_cores: List of CPU cores to pin to (e.g., [0, 1]). 
                      None = use all cores (no affinity).
        
        Returns:
            IsolationState with isolation status and notes.
        """
        if not self.enable:
            self._state.isolated = False
            self._state.isolation_notes = ["Process isolation disabled"]
            return self._state

        self._save_original_state()

        if cpu_cores:
            self._set_affinity(cpu_cores)

        self._set_high_priority()
        self._warmup()

        self._state.isolated = True
        return self._state

    def restore(self) -> bool:
        """
        Restore original process state (priority, affinity).
        
        Returns:
            True if restoration was successful, False otherwise.
        """
        if not self._state.isolated:
            return True

        success = True

        # Restore affinity
        if self._state.affinity is not None:
            try:
                self.process.cpu_affinity(self._state.affinity)
            except (AttributeError, psutil.Error, OSError) as e:
                logger.debug("Could not restore CPU affinity: %s", e)
                success = False

        # Restore priority
        if self._state.nice is not None:
            try:
                self.process.nice(self._state.nice)
            except (psutil.Error, OSError) as e:
                logger.debug("Could not restore priority: %s", e)
                success = False

        self._state.isolated = False
        return success

    @staticmethod
    def get_available_cores() -> List[int]:
        """Returns list of available CPU core IDs."""
        try:
            # Try to get affinity if supported
            proc = psutil.Process(os.getpid())
            return proc.cpu_affinity()
        except (AttributeError, psutil.Error):
            # Fallback: all logical cores
            count = psutil.cpu_count(logical=True)
            return list(range(count)) if count else [0]

    @staticmethod
    def set_priority_temporary(priority_class) -> bool:
        """
        Temporarily set process priority (for testing).
        
        Args:
            priority_class: psutil priority constant (e.g., psutil.HIGH_PRIORITY_CLASS)
        
        Returns:
            True if successful, False otherwise.
        """
        try:
            proc = psutil.Process(os.getpid())
            original = proc.nice()
            proc.nice(priority_class)
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
                # REALTIME_PRIORITY_CLASS could destabilize the system.
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
           (prevents frequency-scaling artifacts on the very first run).
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