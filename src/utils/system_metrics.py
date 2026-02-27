"""
System Resource Monitoring Module
utils/system_metrics.py

Measures CPU, RAM, and I/O metrics in a background thread while a compressor runs.

Classes:
  SystemMonitor    : Adaptive background-thread sampler for CPU / RAM / I/O.
  ProcessIsolator  : Raises process priority and optionally pins CPU cores for
                     more reproducible benchmark measurements.
  ScenarioAnalyzer : Identifies best / worst case results from a result set.
  ScenarioMetrics  : Per-scenario summary data (compression ratio, RAM, CPU, I/O).

Important measurement notes:
  - CPU values can exceed 100 % on multi-core systems (e.g. 200 % = 2 cores fully busy).
    Use cpu_percent_normalized / peak_cpu_percent_normalized for 0-100 % display.
  - RAM measurements include the entire Python process baseline (~300 MB for GUI apps);
    they are NOT a measure of compression-algorithm memory overhead alone.
  - I/O readings may be lower than file size due to OS read-ahead caching.
  - CLI-based compressors (cwebp.exe, optipng.exe, …) spawn short-lived child processes.
    Their I/O is tracked via a persistent PID cache (_seen_pids) so the data is not
    lost when the process exits before the next sampling tick.
"""

import gc
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import psutil

logger = logging.getLogger(__name__)

# Number of logical CPU cores — used for normalisation throughout this module.
_LOGICAL_CORES: int = psutil.cpu_count(logical=True) or 1


# ============================================================================
# Data structures
# ============================================================================

@dataclass
class SystemSnapshot:
    """Single point-in-time resource sample collected by the monitor thread."""
    timestamp:           float
    cpu_percent:         float   # total system-wide CPU usage (%)
    process_cpu_percent: float   # this process + children, cumulative across cores (%)
    ram_used_mb:         float   # RSS of this process + children (MB)
    ram_available_mb:    float   # free system RAM (MB)
    io_read_mb:          float   # cumulative process read bytes (MB)
    io_write_mb:         float   # cumulative process write bytes (MB)


@dataclass
class IsolationState:
    """Process state saved before isolation so it can be restored afterwards."""
    affinity:         Optional[List[int]] = None
    nice:             Optional[int]       = None
    isolated:         bool                = False
    isolation_notes:  List[str]           = field(default_factory=list)


@dataclass
class SystemMetrics:
    """
    Aggregated resource statistics for one complete benchmark run.

    Raw CPU fields (avg_process_cpu, max_process_cpu) are cumulative across all
    logical cores — they can exceed 100 % on multi-core systems (200 % means two
    cores fully utilised).  Use cpu_percent_normalized / peak_cpu_percent_normalized
    for a 0–100 % display value that is comparable across machines.

    I/O fields include data from short-lived child processes (CLI compressors) thanks
    to the persistent PID cache in SystemMonitor — they are not lost when a child
    process exits between two sampling ticks.
    """

    # CPU — raw cumulative values (may exceed 100 % on multi-core systems)
    avg_cpu_percent: float   # average total system-wide CPU (%)
    max_cpu_percent: float   # peak total system-wide CPU (%)
    avg_process_cpu: float   # average process+children CPU, cumulative (%)
    max_process_cpu: float   # peak process+children CPU, cumulative (%)

    # Memory  (includes Python + GUI baseline — not compression overhead alone)
    avg_ram_mb:      float   # average RSS (MB)
    peak_ram_mb:     float   # maximum RSS observed during the run (MB)
    ram_baseline_mb: float   # RSS at start() — the Python/GUI overhead floor

    @property
    def net_peak_ram_mb(self) -> float:
        """
        Peak RAM above the baseline measured at start().

        Use this value when comparing runs across sessions — it removes the
        Python / GUI memory floor (~200-450 MB) that grows with each run
        because previous results stay in memory.
        """
        return max(0.0, self.peak_ram_mb - self.ram_baseline_mb)

    @property
    def net_avg_ram_mb(self) -> float:
        """Average RAM above the baseline measured at start()."""
        return max(0.0, self.avg_ram_mb - self.ram_baseline_mb)

    # I/O  (delta from start to end, including exited child processes)
    total_io_read_mb:  float
    total_io_write_mb: float

    # Timing and sample diagnostics
    duration_seconds: float
    sample_count:     int

    # Core count recorded at measurement time (used for normalisation)
    logical_core_count: int = field(default_factory=lambda: _LOGICAL_CORES)

    # Measurement quality flags
    is_reliable:         bool  = True
    reliability_score:   float = 1.0
    measurement_quality: str   = "good"

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
        """
        Average process CPU normalised to 0–100 %.

        Divides the cumulative value by the number of logical cores recorded
        at measurement time so the result is comparable across machines with
        different core counts.
        """
        return min(100.0, self.avg_process_cpu / self.logical_core_count)

    @property
    def peak_cpu_percent_normalized(self) -> float:
        """Peak process CPU normalised to 0–100 %."""
        return min(100.0, self.max_process_cpu / self.logical_core_count)


# ============================================================================
# System monitor
# ============================================================================

class SystemMonitor:
    """
    Measures CPU, RAM, and I/O in a background thread during compression.

    Key design decisions:
      - cpu_percent(interval=None) is called in __init__ so the first real sample
        is not always 0.0 (psutil requires one warm-up call to initialise its counter).
      - time.perf_counter() is used instead of time.time() for sub-millisecond accuracy.
      - The sampling interval adapts to the input file size: smaller files trigger
        faster sampling so brief operations are captured.
      - Child processes (e.g. cwebp.exe, optipng.exe) are tracked via a persistent
        PID→I/O cache (_seen_pids).  When a child process exits between two sampling
        ticks its last-seen I/O counters are retained, so CLI-based compressors are
        no longer under-reported.

    CPU interpretation:
      Reported values are *cumulative across all logical cores*.  On a 4-core machine
      a fully-loaded single-threaded process shows ~100 %; a 4-thread workload can show
      ~400 %.  Use SystemMetrics.cpu_percent_normalized for a 0–100 % display value.
    """

    # Sampling interval thresholds (adaptive mode)
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
        adaptive:          bool  = True,
        force_pre_post:    bool  = True,
    ) -> None:
        self.base_sampling_interval = sampling_interval
        self.sampling_interval      = sampling_interval
        self.adaptive               = adaptive
        self.force_pre_post         = force_pre_post

        self.process = psutil.Process(os.getpid())

        # Warm up psutil's internal CPU counter so the first real sample is not 0.
        self.process.cpu_percent(interval=None)
        psutil.cpu_percent(interval=None)
        for child in self.process.children(recursive=True):
            try:
                child.cpu_percent(interval=None)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        self.snapshots:        List[SystemSnapshot]       = []
        self.is_monitoring:    bool                       = False
        self.monitor_thread:   Optional[threading.Thread] = None

        self.start_time: float = 0.0
        self.end_time:   float = 0.0

        self.pre_snapshot:  Optional[SystemSnapshot] = None
        self.post_snapshot: Optional[SystemSnapshot] = None

        # Persistent PID → (io_read_mb, io_write_mb) cache.
        #
        # Purpose: CLI compressors (cwebp.exe, optipng.exe, …) are short-lived child
        # processes.  They can finish and their PID can disappear between two sampling
        # ticks.  If we only read io_counters() for *live* processes we silently lose
        # all I/O from processes that exited in between.
        #
        # Solution: every time we successfully read io_counters() for a PID we store
        # the latest cumulative value here.  When the PID is gone on the next tick we
        # keep the last-known value.  _calculate_metrics() then sums the *deltas*
        # (end − start) for every PID seen during the run, regardless of whether the
        # process is still alive at stop() time.
        #
        # Thread safety: only the monitor thread writes; _calculate_metrics() reads
        # only after the thread has been joined, so no lock is needed.
        self._seen_pids: Dict[int, Tuple[float, float]] = {}  # pid → (read_mb, write_mb)
        self._initial_pids: Dict[int, Tuple[float, float]] = {}  # snapshot at start()

    # ---- Public API ----

    def start(self, file_size_bytes: Optional[int] = None) -> None:
        """
        Begin monitoring.  Safe to call when already running (no-op).

        Args:
            file_size_bytes: Source file size used to choose the adaptive interval.
        """
        if self.is_monitoring:
            return

        if file_size_bytes is not None and self.adaptive:
            self._set_adaptive_sampling(file_size_bytes)

        self.snapshots     = []
        self._seen_pids    = {}
        self._initial_pids = {}
        self.is_monitoring = True
        self.start_time    = time.perf_counter()

        if self.force_pre_post:
            self.pre_snapshot = self._take_snapshot()

        # RAM baseline: the RSS of the entire Python process at the moment
        # monitoring starts.  Subtracting this from peak/avg gives a "net RAM"
        # value that is comparable across runs in the same session regardless of
        # how much memory previous benchmark results occupy.
        self._ram_baseline_mb: float = (
            self.pre_snapshot.ram_used_mb if self.pre_snapshot else 0.0
        )

        # Record the I/O baseline for all currently running PIDs (parent + any
        # pre-existing children).  This prevents their background I/O (DLL loads,
        # Python imports, etc.) from contaminating the delta we attribute to
        # compression.
        self._initial_pids = dict(self._seen_pids)

        self.monitor_thread = threading.Thread(
            target=self._monitor_loop, daemon=True, name="SystemMonitor"
        )
        self.monitor_thread.start()

    def stop(self) -> SystemMetrics:
        """Stop monitoring and return aggregated metrics."""
        self.is_monitoring = False
        self.end_time      = time.perf_counter()

        if self.force_pre_post:
            self.post_snapshot = self._take_snapshot()

        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)

        return self._calculate_metrics()

    # ---- Internal ----

    def _monitor_loop(self) -> None:
        while self.is_monitoring:
            try:
                self.snapshots.append(self._take_snapshot())
            except (psutil.Error, OSError) as exc:
                logger.debug("SystemMonitor snapshot error: %s", exc)
            time.sleep(self.sampling_interval)

    def _take_snapshot(self) -> SystemSnapshot:
        """
        Collect one resource sample.

        CPU, RAM, and I/O are summed across the parent process and all *currently
        live* child processes.  I/O for processes that have already exited is
        preserved in _seen_pids and included in the final delta calculation.

        New child processes (compressor CLI tools) receive an immediate
        cpu_percent(interval=None) warm-up call on their first appearance so
        the *next* tick returns a meaningful value instead of 0.0.
        """
        cpu_system = psutil.cpu_percent(interval=0)

        # Build a fresh list of live processes each tick because compressor child
        # processes are created *after* start() is called.
        all_processes = [self.process]
        try:
            all_processes.extend(self.process.children(recursive=True))
        except (psutil.Error, OSError):
            pass

        if logger.isEnabledFor(logging.DEBUG) and len(all_processes) > 1:
            logger.debug(
                "Snapshot: %d process(es) — PIDs %s",
                len(all_processes),
                ", ".join(str(p.pid) for p in all_processes),
            )

        # ---- CPU ----
        # Warm up the cpu_percent counter for any PID we haven't seen before.
        # Without this, the first cpu_percent(interval=0) call for a new PID
        # always returns 0.0 because psutil has no prior measurement to diff against.
        # The warm-up call itself returns 0.0 and is discarded; the *next* tick
        # will return the real accumulated value since this warm-up call.
        cpu_process = 0.0
        for proc in all_processes:
            try:
                if proc.pid not in self._seen_pids:
                    # First time we see this PID — initialise psutil's counter.
                    proc.cpu_percent(interval=None)
                    # Skip this tick's CPU contribution; it would be 0.0 anyway.
                else:
                    cpu_process += proc.cpu_percent(interval=0)
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.Error):
                pass

        # ---- RAM ----
        ram_used_mb = 0.0
        for proc in all_processes:
            try:
                ram_used_mb += proc.memory_info().rss / (1024 * 1024)
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.Error):
                pass

        # ---- I/O (live processes only; _seen_pids caches exited ones) ----
        # For each live process we read its cumulative I/O and update _seen_pids.
        # The snapshot carries the *current total* across all live processes.
        # Exited-process I/O is not added to the snapshot value — it is added
        # separately in _calculate_metrics() via the _seen_pids delta.
        io_read_live = io_write_live = 0.0
        for proc in all_processes:
            try:
                io = proc.io_counters()
                r  = io.read_bytes  / (1024 * 1024)
                w  = io.write_bytes / (1024 * 1024)
                self._seen_pids[proc.pid] = (r, w)
                io_read_live  += r
                io_write_live += w
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.Error, AttributeError):
                # Process exited between the children() call and io_counters() — the
                # last value is already stored in _seen_pids from a previous tick.
                pass

        vmem = psutil.virtual_memory()

        return SystemSnapshot(
            timestamp           = time.perf_counter(),
            cpu_percent         = cpu_system,
            process_cpu_percent = cpu_process,
            ram_used_mb         = ram_used_mb,
            ram_available_mb    = vmem.available / (1024 * 1024),
            io_read_mb          = io_read_live,
            io_write_mb         = io_write_live,
        )

    def _set_adaptive_sampling(self, file_size_bytes: int) -> None:
        """
        Choose a sampling interval proportional to the input file size.
        Smaller files need faster sampling to capture short-lived operations.

        The interval is clamped to FAST_INTERVAL (10 ms) as a lower bound even
        for large files — this ensures at least ~10 samples for operations that
        complete in ~100 ms (e.g. Pillow-TIFF, LibPNG decompression).
        """
        kb = file_size_bytes / 1024
        if kb < 10:
            interval = self.ULTRA_FAST_INTERVAL
        elif kb < 100:
            interval = self.FAST_INTERVAL
        elif kb < 1024:
            interval = self.NORMAL_INTERVAL
        else:
            # For large files use NORMAL_INTERVAL, not SLOW_INTERVAL.
            # SLOW_INTERVAL (100 ms) produces too few samples for compressors
            # that finish in under 300 ms (e.g. LibPNG, Pillow on a ~1 MB file).
            interval = self.NORMAL_INTERVAL

        self.sampling_interval = interval
        logger.debug(
            "Adaptive sampling: file_size=%.1f KB → interval=%.3f s",
            kb, self.sampling_interval,
        )

    def _calculate_metrics(self) -> SystemMetrics:
        """
        Aggregate all snapshots into a single SystemMetrics summary.

        I/O strategy
        ------------
        We compute I/O as the sum of per-PID deltas (end − start) using _seen_pids
        and _initial_pids.  This correctly captures:

          1. The parent Python process.
          2. Long-lived child processes (still alive at stop()).
          3. Short-lived child processes (exited before stop()) — their last-seen
             cumulative value in _seen_pids minus their baseline in _initial_pids
             gives the net I/O they performed during the benchmark run.

        The snapshot-based first-to-last delta (old approach) is kept as a fallback
        for environments where io_counters() is unavailable (some VMs / containers).
        """
        duration = self.end_time - self.start_time

        all_snaps: List[SystemSnapshot] = []
        if self.pre_snapshot:
            all_snaps.append(self.pre_snapshot)
        all_snaps.extend(self.snapshots)
        if self.post_snapshot:
            all_snaps.append(self.post_snapshot)

        if not all_snaps:
            return self._empty_metrics(duration)

        cpu_vals      = [s.cpu_percent        for s in all_snaps]
        cpu_proc_vals = [s.process_cpu_percent for s in all_snaps]
        ram_vals      = [s.ram_used_mb         for s in all_snaps]

        # ---- I/O delta via persistent PID cache ----
        # Sum (final − initial) for every PID seen during the run.
        # PIDs that existed before start() use their _initial_pids baseline;
        # PIDs spawned during the run (compressor children) have no baseline
        # entry, so their full cumulative value is attributed to the run.
        io_read_delta = io_write_delta = 0.0
        if self._seen_pids:
            for pid, (end_r, end_w) in self._seen_pids.items():
                start_r, start_w = self._initial_pids.get(pid, (0.0, 0.0))
                io_read_delta  += max(0.0, end_r - start_r)
                io_write_delta += max(0.0, end_w - start_w)
            logger.debug(
                "I/O delta via PID cache: read=%.3f MB write=%.3f MB (%d PIDs)",
                io_read_delta, io_write_delta, len(self._seen_pids),
            )
        else:
            # Fallback: snapshot-based first-to-last delta.
            if len(all_snaps) >= 2:
                io_read_delta  = max(0.0, all_snaps[-1].io_read_mb  - all_snaps[0].io_read_mb)
                io_write_delta = max(0.0, all_snaps[-1].io_write_mb - all_snaps[0].io_write_mb)
            elif all_snaps:
                io_read_delta  = all_snaps[0].io_read_mb
                io_write_delta = all_snaps[0].io_write_mb

        sample_count = len(all_snaps)
        is_reliable  = (
            sample_count >= self.MIN_SAMPLES_RELIABLE
            and duration  >= self.MIN_DURATION_RELIABLE
        )
        reliability_score = min(1.0, sample_count / self.MIN_SAMPLES_RELIABLE)

        if sample_count >= self.MIN_SAMPLES_RELIABLE:
            quality = "good"
        elif sample_count >= self.MIN_SAMPLES_FAIR:
            quality = "fair"
        else:
            quality = "poor"

        n = len(cpu_vals)
        return SystemMetrics(
            avg_cpu_percent     = sum(cpu_vals)      / n,
            max_cpu_percent     = max(cpu_vals),
            avg_process_cpu     = sum(cpu_proc_vals) / n,
            max_process_cpu     = max(cpu_proc_vals),
            avg_ram_mb          = sum(ram_vals)      / n,
            peak_ram_mb         = max(ram_vals),
            ram_baseline_mb     = getattr(self, "_ram_baseline_mb", 0.0),
            total_io_read_mb    = io_read_delta,
            total_io_write_mb   = io_write_delta,
            duration_seconds    = duration,
            sample_count        = sample_count,
            logical_core_count  = _LOGICAL_CORES,
            is_reliable         = is_reliable,
            reliability_score   = reliability_score,
            measurement_quality = quality,
        )

    @staticmethod
    def _empty_metrics(duration: float) -> SystemMetrics:
        """Return an all-zero SystemMetrics when no samples were collected."""
        return SystemMetrics(
            avg_cpu_percent     = 0.0,
            max_cpu_percent     = 0.0,
            avg_process_cpu     = 0.0,
            max_process_cpu     = 0.0,
            avg_ram_mb          = 0.0,
            peak_ram_mb         = 0.0,
            ram_baseline_mb     = 0.0,
            total_io_read_mb    = 0.0,
            total_io_write_mb   = 0.0,
            duration_seconds    = duration,
            sample_count        = 0,
            logical_core_count  = _LOGICAL_CORES,
            is_reliable         = False,
            reliability_score   = 0.0,
            measurement_quality = "none",
        )


# ============================================================================
# Process isolation  (Windows only)
# ============================================================================

class ProcessIsolator:
    """
    Raises process priority and optionally pins CPU cores to reduce measurement noise.

    Windows implementation:
      - Priority: HIGH_PRIORITY_CLASS via psutil.  Does NOT require administrator.
        REALTIME_PRIORITY_CLASS is intentionally avoided as it can destabilise the system.
      - CPU affinity: set via psutil.Process.cpu_affinity().  No admin required.

    Usage:
        isolator = ProcessIsolator()
        isolator.isolate(high_priority=True)
        # ... run benchmark ...
        isolator.restore()
    """

    def __init__(self, enable: bool = True) -> None:
        self.enable  = enable
        self.process = psutil.Process(os.getpid())
        self._state  = IsolationState()

    def isolate(
        self,
        cpu_cores:     Optional[List[int]] = None,
        high_priority: bool                = True,
    ) -> IsolationState:
        """
        Apply process isolation settings.

        Args:
            cpu_cores:     Core IDs to pin to (e.g. [0]).  None = no affinity change.
            high_priority: If True, raise to HIGH_PRIORITY_CLASS.

        Returns:
            IsolationState describing what was changed and any notes.
        """
        if not self.enable:
            self._state.isolated = False
            self._state.isolation_notes = ["Process isolation disabled."]
            return self._state

        self._save_state()

        if cpu_cores:
            self._set_affinity(cpu_cores)

        if high_priority:
            self._set_high_priority()

        self._warmup()

        self._state.isolated = True
        return self._state

    def restore(self) -> bool:
        """
        Restore the process to its pre-isolation priority and affinity.

        Returns:
            True if both attributes were restored successfully.
        """
        if not self._state.isolated:
            return True

        success = True

        if self._state.affinity is not None:
            try:
                self.process.cpu_affinity(self._state.affinity)
            except (AttributeError, psutil.Error, OSError) as exc:
                logger.debug("Could not restore CPU affinity: %s", exc)
                success = False

        if self._state.nice is not None:
            try:
                self.process.nice(self._state.nice)
            except (psutil.Error, OSError) as exc:
                logger.debug("Could not restore process priority: %s", exc)
                success = False

        self._state.isolated = False
        return success

    @staticmethod
    def get_available_cores() -> List[int]:
        """Return the list of CPU core IDs visible to this process."""
        try:
            return psutil.Process(os.getpid()).cpu_affinity()
        except (AttributeError, psutil.Error):
            count = psutil.cpu_count(logical=True)
            return list(range(count)) if count else [0]

    # ---- Private helpers ----

    def _save_state(self) -> None:
        """Record current affinity and nice level for later restoration."""
        try:
            self._state.affinity = self.process.cpu_affinity()
        except (AttributeError, psutil.Error, OSError):
            self._state.affinity = None

        try:
            self._state.nice = self.process.nice()
        except (psutil.Error, OSError):
            self._state.nice = None

    def _set_affinity(self, cpu_cores: List[int]) -> None:
        """Pin the process to the requested CPU cores (skipping unavailable ones)."""
        try:
            available = self.get_available_cores()
            valid     = [c for c in cpu_cores if c in available]

            if not valid:
                self._state.isolation_notes.append(
                    f"CPU affinity: requested cores {cpu_cores} are not available "
                    f"(available: {available}). Affinity not changed."
                )
                return

            if len(valid) < len(cpu_cores):
                self._state.isolation_notes.append(
                    f"CPU affinity: some cores unavailable; using {valid}."
                )

            self.process.cpu_affinity(valid)
            self._state.isolation_notes.append(f"CPU affinity: pinned to cores {valid}.")

        except (AttributeError, psutil.Error, OSError) as exc:
            self._state.isolation_notes.append(f"CPU affinity: cannot set ({exc}).")

    def _set_high_priority(self) -> None:
        """
        Raise process priority to HIGH_PRIORITY_CLASS on Windows.
        Falls back gracefully if the call fails.
        """
        try:
            # HIGH_PRIORITY_CLASS does not require administrator on Windows.
            # REALTIME_PRIORITY_CLASS is deliberately not used (system destabilisation risk).
            self.process.nice(psutil.HIGH_PRIORITY_CLASS)
            self._state.isolation_notes.append("Priority: HIGH_PRIORITY_CLASS.")
        except (PermissionError, psutil.AccessDenied, psutil.Error, OSError) as exc:
            self._state.isolation_notes.append(
                f"Priority: could not raise to HIGH_PRIORITY_CLASS ({exc}). "
                "Check UAC settings or run as administrator."
            )

    @staticmethod
    def _warmup() -> None:
        """
        Short warm-up before the first measurement:
          1. Force a GC cycle so the collector does not interrupt timing.
          2. 50 ms busy-wait to lift the CPU out of a low-power / throttled state,
             preventing frequency-scaling artefacts on the very first run.
        """
        gc.collect()
        deadline = time.perf_counter() + 0.05
        _x = 0
        while time.perf_counter() < deadline:
            _x += 1


# ============================================================================
# Scenario analysis
# ============================================================================

@dataclass
class ScenarioMetrics:
    """Metrics for a single best / worst case scenario entry."""

    scenario_type:     str            # "best" or "worst"
    file_path:         Path
    file_size_mb:      float
    compression_ratio: float
    system_metrics:    SystemMetrics

    @property
    def ram_per_mb(self) -> float:
        """Peak RAM (MB) per MB of input file."""
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
    """Ranks benchmark results and identifies the best and worst case."""

    SUPPORTED_METRICS = {
        "compression_ratio": ("Compression Ratio", True),    # (label, higher_is_better)
        "ram_usage":         ("RAM Usage",          False),
        "cpu_usage":         ("CPU Usage",          False),
        "io_total":          ("I/O Total",          False),
    }

    @staticmethod
    def identify_scenarios(results: List, metric: str = "compression_ratio") -> dict:
        """
        Return {"best": ScenarioMetrics, "worst": ScenarioMetrics} for the given metric.
        Returns {"best": None, "worst": None} when fewer than two valid results exist.
        """
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
            if metric == "compression_ratio": return r.metrics.compression_ratio
            if metric == "ram_usage":         return r.system_metrics.peak_ram_mb
            if metric == "cpu_usage":         return r.system_metrics.avg_process_cpu
            return r.system_metrics.io_total_mb

        ranked = sorted(valid, key=sort_key, reverse=higher_is_better)

        def to_scenario(r, stype: str) -> ScenarioMetrics:
            return ScenarioMetrics(
                scenario_type     = stype,
                file_path         = r.image_path,
                file_size_mb      = r.metrics.original_size / (1024 * 1024),
                compression_ratio = r.metrics.compression_ratio,
                system_metrics    = r.system_metrics,
            )

        return {
            "best":  to_scenario(ranked[0],  "best"),
            "worst": to_scenario(ranked[-1], "worst"),
        }

    @staticmethod
    def print_scenario_comparison(scenarios: dict, log_callback) -> None:
        """Write a best-vs-worst comparison to log_callback."""
        best  = scenarios.get("best")
        worst = scenarios.get("worst")
        if not best or not worst:
            return

        log_callback(f"\n{'=' * 70}")
        log_callback("BEST vs WORST CASE ANALYSIS")
        log_callback(f"{'=' * 70}")

        for label, s in [("BEST CASE", best), ("WORST CASE", worst)]:
            log_callback(f"\n{label}:")
            log_callback(f"  File:              {s.file_path.name}")
            log_callback(f"  Input size:        {s.file_size_mb:.2f} MB")
            log_callback(f"  Compression ratio: {s.compression_ratio:.2f}x")
            log_callback(f"  Peak RAM:          {s.system_metrics.peak_ram_mb:.1f} MB")
            log_callback(
                f"  Avg CPU:           {s.system_metrics.cpu_percent_normalized:.1f}% "
                f"(raw {s.system_metrics.avg_process_cpu:.1f}%, "
                f"{s.system_metrics.logical_core_count} cores)"
            )
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