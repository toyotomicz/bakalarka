"""
System Resource Monitoring Module

Measures CPU, RAM, and I/O metrics in a background thread while a compressor runs

Classes:
    SystemMonitor    : Adaptive background-thread sampler for CPU / RAM / I/O
    ScenarioAnalyzer : Identifies best / worst case results from a result set
    ScenarioMetrics  : Per-scenario summary data (compression ratio, RAM, CPU, I/O)

Important measurement notes:
    - CPU values can exceed 100 % on multi-core systems (e.g. 200 % = 2 cores fully busy).  
        Use cpu_percent_normalized / peak_cpu_percent_normalized for 0–100 %
        display values that are comparable across machines.
    - RAM measurements include the entire Python process baseline (~300 MB for GUI apps)
        they are NOT a measure of compression-algorithm memory overhead alone.
        Use net_peak_ram_mb / net_avg_ram_mb to subtract the baseline.
    - I/O readings may be lower than file size due to OS read-ahead caching.
    - CLI-based compressors (cwebp.exe, optipng.exe, etc.) spawn short-lived child
        processes. Their I/O is tracked via a persistent PID cache (_seen_pids) so
        data is not lost when the process exits before the next sampling tick.
"""

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import psutil

logger = logging.getLogger(__name__)

# Number of logical CPU cores, used for normalisation throughout this module
_LOGICAL_CORES: int = psutil.cpu_count(logical=True) or 1


# Data structures

@dataclass
class SystemSnapshot:
    """
    Single point-in-time resource sample collected by the monitor thread

    Attributes:
        timestamp: perf_counter() value at sample time.
        cpu_percent: Total system-wide CPU usage (in %).
        process_cpu_percent: This process + children, cumulative across cores (%).
            Can exceed 100 % on multi-core systems.
        ram_used_mb: RSS of this process + children (MB).
        ram_available_mb: Free system RAM (MB).
        io_read_mb: Cumulative process read bytes converted to MB.
        io_write_mb: Cumulative process write bytes converted to MB.
    """

    timestamp:           float
    cpu_percent:         float
    process_cpu_percent: float
    ram_used_mb:         float
    ram_available_mb:    float
    io_read_mb:          float
    io_write_mb:         float


@dataclass
class SystemMetrics:
    """
    Aggregated resource statistics for one complete benchmark run.

    Raw CPU fields (avg_process_cpu, max_process_cpu) are cumulative across all
    logical cores, they can exceed 100 % on multi-core systems (200 % means two
    cores fully utilised).  Use cpu_percent_normalized / peak_cpu_percent_normalized
    for a 0–100 % display value that is comparable across machines.

    I/O fields include data from short-lived child processes (CLI compressors)
    thanks to the persistent PID cache in SystemMonitor, their I/O is not lost
    when they exit between two sampling ticks.

    Attributes:
        avg_cpu_percent: Average total system-wide CPU (%).
        max_cpu_percent: Peak total system-wide CPU (%).
        avg_process_cpu: Average process + children CPU, cumulative (%).
        max_process_cpu: Peak process + children CPU, cumulative (%).
            Note: when the whole-run GetProcessTimes() delta method is used,
            max_process_cpu equals avg_process_cpu because only one aggregate
            delta is computed, it is not a sample-by-sample maximum.
        avg_ram_mb: Average RSS of this process + children (MB).
        peak_ram_mb: Maximum RSS observed during the run (MB).
        ram_baseline_mb: RSS at start(), the Python / GUI overhead floor.
        total_io_read_mb: Net read bytes during the run (MB).
        total_io_write_mb: Net write bytes during the run (MB).
        duration_seconds: Wall-clock duration of the monitored interval.
        sample_count: Number of snapshots collected.
        logical_core_count: Core count used for CPU normalisation.
        is_reliable: True when enough samples were collected for reliable stats.
        reliability_score: Fraction of the minimum required sample count (0–1).
        measurement_quality: 'good', 'fair', 'poor', or 'none'.
    """

    # CPU, raw cumulative values (may exceed 100 % on multi-core systems)
    avg_cpu_percent: float
    max_cpu_percent: float
    avg_process_cpu: float
    max_process_cpu: float

    # Memory (includes Python + GUI baseline, not compression overhead alone)
    avg_ram_mb:      float
    peak_ram_mb:     float
    ram_baseline_mb: float

    # I/O (delta from start to end, including exited child processes)
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
    def net_peak_ram_mb(self) -> float:
        """
        Peak RAM above the baseline measured at start()

        Use this value when comparing runs across sessions, it removes the
        Python / GUI memory floor (~200–450 MB) that grows with each run because
        previous benchmark results accumulate in memory.

        Returns:
            Max(0, peak_ram_mb - ram_baseline_mb).
        """
        return max(0.0, self.peak_ram_mb - self.ram_baseline_mb)

    @property
    def net_avg_ram_mb(self) -> float:
        """
        Average RAM above the baseline measured at start()

        Returns:
            Max(0, avg_ram_mb - ram_baseline_mb).
        """
        return max(0.0, self.avg_ram_mb - self.ram_baseline_mb)

    @property
    def io_total_mb(self) -> float:
        """Total I/O (read + write) in MB."""
        return self.total_io_read_mb + self.total_io_write_mb

    @property
    def ram_efficiency_mb_per_sec(self) -> float:
        """
        Average RAM usage per second of operation

        Returns:
            0.0 when duration_seconds is zero.
        """
        if self.duration_seconds == 0:
            return 0.0
        return self.avg_ram_mb / self.duration_seconds

    @property
    def cpu_percent_normalized(self) -> float:
        """
        Average process CPU normalised to 0–100 %

        Raw psutil values are cumulative across all logical cores (e.g. 200 %
        means two cores fully utilised).  This property divides by
        logical_core_count, which equals the number of pinned cores when CPU
        affinity is active, or the total machine core count otherwise.

        The result is clamped to [0, 100] because psutil occasionally reports
        momentary values fractionally above the theoretical maximum due to
        kernel scheduling granularity, these are measurement artefacts.

        Returns:
            Process CPU usage as a percentage of one core's capacity (0–100)
        """
        if self.logical_core_count <= 0:
            return 0.0
        return min(100.0, max(0.0, self.avg_process_cpu / self.logical_core_count))

    @property
    def peak_cpu_percent_normalized(self) -> float:
        """
        Peak process CPU normalised to 0–100 %

        Same normalisation and clamping as cpu_percent_normalized.

        Note:
            When the whole-run GetProcessTimes() delta method is used,
            this value equals cpu_percent_normalized because only one aggregate
            delta is available, it is not a true sample-by-sample maximum.
        """
        if self.logical_core_count <= 0:
            return 0.0
        return min(100.0, max(0.0, self.max_process_cpu / self.logical_core_count))


# System monitor

class SystemMonitor:
    """
    Measures CPU, RAM, and I/O in a background thread during compression.

    Design decisions:
        - cpu_percent(interval=None) is called in __init__ so the first real
            sample is not always 0.0 (psutil requires one warm-up call to
            initialise its internal counter).
        - time.perf_counter() is used instead of time.time() for sub-millisecond accuracy.
        - The sampling interval adapts to the input file size: smaller files
            trigger faster sampling so brief operations are captured.
        - Child processes (e.g. cwebp.exe, optipng.exe) are tracked via a
            persistent PID → I/O cache (_seen_pids).  When a child exits between
            two sampling ticks its last-seen I/O counters are retained,
            so CLI-based compressors are no longer under-reported.

    CPU interpretation:
        Reported values are cumulative across all logical cores.  
        On a 4-core machine a fully-loaded single-threaded process shows ~100 %; 
        a 4-thread workload can show ~400 %.  
        Use SystemMetrics.cpu_percent_normalized for a 0–100 % display value.

    Class attributes:
        ULTRA_FAST_INTERVAL: 1 ms sampling, used for files < 10 KB.
        FAST_INTERVAL: 10 ms sampling, used for files < 100 KB.
        NORMAL_INTERVAL: 50 ms sampling, used for all other files.
        SLOW_INTERVAL: 100 ms sampling, defined but not currently used;
            reserved for future very-large-file handling.
        MIN_SAMPLES_RELIABLE: Minimum sample count for 'good' quality rating.
        MIN_SAMPLES_FAIR: Minimum sample count for 'fair' quality rating.
        MIN_DURATION_RELIABLE: Minimum wall-clock duration for reliable stats.
    """

    ULTRA_FAST_INTERVAL = 0.001   # 1 ms , files < 10 KB
    FAST_INTERVAL       = 0.010   # 10 ms, files < 100 KB
    NORMAL_INTERVAL     = 0.050   # 50 ms, all other files
    # SLOW_INTERVAL is defined for completeness but not used in _set_adaptive_sampling
    # It produces too few samples for compressors that finish in under 300 ms
    SLOW_INTERVAL       = 0.100   # 100 ms, not currently applied

    MIN_SAMPLES_RELIABLE  = 5
    MIN_SAMPLES_FAIR      = 3
    MIN_DURATION_RELIABLE = 0.010  # 10 ms

    def __init__(
        self,
        sampling_interval: float = 0.05,
        adaptive:          bool  = True,
        force_pre_post:    bool  = True,
    ) -> None:
        """
        Initialize the monitor with the given sampling interval and options

        Args:
            sampling_interval: Default interval between samples in seconds.
                Overridden by adaptive logic if adaptive=True.
            adaptive: When True, automatically tighten the sampling interval
                for small input files so brief operations are captured.
            force_pre_post: When True, collect one snapshot immediately before
                the first monitor tick (pre) and one immediately after stop()
                (post) to bookend the measured interval.
        """
        self.base_sampling_interval = sampling_interval
        self.sampling_interval      = sampling_interval
        self.adaptive               = adaptive
        self.force_pre_post         = force_pre_post

        self.process = psutil.Process(os.getpid())

        # Warm up psutil internal CPU counter so the first real sample is not 0
        # Without this initialisation call, cpu_percent(interval=None) always returns
        # 0.0 because psutil has no prior measurement to diff against
        self.process.cpu_percent(interval=None)
        psutil.cpu_percent(interval=None)
        for child in self.process.children(recursive=True):
            try:
                child.cpu_percent(interval=None)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        self.snapshots:      List[SystemSnapshot]       = []
        self.is_monitoring:  bool                       = False
        self.monitor_thread: Optional[threading.Thread] = None

        self.start_time: float = 0.0
        self.end_time:   float = 0.0

        self.pre_snapshot:  Optional[SystemSnapshot] = None
        self.post_snapshot: Optional[SystemSnapshot] = None

        # Persistent PID → (io_read_mb, io_write_mb) cache
        #
        # CLI compressors (cwebp.exe, optipng.exe, etc.) are short-lived child processes.
        # They can finish and their PID can disappear between two sampling ticks.  
        # Reading io_counters() only for *live* processes silently
        # loses all I/O from processes that exited in between.
        #
        # Solution: every time we successfully read io_counters() for a PID we store
        # the latest cumulative value here.  When the PID is gone on the next tick we
        # keep the last-known value. _calculate_metrics() then sums the *deltas*
        # (end − start) for every PID seen during the run, regardless of whether the
        # process is still alive at stop() time.
        #
        # Thread safety: only the monitor thread writes; _calculate_metrics() reads
        # only after the thread has been joined, so no lock is needed.
        self._seen_pids:    Dict[int, Tuple[float, float]] = {}  # pid: (read_mb, write_mb)
        self._initial_pids: Dict[int, Tuple[float, float]] = {}  # baseline snapshot at start()

        # Persistent CPU-time cache, mirrors _seen_pids but for CPU accounting.
        # Updated every tick for every live PID so that short-lived child processes
        # have their CPU times captured even if they exit before stop() is called.
        self._seen_cpu:    Dict[int, Tuple[float, float]] = {}  # pid: (user_s, system_s)
        self._initial_cpu: Dict[int, Tuple[float, float]] = {}  # baseline at start()


    # Public API

    def start(
        self,
        file_size_bytes:   Optional[int] = None,
        pinned_core_count: Optional[int] = None,
    ) -> None:
        """
        Begin monitoring.  Safe to call when already running (no-op).

        Args:
            file_size_bytes: Source file size in bytes, used to choose the
                adaptive sampling interval.
            pinned_core_count: Number of CPU cores the process is pinned to via
                affinity. When set, CPU normalization uses this value instead
                of the total logical core count, so a single-threaded compressor
                pinned to 1 core correctly shows 0–100 % instead of 0–(100/N) %.
        """
        if self.is_monitoring:
            return

        # Store the effective core count for normalization
        # Fall back to the machine total when no affinity is active
        self._effective_core_count: int = (
            max(1, pinned_core_count) if pinned_core_count else _LOGICAL_CORES
        )

        if file_size_bytes is not None and self.adaptive:
            self._set_adaptive_sampling(file_size_bytes)

        self.snapshots     = []
        self._seen_pids    = {}
        self._initial_pids = {}
        self._seen_cpu     = {}
        self._initial_cpu  = {}
        self.is_monitoring = True
        self.start_time    = time.perf_counter()

        if self.force_pre_post:
            self.pre_snapshot = self._take_snapshot()

        # RAM baseline: RSS of the entire Python process at monitoring start.
        # Subtracting this from peak / avg gives a "net RAM" value that is
        # comparable across runs in the same session regardless of how much
        # memory previous benchmark results occupy.
        self._ram_baseline_mb: float = (
            self.pre_snapshot.ram_used_mb if self.pre_snapshot else 0.0
        )

        # Record the I/O baseline for all currently running PIDs (parent + any pre-existing children)
        # so their background I/O (DLL loads, Python imports, etc.)
        # does not contaminate the delta attributed to compression
        self._initial_pids = dict(self._seen_pids)

        # Record CPU-time baseline for the same set of PIDs
        self._initial_cpu = dict(self._seen_cpu)

        self.monitor_thread = threading.Thread(
            target=self._monitor_loop, daemon=True, name="SystemMonitor"
        )
        self.monitor_thread.start()

    def stop(self) -> SystemMetrics:
        """
        Stop monitoring and return aggregated metrics

        Returns:
            SystemMetrics summarising the interval between start() and stop().
        """
        self.is_monitoring = False
        self.end_time      = time.perf_counter()

        if self.force_pre_post:
            self.post_snapshot = self._take_snapshot()

        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)

        # Update _seen_cpu one final time for any still-running processes.
        # Processes that already exited have their last-known CPU times in
        # _seen_cpu from earlier ticks, those are already the correct final values.
        all_procs = [self.process]
        try:
            all_procs.extend(self.process.children(recursive=True))
        except (psutil.Error, OSError):
            pass
        for proc in all_procs:
            times = self._read_cpu_times(proc)
            if times is not None:
                self._seen_cpu[proc.pid] = times

        return self._calculate_metrics()
    
    
    # Internal helpers

    def _read_cpu_times(self, proc: "psutil.Process") -> Optional[Tuple[float, float]]:
        """
        Read (user_seconds, system_seconds) from GetProcessTimes() for one process

        Args:
            proc: psutil.Process to query.

        Returns:
            (user_s, system_s) tuple, or None if the process is gone or access
            is denied.
        """
        try:
            t = proc.cpu_times()
            return (t.user, t.system)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.Error):
            return None

    def _monitor_loop(self) -> None:
        """Background thread body: collect snapshots until is_monitoring is False."""
        while self.is_monitoring:
            try:
                self.snapshots.append(self._take_snapshot())
            except (psutil.Error, OSError) as exc:
                logger.debug("SystemMonitor snapshot error: %s", exc)
            time.sleep(self.sampling_interval)

    def _take_snapshot(self) -> SystemSnapshot:
        """
        Collect one resource sample

        CPU, RAM, and I/O are summed across the parent process and all currently
        live child processes.  I/O for processes that have already exited is
        preserved in _seen_pids and included in the final delta calculation.

        New child processes (compressor CLI tools) receive an immediate
        cpu_percent(interval=None) warm-up call on their first appearance so
        the *next* tick returns a meaningful value instead of 0.0.

        Returns:
            SystemSnapshot with the current resource state.
        """
        cpu_system = psutil.cpu_percent(interval=0)

        # Rebuild the process list each tick because compressor child processes
        # are created after start() is called.
        all_processes = [self.process]
        try:
            all_processes.extend(self.process.children(recursive=True))
        except (psutil.Error, OSError):
            pass

        if logger.isEnabledFor(logging.DEBUG) and len(all_processes) > 1:
            logger.debug(
                "Snapshot: %d process(es), PIDs %s",
                len(all_processes),
                ", ".join(str(p.pid) for p in all_processes),
            )

        # ***CPU***
        # Warm up the cpu_percent counter for any PID we haven't seen before.
        # Without this, the first cpu_percent(interval=0) call for a new PID always
        # returns 0.0 because psutil has no prior measurement to diff against.
        # The warm-up call itself returns 0.0 and is discarded; the *next* tick
        # will return the real accumulated value since the warm-up call.
        cpu_process = 0.0
        for proc in all_processes:
            try:
                if proc.pid not in self._seen_pids:
                    proc.cpu_percent(interval=None)  # warm-up; return value discarded
                else:
                    cpu_process += proc.cpu_percent(interval=0)
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.Error):
                pass

        # ***RAM***
        ram_used_mb = 0.0
        for proc in all_processes:
            try:
                ram_used_mb += proc.memory_info().rss / (1024 * 1024)
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.Error):
                pass

        # ***I/O (live processes only; _seen_pids caches exited ones)***
        # For each live process we read its cumulative I/O and update _seen_pids.
        # The snapshot carries the *current total* across all live processes.
        # Exited-process I/O is not added to the snapshot value, it is added
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
                # Process exited between the children() call and io_counters(),
                # the last value is already stored in _seen_pids from a prior tick.
                pass

        # ***CPU times cache (mirrors _seen_pids strategy for I/O)***
        # Read GetProcessTimes() for every live process and persist the latest
        # cumulative value.  When a process exits between ticks its last entry
        # stays in _seen_cpu, giving _calculate_metrics() a correct final delta.
        for proc in all_processes:
            try:
                t = proc.cpu_times()
                self._seen_cpu[proc.pid] = (t.user, t.system)
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.Error):
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
        The interval is capped at NORMAL_INTERVAL even for large files because
        SLOW_INTERVAL (100 ms) produces too few samples for compressors that
        finish in under 300 ms (e.g. Pillow-TIFF on a ~1 MB file).

        Args:
            file_size_bytes: Source file size used to pick the sampling tier.
        """
        kb = file_size_bytes / 1024
        if kb < 10:
            interval = self.ULTRA_FAST_INTERVAL
        elif kb < 100:
            interval = self.FAST_INTERVAL
        else:
            # Use NORMAL_INTERVAL for everything >= 100 KB including large files.
            # SLOW_INTERVAL is intentionally skipped, see class docstring.
            interval = self.NORMAL_INTERVAL

        self.sampling_interval = interval
        logger.debug(
            "Adaptive sampling: file_size=%.1f KB → interval=%.3f s",
            kb, self.sampling_interval,
        )

    def _calculate_metrics(self) -> SystemMetrics:
        """
        Aggregate all snapshots into a single SystemMetrics summary.

        I/O strategy:
            Compute I/O as the sum of per-PID deltas (end − start) using
            _seen_pids and _initial_pids.  This correctly captures:

                1. The parent Python process
                2. Long-lived child processes (still alive at stop())
                3. Short-lived child processes (exited before stop()), their
                    last-seen cumulative value in _seen_pids minus their baseline
                    in _initial_pids gives the net I/O during the benchmark run

            The snapshot-based first-to-last delta is kept as a fallback for
            environments where io_counters() is unavailable (some VMs / containers).

        CPU strategy:
            Compute a single whole-run delta using GetProcessTimes() (via _seen_cpu / _initial_cpu). 
            This eliminates per-sample FILETIME quantisation error (~15.6 ms on Windows)
            that caused occasional >100 % readings with the sample-average approach.
            
            Because only one aggregate delta is produced, avg_process_cpu and
            max_process_cpu are both set to the same value, max_process_cpu is
            not a true sample-by-sample maximum in this code path.

        Returns:
            SystemMetrics summarising the interval between start() and stop().
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

        ram_vals = [s.ram_used_mb  for s in all_snaps]
        cpu_vals = [s.cpu_percent  for s in all_snaps]

        # ***CPU: whole-run delta via persistent GetProcessTimes() cache***
        # For every PID seen during the run:
        #   delta = (user_end + system_end) − (user_start + system_start)
        # Sum all deltas, divide by wall time: total CPU utilisation percentage.
        #
        # Shortlived child processes (optipng.exe, cwebp.exe, etc.) are included
        # because _seen_cpu is updated every monitor tick: even if the process
        # exits before stop(), its last-known CPU times remain in _seen_cpu.
        cpu_start = getattr(self, "_initial_cpu", {})
        cpu_end   = getattr(self, "_seen_cpu",    {})

        total_cpu_delta = 0.0
        for pid, (end_user, end_sys) in cpu_end.items():
            start_user, start_sys = cpu_start.get(pid, (0.0, 0.0))
            total_cpu_delta += max(0.0, (end_user - start_user) + (end_sys - start_sys))

        if duration > 0 and total_cpu_delta > 0:
            effective_cores = getattr(self, "_effective_core_count", _LOGICAL_CORES)
            raw_whole_run   = (total_cpu_delta / duration) * 100.0
            # Clamp to the physical ceiling (100 % x pinned cores).
            raw_whole_run   = min(raw_whole_run, 100.0 * effective_cores)
            avg_process_cpu = raw_whole_run
            # With the whole-run delta method we have only one aggregate value;
            # set max equal to avg rather than leaving it misleadingly at 0.
            max_process_cpu = raw_whole_run
        else:
            # Fallback: per-sample average (VMs / containers without cpu_times()).
            cpu_proc_vals   = [s.process_cpu_percent for s in all_snaps]
            avg_process_cpu = sum(cpu_proc_vals) / len(cpu_proc_vals) if cpu_proc_vals else 0.0
            max_process_cpu = max(cpu_proc_vals) if cpu_proc_vals else 0.0

        # ***I/O delta via persistent PID cache***
        # Sum (final − initial) for every PID seen during the run.
        # PIDs that existed before start() use their _initial_pids baseline;
        # PIDs spawned during the run have no baseline entry, so their full
        # cumulative value is attributed to the run.
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
            avg_cpu_percent     = sum(cpu_vals) / n if n else 0.0,
            max_cpu_percent     = max(cpu_vals) if cpu_vals else 0.0,
            avg_process_cpu     = avg_process_cpu,
            max_process_cpu     = max_process_cpu,
            avg_ram_mb          = sum(ram_vals) / n,
            peak_ram_mb         = max(ram_vals),
            ram_baseline_mb     = getattr(self, "_ram_baseline_mb", 0.0),
            total_io_read_mb    = io_read_delta,
            total_io_write_mb   = io_write_delta,
            duration_seconds    = duration,
            sample_count        = sample_count,
            logical_core_count  = getattr(self, "_effective_core_count", _LOGICAL_CORES),
            is_reliable         = is_reliable,
            reliability_score   = reliability_score,
            measurement_quality = quality,
        )

    @staticmethod
    def _empty_metrics(duration: float) -> SystemMetrics:
        """
        Return an all-zero SystemMetrics when no samples were collected.

        Args:
            duration: Wall-clock duration of the monitored interval.

        Returns:
            SystemMetrics with all numeric fields set to zero and
            measurement_quality='none'.
        """
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


# Scenario analysis

@dataclass
class ScenarioMetrics:
    """
    Metrics for a single best / worst case scenario entry

    Attributes:
        scenario_type: 'best' or 'worst'.
        file_path: Path to the source image for this scenario.
        file_size_mb: Input file size in MB (uncompressed pixel data).
        compression_ratio: Achieved compression ratio.
        system_metrics: Full resource measurements for this run.
    """

    scenario_type:     str
    file_path:         Path
    file_size_mb:      float
    compression_ratio: float
    system_metrics:    SystemMetrics

    @property
    def ram_per_mb(self) -> float:
        """
        Peak RAM (MB) per MB of input file

        Returns:
            0.0 when file_size_mb is zero.
        """
        if self.file_size_mb == 0:
            return 0.0
        return self.system_metrics.peak_ram_mb / self.file_size_mb

    @property
    def cpu_efficiency(self) -> float:
        """
        CPU seconds consumed per MB of input file

        Returns:
            0.0 when file_size_mb is zero.
        """
        if self.file_size_mb == 0:
            return 0.0
        cpu_time = (
            self.system_metrics.avg_process_cpu / 100.0
        ) * self.system_metrics.duration_seconds
        return cpu_time / self.file_size_mb


class ScenarioAnalyzer:
    """
    Ranks benchmark results and identifies the best and worst case scenarios for a given metric

    Class attributes:
        SUPPORTED_METRICS: Dict mapping metric key → (label, higher_is_better)
    """

    SUPPORTED_METRICS = {
        "compression_ratio": ("Compression Ratio", True),    # higher = better
        "ram_usage":         ("RAM Usage",          False),  # lower = better
        "cpu_usage":         ("CPU Usage",          False),
        "io_total":          ("I/O Total",          False),
    }

    @staticmethod
    def identify_scenarios(results: List, metric: str = "compression_ratio") -> dict:
        """
        Return the best and worst case BenchmarkResults for the given metric.

        Args:
            results: List of BenchmarkResult objects to analyse.
            metric: One of the keys in SUPPORTED_METRICS.

        Returns:
            Dict with keys 'best' and 'worst', each holding a ScenarioMetrics
            object.  Both are None when fewer than two valid results exist.
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
        """
        Write a best-vs-worst comparison to log_callback.

        Args:
            scenarios: Dict with 'best' and 'worst' ScenarioMetrics values.
            log_callback: Callable(str) for progress / log output.
        """
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
                f"  Avg CPU:           {s.system_metrics.cpu_percent_normalized:.1f}%"
            )
            log_callback(f"  Total I/O:         {s.system_metrics.io_total_mb:.2f} MB")

        if best.system_metrics.peak_ram_mb > 0:
            ram_ratio = worst.system_metrics.peak_ram_mb / best.system_metrics.peak_ram_mb
            log_callback(f"\n  RAM variation:  {ram_ratio:.2f}x")

        if best.system_metrics.cpu_percent_normalized > 0:
            cpu_ratio = (
                worst.system_metrics.cpu_percent_normalized
                / best.system_metrics.cpu_percent_normalized
            )
            log_callback(f"  CPU variation:  {cpu_ratio:.2f}x")