"""
System Resource Monitoring Module
Monitors CPU, RAM, and I/O metrics during compression benchmarks
"""

import os
import time
import threading
import psutil
from dataclasses import dataclass
from typing import Optional, List
from pathlib import Path


@dataclass
class SystemSnapshot:
    """Single point-in-time system resource measurement"""
    timestamp: float
    cpu_percent: float  # Overall CPU usage
    process_cpu_percent: float  # This process CPU usage
    ram_used_mb: float  # Process RAM usage in MB
    ram_available_mb: float  # System available RAM in MB
    io_read_mb: float  # Cumulative IO read in MB
    io_write_mb: float  # Cumulative IO write in MB


@dataclass
class SystemMetrics:
    """Aggregated system resource metrics for a benchmark run"""
    # CPU metrics
    avg_cpu_percent: float
    max_cpu_percent: float
    avg_process_cpu: float
    max_process_cpu: float
    
    # Memory metrics
    avg_ram_mb: float
    max_ram_mb: float
    peak_ram_mb: float  # Absolute peak during operation
    
    # I/O metrics
    total_io_read_mb: float
    total_io_write_mb: float
    
    # Timing
    duration_seconds: float
    sample_count: int
    
    @property
    def ram_efficiency_mb_per_sec(self) -> float:
        """Average RAM usage per second of operation"""
        if self.duration_seconds == 0:
            return 0.0
        return self.avg_ram_mb / self.duration_seconds
    
    @property
    def io_total_mb(self) -> float:
        """Total I/O (read + write)"""
        return self.total_io_read_mb + self.total_io_write_mb


class SystemMonitor:
    """
    Monitors system resources during benchmark execution
    Runs in a background thread with configurable sampling rate
    """
    
    def __init__(self, sampling_interval: float = 0.1):
        """
        Args:
            sampling_interval: Time between samples in seconds (default 100ms)
        """
        self.sampling_interval = sampling_interval
        self.process = psutil.Process(os.getpid())
        
        self.snapshots: List[SystemSnapshot] = []
        self.is_monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        
        # Baseline measurements
        self.baseline_io_read = 0
        self.baseline_io_write = 0
        self.start_time = 0
        self.end_time = 0
    
    def start(self):
        """Start monitoring in background thread"""
        if self.is_monitoring:
            return
        
        self.snapshots = []
        self.is_monitoring = True
        self.start_time = time.time()
        
        # Get baseline I/O
        try:
            io_counters = self.process.io_counters()
            self.baseline_io_read = io_counters.read_bytes / (1024 * 1024)
            self.baseline_io_write = io_counters.write_bytes / (1024 * 1024)
        except:
            self.baseline_io_read = 0
            self.baseline_io_write = 0
        
        # Start monitoring thread
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
    
    def stop(self) -> SystemMetrics:
        """
        Stop monitoring and return aggregated metrics
        
        Returns:
            SystemMetrics with aggregated resource usage
        """
        self.is_monitoring = False
        self.end_time = time.time()
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        
        return self._calculate_metrics()
    
    def _monitor_loop(self):
        """Background monitoring loop"""
        while self.is_monitoring:
            try:
                snapshot = self._take_snapshot()
                self.snapshots.append(snapshot)
            except Exception:
                pass  # Silently ignore monitoring errors
            
            time.sleep(self.sampling_interval)
    
    def _take_snapshot(self) -> SystemSnapshot:
        """Capture current system state"""
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=0)
        process_cpu = self.process.cpu_percent(interval=0)
        
        # Memory usage
        mem_info = self.process.memory_info()
        ram_used = mem_info.rss / (1024 * 1024)  # MB
        
        virtual_mem = psutil.virtual_memory()
        ram_available = virtual_mem.available / (1024 * 1024)  # MB
        
        # I/O counters
        try:
            io_counters = self.process.io_counters()
            io_read = io_counters.read_bytes / (1024 * 1024)
            io_write = io_counters.write_bytes / (1024 * 1024)
        except:
            io_read = 0
            io_write = 0
        
        return SystemSnapshot(
            timestamp=time.time(),
            cpu_percent=cpu_percent,
            process_cpu_percent=process_cpu,
            ram_used_mb=ram_used,
            ram_available_mb=ram_available,
            io_read_mb=io_read,
            io_write_mb=io_write
        )
    
    def _calculate_metrics(self) -> SystemMetrics:
        """Calculate aggregated metrics from snapshots"""
        if not self.snapshots:
            return SystemMetrics(
                avg_cpu_percent=0, max_cpu_percent=0,
                avg_process_cpu=0, max_process_cpu=0,
                avg_ram_mb=0, max_ram_mb=0, peak_ram_mb=0,
                total_io_read_mb=0, total_io_write_mb=0,
                duration_seconds=0, sample_count=0
            )
        
        # CPU metrics
        cpu_percents = [s.cpu_percent for s in self.snapshots]
        process_cpus = [s.process_cpu_percent for s in self.snapshots]
        
        avg_cpu = sum(cpu_percents) / len(cpu_percents)
        max_cpu = max(cpu_percents)
        avg_process_cpu = sum(process_cpus) / len(process_cpus)
        max_process_cpu = max(process_cpus)
        
        # Memory metrics
        ram_values = [s.ram_used_mb for s in self.snapshots]
        avg_ram = sum(ram_values) / len(ram_values)
        max_ram = max(ram_values)
        peak_ram = max(ram_values)
        
        # I/O metrics (delta from baseline)
        last_snapshot = self.snapshots[-1]
        io_read_delta = last_snapshot.io_read_mb - self.baseline_io_read
        io_write_delta = last_snapshot.io_write_mb - self.baseline_io_write
        
        # Duration
        duration = self.end_time - self.start_time
        
        return SystemMetrics(
            avg_cpu_percent=avg_cpu,
            max_cpu_percent=max_cpu,
            avg_process_cpu=avg_process_cpu,
            max_process_cpu=max_process_cpu,
            avg_ram_mb=avg_ram,
            max_ram_mb=max_ram,
            peak_ram_mb=peak_ram,
            total_io_read_mb=max(0, io_read_delta),
            total_io_write_mb=max(0, io_write_delta),
            duration_seconds=duration,
            sample_count=len(self.snapshots)
        )


class ProcessIsolator:
    """
    Provides process isolation for more accurate benchmarking
    Uses process affinity and priority adjustment
    """
    
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.original_affinity = None
        self.original_priority = None
    
    def isolate(self, cpu_cores: Optional[List[int]] = None, high_priority: bool = True):
        """
        Isolate process for benchmarking
        
        Args:
            cpu_cores: List of CPU cores to pin to (None = all cores)
            high_priority: Set process to high priority
        """
        try:
            # Save original settings
            self.original_affinity = self.process.cpu_affinity()
            self.original_priority = self.process.nice()
            
            # Set CPU affinity if specified
            if cpu_cores is not None and hasattr(self.process, 'cpu_affinity'):
                self.process.cpu_affinity(cpu_cores)
            
            # Set high priority if requested
            if high_priority:
                if os.name == 'nt':  # Windows
                    self.process.nice(psutil.HIGH_PRIORITY_CLASS)
                else:  # Unix-like
                    self.process.nice(-10)  # Negative = higher priority
        
        except Exception as e:
            print(f"Warning: Could not fully isolate process: {e}")
    
    def restore(self):
        """Restore original process settings"""
        try:
            if self.original_affinity is not None:
                self.process.cpu_affinity(self.original_affinity)
            if self.original_priority is not None:
                self.process.nice(self.original_priority)
        except Exception:
            pass


@dataclass
class ScenarioMetrics:
    """Metrics for best/worst case scenarios"""
    scenario_type: str  # "best" or "worst"
    file_path: Path
    file_size_mb: float
    compression_ratio: float
    system_metrics: SystemMetrics
    
    @property
    def ram_per_mb(self) -> float:
        """RAM usage per MB of input file"""
        if self.file_size_mb == 0:
            return 0.0
        return self.system_metrics.peak_ram_mb / self.file_size_mb
    
    @property
    def cpu_efficiency(self) -> float:
        """CPU seconds per MB processed"""
        if self.file_size_mb == 0:
            return 0.0
        cpu_time = (self.system_metrics.avg_process_cpu / 100.0) * self.system_metrics.duration_seconds
        return cpu_time / self.file_size_mb


class ScenarioAnalyzer:
    """
    Analyzes benchmark results to identify best/worst case scenarios
    """
    
    @staticmethod
    def identify_scenarios(results: List, metric: str = "compression_ratio") -> dict:
        """
        Identify best and worst case scenarios based on a metric
        
        Args:
            results: List of benchmark results with system_metrics
            metric: Metric to use for comparison
                   ("compression_ratio", "ram_usage", "cpu_usage", "io_total")
        
        Returns:
            Dict with "best" and "worst" ScenarioMetrics
        """
        if not results:
            return {"best": None, "worst": None}
        
        valid_results = [r for r in results if hasattr(r, 'system_metrics') and r.system_metrics]
        
        if not valid_results:
            return {"best": None, "worst": None}
        
        # Sort based on metric
        if metric == "compression_ratio":
            sorted_results = sorted(valid_results, 
                                   key=lambda r: r.metrics.compression_ratio, 
                                   reverse=True)
        elif metric == "ram_usage":
            sorted_results = sorted(valid_results,
                                   key=lambda r: r.system_metrics.peak_ram_mb)
        elif metric == "cpu_usage":
            sorted_results = sorted(valid_results,
                                   key=lambda r: r.system_metrics.avg_process_cpu)
        elif metric == "io_total":
            sorted_results = sorted(valid_results,
                                   key=lambda r: r.system_metrics.io_total_mb)
        else:
            return {"best": None, "worst": None}
        
        best = sorted_results[0]
        worst = sorted_results[-1]
        
        return {
            "best": ScenarioMetrics(
                scenario_type="best",
                file_path=best.image_path,
                file_size_mb=best.metrics.original_size / (1024 * 1024),
                compression_ratio=best.metrics.compression_ratio,
                system_metrics=best.system_metrics
            ),
            "worst": ScenarioMetrics(
                scenario_type="worst",
                file_path=worst.image_path,
                file_size_mb=worst.metrics.original_size / (1024 * 1024),
                compression_ratio=worst.metrics.compression_ratio,
                system_metrics=worst.system_metrics
            )
        }
    
    @staticmethod
    def print_scenario_comparison(scenarios: dict, log_callback):
        """Print detailed comparison of best vs worst scenarios"""
        best = scenarios.get("best")
        worst = scenarios.get("worst")
        
        if not best or not worst:
            return
        
        log_callback(f"\n{'='*70}")
        log_callback("BEST vs WORST CASE ANALYSIS")
        log_callback(f"{'='*70}")
        
        log_callback("\nBEST CASE:")
        log_callback(f"  File: {best.file_path.name}")
        log_callback(f"  Size: {best.file_size_mb:.2f} MB")
        log_callback(f"  Compression Ratio: {best.compression_ratio:.2f}x")
        log_callback(f"  Peak RAM: {best.system_metrics.peak_ram_mb:.2f} MB")
        log_callback(f"  Avg CPU: {best.system_metrics.avg_process_cpu:.1f}%")
        log_callback(f"  I/O Total: {best.system_metrics.io_total_mb:.2f} MB")
        log_callback(f"  RAM/MB: {best.ram_per_mb:.2f} MB RAM per MB file")
        
        log_callback("\nWORST CASE:")
        log_callback(f"  File: {worst.file_path.name}")
        log_callback(f"  Size: {worst.file_size_mb:.2f} MB")
        log_callback(f"  Compression Ratio: {worst.compression_ratio:.2f}x")
        log_callback(f"  Peak RAM: {worst.system_metrics.peak_ram_mb:.2f} MB")
        log_callback(f"  Avg CPU: {worst.system_metrics.avg_process_cpu:.1f}%")
        log_callback(f"  I/O Total: {worst.system_metrics.io_total_mb:.2f} MB")
        log_callback(f"  RAM/MB: {worst.ram_per_mb:.2f} MB RAM per MB file")
        
        # Comparison
        ram_ratio = worst.system_metrics.peak_ram_mb / best.system_metrics.peak_ram_mb if best.system_metrics.peak_ram_mb > 0 else 0
        cpu_ratio = worst.system_metrics.avg_process_cpu / best.system_metrics.avg_process_cpu if best.system_metrics.avg_process_cpu > 0 else 0
        
        log_callback("\nCOMPARISON:")
        log_callback(f"  RAM Variation: {ram_ratio:.2f}x (worst uses {ram_ratio:.1f}x more RAM)")
        log_callback(f"  CPU Variation: {cpu_ratio:.2f}x (worst uses {cpu_ratio:.1f}x more CPU)")