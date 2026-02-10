"""
System Resource Monitoring Module - ENHANCED VERSION
Monitors CPU, RAM, and I/O metrics during compression benchmarks

FEATURES:
- Adaptive sampling for small/large files
- Pre/post snapshots for ultra-fast operations
- Quality indicators (is_reliable, reliability_score, measurement_quality)
- ScenarioAnalyzer included
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
    cpu_percent: float
    process_cpu_percent: float
    ram_used_mb: float
    ram_available_mb: float
    io_read_mb: float
    io_write_mb: float


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
    
    # I/O metrics
    total_io_read_mb: float
    total_io_write_mb: float
    
    # Timing
    duration_seconds: float
    sample_count: int
    
    # Quality indicators (ENHANCED)
    is_reliable: bool = True
    reliability_score: float = 1.0
    measurement_quality: str = "good"
    
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
    Enhanced system monitor with adaptive sampling
    
    KEY FEATURES:
    - Adaptive sampling: adjusts rate based on file size
    - Pre/post snapshots: captures start/end even if too fast
    - Quality metrics: warns about unreliable measurements
    """
    
    # Adaptive sampling rates
    ULTRA_FAST_INTERVAL = 0.001   # 1ms for < 10KB files
    FAST_INTERVAL = 0.01          # 10ms for < 100KB
    NORMAL_INTERVAL = 0.05        # 50ms for < 1MB
    SLOW_INTERVAL = 0.1           # 100ms for > 1MB
    
    # Quality thresholds
    MIN_SAMPLES_RELIABLE = 5
    MIN_SAMPLES_FAIR = 3
    MIN_DURATION_RELIABLE = 0.01  # 10ms
    
    def __init__(self, 
                 sampling_interval: float = 0.05,
                 adaptive: bool = True,
                 force_pre_post: bool = True):
        """
        Args:
            sampling_interval: Base sampling interval (seconds)
            adaptive: Enable adaptive sampling based on file size
            force_pre_post: Always capture pre/post snapshots
        """
        self.base_sampling_interval = sampling_interval
        self.sampling_interval = sampling_interval
        self.adaptive = adaptive
        self.force_pre_post = force_pre_post
        
        self.process = psutil.Process(os.getpid())
        # Inicializační volání pro korektní baseline
        self.process.cpu_percent(interval=None)
        psutil.cpu_percent(interval=None)
        
        self.snapshots: List[SystemSnapshot] = []
        self.is_monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        
        # Baseline measurements
        self.baseline_io_read = 0
        self.baseline_io_write = 0
        self.start_time = 0
        self.end_time = 0
        
        # Pre/post snapshots
        self.pre_snapshot: Optional[SystemSnapshot] = None
        self.post_snapshot: Optional[SystemSnapshot] = None
    
    def _estimate_duration(self, file_size_bytes: int) -> float:
        """Estimate operation duration based on file size"""
        if file_size_bytes < 10_000:  # < 10 KB
            return 0.001
        elif file_size_bytes < 100_000:  # < 100 KB
            return 0.01
        elif file_size_bytes < 1_000_000:  # < 1 MB
            return 0.1
        else:
            return file_size_bytes / (50 * 1024 * 1024)
    
    def _set_adaptive_sampling(self, file_size_bytes: int):
        """Adjust sampling interval based on file size"""
        if not self.adaptive:
            return
        
        estimated_duration = self._estimate_duration(file_size_bytes)
        
        if estimated_duration < 0.01:
            self.sampling_interval = self.ULTRA_FAST_INTERVAL
        elif estimated_duration < 0.1:
            self.sampling_interval = self.FAST_INTERVAL
        elif estimated_duration < 1.0:
            self.sampling_interval = self.NORMAL_INTERVAL
        else:
            self.sampling_interval = self.SLOW_INTERVAL
    
    def start(self, file_size_bytes: Optional[int] = None):
        """Start monitoring"""
        if self.is_monitoring:
            return
        
        # Adaptive sampling
        if file_size_bytes and self.adaptive:
            self._set_adaptive_sampling(file_size_bytes)
        
        self.snapshots = []
        self.is_monitoring = True
        self.start_time = time.time()
        
        # Pre-snapshot
        if self.force_pre_post:
            self.pre_snapshot = self._take_snapshot()
        
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
        """Stop monitoring and return metrics"""
        self.is_monitoring = False
        self.end_time = time.time()
        
        # Post-snapshot
        if self.force_pre_post:
            self.post_snapshot = self._take_snapshot()
        
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
                pass
            
            time.sleep(self.sampling_interval)
    
    def _take_snapshot(self) -> SystemSnapshot:
        """Capture current system state"""
        cpu_percent = psutil.cpu_percent(interval=0)
        process_cpu = self.process.cpu_percent(interval=0)
        
        mem_info = self.process.memory_info()
        ram_used = mem_info.rss / (1024 * 1024)
        
        virtual_mem = psutil.virtual_memory()
        ram_available = virtual_mem.available / (1024 * 1024)
        
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
    
    def _assess_quality(self, duration: float, sample_count: int) -> tuple:
        """Assess measurement quality"""
        # Duration-based reliability
        if duration < 0.001:
            return False, 0.0, "unreliable"
        elif duration < self.MIN_DURATION_RELIABLE:
            duration_score = duration / self.MIN_DURATION_RELIABLE
        else:
            duration_score = 1.0
        
        # Sample-based reliability
        if sample_count == 0:
            sample_score = 0.0
        elif sample_count < self.MIN_SAMPLES_FAIR:
            sample_score = sample_count / self.MIN_SAMPLES_FAIR * 0.5
        elif sample_count < self.MIN_SAMPLES_RELIABLE:
            sample_score = 0.5 + (sample_count - self.MIN_SAMPLES_FAIR) / \
                          (self.MIN_SAMPLES_RELIABLE - self.MIN_SAMPLES_FAIR) * 0.3
        else:
            sample_score = 0.8 + min(sample_count / 20, 0.2)
        
        # Combined score
        reliability_score = (duration_score * 0.4 + sample_score * 0.6)
        
        # Categorize
        if reliability_score >= 0.8:
            quality = "good"
            is_reliable = True
        elif reliability_score >= 0.5:
            quality = "fair"
            is_reliable = True
        elif reliability_score >= 0.3:
            quality = "poor"
            is_reliable = False
        else:
            quality = "unreliable"
            is_reliable = False
        
        return is_reliable, reliability_score, quality
    
    def _calculate_metrics(self) -> SystemMetrics:
        """Calculate aggregated metrics with quality assessment"""
        duration = self.end_time - self.start_time
        
        # Combine all snapshots
        all_snapshots = self.snapshots.copy()
        if self.pre_snapshot:
            all_snapshots.insert(0, self.pre_snapshot)
        if self.post_snapshot:
            all_snapshots.append(self.post_snapshot)
        
        if not all_snapshots:
            is_reliable, score, quality = self._assess_quality(duration, 0)
            return SystemMetrics(
                avg_cpu_percent=0, max_cpu_percent=0,
                avg_process_cpu=0, max_process_cpu=0,
                avg_ram_mb=0, max_ram_mb=0,
                total_io_read_mb=0, total_io_write_mb=0,
                duration_seconds=duration, sample_count=0,
                is_reliable=is_reliable,
                reliability_score=score,
                measurement_quality=quality
            )
        
        # CPU metrics
        cpu_percents = [s.cpu_percent for s in all_snapshots]
        process_cpus = [s.process_cpu_percent for s in all_snapshots]
        
        avg_cpu = sum(cpu_percents) / len(cpu_percents)
        max_cpu = max(cpu_percents)
        avg_process_cpu = sum(process_cpus) / len(process_cpus)
        max_process_cpu = max(process_cpus)
        
        # Memory metrics
        ram_values = [s.ram_used_mb for s in all_snapshots]
        avg_ram = sum(ram_values) / len(ram_values)
        max_ram = max(ram_values)
        
        # I/O metrics
        if self.post_snapshot:
            io_read_delta = self.post_snapshot.io_read_mb - self.baseline_io_read
            io_write_delta = self.post_snapshot.io_write_mb - self.baseline_io_write
        elif all_snapshots:
            last_snapshot = all_snapshots[-1]
            io_read_delta = last_snapshot.io_read_mb - self.baseline_io_read
            io_write_delta = last_snapshot.io_write_mb - self.baseline_io_write
        else:
            io_read_delta = 0
            io_write_delta = 0
        
        # Quality assessment
        is_reliable, score, quality = self._assess_quality(duration, len(all_snapshots))
        
        return SystemMetrics(
            avg_cpu_percent=avg_cpu,
            max_cpu_percent=max_cpu,
            avg_process_cpu=avg_process_cpu,
            max_process_cpu=max_process_cpu,
            avg_ram_mb=avg_ram,
            max_ram_mb=max_ram,
            total_io_read_mb=max(0, io_read_delta),
            total_io_write_mb=max(0, io_write_delta),
            duration_seconds=duration,
            sample_count=len(all_snapshots),
            is_reliable=is_reliable,
            reliability_score=score,
            measurement_quality=quality
        )


class ProcessIsolator:
    """Process isolation for accurate benchmarking"""
    
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.original_affinity = None
        self.original_priority = None
    
    def isolate(self, cpu_cores: Optional[List[int]] = None, high_priority: bool = True):
        """Isolate process for benchmarking"""
        try:
            self.original_affinity = self.process.cpu_affinity()
            self.original_priority = self.process.nice()
            
            if cpu_cores is not None and hasattr(self.process, 'cpu_affinity'):
                self.process.cpu_affinity(cpu_cores)
            
            if high_priority:
                if os.name == 'nt':
                    self.process.nice(psutil.HIGH_PRIORITY_CLASS)
                else:
                    self.process.nice(-10)
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
    scenario_type: str
    file_path: Path
    file_size_mb: float
    compression_ratio: float
    system_metrics: SystemMetrics
    
    @property
    def ram_per_mb(self) -> float:
        if self.file_size_mb == 0:
            return 0.0
        return self.system_metrics.max_ram_mb / self.file_size_mb
    
    @property
    def cpu_efficiency(self) -> float:
        if self.file_size_mb == 0:
            return 0.0
        cpu_time = (self.system_metrics.avg_process_cpu / 100.0) * self.system_metrics.duration_seconds
        return cpu_time / self.file_size_mb


class ScenarioAnalyzer:
    """Analyzes benchmark results to identify best/worst case scenarios"""
    
    @staticmethod
    def identify_scenarios(results: List, metric: str = "compression_ratio") -> dict:
        if not results:
            return {"best": None, "worst": None}
        
        valid_results = [r for r in results if hasattr(r, 'system_metrics') and r.system_metrics]
        
        if not valid_results:
            return {"best": None, "worst": None}
        
        if metric == "compression_ratio":
            sorted_results = sorted(valid_results, 
                                   key=lambda r: r.metrics.compression_ratio, 
                                   reverse=True)
        elif metric == "ram_usage":
            sorted_results = sorted(valid_results,
                                   key=lambda r: r.system_metrics.max_ram_mb)
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
        log_callback(f"  Max RAM: {best.system_metrics.max_ram_mb:.2f} MB")
        log_callback(f"  Avg CPU: {best.system_metrics.avg_process_cpu:.1f}%")
        log_callback(f"  I/O Total: {best.system_metrics.io_total_mb:.2f} MB")
        
        log_callback("\nWORST CASE:")
        log_callback(f"  File: {worst.file_path.name}")
        log_callback(f"  Size: {worst.file_size_mb:.2f} MB")
        log_callback(f"  Compression Ratio: {worst.compression_ratio:.2f}x")
        log_callback(f"  Max RAM: {worst.system_metrics.max_ram_mb:.2f} MB")
        log_callback(f"  Avg CPU: {worst.system_metrics.avg_process_cpu:.1f}%")
        log_callback(f"  I/O Total: {worst.system_metrics.io_total_mb:.2f} MB")
        
        ram_ratio = worst.system_metrics.max_ram_mb / best.system_metrics.max_ram_mb if best.system_metrics.max_ram_mb > 0 else 0
        cpu_ratio = worst.system_metrics.avg_process_cpu / best.system_metrics.avg_process_cpu if best.system_metrics.avg_process_cpu > 0 else 0
        
        log_callback("\nCOMPARISON:")
        log_callback(f"  RAM Variation: {ram_ratio:.2f}x")
        log_callback(f"  CPU Variation: {cpu_ratio:.2f}x")