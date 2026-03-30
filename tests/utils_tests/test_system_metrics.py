"""
Tests for utils/system_metrics.py.

Covers SystemMetrics computed properties, SystemMonitor start/stop with
adaptive sampling, ScenarioMetrics derived values, and
ScenarioAnalyzer.identify_scenarios().
"""

import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from utils.system_metrics import (
    ScenarioAnalyzer,
    ScenarioMetrics,
    SystemMetrics,
    SystemMonitor,
    SystemSnapshot,
)


# Helpers

def _make_metrics(
    avg_process_cpu: float = 100.0,
    max_process_cpu: float = 150.0,
    avg_ram_mb: float = 500.0,
    peak_ram_mb: float = 600.0,
    ram_baseline_mb: float = 300.0,
    total_io_read_mb: float = 10.0,
    total_io_write_mb: float = 5.0,
    duration_seconds: float = 1.0,
    sample_count: int = 10,
    logical_core_count: int = 4,
    is_reliable: bool = True,
    reliability_score: float = 1.0,
    measurement_quality: str = "good",
) -> SystemMetrics:
    """
    Construct a SystemMetrics instance with sensible defaults.

    Args:
        avg_process_cpu: Average per-core CPU usage reported by psutil.
        max_process_cpu: Peak per-core CPU usage reported by psutil.
        avg_ram_mb: Average RSS in MB during the measurement window.
        peak_ram_mb: Maximum RSS in MB during the measurement window.
        ram_baseline_mb: RSS in MB before the operation started.
        total_io_read_mb: Total bytes read, converted to MB.
        total_io_write_mb: Total bytes written, converted to MB.
        duration_seconds: Wall-clock length of the measurement window.
        sample_count: Number of samples collected.
        logical_core_count: Logical CPU cores visible to the process.
        is_reliable: Whether the measurement met reliability thresholds.
        reliability_score: Numeric reliability score in [0, 1].
        measurement_quality: Human-readable quality label.

    Returns:
        A fully populated SystemMetrics dataclass instance.
    """
    return SystemMetrics(
        avg_cpu_percent=50.0,
        max_cpu_percent=80.0,
        avg_process_cpu=avg_process_cpu,
        max_process_cpu=max_process_cpu,
        avg_ram_mb=avg_ram_mb,
        peak_ram_mb=peak_ram_mb,
        ram_baseline_mb=ram_baseline_mb,
        total_io_read_mb=total_io_read_mb,
        total_io_write_mb=total_io_write_mb,
        duration_seconds=duration_seconds,
        sample_count=sample_count,
        logical_core_count=logical_core_count,
        is_reliable=is_reliable,
        reliability_score=reliability_score,
        measurement_quality=measurement_quality,
    )


# SystemMetrics - computed properties

class TestSystemMetrics:
    """Verify the computed properties of SystemMetrics."""

    def test_io_total_mb(self):
        m = _make_metrics(total_io_read_mb=10.0, total_io_write_mb=5.0)
        assert m.io_total_mb == pytest.approx(15.0)

    def test_net_peak_ram_mb(self):
        m = _make_metrics(peak_ram_mb=600.0, ram_baseline_mb=300.0)
        assert m.net_peak_ram_mb == pytest.approx(300.0)

    def test_net_peak_ram_mb_clamps_to_zero_when_baseline_exceeds_peak(self):
        m = _make_metrics(peak_ram_mb=200.0, ram_baseline_mb=300.0)
        assert m.net_peak_ram_mb == 0.0

    def test_net_avg_ram_mb(self):
        m = _make_metrics(avg_ram_mb=400.0, ram_baseline_mb=300.0)
        assert m.net_avg_ram_mb == pytest.approx(100.0)

    def test_cpu_percent_normalized_basic(self):
        """100% across 4 cores normalises to 25%."""
        m = _make_metrics(avg_process_cpu=100.0, logical_core_count=4)
        assert m.cpu_percent_normalized == pytest.approx(25.0)

    def test_cpu_percent_normalized_clamps_to_100(self):
        """psutil may report values above the theoretical maximum; they must clamp to 100."""
        m = _make_metrics(avg_process_cpu=99999.0, logical_core_count=1)
        assert m.cpu_percent_normalized == pytest.approx(100.0)

    def test_cpu_percent_normalized_returns_zero_for_zero_cores(self):
        m = _make_metrics(avg_process_cpu=100.0, logical_core_count=0)
        assert m.cpu_percent_normalized == 0.0

    def test_peak_cpu_percent_normalized(self):
        m = _make_metrics(max_process_cpu=200.0, logical_core_count=4)
        assert m.peak_cpu_percent_normalized == pytest.approx(50.0)

    def test_ram_efficiency_mb_per_sec(self):
        m = _make_metrics(avg_ram_mb=600.0, duration_seconds=2.0)
        assert m.ram_efficiency_mb_per_sec == pytest.approx(300.0)

    def test_ram_efficiency_returns_zero_for_zero_duration(self):
        m = _make_metrics(avg_ram_mb=600.0, duration_seconds=0.0)
        assert m.ram_efficiency_mb_per_sec == 0.0


# SystemMonitor - start / stop

def _mock_io():
    """
    Return a fake io_counters() result that avoids /proc/<pid>/io issues.

    Returns:
        MagicMock with read_bytes and write_bytes set to zero.
    """
    io = MagicMock()
    io.read_bytes = 0
    io.write_bytes = 0
    return io


class TestSystemMonitor:
    """
    Test the monitor using a real background thread for less than 200 ms.

    io_counters() is mocked because the sandbox exposes a non-standard
    /proc/<pid>/io layout.
    """

    def _patched_monitor(self) -> SystemMonitor:
        """Create a SystemMonitor with io_counters mocked out.

        Returns:
            SystemMonitor instance safe to run in the test sandbox.
        """
        monitor = SystemMonitor()
        monitor.process.io_counters = MagicMock(return_value=_mock_io())
        return monitor

    def test_start_and_stop_does_not_raise(self):
        monitor = self._patched_monitor()
        monitor.start(file_size_bytes=1024)
        time.sleep(0.05)
        monitor.stop()

    def test_stop_returns_system_metrics(self):
        monitor = self._patched_monitor()
        monitor.start(file_size_bytes=1024)
        time.sleep(0.05)
        metrics = monitor.stop()
        assert isinstance(metrics, SystemMetrics)

    def test_metrics_duration_is_positive(self):
        monitor = self._patched_monitor()
        monitor.start(file_size_bytes=1024)
        time.sleep(0.05)
        metrics = monitor.stop()
        assert metrics.duration_seconds > 0.0

    def test_stop_without_sleep_returns_system_metrics(self):
        """stop() called immediately after start() must still return valid metrics."""
        monitor = self._patched_monitor()
        monitor.start(file_size_bytes=512)
        metrics = monitor.stop()
        assert isinstance(metrics, SystemMetrics)

    def test_adaptive_sampling_uses_short_interval_for_small_files(self):
        monitor = SystemMonitor()
        monitor._set_adaptive_sampling(file_size_bytes=100)
        assert monitor.sampling_interval <= SystemMonitor.FAST_INTERVAL

    def test_adaptive_sampling_uses_normal_interval_for_large_files(self):
        monitor = SystemMonitor()
        monitor._set_adaptive_sampling(file_size_bytes=10 * 1024 * 1024)
        assert monitor.sampling_interval >= SystemMonitor.NORMAL_INTERVAL

    def test_second_start_call_is_noop(self):
        """A second call to start() while already running must not spawn a new thread."""
        monitor = self._patched_monitor()
        monitor.start(file_size_bytes=1024)
        thread_before = monitor.monitor_thread
        monitor.start(file_size_bytes=1024)
        assert monitor.monitor_thread is thread_before
        monitor.stop()


# ScenarioMetrics - computed properties

class TestScenarioMetrics:
    """Verify ram_per_mb and cpu_efficiency derived properties of ScenarioMetrics."""

    def _make_scenario(
        self,
        peak_ram_mb: float = 600.0,
        file_size_mb: float = 2.0,
        avg_process_cpu: float = 100.0,
        duration: float = 1.0,
    ) -> ScenarioMetrics:
        """
        Build a ScenarioMetrics instance with the given parameters.

        Args:
            peak_ram_mb: Peak RSS during compression.
            file_size_mb: Size of the source file in MB.
            avg_process_cpu: Average CPU usage during compression.
            duration: Wall-clock duration of the compression in seconds.

        Returns:
            A ScenarioMetrics instance backed by a real SystemMetrics object.
        """
        return ScenarioMetrics(
            scenario_type="best",
            file_path=Path("dummy.png"),
            file_size_mb=file_size_mb,
            compression_ratio=2.0,
            system_metrics=_make_metrics(
                peak_ram_mb=peak_ram_mb,
                avg_process_cpu=avg_process_cpu,
                duration_seconds=duration,
            ),
        )

    def test_ram_per_mb(self):
        s = self._make_scenario(peak_ram_mb=600.0, file_size_mb=3.0)
        assert s.ram_per_mb == pytest.approx(200.0)

    def test_ram_per_mb_returns_zero_for_zero_file_size(self):
        s = self._make_scenario(file_size_mb=0.0)
        assert s.ram_per_mb == 0.0

    def test_cpu_efficiency(self):
        """cpu_time = (100 / 100) * 2.0 = 2.0 s, so efficiency = 2.0 s / 2.0 MB = 1.0."""
        s = self._make_scenario(avg_process_cpu=100.0, duration=2.0, file_size_mb=2.0)
        assert s.cpu_efficiency == pytest.approx(1.0)

    def test_cpu_efficiency_returns_zero_for_zero_file_size(self):
        s = self._make_scenario(file_size_mb=0.0)
        assert s.cpu_efficiency == 0.0


# ScenarioAnalyzer - identify_scenarios()

def _make_result_mock(compression_ratio: float, peak_ram_mb: float = 300.0) -> MagicMock:
    """
    Create a benchmark result mock compatible with ScenarioAnalyzer.

    Args:
        compression_ratio: Compression ratio to assign to the result.
        peak_ram_mb: Peak RAM usage in MB to embed in the system metrics.

    Returns:
        MagicMock whose attributes mirror the real benchmark result structure.
    """
    result = MagicMock()
    result.image_path = Path(f"image_{compression_ratio}.png")
    result.metrics.success = True
    result.metrics.compression_ratio = compression_ratio
    result.metrics.original_size = 2 * 1024 * 1024  # 2 MB
    result.system_metrics = _make_metrics(
        peak_ram_mb=peak_ram_mb,
        avg_process_cpu=100.0,
        total_io_read_mb=10.0,
        total_io_write_mb=5.0,
    )
    return result


class TestScenarioAnalyzer:
    """Verify identify_scenarios() ranking logic and edge-case handling."""

    def test_identifies_best_and_worst_by_compression_ratio(self):
        results = [
            _make_result_mock(compression_ratio=1.5),
            _make_result_mock(compression_ratio=3.0),
            _make_result_mock(compression_ratio=2.0),
        ]
        scenarios = ScenarioAnalyzer.identify_scenarios(results, "compression_ratio")

        assert scenarios["best"].compression_ratio == pytest.approx(3.0)
        assert scenarios["worst"].compression_ratio == pytest.approx(1.5)

    def test_returns_none_for_fewer_than_two_results(self):
        results = [_make_result_mock(compression_ratio=2.0)]
        scenarios = ScenarioAnalyzer.identify_scenarios(results, "compression_ratio")
        assert scenarios["best"] is None
        assert scenarios["worst"] is None

    def test_returns_none_for_empty_results(self):
        scenarios = ScenarioAnalyzer.identify_scenarios([], "compression_ratio")
        assert scenarios["best"] is None

    def test_unknown_metric_returns_none(self):
        results = [_make_result_mock(2.0), _make_result_mock(3.0)]
        scenarios = ScenarioAnalyzer.identify_scenarios(results, "nonexistent_metric")
        assert scenarios["best"] is None

    def test_ram_usage_metric_lower_is_better(self):
        """For RAM usage, the result with lower peak RAM must be ranked best."""
        results = [
            _make_result_mock(compression_ratio=2.0, peak_ram_mb=100.0),
            _make_result_mock(compression_ratio=2.0, peak_ram_mb=500.0),
        ]
        scenarios = ScenarioAnalyzer.identify_scenarios(results, "ram_usage")

        assert scenarios["best"].system_metrics.peak_ram_mb == pytest.approx(100.0)
        assert scenarios["worst"].system_metrics.peak_ram_mb == pytest.approx(500.0)

    def test_failed_results_are_excluded(self):
        """A failed result must be excluded, leaving only one success, so both slots are None."""
        good = _make_result_mock(compression_ratio=2.0)
        bad = _make_result_mock(compression_ratio=5.0)
        bad.metrics.success = False

        scenarios = ScenarioAnalyzer.identify_scenarios([good, bad], "compression_ratio")
        assert scenarios["best"] is None