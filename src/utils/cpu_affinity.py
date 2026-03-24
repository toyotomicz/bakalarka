"""
CPU affinity and process isolation

Handles pinning the benchmark process to a specific logical CPU core and
optionally raising its scheduling priority for reproducible timing.

Classes:
    IsolationConfig : What isolation the user wants (high priority + optional core pin).
    IsolationState  : Saved pre-isolation state.
    ProcessIsolator : Sets / restores affinity and priority; provides a CPU warmup.

Design notes:
    - This module has NO dependency on system_metrics.py or benchmark_shared.py.
        It is a pure utility that can be imported by any layer of the application.
    - CPU affinity is a Windows concept (SetThreadAffinityMask / SetProcessAffinityMask).
        psutil exposes it cross-platform where the OS supports it; on platforms where
        cpu_affinity() is unavailable the calls fail gracefully with a logged warning.
    - HIGH_PRIORITY_CLASS is used for priority elevation. REALTIME_PRIORITY_CLASS is
        deliberately avoided, it can starve system threads and destabilize the machine.
    - Core 0 is technically valid but is discouraged for benchmarking on Windows because
        it handles the majority of hardware IRQs, introducing timing jitter.

Usage::

    from utils.cpu_affinity import IsolationConfig, ProcessIsolator

    cfg = IsolationConfig(high_priority=True, cpu_core=1)
    isolator = ProcessIsolator(cfg)
    state = isolator.isolate()
    for note in state.isolation_notes:
        print(note)
    # ... run benchmark ...
    isolator.restore()
"""

import gc
import logging
import os
import time
from dataclasses import dataclass, field
from typing import List, Optional

import psutil

logger = logging.getLogger(__name__)


# Data structures

@dataclass
class IsolationConfig:
    """
    Describes what kind of process isolation to apply for a benchmark run.

    This is the single place where both isolation knobs live. Instead of
    scattering high_priority and cpu_core across multiple caller configs,
    callers build one IsolationConfig and hand it to ProcessIsolator.

    Attributes:
        high_priority: Raise the process to HIGH_PRIORITY_CLASS before measuring.
            Does NOT require administrator privileges on Windows.
        cpu_core: Pin the process to this logical core index (e.g. 1).
            None means no affinity change.
            Core 0 is valid but discouraged on Windows due to high IRQ load.
    """

    high_priority: bool          = False
    cpu_core:      Optional[int] = None

    @property
    def enabled(self) -> bool:
        """True when any isolation is requested."""
        return self.high_priority or self.cpu_core is not None


@dataclass
class IsolationState:
    """
    Process state captured before isolation so it can be restored afterwards.

    Attributes:
        affinity: Original CPU affinity (list of logical core IDs).
            None if unavailable or if querying affinity failed.

        nice: Original scheduling priority.
            On Unix systems this is the nice value (int).
            On Windows this is the process priority class (psutil constant).
            None if unavailable or if querying priority failed.

        isolated: True after isolate() has been successfully applied;
            set to False by restore().

        isolation_notes: Human readable log of actions taken (or skipped)
            during isolate() and restore(), suitable for debugging or UI display.

        pinned_cores: Logical core IDs that were actually applied for affinity.
            May differ from requested cores if invalid cores were filtered out.
            Empty when no pinning occurred.
    """

    affinity:        Optional[List[int]] = None
    nice:            Optional[int]       = None
    isolated:        bool                = False
    isolation_notes: List[str]           = field(default_factory=list)
    pinned_cores:    List[int]           = field(default_factory=list)

    @property
    def pinned_core_count(self) -> int:
        """Number of cores actually pinned. 0 when no affinity was set."""
        return len(self.pinned_cores)


# ProcessIsolator

class ProcessIsolator:
    """
    Pins the current process to a specific CPU core and / or raises its priority.

    Reducing OS scheduling noise during benchmarking improves timing
    reproducibility, particularly for short running compressors.

    Typical usage:
        cfg = IsolationConfig(high_priority=True, cpu_core=1)
        isolator = ProcessIsolator(cfg)
        state = isolator.isolate()
        for note in state.isolation_notes:
            log(note)
        try:
            run_benchmark()
        finally:
            isolator.restore()

    Thread safety:
        isolate() and restore() must be called from the same thread (the one whose
        affinity should be changed). The monitor thread in SystemMonitor is
        unaffected because affinity is set at the process level on Windows.

    Attributes:
        config: The IsolationConfig describing what isolation to apply.
        process: psutil.Process handle for the current process.
    """

    def __init__(self, config: IsolationConfig) -> None:
        """
        Initialise the isolator.

        Args:
            config: Describes what isolation to apply.
                Use IsolationConfig() (all defaults) for a no-op isolator.
        """
        self.config  = config
        self.process = psutil.Process(os.getpid())
        self._state  = IsolationState()

    # Public API

    def isolate(self) -> IsolationState:
        """
        Apply isolation settings and return the resulting IsolationState.

        When config.enabled is False this is a no-op and isolation_notes will say so. 
        Otherwise the sequence is:
            1) Snapshot current affinity and priority.
            2) Pin to the requested CPU core (if specified).
            3) Raise to HIGH_PRIORITY_CLASS (if requested).
            4) Run a short CPU warmup to exit power-saving states.

        Returns:
            IsolationState with notes describing every action taken.
        """
        if not self.config.enabled:
            self._state.isolated = False
            self._state.isolation_notes = ["Process isolation disabled."]
            return self._state

        self._save_state()

        if self.config.cpu_core is not None:
            self._set_affinity([self.config.cpu_core])

        if self.config.high_priority:
            self._set_high_priority()

        self._warmup()

        self._state.isolated = True
        return self._state

    def restore(self) -> bool:
        """
        Restore the process to its pre-isolation affinity and priority.

        Safe to call even if isolate() was never called or isolation was disabled;
        it is a no-op in those cases.

        Returns:
            True if all attributes were restored successfully, False if any
            restore call raised an exception (details are logged at DEBUG level).
        """
        if not self._state.isolated:
            return True

        success = True

        if self._state.affinity is not None:
            try:
                self.process.cpu_affinity(self._state.affinity)
                logger.debug("CPU affinity restored to %s.", self._state.affinity)
            except (AttributeError, psutil.Error, OSError) as exc:
                logger.debug("Could not restore CPU affinity: %s", exc)
                success = False

        if self._state.nice is not None:
            try:
                self.process.nice(self._state.nice)
                logger.debug("Process priority restored to %s.", self._state.nice)
            except (psutil.Error, OSError) as exc:
                logger.debug("Could not restore process priority: %s", exc)
                success = False

        self._state.isolated     = False
        self._state.pinned_cores = []
        return success

    @staticmethod
    def get_available_cores() -> List[int]:
        """
        Return the logical CPU core IDs currently visible to this process.

        This reflects the process affinity mask set by the OS or the parent process.

        Returns:
            List of zero-based core indices.
        """
        try:
            return psutil.Process(os.getpid()).cpu_affinity()
        except (AttributeError, psutil.Error):
            count = psutil.cpu_count(logical=True)
            return list(range(count)) if count else [0]

    # Private helpers

    def _save_state(self) -> None:
        """Snapshot the current affinity mask and nice level for later restoration."""
        try:
            self._state.affinity = self.process.cpu_affinity()
        except (AttributeError, psutil.Error, OSError):
            self._state.affinity = None

        try:
            self._state.nice = self.process.nice()
        except (psutil.Error, OSError):
            self._state.nice = None

    def _set_affinity(self, cpu_cores: List[int]) -> None:
        """
        Pin the process to the requested CPU cores.

        Cores not in get_available_cores() are silently dropped. If none of
        the requested cores are available the affinity is left unchanged and a
        note is appended to IsolationState.isolation_notes.

        Args:
            cpu_cores: List of logical core indices to pin to.
        """
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
                skipped = [c for c in cpu_cores if c not in available]
                self._state.isolation_notes.append(
                    f"CPU affinity: cores {skipped} unavailable; using {valid}."
                )

            self.process.cpu_affinity(valid)
            self._state.pinned_cores = valid
            self._state.isolation_notes.append(
                f"CPU affinity: pinned to cores {valid}."
            )
            logger.debug("CPU affinity set to %s.", valid)

        except (AttributeError, psutil.Error, OSError) as exc:
            self._state.isolation_notes.append(
                f"CPU affinity: cannot set ({exc})."
            )
            logger.debug("Failed to set CPU affinity: %s", exc)

    def _set_high_priority(self) -> None:
        """
        Raise process priority to HIGH_PRIORITY_CLASS on Windows.

        HIGH_PRIORITY_CLASS does not require administrator privileges.
        REALTIME_PRIORITY_CLASS is deliberately avoided, it can prevent the OS
        scheduler from servicing system threads and destabilize the machine.
        Falls back gracefully on non-Windows platforms or permission errors.
        """
        try:
            self.process.nice(psutil.HIGH_PRIORITY_CLASS)
            self._state.isolation_notes.append("Priority: HIGH_PRIORITY_CLASS.")
            logger.debug("Process priority set to HIGH_PRIORITY_CLASS.")
        except (PermissionError, psutil.AccessDenied, psutil.Error, OSError) as exc:
            self._state.isolation_notes.append(
                f"Priority: could not raise to HIGH_PRIORITY_CLASS ({exc}). "
                "Check UAC settings or run as administrator."
            )
            logger.debug("Failed to set HIGH_PRIORITY_CLASS: %s", exc)

    @staticmethod
    def _warmup() -> None:
        """
        Short warm-up sequence to prepare the CPU for accurate timing.

        1) Force a full GC cycle so the garbage collector does not interrupt
            timing on the first benchmark run.
        2) 50 ms busy-wait to pull the CPU out of a low-power / frequency-scaled
            idle state, preventing turbo-boost ramp-up artifacts on the first run.
        """
        gc.collect()
        deadline = time.perf_counter() + 0.05
        _x = 0
        while time.perf_counter() < deadline:
            _x += 1  # keeps the loop body non-empty, value is intentionally discarded