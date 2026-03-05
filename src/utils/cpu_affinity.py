"""
CPU Affinity and Process Isolation
utils/cpu_affinity.py

Handles pinning the benchmark process to a specific logical CPU core and
optionally raising its scheduling priority for reproducible timing.

Classes:
  IsolationConfig : What isolation the user wants (high priority + optional core pin).
  IsolationState  : Saved pre-isolation state (affinity mask + nice level).
  ProcessIsolator : Sets / restores affinity and priority; provides warmup.

Design notes:
  - This module has NO dependency on system_metrics.py or benchmark_shared.py.
    It is a pure utility that can be imported by any layer of the application.
  - CPU affinity is a Windows concept (SetThreadAffinityMask / SetProcessAffinityMask).
    psutil exposes it cross-platform where the OS supports it; on platforms where
    cpu_affinity() is unavailable the calls fail gracefully with a logged warning.
  - HIGH_PRIORITY_CLASS is used for priority elevation.  REALTIME_PRIORITY_CLASS is
    deliberately avoided — it can starve system threads and destabilise the machine.
  - Core 0 is technically valid but is discouraged for benchmarking on Windows because
    it handles the majority of hardware IRQs, introducing timing jitter.

Usage:
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


# ============================================================================
# Data structures
# ============================================================================

@dataclass
class IsolationConfig:
    """
    What kind of isolation the user wants for a benchmark run.

    This is the single place where both isolation knobs live — instead of
    scattering isolate_process and cpu_affinity_core across multiple
    caller configs, callers build one IsolationConfig and hand it to
    ProcessIsolator.

    high_priority : Raise the process to HIGH_PRIORITY_CLASS before measuring.
                    Does NOT require administrator privileges on Windows.
    cpu_core      : Pin the process to this logical core index (e.g. 1).
                    None means no affinity change.
                    Core 0 is valid but discouraged on Windows (high IRQ load).
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
    Process state captured before isolation so it can be fully restored afterwards.

    affinity        : Original CPU affinity mask (list of core IDs).
                      None when the platform does not support cpu_affinity().
    nice            : Original scheduling priority / nice value.
                      None when the platform does not support nice().
    isolated        : True while isolation is active; set to False by restore().
    isolation_notes : Human-readable log of every action taken (or skipped) during
                      isolate() and restore(), suitable for display in a GUI or log.
    pinned_cores    : The core IDs that were actually pinned (subset of the requested
                      cores that were available).  Empty list when no pinning occurred.
    """
    affinity:        Optional[List[int]] = None
    nice:            Optional[int]       = None
    isolated:        bool                = False
    isolation_notes: List[str]           = field(default_factory=list)
    pinned_cores:    List[int]           = field(default_factory=list)

    @property
    def pinned_core_count(self) -> int:
        """Number of cores actually pinned.  0 when no affinity was set."""
        return len(self.pinned_cores)


# ============================================================================
# ProcessIsolator
# ============================================================================

class ProcessIsolator:
    """
    Pins the current process to a specific CPU core and / or raises its
    scheduling priority to reduce measurement noise during benchmarking.

    Typical usage
    -------------
        cfg = IsolationConfig(high_priority=True, cpu_core=1)
        isolator = ProcessIsolator(cfg)
        state = isolator.isolate()
        for note in state.isolation_notes:
            log(note)
        try:
            run_benchmark()
        finally:
            isolator.restore()

    Thread safety
    -------------
    isolate() and restore() must be called from the same thread (the one whose
    affinity should be changed).  The monitor thread in SystemMonitor is unaffected
    because affinity is set at the process level on Windows.
    """

    def __init__(self, config: IsolationConfig) -> None:
        """
        Args:
            config: Describes what isolation to apply.
                    Use IsolationConfig() (all defaults) for a no-op isolator.
        """
        self.config  = config
        self.process = psutil.Process(os.getpid())
        self._state  = IsolationState()

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def isolate(self) -> IsolationState:
        """
        Apply the isolation settings from self.config and return IsolationState.

        When config.enabled is False this is a no-op — isolation_notes will say so.

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

        Safe to call even if isolate() was never called or if isolation was
        disabled — it is a no-op in those cases.

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
        Return the list of logical CPU core IDs currently visible to this process.

        On Windows this reflects the process affinity mask set by the OS or the
        parent process.  Falls back to all logical cores when cpu_affinity() is
        unavailable (Linux without CAP_SYS_NICE, some container environments).
        """
        try:
            return psutil.Process(os.getpid()).cpu_affinity()
        except (AttributeError, psutil.Error):
            count = psutil.cpu_count(logical=True)
            return list(range(count)) if count else [0]

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

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

        Cores that are not in get_available_cores() are silently dropped.
        If none of the requested cores are available the affinity is left
        unchanged and a note is appended to IsolationState.isolation_notes.
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
        REALTIME_PRIORITY_CLASS is deliberately not used — it can prevent
        the OS scheduler from servicing system threads and destabilise the machine.
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
        Short warm-up sequence before the first measurement:

        1. Force a full GC cycle so the garbage collector does not interrupt
           timing during the benchmark run.
        2. 50 ms busy-wait to pull the CPU out of a low-power / frequency-scaled
           idle state, preventing turbo-boost ramp-up artefacts on the first run.
        """
        gc.collect()
        deadline = time.perf_counter() + 0.05
        _x = 0
        while time.perf_counter() < deadline:
            _x += 1