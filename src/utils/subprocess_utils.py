"""
Subprocess utilities

Provides run_with_affinity(), a drop-in replacement for subprocess.run() that
pins the child process to the same CPU affinity mask as the current process.

Problem:
    subprocess.run() / Popen() on Windows does NOT inherit the parent's CPU affinity mask.
    Every new process receives the system-default affinity (all cores).
    This means CLI-based compressors (optipng.exe, oxipng.exe, cwebp.exe, etc.) 
    silently escape the affinity pin that BenchmarkRunner sets,
    making CPU measurements meaningless for those compressors:

        Parent pinned to core 1:   measured at ~95 % CPU  (correct)
        optipng.exe (all cores):   measured at  ~5 % CPU  (wrong, child uses N cores)

Solution:
    Launch the child process normally, then immediately call
    OpenProcess() + SetProcessAffinityMask() before the process has done any
    meaningful work. This approach:

        - Does NOT require CREATE_SUSPENDED or access to _thread_handle (a fragile
            CPython implementation detail that changed across Python versions).
        - Does NOT require administrator privileges.
        - Works on Python 3.8–3.13+.
        - Has a theoretical race window of a few microseconds between CreateProcess()
            returning and SetProcessAffinityMask() executing, but in practice the child
            spends its first milliseconds loading the C runtime and DLLs before doing
            any compression work.

Usage:

    from utils.subprocess_utils import run_with_affinity

    # Drop-in replacement for subprocess.run():
    result = run_with_affinity(cmd, capture_output=True, text=True)

    # Explicitly pass a mask (overrides the auto-detected parent affinity):
    result = run_with_affinity(cmd, affinity_mask=0b0010)  # core 1 only
"""

import ctypes
import ctypes.wintypes as wintypes
import logging
import os
import subprocess
from typing import List, Optional, Union

logger = logging.getLogger(__name__)


# Windows kernel32 setup

_kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]

_PROCESS_SET_INFORMATION   = 0x0200
_PROCESS_QUERY_INFORMATION = 0x0400

_kernel32.OpenProcess.restype  = wintypes.HANDLE
_kernel32.OpenProcess.argtypes = [
    wintypes.DWORD,  # dwDesiredAccess
    wintypes.BOOL,   # bInheritHandle
    wintypes.DWORD,  # dwProcessId
]

_kernel32.SetProcessAffinityMask.restype  = wintypes.BOOL
_kernel32.SetProcessAffinityMask.argtypes = [
    wintypes.HANDLE,  # hProcess
    ctypes.c_size_t,  # dwProcessAffinityMask (DWORD_PTR = size_t on 64-bit)
]

_kernel32.CloseHandle.restype  = wintypes.BOOL
_kernel32.CloseHandle.argtypes = [wintypes.HANDLE]

_CREATE_NO_WINDOW = 0x08000000


# Internal helpers

def _get_current_affinity_mask() -> Optional[int]:
    """
    Return the CPU affinity mask of the current process as an integer bitmask.

    Bit N = 1 means this process may run on logical core N.

    Returns:
        Integer bitmask, or None if psutil is unavailable or the call fails.
    """
    try:
        import psutil
        cores = psutil.Process(os.getpid()).cpu_affinity()
        mask  = 0
        for core in cores:
            mask |= (1 << core)
        return mask
    except Exception as exc:
        logger.debug("Could not read current process affinity: %s", exc)
        return None


def _apply_affinity_to_pid(pid: int, mask: int) -> bool:
    """
    Open a running process by PID and apply a CPU affinity mask.

    Args:
        pid: Target process ID.
        mask: CPU affinity bitmask (bit N = 1 → may run on core N).

    Returns:
        True on success, False on any failure (logged at WARNING level).
        The process handle is always closed, even on failure.
    """
    handle = _kernel32.OpenProcess(
        _PROCESS_SET_INFORMATION | _PROCESS_QUERY_INFORMATION,
        False,
        pid,
    )

    if not handle:
        err = ctypes.get_last_error()
        logger.debug(
            "OpenProcess failed for PID %d (error %d), "
            "child will run with default affinity.", pid, err
        )
        return False

    try:
        ok = _kernel32.SetProcessAffinityMask(handle, ctypes.c_size_t(mask))
        if not ok:
            err = ctypes.get_last_error()
            logger.warning(
                "SetProcessAffinityMask failed for PID %d (error %d), "
                "child will run with default affinity.", pid, err
            )
            return False

        logger.debug(
            "Child PID %d: affinity set to mask 0x%X (cores %s).",
            pid, mask, [i for i in range(64) if mask & (1 << i)],
        )
        return True

    finally:
        _kernel32.CloseHandle(handle)


# Public API

def run_with_affinity(
    cmd: Union[List[str], str],
    affinity_mask: Optional[int] = None,
    **subprocess_kwargs,
) -> subprocess.CompletedProcess:
    """
    Run a subprocess and immediately pin it to the specified CPU affinity mask.

    1. Launch the child process via Popen().
    2. Immediately call OpenProcess() + SetProcessAffinityMask() on the new PID.
    3. Wait for completion and return a CompletedProcess identical to what
        subprocess.run() would return.

    The window between process launch and affinity assignment is ~5 microseconds.
    The child spends its first ~1–5 ms loading the C runtime and DLLs before
    doing any compression work, so no computation occurs on the wrong core.

    Args:
        cmd: Command and arguments, passed directly to Popen().
        affinity_mask: Integer bitmask (bit N = 1: may run on core N).
            None (default) → copy the current process affinity automatically.
            If the current affinity cannot be read the child runs with the
            system default (all cores) and a debug message is logged.
        **subprocess_kwargs: All keyword arguments forwarded to subprocess.run()
            / Popen() unchanged: capture_output, text, timeout, cwd, env, etc.

    Returns:
        subprocess.CompletedProcess with returncode, stdout, and stderr.
    """
    if affinity_mask is None:
        affinity_mask = _get_current_affinity_mask()

    if affinity_mask is None:
        logger.debug("run_with_affinity: mask unknown, falling back to subprocess.run().")
        return subprocess.run(cmd, **subprocess_kwargs)

    timeout = subprocess_kwargs.pop("timeout", None)
    check   = subprocess_kwargs.pop("check",   False)

    if subprocess_kwargs.pop("capture_output", False):
        subprocess_kwargs.setdefault("stdout", subprocess.PIPE)
        subprocess_kwargs.setdefault("stderr", subprocess.PIPE)

    existing_flags = subprocess_kwargs.pop("creationflags", 0)
    subprocess_kwargs["creationflags"] = existing_flags | _CREATE_NO_WINDOW

    proc = subprocess.Popen(cmd, **subprocess_kwargs)

    _apply_affinity_to_pid(proc.pid, affinity_mask)

    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        stdout, stderr = proc.communicate()
        raise subprocess.TimeoutExpired(
            proc.args, timeout, output=stdout, stderr=stderr
        )
    except Exception:
        proc.kill()
        raise

    completed = subprocess.CompletedProcess(
        args=proc.args,
        returncode=proc.poll(),
        stdout=stdout,
        stderr=stderr,
    )

    if check and completed.returncode != 0:
        completed.check_returncode()

    return completed