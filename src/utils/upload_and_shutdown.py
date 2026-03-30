"""
Upload benchmark JSONs to filebin.net and optionally shut down the PC

Provides a single public entry point upload_and_maybe_shutdown() that runs
the entire upload + shutdown sequence in a background daemon thread so that
the GUI remains responsive during the transfer.
"""

import subprocess
import sys
import threading
from pathlib import Path
from typing import Callable, List


# Upload single file

def upload_filebin(
    json_path: Path,
    bin_name: str,
    log: Callable[[str], None],
) -> bool:
    """
    Upload one file to filebin.net/<bin_name>.

    Args:
        json_path: Local path of the JSON file to upload.
        bin_name: Name of the filebin.net shared bin to upload into.
        log: Callable that accepts one log line at a time.

    Returns:
        True on a successful HTTP 200 or 201 response, False on any error.
    """

    # Lazy import to avoid making requests a hard dependency for the whole program
    try:
        import requests 
    except ImportError:
        log("ERROR: requests is not installed")
        return False
    
    file_url = f"https://filebin.net/{bin_name}/{json_path.name}"
    log(f"Uploading {json_path.name} ...")

    try:
        with open(json_path, "rb") as fh:
            resp = requests.post(
                file_url,
                data=fh,
                headers={
                    "Content-Type": "application/octet-stream",
                    "Accept":       "application/json",
                },
                timeout=120,
            )

        if resp.status_code in (200, 201):
            log(f"  OK:  {file_url}")
            return True
        else:
            log(f"  FAILED  HTTP {resp.status_code}: {resp.text[:200]}")
            return False

    except Exception as exc:
        log(f"  ERROR: {exc}")
        return False


# Clipboard helper

def _copy_to_clipboard(text: str, log: Callable[[str], None]) -> None:
    """
    Copy text to the system clipboard via a hidden Tk root window

    Silently logs an error and continues when the clipboard is unavailable

    Args:
        text: String to place on the clipboard.
        log: Callable that accepts one log line at a time.
    """
    try:
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        root.clipboard_clear()
        root.clipboard_append(text)
        root.update()  # required for the clipboard content to persist after root is destroyed
        log("Bin URL copied to clipboard.")
    except Exception as exc:
        log(f"Could not copy URL to clipboard: {exc}")


# Shutdown

def shutdown_pc(delay_seconds: int = 60, log: Callable[[str], None] = print) -> None:
    """
    Schedule an OS-level shutdown

    Args:
        delay_seconds: Seconds to wait before the machine powers off.
            On Linux the delay is rounded up to the nearest minute because the
            ``shutdown`` command only accepts whole minutes.
        log: Callable that accepts one log line at a time.
    """
    log("")
    log(f"PC will shut down in {delay_seconds} seconds.")
    log("Run  shutdown /a  in CMD to cancel.")
    subprocess.run(["shutdown", "/s", "/t", str(delay_seconds)], check=False)


# Public entry point

def upload_and_maybe_shutdown(
    json_paths: List[Path],
    bin_name: str,
    log: Callable[[str], None],
    shutdown_after: bool = False,
    shutdown_delay: int  = 60,
) -> None:
    """
    Upload all json_paths to filebin.net and optionally shut down the PC

    Runs entirely in a background daemon thread so the GUI stays responsive
    during the upload.  Shutdown (if enabled) only triggers after ALL files
    upload successfully, if any file fails, shutdown is cancelled.

    Args:
        json_paths: Ordered list of local JSON files to upload.
        bin_name: Name of the filebin.net shared bin to upload into.
        log: Callable that accepts one log line at a time (marshalled to the
            UI thread by the caller if needed).
        shutdown_after: When True, schedule a system shutdown after all
            uploads succeed.
        shutdown_delay: Seconds to wait before the machine shuts down.
    """

    def _worker() -> None:
        bin_url = f"https://filebin.net/{bin_name}"

        log("")
        log(f"Uploading {len(json_paths)} file(s) to bin '{bin_name}' ...")

        all_ok = True
        for path in json_paths:
            ok = upload_filebin(path, bin_name, log)
            if not ok:
                all_ok = False

        if all_ok:
            log("")
            log(f"  Bin  : {bin_url}  - open on any PC to see all files")
            log("")
            _copy_to_clipboard(bin_url, log)
        else:
            log("")
            log("WARNING: One or more uploads failed, shutdown cancelled.")
            return

        if shutdown_after:
            shutdown_pc(delay_seconds=shutdown_delay, log=log)

    threading.Thread(target=_worker, daemon=True).start()