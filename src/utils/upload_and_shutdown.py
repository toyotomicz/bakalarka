"""
utils/upload_and_shutdown.py

Upload one or more benchmark JSONs to filebin.net, then optionally shut down.
"""

import sys
import subprocess
import threading
from pathlib import Path
from typing import Callable, List, Optional


# ---------------------------------------------------------------------------
# Upload single file
# ---------------------------------------------------------------------------

def upload_filebin(
    json_path: Path,
    bin_name: str,
    log: Callable[[str], None],
) -> bool:
    """Upload one file to filebin.net/<bin_name>. Returns True on success."""
    try:
        import requests
    except ImportError:
        log("ERROR: 'requests' not installed. Run: pip install requests")
        return False

    file_url = f"https://filebin.net/{bin_name}/{json_path.name}"
    log(f"Uploading {json_path.name} ...")

    try:
        with open(json_path, "rb") as f:
            resp = requests.post(
                file_url,
                data=f,
                headers={
                    "Content-Type": "application/octet-stream",
                    "Accept":       "application/json",
                },
                timeout=120,
            )

        if resp.status_code in (200, 201):
            log(f"  OK  →  {file_url}")
            return True
        else:
            log(f"  FAILED  HTTP {resp.status_code}: {resp.text[:200]}")
            return False

    except Exception as exc:
        log(f"  ERROR: {exc}")
        return False


# ---------------------------------------------------------------------------
# Clipboard helper
# ---------------------------------------------------------------------------

def _copy_to_clipboard(text: str, log: Callable[[str], None]) -> None:
    try:
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        root.clipboard_clear()
        root.clipboard_append(text)
        root.update()
        log("Bin URL copied to clipboard.")
    except Exception as exc:
        log(f"Could not copy URL to clipboard: {exc}")


# ---------------------------------------------------------------------------
# Shutdown
# ---------------------------------------------------------------------------

def shutdown_pc(delay_seconds: int = 60, log: Callable[[str], None] = print) -> None:
    log("")
    log(f"PC will shut down in {delay_seconds} seconds.")
    if sys.platform == "win32":
        log("Run  shutdown /a  in Command Prompt to cancel.")
        subprocess.run(["shutdown", "/s", "/t", str(delay_seconds)], check=False)
    else:
        minutes = max(1, delay_seconds // 60)
        log("Run  sudo shutdown -c  to cancel.")
        subprocess.run(["shutdown", "-h", f"+{minutes}"], check=False)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def upload_and_maybe_shutdown(
    json_paths: List[Path],
    bin_name: str,
    log: Callable[[str], None],
    shutdown_after: bool = False,
    shutdown_delay: int = 60,
) -> None:
    """
    Upload all json_paths to filebin.net/<bin_name> in a background thread.
    Shutdown (if enabled) only triggers after ALL files upload successfully.
    If any file fails, shutdown is cancelled.
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
            log("╔══════════════════════════════════════════════════════════════════╗")
            log(f"  All uploads successful!")
            log(f"  Bin  : {bin_url}  <- open on any PC to see all files")
            log(f"  ZIP  : {bin_url}/archive  <- download everything at once")
            log("╚══════════════════════════════════════════════════════════════════╝")
            log("")
            _copy_to_clipboard(bin_url, log)
        else:
            log("")
            log("WARNING: One or more uploads failed — shutdown cancelled.")
            return

        if shutdown_after:
            shutdown_pc(delay_seconds=shutdown_delay, log=log)

    threading.Thread(target=_worker, daemon=True).start()