"""
Tests for utils/upload_and_shutdown.py.

Covers upload_filebin() (success, HTTP error, network exception, missing
requests package) and upload_and_maybe_shutdown() (background thread,
shutdown on success / failure / disabled).
"""

import sys
import threading
import time
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

from utils.upload_and_shutdown import upload_filebin, upload_and_maybe_shutdown


# Helpers

def _collect_log():
    """
    Create a simple log collector.

    Returns:
        Tuple of (message list, append callback) for capturing log output.
    """
    msgs = []
    return msgs, msgs.append


def _make_requests_mock(status_code: int = 200, text: str = "") -> ModuleType:
    """
    Build a fake ``requests`` module whose post() returns the given HTTP status.

    Args:
        status_code: HTTP status code to return from post().
        text: Response body text.

    Returns:
        MagicMock whose post() returns a response with the given status.
    """
    mock_resp = MagicMock()
    mock_resp.status_code = status_code
    mock_resp.text = text

    mock_requests = MagicMock()
    mock_requests.post.return_value = mock_resp
    return mock_requests


# upload_filebin()

class TestUploadFilebin:
    """Verify upload_filebin() return values and log messages for each scenario."""

    def test_http_200_returns_true(self, tmp_path):
        json_file = tmp_path / "result.json"
        json_file.write_text('{"data": 1}')
        msgs, log = _collect_log()

        with patch.dict(sys.modules, {"requests": _make_requests_mock(200)}):
            result = upload_filebin(json_file, "testbin", log)

        assert result is True
        assert any("OK" in m for m in msgs)

    def test_http_201_returns_true(self, tmp_path):
        json_file = tmp_path / "result.json"
        json_file.write_text("{}")
        _, log = _collect_log()

        with patch.dict(sys.modules, {"requests": _make_requests_mock(201)}):
            result = upload_filebin(json_file, "testbin", log)

        assert result is True

    def test_http_error_returns_false(self, tmp_path):
        json_file = tmp_path / "result.json"
        json_file.write_text("{}")
        msgs, log = _collect_log()

        with patch.dict(sys.modules, {"requests": _make_requests_mock(500, "Server Error")}):
            result = upload_filebin(json_file, "testbin", log)

        assert result is False
        assert any("FAILED" in m or "500" in m for m in msgs)

    def test_network_exception_returns_false(self, tmp_path):
        json_file = tmp_path / "result.json"
        json_file.write_text("{}")
        msgs, log = _collect_log()

        mock_req = MagicMock()
        mock_req.post.side_effect = ConnectionError("network down")

        with patch.dict(sys.modules, {"requests": mock_req}):
            result = upload_filebin(json_file, "testbin", log)

        assert result is False
        assert any("ERROR" in m for m in msgs)

    def test_missing_requests_package_returns_false(self, tmp_path):
        """Setting requests to None in sys.modules triggers ImportError inside the function."""
        json_file = tmp_path / "result.json"
        json_file.write_text("{}")
        msgs, log = _collect_log()

        with patch.dict(sys.modules, {"requests": None}):
            result = upload_filebin(json_file, "testbin", log)

        assert result is False
        assert any("not installed" in m.lower() or "ERROR" in m for m in msgs)


# upload_and_maybe_shutdown()

class TestUploadAndMaybeShutdown:
    """Verify the threading and shutdown behaviour of upload_and_maybe_shutdown()."""

    def _wait_for_daemon_threads(self, timeout: float = 2.0) -> None:
        """
        Block until all daemon threads spawned by the function have finished.

        Args:
            timeout: Maximum seconds to wait before returning.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            alive = [
                t for t in threading.enumerate()
                if t.daemon and t is not threading.main_thread()
            ]
            if not alive:
                break
            time.sleep(0.02)

    def test_uploads_every_file(self, tmp_path):
        files = [tmp_path / "a.json", tmp_path / "b.json"]
        for f in files:
            f.write_text("{}")
        _, log = _collect_log()

        mock_req = _make_requests_mock(200)
        with patch.dict(sys.modules, {"requests": mock_req}):
            upload_and_maybe_shutdown(files, "mybin", log)
            self._wait_for_daemon_threads()

        assert mock_req.post.call_count == 2

    def test_shutdown_is_called_after_full_success(self, tmp_path):
        json_file = tmp_path / "result.json"
        json_file.write_text("{}")
        _, log = _collect_log()

        with patch.dict(sys.modules, {"requests": _make_requests_mock(200)}), \
             patch("utils.upload_and_shutdown.shutdown_pc") as mock_shutdown:
            upload_and_maybe_shutdown(
                [json_file], "mybin", log, shutdown_after=True, shutdown_delay=30
            )
            self._wait_for_daemon_threads()

        mock_shutdown.assert_called_once_with(delay_seconds=30, log=log)

    def test_shutdown_is_not_called_after_upload_failure(self, tmp_path):
        json_file = tmp_path / "result.json"
        json_file.write_text("{}")
        msgs, log = _collect_log()

        with patch.dict(sys.modules, {"requests": _make_requests_mock(500, "err")}), \
             patch("utils.upload_and_shutdown.shutdown_pc") as mock_shutdown:
            upload_and_maybe_shutdown(
                [json_file], "mybin", log, shutdown_after=True
            )
            self._wait_for_daemon_threads()

        mock_shutdown.assert_not_called()
        assert any("failed" in m.lower() or "cancelled" in m.lower() for m in msgs)

    def test_shutdown_is_not_called_when_disabled(self, tmp_path):
        json_file = tmp_path / "result.json"
        json_file.write_text("{}")
        _, log = _collect_log()

        with patch.dict(sys.modules, {"requests": _make_requests_mock(200)}), \
             patch("utils.upload_and_shutdown.shutdown_pc") as mock_shutdown:
            upload_and_maybe_shutdown(
                [json_file], "mybin", log, shutdown_after=False
            )
            self._wait_for_daemon_threads()

        mock_shutdown.assert_not_called()

    def test_bin_url_appears_in_log(self, tmp_path):
        json_file = tmp_path / "result.json"
        json_file.write_text("{}")
        msgs, log = _collect_log()

        with patch.dict(sys.modules, {"requests": _make_requests_mock(200)}):
            upload_and_maybe_shutdown([json_file], "myspecialbin", log)
            self._wait_for_daemon_threads()

        assert any("myspecialbin" in m for m in msgs)