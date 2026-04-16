"""
Tests for utils/subprocess_utils.py
"""

import subprocess
import sys
from unittest.mock import MagicMock, patch
import psutil

import pytest

import utils.subprocess_utils as spu

# _get_current_affinity_mask

class TestGetCurrentAffinityMask:
    def test_returns_nonzero_int_or_none(self):
        mask = spu._get_current_affinity_mask()
        # May be None on platforms without cpu_affinity() support
        assert mask is None or (isinstance(mask, int) and mask > 0)

    def test_returns_none_when_psutil_raises(self):
        with patch("psutil.Process", side_effect=Exception("no psutil")):
            mask = spu._get_current_affinity_mask()
        assert mask is None

    def test_mask_bits_match_core_list(self):
        """Bit N must be set exactly when core N appears in the affinity list."""
        mock_proc = MagicMock()
        mock_proc.cpu_affinity.return_value = [0, 2]   # cores 0 and 2
        with patch("psutil.Process", return_value=mock_proc):
            mask = spu._get_current_affinity_mask()
        # Expected: bit 0 = 1, bit 2 = 1  →  0b0101 = 5
        assert mask == 0b0101


# run_with_affinity - real Windows path

@pytest.mark.skipif(sys.platform != "win32", reason="Windows-only path")
class TestRunWithAffinityWindows:
    """Exercises the real Windows code path – runs only on Windows CI."""

    def test_echo_command_succeeds(self):
        result = spu.run_with_affinity(
            ["cmd", "/c", "echo", "hello"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "hello" in result.stdout

    def test_invalid_command_raises(self):
        with pytest.raises((FileNotFoundError, OSError)):
            spu.run_with_affinity(["__nonexistent_binary__"], check=False)


# run_with_affinity - Windows logic with mocks

class TestRunWithAffinityWindowsMocked:
    """
    Tests the Windows logic of run_with_affinity() using mocks.
    Runs on all platforms.
    """

    def test_affinity_is_applied_and_result_returned(self):
        mock_proc = MagicMock()
        mock_proc.pid = 12345
        mock_proc.poll.return_value = 0
        mock_proc.communicate.return_value = (b"output", b"")

        with patch.object(spu, "_IS_WINDOWS", True), \
             patch.object(spu, "_CREATE_NO_WINDOW", 0x08000000, create=True), \
             patch("utils.subprocess_utils._get_current_affinity_mask", return_value=0b0010), \
             patch("utils.subprocess_utils._apply_affinity_to_pid", return_value=True) as mock_apply, \
             patch("utils.subprocess_utils.subprocess.Popen", return_value=mock_proc):

            result = spu.run_with_affinity(["dummy_cmd"], capture_output=True)

        mock_apply.assert_called_once_with(12345, 0b0010)
        assert result.returncode == 0

    def test_falls_back_to_subprocess_run_when_mask_unknown(self):
        """When the affinity mask cannot be read, subprocess.run() is used directly."""
        fake = subprocess.CompletedProcess(args=["cmd"], returncode=0)

        with patch.object(spu, "_IS_WINDOWS", True), \
             patch("utils.subprocess_utils._get_current_affinity_mask", return_value=None), \
             patch("utils.subprocess_utils.subprocess.run", return_value=fake) as mock_run:

            result = spu.run_with_affinity(["cmd"])

        mock_run.assert_called_once()
        assert result.returncode == 0

    def test_explicit_affinity_mask_overrides_auto_detect(self):
        mock_proc = MagicMock()
        mock_proc.pid = 99
        mock_proc.poll.return_value = 0
        mock_proc.communicate.return_value = (b"", b"")

        with patch.object(spu, "_IS_WINDOWS", True), \
             patch.object(spu, "_CREATE_NO_WINDOW", 0x08000000, create=True), \
             patch("utils.subprocess_utils._apply_affinity_to_pid", return_value=True) as mock_apply, \
             patch("utils.subprocess_utils.subprocess.Popen", return_value=mock_proc):

            spu.run_with_affinity(["cmd"], affinity_mask=0b1000)

        mock_apply.assert_called_once_with(99, 0b1000)

    def test_check_raises_on_nonzero_returncode(self):
        mock_proc = MagicMock()
        mock_proc.pid = 1
        mock_proc.poll.return_value = 1
        mock_proc.communicate.return_value = (b"", b"error")
        mock_proc.args = ["bad_cmd"]

        with patch.object(spu, "_IS_WINDOWS", True), \
             patch.object(spu, "_CREATE_NO_WINDOW", 0x08000000, create=True), \
             patch("utils.subprocess_utils._get_current_affinity_mask", return_value=0b0001), \
             patch("utils.subprocess_utils._apply_affinity_to_pid", return_value=True), \
             patch("utils.subprocess_utils.subprocess.Popen", return_value=mock_proc):

            with pytest.raises(subprocess.CalledProcessError):
                spu.run_with_affinity(["bad_cmd"], check=True)
