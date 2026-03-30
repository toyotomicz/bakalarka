"""
Tests for utils/cpu_affinity.py.

Covers IsolationConfig, IsolationState, and ProcessIsolator including
isolate(), restore(), and get_available_cores().
"""

import os
from unittest.mock import MagicMock, patch

import psutil
import pytest

from utils.cpu_affinity import IsolationConfig, IsolationState, ProcessIsolator


# IsolationConfig

class TestIsolationConfig:
    """Verify the enabled property logic of IsolationConfig."""

    def test_defaults_disabled(self):
        cfg = IsolationConfig()
        assert cfg.high_priority is False
        assert cfg.cpu_core is None
        assert cfg.enabled is False

    def test_high_priority_enables(self):
        cfg = IsolationConfig(high_priority=True)
        assert cfg.enabled is True

    def test_cpu_core_enables(self):
        cfg = IsolationConfig(cpu_core=2)
        assert cfg.enabled is True

    def test_both_enables(self):
        cfg = IsolationConfig(high_priority=True, cpu_core=0)
        assert cfg.enabled is True


# IsolationState

class TestIsolationState:
    """Verify IsolationState computed properties and default values."""

    def test_pinned_core_count_empty(self):
        state = IsolationState()
        assert state.pinned_core_count == 0

    def test_pinned_core_count_filled(self):
        state = IsolationState(pinned_cores=[1, 2, 3])
        assert state.pinned_core_count == 3

    def test_isolation_notes_default_empty(self):
        state = IsolationState()
        assert state.isolation_notes == []

    def test_isolated_default_false(self):
        state = IsolationState()
        assert state.isolated is False


# Helpers

def _make_process_mock(affinity=None, nice_val=0):
    """
    Return a psutil.Process mock with configurable affinity and nice values.

    Args:
        affinity: List of core indices to return from cpu_affinity().
        nice_val: Value to return from nice().

    Returns:
        A MagicMock that behaves like a psutil.Process instance.
    """
    proc = MagicMock()
    proc.cpu_affinity.return_value = affinity if affinity is not None else [0, 1]
    proc.nice.return_value = nice_val
    return proc


# ProcessIsolator - disabled config

class TestProcessIsolatorDisabled:
    """Verify that a disabled IsolationConfig causes isolate() to be a no-op."""

    def test_disabled_config_is_noop(self):
        cfg = IsolationConfig()
        isolator = ProcessIsolator(cfg)

        state = isolator.isolate()

        assert state.isolated is False
        assert any("disabled" in note.lower() for note in state.isolation_notes)

    def test_restore_when_not_isolated_returns_true(self):
        cfg = IsolationConfig()
        isolator = ProcessIsolator(cfg)
        assert isolator.restore() is True


# ProcessIsolator - isolate() enabled

class TestProcessIsolatorIsolate:
    """Verify isolate() behaviour when affinity and/or priority isolation is enabled."""

    @patch("utils.cpu_affinity.psutil.Process")
    def test_affinity_set_when_core_available(self, mock_process_cls):
        proc = _make_process_mock(affinity=[0, 1, 2])
        mock_process_cls.return_value = proc

        cfg = IsolationConfig(cpu_core=1)
        isolator = ProcessIsolator(cfg)
        with patch.object(ProcessIsolator, "get_available_cores", return_value=[0, 1, 2]):
            state = isolator.isolate()

        assert state.isolated is True
        assert 1 in state.pinned_cores
        assert any("pinned" in note.lower() for note in state.isolation_notes)

    @patch("utils.cpu_affinity.psutil.Process")
    def test_affinity_skipped_when_core_unavailable(self, mock_process_cls):
        """isolate() must complete without raising when the requested core is absent."""
        proc = _make_process_mock(affinity=[0])
        mock_process_cls.return_value = proc

        cfg = IsolationConfig(cpu_core=5)
        isolator = ProcessIsolator(cfg)
        with patch.object(ProcessIsolator, "get_available_cores", return_value=[0]):
            state = isolator.isolate()

        assert state.pinned_cores == []
        assert any("not available" in note.lower() for note in state.isolation_notes)

    @patch("utils.cpu_affinity.psutil.Process")
    def test_high_priority_sets_nice(self, mock_process_cls):
        proc = _make_process_mock()
        mock_process_cls.return_value = proc

        cfg = IsolationConfig(high_priority=True)
        isolator = ProcessIsolator(cfg)
        with patch.object(ProcessIsolator, "get_available_cores", return_value=[0, 1]), \
             patch("utils.cpu_affinity.psutil.HIGH_PRIORITY_CLASS", 128, create=True):
            state = isolator.isolate()

        proc.nice.assert_called_with(128)
        assert state.isolated is True

    @patch("utils.cpu_affinity.psutil.Process")
    def test_high_priority_failure_handled_gracefully(self, mock_process_cls):
        """AccessDenied during priority elevation must not propagate as an exception."""
        proc = _make_process_mock()
        proc.nice.side_effect = [0, psutil.AccessDenied(0)]
        mock_process_cls.return_value = proc

        cfg = IsolationConfig(high_priority=True)
        isolator = ProcessIsolator(cfg)
        with patch.object(ProcessIsolator, "get_available_cores", return_value=[0, 1]), \
             patch("utils.cpu_affinity.psutil.HIGH_PRIORITY_CLASS", 128, create=True):
            state = isolator.isolate()

        assert any(
            "could not" in note.lower() or "check uac" in note.lower()
            for note in state.isolation_notes
        )


# ProcessIsolator - restore()

class TestProcessIsolatorRestore:
    """Verify that restore() resets affinity and priority to their original values."""

    @patch("utils.cpu_affinity.psutil.Process")
    def test_restore_resets_affinity_and_priority(self, mock_process_cls):
        proc = _make_process_mock(affinity=[0, 1], nice_val=0)
        mock_process_cls.return_value = proc

        cfg = IsolationConfig(cpu_core=1, high_priority=True)
        isolator = ProcessIsolator(cfg)
        with patch.object(ProcessIsolator, "get_available_cores", return_value=[0, 1]), \
             patch("utils.cpu_affinity.psutil.HIGH_PRIORITY_CLASS", 128, create=True):
            isolator.isolate()

        result = isolator.restore()

        assert result is True
        assert isolator._state.isolated is False
        assert proc.cpu_affinity.call_count >= 2

    @patch("utils.cpu_affinity.psutil.Process")
    def test_restore_returns_false_on_affinity_error(self, mock_process_cls):
        proc = _make_process_mock(affinity=[0, 1])
        mock_process_cls.return_value = proc

        cfg = IsolationConfig(cpu_core=1)
        isolator = ProcessIsolator(cfg)
        with patch.object(ProcessIsolator, "get_available_cores", return_value=[0, 1]):
            isolator.isolate()

        proc.cpu_affinity.side_effect = psutil.Error("test error")
        result = isolator.restore()

        assert result is False


# ProcessIsolator - get_available_cores()

class TestGetAvailableCores:
    """Verify that get_available_cores() returns a non-empty list of integers."""

    def test_returns_list_of_ints(self):
        cores = ProcessIsolator.get_available_cores()
        assert isinstance(cores, list)
        assert all(isinstance(c, int) for c in cores)
        assert len(cores) >= 1

    @patch("utils.cpu_affinity.psutil.Process")
    def test_fallback_when_affinity_unavailable(self, mock_process_cls):
        """Falls back to range(cpu_count) when cpu_affinity() raises AttributeError."""
        proc = MagicMock()
        proc.cpu_affinity.side_effect = AttributeError("not supported")
        mock_process_cls.return_value = proc

        with patch("utils.cpu_affinity.psutil.cpu_count", return_value=4):
            cores = ProcessIsolator.get_available_cores()

        assert cores == [0, 1, 2, 3]