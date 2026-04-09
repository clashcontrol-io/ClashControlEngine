"""Tests for the daemon and protocol helpers."""
import json
import os
import socket
import sys
import time

import pytest

from clashcontrol_engine import daemon, protocol


# ── Fixtures ───────────────────────────────────────────────────────

@pytest.fixture
def isolated_state(tmp_path, monkeypatch):
    """Redirect daemon.state_dir() into a temp directory for the whole test."""
    monkeypatch.setenv("CC_ENGINE_STATE_DIR", str(tmp_path))
    yield tmp_path


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# ── PID file helpers ───────────────────────────────────────────────

def test_state_dir_respects_env_var(isolated_state):
    assert daemon.state_dir() == isolated_state
    assert daemon.pid_file().parent == isolated_state


def test_write_and_read_pid(isolated_state):
    daemon._write_pid(12345, "localhost", 19800)
    entry = daemon._read_pid()
    assert entry is not None
    pid, info = entry
    assert pid == 12345
    assert info["host"] == "localhost"
    assert info["port"] == 19800
    assert "started_at" in info


def test_read_pid_missing(isolated_state):
    assert daemon._read_pid() is None


def test_read_pid_corrupt(isolated_state):
    daemon.pid_file().write_text("not-json")
    assert daemon._read_pid() is None


def test_clear_pid_idempotent(isolated_state):
    daemon._clear_pid()  # no file
    daemon._write_pid(1, "localhost", 19800)
    daemon._clear_pid()
    assert not daemon.pid_file().exists()


def test_clear_pid_if_mine_matches(isolated_state):
    daemon._write_pid(4242, "localhost", 19800)
    daemon._clear_pid_if_mine(4242)
    assert not daemon.pid_file().exists()


def test_clear_pid_if_mine_no_match(isolated_state):
    daemon._write_pid(4242, "localhost", 19800)
    daemon._clear_pid_if_mine(9999)
    # Different PID owner — file must stay
    assert daemon.pid_file().exists()


# ── Liveness ───────────────────────────────────────────────────────

def test_is_pid_alive_self():
    assert daemon._is_pid_alive(os.getpid()) is True


def test_is_pid_alive_invalid():
    assert daemon._is_pid_alive(0) is False
    assert daemon._is_pid_alive(-1) is False


def test_is_pid_alive_nonexistent():
    # Very high PID that's extremely unlikely to be in use
    assert daemon._is_pid_alive(2**31 - 2) is False


# ── current_status ─────────────────────────────────────────────────

def test_status_stopped(isolated_state):
    state, info = daemon.current_status()
    assert state == "stopped"
    assert info == {}


def test_status_stale(isolated_state):
    daemon._write_pid(2**31 - 2, "localhost", 19800)
    state, info = daemon.current_status()
    assert state == "stale"
    assert info["pid"] == 2**31 - 2


def test_status_running_for_self(isolated_state):
    daemon._write_pid(os.getpid(), "localhost", 19800)
    state, info = daemon.current_status()
    assert state == "running"
    assert info["pid"] == os.getpid()


def test_stop_clears_stale_pid(isolated_state):
    daemon._write_pid(2**31 - 2, "localhost", 19800)
    result = daemon.stop_daemon()
    assert result is False
    assert not daemon.pid_file().exists()


def test_stop_when_nothing_running(isolated_state):
    assert daemon.stop_daemon() is False


# ── Full spawn/stop lifecycle ──────────────────────────────────────

@pytest.mark.skipif(
    sys.platform == "win32",
    reason="subprocess detachment semantics differ on Windows CI",
)
def test_daemon_lifecycle(isolated_state):
    """Spawn a real detached engine, verify it's reachable, then stop it."""
    port = _free_port()
    try:
        pid = daemon.start_daemon("127.0.0.1", port, wait_seconds=15.0)
        assert pid > 0
        assert daemon._is_pid_alive(pid)

        state, info = daemon.current_status()
        assert state == "running"
        assert info["port"] == port

        assert daemon._probe_http("127.0.0.1", port, timeout=2.0)
    finally:
        daemon.stop_daemon(timeout=10.0)

    assert not daemon.pid_file().exists()
    state, _ = daemon.current_status()
    assert state == "stopped"


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="subprocess detachment semantics differ on Windows CI",
)
def test_start_daemon_refuses_double_launch(isolated_state):
    port = _free_port()
    try:
        daemon.start_daemon("127.0.0.1", port, wait_seconds=15.0)
        with pytest.raises(RuntimeError, match="already running"):
            daemon.start_daemon("127.0.0.1", port, wait_seconds=2.0)
    finally:
        daemon.stop_daemon(timeout=10.0)


# ── Protocol module ────────────────────────────────────────────────

def test_is_protocol_url():
    assert protocol.is_protocol_url("clashcontrol://start") is True
    assert protocol.is_protocol_url("CLASHCONTROL://x") is True
    assert protocol.is_protocol_url("https://clashcontrol.io") is False
    assert protocol.is_protocol_url("") is False
    assert protocol.is_protocol_url(None) is False


@pytest.mark.skipif(
    sys.platform != "linux",
    reason="Linux backend tests only",
)
def test_linux_protocol_install_and_uninstall(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    assert protocol.protocol_status() is False

    path = protocol.install_protocol()
    assert path.exists()
    assert protocol.protocol_status() is True

    contents = path.read_text()
    assert "MimeType=x-scheme-handler/clashcontrol" in contents
    assert "clashcontrol_engine --open" in contents

    assert protocol.uninstall_protocol() is True
    assert protocol.protocol_status() is False
    # uninstall on clean slate is a no-op
    assert protocol.uninstall_protocol() is False
