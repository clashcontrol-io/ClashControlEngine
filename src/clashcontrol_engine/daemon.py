"""
Background daemon management for the clashcontrol-engine server.

Provides start/stop/status operations backed by a PID file so the server
can persist across terminal sessions without requiring an OS service
manager. The PID file lives at ``~/.clashcontrol/engine.pid`` by default;
this can be overridden with ``$CC_ENGINE_STATE_DIR`` (primarily for tests).
"""
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

__all__ = [
    "state_dir",
    "pid_file",
    "log_file",
    "current_status",
    "start_daemon",
    "stop_daemon",
    "engine_argv",
]


def engine_argv(*extra_args) -> list:
    """Return the argv list that re-invokes the engine with *extra_args*.

    Works for both ``pip install``-style environments (invokes the current
    Python interpreter with ``-m clashcontrol_engine``) and PyInstaller
    single-file binaries (invokes the frozen executable directly). Used by
    ``start_daemon`` and the URL scheme handler so every path launches the
    engine the same way regardless of how it was installed.

    For frozen binaries, prefers the canonical install location over the
    currently running executable — the user may be running from
    ``~/Downloads`` and intend to delete that file after install
    completes. Once ``install.ensure_installed`` has placed a copy at
    the install path, every subsequent spawn points there instead.
    """
    if getattr(sys, "frozen", False):
        from . import install as _install
        canonical = _install.installed_binary()
        binary = str(canonical) if canonical is not None else sys.executable
        return [binary, *extra_args]
    return [sys.executable, "-m", "clashcontrol_engine", *extra_args]


def state_dir() -> Path:
    """Return the directory used for PID + log files, creating it if needed."""
    override = os.environ.get("CC_ENGINE_STATE_DIR")
    d = Path(override) if override else Path.home() / ".clashcontrol"
    d.mkdir(parents=True, exist_ok=True)
    return d


def pid_file() -> Path:
    return state_dir() / "engine.pid"


def log_file() -> Path:
    return state_dir() / "engine.log"


# ── PID file helpers ────────────────────────────────────────────────

def _read_pid():
    p = pid_file()
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text())
        return int(data["pid"]), data
    except (ValueError, KeyError, json.JSONDecodeError, OSError):
        return None


def _write_pid(pid: int, host: str, port: int) -> None:
    pid_file().write_text(json.dumps({
        "pid": pid,
        "host": host,
        "port": port,
        "started_at": time.time(),
    }))


def _clear_pid() -> None:
    p = pid_file()
    try:
        p.unlink()
    except FileNotFoundError:
        pass


def _clear_pid_if_mine(my_pid: int) -> None:
    """Remove the PID file only if it still points at *my_pid*.

    Used by ``server.run_server`` as an ``atexit`` hook so a foreground
    engine tidies up after itself without clobbering a newer daemon.
    """
    entry = _read_pid()
    if entry is None:
        return
    pid, _ = entry
    if pid == my_pid:
        _clear_pid()


# ── Liveness checks ─────────────────────────────────────────────────

def _is_pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        return _is_pid_alive_windows(pid)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Process exists but is owned by another user — still "alive".
        return True
    return True


def _is_pid_alive_windows(pid: int) -> bool:
    import ctypes

    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    STILL_ACTIVE = 259
    kernel32 = ctypes.windll.kernel32
    handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        return False
    try:
        code = ctypes.c_ulong()
        if kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
            return code.value == STILL_ACTIVE
        return False
    finally:
        kernel32.CloseHandle(handle)


def _probe_http(host: str, port: int, timeout: float = 1.0) -> bool:
    # "localhost" sometimes resolves to IPv6 only on Windows; use 127.0.0.1
    # for the probe so we don't false-negative.
    probe_host = "127.0.0.1" if host in ("localhost", "0.0.0.0") else host
    url = f"http://{probe_host}:{port}/status"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.status == 200
    except (urllib.error.URLError, ConnectionError, OSError, ValueError):
        return False


# ── Public API ──────────────────────────────────────────────────────

def current_status():
    """Return ``(state, info)`` where state is ``'running'``, ``'stale'``, or ``'stopped'``."""
    entry = _read_pid()
    if entry is None:
        return "stopped", {}
    pid, info = entry
    if _is_pid_alive(pid):
        return "running", {"pid": pid, **info}
    return "stale", {"pid": pid, **info}


def start_daemon(host: str, port: int, wait_seconds: float = 10.0) -> int:
    """Spawn the engine as a detached background process.

    Returns the child PID. Raises ``RuntimeError`` if one is already
    running or if the child exits before the HTTP server comes up.
    """
    state, info = current_status()
    if state == "running":
        running_host = info.get("host", host)
        running_port = info.get("port", port)
        raise RuntimeError(
            f"clashcontrol-engine already running (pid={info['pid']}) "
            f"on http://{running_host}:{running_port}"
        )
    if state == "stale":
        _clear_pid()

    log = log_file().open("ab")
    cmd = engine_argv("--foreground", "--host", host, "--port", str(port))

    popen_kwargs = dict(
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=log,
        close_fds=True,
        cwd=str(Path.home()),
    )
    if os.name == "nt":
        DETACHED_PROCESS = 0x00000008
        CREATE_NEW_PROCESS_GROUP = 0x00000200
        popen_kwargs["creationflags"] = DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
    else:
        popen_kwargs["start_new_session"] = True

    proc = subprocess.Popen(cmd, **popen_kwargs)

    deadline = time.time() + wait_seconds
    while time.time() < deadline:
        if proc.poll() is not None:
            _clear_pid()
            raise RuntimeError(
                f"engine exited during startup (code={proc.returncode}); "
                f"see {log_file()} for details"
            )
        if _probe_http(host, port, timeout=0.5):
            # server.run_server will have written the PID file itself, but
            # fall back to writing it from the parent if it didn't (e.g. on
            # older installs that don't import daemon).
            if _read_pid() is None:
                _write_pid(proc.pid, host, port)
            return proc.pid
        time.sleep(0.1)

    # HTTP never came up, but the process is still alive. Record what we
    # know and let the caller decide how to surface that.
    if _read_pid() is None:
        _write_pid(proc.pid, host, port)
    return proc.pid


def stop_daemon(timeout: float = 5.0) -> bool:
    """Stop a running daemon. Returns True if something was stopped."""
    entry = _read_pid()
    if entry is None:
        return False
    pid, _ = entry
    if not _is_pid_alive(pid):
        _clear_pid()
        return False

    _send_terminate(pid)

    deadline = time.time() + timeout
    while time.time() < deadline:
        if not _is_pid_alive(pid):
            _clear_pid()
            return True
        time.sleep(0.1)

    _send_kill(pid)
    _clear_pid()
    return True


def _send_terminate(pid: int) -> None:
    try:
        if os.name == "nt":
            _terminate_windows(pid)
        else:
            os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass


def _send_kill(pid: int) -> None:
    try:
        if os.name == "nt":
            _terminate_windows(pid)
        else:
            os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _terminate_windows(pid: int) -> None:
    import ctypes

    PROCESS_TERMINATE = 0x0001
    kernel32 = ctypes.windll.kernel32
    handle = kernel32.OpenProcess(PROCESS_TERMINATE, False, pid)
    if not handle:
        return
    try:
        kernel32.TerminateProcess(handle, 0)
    finally:
        kernel32.CloseHandle(handle)
