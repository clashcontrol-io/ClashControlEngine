"""Self-installation of the frozen binary to a canonical per-user location.

A downloaded PyInstaller one-file binary ends up wherever the user saved
it — typically ``~/Downloads``. If the URL scheme handler and the daemon
both reference that path, the user is stuck: they can't clean up their
Downloads folder without breaking the integration, and if they do delete
the file, the next ``clashcontrol://`` click points at thin air.

The install flow solves this by copying the binary, on first run, to a
stable per-user directory. The URL scheme handler is registered with the
canonical path (via ``daemon.engine_argv``), the daemon is spawned from
the canonical path, and the original download is free to be deleted.

For a ``pip install``-based setup there's no single binary to copy, so
``ensure_installed`` is a no-op and ``daemon.engine_argv`` falls back to
``python -m clashcontrol_engine`` as before.
"""
import os
import shutil
import sys
from pathlib import Path
from typing import Optional

__all__ = [
    "install_dir",
    "install_path",
    "installed_binary",
    "is_frozen",
    "ensure_installed",
    "remove_installed",
]


def is_frozen() -> bool:
    """True iff we're running inside a PyInstaller one-file binary."""
    return bool(getattr(sys, "frozen", False))


def install_dir() -> Path:
    """Canonical per-user directory for the installed engine binary.

    - Windows: ``%LOCALAPPDATA%\\ClashControl``
    - macOS:   ``~/Library/Application Support/ClashControl``
    - Linux:   ``$XDG_DATA_HOME/clashcontrol`` (default ``~/.local/share``)
    """
    override = os.environ.get("CC_ENGINE_INSTALL_DIR")
    if override:
        return Path(override)

    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA")
        root = Path(base) if base else Path.home() / "AppData" / "Local"
        return root / "ClashControl"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "ClashControl"
    # Linux / other Unix: XDG data home
    base = os.environ.get("XDG_DATA_HOME")
    root = Path(base) if base else Path.home() / ".local" / "share"
    return root / "clashcontrol"


def install_path() -> Path:
    """Canonical full path to the installed engine binary."""
    name = "clashcontrol-engine.exe" if sys.platform == "win32" else "clashcontrol-engine"
    return install_dir() / name


def installed_binary() -> Optional[Path]:
    """Return the canonical install path if an installed binary exists."""
    p = install_path()
    return p if p.exists() else None


def _same_file(a: Path, b: Path) -> bool:
    try:
        return a.resolve() == b.resolve()
    except OSError:
        return False


def ensure_installed() -> Optional[Path]:
    """Copy the current frozen binary to the canonical install location.

    Returns the install path on success, or ``None`` when there is
    nothing to install (we're running from a pip install, not a frozen
    binary). Idempotent: if the current process is already running from
    the install path, just returns it. If the destination is locked by
    a running daemon at the same path, leaves the existing file in
    place and returns its path.
    """
    if not is_frozen():
        return None

    source = Path(sys.executable)
    dest = install_path()

    if _same_file(source, dest):
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)

    # Two-step copy: write to a sibling ``.new`` file, then atomically
    # rename over the destination. Lets us handle a locked destination
    # (Windows holds an exclusive lock on a running .exe) without
    # half-overwriting it.
    temp = dest.with_name(dest.name + ".new")
    try:
        shutil.copy2(source, temp)
    except OSError:
        return None

    try:
        if dest.exists():
            try:
                dest.unlink()
            except PermissionError:
                # Destination is locked — almost certainly by a daemon
                # still running from the old copy. Leave it alone; the
                # caller is expected to stop the daemon first if it
                # wants the binary upgraded.
                temp.unlink()
                return dest
        os.replace(temp, dest)
    except OSError:
        try:
            temp.unlink()
        except OSError:
            pass
        return None

    if os.name != "nt":
        try:
            os.chmod(dest, 0o755)
        except OSError:
            pass

    return dest


def remove_installed() -> bool:
    """Delete the canonical install binary, if present.

    Returns ``True`` if a file was removed. Safe to call when nothing is
    installed. A ``PermissionError`` from a locked file is swallowed —
    the uninstall flow has already tried to stop the daemon by the time
    this runs, so a locked file here indicates a race the user can
    resolve manually.
    """
    p = install_path()
    try:
        p.unlink()
        return True
    except FileNotFoundError:
        return False
    except OSError:
        return False
