"""Process startup bootstrap.

Kept separate from ``__main__`` so ``configure_io`` can be imported and
exercised by unit tests without actually running the CLI.

Responsible for making the process's standard streams usable no matter
how it was launched:

- On a ``--noconsole`` PyInstaller build, Windows does not attach a
  console to the process by default. ``sys.stdout`` / ``sys.stderr`` are
  set to ``None`` by the PyInstaller bootloader. Any ``print()`` call
  would raise.
- If the user launched us from ``cmd.exe``, the parent *does* have a
  console we can share via ``AttachConsole(ATTACH_PARENT_PROCESS)``,
  which lets ``--foreground`` and friends still produce terminal output.
- If the user double-clicked or went via the URL scheme, there is no
  parent console. We replace ``sys.stdout`` / ``sys.stderr`` with a
  silent sink so ``print()`` is a no-op rather than a crash, and the
  CLI commands use ``has_console_output()`` to decide whether to
  surface results via a Windows ``MessageBox`` instead.
- After stdio is usable, we reconfigure it to UTF-8 with
  ``errors="replace"`` so legacy parent encodings (cp1252 when Windows
  redirects a daemon child's stdout to a log file) can't crash any
  ``print()`` call containing non-ASCII characters.
"""
import sys

__all__ = [
    "configure_io",
    "has_console_output",
]

# Updated by ``configure_io``. Defaults to True so that pure pip/test
# runs (where sys.stdout is always a real pipe or terminal) don't get
# misrouted to a MessageBox when CLI commands check before printing.
_HAS_CONSOLE_OUTPUT = True


def has_console_output() -> bool:
    """True iff ``print()`` currently produces output the user can see.

    False when we're in a ``--noconsole`` build with no parent console
    to attach to (double-click / URL scheme launch). CLI commands use
    this to decide whether to print or pop a Windows ``MessageBox``.
    """
    return _HAS_CONSOLE_OUTPUT


class _NullStream:
    """Drop-in replacement for ``sys.stdout`` when no console exists.

    Supports the subset of the ``TextIOBase`` interface that ``print``,
    ``traceback`` and PyInstaller's bootloader touch. Pretends to have
    a ``reconfigure`` method so ``configure_io``'s UTF-8 upgrade loop
    is a no-op instead of an ``AttributeError``.
    """

    encoding = "utf-8"
    errors = "replace"

    def write(self, _data):
        return 0

    def flush(self):
        pass

    def isatty(self):
        return False

    def fileno(self):
        raise OSError("no fileno")

    def close(self):
        pass

    def reconfigure(self, **_kwargs):
        pass


def _stdio_is_usable(stream) -> bool:
    """True if *stream* behaves like a live text stream."""
    if stream is None:
        return False
    try:
        stream.write("")
    except Exception:
        return False
    return True


def _attach_parent_console() -> bool:
    """Windows-only: attach a GUI-subsystem build to the parent console.

    When the user runs our ``--noconsole`` build from ``cmd.exe``, the
    parent cmd has a console we can share via ``AttachConsole``. That
    lets terminal-invoked commands (``--foreground``, ``--status``, ...)
    still produce visible output. Returns True on success, False when
    there is no parent console (double-click or URL scheme launch).
    """
    if sys.platform != "win32":
        return False
    try:
        import ctypes
        ATTACH_PARENT_PROCESS = -1  # DWORD(-1) sentinel
        kernel32 = ctypes.windll.kernel32
        if not kernel32.AttachConsole(ATTACH_PARENT_PROCESS):
            return False
        # Rebind Python's stdio to the attached console. Opening
        # CONIN$ / CONOUT$ is the standard Win32 dance for this.
        try:
            sys.stdin = open("CONIN$", "r", encoding="utf-8", errors="replace")
        except OSError:
            pass
        try:
            sys.stdout = open("CONOUT$", "w", encoding="utf-8", errors="replace")
        except OSError:
            return False
        try:
            sys.stderr = open("CONOUT$", "w", encoding="utf-8", errors="replace")
        except OSError:
            pass
        return True
    except Exception:
        return False


def _ensure_usable_stdio() -> None:
    """Replace any ``None`` or broken standard stream with a silent sink."""
    if not _stdio_is_usable(sys.stdout):
        sys.stdout = _NullStream()
    if not _stdio_is_usable(sys.stderr):
        sys.stderr = _NullStream()
    if sys.stdin is None:
        # No _NullStream substitute for stdin — nothing we do reads from
        # it in practice, and faking a real input stream is asking for
        # trouble. Just leave it as None; Python tolerates that.
        pass


def configure_io() -> None:
    """Make stdio usable, then upgrade the encoding to UTF-8.

    Order matters:
      1. Record whether stdio was already live at entry (console-build
         launched from a terminal, or pip install).
      2. If stdio is dead and we're on Windows, try to attach to the
         parent process's console.
      3. Replace any still-dead stream with a silent sink so ``print``
         is a no-op rather than a crash.
      4. Reconfigure the live streams to UTF-8 with ``errors="replace"``
         so a legacy parent encoding can't crash a banner print.

    Safe to call more than once; subsequent calls are idempotent.
    """
    global _HAS_CONSOLE_OUTPUT

    had_stdout = _stdio_is_usable(sys.stdout)
    attached = False
    if not had_stdout:
        attached = _attach_parent_console()

    _ensure_usable_stdio()

    _HAS_CONSOLE_OUTPUT = had_stdout or attached

    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass
