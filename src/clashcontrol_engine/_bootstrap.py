"""Process startup bootstrap.

Kept separate from ``__main__`` so ``configure_io`` can be imported and
exercised by unit tests without actually running the CLI.
"""
import sys

__all__ = ["configure_io"]


def configure_io() -> None:
    """Upgrade stdout/stderr to UTF-8.

    A legacy parent encoding must never be able to crash the child. On
    Windows, ``subprocess.Popen(stdout=log_file)`` hands the child a
    cp1252-encoded stdout, and any ``print()`` containing a character
    outside the 0x00-0xFF range raises ``UnicodeEncodeError`` during
    startup — before ``server.run_server`` even gets to bind a socket.

    We reconfigure both streams to UTF-8 with ``errors="replace"`` so a
    stray non-ASCII byte is degraded to ``?`` rather than killing the
    engine. Safe to call more than once; no-op on streams that don't
    support ``reconfigure`` (Python <3.7, or oddball wrappers in tests).
    """
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass
