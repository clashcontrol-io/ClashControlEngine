"""CLI entry point for clashcontrol-engine."""
import argparse
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        prog='clashcontrol-engine',
        description='Local clash detection server for ClashControl',
    )
    parser.add_argument(
        '--port', type=int,
        default=int(os.environ.get('CC_ENGINE_PORT', 19800)),
        help='HTTP port (default: 19800, WebSocket on PORT+1)',
    )
    parser.add_argument(
        '--host', type=str,
        default=os.environ.get('CC_ENGINE_HOST', 'localhost'),
        help='Bind address (default: localhost)',
    )

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        '--foreground', action='store_true',
        help='Run the HTTP/WebSocket server in the foreground (Ctrl-C to stop)',
    )
    mode.add_argument(
        '--uninstall', action='store_true',
        help='Remove the clashcontrol:// URL scheme and stop the running engine',
    )
    mode.add_argument(
        '--daemon', action='store_true',
        help='Start the engine as a detached background process and exit',
    )
    mode.add_argument(
        '--stop', action='store_true',
        help='Stop a running background engine and exit',
    )
    mode.add_argument(
        '--status', action='store_true',
        help='Report whether the engine is running and exit',
    )
    mode.add_argument(
        '--install', action='store_true',
        help=argparse.SUPPRESS,  # Kept as an explicit alias for the default
    )
    mode.add_argument(
        '--open', metavar='URL', default=None,
        help=argparse.SUPPRESS,  # Invoked by the URL scheme handler
    )

    args = parser.parse_args()

    if args.uninstall:
        sys.exit(_cmd_uninstall())
    if args.daemon:
        sys.exit(_cmd_daemon(args.host, args.port))
    if args.stop:
        sys.exit(_cmd_stop())
    if args.status:
        sys.exit(_cmd_status())
    if args.open is not None:
        sys.exit(_cmd_open(args.open, args.host, args.port))
    if args.foreground:
        try:
            from .server import run_server
            run_server(host=args.host, port=args.port)
        except KeyboardInterrupt:
            sys.exit(0)
        return

    # Default: first-run install flow. Idempotent — safe to invoke every
    # time. Double-clicking a PyInstaller binary lands here, which is what
    # makes the flow genuinely one-click.
    sys.exit(_cmd_install(args.host, args.port))


def _cmd_install(host, port):
    """First-run install: self-install + URL scheme + start the engine.

    Steps, in order:
      1. Stop any previously running daemon so the canonical binary
         path is unlocked and can be overwritten with a fresh version.
      2. Copy the currently running frozen binary to the canonical
         per-user install location (no-op for pip installs).
      3. Register the ``clashcontrol://`` URL scheme handler. The
         handler command resolves through ``daemon.engine_argv``, so
         it will point at the install location, not at the download.
      4. Start a fresh daemon from the canonical install location.
      5. Tell the user it's safe to delete the downloaded binary.
    """
    from . import daemon, install, protocol

    # 1. Stop any prior daemon so we can overwrite its on-disk binary.
    state, info = daemon.current_status()
    if state == "running":
        prior_pid = info.get("pid")
        print(f"[CC Engine] Stopping previous engine (pid={prior_pid}) to upgrade")
        daemon.stop_daemon(timeout=10.0)
    elif state == "stale":
        daemon._clear_pid()

    # 2. Copy ourselves to the canonical install location. Frozen binary
    #    only — a pip install is already in a stable site-packages dir
    #    and has nothing to copy.
    source_path = Path(sys.executable).resolve() if install.is_frozen() else None
    installed = install.ensure_installed()
    relocated = (
        installed is not None
        and source_path is not None
        and source_path != installed.resolve()
    )
    if relocated:
        print(f"[CC Engine] Installed engine binary to {installed}")

    # 3. Register URL scheme. ``protocol.install_protocol`` builds its
    #    command through ``daemon.engine_argv``, which now prefers the
    #    freshly installed canonical path.
    location = protocol.install_protocol()
    print(f"[CC Engine] Registered clashcontrol:// handler at {location}")

    # 4. Start the engine. ``daemon.start_daemon`` also routes through
    #    ``engine_argv``, so the child process is spawned from the
    #    canonical path — not from ~/Downloads.
    try:
        pid = daemon.start_daemon(host, port)
    except RuntimeError as e:
        print(f"[CC Engine] {e}", file=sys.stderr)
        return 1
    print(f"[CC Engine] Engine started (pid={pid}) on http://{host}:{port}")

    # 5. Tell the user the download is now disposable.
    if relocated:
        print(f"[CC Engine] You can safely delete {source_path}")

    print()
    print("[CC Engine] Install complete.")
    print("[CC Engine] Open ClashControl - it will connect automatically.")
    print("[CC Engine] Next time, just click Connect in ClashControl and the")
    print("[CC Engine] engine will start on demand. Nothing auto-runs at login.")
    return 0


def _cmd_uninstall():
    from . import daemon, install, protocol

    removed = protocol.uninstall_protocol()
    stopped = daemon.stop_daemon()
    # Stop must happen before remove_installed — on Windows the install
    # binary is locked for the lifetime of the daemon process.
    deleted = install.remove_installed()

    if removed:
        print("[CC Engine] clashcontrol:// handler removed")
    else:
        print("[CC Engine] No URL handler was registered")
    if stopped:
        print("[CC Engine] Stopped running engine")
    else:
        print("[CC Engine] Engine was not running")
    if deleted:
        print(f"[CC Engine] Removed installed binary at {install.install_path()}")
    return 0


def _cmd_daemon(host, port):
    from . import daemon
    try:
        pid = daemon.start_daemon(host, port)
    except RuntimeError as e:
        print(f"[CC Engine] {e}", file=sys.stderr)
        return 1
    print(f"[CC Engine] Started detached (pid={pid}) on http://{host}:{port}")
    print(f"[CC Engine] PID file: {daemon.pid_file()}")
    print(f"[CC Engine] Log file: {daemon.log_file()}")
    print(f"[CC Engine] Stop with: clashcontrol-engine --stop")
    return 0


def _cmd_stop():
    from . import daemon
    if daemon.stop_daemon():
        print("[CC Engine] Stopped")
        return 0
    print("[CC Engine] Not running")
    return 1


def _cmd_status():
    from . import daemon, protocol
    state, info = daemon.current_status()
    scheme_on = protocol.protocol_status()

    if state == "running":
        host = info.get("host", "localhost")
        port = info.get("port", 19800)
        print(
            f"[CC Engine] Running (pid={info['pid']}) on http://{host}:{port}"
        )
        print(f"[CC Engine] URL handler: {'registered' if scheme_on else 'not registered'}")
        return 0
    if state == "stale":
        print(f"[CC Engine] Stale PID file (pid={info['pid']} not alive)")
        print(f"[CC Engine] URL handler: {'registered' if scheme_on else 'not registered'}")
        return 2
    print("[CC Engine] Not running")
    print(f"[CC Engine] URL handler: {'registered' if scheme_on else 'not registered'}")
    return 1


def _cmd_open(url, host, port):
    """Invoked by the OS when a ``clashcontrol://`` URL is activated.

    The URL body is currently ignored — we just ensure the daemon is
    running. This is idempotent: if the engine is already up, return
    success without touching it.
    """
    from . import daemon, protocol

    if not protocol.is_protocol_url(url):
        print(f"[CC Engine] --open expects a clashcontrol:// URL, got: {url!r}",
              file=sys.stderr)
        return 2

    state, _ = daemon.current_status()
    if state == "running":
        return 0
    if state == "stale":
        daemon._clear_pid()

    try:
        daemon.start_daemon(host, port)
    except RuntimeError as e:
        print(f"[CC Engine] {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    main()
