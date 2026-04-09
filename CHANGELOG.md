# Changelog

## 0.2.2 — 2026-04-09

- **No more cmd window on Windows.** The Windows PyInstaller build now
  ships as a GUI-subsystem (`--noconsole`) binary, so nothing flashes
  on the desktop when the `clashcontrol://` URL handler fires or when
  the user double-clicks the downloaded installer from Explorer. The
  cost of going windowed — no stdio at all in the default launch path
  — is absorbed by a new bootstrap rescue layer so terminal users and
  the install flow don't lose their feedback:
  - `_bootstrap.configure_io` now tries `AttachConsole(ATTACH_PARENT_PROCESS)`
    when stdio is dead at entry, which restores visible output for
    `--foreground`, `--status`, `--help` and friends when the user
    launched us from `cmd.exe`.
  - When there is no parent console (double-click or URL-scheme
    launch), `sys.stdout` / `sys.stderr` are swapped for a silent
    `_NullStream` so `print()` becomes a no-op instead of an
    `AttributeError`, and a new `has_console_output()` predicate lets
    the CLI know output is going nowhere.
  - `cli._show_result()` uses that predicate to pop a Windows
    `MessageBox` for the final install/uninstall verdict only when
    the console path is silent — terminal users aren't nagged with a
    dialog on top of the progress lines they've already seen.
- New regression tests cover the `--noconsole` bootloader state
  (`sys.stdout = sys.stderr = None`), the live-stdio common case, and
  the `_NullStream` `TextIOBase` subset that `print` / `traceback` /
  `reconfigure` actually touch.

## 0.2.1 — 2026-04-09

- **URL-scheme handler is now fire-and-forget.** Clicking Connect in
  ClashControl pinned a Windows console window open for up to ten
  seconds because ``clashcontrol-engine --open`` blocked on the HTTP
  readiness probe inside ``start_daemon``. Two changes fix this:
  - ``start_daemon`` learns a ``wait_for_ready`` parameter. When
    False, the parent spawns the child, writes a provisional PID
    file, and returns immediately instead of polling for up to ten
    seconds. ``_cmd_open`` uses this mode — the invoker (ClashControl
    itself) runs its own connection retry loop, so blocking the
    URL-scheme parent serves no purpose.
  - ``_cmd_open`` hides its attached console window via
    ``ShowWindow(GetConsoleWindow(), SW_HIDE)`` as its very first
    action on Windows. The window is still visible briefly during
    PyInstaller bootloader extraction (that runs before any of our
    Python code), but disappears the instant our code starts.
- **Self-installing binary.** The downloaded PyInstaller executable
  now copies itself to a canonical per-user location on first run, and
  the URL scheme handler + daemon spawn both reference that canonical
  path instead of whatever directory the user downloaded to. After
  install, the download is disposable — delete it from `~/Downloads`,
  the integration keeps working. Install locations:
  - Windows: `%LOCALAPPDATA%\ClashControl\clashcontrol-engine.exe`
  - macOS:   `~/Library/Application Support/ClashControl/clashcontrol-engine`
  - Linux:   `$XDG_DATA_HOME/clashcontrol/clashcontrol-engine`
    (default `~/.local/share/clashcontrol/clashcontrol-engine`)
  Override the install directory with `CC_ENGINE_INSTALL_DIR` if
  needed. `--uninstall` now also removes the canonical binary.
- **Install flow is upgrade-aware.** Re-running the install from a
  newer downloaded binary now stops the existing daemon first, then
  overwrites the canonical binary, then re-registers and re-starts.
  Previously, an upgrade silently no-op'd because the "already
  running" branch short-circuited before the new binary ever took
  over.
- **Fix one-click install on Windows.** The detached daemon crashed
  silently on startup because `subprocess.Popen(stdout=log_file)` handed
  the child a cp1252-encoded stdout, and the startup banner contained
  characters (`->` arrows, em-dashes) that cp1252 couldn't encode. The
  child raised `UnicodeEncodeError` before it could bind its port,
  leaving the clashcontrol:// handler to spawn a process that never
  became reachable. Fix is two layers:
  - `__main__.py` now reconfigures stdout/stderr to UTF-8 with
    `errors="replace"` at the very first opportunity, via a new
    `_bootstrap.configure_io` helper. Guards against any future
    non-ASCII print crashing a log-redirected child.
  - The startup banner and install messages are now pure ASCII
    (`->` instead of `→`, `-` instead of em-dashes) as a belt to go
    with the UTF-8 reconfigure suspenders.
- Regression test spawns a subprocess under
  `PYTHONIOENCODING=ascii:strict` and verifies `configure_io` lets it
  print U+2192 and U+2014 without crashing.

## 0.2.0 — 2026-04-09

- **One-click install.** Running `clashcontrol-engine` with no arguments
  is now the install flow: it registers a per-user `clashcontrol://`
  URL scheme handler, spawns a detached engine, and exits. Double-click
  the downloaded PyInstaller binary on any platform — that's the whole
  setup. The daemon spawner is frozen-aware, so both `pip install` and
  the standalone binaries follow the same path.
- **Linux/macOS release assets are now tarballs** so the executable bit
  is preserved end-to-end. No more `chmod +x` on download. Windows
  continues to ship a raw `.exe`.
- **README restructured** to treat the standalone binaries as the only
  documented install path for end users; the `pip install` route moves
  to a short "Development" section aimed at contributors working from
  source.
- **Engine persistence via URL scheme.** On subsequent visits, clicking
  **Connect** in ClashControl navigates to `clashcontrol://start`, which
  the OS routes to the registered handler and launches the engine on
  demand. No more "reinstall to reconnect". No auto-start at login — the
  engine only runs when you ask for it.
- **`--foreground` flag** replaces the old "no-args runs foreground"
  behaviour for power users who want live logs.
- New CLI flags: `--uninstall`, `--daemon`, `--stop`, `--status`,
  `--foreground`. `--daemon` spawns a detached background process with
  a PID file at `~/.clashcontrol/engine.pid` and a log at
  `~/.clashcontrol/engine.log`.
- Scheme handler is registered per-user (no sudo/admin): LaunchAgent
  `.app` bundle on macOS, XDG `.desktop` file on Linux, HKCU registry
  keys on Windows.

## 0.1.0 — 2026-04-01

Initial release. Extracted from the [ClashControl](https://github.com/clashcontrol-io/ClashControl) monorepo into its own repository.

- Multi-threaded exact mesh intersection using all CPU cores
- Möller 1997 triangle-triangle intersection with degenerate triangle handling
- Dual-BVH traversal for efficient narrow phase
- Sweep-and-prune broad phase with dynamic axis selection
- Hard clash detection (exact triangle intersection)
- Soft clash detection (clearance gap checking with KD-tree or spatial hash fallback)
- HTTP server on `localhost:19800` with `/status` and `/detect` endpoints
- WebSocket server on port 19801 for real-time progress updates
- CORS support for browser addon integration
- IFC type pair exclusion rules
- Optional scipy dependency for faster distance calculations
