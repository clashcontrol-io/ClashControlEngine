# Changelog

## 0.2.1 — 2026-04-09

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
