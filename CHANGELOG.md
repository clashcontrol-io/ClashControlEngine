# Changelog

## Unreleased

- **Engine persistence.** `clashcontrol-engine --install` now registers a
  user-level `clashcontrol://` URL scheme handler and starts the engine
  immediately, so ClashControl connects on first run. On subsequent
  runs, clicking **Connect** in ClashControl navigates to
  `clashcontrol://start`, which the OS routes to the registered handler
  and launches the engine on demand. No more "reinstall to reconnect".
  No auto-start at login — the engine only runs when you ask for it.
- New CLI flags: `--install`, `--uninstall`, `--daemon`, `--stop`,
  `--status`. `--daemon` spawns a detached background process with a
  PID file at `~/.clashcontrol/engine.pid` and a log at
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
