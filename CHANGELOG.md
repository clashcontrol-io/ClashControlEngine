# Changelog

## Unreleased (0.3.0)

Behaviour changes (depth semantics + CORS restriction) justify the minor
bump. The browser addon reads the engine version from `/status` and is
version-agnostic.

### Correctness
- **All-vs-all runs no longer self-clash or double-count.** When both
  rule sides select the same element set, the broad phase now generates
  each unordered pair exactly once and never emits `(i, i)` self-pairs.
  Previously every element was narrow-phased against itself (shared-edge
  triangles registered as false "self clashes") and every real pair was
  tested twice.
- **Coplanar pairs: explicit touch-not-clash policy + NaN guard.**
  `tri_tri_intersect` now returns early (no hit) for (near-)coplanar
  triangle pairs, with the policy documented in code: flush surface
  contact (a wall's bottom face lying in a slab's top-face plane) is
  touching, not interpenetration — reporting it would flood every model
  with false positives at ordinary support contacts. Volumetric overlaps
  are still caught via their non-coplanar face pairs. The early-out also
  removes the 0/0 interval-division path for near-coplanar input.
- **Honest penetration depth.** Hard clashes now report the overlap of
  the two elements' AABBs along the minimum-overlap axis — a cheap,
  documented *upper bound* on true penetration
  (`depth_semantics: "aabb_overlap_estimate"`, `distance` stays negative
  mm for API compat). The old value was the length of one intersection-
  line segment, which measures the size of the crossing region, not
  penetration. The bogus `volume` field (previously `depth * 0.001`,
  not a volume) is now `null` for hard clashes.
- **Exact point-to-triangle clearance.** Soft-clash distance adds a
  vectorized Ericson closest-point-on-triangle refinement in both
  directions on top of the vertex-vertex KD-tree query. Vertex-only
  distance badly overestimated the gap between large faces whose
  vertices are far apart (two big parallel slabs reported metres of
  clearance across a 100 mm gap).

### Performance
- Real sorted-endpoint sweep-and-prune (start pointer + sweep-axis
  break) instead of rescanning the full candidate list per element.
- Per-element BVH and clearance-KD-tree caches, built once per detection
  run and reused across candidate pairs.
- The process pool now ships geometry to each worker once (pool
  initializer); tasks are `(index_a, index_b)` tuples instead of
  pickling both meshes for every pair.
- Single BVH traversal collecting up to 24 intersection points, dropping
  the redundant 3-point probe pass.

### Server
- `ThreadingHTTPServer`: `/status` keeps answering during a long
  `/detect`.
- Malformed `Content-Length` → 400; bodies over 64 MB → 413.
- CORS allow-list: only `https://clashcontrol.io`,
  `https://www.clashcontrol.io` and localhost origins get CORS headers.
- `Access-Control-Allow-Private-Network: true` on preflight (Chrome PNA).
- Failed GitHub update lookups are no longer cached for an hour.
- WebSocket `phase` messages (`Building BVH` / `Narrow phase` /
  `Finalising`) so the addon can show engine phases live.

### Updater / daemon
- `is_newer` handles pre-release tags (`0.2.6-rc1`) and treats malformed
  tags as not-newer instead of raising.
- Binaries are verified against the release's `SHA256SUMS` asset before
  the swap (releases without one log a warning and proceed).
- Windows binary swap restores the `.old` binary if the second rename
  fails; the updater releases the listen sockets before spawning the
  replacement daemon, and startup retries `EADDRINUSE` for ~5 s.
- The update worker reuses the release info fetched at trigger time
  instead of re-fetching (TOCTOU).

### Release pipeline
- **Auto releases now actually get binaries.** The tag created by
  `auto-release.yml` uses the default `GITHUB_TOKEN`, which GitHub
  prevents from triggering `release.yml`'s tag-push build — so every
  auto release shipped without assets. `auto-release.yml` now calls
  `release.yml` as a reusable workflow in the same run, and a new
  `checksums` job publishes `SHA256SUMS` alongside the binaries.

## 0.2.6 — 2026-06-10

- Contract fixes from a cross-repo audit with the ClashControl addon:
  - `/detect` clash objects carry `modelAId`/`modelBId` so the addon
    resolves elements per-model (O(1)) instead of scanning all models.
  - `GET /update` adds `update_version`/`update_url` aliases alongside
    `latest`/`release_url` — the addon's update banner reads those names.

## 0.2.5 — 2026-04-30

- CI: `workflow_dispatch` trigger on the build workflow for manual
  re-runs of a release build.

## 0.2.4 — 2026-04-30

- CI: build workflow triggers on tag push instead of `release published`
  (an attempt to make auto releases build binaries; superseded by the
  workflow_call approach in 0.3.0).

## 0.2.3 — 2026-04-10

- **Self-update.** `GET /update` reports the latest GitHub release
  (cached 1 h); `POST /update` downloads the new binary, hot-swaps the
  installed executable and restarts the daemon (frozen installs only —
  pip installs get a manual-upgrade message).
- Auto-release workflow: every push to `main` bumps the patch version
  and cuts a GitHub release.

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
