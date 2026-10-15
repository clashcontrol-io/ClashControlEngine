# clashcontrol-engine

Local clash detection server for [ClashControl](https://clashcontrol.io) — multi-process exact mesh intersection on your machine, across all CPU cores.

## Install

Download the file for your OS and open it. That's the whole setup.

- **Windows** — [clashcontrol-engine-win.exe](https://github.com/clashcontrol-io/ClashControlEngine/releases/latest/download/clashcontrol-engine-win.exe). Double-click.
- **macOS** — [clashcontrol-engine-mac.tar.gz](https://github.com/clashcontrol-io/ClashControlEngine/releases/latest/download/clashcontrol-engine-mac.tar.gz). Double-click the `.tar.gz` (Finder extracts it), then double-click the binary inside.
- **Linux** — [clashcontrol-engine-linux.tar.gz](https://github.com/clashcontrol-io/ClashControlEngine/releases/latest/download/clashcontrol-engine-linux.tar.gz). Extract and run:
  ```bash
  tar -xzf clashcontrol-engine-linux.tar.gz
  ./clashcontrol-engine-linux --install
  ```
  (`--install` is the explicit form of the default no-arguments flow —
  double-clicking the binary does the same thing.)

No Python, no `pip install`, no `chmod`, no install wizard. The executable bit is preserved through the tarball.

The first run does two things and exits: registers a `clashcontrol://` URL scheme handler for your user, and starts the engine as a detached background process. Open ClashControl in your browser — it connects automatically.

## Usage

From then on, you don't touch the engine at all. Whenever you want to use it, click **Connect** in ClashControl. That navigates to `clashcontrol://start`, the OS routes it to the handler the first run registered, and the engine comes up on demand. Nothing auto-runs at login — the engine only runs when you ask for it.

To uninstall the handler and stop the engine:

```bash
clashcontrol-engine --uninstall
```

### Manual control

If you'd rather manage the engine yourself, the primitives are:

```
clashcontrol-engine --foreground   # run in foreground (Ctrl-C to stop, live logs)
clashcontrol-engine --daemon       # run detached; PID at ~/.clashcontrol/engine.pid
clashcontrol-engine --status       # is it running?
clashcontrol-engine --stop         # stop a detached engine
clashcontrol-engine --uninstall    # remove URL handler + stop engine
```

### Options

```
--port PORT    HTTP port (default: 19800, WebSocket on PORT+1)
--host HOST    Bind address (default: localhost)
```

### Environment variables

- `CC_ENGINE_PORT` — same as `--port`
- `CC_ENGINE_HOST` — same as `--host`
- `CC_ENGINE_STATE_DIR` — where the PID and log files live (default `~/.clashcontrol`)
- `CC_ENGINE_ALLOWED_ORIGINS` — comma-separated **exact** extra origins allowed to use the engine, for self-hosting or developing the web app (e.g. `http://localhost:8000`). No wildcards.

## Security

The engine only listens on loopback, but any page open in your browser can reach `127.0.0.1`, so it does not trust the network path alone:

- **Origin allow-list, enforced server-side.** A request carrying an `Origin` header must come from `https://www.clashcontrol.io` / `https://clashcontrol.io` (or an origin you listed in `CC_ENGINE_ALLOWED_ORIGINS`); anything else, including `null`, gets `403` before any work runs. This covers WebSocket handshakes too.
- **Host check.** Requests whose `Host` is not a loopback name get `421` (DNS-rebinding guard).
- **Pairing token.** The web app calls `POST /pair` and receives a random token valid for 1 hour (held in memory only). Browser requests must send it as `X-CC-Pairing-Token` (or `?cc_token=` on the WebSocket) for everything except `GET /status`. Only an allow-listed origin can read the token.
- **Private Network Access.** Preflights from the allowed origin are answered with `Access-Control-Allow-Private-Network: true`, as Chromium requires for public sites talking to localhost.
- Requests with no `Origin` (the CLI, `curl`, local tooling) are not browser cross-origin traffic and stay governed by the loopback bind and the Host check.

## What it does

Runs an HTTP + WebSocket server on `localhost:19800` that ClashControl connects to for clash detection. Uses all CPU cores (worker processes) for parallel exact triangle-triangle intersection testing.

| | Browser engine | Local engine |
|---|---|---|
| Parallelism | single thread | all CPU cores (multi-process) |
| Accuracy | OBB approximation | Exact triangle intersection |
| Speed (10K elements) | ~60s | ~15-20s |
| With `[fast]` extra | — | ~1-3s |

The standalone binaries bundle Numba JIT compilation and scipy KD-tree (the `[fast]` extra) for ~20-50x per-core speedup on triangle intersection. Without them the engine runs the same algorithms in pure Python/numpy.

The browser engine is used automatically as a fallback when this server isn't running.

## Development

Contributors can install from source with pip:

```bash
git clone https://github.com/clashcontrol-io/ClashControlEngine
cd ClashControlEngine
pip install -e ".[fast]"
pytest tests/
```

This gives you the same `clashcontrol-engine` CLI but pointing at the working tree, so edits are picked up live.

## License

[SSPL v1](LICENSE) — same license as [ClashControl](https://github.com/clashcontrol-io/ClashControl).
