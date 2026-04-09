# clashcontrol-engine

Local clash detection server for [ClashControl](https://clashcontrol.io) — multi-threaded exact mesh intersection on your machine.

## Install

### Standalone (no Python required)

Download the latest executable for your platform:

- [Windows (.exe)](https://github.com/clashcontrol-io/ClashControlEngine/releases/latest/download/clashcontrol-engine-win.exe)
- [macOS](https://github.com/clashcontrol-io/ClashControlEngine/releases/latest/download/clashcontrol-engine-mac)
- [Linux](https://github.com/clashcontrol-io/ClashControlEngine/releases/latest/download/clashcontrol-engine-linux)

Download, run the file, done. No install wizard needed.

### pip (Python 3.8+)

```bash
pip install clashcontrol-engine
```

For faster performance (Numba JIT compilation + scipy KD-tree):

```bash
pip install clashcontrol-engine[fast]
```

## Usage

First-run install (do this once):

```bash
clashcontrol-engine --install
```

This registers a `clashcontrol://` URL scheme handler for your user and
starts the engine immediately. Open ClashControl in your browser — it
connects automatically.

From then on, you don't need to touch the terminal. Whenever you want
to use the local engine again, just click **Connect** in ClashControl.
That navigates to `clashcontrol://start`, the OS routes it to the
handler this install registered, and the engine comes up on demand.
Nothing auto-runs at login — the engine only runs when you ask for it.

To uninstall the handler and stop the engine:

```bash
clashcontrol-engine --uninstall
```

### Manual control

If you'd rather manage the engine yourself, the primitives are:

```
clashcontrol-engine             # run in foreground (Ctrl-C to stop)
clashcontrol-engine --daemon    # run detached; PID at ~/.clashcontrol/engine.pid
clashcontrol-engine --status    # is it running?
clashcontrol-engine --stop      # stop a detached engine
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

## What it does

Runs an HTTP + WebSocket server on `localhost:19800` that ClashControl connects to for clash detection. Uses all CPU cores for parallel exact triangle-triangle intersection testing.

| | Browser engine | Local engine |
|---|---|---|
| Threads | 1 | All CPU cores |
| Accuracy | OBB approximation | Exact triangle intersection |
| Speed (10K elements) | ~60s | ~15-20s |
| With `[fast]` extra | — | ~1-3s |

The `[fast]` extra adds Numba JIT compilation (~20-50x per-core speedup on triangle intersection) and scipy KD-tree for distance calculations. Without it, the engine runs the same algorithms in pure Python/numpy.

The browser engine is used automatically as a fallback when this server isn't running.

## Requirements

- Python 3.8+
- numpy

## License

[SSPL v1](LICENSE) — same license as [ClashControl](https://github.com/clashcontrol-io/ClashControl).
