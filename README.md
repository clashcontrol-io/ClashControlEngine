# clashcontrol-engine

Local clash detection server for [ClashControl](https://clashcontrol.io) — multi-threaded exact mesh intersection on your machine.

## Install

### Standalone (no Python required)

Download the latest executable for your platform:

- [Windows (.exe)](https://github.com/clashcontrol-io/ClashControlEngine/releases/latest/download/clashcontrol-engine-win.exe)
- [macOS](https://github.com/clashcontrol-io/ClashControlEngine/releases/latest/download/clashcontrol-engine-mac)
- [Linux](https://github.com/clashcontrol-io/ClashControlEngine/releases/latest/download/clashcontrol-engine-linux)

Download, run the file, done. No install wizard needed.

> **Note:** Standalone executables are available once a [release is published](https://github.com/clashcontrol-io/ClashControlEngine/releases). Until then, use pip install below.

### pip (Python 3.8+)

```bash
pip install clashcontrol-engine
```

For faster performance (Numba JIT compilation + scipy KD-tree):

```bash
pip install clashcontrol-engine[fast]
```

## Usage

```bash
clashcontrol-engine
```

That's it. Open ClashControl in your browser, enable the **Local Clash Engine** addon in Settings, and run clash detection as usual.

### Options

```
--port PORT    HTTP port (default: 19800, WebSocket on PORT+1)
--host HOST    Bind address (default: localhost)
```

### Environment variables

- `CC_ENGINE_PORT` — same as `--port`
- `CC_ENGINE_HOST` — same as `--host`

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
