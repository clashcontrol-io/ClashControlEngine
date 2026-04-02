ClashControlEngine repository

## Overview

Local clash detection server for [ClashControl](https://clashcontrol.io). Multi-threaded exact mesh intersection using Möller triangle-triangle tests with dual-BVH traversal and sweep-and-prune broad phase.

## Architecture

- `src/clashcontrol_engine/cli.py` — CLI entry point (`clashcontrol-engine` command)
- `src/clashcontrol_engine/server.py` — HTTP + WebSocket server (port 19800)
- `src/clashcontrol_engine/engine.py` — Orchestration, element parsing, parallel dispatch
- `src/clashcontrol_engine/intersection.py` — Narrow phase: Möller 1997, BVH, min-distance
- `src/clashcontrol_engine/sweep.py` — Broad phase: sweep-and-prune AABB filtering

## Development

```bash
pip install -e ".[fast]"       # install with scipy KD-tree
pytest tests/                  # run test suite
```

## Key conventions

- Python package: `clashcontrol-engine` (hyphenated) / `clashcontrol_engine` (import)
- Env vars: `CC_ENGINE_PORT`, `CC_ENGINE_HOST`
- License: SSPL v1 (same as ClashControl)
