# Changelog

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
