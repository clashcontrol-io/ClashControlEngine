"""
HTTP + WebSocket server for ClashControl local clash detection.

- HTTP on PORT (default 19800): GET /status, GET /update, POST /update,
  POST /detect, OPTIONS (CORS)
- WebSocket on PORT+1 (default 19801): progress updates during detection
"""
import asyncio
import atexit
import json
import multiprocessing
import os
import re
import threading
import time
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from threading import Thread
from urllib.request import Request, urlopen
from urllib.error import URLError

from . import __version__, daemon as _daemon
from . import updater as _updater
from .engine import detect_clashes, BACKENDS

PORT = int(os.environ.get('CC_ENGINE_PORT', 19800))
HOST = os.environ.get('CC_ENGINE_HOST', 'localhost')

# Reject request bodies larger than this (64 MB) with 413.
MAX_BODY_BYTES = 64 * 1024 * 1024

# Origins allowed to talk to the engine: the ClashControl web app and
# local development servers. Anything else gets no CORS headers.
_ALLOWED_ORIGIN_RE = re.compile(
    r'^(?:'
    r'https://(?:www\.)?clashcontrol\.io'
    r'|https?://localhost(?::\d+)?'
    r'|https?://127\.0\.0\.1(?::\d+)?'
    r'|https?://\[::1\](?::\d+)?'
    r')$'
)

_ws_clients = set()
_loop = None
_active_host = HOST
_active_port = PORT
_http_server = None
_ws_server = None


def _origin_allowed(origin):
    return bool(origin) and _ALLOWED_ORIGIN_RE.match(origin) is not None

# ---------------------------------------------------------------------------
# Update-check cache (queried lazily by GET /update)
# ---------------------------------------------------------------------------
_GITHUB_RELEASES_URL = (
    'https://api.github.com/repos/clashcontrol-io/ClashControlEngine/releases/latest'
)
_UPDATE_CACHE_TTL = 3600  # seconds
_update_cache = None  # type: dict | None
_update_cache_time: float = 0.0
_update_cache_lock = threading.Lock()


def _fetch_update_info():
    """Return {current, latest, update_available, release_url}, cached for 1 h.

    Only successful lookups are cached — a transient network failure
    should not suppress update checks for a whole hour.
    """
    global _update_cache, _update_cache_time

    now = time.monotonic()
    with _update_cache_lock:
        if _update_cache is not None and (now - _update_cache_time) < _UPDATE_CACHE_TTL:
            return _update_cache

        result = _query_github_latest()
        if 'error' not in result:
            _update_cache = result
            _update_cache_time = now
        return result


def _query_github_latest():
    try:
        req = Request(
            _GITHUB_RELEASES_URL,
            headers={'User-Agent': f'clashcontrol-engine/{__version__}'},
        )
        with urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())

        latest_tag = data.get('tag_name', '').lstrip('v')
        release_url = data.get('html_url', '')
        update_available = _is_newer(latest_tag, __version__)

        return {
            'current': __version__,
            'latest': latest_tag,
            'update_available': update_available,
            'release_url': release_url,
            # Aliases the ClashControl addon reads (kept alongside the
            # original names for backward compatibility).
            'update_version': latest_tag,
            'update_url': release_url,
        }
    except Exception:
        return {
            'current': __version__,
            'latest': None,
            'update_available': False,
            'release_url': None,
            'error': 'Unable to reach GitHub releases API',
        }


def _is_newer(candidate, current):
    return _updater.is_newer(candidate, current)


class Handler(BaseHTTPRequestHandler):

    def do_GET(self):
        if self.path == '/status':
            self._json_response(200, {
                'status': 'ready',
                'version': __version__,
                'cores': multiprocessing.cpu_count(),
                'backends': BACKENDS,
            })
        elif self.path == '/update':
            self._json_response(200, _fetch_update_info())
        else:
            self.send_response(404)
            self.end_headers()

    def _read_body(self):
        """Parse Content-Length and read the body.

        Returns the body bytes, or None after sending an error response
        (400 for a malformed length, 413 for an oversized body).
        """
        raw_length = self.headers.get('Content-Length', 0)
        try:
            content_length = int(raw_length)
        except (TypeError, ValueError):
            self._json_response(400, {'error': f'Invalid Content-Length: {raw_length!r}'})
            return None
        if content_length < 0:
            self._json_response(400, {'error': f'Invalid Content-Length: {raw_length!r}'})
            return None
        if content_length > MAX_BODY_BYTES:
            self._json_response(413, {
                'error': f'Request body too large ({content_length} bytes; '
                         f'max {MAX_BODY_BYTES})',
            })
            return None
        return self.rfile.read(content_length)

    def do_POST(self):
        if self.path == '/update':
            self._json_response(202, _updater.trigger(_active_host, _active_port))
        elif self.path == '/detect':
            body = self._read_body()
            if body is None:
                return

            try:
                payload = json.loads(body)
            except (json.JSONDecodeError, ValueError) as e:
                self._json_response(400, {'error': f'Invalid JSON: {e}'})
                return

            try:
                def on_progress(done, total):
                    self._broadcast_ws({
                        'type': 'progress',
                        'done': done,
                        'total': total,
                        'pct': round(done / total * 100) if total else 0,
                    })

                def on_phase(label):
                    # The browser addon (local-engine.js) reads msg.label
                    # and displays it as the current phase in the chat
                    # bubble; 'phase' is included for symmetry/back-compat.
                    self._broadcast_ws({
                        'type': 'phase',
                        'phase': label,
                        'label': label,
                    })

                result = detect_clashes(payload, on_progress=on_progress,
                                        on_phase=on_phase)
                self._json_response(200, result)

                self._broadcast_ws({
                    'type': 'complete',
                    'clashCount': result['stats']['clashCount'],
                    'duration_ms': result['stats']['duration_ms'],
                })

            except Exception as e:
                self._json_response(500, {'error': str(e)})
        else:
            self.send_response(404)
            self.end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        if self._cors_headers():
            self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
            self.send_header('Access-Control-Allow-Headers', 'Content-Type')
            # Chrome's Private Network Access preflight: the page on
            # clashcontrol.io (public) is talking to localhost (private).
            self.send_header('Access-Control-Allow-Private-Network', 'true')
        self.end_headers()

    def _json_response(self, status, data):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self._cors_headers()
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _cors_headers(self):
        """Reflect the Origin only for allowed origins (clashcontrol.io
        and localhost dev servers). Returns True if CORS headers were
        sent, False for disallowed/absent origins (no CORS headers, so
        browsers block cross-origin reads)."""
        origin = self.headers.get('Origin', '')
        if not _origin_allowed(origin):
            return False
        self.send_header('Access-Control-Allow-Origin', origin)
        self.send_header('Vary', 'Origin')
        return True

    def _broadcast_ws(self, msg):
        if not _ws_clients or _loop is None:
            return
        text = json.dumps(msg)
        for ws in list(_ws_clients):
            try:
                asyncio.run_coroutine_threadsafe(ws.send(text), _loop)
            except Exception:
                pass

    def log_message(self, fmt, *args):
        print(f"[CC Engine] {args[0]}")


async def _ws_handler(websocket):
    _ws_clients.add(websocket)
    try:
        async for _ in websocket:
            pass  # Send-only channel
    finally:
        _ws_clients.discard(websocket)


def _bind_with_retry(factory, what, timeout_s=5.0):
    """Call *factory* retrying EADDRINUSE for up to *timeout_s* seconds.

    Used on startup so a freshly-spawned engine (e.g. right after a
    self-update restart) can win the port from a predecessor that is
    still tearing down its sockets.
    """
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            return factory()
        except OSError as exc:
            if getattr(exc, 'errno', None) != 98 and 'in use' not in str(exc).lower():
                raise
            if time.monotonic() >= deadline:
                raise
            print(f"[CC Engine] {what} port busy - retrying ...")
            time.sleep(0.25)


def release_listen_sockets():
    """Close the HTTP (and WebSocket) listen sockets.

    Called by the updater right before spawning the replacement daemon so
    the new process can bind the ports without racing this one's exit.
    """
    global _http_server
    if _http_server is not None:
        try:
            _http_server.shutdown()
            _http_server.server_close()
        except Exception:
            pass
        _http_server = None
    if _loop is not None and _ws_server is not None:
        try:
            def _close_ws():
                _ws_server.close()
            _loop.call_soon_threadsafe(_close_ws)
        except Exception:
            pass


def run_server(host=None, port=None):
    """Start the HTTP + WebSocket server."""
    global _loop, _active_host, _active_port, _http_server, _ws_server

    host = host or HOST
    port = port or PORT
    _active_host = host
    _active_port = port
    ws_port = port + 1

    print(f"[CC Engine] ClashControl Local Engine v{__version__}")
    print(f"[CC Engine] HTTP  -> http://{host}:{port}")
    print(f"[CC Engine] WS    -> ws://{host}:{ws_port}")
    print(f"[CC Engine] Cores -> {multiprocessing.cpu_count()}")
    print(f"[CC Engine] Accel -> {', '.join(BACKENDS)}")
    print(f"[CC Engine] Ready for connections")

    # Publish our PID so --stop / --status (and ClashControl's "already
    # running?" probe) can find us. Cleared on normal exit via atexit.
    my_pid = os.getpid()
    _daemon._write_pid(my_pid, host, port)
    atexit.register(_daemon._clear_pid_if_mine, my_pid)

    # HTTP server in a daemon thread. ThreadingHTTPServer so /status
    # probes keep answering while a long /detect runs.
    http_server = _bind_with_retry(
        lambda: ThreadingHTTPServer((host, port), Handler), 'HTTP')
    _http_server = http_server
    http_thread = Thread(target=http_server.serve_forever, daemon=True)
    http_thread.start()

    # WebSocket server in asyncio event loop (main thread)
    _loop = asyncio.new_event_loop()
    asyncio.set_event_loop(_loop)

    try:
        import websockets

        async def _start_ws():
            global _ws_server
            server = await websockets.serve(_ws_handler, host, ws_port)
            _ws_server = server
            await server.serve_forever()

        _loop.run_until_complete(_start_ws())
    except ImportError:
        print("[CC Engine] websockets not installed - progress updates disabled")
        print("[CC Engine] Install with: pip install websockets")
        # Keep running with just HTTP - block main thread
        try:
            import signal
            signal.pause()
        except (AttributeError, KeyboardInterrupt):
            # signal.pause() not available on Windows - use thread join
            try:
                http_thread.daemon = False
                http_thread.join()
            except KeyboardInterrupt:
                pass
    except KeyboardInterrupt:
        pass
    finally:
        print("\n[CC Engine] Shutting down")
        if _http_server is not None:
            http_server.shutdown()
