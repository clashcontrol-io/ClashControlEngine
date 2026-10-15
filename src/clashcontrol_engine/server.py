"""
HTTP + WebSocket server for ClashControl local clash detection.

- HTTP on PORT (default 19800): GET /status, GET /update, POST /update,
  POST /detect, OPTIONS (CORS)
- WebSocket on PORT+1 (default 19801): progress updates during detection

Access control (see pairing.py): loopback Host only, an exact Origin
allow-list enforced server-side (403 before any work), and a short-lived
pairing token from POST /pair on everything a browser asks for except
/status.
"""
import asyncio
import atexit
import json
import multiprocessing
import os
import threading
import time
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from threading import Thread
from urllib.request import Request, urlopen
from urllib.error import URLError

from urllib.parse import urlsplit, parse_qs

from . import __version__, daemon as _daemon
from . import pairing as _pairing
from . import updater as _updater
from .engine import detect_clashes, BACKENDS, CAPABILITIES, PROTOCOL_VERSION

PORT = int(os.environ.get('CC_ENGINE_PORT', 19800))
HOST = os.environ.get('CC_ENGINE_HOST', 'localhost')

# Reject request bodies larger than this (64 MB) with 413.
MAX_BODY_BYTES = 64 * 1024 * 1024

# Header carrying the pairing token on HTTP requests; ?cc_token= on the WebSocket.
PAIRING_HEADER = 'X-CC-Pairing-Token'

# Readable by an allowed origin without a token, so the app can detect the
# engine (and pair) before it holds one. Nothing sensitive: version + cores.
_PUBLIC_PATHS = ('/status',)

_ws_clients = set()
_loop = None
_active_host = HOST
_active_port = PORT
_http_server = None
_ws_server = None


def _origin_allowed(origin):
    return _pairing.origin_allowed(origin)

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

    # ── Access control ────────────────────────────────────────────
    def _admit(self):
        """Host + Origin gate. Runs first on every request, before routing.

        Returns True to continue. Otherwise a response has already been sent:
        421 for a non-loopback Host (DNS rebinding), 403 for any Origin that is
        not allow-listed. The 403 carries no CORS headers.
        """
        if not _pairing.host_allowed(self.headers.get('Host', '')):
            self._json_response(421, {'error': 'Misdirected request',
                                      'code': 'host_not_allowed'}, cors=False)
            return False
        origin = self.headers.get('Origin')
        if origin is not None and not _pairing.origin_allowed(origin):
            print(f"[CC Engine] Rejected request from origin: {origin!r}")
            self._json_response(403, {'error': 'Origin not allowed',
                                      'code': 'origin_not_allowed'}, cors=False)
            return False
        return True

    def _require_pairing(self):
        """Browser-originated requests (they carry Origin) must present a valid
        pairing token. Origin-less callers are local tooling (CLI, curl)."""
        if self.headers.get('Origin') is None:
            return True
        if _pairing.store.valid(self.headers.get(PAIRING_HEADER, '')):
            return True
        self._json_response(401, {'error': 'Pairing required',
                                  'code': 'pairing_required'})
        return False

    def _pair(self):
        # Only a real, allow-listed browser origin can pair; the response is
        # CORS-readable by that origin alone, so no other site ever sees a token.
        if self.headers.get('Origin') is None:
            self._json_response(400, {'error': 'Origin header required',
                                      'code': 'origin_required'})
            return
        token, ttl = _pairing.store.issue()
        self._json_response(200, {'token': token, 'expiresIn': ttl, 'protocol': 1})

    def do_GET(self):
        if not self._admit():
            return
        if self.path not in _PUBLIC_PATHS and not self._require_pairing():
            return
        if self.path == '/status':
            self._json_response(200, {
                'status': 'ready',
                'version': __version__,
                'cores': multiprocessing.cpu_count(),
                'backends': BACKENDS,
                # Wire-contract version + which rule fields are honored engine-side,
                # so the ClashControl client can negotiate instead of hand-maintaining
                # a snapshot (V7 P1.1). Older clients ignore these extra keys.
                'protocolVersion': PROTOCOL_VERSION,
                'capabilities': CAPABILITIES,
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
        if not self._admit():
            return
        if self.path == '/pair':
            self._pair()
            return
        if not self._require_pairing():
            return
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
        if not self._admit():
            return
        self.send_response(200)
        if self._cors_headers():
            self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
            self.send_header('Access-Control-Allow-Headers',
                             f'Content-Type, {PAIRING_HEADER}')
            self.send_header('Access-Control-Max-Age', '600')
            # Chrome's Private Network Access preflight: the page on
            # clashcontrol.io (public) is talking to localhost (private).
            self.send_header('Access-Control-Allow-Private-Network', 'true')
        self.send_header('Content-Length', '0')
        self.end_headers()

    def _json_response(self, status, data, cors=True):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        if cors:
            self._cors_headers()
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _cors_headers(self):
        """Reflect the Origin only for allow-listed origins. Returns True if
        CORS headers were sent, False for absent origins (non-browser
        callers; disallowed origins never get this far - see _admit)."""
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


def _ws_request_info(websocket):
    """(path, headers) across websockets' legacy (<=12) and asyncio (>=13) APIs."""
    req = getattr(websocket, 'request', None)
    if req is not None and hasattr(req, 'path'):
        return req.path, req.headers
    return getattr(websocket, 'path', ''), getattr(websocket, 'request_headers', {})


def _ws_authorized(path, headers):
    """Browsers (Origin present) must present ?cc_token=; Origin-less local
    tooling is trusted like on the HTTP side. Origin itself is enforced by the
    handshake (`origins=` in websockets.serve)."""
    if headers.get('Origin') is None:
        return True
    token = (parse_qs(urlsplit(path or '').query).get('cc_token') or [''])[0]
    return _pairing.store.valid(token)


async def _ws_handler(websocket):
    path, headers = _ws_request_info(websocket)
    if not _ws_authorized(path, headers):
        await websocket.close(code=1008, reason='pairing required')
        return
    _ws_clients.add(websocket)
    try:
        async for _ in websocket:
            pass  # Send-only channel
    finally:
        _ws_clients.discard(websocket)


async def _start_ws_server(host, ws_port):
    """Start the progress WebSocket server with origin enforcement.

    origins=: the handshake itself 403s any browser origin that is not
    allow-listed. None admits Origin-less (non-browser) clients; those are
    local tooling, same as on the HTTP side.
    """
    import websockets
    return await websockets.serve(
        _ws_handler, host, ws_port,
        origins=list(_pairing.allowed_origins()) + [None])


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
            server = await _start_ws_server(host, ws_port)
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
