"""
Origin allow-list and short-lived pairing tokens for the local engine.

Why: the engine listens on loopback, but every page the user has open can
reach 127.0.0.1. Without enforcement a hostile site could start clash jobs
(or trigger a self-update) through the victim's browser. Three layers:

1. Origin: a request that carries an ``Origin`` header must come from the
   ClashControl web app (or an origin the user explicitly opted into with
   ``CC_ENGINE_ALLOWED_ORIGINS``). Anything else is rejected with 403
   *before* any work is done - not merely denied CORS headers, because a
   "simple" cross-site POST is executed whether or not the response is
   readable.
2. Host: must be a loopback name, which defeats DNS rebinding.
3. Pairing token: the app obtains a random, short-lived token from
   ``POST /pair`` (only answerable to an allowed origin, and the response is
   only readable by that origin) and presents it on everything else.

Requests without an ``Origin`` header are not browser cross-origin requests
(browsers always attach one to cross-origin POSTs and WebSocket handshakes),
so they are local tooling and stay governed by the loopback bind + Host check.
"""
import hashlib
import os
import secrets
import threading
import time

APP_ORIGINS = (
    'https://www.clashcontrol.io',
    'https://clashcontrol.io',
)

# Token lifetime. Clients re-pair transparently on a 401, so this can be short.
TOKEN_TTL_SECONDS = 3600
# Cap on simultaneously valid tokens (one per open tab is plenty).
MAX_ACTIVE_TOKENS = 32

_LOOPBACK_HOSTS = ('127.0.0.1', 'localhost', '[::1]', '::1')


def allowed_origins():
    """App origins plus any exact origins listed in CC_ENGINE_ALLOWED_ORIGINS.

    The extra list is for self-hosting / local development of the web app
    (e.g. ``http://localhost:8000``). Entries must be exact origins; there is
    no wildcard support on purpose.
    """
    extra = [o.strip().rstrip('/') for o in
             os.environ.get('CC_ENGINE_ALLOWED_ORIGINS', '').split(',')]
    return tuple(APP_ORIGINS) + tuple(o for o in extra if o and o != '*')


def origin_allowed(origin):
    return bool(origin) and origin in allowed_origins()


def host_allowed(host_header):
    """True when the Host header names a loopback address (any port)."""
    if not host_header:
        return False
    host = host_header.strip().lower()
    if host.startswith('['):                      # [::1]:19800
        end = host.find(']')
        host = host[:end + 1] if end != -1 else host
    else:
        host = host.rsplit(':', 1)[0] if host.count(':') == 1 else host
    return host in _LOOPBACK_HOSTS


def _digest(token):
    return hashlib.sha256(token.encode('utf-8')).hexdigest()


class PairingStore:
    """In-memory token store. Tokens die with the process (and after the TTL)."""

    def __init__(self, ttl=TOKEN_TTL_SECONDS, max_tokens=MAX_ACTIVE_TOKENS,
                 clock=time.monotonic):
        self._ttl = ttl
        self._max = max_tokens
        self._clock = clock
        self._lock = threading.Lock()
        self._tokens = {}          # sha256(token) -> expiry (monotonic seconds)

    def _purge(self, now):
        for key in [k for k, exp in self._tokens.items() if exp <= now]:
            del self._tokens[key]

    def issue(self):
        """Create a token. Returns (token, ttl_seconds)."""
        token = secrets.token_urlsafe(32)
        now = self._clock()
        with self._lock:
            self._purge(now)
            while len(self._tokens) >= self._max:   # evict the oldest
                oldest = min(self._tokens, key=self._tokens.get)
                del self._tokens[oldest]
            self._tokens[_digest(token)] = now + self._ttl
        return token, self._ttl

    def valid(self, token):
        if not token or not isinstance(token, str):
            return False
        now = self._clock()
        with self._lock:
            self._purge(now)
            return _digest(token) in self._tokens

    def clear(self):
        with self._lock:
            self._tokens.clear()


# Process-wide store shared by the HTTP and WebSocket servers.
store = PairingStore()
