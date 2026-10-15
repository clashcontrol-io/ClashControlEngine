"""Tests for the HTTP server: status, detect, body limits, CORS."""
import json
import socket
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import numpy as np
import pytest

from clashcontrol_engine.server import Handler, MAX_BODY_BYTES, _origin_allowed


APP_ORIGIN = 'https://www.clashcontrol.io'


@pytest.fixture(scope='module')
def server_port():
    httpd = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield port
    httpd.shutdown()
    httpd.server_close()


def _get(port, path, origin=None, token=None):
    headers = {'Origin': origin} if origin else {}
    if token:
        headers['X-CC-Pairing-Token'] = token
    req = urllib.request.Request(f'http://127.0.0.1:{port}{path}', headers=headers)
    with urllib.request.urlopen(req, timeout=10) as resp:
        return resp.status, dict(resp.headers), json.loads(resp.read())


def _post_json(port, path, payload, origin=None, token=None):
    body = json.dumps(payload).encode()
    headers = {'Content-Type': 'application/json'}
    if origin:
        headers['Origin'] = origin
    if token:
        headers['X-CC-Pairing-Token'] = token
    req = urllib.request.Request(
        f'http://127.0.0.1:{port}{path}', data=body, headers=headers)
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.status, dict(resp.headers), json.loads(resp.read())


def _pair(port, origin=APP_ORIGIN):
    req = urllib.request.Request(
        f'http://127.0.0.1:{port}/pair', data=b'', method='POST',
        headers={'Origin': origin})
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read())['token']


def _status_of(port, method, path, headers=None, data=None):
    """Return (status, headers, body) without raising on HTTP errors."""
    req = urllib.request.Request(f'http://127.0.0.1:{port}{path}', data=data,
                                 method=method, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, dict(resp.headers), resp.read()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read()


def _raw_request(port, data: bytes) -> bytes:
    """Send raw bytes, return the full raw response (read to close)."""
    with socket.create_connection(('127.0.0.1', port), timeout=10) as s:
        s.sendall(data)
        chunks = []
        while True:
            chunk = s.recv(65536)
            if not chunk:
                break
            chunks.append(chunk)
    return b''.join(chunks)


def _make_box(center, half_size):
    cx, cy, cz = center
    h = half_size
    verts = np.array([
        [cx-h, cy-h, cz-h], [cx+h, cy-h, cz-h],
        [cx+h, cy+h, cz-h], [cx-h, cy+h, cz-h],
        [cx-h, cy-h, cz+h], [cx+h, cy-h, cz+h],
        [cx+h, cy+h, cz+h], [cx-h, cy+h, cz+h],
    ], dtype=np.float32)
    faces = np.array([
        [0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6],
        [0, 4, 5], [0, 5, 1], [2, 6, 7], [2, 7, 3],
        [0, 7, 4], [0, 3, 7], [1, 5, 6], [1, 6, 2],
    ], dtype=np.int32)
    return verts, faces


# ── /status ───────────────────────────────────────────────────────

def test_status_shape(server_port):
    status, headers, data = _get(server_port, '/status', origin=APP_ORIGIN)
    assert status == 200
    assert data['status'] == 'ready'
    assert isinstance(data['version'], str)
    assert isinstance(data['cores'], int) and data['cores'] >= 1
    assert isinstance(data['backends'], list) and 'numpy' in data['backends']


def test_status_advertises_protocol_and_capabilities(server_port):
    """V7 P1.1: /status publishes a protocol version + a rule-capability map so
    the ClashControl client can negotiate instead of hand-maintaining a snapshot."""
    _, _, data = _get(server_port, '/status', origin=APP_ORIGIN)
    assert isinstance(data['protocolVersion'], int) and data['protocolVersion'] >= 1
    caps = data['capabilities']
    assert caps['protocolVersion'] == data['protocolVersion']
    # Exact-id/all scope only — the client must resolve rich selectors itself.
    assert caps['modelScope'] == 'exact'
    rules = caps['rules']
    # Honored engine-side (broad phase + narrow phase).
    for honored in ('mode', 'maxGap', 'excludeTypePairs'):
        assert rules[honored] is True, honored
    # Not honored — the client must apply these after the fact or fall back.
    # minGap and excludeSelf (CLAUDE.md Items 4 and 5-followup) belong here,
    # not above: minGap is never read anywhere in this engine
    # (sweep.py/intersection.py/engine.py); excludeSelf IS read
    # (sweep.py's `rules.get('excludeSelf', False)`) but only ever affects a
    # narrow edge case (a disjoint elements_a/elements_b that happens to
    # share one literal element) -- for the shipped default scope
    # (modelA/modelB both 'all'), elements_a is elements_b, sweep.py's
    # same_set branch runs instead, and it has no same-MODEL concept at
    # all. See test_excludeSelf_capability_is_honest below for the
    # reproduction. The browser adapter applies excludeSelf client-side
    # after the fact — see addons/local-engine.js.
    for unsupported in ('minGap', 'excludeSelf', 'excludeTypes', 'includeSpaces', 'toleranceByTypePair',
                        'minOverlapVolM3', 'duplicates', 'useSemanticFilter',
                        'excludeSameDiscipline', 'disciplineMatrix', 'changeAware'):
        assert rules[unsupported] is False, unsupported
    assert caps['overlapVolume'] is False


def test_excludeSelf_capability_is_honest_not_actually_implemented(server_port):
    """Pin the advertised excludeSelf capability against reality: for the
    shipped default scope (modelA/modelB both 'all', so the caller passes
    the SAME elements list on both sides), excludeSelf=True and
    excludeSelf=False must produce byte-identical candidate sets that still
    include same-model pairs -- proving the engine's broad phase has no
    same-model exclusion for this scope, regardless of what excludeSelf
    says. If this ever starts failing because sweep_and_prune genuinely
    started dropping same-model pairs, flip the capability back to True in
    the same change -- don't just delete this test."""
    from clashcontrol_engine.sweep import sweep_and_prune

    elements = [
        {'id': 'A1', 'model_id': 'A', 'bbox_min': [0, 0, 0], 'bbox_max': [2, 2, 2], 'ifcType': 'IfcWall'},
        {'id': 'A2', 'model_id': 'A', 'bbox_min': [1, 1, 1], 'bbox_max': [3, 3, 3], 'ifcType': 'IfcWall'},
        {'id': 'B10', 'model_id': 'B', 'bbox_min': [10, 10, 10], 'bbox_max': [12, 12, 12], 'ifcType': 'IfcDuct'},
        {'id': 'B11', 'model_id': 'B', 'bbox_min': [11, 11, 11], 'bbox_max': [13, 13, 13], 'ifcType': 'IfcDuct'},
    ]

    cands_true = sweep_and_prune(elements, elements, 0.0, {'excludeSelf': True})
    cands_false = sweep_and_prune(elements, elements, 0.0, {'excludeSelf': False})
    assert cands_true == cands_false, 'excludeSelf must not change the candidate set for the default (same-list) scope'

    same_model_pairs = [
        (elements[i]['id'], elements[j]['id'])
        for i, j in cands_true
        if elements[i]['model_id'] == elements[j]['model_id']
    ]
    assert same_model_pairs, 'same-model pairs must still be present -- excludeSelf does not filter them here'

    _, _, data = _get(server_port, '/status', origin=APP_ORIGIN)
    assert data['capabilities']['rules']['excludeSelf'] is False


def test_minGap_capability_is_honest_not_actually_implemented(server_port):
    """Pin the advertised minGap capability against reality: grep the whole
    engine package for any real use of a 'minGap' rule field. If this ever
    starts failing because someone genuinely implemented minGap, flip the
    capability back to True in the same change -- don't just delete this
    test."""
    import pathlib
    engine_pkg = pathlib.Path(__file__).parent.parent / 'src' / 'clashcontrol_engine'
    hits = []
    for py_file in engine_pkg.glob('*.py'):
        text = py_file.read_text()
        # The CAPABILITIES literal itself is the one allowed mention.
        for line in text.splitlines():
            stripped = line.strip()
            if 'minGap' in stripped and "'minGap':" not in stripped and not stripped.startswith('#'):
                hits.append(f'{py_file.name}: {stripped}')
    assert hits == [], (
        "minGap is referenced outside the CAPABILITIES literal -- if it's "
        "genuinely implemented now, flip CAPABILITIES['rules']['minGap'] "
        f"back to True. Found: {hits}"
    )

    _, _, data = _get(server_port, '/status', origin=APP_ORIGIN)
    assert data['capabilities']['rules']['minGap'] is False


# ── /detect happy path ────────────────────────────────────────────

def test_detect_happy_path(server_port):
    verts_a, faces_a = _make_box([0, 0, 0], 1.0)
    verts_b, faces_b = _make_box([0.5, 0, 0], 1.0)
    payload = {
        'elements': [
            {'id': 1, 'modelId': 'm', 'ifcType': 'IfcWall', 'name': 'a',
             'storey': '', 'discipline': 'other',
             'vertices': verts_a.flatten().tolist(),
             'indices': faces_a.flatten().tolist()},
            {'id': 2, 'modelId': 'm', 'ifcType': 'IfcDuct', 'name': 'b',
             'storey': '', 'discipline': 'other',
             'vertices': verts_b.flatten().tolist(),
             'indices': faces_b.flatten().tolist()},
        ],
        'rules': {'mode': 'hard'},
    }
    token = _pair(server_port)
    status, headers, data = _post_json(server_port, '/detect', payload,
                                       origin=APP_ORIGIN, token=token)
    assert status == 200
    assert len(data['clashes']) == 1
    assert data['stats']['candidatePairs'] == 1
    assert headers.get('Access-Control-Allow-Origin') == APP_ORIGIN


def test_detect_invalid_json(server_port):
    req = urllib.request.Request(
        f'http://127.0.0.1:{server_port}/detect',
        data=b'{not json',
        headers={'Content-Type': 'application/json'})
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        urllib.request.urlopen(req, timeout=10)
    assert excinfo.value.code == 400


# ── Body limits ───────────────────────────────────────────────────

def test_malformed_content_length_is_400(server_port):
    raw = _raw_request(
        server_port,
        b'POST /detect HTTP/1.1\r\n'
        b'Host: 127.0.0.1\r\n'
        b'Content-Length: abc\r\n'
        b'\r\n')
    assert raw.startswith(b'HTTP/1.0 400') or raw.startswith(b'HTTP/1.1 400')


def test_oversized_body_is_413(server_port):
    # Declare a body over the cap; the server must refuse before reading it.
    raw = _raw_request(
        server_port,
        b'POST /detect HTTP/1.1\r\n'
        b'Host: 127.0.0.1\r\n'
        b'Content-Length: ' + str(MAX_BODY_BYTES + 1).encode() + b'\r\n'
        b'\r\n')
    assert raw.startswith(b'HTTP/1.0 413') or raw.startswith(b'HTTP/1.1 413')


# ── CORS / preflight ──────────────────────────────────────────────

def test_options_preflight_allowed_origin(server_port):
    req = urllib.request.Request(
        f'http://127.0.0.1:{server_port}/detect',
        method='OPTIONS',
        headers={'Origin': APP_ORIGIN,
                 'Access-Control-Request-Method': 'POST',
                 'Access-Control-Request-Private-Network': 'true'})
    with urllib.request.urlopen(req, timeout=10) as resp:
        headers = dict(resp.headers)
    assert headers.get('Access-Control-Allow-Origin') == APP_ORIGIN
    assert headers.get('Access-Control-Allow-Private-Network') == 'true'
    assert 'POST' in headers.get('Access-Control-Allow-Methods', '')
    assert 'Content-Type' in headers.get('Access-Control-Allow-Headers', '')


def test_options_preflight_denied_origin_is_403(server_port):
    status, headers, _ = _status_of(
        server_port, 'OPTIONS', '/detect', {'Origin': 'https://evil.example.com'})
    assert status == 403
    assert 'Access-Control-Allow-Origin' not in headers
    assert 'Access-Control-Allow-Private-Network' not in headers


def test_preflight_allows_pairing_header(server_port):
    status, headers, _ = _status_of(
        server_port, 'OPTIONS', '/detect',
        {'Origin': APP_ORIGIN, 'Access-Control-Request-Private-Network': 'true'})
    assert status == 200
    assert 'X-CC-Pairing-Token' in headers.get('Access-Control-Allow-Headers', '')


@pytest.mark.parametrize('origin,allowed', [
    ('https://clashcontrol.io', True),
    ('https://www.clashcontrol.io', True),
    # Dev origins are opt-in via CC_ENGINE_ALLOWED_ORIGINS, not default.
    ('http://localhost:8000', False),
    ('http://localhost', False),
    ('http://127.0.0.1:5500', False),
    ('http://[::1]:8000', False),
    ('https://evil.example.com', False),
    ('https://clashcontrol.io.evil.com', False),
    ('https://xclashcontrol.io', False),
    ('http://www.clashcontrol.io', False),
    ('http://192.168.1.10:8000', False),
    ('null', False),
    ('', False),
])
def test_origin_allowlist(origin, allowed):
    assert _origin_allowed(origin) is allowed


def test_extra_origins_from_env(monkeypatch):
    monkeypatch.setenv('CC_ENGINE_ALLOWED_ORIGINS',
                       'http://localhost:8000, https://self.example/ ,*')
    assert _origin_allowed('http://localhost:8000')
    assert _origin_allowed('https://self.example')
    assert not _origin_allowed('*')
    assert not _origin_allowed('http://localhost:9999')


# ── Hostile-origin enforcement (executed server-side, not just CORS) ──

EVIL = 'https://evil.example.com'


@pytest.mark.parametrize('method,path', [
    ('GET', '/status'), ('GET', '/update'), ('POST', '/update'),
    ('POST', '/detect'), ('POST', '/pair'),
])
def test_hostile_origin_rejected_everywhere(server_port, method, path):
    data = b'' if method == 'POST' else None
    status, headers, body = _status_of(
        server_port, method, path, {'Origin': EVIL}, data)
    assert status == 403
    assert json.loads(body)['code'] == 'origin_not_allowed'
    assert 'Access-Control-Allow-Origin' not in headers


def test_null_origin_rejected(server_port):
    status, _, _ = _status_of(server_port, 'GET', '/status', {'Origin': 'null'})
    assert status == 403


def test_simple_cross_site_post_does_not_run_a_job(server_port):
    """A text/plain POST needs no CORS preflight, so the browser sends it
    straight away. It must be refused before the body is even parsed."""
    payload = json.dumps({'elements': [], 'rules': {}}).encode()
    status, _, body = _status_of(
        server_port, 'POST', '/detect',
        {'Origin': EVIL, 'Content-Type': 'text/plain'}, payload)
    assert status == 403
    assert b'clashes' not in body


def test_dns_rebinding_host_rejected(server_port):
    raw = _raw_request(
        server_port,
        b'GET /status HTTP/1.1\r\nHost: attacker.example:19800\r\n\r\n')
    assert b' 421 ' in raw.split(b'\r\n')[0]


@pytest.mark.parametrize('host', [b'127.0.0.1:19800', b'localhost:19800',
                                  b'[::1]:19800', b'localhost'])
def test_loopback_hosts_accepted(server_port, host):
    raw = _raw_request(server_port, b'GET /status HTTP/1.1\r\nHost: ' + host + b'\r\n\r\n')
    assert b' 200 ' in raw.split(b'\r\n')[0]


# ── Pairing ───────────────────────────────────────────────────────

def test_status_readable_by_app_without_token(server_port):
    status, headers, data = _get(server_port, '/status', origin=APP_ORIGIN)
    assert status == 200
    assert headers.get('Access-Control-Allow-Origin') == APP_ORIGIN


@pytest.mark.parametrize('method,path', [
    ('GET', '/update'), ('POST', '/update'), ('POST', '/detect')])
def test_browser_request_without_token_is_401(server_port, method, path):
    data = b'{}' if method == 'POST' else None
    status, headers, body = _status_of(
        server_port, method, path, {'Origin': APP_ORIGIN}, data)
    assert status == 401
    assert json.loads(body)['code'] == 'pairing_required'
    # 401 must be CORS-readable so the app can tell it to re-pair.
    assert headers.get('Access-Control-Allow-Origin') == APP_ORIGIN


def test_wrong_token_is_401(server_port):
    status, _, _ = _status_of(
        server_port, 'POST', '/detect',
        {'Origin': APP_ORIGIN, 'X-CC-Pairing-Token': 'not-a-real-token'}, b'{}')
    assert status == 401


def test_pair_issues_working_token(server_port):
    token = _pair(server_port)
    assert len(token) >= 32
    status, _, _ = _status_of(
        server_port, 'GET', '/update',
        {'Origin': APP_ORIGIN, 'X-CC-Pairing-Token': token})
    assert status == 200


def test_pair_tokens_are_unique(server_port):
    assert _pair(server_port) != _pair(server_port)


def test_pair_requires_origin(server_port):
    status, _, body = _status_of(server_port, 'POST', '/pair', data=b'')
    assert status == 400
    assert json.loads(body)['code'] == 'origin_required'


def test_originless_local_tooling_needs_no_token(server_port):
    """CLI / curl / daemon probes carry no Origin and are not browser traffic."""
    status, _, _ = _status_of(server_port, 'GET', '/update')
    assert status == 200


# ── Token store ───────────────────────────────────────────────────

def test_token_expires():
    from clashcontrol_engine.pairing import PairingStore
    now = [1000.0]
    st = PairingStore(ttl=60, clock=lambda: now[0])
    token, ttl = st.issue()
    assert ttl == 60 and st.valid(token)
    now[0] += 59
    assert st.valid(token)
    now[0] += 2
    assert not st.valid(token)


def test_token_store_is_capped():
    from clashcontrol_engine.pairing import PairingStore
    st = PairingStore(max_tokens=3)
    tokens = [st.issue()[0] for _ in range(5)]
    assert not st.valid(tokens[0])      # oldest evicted
    assert st.valid(tokens[-1])


@pytest.mark.parametrize('bad', [None, '', 123, b'x'])
def test_token_store_rejects_junk(bad):
    from clashcontrol_engine.pairing import PairingStore
    assert not PairingStore().valid(bad)


@pytest.mark.parametrize('host,ok', [
    ('127.0.0.1', True), ('127.0.0.1:19800', True), ('localhost:1', True),
    ('LOCALHOST', True), ('[::1]:19800', True), ('[::1]', True),
    ('attacker.example', False), ('attacker.example:19800', False),
    ('127.0.0.1.evil.com', False), ('', False), (None, False),
])
def test_host_allowed(host, ok):
    from clashcontrol_engine.pairing import host_allowed
    assert host_allowed(host) is ok
