"""Tests for the HTTP server: status, detect, body limits, CORS."""
import json
import socket
import threading
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


def _get(port, path, origin=None):
    headers = {'Origin': origin} if origin else {}
    req = urllib.request.Request(f'http://127.0.0.1:{port}{path}', headers=headers)
    with urllib.request.urlopen(req, timeout=10) as resp:
        return resp.status, dict(resp.headers), json.loads(resp.read())


def _post_json(port, path, payload, origin=None):
    body = json.dumps(payload).encode()
    headers = {'Content-Type': 'application/json'}
    if origin:
        headers['Origin'] = origin
    req = urllib.request.Request(
        f'http://127.0.0.1:{port}{path}', data=body, headers=headers)
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.status, dict(resp.headers), json.loads(resp.read())


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
    for honored in ('mode', 'maxGap', 'minGap', 'excludeSelf', 'excludeTypePairs'):
        assert rules[honored] is True, honored
    # Not honored — the client must apply these after the fact or fall back.
    for unsupported in ('excludeTypes', 'includeSpaces', 'toleranceByTypePair',
                        'minOverlapVolM3', 'duplicates', 'useSemanticFilter',
                        'excludeSameDiscipline', 'disciplineMatrix', 'changeAware'):
        assert rules[unsupported] is False, unsupported
    assert caps['overlapVolume'] is False


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
    status, headers, data = _post_json(server_port, '/detect', payload,
                                       origin=APP_ORIGIN)
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


def test_options_preflight_denied_origin(server_port):
    req = urllib.request.Request(
        f'http://127.0.0.1:{server_port}/detect',
        method='OPTIONS',
        headers={'Origin': 'https://evil.example.com'})
    with urllib.request.urlopen(req, timeout=10) as resp:
        headers = dict(resp.headers)
    assert 'Access-Control-Allow-Origin' not in headers
    assert 'Access-Control-Allow-Private-Network' not in headers


def test_cors_reflection_denied_on_get(server_port):
    status, headers, _ = _get(server_port, '/status',
                              origin='https://evil.example.com')
    assert status == 200  # still served — just no CORS grant
    assert 'Access-Control-Allow-Origin' not in headers


@pytest.mark.parametrize('origin,allowed', [
    ('https://clashcontrol.io', True),
    ('https://www.clashcontrol.io', True),
    ('http://localhost:8000', True),
    ('http://localhost', True),
    ('https://localhost:8443', True),
    ('http://127.0.0.1:5500', True),
    ('http://[::1]:8000', True),
    ('https://evil.example.com', False),
    ('https://clashcontrol.io.evil.com', False),
    ('https://xclashcontrol.io', False),
    ('http://192.168.1.10:8000', False),
    ('null', False),
    ('', False),
])
def test_origin_allowlist(origin, allowed):
    assert _origin_allowed(origin) is allowed
