"""WebSocket progress channel: origin enforcement + pairing token."""
import asyncio
import socket

import pytest

websockets = pytest.importorskip('websockets')

from clashcontrol_engine import pairing, server

APP_ORIGIN = 'https://www.clashcontrol.io'


def _free_port():
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


def _code(exc):
    """Close code across websockets' legacy and asyncio implementations."""
    rcvd = getattr(exc, 'rcvd', None)
    return rcvd.code if rcvd is not None else getattr(exc, 'code', None)


async def _with_server(client):
    port = _free_port()
    srv = await server._start_ws_server('127.0.0.1', port)
    try:
        return await client(f'ws://127.0.0.1:{port}')
    finally:
        srv.close()
        await srv.wait_closed()
        server._ws_clients.clear()


def _run(client):
    return asyncio.run(_with_server(client))


async def _stays_open(uri, **kw):
    async with websockets.connect(uri, **kw) as ws:
        try:
            await asyncio.wait_for(ws.recv(), timeout=0.5)
        except asyncio.TimeoutError:
            return True            # send-only channel: silence == still open
        except websockets.exceptions.ConnectionClosed as e:
            return _code(e)
    return False


def test_hostile_origin_handshake_rejected():
    async def client(uri):
        with pytest.raises(websockets.exceptions.InvalidHandshake):
            async with websockets.connect(uri, origin='https://evil.example.com'):
                pass
    _run(client)


def test_null_origin_handshake_rejected():
    async def client(uri):
        with pytest.raises(websockets.exceptions.InvalidHandshake):
            async with websockets.connect(uri, origin='null'):
                pass
    _run(client)


def test_app_origin_without_token_closed_1008():
    async def client(uri):
        return await _stays_open(uri, origin=APP_ORIGIN)
    assert _run(client) == 1008


def test_app_origin_with_bad_token_closed_1008():
    async def client(uri):
        return await _stays_open(uri + '?cc_token=nope', origin=APP_ORIGIN)
    assert _run(client) == 1008


def test_app_origin_with_valid_token_stays_open():
    token, _ = pairing.store.issue()

    async def client(uri):
        return await _stays_open(uri + f'?cc_token={token}', origin=APP_ORIGIN)
    assert _run(client) is True


def test_originless_local_client_allowed():
    async def client(uri):
        return await _stays_open(uri)
    assert _run(client) is True
