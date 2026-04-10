"""
Binary self-update: fetch the latest GitHub release asset, hot-swap the
installed binary, and restart the engine daemon.

Only works for frozen (PyInstaller) installs.  pip-based installs return
an explanatory error so callers can surface a manual-upgrade message.

Public API
----------
trigger(host, port) -> dict
    Start a background update if one is available.
    Returns immediately with a status dict:

    {"status": "updating",   "current": "0.2.2", "latest": "0.2.3"}
    {"status": "up_to_date", "version": "0.2.2"}
    {"status": "error",      "error": "<reason>"}

is_newer(candidate, current) -> bool
    Semver comparison helper (re-exported for server.py).
"""
import json
import os
import subprocess
import sys
import tarfile
import threading
from pathlib import Path
from urllib.request import Request, urlopen

from . import __version__
from . import daemon as _daemon
from . import install as _install

_RELEASES_API = (
    'https://api.github.com/repos/clashcontrol-io/ClashControlEngine/releases/latest'
)

# Guard against concurrent update attempts
_update_lock = threading.Lock()


# ── Helpers ─────────────────────────────────────────────────────────────────

def is_newer(candidate, current):
    """Return True if *candidate* version string is strictly newer than *current*."""
    def _parse(v):
        try:
            return tuple(int(x) for x in v.split('.'))
        except Exception:
            return (0,)
    return _parse(candidate) > _parse(current)


def _platform_asset():
    if sys.platform == 'win32':
        return 'clashcontrol-engine-win.exe'
    if sys.platform == 'darwin':
        return 'clashcontrol-engine-mac.tar.gz'
    return 'clashcontrol-engine-linux.tar.gz'


def _fetch_release():
    req = Request(
        _RELEASES_API,
        headers={'User-Agent': f'clashcontrol-engine/{__version__}'},
    )
    with urlopen(req, timeout=10) as resp:
        return json.loads(resp.read())


def _download_asset(url, dest: Path):
    """Download *url* to *dest*, extracting from a tar.gz archive when needed."""
    req = Request(url, headers={'User-Agent': f'clashcontrol-engine/{__version__}'})
    with urlopen(req, timeout=120) as resp:
        asset_name = _platform_asset()
        if asset_name.endswith('.tar.gz'):
            tmp = dest.with_suffix('.tmp.tar.gz')
            try:
                tmp.write_bytes(resp.read())
                with tarfile.open(tmp) as tf:
                    # Archive contains a single file; name = asset without .tar.gz
                    member_name = asset_name[:-7]
                    dest.write_bytes(tf.extractfile(tf.getmember(member_name)).read())
            finally:
                try:
                    tmp.unlink()
                except OSError:
                    pass
        else:
            dest.write_bytes(resp.read())

    if os.name != 'nt':
        os.chmod(dest, 0o755)


# ── Core update logic (runs in background thread) ────────────────────────────

def _replace_binary(new_binary: Path, install_path: Path):
    """Atomically replace the installed binary with *new_binary*."""
    if sys.platform == 'win32':
        # Windows: cannot overwrite a running exe, but rename works fine.
        # Move the current binary aside, then rename the new one into place.
        old = install_path.with_suffix('.old')
        try:
            old.unlink()
        except OSError:
            pass
        os.rename(install_path, old)
        os.rename(new_binary, install_path)
    else:
        # Unix: os.replace is atomic even on a running binary (inode swap).
        os.replace(new_binary, install_path)
        os.chmod(install_path, 0o755)


def _spawn_updated_daemon(install_path: Path, host: str, port: int):
    """Spawn the new binary as a detached daemon."""
    log = open(_daemon.log_file(), 'ab')
    kwargs = dict(
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=log,
        close_fds=True,
        cwd=str(Path.home()),
    )
    if os.name == 'nt':
        DETACHED_PROCESS    = 0x00000008
        CREATE_NEW_PROCESS_GROUP = 0x00000200
        kwargs['creationflags'] = DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
    else:
        kwargs['start_new_session'] = True

    subprocess.Popen(
        [str(install_path), '--foreground', '--host', host, '--port', str(port)],
        **kwargs,
    )


def _run_update(host: str, port: int):
    """Background worker: download → replace → restart."""
    with _update_lock:
        try:
            release = _fetch_release()
        except Exception as exc:
            print(f'[CC Engine] Update check failed: {exc}')
            return

        tag = release.get('tag_name', '').lstrip('v')
        if not is_newer(tag, __version__):
            print(f'[CC Engine] Already up to date ({__version__})')
            return

        asset_name = _platform_asset()
        asset = next(
            (a for a in release.get('assets', []) if a['name'] == asset_name),
            None,
        )
        if asset is None:
            print(f'[CC Engine] Release {release["tag_name"]} has no asset {asset_name!r}')
            return

        install_path = _install.install_path()
        new_binary = install_path.with_name(install_path.name + '.new')

        print(f'[CC Engine] Downloading {asset_name} ({__version__} -> {tag}) ...')
        try:
            _download_asset(asset['browser_download_url'], new_binary)
        except Exception as exc:
            print(f'[CC Engine] Update download failed: {exc}')
            try:
                new_binary.unlink()
            except OSError:
                pass
            return

        try:
            _replace_binary(new_binary, install_path)
        except Exception as exc:
            print(f'[CC Engine] Update replace failed: {exc}')
            try:
                new_binary.unlink()
            except OSError:
                pass
            return

        try:
            _spawn_updated_daemon(install_path, host, port)
        except Exception as exc:
            print(f'[CC Engine] Update restart failed: {exc}')
            return

        print(f'[CC Engine] Updated to {release["tag_name"]} — restarting')
        # Hard exit: skip atexit so the new daemon owns the PID file.
        os._exit(0)


# ── Public entry point ───────────────────────────────────────────────────────

def trigger(host: str, port: int) -> dict:
    """Check for a newer release and start a background update if one exists.

    Returns a status dict immediately — the actual download/restart happens
    in a daemon thread after this returns.
    """
    if not _install.is_frozen():
        return {
            'status': 'error',
            'error': (
                'Self-update requires the installed binary. '
                'Run: pip install --upgrade clashcontrol-engine'
            ),
        }

    if _update_lock.locked():
        return {'status': 'updating', 'current': __version__}

    try:
        release = _fetch_release()
    except Exception as exc:
        return {'status': 'error', 'error': f'Cannot reach GitHub: {exc}'}

    tag = release.get('tag_name', '').lstrip('v')
    if not is_newer(tag, __version__):
        return {'status': 'up_to_date', 'version': __version__}

    threading.Thread(target=_run_update, args=(host, port), daemon=True).start()
    return {'status': 'updating', 'current': __version__, 'latest': tag}
