"""
Per-user registration of the ``clashcontrol://`` URL scheme.

Registering the scheme lets ClashControl's in-browser "Connect" button
launch the local engine on demand — clicking the button navigates to a
``clashcontrol://start`` URL, the OS hands it to the handler installed
here, and the handler spawns the daemon.

All three platforms register at the user scope (no sudo / admin prompt):

- **macOS**  — synthesises a minimal ``.app`` bundle under
  ``~/Applications/ClashControl Engine.app`` and asks LaunchServices to
  register it via ``lsregister``.
- **Linux**  — writes a ``.desktop`` file to
  ``~/.local/share/applications`` with ``MimeType=x-scheme-handler/clashcontrol``
  and calls ``xdg-mime default`` plus ``update-desktop-database``.
- **Windows** — writes per-user registry keys under
  ``HKCU\\Software\\Classes\\clashcontrol``.

The registered command points back at the current Python interpreter so
``pip install --user`` installs and virtualenvs keep working.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

from . import daemon

__all__ = [
    "SCHEME",
    "install_protocol",
    "uninstall_protocol",
    "protocol_status",
    "protocol_path",
    "is_protocol_url",
]

SCHEME = "clashcontrol"
_BUNDLE_ID = "io.clashcontrol.engine"
_APP_NAME = "ClashControl Engine"


def is_protocol_url(value: str) -> bool:
    """Return True if *value* looks like a ``clashcontrol://...`` URL."""
    return isinstance(value, str) and value.lower().startswith(f"{SCHEME}://")


# ── macOS: synthesise a minimal .app bundle ────────────────────────

def _macos_app_dir() -> Path:
    return Path.home() / "Applications" / f"{_APP_NAME}.app"


def _macos_install() -> Path:
    app = _macos_app_dir()
    macos_dir = app / "Contents" / "MacOS"
    macos_dir.mkdir(parents=True, exist_ok=True)

    launcher = macos_dir / "clashcontrol-engine-launcher"
    argv = daemon.engine_argv("--daemon")
    launcher.write_text(
        "#!/bin/bash\n"
        f'exec {" ".join(_shell_quote(a) for a in argv)}\n'
    )
    launcher.chmod(0o755)

    info_plist = app / "Contents" / "Info.plist"
    info_plist.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" '
        '"http://www.apple.com/DTDs/PropertyList-1.0.dtd">\n'
        '<plist version="1.0">\n'
        '<dict>\n'
        f'    <key>CFBundleIdentifier</key>\n    <string>{_BUNDLE_ID}</string>\n'
        f'    <key>CFBundleName</key>\n    <string>{_APP_NAME}</string>\n'
        '    <key>CFBundleExecutable</key>\n'
        '    <string>clashcontrol-engine-launcher</string>\n'
        '    <key>CFBundlePackageType</key>\n    <string>APPL</string>\n'
        '    <key>CFBundleInfoDictionaryVersion</key>\n    <string>6.0</string>\n'
        '    <key>CFBundleVersion</key>\n    <string>1</string>\n'
        '    <key>LSUIElement</key>\n    <true/>\n'
        '    <key>CFBundleURLTypes</key>\n'
        '    <array>\n'
        '        <dict>\n'
        f'            <key>CFBundleURLName</key>\n            <string>{_BUNDLE_ID}</string>\n'
        '            <key>CFBundleURLSchemes</key>\n'
        '            <array>\n'
        f'                <string>{SCHEME}</string>\n'
        '            </array>\n'
        '        </dict>\n'
        '    </array>\n'
        '</dict>\n'
        '</plist>\n'
    )

    _run([
        "/System/Library/Frameworks/CoreServices.framework/Frameworks/"
        "LaunchServices.framework/Support/lsregister",
        "-f", str(app),
    ], check=False)
    return app


def _macos_uninstall() -> bool:
    app = _macos_app_dir()
    if not app.exists():
        return False
    _run([
        "/System/Library/Frameworks/CoreServices.framework/Frameworks/"
        "LaunchServices.framework/Support/lsregister",
        "-u", str(app),
    ], check=False)
    shutil.rmtree(app, ignore_errors=True)
    return True


# ── Linux: XDG .desktop + xdg-mime ─────────────────────────────────

def _linux_desktop_path() -> Path:
    base = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return Path(base) / "applications" / "clashcontrol-engine.desktop"


def _linux_install() -> Path:
    path = _linux_desktop_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    # %u is the per-desktop-entry placeholder for a single URL; the engine
    # ignores its value and just starts the daemon.
    argv = daemon.engine_argv("--open", "%u")
    exec_cmd = " ".join(_shell_quote(a) for a in argv)
    path.write_text(
        "[Desktop Entry]\n"
        "Type=Application\n"
        f"Name={_APP_NAME}\n"
        "Comment=Local clash detection server for ClashControl\n"
        f"Exec={exec_cmd}\n"
        "Terminal=false\n"
        "NoDisplay=true\n"
        f"MimeType=x-scheme-handler/{SCHEME};\n"
        "Categories=Network;\n"
    )

    apps_dir = path.parent
    _run(["update-desktop-database", str(apps_dir)], check=False)
    _run(
        ["xdg-mime", "default", path.name, f"x-scheme-handler/{SCHEME}"],
        check=False,
    )
    return path


def _linux_uninstall() -> bool:
    path = _linux_desktop_path()
    if not path.exists():
        return False
    path.unlink()
    _run(["update-desktop-database", str(path.parent)], check=False)
    return True


# ── Windows: HKCU registry keys ────────────────────────────────────

def _windows_install() -> str:
    import winreg  # type: ignore[import-not-found]

    argv = daemon.engine_argv("--open", "%1")
    # On a pip install, prefer pythonw.exe (no console flash) over python.exe.
    if not getattr(sys, "frozen", False):
        pyw = Path(argv[0]).with_name("pythonw.exe")
        if pyw.exists():
            argv[0] = str(pyw)
    command = " ".join(
        f'"{a}"' if (" " in a or a == "%1") else a for a in argv
    )

    key_path = f"Software\\Classes\\{SCHEME}"
    with winreg.CreateKey(winreg.HKEY_CURRENT_USER, key_path) as key:
        winreg.SetValueEx(key, None, 0, winreg.REG_SZ, f"URL:{_APP_NAME}")
        winreg.SetValueEx(key, "URL Protocol", 0, winreg.REG_SZ, "")
    with winreg.CreateKey(
        winreg.HKEY_CURRENT_USER, f"{key_path}\\shell\\open\\command"
    ) as key:
        winreg.SetValueEx(key, None, 0, winreg.REG_SZ, command)
    return f"HKCU\\{key_path}"


def _windows_uninstall() -> bool:
    import winreg  # type: ignore[import-not-found]

    key_path = f"Software\\Classes\\{SCHEME}"
    removed = False
    for sub in (
        f"{key_path}\\shell\\open\\command",
        f"{key_path}\\shell\\open",
        f"{key_path}\\shell",
        key_path,
    ):
        try:
            winreg.DeleteKey(winreg.HKEY_CURRENT_USER, sub)
            removed = True
        except FileNotFoundError:
            continue
        except OSError:
            continue
    return removed


def _windows_status() -> bool:
    import winreg  # type: ignore[import-not-found]

    try:
        with winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            f"Software\\Classes\\{SCHEME}\\shell\\open\\command",
        ):
            return True
    except FileNotFoundError:
        return False
    except OSError:
        return False


# ── Public dispatch ─────────────────────────────────────────────────

def _backend():
    if sys.platform == "darwin":
        return "macos"
    if sys.platform == "win32":
        return "windows"
    return "linux"


def protocol_path():
    """Return a human-readable path/location for the registered handler."""
    b = _backend()
    if b == "macos":
        return _macos_app_dir()
    if b == "windows":
        return f"HKCU\\Software\\Classes\\{SCHEME}"
    return _linux_desktop_path()


def protocol_status() -> bool:
    """Return True if the ``clashcontrol://`` handler is currently registered."""
    b = _backend()
    if b == "macos":
        return _macos_app_dir().exists()
    if b == "windows":
        return _windows_status()
    return _linux_desktop_path().exists()


def install_protocol():
    """Register the ``clashcontrol://`` URL scheme handler for the current user."""
    b = _backend()
    if b == "macos":
        return _macos_install()
    if b == "windows":
        return _windows_install()
    return _linux_install()


def uninstall_protocol() -> bool:
    """Remove the URL scheme registration if present."""
    b = _backend()
    if b == "macos":
        return _macos_uninstall()
    if b == "windows":
        return _windows_uninstall()
    return _linux_uninstall()


# ── Helpers ─────────────────────────────────────────────────────────

def _shell_quote(arg: str) -> str:
    if not arg:
        return "''"
    if all(c.isalnum() or c in "@%+=:,./-_" for c in arg):
        return arg
    return "'" + arg.replace("'", "'\\''") + "'"


def _run(cmd, check=True):
    """Run a subprocess silently. Returns CompletedProcess or None on miss."""
    if not shutil.which(cmd[0]) and not Path(cmd[0]).exists():
        return None
    try:
        return subprocess.run(
            cmd,
            check=check,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None
