"""Tests for the self-installation module."""
import os
import shutil
import sys
from pathlib import Path

import pytest

from clashcontrol_engine import daemon, install


# ── Fixtures ───────────────────────────────────────────────────────

@pytest.fixture
def isolated_install(tmp_path, monkeypatch):
    """Redirect install.install_dir() into a temp directory."""
    target = tmp_path / "installdir"
    monkeypatch.setenv("CC_ENGINE_INSTALL_DIR", str(target))
    yield target


@pytest.fixture
def fake_frozen(monkeypatch, tmp_path):
    """Simulate running inside a PyInstaller frozen binary.

    Creates a fake "download" binary under tmp_path/Downloads and points
    sys.executable at it, plus sets sys.frozen so install.is_frozen()
    returns True. Returns the path to the fake downloaded binary.
    """
    downloads = tmp_path / "Downloads"
    downloads.mkdir()
    fake_exe = downloads / "clashcontrol-engine-downloaded.exe"
    fake_exe.write_bytes(b"#!/bin/sh\necho fake cce binary\n")
    if os.name != "nt":
        os.chmod(fake_exe, 0o755)

    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(fake_exe))
    yield fake_exe


# ── install_dir / install_path ────────────────────────────────────

def test_install_dir_respects_env_override(isolated_install):
    assert install.install_dir() == isolated_install


def test_install_path_under_install_dir(isolated_install):
    assert install.install_path().parent == isolated_install
    # Name varies by platform but always shares the engine stem
    assert install.install_path().name.startswith("clashcontrol-engine")


def test_installed_binary_returns_none_when_absent(isolated_install):
    assert install.installed_binary() is None


def test_installed_binary_returns_path_when_present(isolated_install):
    p = install.install_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"binary")
    assert install.installed_binary() == p


# ── is_frozen ──────────────────────────────────────────────────────

def test_is_frozen_false_in_pytest():
    # pytest runs under a regular python interpreter
    assert install.is_frozen() is False


def test_is_frozen_true_when_sys_frozen_set(monkeypatch):
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    assert install.is_frozen() is True


# ── ensure_installed ──────────────────────────────────────────────

def test_ensure_installed_noop_when_not_frozen(isolated_install):
    assert install.ensure_installed() is None
    assert not install.install_path().exists()


def test_ensure_installed_copies_frozen_binary(isolated_install, fake_frozen):
    result = install.ensure_installed()
    assert result is not None
    assert result == install.install_path()
    assert result.exists()
    # Content preserved
    assert result.read_bytes() == fake_frozen.read_bytes()
    # Source is untouched
    assert fake_frozen.exists()


def test_ensure_installed_sets_executable_bit_on_unix(isolated_install, fake_frozen):
    if os.name == "nt":
        pytest.skip("no executable bit on Windows")
    result = install.ensure_installed()
    assert result is not None
    mode = result.stat().st_mode & 0o777
    assert mode & 0o100, f"expected exec bit, got {oct(mode)}"


def test_ensure_installed_idempotent_from_install_path(isolated_install, monkeypatch, tmp_path):
    """When the current process is already running from the install path,
    ensure_installed should detect that and return without recopying."""
    target = install.install_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"already installed")
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(target))

    result = install.ensure_installed()
    assert result == target
    # Content unchanged (no self-overwrite corruption)
    assert target.read_bytes() == b"already installed"


def test_ensure_installed_overwrites_existing(isolated_install, fake_frozen):
    """A fresh download should replace an older installed binary."""
    target = install.install_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"old version")

    result = install.ensure_installed()
    assert result == target
    assert target.read_bytes() == fake_frozen.read_bytes()
    assert b"old version" not in target.read_bytes()


# ── remove_installed ──────────────────────────────────────────────

def test_remove_installed_when_absent(isolated_install):
    assert install.remove_installed() is False


def test_remove_installed_when_present(isolated_install):
    p = install.install_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"binary")
    assert install.remove_installed() is True
    assert not p.exists()


# ── daemon.engine_argv integration ────────────────────────────────

def test_engine_argv_unfrozen_uses_python_module():
    """Non-frozen path: argv invokes ``python -m clashcontrol_engine``."""
    argv = daemon.engine_argv("--foreground", "--port", "1234")
    assert argv[0] == sys.executable
    assert argv[1:3] == ["-m", "clashcontrol_engine"]
    assert argv[-2:] == ["--port", "1234"]


def test_engine_argv_frozen_prefers_canonical_over_sys_executable(
    isolated_install, fake_frozen
):
    """Once ensure_installed has run, engine_argv must point at the install
    location — not at the (soon to be deleted) downloaded binary."""
    install.ensure_installed()
    argv = daemon.engine_argv("--foreground")
    assert argv[0] == str(install.install_path())
    assert argv[0] != str(fake_frozen)
    assert argv[1:] == ["--foreground"]


def test_engine_argv_frozen_falls_back_when_not_yet_installed(
    isolated_install, fake_frozen
):
    """Before ensure_installed runs (install.install_path doesn't exist
    yet), engine_argv should fall back to sys.executable so the first
    pre-install spawn can still happen if anything needs it."""
    assert not install.install_path().exists()
    argv = daemon.engine_argv("--foreground")
    assert argv[0] == str(fake_frozen)
