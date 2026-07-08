"""Tests for the self-updater helpers."""
import pytest

from clashcontrol_engine.updater import is_newer, _parse_version


# ── Version parsing ───────────────────────────────────────────────

def test_parse_version_plain():
    assert _parse_version('0.2.6') == (0, 2, 6)


def test_parse_version_v_prefix():
    assert _parse_version('v1.10.3') == (1, 10, 3)


def test_parse_version_prerelease_suffix():
    assert _parse_version('0.2.6-rc1') == (0, 2, 6)
    assert _parse_version('0.2.6+build5') == (0, 2, 6)


def test_parse_version_malformed_raises():
    with pytest.raises(ValueError):
        _parse_version('not-a-version')
    with pytest.raises(ValueError):
        _parse_version('')


# ── is_newer ──────────────────────────────────────────────────────

def test_is_newer_basic():
    assert is_newer('0.2.7', '0.2.6') is True
    assert is_newer('0.3.0', '0.2.9') is True
    assert is_newer('1.0.0', '0.9.9') is True
    assert is_newer('0.2.6', '0.2.6') is False
    assert is_newer('0.2.5', '0.2.6') is False


def test_is_newer_prerelease_tags():
    # Pre-release suffix is stripped for the numeric compare
    assert is_newer('0.2.6-rc1', '0.2.5') is True
    assert is_newer('0.2.6-rc1', '0.2.6') is False
    assert is_newer('v0.2.7-beta.2', '0.2.6') is True


def test_is_newer_malformed_latest_is_not_newer():
    # A mistyped GitHub tag must never break the update check
    assert is_newer('garbage', '0.2.6') is False
    assert is_newer('', '0.2.6') is False
    assert is_newer(None, '0.2.6') is False
    assert is_newer('0.2.x', '0.2.6') is False


def test_is_newer_malformed_current_is_not_newer():
    assert is_newer('0.2.7', 'garbage') is False
