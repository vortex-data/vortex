# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Tests for the revalidate retries of `scripts/refresh-bench-site.py`."""

import importlib.util
import urllib.error
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
BASE = "https://bench.example"


@pytest.fixture
def refresh() -> ModuleType:
    spec = importlib.util.spec_from_file_location("refresh_bench_site", REPO_ROOT / "scripts/refresh-bench-site.py")
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(f"{BASE}/api/revalidate", code, "error", None, None)  # ty: ignore[invalid-argument-type]


def _fake_site(refresh: ModuleType, monkeypatch: pytest.MonkeyPatch, posts: list[Exception | None]) -> list[str]:
    """Replace `_http`. Each POST takes the next entry of `posts`: an error to raise, or None for
    success. Every GET succeeds. Returns the log of requests."""
    log: list[str] = []

    def http(method: str, url: str, token: str | None, timeout: float) -> bytes:
        log.append(f"{method} {url.removeprefix(BASE)}")
        if method == "POST":
            assert token == "token"
            outcome = posts.pop(0)
            if outcome is not None:
                raise outcome
            return b'{"revalidated":true}'
        if url.endswith("/api/groups"):
            return b'{"groups":[{"slug":"a"}]}'
        return b"{}"

    monkeypatch.setattr(refresh, "_http", http)
    return log


@pytest.mark.parametrize(
    "error",
    [_http_error(503), _http_error(429), urllib.error.URLError("connection refused"), TimeoutError("timed out")],
)
def test_retries_transient_errors_then_warms(refresh, monkeypatch, error):
    log = _fake_site(refresh, monkeypatch, [error, error, None])
    sleeps: list[float] = []
    assert refresh.refresh_site_cache(BASE, "token", 1.0, sleep=sleeps.append)
    assert sleeps == list(refresh._RETRY_DELAYS_S[:2])
    assert log == [
        "POST /api/revalidate",
        "POST /api/revalidate",
        "POST /api/revalidate",
        "GET /",
        "GET /api/groups",
        "GET /api/group/a?n=100",
    ]


def test_does_not_retry_an_unauthorized_token(refresh, monkeypatch):
    log = _fake_site(refresh, monkeypatch, [_http_error(401)])
    sleeps: list[float] = []
    assert not refresh.refresh_site_cache(BASE, "token", 1.0, sleep=sleeps.append)
    assert sleeps == []
    assert log == ["POST /api/revalidate"]


def test_fails_without_warming_when_retries_run_out(refresh, monkeypatch):
    attempts = len(refresh._RETRY_DELAYS_S) + 1
    log = _fake_site(refresh, monkeypatch, [_http_error(502)] * attempts)
    sleeps: list[float] = []
    assert not refresh.refresh_site_cache(BASE, "token", 1.0, sleep=sleeps.append)
    assert sleeps == list(refresh._RETRY_DELAYS_S)
    assert log == ["POST /api/revalidate"] * attempts
