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


def test_asks_for_a_new_identity_token_for_each_attempt(refresh, monkeypatch):
    tokens: list[str] = []

    def http(method: str, url: str, token: str | None, timeout: float) -> bytes:
        if method == "POST":
            assert token is not None
            tokens.append(token)
            if len(tokens) == 1:
                raise _http_error(503)
        return b'{"groups":[]}'

    monkeypatch.setattr(refresh, "_http", http)
    minted = iter(["first", "second"])

    assert refresh.refresh_site_cache(BASE, lambda: next(minted), 1.0, sleep=lambda _: None)
    assert tokens == ["first", "second"]


def test_uses_the_identity_token_when_there_is_no_shared_token(refresh, monkeypatch):
    monkeypatch.setenv("BENCH_SITE_BASE_URL", f"{BASE}/")
    monkeypatch.delenv("BENCH_REVALIDATE_TOKEN", raising=False)
    monkeypatch.setenv("ACTIONS_ID_TOKEN_REQUEST_URL", "https://token.example/?api-version=2")
    monkeypatch.setenv("ACTIONS_ID_TOKEN_REQUEST_TOKEN", "runtime")
    monkeypatch.setattr("sys.argv", ["refresh-bench-site.py"])
    audiences: list[str] = []
    calls: list[tuple[str, str]] = []

    def identity(audience: str, timeout: float = 30.0) -> str:
        audiences.append(audience)
        return "identity"

    def refresh_site_cache(base_url: str, token, timeout: float) -> bool:
        calls.append((base_url, token()))
        return True

    monkeypatch.setattr(refresh, "identity_token", identity)
    monkeypatch.setattr(refresh, "refresh_site_cache", refresh_site_cache)

    assert refresh.main() == 0
    assert calls == [(f"{BASE}/", "identity")]

    # The audience is the URL of the website with no `/` at the end.
    assert set(audiences) == {BASE}


def test_does_nothing_without_a_token_of_either_kind(refresh, monkeypatch, capsys):
    monkeypatch.setenv("BENCH_SITE_BASE_URL", BASE)
    monkeypatch.delenv("BENCH_REVALIDATE_TOKEN", raising=False)
    monkeypatch.delenv("ACTIONS_ID_TOKEN_REQUEST_URL", raising=False)
    monkeypatch.setattr("sys.argv", ["refresh-bench-site.py"])

    assert refresh.main() == 0
    assert "skipping site refresh" in capsys.readouterr().out
