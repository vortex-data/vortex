# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Unit tests for post-ingest.py's best-effort site-cache refresh hook.

These are pure-stdlib tests (no Docker, no psycopg): they exercise
`refresh_site_cache` by monkeypatching `urllib.request.urlopen`, asserting the
bearer header is sent and that every failure is swallowed so the hook can never
change the ingest exit code.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent


def _load_module(filename: str, modname: str):
    path = SCRIPTS_DIR / filename
    spec = importlib.util.spec_from_file_location(modname, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


post_ingest = _load_module("post-ingest.py", "post_ingest")


class _FakeResponse:
    def __init__(self, body: bytes = b"{}"):
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_refresh_posts_revalidate_with_bearer(monkeypatch):
    calls: list[tuple[str, str, dict[str, str], bytes | None]] = []

    def fake_urlopen(req, timeout=None):
        calls.append((req.full_url, req.get_method(), dict(req.headers), req.data))
        return _FakeResponse(b'{"groups": []}')

    monkeypatch.setattr(post_ingest.urllib.request, "urlopen", fake_urlopen)
    post_ingest.refresh_site_cache("https://example.test/", "tok", 5.0)

    revalidate = [c for c in calls if c[0].endswith("/api/revalidate")]
    assert revalidate, "expected a POST to /api/revalidate"
    # The revalidate request must use the POST method.
    assert revalidate[0][1] == "POST", "revalidate must be a POST request"
    # urllib title-cases header keys, so the bearer lives under "Authorization".
    assert revalidate[0][2].get("Authorization") == "Bearer tok"
    # Revalidate must be the first request issued, before any warm GETs.
    assert calls[0][0].endswith("/api/revalidate"), "revalidate must precede warm GETs"


def test_refresh_skips_warm_when_revalidate_fails(monkeypatch):
    """When the revalidate POST fails, no warm GET must be issued.

    Warming after a failed flush would repopulate the Data Cache with stale data.
    The function must still return normally (never raise).
    """
    calls: list[str] = []

    def fake_urlopen(req, timeout=None):
        calls.append(req.full_url)
        if len(calls) == 1:
            raise OSError("revalidate failed")
        return _FakeResponse(b'{"groups": []}')

    monkeypatch.setattr(post_ingest.urllib.request, "urlopen", fake_urlopen)
    result = post_ingest.refresh_site_cache("https://example.test", "tok", 5.0)

    # Must not raise.
    assert result is None
    # Only the one revalidate attempt should have been made; no warm GETs follow.
    assert len(calls) == 1, f"expected 1 call (the failed revalidate), got {len(calls)}: {calls}"
    assert calls[0].endswith("/api/revalidate")


def test_refresh_swallows_all_failures(monkeypatch):
    def boom(req, timeout=None):
        raise OSError("connection refused")

    monkeypatch.setattr(post_ingest.urllib.request, "urlopen", boom)
    # Must not raise: a cache-refresh failure can never fail an ingest.
    assert post_ingest.refresh_site_cache("https://example.test", "tok", 5.0) is None


def test_warm_default_windows_issues_expected_gets(monkeypatch):
    """The warm pass GETs the landing page, /api/groups, and one /api/group/{slug}?n=100
    per discovered slug.

    Order is non-deterministic (ThreadPoolExecutor), so we assert the SET of
    warm URLs rather than a fixed sequence.
    """
    captured: list[str] = []

    def fake_urlopen(req, timeout=None):
        captured.append(req.full_url)
        if req.full_url.endswith("/api/groups"):
            body = b'{"groups":[{"slug":"g1"},{"slug":"g2"}]}'
        else:
            body = b"{}"
        return _FakeResponse(body)

    monkeypatch.setattr(post_ingest.urllib.request, "urlopen", fake_urlopen)
    post_ingest.refresh_site_cache("https://example.test/", "tok", 5.0)

    urls = set(captured)
    # The revalidate POST is always first (tested separately); include it here
    # so the assertion list is complete.
    assert "https://example.test/api/revalidate" in urls
    assert "https://example.test/" in urls
    assert "https://example.test/api/groups" in urls
    assert "https://example.test/api/group/g1?n=100" in urls
    assert "https://example.test/api/group/g2?n=100" in urls


def test_main_postgres_refresh_failure_still_exits_zero(monkeypatch, tmp_path):
    """When refresh_site_cache raises (or fails), _main_postgres still returns 0.

    This pins the "best-effort, never changes the exit code" contract: even if
    the refresh throws, the successful write must be reported as exit code 0.
    """
    import types

    # Provide the required env vars so the refresh branch is entered.
    monkeypatch.setenv("BENCH_SITE_BASE_URL", "https://example.test")
    monkeypatch.setenv("BENCH_REVALIDATE_TOKEN", "tok")

    # Make urlopen raise so refresh_site_cache encounters a failure.
    def boom(req, timeout=None):
        raise OSError("network error")

    monkeypatch.setattr(post_ingest.urllib.request, "urlopen", boom)

    # Stub out all DB/git dependencies so _main_postgres reaches the refresh
    # call without needing a real Postgres connection or git history.
    closed = {"n": 0}
    conn = types.SimpleNamespace(close=lambda: closed.__setitem__("n", closed["n"] + 1))
    monkeypatch.setattr(post_ingest, "read_records", lambda path: [])
    monkeypatch.setattr(
        post_ingest,
        "build_commit",
        lambda *a, **k: {
            "sha": "a" * 40,
            "timestamp": "2026-01-02T03:04:05+00:00",
            "message": "msg",
            "author_name": "A",
            "author_email": "a@example.com",
            "committer_name": "A",
            "committer_email": "a@example.com",
            "tree_sha": "0" * 40,
            "url": "https://example.com/commit/" + "a" * 40,
        },
    )
    monkeypatch.setattr(post_ingest, "connect_postgres", lambda dsn, region: conn)
    monkeypatch.setattr(post_ingest, "ingest_postgres", lambda c, commit, records: (0, 0))

    import argparse
    from pathlib import Path

    args = argparse.Namespace(
        jsonl_path=Path("x.jsonl"),
        commit_sha="a" * 40,
        repo_url="https://example.com/repo",
        git_dir=None,
        postgres="dsn",
        region=None,
        timeout=5.0,
    )
    rc = post_ingest._main_postgres(args)
    # A refresh failure must never change the ingest exit code.
    assert rc == 0
    assert closed["n"] == 1  # the connection was still closed in the finally block


def test_main_postgres_skips_refresh_when_base_url_absent(monkeypatch, tmp_path):
    """When BENCH_SITE_BASE_URL is unset, refresh_site_cache is never called."""
    import types

    monkeypatch.delenv("BENCH_SITE_BASE_URL", raising=False)
    monkeypatch.setenv("BENCH_REVALIDATE_TOKEN", "tok")

    refresh_calls: list[str] = []
    monkeypatch.setattr(post_ingest, "refresh_site_cache", lambda *a, **k: refresh_calls.append("called"))

    conn = types.SimpleNamespace(close=lambda: None)
    monkeypatch.setattr(post_ingest, "read_records", lambda path: [])
    monkeypatch.setattr(
        post_ingest,
        "build_commit",
        lambda *a, **k: {
            "sha": "b" * 40,
            "timestamp": "2026-01-02T00:00:00+00:00",
            "message": "m",
            "author_name": "B",
            "author_email": "b@example.com",
            "committer_name": "B",
            "committer_email": "b@example.com",
            "tree_sha": "0" * 40,
            "url": "https://example.com/commit/" + "b" * 40,
        },
    )
    monkeypatch.setattr(post_ingest, "connect_postgres", lambda dsn, region: conn)
    monkeypatch.setattr(post_ingest, "ingest_postgres", lambda c, commit, records: (0, 0))

    import argparse
    from pathlib import Path

    args = argparse.Namespace(
        jsonl_path=Path("x.jsonl"),
        commit_sha="b" * 40,
        repo_url="https://example.com/repo",
        git_dir=None,
        postgres="dsn",
        region=None,
        timeout=5.0,
    )
    rc = post_ingest._main_postgres(args)
    assert rc == 0
    assert refresh_calls == [], "refresh_site_cache must not be called when BENCH_SITE_BASE_URL is absent"


def test_main_postgres_skips_refresh_when_token_absent(monkeypatch, tmp_path):
    """When BENCH_REVALIDATE_TOKEN is unset, refresh_site_cache is never called."""
    import types

    monkeypatch.setenv("BENCH_SITE_BASE_URL", "https://example.test")
    monkeypatch.delenv("BENCH_REVALIDATE_TOKEN", raising=False)

    refresh_calls: list[str] = []
    monkeypatch.setattr(post_ingest, "refresh_site_cache", lambda *a, **k: refresh_calls.append("called"))

    conn = types.SimpleNamespace(close=lambda: None)
    monkeypatch.setattr(post_ingest, "read_records", lambda path: [])
    monkeypatch.setattr(
        post_ingest,
        "build_commit",
        lambda *a, **k: {
            "sha": "c" * 40,
            "timestamp": "2026-01-02T00:00:00+00:00",
            "message": "m",
            "author_name": "C",
            "author_email": "c@example.com",
            "committer_name": "C",
            "committer_email": "c@example.com",
            "tree_sha": "0" * 40,
            "url": "https://example.com/commit/" + "c" * 40,
        },
    )
    monkeypatch.setattr(post_ingest, "connect_postgres", lambda dsn, region: conn)
    monkeypatch.setattr(post_ingest, "ingest_postgres", lambda c, commit, records: (0, 0))

    import argparse
    from pathlib import Path

    args = argparse.Namespace(
        jsonl_path=Path("x.jsonl"),
        commit_sha="c" * 40,
        repo_url="https://example.com/repo",
        git_dir=None,
        postgres="dsn",
        region=None,
        timeout=5.0,
    )
    rc = post_ingest._main_postgres(args)
    assert rc == 0
    assert refresh_calls == [], "refresh_site_cache must not be called when BENCH_REVALIDATE_TOKEN is absent"
