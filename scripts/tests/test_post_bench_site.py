# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Tests for `scripts/post-bench-site.py`: the form, the profile archive and the retries."""

import email
import email.policy
import importlib.util
import io
import subprocess
import tarfile
import urllib.error
from email.message import Message
from pathlib import Path
from types import ModuleType
from typing import cast

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
BASE = "https://bench.example"
Parts = dict[str, tuple[str, bytes]]
PARTS: Parts = {"commit": ("application/json", b'{"commit_sha": "abc"}')}


@pytest.fixture
def poster() -> ModuleType:
    spec = importlib.util.spec_from_file_location("post_bench_site", REPO_ROOT / "scripts/post-bench-site.py")
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _http_error(code: int, retry_after: str | None = None) -> urllib.error.HTTPError:
    headers = Message()
    if retry_after is not None:
        headers["retry-after"] = retry_after
    return urllib.error.HTTPError(f"{BASE}/api/ingest", code, "error", headers, io.BytesIO(b'{"error":"x"}'))


class FakeSite:
    """The website and the token endpoint. Each post takes the next entry of `outcomes`: an error
    to raise, or None for success."""

    def __init__(self, outcomes: list[Exception | None]) -> None:
        self.outcomes = outcomes
        self.tokens: list[str] = []
        self.posts: list[tuple[str, str]] = []
        self.sleeps: list[float] = []

    def token(self, audience: str) -> str:
        self.tokens.append(audience)
        return f"token-{len(self.tokens)}"

    def send(self, url: str, token: str, content_type: str, body: bytes, timeout: float) -> bytes:
        self.posts.append((url, token))
        outcome = self.outcomes.pop(0)
        if outcome is not None:
            raise outcome
        return b'{"commit_sha": "abc", "inserted": {"commits": 1}, "updated": {}, "version": "v2"}'

    def post(self, poster: ModuleType, parts: Parts = PARTS) -> dict[str, object] | None:
        answer = poster.post(
            f"{BASE}/", parts, 5.0, token=self.token, send=self.send, delays=(1, 2), sleep=self.sleeps.append
        )
        return cast("dict[str, object] | None", answer)


def test_the_form_has_each_part_under_its_name(poster: ModuleType) -> None:
    content_type, body = poster.multipart(
        {
            "commit": ("application/json", b'{"a": 1}'),
            "profiles": ("application/gzip", b"\x1f\x8b\r\n--binary\x00"),
        }
    )

    message = email.message_from_bytes(
        f"Content-Type: {content_type}\r\n\r\n".encode() + body, policy=email.policy.HTTP
    )
    parts = {part.get_param("name", header="content-disposition"): part for part in message.iter_parts()}

    assert list(parts) == ["commit", "profiles"]
    assert parts["commit"].get_payload(decode=True) == b'{"a": 1}'
    assert parts["commit"].get_content_type() == "application/json"
    assert parts["profiles"].get_payload(decode=True) == b"\x1f\x8b\r\n--binary\x00"


def test_a_post_uses_the_site_url_as_the_audience(poster: ModuleType) -> None:
    site = FakeSite([None])

    answer = site.post(poster)

    assert answer is not None and answer["version"] == "v2"
    assert site.tokens == [BASE]
    assert site.posts == [(f"{BASE}/api/ingest", "token-1")]
    assert site.sleeps == []


def test_a_transient_failure_is_sent_again_with_a_new_token(poster: ModuleType) -> None:
    site = FakeSite([_http_error(503, retry_after="15"), OSError("connection reset"), None])

    assert site.post(poster) is not None

    # The first wait is the Retry-After of the website, which is longer than the default.
    assert site.sleeps == [15, 2]
    assert [token for _, token in site.posts] == ["token-1", "token-2", "token-3"]


@pytest.mark.parametrize("code", [400, 401, 403, 409, 413, 422])
def test_a_refused_post_is_not_sent_again(poster: ModuleType, code: int, capsys: pytest.CaptureFixture[str]) -> None:
    site = FakeSite([_http_error(code)])

    assert site.post(poster) is None

    assert len(site.posts) == 1
    assert site.sleeps == []
    assert f'HTTP {code} {{"error":"x"}}' in capsys.readouterr().err


def test_a_post_gives_up_after_the_last_attempt(poster: ModuleType) -> None:
    site = FakeSite([_http_error(502), _http_error(504), _http_error(503)])

    assert site.post(poster) is None

    assert len(site.posts) == 3
    assert site.sleeps == [1, 2]


def test_a_post_larger_than_the_limit_is_not_sent(poster: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(poster, "MAX_BODY_BYTES", 64)
    site = FakeSite([None])

    assert site.post(poster, {"profiles": ("application/gzip", b"x" * 100)}) is None

    assert site.posts == []


def test_the_archive_has_only_the_callgrind_files(poster: ModuleType, tmp_path: Path) -> None:
    folder = tmp_path / "codspeed-profiles-shard-1" / "profile.abc.out"
    folder.mkdir(parents=True)
    (folder / "101.out").write_text("fn=(1) main\n1 10\n")
    (folder / "102.out").write_text("fn=(1) main\n1 20\n")
    (folder / "valgrind.101.log").write_text("log")
    (tmp_path / "notes.txt").write_text("notes")

    archive = poster.profile_archive(tmp_path)

    assert archive is not None
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
        assert tar.getnames() == [
            "codspeed-profiles-shard-1/profile.abc.out/101.out",
            "codspeed-profiles-shard-1/profile.abc.out/102.out",
        ]

    assert poster.profile_archive(tmp_path / "codspeed-profiles-shard-1" / "missing") is None


def test_the_commit_metadata_comes_from_git(poster: ModuleType) -> None:
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()

    commit = poster.build_commit(sha, REPO_ROOT)

    assert commit["commit_sha"] == sha
    assert set(commit) == {
        "commit_sha",
        "timestamp",
        "message",
        "author_name",
        "author_email",
        "committer_name",
        "committer_email",
        "tree_sha",
    }
    assert len(commit["tree_sha"]) == 40
    assert "T" in commit["timestamp"]


def test_the_script_does_nothing_without_the_site_url(
    poster: ModuleType, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("BENCH_SITE_INGEST_URL", raising=False)
    monkeypatch.setattr("sys.argv", ["post-bench-site.py", "--results", "missing.jsonl"])

    assert poster.main() == 0
    assert "skipping" in capsys.readouterr().out
