#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Post the benchmark results and the CodSpeed profiles of one commit to the benchmarks website.

The website writes them to its store. The job proves its identity with a GitHub Actions identity
token (OIDC), so there is no secret: the website checks the signature of the token, and accepts
only the workflows of `develop` that it trusts. The commit, the branch and the repository come
from the token, not from this script.

The request is `POST {BENCH_SITE_INGEST_URL}/api/ingest` with a multipart form:

- `commit`: the metadata of the commit, from `git show`. Always sent, so that each post is
  sufficient by itself.
- `results`: the JSONL file of `--results`, if it has records.
- `profiles`: a `tar.gz` of the Callgrind files under the `--profiles` directory, if there are any.

The job needs `permissions: id-token: write`. The audience of the token is the URL of the website.

The script does nothing when `BENCH_SITE_INGEST_URL` is not set. A post that fails with a
transport error, HTTP 429 or HTTP 5xx is sent again a few times, each time with a new token. The
website updates rows by their key, so a second post of the same data changes nothing. Any other
failure prints the answer of the website and exits non-zero.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import secrets
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from pathlib import Path

# The website refuses a larger request. One CodSpeed shard is far below this.
MAX_BODY_BYTES = 32 * 1024 * 1024

# Waits between attempts. Other jobs of the run post at the same time, and the website takes one
# post at a time, so a post can wait for some minutes.
_RETRY_DELAYS_S = (5, 15, 45, 90, 120)

_COMMIT_FORMATS = {
    "timestamp": "%cI",
    "message": "%s",
    "author_name": "%an",
    "author_email": "%ae",
    "committer_name": "%cn",
    "committer_email": "%ce",
    "tree_sha": "%T",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", type=Path, help="The JSONL file of benchmark records to post.")
    parser.add_argument(
        "--profiles",
        type=Path,
        help="A directory with the Callgrind files of a CodSpeed run (`<pid>.out`), at any depth.",
    )
    parser.add_argument(
        "--commit-sha",
        default=os.environ.get("GITHUB_SHA"),
        help="The commit of the run (default: $GITHUB_SHA). The website refuses another commit.",
    )
    parser.add_argument("--git-dir", type=Path, help="The checkout to read the commit from.")
    parser.add_argument(
        "--timeout",
        type=float,
        default=600.0,
        help="Timeout of each post in seconds (default: 600).",
    )
    return parser.parse_args()


def build_commit(sha: str, git_dir: Path | None) -> dict[str, str]:
    """The metadata of one commit, from `git show`."""
    commit = {"commit_sha": sha}

    for name, fmt in _COMMIT_FORMATS.items():
        result = subprocess.run(
            ["git", "show", "-s", f"--format={fmt}", sha],
            cwd=git_dir,
            check=True,
            capture_output=True,
            text=True,
        )
        commit[name] = result.stdout.strip()

    return commit


def profile_files(directory: Path) -> list[Path]:
    """The Callgrind files under `directory`. CodSpeed names them `<pid>.out`; the Valgrind logs
    next to them end in `.log`."""
    return sorted(
        path for path in directory.rglob("*.out*") if path.is_file() and not path.name.endswith((".log", ".gz"))
    )


def profile_archive(directory: Path) -> bytes | None:
    """A `tar.gz` of the Callgrind files under `directory`, or None if there are none."""
    files = profile_files(directory)
    if not files:
        return None

    out = io.BytesIO()
    with tarfile.open(fileobj=out, mode="w:gz") as tar:
        for path in files:
            tar.add(path, arcname=str(path.relative_to(directory)))

    return out.getvalue()


def multipart(parts: dict[str, tuple[str, bytes]]) -> tuple[str, bytes]:
    """Encode `{name: (content type, data)}` as a multipart form. Returns the content type of the
    request and its body."""
    boundary = f"vortex-bench-{secrets.token_hex(16)}"
    body = io.BytesIO()

    for name, (content_type, data) in parts.items():
        body.write(f"--{boundary}\r\n".encode())
        body.write(f'Content-Disposition: form-data; name="{name}"; filename="{name}"\r\n'.encode())
        body.write(f"Content-Type: {content_type}\r\n\r\n".encode())
        body.write(data)
        body.write(b"\r\n")

    body.write(f"--{boundary}--\r\n".encode())
    return f"multipart/form-data; boundary={boundary}", body.getvalue()


def identity_token(audience: str, timeout: float = 30.0) -> str:
    """Ask GitHub for the identity token of this job, for one audience."""
    url = os.environ.get("ACTIONS_ID_TOKEN_REQUEST_URL")
    bearer = os.environ.get("ACTIONS_ID_TOKEN_REQUEST_TOKEN")
    if not url or not bearer:
        raise SystemExit("error: no identity token is available. The job needs `permissions: id-token: write`.")

    request = urllib.request.Request(
        f"{url}&audience={urllib.parse.quote(audience, safe='')}",
        headers={"authorization": f"bearer {bearer}", "accept": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)["value"]


def _send(url: str, token: str, content_type: str, body: bytes, timeout: float) -> bytes:
    """Issue one post and return the body. Raises on any non-2xx or transport error."""
    request = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "authorization": f"Bearer {token}",
            "content-type": content_type,
            "accept": "application/json",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def _is_transient(exc: Exception) -> bool:
    """True for a transport error, HTTP 429 or HTTP 5xx. Other HTTP errors, such as 403 for a job
    that the website does not trust, do not change when the post is sent again."""
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code == 429 or exc.code >= 500
    return isinstance(exc, OSError)


def _delay(exc: Exception, default: float) -> float:
    """The wait before the next attempt: the `Retry-After` of the website if it gives one."""
    if isinstance(exc, urllib.error.HTTPError):
        value = exc.headers.get("retry-after") if exc.headers else None
        if value and value.isdigit():
            return max(default, float(value))
    return default


def _describe(exc: Exception) -> str:
    if isinstance(exc, urllib.error.HTTPError):
        try:
            answer = exc.read().decode(errors="replace").strip()
        except Exception:  # noqa: BLE001 -- the status alone is still useful.
            answer = ""
        return f"HTTP {exc.code} {answer}".strip()
    return str(exc)


def post(
    base_url: str,
    parts: dict[str, tuple[str, bytes]],
    timeout: float,
    *,
    token: Callable[[str], str] = identity_token,
    send: Callable[[str, str, str, bytes, float], bytes] = _send,
    delays: tuple[float, ...] = _RETRY_DELAYS_S,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, object] | None:
    """Post the parts. Returns the answer of the website, or None when the post fails."""
    base = base_url.rstrip("/")
    content_type, body = multipart(parts)
    if len(body) > MAX_BODY_BYTES:
        print(
            f"error: the post has {len(body) >> 20} MiB and the website takes "
            f"{MAX_BODY_BYTES >> 20} MiB at most. Post fewer profiles at a time.",
            file=sys.stderr,
        )
        return None

    for attempt in range(len(delays) + 1):
        try:
            # A token is valid for a short time, so each attempt asks for a new one.
            return json.loads(send(f"{base}/api/ingest", token(base), content_type, body, timeout))
        except Exception as exc:  # noqa: BLE001 -- reported through the exit code.
            described = _describe(exc)
            if attempt == len(delays) or not _is_transient(exc):
                print(f"error: post to {base} failed: {described}", file=sys.stderr)
                return None
            wait = _delay(exc, delays[attempt])
            print(f"warning: post to {base} failed ({described}); retrying in {wait:g}s", file=sys.stderr)
            sleep(wait)
    return None


def main() -> int:
    args = parse_args()
    base_url = os.environ.get("BENCH_SITE_INGEST_URL")
    if not base_url:
        print("BENCH_SITE_INGEST_URL is not set; skipping the post to the benchmarks website.")
        return 0
    if not args.commit_sha:
        print("error: give --commit-sha or set GITHUB_SHA", file=sys.stderr)
        return 1

    commit = build_commit(args.commit_sha.strip().lower(), args.git_dir)
    parts = {"commit": ("application/json", json.dumps(commit).encode())}

    if args.results is not None:
        results = args.results.read_bytes()
        if results.strip():
            parts["results"] = ("application/jsonl", results)
        else:
            print(f"{args.results} has no records; posting only the commit.")

    if args.profiles is not None:
        archive = profile_archive(args.profiles)
        if archive is None:
            print(f"warning: no Callgrind files under {args.profiles}", file=sys.stderr)
        else:
            parts["profiles"] = ("application/gzip", archive)

    answer = post(base_url, parts, args.timeout)
    if answer is None:
        return 1

    print(
        f"posted {', '.join(parts)} for {answer.get('commit_sha')}: "
        f"inserted {answer.get('inserted')}, updated {answer.get('updated')}, "
        f"data version {answer.get('version')}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
