#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Wrap a `--gh-json-v3` JSONL file in an envelope and POST to /api/ingest.

Reads bare v3 records from a JSONL file produced by `vortex-bench --gh-json-v3`,
fills the `commit` envelope by shelling out to `git show`, and POSTs the
envelope to `<server>/api/ingest` with a bearer token.

Standard library only -- urllib, json, subprocess. No retries, no spool, no
outbox. See `benchmarks-website/planning/02-contracts.md` and
`benchmarks-website/planning/components/emitter.md`.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="POST a v3 JSONL records file to /api/ingest.",
    )
    parser.add_argument(
        "jsonl_path",
        type=Path,
        help="Path to the JSONL file written by vortex-bench --gh-json-v3.",
    )
    parser.add_argument(
        "--server",
        required=True,
        help="Server base URL, e.g. http://localhost:8080.",
    )
    parser.add_argument(
        "--commit-sha",
        required=True,
        help="40-hex commit SHA. Usually ${{ github.sha }} in CI.",
    )
    parser.add_argument(
        "--benchmark-id",
        required=True,
        help="Run identifier echoed back in run_meta.benchmark_id.",
    )
    parser.add_argument(
        "--token-env",
        default="INGEST_BEARER_TOKEN",
        help="Env var holding the bearer token (default: INGEST_BEARER_TOKEN).",
    )
    parser.add_argument(
        "--repo-url",
        default="https://github.com/vortex-data/vortex",
        help="Base repo URL used to build the commits.url field.",
    )
    parser.add_argument(
        "--git-dir",
        type=Path,
        default=None,
        help="Run `git show` in this directory (default: current directory).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="HTTP timeout in seconds (default: 30).",
    )
    return parser.parse_args()


def read_records(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as fp:
        for line_no, line in enumerate(fp, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(
                    f"{path}:{line_no}: invalid JSON: {exc}"
                ) from exc
    return records


def git_show_field(sha: str, fmt: str, cwd: Path | None) -> str:
    """Run `git show -s --format=<fmt> <sha>` and return its stdout (stripped)."""
    result = subprocess.run(
        ["git", "show", "-s", f"--format={fmt}", sha],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def build_commit(sha: str, repo_url: str, git_dir: Path | None) -> dict:
    sha = sha.strip().lower()
    if len(sha) != 40 or any(c not in "0123456789abcdef" for c in sha):
        raise SystemExit(f"commit SHA must be 40-hex lowercase, got: {sha!r}")

    timestamp = git_show_field(sha, "%cI", git_dir)
    message = git_show_field(sha, "%s", git_dir)
    author_name = git_show_field(sha, "%an", git_dir)
    author_email = git_show_field(sha, "%ae", git_dir)
    committer_name = git_show_field(sha, "%cn", git_dir)
    committer_email = git_show_field(sha, "%ce", git_dir)
    tree_sha = git_show_field(sha, "%T", git_dir)

    return {
        "sha": sha,
        "timestamp": timestamp,
        "message": message,
        "author_name": author_name,
        "author_email": author_email,
        "committer_name": committer_name,
        "committer_email": committer_email,
        "tree_sha": tree_sha,
        "url": f"{repo_url.rstrip('/')}/commit/{sha}",
    }


def post(server: str, envelope: dict, token: str, timeout: float) -> tuple[int, bytes]:
    body = json.dumps(envelope).encode("utf-8")
    url = f"{server.rstrip('/')}/api/ingest"
    request = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def main() -> int:
    args = parse_args()

    token = os.environ.get(args.token_env)
    if not token:
        print(
            f"error: env var {args.token_env} is not set",
            file=sys.stderr,
        )
        return 2

    records = read_records(args.jsonl_path)
    commit = build_commit(args.commit_sha, args.repo_url, args.git_dir)

    envelope = {
        "run_meta": {
            "benchmark_id": args.benchmark_id,
            "schema_version": SCHEMA_VERSION,
            "started_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
        "commit": commit,
        "records": records,
    }

    status, body = post(args.server, envelope, token, args.timeout)
    body_text = body.decode("utf-8", errors="replace")

    if status >= 400:
        print(
            f"error: POST {args.server}/api/ingest -> {status}\n{body_text}",
            file=sys.stderr,
        )
        return 1

    print(body_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
