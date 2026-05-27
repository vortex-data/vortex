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

Standard library only -- urllib, json, subprocess. The default envelope size
(60 MiB, just under the server's 64 MiB body limit) is sized so a single
JSONL run normally posts in one envelope -- preserving the "per-file
all-or-nothing" contract the server documents. If the JSONL is large enough
that splitting kicks in, the script emits a warning and proceeds with the
chunked semantics (per-chunk commit, mid-chunk failure leaves earlier chunks
ingested; subsequent retries re-upsert via the server's ON CONFLICT
idempotency on `measurement_id`).

Wire-contract pointers (kept in sync as a coordinated change per
`benchmarks-website/AGENTS.md`):

- `benchmarks-website/server/src/records.rs` - envelope + per-record wire
  shapes that the server deserializes.
- `vortex-bench/src/v3.rs` - bare-record producer that writes the JSONL this
  script wraps.
- `benchmarks-website/server/src/schema.rs` - `SCHEMA_VERSION` source of
  truth that the `SCHEMA_VERSION` constant below MUST equal at every bump.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

# MUST equal `benchmarks-website/server/src/schema.rs::SCHEMA_VERSION`.
# Bumping this is a coordinated change across schema.rs, records.rs, v3.rs,
# and this script. See `benchmarks-website/AGENTS.md` ("Wire shapes are a
# coordinated change") for the full list of coupled sites.
SCHEMA_VERSION = 1
# Default sized to fit comfortably under the server's 64 MiB ingest body
# limit while leaving headroom for HTTP and JSON framing overhead. The
# point is to keep a normal JSONL run in one envelope so the documented
# "per-file all-or-nothing" contract holds.
DEFAULT_MAX_ENVELOPE_BYTES = 60 * 1024 * 1024


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
    parser.add_argument(
        "--max-envelope-bytes",
        type=int,
        default=DEFAULT_MAX_ENVELOPE_BYTES,
        help=(f"Maximum encoded JSON bytes per POST before splitting records (default: {DEFAULT_MAX_ENVELOPE_BYTES})."),
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
                raise SystemExit(f"{path}:{line_no}: invalid JSON: {exc}") from exc
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


def build_envelope(run_meta: dict, commit: dict, records: list[dict]) -> dict:
    return {
        "run_meta": run_meta,
        "commit": commit,
        "records": records,
    }


def encode_envelope(envelope: dict) -> bytes:
    return json.dumps(envelope, separators=(",", ":")).encode("utf-8")


def chunk_envelopes(
    run_meta: dict,
    commit: dict,
    records: list[dict],
    max_envelope_bytes: int,
) -> list[tuple[dict, bytes]]:
    if max_envelope_bytes <= 0:
        raise SystemExit("--max-envelope-bytes must be positive")
    if not records:
        envelope = build_envelope(run_meta, commit, [])
        return [(envelope, encode_envelope(envelope))]

    # Cost model: re-encoding the growing envelope per record was O(N^2) in
    # the size of the JSONL on the CI hot path. Instead track each record's
    # encoded size once (`json.dumps(record)`) and reason about the
    # cumulative chunk size via that plus per-record separator overhead
    # plus a one-time envelope shell. Cross-check at chunk flush time by
    # encoding the actual chunk and `assert len(body) <= cap` so a
    # misestimation surfaces here, not at the server.
    shell = build_envelope(run_meta, commit, [])
    shell_bytes = len(encode_envelope(shell))
    # `,` between records inside the JSON array.
    record_sep_bytes = 1

    def encoded_size(records_chunk: list[dict]) -> int:
        env = build_envelope(run_meta, commit, records_chunk)
        return len(encode_envelope(env))

    encoded_records: list[bytes] = [json.dumps(r, separators=(",", ":")).encode("utf-8") for r in records]

    chunks: list[tuple[dict, bytes]] = []
    batch: list[dict] = []
    batch_payload_bytes = 0

    def flush_batch() -> None:
        env = build_envelope(run_meta, commit, batch)
        body = encode_envelope(env)
        assert len(body) <= max_envelope_bytes, (
            f"chunk_envelopes invariant violated: {len(body)} > {max_envelope_bytes}"
        )
        chunks.append((env, body))

    for i, record in enumerate(records):
        record_bytes = len(encoded_records[i])
        # First record in a fresh batch has no separator before it.
        delta = record_bytes if not batch else record_bytes + record_sep_bytes
        projected = shell_bytes + batch_payload_bytes + delta

        if not batch and projected > max_envelope_bytes:
            # First record in a fresh batch ALONE exceeds the cap. Refuse
            # rather than ship an over-cap chunk that the server will 413.
            raise SystemExit(
                f"single record exceeds --max-envelope-bytes "
                f"(record {i} encodes to {record_bytes} bytes; "
                f"envelope shell adds {shell_bytes}; cap is {max_envelope_bytes}). "
                f"Raise --max-envelope-bytes, or trim the record before posting."
            )

        if batch and projected > max_envelope_bytes:
            flush_batch()
            batch = [record]
            batch_payload_bytes = record_bytes
        else:
            batch.append(record)
            batch_payload_bytes += delta

    if batch:
        flush_batch()
    return chunks


def post(server: str, body: bytes, token: str, timeout: float) -> tuple[int, bytes]:
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
    run_meta = {
        "benchmark_id": args.benchmark_id,
        "schema_version": SCHEMA_VERSION,
        "started_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    chunks = chunk_envelopes(run_meta, commit, records, args.max_envelope_bytes)
    if len(chunks) > 1:
        print(
            f"warning: JSONL exceeds --max-envelope-bytes ({args.max_envelope_bytes} B); "
            f"splitting into {len(chunks)} envelopes. Each chunk commits independently on "
            "the server, so a mid-stream failure leaves earlier chunks ingested. "
            "Re-run on failure to re-upsert via measurement_id ON CONFLICT.",
            file=sys.stderr,
        )
    inserted = 0
    updated = 0
    raw_bodies: list[str] = []
    for idx, (envelope, body) in enumerate(chunks, start=1):
        if len(chunks) > 1:
            print(
                f"POST chunk {idx}/{len(chunks)} records={len(envelope['records'])} bytes={len(body)}",
                file=sys.stderr,
            )
        status, response = post(args.server, body, token, args.timeout)
        body_text = response.decode("utf-8", errors="replace")

        if status >= 400:
            print(
                f"error: POST {args.server}/api/ingest chunk {idx}/{len(chunks)} -> {status}\n{body_text}",
                file=sys.stderr,
            )
            return 1

        try:
            parsed = json.loads(body_text)
            inserted += int(parsed.get("inserted", 0))
            updated += int(parsed.get("updated", 0))
        except (TypeError, ValueError, json.JSONDecodeError):
            raw_bodies.append(body_text)

    if raw_bodies:
        print("\n".join(raw_bodies))
    else:
        print(
            json.dumps(
                {
                    "chunks": len(chunks),
                    "records": len(records),
                    "inserted": inserted,
                    "updated": updated,
                },
                separators=(",", ":"),
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
