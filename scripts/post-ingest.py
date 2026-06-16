#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Ingest a `--gh-json-v3` JSONL file into the benchmarks store.

Reads bare v3 records from a JSONL file produced by `vortex-bench --gh-json-v3`
and fills the `commit` fields by shelling out to `git show`. Two mutually
exclusive ingest modes select the destination substrate:

- `--server <url>` (v3): wraps the records in an envelope and POSTs to
  `<server>/api/ingest` with a bearer token. Standard library only -- urllib,
  json, subprocess.
- `--postgres <dsn>` (v4): computes the server-internal `measurement_id`
  locally (via `_measurement_id.py`) and upserts directly into the RDS
  Postgres tables with `INSERT ... ON CONFLICT (measurement_id) DO UPDATE`,
  over a verify-full TLS connection, authenticating with an RDS IAM auth token
  when the DSN carries no password. Requires `psycopg`, `boto3`, and `xxhash`
  from the project environment; these are imported lazily inside the
  `--postgres` code path so the v3 `--server` path stays standard-library-only
  under a bare `python3` (CI invokes `python3 scripts/post-ingest.py`, not
  `uv run`, and the v3 path is in production until the Phase 5 cutover). The
  PEP 723 metadata block below intentionally keeps `dependencies = []` for that
  reason; run the `--postgres` mode from the repo's uv environment.

The default envelope size (60 MiB, just under the server's 64 MiB body limit)
is sized so a single JSONL run normally posts in one envelope -- preserving the
"per-file all-or-nothing" contract the server documents. If the JSONL is large
enough that splitting kicks in, the script emits a warning and proceeds with the
chunked semantics (per-chunk commit, mid-chunk failure leaves earlier chunks
ingested; subsequent retries re-upsert via the server's ON CONFLICT idempotency
on `measurement_id`). The `--postgres` mode applies a whole JSONL file in one
transaction (all-or-nothing), with no chunking.

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
import math
import os
import subprocess
import sys
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
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
        description=(
            "Ingest a v3 JSONL records file: --server POSTs a v3 envelope to "
            "/api/ingest; --postgres upserts directly into the RDS Postgres tables."
        ),
    )
    parser.add_argument(
        "jsonl_path",
        type=Path,
        help="Path to the JSONL file written by vortex-bench --gh-json-v3.",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--server",
        help=(
            "v3 mode: server base URL, e.g. http://localhost:8080. Wraps the "
            "records in an envelope and POSTs to <server>/api/ingest with a "
            "bearer token. Mutually exclusive with --postgres."
        ),
    )
    mode.add_argument(
        "--postgres",
        metavar="DSN",
        help=(
            "v4 mode: libpq DSN for the RDS Postgres ingest target, e.g. "
            "'postgresql://bench_ingest@host:5432/benchmarks?sslmode=verify-full"
            "&sslrootcert=/path/rds-ca.pem'. Computes measurement_id locally and "
            "upserts via INSERT ... ON CONFLICT DO UPDATE. If the DSN carries no "
            "password, an RDS IAM auth token is minted for the DSN's user. "
            "Mutually exclusive with --server."
        ),
    )
    parser.add_argument(
        "--region",
        default=None,
        help=(
            "AWS region for RDS IAM token minting (--postgres mode). Precedence: "
            "this explicit --region, then the boto3 session region, then the region "
            "parsed from the RDS hostname."
        ),
    )
    parser.add_argument(
        "--commit-sha",
        required=True,
        help="40-hex commit SHA. Usually ${{ github.sha }} in CI.",
    )
    parser.add_argument(
        "--benchmark-id",
        default=None,
        help="Run identifier echoed back in run_meta.benchmark_id. Required for "
        "--server mode; unused by --postgres mode (the v4 tables have no "
        "benchmark_id column).",
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
    # The JSONL comes from the project's own `vortex-bench` CI on the same run (trusted UTF-8
    # input); a malformed line still fails loud with `path:line` context for debuggability.
    records: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return records


def git_show_field(sha: str, fmt: str, cwd: Path | None) -> str:
    """Run `git show -s --format=<fmt> <sha>` and return its stdout (stripped).

    The metadata comes from the project's own (UTF-8) git history.
    """
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


# --------------------------------------------------------------------------
# Postgres dual-write mode (--postgres)
# --------------------------------------------------------------------------
# The v4 ingest path. `psycopg`, `boto3`, and `_measurement_id` (which pulls in
# `xxhash`) are imported lazily inside the functions below so the v3 `--server`
# path above stays standard-library-only under a bare `python3`.

# Per-`kind` record field sets, mirroring the `#[serde(deny_unknown_fields)]`
# structs in `benchmarks-website/server/src/records.rs`. `required` plus
# `optional` is the exact set of fields a record of that `kind` may carry
# (besides `kind` itself); an unknown field is rejected loudly, preserving the
# v3 server's deny_unknown_fields behavior so producer/schema drift surfaces
# instead of silently dropping data. `optional` fields default to NULL.
_RECORD_FIELDS: dict[str, tuple[frozenset[str], frozenset[str]]] = {
    "query_measurement": (
        frozenset(
            {
                "commit_sha",
                "dataset",
                "query_idx",
                "storage",
                "engine",
                "format",
                "value_ns",
                "all_runtimes_ns",
            }
        ),
        frozenset(
            {
                "dataset_variant",
                "scale_factor",
                "peak_physical",
                "peak_virtual",
                "physical_delta",
                "virtual_delta",
                "env_triple",
            }
        ),
    ),
    "compression_time": (
        frozenset({"commit_sha", "dataset", "format", "op", "value_ns", "all_runtimes_ns"}),
        frozenset({"dataset_variant", "env_triple"}),
    ),
    "compression_size": (
        frozenset({"commit_sha", "dataset", "format", "value_bytes"}),
        frozenset({"dataset_variant"}),
    ),
    "random_access_time": (
        frozenset({"commit_sha", "dataset", "format", "value_ns", "all_runtimes_ns"}),
        frozenset({"env_triple"}),
    ),
    "vector_search_run": (
        frozenset(
            {
                "commit_sha",
                "dataset",
                "layout",
                "flavor",
                "threshold",
                "value_ns",
                "all_runtimes_ns",
                "matches",
                "rows_scanned",
                "bytes_scanned",
                "iterations",
            }
        ),
        frozenset({"env_triple"}),
    ),
}

_MEASUREMENT_ID_MODULE = None


def _measurement_id_module():
    """Lazily load `scripts/_measurement_id.py` by path (cached).

    Loaded by file path rather than `import _measurement_id` so it resolves
    regardless of cwd / `sys.path` (the test harness loads sibling scripts the
    same way). Importing it pulls in `xxhash`, so it only happens here on the
    `--postgres` path.
    """
    global _MEASUREMENT_ID_MODULE
    if _MEASUREMENT_ID_MODULE is None:
        import importlib.util

        path = Path(__file__).resolve().parent / "_measurement_id.py"
        spec = importlib.util.spec_from_file_location("_measurement_id", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _MEASUREMENT_ID_MODULE = module
    return _MEASUREMENT_ID_MODULE


def _validate_record_fields(record: object, index: int) -> str:
    """Validate a record's `kind` and field set; return the `kind`.

    Mirrors the server's deny_unknown_fields + required-field deserialization:
    a non-object record, an unknown `kind`, an unknown field, or a missing
    required field is a loud error keyed by the record's index.
    """
    if not isinstance(record, dict):
        raise SystemExit(f"record {index}: expected a JSON object, got {type(record).__name__}")
    kind = record.get("kind")
    # `isinstance(kind, str)` first: a non-scalar kind (e.g. a list) is unhashable
    # and would raise TypeError at the `in` membership check rather than this
    # controlled record-indexed error.
    if not isinstance(kind, str) or kind not in _RECORD_FIELDS:
        raise SystemExit(f"record {index}: unknown kind {kind!r}; expected one of {sorted(_RECORD_FIELDS)}")
    required, optional = _RECORD_FIELDS[kind]
    present = set(record) - {"kind"}
    unknown = present - required - optional
    if unknown:
        raise SystemExit(f"record {index} ({kind}): unknown field(s) {sorted(unknown)} (deny_unknown_fields)")
    missing = required - present
    if missing:
        raise SystemExit(f"record {index} ({kind}): missing required field(s) {sorted(missing)}")
    return kind


def _require_finite(value: object, field: str, kind: str, index: int) -> None:
    """Raise loudly unless `value` is a finite number.

    Guards every f64 dim that feeds a `measurement_id` hash (currently only
    `vector_search_run.threshold`). A NaN/Inf would hash differently across the
    Rust `to_bits()` and Python `struct.pack('<d', ...)` encodings, silently
    producing a duplicate row instead of an upsert; fail loud instead.
    """
    is_finite_number = isinstance(value, (int, float)) and not isinstance(value, bool)
    if is_finite_number:
        is_finite_number = math.isfinite(value)
    if not is_finite_number:
        raise SystemExit(
            f"record {index} ({kind}): {field}={value!r} is not a finite number; refusing to "
            "write a row whose measurement_id would diverge between the Rust and Python hashers "
            "(NaN/Inf bit patterns differ across struct.pack and to_bits)."
        )


# i32/i64 bounds enforced by the Postgres INTEGER/BIGINT columns and the v3 serde.
_INT32_MIN, _INT32_MAX = -(2**31), 2**31 - 1
_INT64_MIN, _INT64_MAX = -(2**63), 2**63 - 1


def _require_int(value: object, field: str, kind: str, index: int, *, bits: int) -> None:
    """Raise loudly unless `value` is a plain (non-bool) integer within i32/i64 range.

    The integer columns bind straight to INTEGER/BIGINT; psycopg adapts a Python
    float to float8 and Postgres assignment-casts (rounds) it, so a JSON float
    would silently persist a rounded value where the v3 server's serde `i32`/`i64`
    rejects it. An out-of-range integer would otherwise fail late as an uncaught
    `struct.error` (i32 hash dims via `_write_i32`) or a raw Postgres 22003
    overflow; validate the type AND width here so a malformed scalar fails loud
    (record-indexed) instead, matching the v3 serde boundary.
    """
    lo, hi = (_INT32_MIN, _INT32_MAX) if bits == 32 else (_INT64_MIN, _INT64_MAX)
    if isinstance(value, bool) or not isinstance(value, int):
        raise SystemExit(f"record {index} ({kind}): {field} must be an integer, got {value!r}")
    if not (lo <= value <= hi):
        raise SystemExit(f"record {index} ({kind}): {field}={value!r} is out of int{bits} range")


def _require_int_list(value: object, field: str, kind: str, index: int) -> None:
    """Raise loudly unless `value` is a JSON array of plain (non-bool) i64 integers.

    Guards the `all_runtimes_ns` -> `bigint[]` bind: the explicit `::bigint[]`
    cast is permissive in ways the v3 server's `Vec<i64>` serde is not. psycopg
    sends the string `"{}"` as text and the cast parses it into an empty array, a
    `[1, null]` list adapts to `{1,NULL}`, and an out-of-i64 element hits a raw
    Postgres 22003 -- each diverges from or fails differently than the v3 path.
    Validate the element type AND i64 range so a malformed value fails loud.
    """
    if not isinstance(value, list) or any(isinstance(x, bool) or not isinstance(x, int) for x in value):
        raise SystemExit(f"record {index} ({kind}): {field} must be a JSON array of integers, got {value!r}")
    if any(not (_INT64_MIN <= x <= _INT64_MAX) for x in value):
        raise SystemExit(f"record {index} ({kind}): {field} has an element out of int64 range")


def _require_str(value: object, field: str, kind: str, index: int) -> None:
    """Raise loudly unless `value` is a string. Mirrors the v3 serde `String` fields."""
    if not isinstance(value, str):
        raise SystemExit(f"record {index} ({kind}): {field} must be a string, got {value!r}")


def _require_opt_str(value: object, field: str, kind: str, index: int) -> None:
    """Raise loudly unless `value` is a string or null. Mirrors `Option<String>`."""
    if value is not None and not isinstance(value, str):
        raise SystemExit(f"record {index} ({kind}): {field} must be a string or null, got {value!r}")


def _memory_quartet_consistent(r: dict) -> bool:
    """The four `query_measurements` memory columns are all set or all absent.

    Mirrors `ingest.rs::memory_quartet_consistent`: a partial quartet is a
    validation error, not a half-populated row.
    """
    present = [
        r.get("peak_physical") is not None,
        r.get("peak_virtual") is not None,
        r.get("physical_delta") is not None,
        r.get("virtual_delta") is not None,
    ]
    return not any(present) or all(present)


# Per-kind field -> type token, mirroring the v3 server's typed serde boundary in
# `records.rs`. The direct-to-Postgres writer can no longer lean on server-side
# deserialization, so every field is type/range-validated here. Tokens:
#   "str"     required String        "opt_str"  Option<String>
#   "i32"     required i32           "i64"      required i64
#   "opt_i64" Option<i64>            "i64_list" Vec<i64> (all_runtimes_ns)
#   "f64"     finite f64 (threshold).
# Required tokens (every non-opt_*) line up with `_RECORD_FIELDS` required sets, so
# `record[field]` is always present by the time `_validate_record_values` runs.
_FIELD_TYPES: dict[str, tuple[tuple[str, str], ...]] = {
    "query_measurement": (
        ("commit_sha", "str"),
        ("dataset", "str"),
        ("dataset_variant", "opt_str"),
        ("scale_factor", "opt_str"),
        ("query_idx", "i32"),
        ("storage", "str"),
        ("engine", "str"),
        ("format", "str"),
        ("value_ns", "i64"),
        ("all_runtimes_ns", "i64_list"),
        ("peak_physical", "opt_i64"),
        ("peak_virtual", "opt_i64"),
        ("physical_delta", "opt_i64"),
        ("virtual_delta", "opt_i64"),
        ("env_triple", "opt_str"),
    ),
    "compression_time": (
        ("commit_sha", "str"),
        ("dataset", "str"),
        ("dataset_variant", "opt_str"),
        ("format", "str"),
        ("op", "str"),
        ("value_ns", "i64"),
        ("all_runtimes_ns", "i64_list"),
        ("env_triple", "opt_str"),
    ),
    "compression_size": (
        ("commit_sha", "str"),
        ("dataset", "str"),
        ("dataset_variant", "opt_str"),
        ("format", "str"),
        ("value_bytes", "i64"),
    ),
    "random_access_time": (
        ("commit_sha", "str"),
        ("dataset", "str"),
        ("format", "str"),
        ("value_ns", "i64"),
        ("all_runtimes_ns", "i64_list"),
        ("env_triple", "opt_str"),
    ),
    "vector_search_run": (
        ("commit_sha", "str"),
        ("dataset", "str"),
        ("layout", "str"),
        ("flavor", "str"),
        ("threshold", "f64"),
        ("value_ns", "i64"),
        ("all_runtimes_ns", "i64_list"),
        ("matches", "i64"),
        ("rows_scanned", "i64"),
        ("bytes_scanned", "i64"),
        ("iterations", "i32"),
        ("env_triple", "opt_str"),
    ),
}


def _validate_record_values(record: dict, kind: str, index: int) -> None:
    """Validate every field's type/range against the v3 server's serde boundary.

    Runs in `ingest_postgres`'s loop (where the record index is known) so every
    failure is reported as `record {index} ({kind}): ...`, matching the v3
    server's indexed per-record errors. Drives field type/range checks off
    `_FIELD_TYPES`, then applies the semantic checks the type alone does not cover
    (the storage enum + memory quartet for query_measurements).
    """
    for field, typ in _FIELD_TYPES[kind]:
        if typ == "str":
            _require_str(record[field], field, kind, index)
        elif typ == "opt_str":
            _require_opt_str(record.get(field), field, kind, index)
        elif typ == "i32":
            _require_int(record[field], field, kind, index, bits=32)
        elif typ == "i64":
            _require_int(record[field], field, kind, index, bits=64)
        elif typ == "opt_i64":
            if record.get(field) is not None:
                _require_int(record[field], field, kind, index, bits=64)
        elif typ == "i64_list":
            _require_int_list(record[field], field, kind, index)
        elif typ == "f64":
            _require_finite(record[field], field, kind, index)

    if kind == "query_measurement":
        if record["storage"] not in ("nvme", "s3"):
            raise SystemExit(
                f"record {index} (query_measurement): storage must be 'nvme' or 's3', got {record['storage']!r}"
            )
        if not _memory_quartet_consistent(record):
            raise SystemExit(
                f"record {index} (query_measurement): memory fields must be populated together (all four or none)"
            )


def _upsert_returning_was_update(conn, sql: str, params: tuple) -> bool:
    """Run an `INSERT ... ON CONFLICT DO UPDATE ... RETURNING (xmax = 0)` upsert
    and return whether the row was an update (vs a fresh insert).

    Classifies inserted-vs-updated atomically from the upsert itself: a fresh
    INSERT leaves the new tuple's system column `xmax = 0`, while `ON CONFLICT
    DO UPDATE` stamps `xmax` with the current transaction id. A preflight SELECT
    (the prior approach) would race the upsert under concurrent re-ingest of the
    same `measurement_id`, miscounting a concurrent loser as inserted; deriving
    the flag from the single atomic statement removes that window. A duplicate
    `measurement_id` within ONE transaction is also handled correctly: the second
    upsert locks the tuple the first created (stamping its xmax), so it is
    classified as an update, matching the v3 server. `sql` must end with
    `RETURNING (xmax = 0) AS inserted`.
    """
    row = conn.execute(sql, params).fetchone()
    return not row[0]


def _insert_query_measurement(conn, mid_mod, r: dict) -> bool:
    """Upsert a `query_measurements` row. Mirrors `ingest.rs::insert_query_measurement`.

    Record values are validated by `_validate_record_values` before dispatch.
    """
    mid = mid_mod.measurement_id_query(
        commit_sha=r["commit_sha"],
        dataset=r["dataset"],
        dataset_variant=r.get("dataset_variant"),
        scale_factor=r.get("scale_factor"),
        query_idx=r["query_idx"],
        storage=r["storage"],
        engine=r["engine"],
        format=r["format"],
    )
    return _upsert_returning_was_update(
        conn,
        """
        INSERT INTO query_measurements (
            measurement_id, commit_sha, dataset, dataset_variant, scale_factor,
            query_idx, storage, engine, format,
            value_ns, all_runtimes_ns,
            peak_physical, peak_virtual, physical_delta, virtual_delta,
            env_triple, commit_timestamp
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::bigint[], %s, %s, %s, %s, %s,
                  (SELECT timestamp FROM commits WHERE commit_sha = %s))
        ON CONFLICT (measurement_id) DO UPDATE SET
            commit_sha       = excluded.commit_sha,
            value_ns         = excluded.value_ns,
            all_runtimes_ns  = excluded.all_runtimes_ns,
            peak_physical    = excluded.peak_physical,
            peak_virtual     = excluded.peak_virtual,
            physical_delta   = excluded.physical_delta,
            virtual_delta    = excluded.virtual_delta,
            env_triple       = excluded.env_triple,
            commit_timestamp = excluded.commit_timestamp
        RETURNING (xmax = 0) AS inserted
        """,
        (
            mid,
            r["commit_sha"],
            r["dataset"],
            r.get("dataset_variant"),
            r.get("scale_factor"),
            r["query_idx"],
            r["storage"],
            r["engine"],
            r["format"],
            r["value_ns"],
            r["all_runtimes_ns"],
            r.get("peak_physical"),
            r.get("peak_virtual"),
            r.get("physical_delta"),
            r.get("virtual_delta"),
            r.get("env_triple"),
            # The denormalized `commit_timestamp` (migration 006) is resolved from the
            # `commits` row this same transaction upserted first, so the read path's
            # latest-per-series summary never sees a NULL from this writer.
            r["commit_sha"],
        ),
    )


def _insert_compression_time(conn, mid_mod, r: dict) -> bool:
    """Upsert a `compression_times` row. Mirrors `ingest.rs::insert_compression_time`."""
    mid = mid_mod.measurement_id_compression_time(
        commit_sha=r["commit_sha"],
        dataset=r["dataset"],
        dataset_variant=r.get("dataset_variant"),
        format=r["format"],
        op=r["op"],
    )
    return _upsert_returning_was_update(
        conn,
        """
        INSERT INTO compression_times (
            measurement_id, commit_sha, dataset, dataset_variant,
            format, op, value_ns, all_runtimes_ns, env_triple
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s::bigint[], %s)
        ON CONFLICT (measurement_id) DO UPDATE SET
            commit_sha      = excluded.commit_sha,
            value_ns        = excluded.value_ns,
            all_runtimes_ns = excluded.all_runtimes_ns,
            env_triple      = excluded.env_triple
        RETURNING (xmax = 0) AS inserted
        """,
        (
            mid,
            r["commit_sha"],
            r["dataset"],
            r.get("dataset_variant"),
            r["format"],
            r["op"],
            r["value_ns"],
            r["all_runtimes_ns"],
            r.get("env_triple"),
        ),
    )


def _insert_compression_size(conn, mid_mod, r: dict) -> bool:
    """Upsert a `compression_sizes` row. Mirrors `ingest.rs::insert_compression_size`."""
    mid = mid_mod.measurement_id_compression_size(
        commit_sha=r["commit_sha"],
        dataset=r["dataset"],
        dataset_variant=r.get("dataset_variant"),
        format=r["format"],
    )
    return _upsert_returning_was_update(
        conn,
        """
        INSERT INTO compression_sizes (
            measurement_id, commit_sha, dataset, dataset_variant,
            format, value_bytes
        ) VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (measurement_id) DO UPDATE SET
            commit_sha   = excluded.commit_sha,
            value_bytes  = excluded.value_bytes
        RETURNING (xmax = 0) AS inserted
        """,
        (
            mid,
            r["commit_sha"],
            r["dataset"],
            r.get("dataset_variant"),
            r["format"],
            r["value_bytes"],
        ),
    )


def _insert_random_access(conn, mid_mod, r: dict) -> bool:
    """Upsert a `random_access_times` row. Mirrors `ingest.rs::insert_random_access`."""
    mid = mid_mod.measurement_id_random_access(
        commit_sha=r["commit_sha"],
        dataset=r["dataset"],
        format=r["format"],
    )
    return _upsert_returning_was_update(
        conn,
        """
        INSERT INTO random_access_times (
            measurement_id, commit_sha, dataset, format,
            value_ns, all_runtimes_ns, env_triple
        ) VALUES (%s, %s, %s, %s, %s, %s::bigint[], %s)
        ON CONFLICT (measurement_id) DO UPDATE SET
            commit_sha      = excluded.commit_sha,
            value_ns        = excluded.value_ns,
            all_runtimes_ns = excluded.all_runtimes_ns,
            env_triple      = excluded.env_triple
        RETURNING (xmax = 0) AS inserted
        """,
        (
            mid,
            r["commit_sha"],
            r["dataset"],
            r["format"],
            r["value_ns"],
            r["all_runtimes_ns"],
            r.get("env_triple"),
        ),
    )


def _insert_vector_search(conn, mid_mod, r: dict) -> bool:
    """Upsert a `vector_search_runs` row. Mirrors `ingest.rs::insert_vector_search`.

    `threshold` is validated finite by `_validate_record_values` before dispatch.
    """
    threshold = float(r["threshold"])
    mid = mid_mod.measurement_id_vector_search(
        commit_sha=r["commit_sha"],
        dataset=r["dataset"],
        layout=r["layout"],
        flavor=r["flavor"],
        threshold=threshold,
    )
    return _upsert_returning_was_update(
        conn,
        """
        INSERT INTO vector_search_runs (
            measurement_id, commit_sha, dataset, layout, flavor, threshold,
            value_ns, all_runtimes_ns, matches, rows_scanned, bytes_scanned,
            iterations, env_triple
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s::bigint[], %s, %s, %s, %s, %s)
        ON CONFLICT (measurement_id) DO UPDATE SET
            commit_sha      = excluded.commit_sha,
            value_ns        = excluded.value_ns,
            all_runtimes_ns = excluded.all_runtimes_ns,
            matches         = excluded.matches,
            rows_scanned    = excluded.rows_scanned,
            bytes_scanned   = excluded.bytes_scanned,
            iterations      = excluded.iterations,
            env_triple      = excluded.env_triple
        RETURNING (xmax = 0) AS inserted
        """,
        (
            mid,
            r["commit_sha"],
            r["dataset"],
            r["layout"],
            r["flavor"],
            threshold,
            r["value_ns"],
            r["all_runtimes_ns"],
            r["matches"],
            r["rows_scanned"],
            r["bytes_scanned"],
            r["iterations"],
            r.get("env_triple"),
        ),
    )


# Dispatch from a record's `kind` to its per-table upsert. Keyed identically to
# `_RECORD_FIELDS`; the two maps are wired together when adding a fact table.
_APPLY_RECORD = {
    "query_measurement": _insert_query_measurement,
    "compression_time": _insert_compression_time,
    "compression_size": _insert_compression_size,
    "random_access_time": _insert_random_access,
    "vector_search_run": _insert_vector_search,
}


def _upsert_commit(conn, commit: dict) -> None:
    """Upsert the `commits` dim row. Mirrors `ingest.rs::upsert_commit`."""
    conn.execute(
        """
        INSERT INTO commits (
            commit_sha, timestamp, message, author_name, author_email,
            committer_name, committer_email, tree_sha, url
        ) VALUES (%s, %s::timestamptz, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (commit_sha) DO UPDATE SET
            timestamp       = excluded.timestamp,
            message         = excluded.message,
            author_name     = excluded.author_name,
            author_email    = excluded.author_email,
            committer_name  = excluded.committer_name,
            committer_email = excluded.committer_email,
            tree_sha        = excluded.tree_sha,
            url             = excluded.url
        """,
        (
            commit["sha"],
            commit["timestamp"],
            commit["message"],
            commit["author_name"],
            commit["author_email"],
            commit["committer_name"],
            commit["committer_email"],
            commit["tree_sha"],
            commit["url"],
        ),
    )


# Mirrors the v3 server's `WRITE_CONFLICT_ATTEMPTS`
# (`benchmarks-website/server/src/ingest.rs`). The DuckDB ingest wrapped
# `apply_envelope_once` in `retry_write_conflicts`; the DuckDB -> Postgres substrate change
# must preserve that behavior, since the dual-write CI runs ~14 concurrent writers.
_WRITE_CONFLICT_ATTEMPTS = 128


def _retry_write_conflicts(op):
    """Retry `op` on a Postgres write conflict, mirroring the v3 server's `retry_write_conflicts`.

    The v3 DuckDB ingest retried on write conflicts up to `WRITE_CONFLICT_ATTEMPTS` times; the
    Postgres writer must preserve that, because the dual-write CI runs many concurrent writers
    whose row-level `ON CONFLICT DO UPDATE` upserts touching the same `commits` / dim rows in
    conflicting orders can deadlock. The retryable analogs on Postgres are deadlock
    (`SQLSTATE 40P01`) and serialization failure (`40001`); both abort one transaction cleanly,
    so re-running the whole transaction is safe. A non-retryable error (e.g. a validation
    `SystemExit`) propagates immediately. Returns `op`'s value on the first success.
    """
    from psycopg import errors as pg_errors

    for attempt in range(1, _WRITE_CONFLICT_ATTEMPTS + 1):
        try:
            return op()
        except (pg_errors.DeadlockDetected, pg_errors.SerializationFailure):
            # The failing `op`'s `with conn.transaction()` block already rolled back, so the
            # connection is idle and the whole transaction can be retried. Re-raise on the
            # final attempt (mirrors v3 returning the error after the last try; the bare
            # retry loop mirrors v3's `std::thread::yield_now()` between attempts).
            if attempt >= _WRITE_CONFLICT_ATTEMPTS:
                raise
    raise AssertionError("unreachable: _retry_write_conflicts exited without return or raise")


def ingest_postgres(conn, commit: dict, records: list[dict]) -> tuple[int, int]:
    """Upsert a commit and its records into Postgres, retrying on write conflicts.

    Wraps `_ingest_postgres_once` in `_retry_write_conflicts` (mirroring the v3 server's
    `apply_envelope = retry_write_conflicts(apply_envelope_once)`). Returns `(inserted, updated)`
    aggregated across all fact tables.
    """
    mid_mod = _measurement_id_module()
    return _retry_write_conflicts(lambda: _ingest_postgres_once(conn, commit, records, mid_mod))


def _ingest_postgres_once(conn, commit: dict, records: list[dict], mid_mod) -> tuple[int, int]:
    """Upsert a commit and its records into Postgres in one transaction (a single attempt).

    Mirrors the v3 server's `apply_envelope_once`: upsert `commits` first, then each fact
    record, classifying each as inserted or updated. Any validation failure rolls the whole
    transaction back (all-or-nothing). Returns `(inserted, updated)` aggregated across all
    fact tables.
    """
    inserted = 0
    updated = 0
    with conn.transaction():
        _upsert_commit(conn, commit)
        for idx, record in enumerate(records):
            kind = _validate_record_fields(record, idx)
            if record["commit_sha"] != commit["sha"]:
                raise SystemExit(
                    f"record {idx} ({kind}): commit_sha {record['commit_sha']!r} does not "
                    f"match envelope commit.sha {commit['sha']!r}"
                )
            _validate_record_values(record, kind, idx)
            if _APPLY_RECORD[kind](conn, mid_mod, record):
                updated += 1
            else:
                inserted += 1
    return inserted, updated


def _region_from_host(host: str) -> str | None:
    """Parse the AWS region out of an RDS endpoint hostname.

    RDS endpoints look like `<name>.<id>.<region>.rds.amazonaws.com` (instance)
    or `<name>.proxy-<id>.<region>.rds.amazonaws.com` (proxy); the region is the
    label immediately before `rds.amazonaws.com`. Returns None for any other
    shape.
    """
    parts = host.split(".")
    if len(parts) >= 4 and parts[-3:] == ["rds", "amazonaws", "com"]:
        return parts[-4]
    return None


def _rds_iam_token(*, host: str, port: int, user: str, region: str | None) -> str:
    """Mint a short-lived RDS IAM auth token to use as the connection password."""
    import boto3

    session = boto3.session.Session()
    resolved = region or session.region_name or _region_from_host(host)
    if not resolved:
        raise SystemExit("could not determine the AWS region for the RDS IAM token; pass --region or set AWS_REGION.")
    client = session.client("rds", region_name=resolved)
    return client.generate_db_auth_token(DBHostname=host, Port=port, DBUsername=user, Region=resolved)


# The least-privilege login role the v4 CI ingest path must authenticate as
# (created by `migrations/004_ingest_role.sql`; SELECT,INSERT,UPDATE only).
# Phase-2 BAN: do not authenticate the ingest write path as `migrator` /
# `GitHubBenchmarkSchemaRole`.
_INGEST_ROLE = "bench_ingest"


def connect_postgres(dsn: str, region: str | None):
    """Open a psycopg connection to the RDS Postgres ingest target.

    Enforces the ingest contract: verify-full TLS, and authentication only as the
    least-privilege `bench_ingest` role -- always, regardless of auth method (IAM
    token or password) or host. This is the production RDS ingest connector; the
    test suite exercises ingest via `ingest_postgres` directly, so there is no
    local-affordance exception to carve out. A host-based "is this local?"
    heuristic would be unreliable anyway because libpq can resolve the host from a
    DSN `hostaddr=` or the `$PGHOST` environment variable, either of which could
    bypass a DSN-host check. Parses `dsn`; if it carries no password, mints an RDS
    IAM auth token for the DSN's user and host (the DSN is expected to also supply
    `sslrootcert` for verify-full to validate against).
    """
    import psycopg
    from psycopg import conninfo

    params = conninfo.conninfo_to_dict(dsn)

    # verify-full is the ingest TLS contract. Default an absent value, but
    # refuse a DSN that explicitly downgrades it rather than silently weakening
    # the internet-reachable ingest connection.
    sslmode = params.get("sslmode")
    if sslmode is None:
        params["sslmode"] = "verify-full"
    elif sslmode != "verify-full":
        raise SystemExit(
            f"--postgres requires sslmode=verify-full for the RDS ingest path; DSN "
            f"specified sslmode={sslmode!r}. Omit it (defaults to verify-full) or set "
            f"it to verify-full."
        )

    user = params.get("user")
    # Least-privilege: always the bench_ingest role. No host heuristic (see the
    # docstring): a misconfigured DSN -- by host=, hostaddr=, or $PGHOST -- must
    # never ingest as migrator/postgres.
    if user != _INGEST_ROLE:
        raise SystemExit(f"--postgres must connect as the least-privilege {_INGEST_ROLE!r} role; DSN user is {user!r}.")

    if not params.get("password"):
        # IAM-token path (production CI ingest): mint a token for the DSN's user.
        host = params.get("host")
        if not host:
            raise SystemExit("--postgres DSN must specify host for IAM token minting")
        try:
            port = int(params.get("port", 5432))
        except (TypeError, ValueError) as exc:
            raise SystemExit(f"--postgres DSN has a non-numeric port: {params.get('port')!r}") from exc
        params["password"] = _rds_iam_token(host=host, port=port, user=user, region=region)

    # Force `search_path=public` so the writer's unqualified table names always resolve to the
    # migration-owned `public.*` tables, regardless of any `search_path` baked into the DSN's
    # `options=-c search_path=...` or the role's default. libpq applies repeated `-c` settings
    # left-to-right (last wins), so appending ours last makes it authoritative even if the DSN
    # already set one.
    existing_options = params.get("options") or ""
    params["options"] = f"{existing_options} -c search_path=public".strip()

    conn = psycopg.connect(**params)
    # Verify the RESOLVED transport actually used TLS, not merely that the DSN requested
    # verify-full. The `sslmode` check above rejects an explicit downgrade, but it cannot stop a
    # hostless / Unix-socket DSN (host omitted, `host=/...`, or libpq resolving via `$PGHOST`)
    # from connecting over a local socket, where libpq silently ignores `sslmode`. Checking
    # `ssl_in_use` post-connect validates the connection that actually happened rather than
    # trusting the DSN string -- the same "verify the resolved state" reasoning the docstring
    # uses to reject a DSN-host heuristic -- and so also closes the `$PGHOST` / `hostaddr` bypass.
    # NOTE: `ssl_in_use` lives on the low-level libpq wrapper `conn.pgconn` (a `pq.PGconn`); it is
    # NOT on the high-level `conn.info` (`ConnectionInfo`), which exposes only host/dbname/user/etc.
    if not conn.pgconn.ssl_in_use:
        conn.close()
        raise SystemExit(
            "--postgres requires a verify-full TLS connection, but the established connection is "
            "not using TLS (a hostless or Unix-socket DSN bypasses sslmode); connect to the RDS "
            "instance over TCP with sslmode=verify-full."
        )
    return conn


def _http(method: str, url: str, token: str | None, timeout: float) -> bytes:
    """Issue one HTTP request and return the body. Raises on any non-2xx or
    transport error; callers in `refresh_site_cache` swallow those."""
    headers = {"accept": "application/json"}
    if token is not None:
        headers["authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, method=method, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _warm_default_windows(base: str, timeout: float) -> None:
    """Best-effort warm pass: prime the freshly invalidated Data Cache for the
    landing page and every group's default last-100 bundle, so the first human
    request after an ingest is already hot. Each request is independent; one
    failure does not abort the others."""
    def warm(url: str) -> None:
        try:
            _http("GET", url, None, timeout)
        except Exception as exc:  # noqa: BLE001 -- warm is best-effort.
            print(f"warning: warm {url} failed: {exc}", file=sys.stderr)

    warm(f"{base}/")
    try:
        groups_body = _http("GET", f"{base}/api/groups", None, timeout)
        slugs = [g["slug"] for g in json.loads(groups_body).get("groups", []) if "slug" in g]
    except Exception as exc:  # noqa: BLE001 -- group discovery is best-effort.
        print(f"warning: warm group discovery failed: {exc}", file=sys.stderr)
        return
    # A whole-bundle recompute is a few seconds cold, so warm with bounded
    # concurrency rather than one slow serial pass.
    with ThreadPoolExecutor(max_workers=4) as pool:
        pool.map(lambda s: warm(f"{base}/api/group/{s}?n=100"), slugs)


def refresh_site_cache(base_url: str, token: str, timeout: float) -> None:
    """Revalidate the site's Data Cache tag, then warm the default windows.

    BEST-EFFORT: every failure is logged to stderr and swallowed so a cache
    refresh can never change the ingest exit code. The warm pass is skipped
    when revalidation fails: warming after a failed flush would repopulate the
    Data Cache with stale data, which is the opposite of the intent.
    """
    base = base_url.rstrip("/")
    try:
        _http("POST", f"{base}/api/revalidate", token, timeout)
    except Exception as exc:  # noqa: BLE001 -- refresh must never raise into ingest.
        print(f"warning: cache revalidate failed: {exc}", file=sys.stderr)
        return  # Skip the warm pass: no point warming a cache that was not flushed.
    _warm_default_windows(base, timeout)


def _main_postgres(args: argparse.Namespace) -> int:
    records = read_records(args.jsonl_path)
    # `build_commit` runs `git show <commit_sha>`, so the SHA must be in the runner's local git
    # history. The v4 ingest step inherits the v3 `--server` step's checkout assumption (the default
    # checkout provides the head SHA); a shallow checkout missing the SHA fails loud here, and the
    # v4 step is best-effort (continue-on-error), so it never fails the job.
    commit = build_commit(args.commit_sha, args.repo_url, args.git_dir)
    conn = connect_postgres(args.postgres, args.region)
    try:
        inserted, updated = ingest_postgres(conn, commit, records)
    finally:
        conn.close()
    print(
        json.dumps(
            {"records": len(records), "inserted": inserted, "updated": updated},
            separators=(",", ":"),
        )
    )
    # Best-effort site-cache refresh after a successful write. No-op unless both
    # env vars are set (so the script stays inert until the ops wiring lands),
    # and it can never fail the ingest. The ops prerequisite (setting the two env
    # vars in Vercel and as GitHub secrets/vars) is documented in the "Ops
    # prerequisite" section of .big-plans/ct__bench-v4-uiux-r3-design.md.
    base_url = os.environ.get("BENCH_SITE_BASE_URL")
    revalidate_token = os.environ.get("BENCH_REVALIDATE_TOKEN")
    if base_url and revalidate_token:
        refresh_site_cache(base_url, revalidate_token, args.timeout)
    return 0


def _main_server(args: argparse.Namespace) -> int:
    if args.benchmark_id is None:
        print("error: --benchmark-id is required in --server mode", file=sys.stderr)
        return 2
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


def main() -> int:
    args = parse_args()
    if args.postgres is not None:
        return _main_postgres(args)
    return _main_server(args)


if __name__ == "__main__":
    raise SystemExit(main())
