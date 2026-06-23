# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Python port of the server-internal `measurement_id` xxhash64 functions.

This is a byte-for-byte port of the `vortex-data/benchmarks-website` repo's
`server/src/db.rs` (`measurement_id_*`, `hasher_for`,
`write_str`, `write_opt_str`, `write_i32`, `write_f64`, `finish`). The v4 ingest
writer computes the same `measurement_id` the v3 Rust server computes, so
re-ingesting an existing `(commit, dim-tuple)` upserts the existing row via
`ON CONFLICT (measurement_id) DO UPDATE` instead of inserting a duplicate.

The equivalence is NOT assumed -- it is pinned by golden vectors generated from
the Rust source of truth (the `measurement_id_golden_vectors` test in the
`vortex-data/benchmarks-website` repo's `server/src/db.rs`). The Python port is
checked against those vectors; any drift in either implementation fails that
check.

## The hash, precisely

`measurement_id` is a canonical xxhash64 (seed 0) over a byte buffer built as:

1. The per-table tag bytes, then a single `0x00` separator (`hasher_for`).
2. For each dimensional field, in the exact order the Rust function writes it:
   - `write_str(s)`     -> `len(utf8(s))` as a little-endian u64, then the utf8
     bytes. The length is the BYTE length, not the character count.
   - `write_opt_str(o)` -> `0x00` for `None`; `0x01` then `write_str(s)` for
     `Some(s)`.
   - `write_i32(v)`     -> 4 little-endian two's-complement bytes.
   - `write_f64(v)`     -> the IEEE-754 bit pattern as 8 little-endian bytes
     (`v.to_bits()` written little-endian == `struct.pack("<d", v)`).
3. The 64-bit digest is bit-cast `u64 -> i64` (Postgres/DuckDB `BIGINT` is
   signed), matching Rust's `hasher.finish() as i64`.

Canonical xxhash64 is streaming-equivalent: feeding the bytes incrementally (as
the Rust `Hasher` does via `write_*`) yields the same digest as hashing the
fully concatenated buffer once (as this module does).

The little-endian integer encodings are load-bearing and assume a little-endian
host. The Rust side serializes integers with native-endian byte order
(twox-hash 2.x `write_u64` / `write_i32` use `to_ne_bytes`, and `write_f64` is
`write_u64(v.to_bits())`), so byte-for-byte compatibility holds only where
`to_ne_bytes == to_le_bytes`. Every target in play (x86_64 / aarch64 CI runners,
dev machines, the RDS Postgres host, the Vercel reader) is little-endian, and the
golden vectors are generated on a little-endian host and pin it there. On a
big-endian host both this module and the Rust `write_*` would have to switch to a
shared explicit endianness.
"""

import struct

import xxhash

# Must match `XxHash64::with_seed(0)` in `db.rs::hasher_for`.
_SEED = 0

# Per-table tag literals. These MUST match the `hasher_for("<tag>")` argument in
# each `db.rs::measurement_id_*` function verbatim; the tag is the table name.
_TAG_QUERY_MEASUREMENTS = "query_measurements"
_TAG_COMPRESSION_TIMES = "compression_times"
_TAG_COMPRESSION_SIZES = "compression_sizes"
_TAG_RANDOM_ACCESS_TIMES = "random_access_times"
_TAG_VECTOR_SEARCH_RUNS = "vector_search_runs"


def _hasher_buf(tag: str) -> bytearray:
    """Start a hash buffer seeded with a per-table tag plus a `0x00` separator.

    Mirrors `db.rs::hasher_for`: two fact tables that share the same dim values
    still produce distinct `measurement_id`s because the tag differs.
    """
    buf = bytearray()
    buf += tag.encode("utf-8")
    buf.append(0)
    return buf


def _write_str(buf: bytearray, s: str) -> None:
    """Append a length-prefixed string: utf8 BYTE length as LE u64, then bytes."""
    encoded = s.encode("utf-8")
    buf += struct.pack("<Q", len(encoded))
    buf += encoded


def _write_opt_str(buf: bytearray, s: str | None) -> None:
    """Append an optional string: `0x00` for None, `0x01` + the string for Some."""
    if s is None:
        buf.append(0)
    else:
        buf.append(1)
        _write_str(buf, s)


def _write_i32(buf: bytearray, v: int) -> None:
    """Append a 32-bit signed integer as 4 little-endian two's-complement bytes."""
    buf += struct.pack("<i", v)


def _write_f64(buf: bytearray, v: float) -> None:
    """Append a 64-bit float as its 8 little-endian IEEE-754 bytes.

    `struct.pack("<d", v)` produces exactly the bytes Rust writes via
    `hasher.write_u64(v.to_bits())` on a little-endian target.
    """
    buf += struct.pack("<d", v)


def _finish(buf: bytearray) -> int:
    """Hash the buffer (xxhash64, seed 0) and bit-cast the u64 digest to i64."""
    digest = xxhash.xxh64(bytes(buf), seed=_SEED).intdigest()
    # Bit-cast u64 -> i64 to match Rust's `hasher.finish() as i64`.
    return digest - (1 << 64) if digest >= (1 << 63) else digest


def measurement_id_query(
    *,
    commit_sha: str,
    dataset: str,
    dataset_variant: str | None,
    scale_factor: str | None,
    query_idx: int,
    storage: str,
    engine: str,
    format: str,
) -> int:
    """`measurement_id` for a `query_measurements` row. Mirrors
    `db.rs::measurement_id_query`."""
    buf = _hasher_buf(_TAG_QUERY_MEASUREMENTS)
    _write_str(buf, commit_sha)
    _write_str(buf, dataset)
    _write_opt_str(buf, dataset_variant)
    _write_opt_str(buf, scale_factor)
    _write_i32(buf, query_idx)
    _write_str(buf, storage)
    _write_str(buf, engine)
    _write_str(buf, format)
    return _finish(buf)


def measurement_id_compression_time(
    *,
    commit_sha: str,
    dataset: str,
    dataset_variant: str | None,
    format: str,
    op: str,
) -> int:
    """`measurement_id` for a `compression_times` row. Mirrors
    `db.rs::measurement_id_compression_time`."""
    buf = _hasher_buf(_TAG_COMPRESSION_TIMES)
    _write_str(buf, commit_sha)
    _write_str(buf, dataset)
    _write_opt_str(buf, dataset_variant)
    _write_str(buf, format)
    _write_str(buf, op)
    return _finish(buf)


def measurement_id_compression_size(
    *,
    commit_sha: str,
    dataset: str,
    dataset_variant: str | None,
    format: str,
) -> int:
    """`measurement_id` for a `compression_sizes` row. Mirrors
    `db.rs::measurement_id_compression_size`."""
    buf = _hasher_buf(_TAG_COMPRESSION_SIZES)
    _write_str(buf, commit_sha)
    _write_str(buf, dataset)
    _write_opt_str(buf, dataset_variant)
    _write_str(buf, format)
    return _finish(buf)


def measurement_id_random_access(
    *,
    commit_sha: str,
    dataset: str,
    format: str,
) -> int:
    """`measurement_id` for a `random_access_times` row. Mirrors
    `db.rs::measurement_id_random_access`. Note: no `dataset_variant`."""
    buf = _hasher_buf(_TAG_RANDOM_ACCESS_TIMES)
    _write_str(buf, commit_sha)
    _write_str(buf, dataset)
    _write_str(buf, format)
    return _finish(buf)


def measurement_id_vector_search(
    *,
    commit_sha: str,
    dataset: str,
    layout: str,
    flavor: str,
    threshold: float,
) -> int:
    """`measurement_id` for a `vector_search_runs` row. Mirrors
    `db.rs::measurement_id_vector_search`. `iterations` is intentionally NOT part
    of the dim tuple -- it is a side count."""
    buf = _hasher_buf(_TAG_VECTOR_SEARCH_RUNS)
    _write_str(buf, commit_sha)
    _write_str(buf, dataset)
    _write_str(buf, layout)
    _write_str(buf, flavor)
    _write_f64(buf, threshold)
    return _finish(buf)


# Dispatch table keyed by the fact-table name, used by the golden-vector test to
# map a vector's `table` field to the matching port function. Keeping it here (vs.
# in the test) means a new fact table is wired in one place alongside the port.
MEASUREMENT_ID_BY_TABLE = {
    _TAG_QUERY_MEASUREMENTS: measurement_id_query,
    _TAG_COMPRESSION_TIMES: measurement_id_compression_time,
    _TAG_COMPRESSION_SIZES: measurement_id_compression_size,
    _TAG_RANDOM_ACCESS_TIMES: measurement_id_random_access,
    _TAG_VECTOR_SEARCH_RUNS: measurement_id_vector_search,
}
