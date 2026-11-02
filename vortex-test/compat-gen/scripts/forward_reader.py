#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""
Read forward-compatibility files with whichever `vortex-data` release is installed.

`compat.py forward` runs this script once per released version, in an isolated environment
holding that version's wheel. It must therefore only use the oldest Python reader API,
`vortex.open(path).to_arrow()`, and must not import anything from this repository.

Each file is compared with its expected Arrow IPC file by value, after converting both to
Python objects: older readers may export a different but equivalent Arrow type, such as
`string` for `string_view`.

Prints a JSON result to stdout, and progress to stderr.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import traceback
from pathlib import Path

import pyarrow as pa

import vortex


def _to_table(value: object) -> pa.Table:
    """Normalize the result of `VortexFile.to_arrow()` across reader versions."""
    if isinstance(value, pa.Table):
        return value
    if isinstance(value, pa.RecordBatchReader):
        return value.read_all()
    if isinstance(value, pa.RecordBatch):
        return pa.Table.from_batches([value])
    if isinstance(value, pa.StructArray):
        return pa.Table.from_struct_array(value)
    if isinstance(value, pa.ChunkedArray) and pa.types.is_struct(value.type):
        return pa.Table.from_batches([pa.RecordBatch.from_struct_array(c) for c in value.chunks])
    raise TypeError(f"unsupported to_arrow() result: {type(value).__name__}")


def _same(actual: object, expected: object) -> bool:
    """Value equality where NaN equals NaN and 0.0 differs from -0.0."""
    if isinstance(expected, float) and isinstance(actual, float):
        if math.isnan(expected):
            return math.isnan(actual)
        return actual == expected and math.copysign(1.0, actual) == math.copysign(1.0, expected)
    if isinstance(expected, dict) and isinstance(actual, dict):
        return actual.keys() == expected.keys() and all(_same(actual[k], expected[k]) for k in expected)
    if isinstance(expected, (list, tuple)) and isinstance(actual, (list, tuple)):
        return len(actual) == len(expected) and all(_same(a, e) for a, e in zip(actual, expected))
    return type(actual) is type(expected) and actual == expected


def _pylist(column: pa.ChunkedArray, expected_type: pa.DataType) -> list[object]:
    """Convert a column to Python values; temporal columns become their integer storage.

    Python's datetime types cannot hold nanoseconds, so a temporal column is first cast to the
    expected type (a safe cast, which fails rather than truncate) and then to its storage.
    """
    if pa.types.is_temporal(expected_type):
        storage = pa.int32() if expected_type.bit_width == 32 else pa.int64()
        return column.cast(expected_type).cast(storage).to_pylist()
    return column.to_pylist()


def _compare(actual: pa.Table, expected: pa.Table) -> str | None:
    """Return a description of the first difference, or None if the tables match."""
    if actual.column_names != expected.column_names:
        return f"columns {actual.column_names} != expected {expected.column_names}"
    if actual.num_rows != expected.num_rows:
        return f"{actual.num_rows} rows != expected {expected.num_rows}"
    for name in expected.column_names:
        actual_values = _pylist(actual.column(name), expected.schema.field(name).type)
        expected_values = _pylist(expected.column(name), expected.schema.field(name).type)
        for row, (a, e) in enumerate(zip(actual_values, expected_values)):
            if not _same(a, e):
                return (
                    f"column {name!r} row {row}: {a!r} != expected {e!r} "
                    f"(read as {actual.schema.field(name).type}, "
                    f"expected {expected.schema.field(name).type})"
                )
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", required=True, type=Path, help="Output directory of `generate-forward`.")
    parser.add_argument("--editions", required=True, help="Comma-separated editions to read.")
    parser.add_argument("--version", required=True, help="The installed vortex-data version, for messages.")
    args = parser.parse_args()

    manifest = json.loads((args.dir / "forward.json").read_text())
    expected_paths = {f["name"]: args.dir / f["expected"] for f in manifest["fixtures"]}
    wanted = set(args.editions.split(","))

    passed: list[dict[str, str]] = []
    failed: list[dict[str, str]] = []
    expected_tables: dict[str, pa.Table] = {}

    for edition in manifest["editions"]:
        if edition["edition"] not in wanted:
            continue
        for name in edition["written"]:
            path = args.dir / edition["edition"] / name
            entry = {"edition": edition["edition"], "name": name}
            try:
                if name not in expected_tables:
                    with pa.memory_map(str(expected_paths[name])) as source:
                        expected_tables[name] = pa.ipc.open_file(source).read_all()
                actual = _to_table(vortex.open(str(path)).to_arrow())
                error = _compare(actual, expected_tables[name])
            except Exception as e:  # noqa: BLE001  # Any reader failure is a compatibility failure.
                # Native errors append a backtrace; keep the result to the message.
                message = str(e).split("Backtrace:")[0].strip()
                error = f"{type(e).__name__}: {message}"
                traceback.print_exc(file=sys.stderr)
            if error is None:
                passed.append(entry)
            else:
                print(f"  v{args.version} FAIL {edition['edition']}/{name}: {error}", file=sys.stderr)
                failed.append({**entry, "error": error})

    print(json.dumps({"version": args.version, "passed": passed, "failed": failed}))


if __name__ == "__main__":
    main()
