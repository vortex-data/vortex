#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Capture file sizes from benchmark data directories and output as JSONL."""

import argparse
import json
import struct
import sys
from pathlib import Path

DUCKDB_CATALOG_FILES = {"duckdb.db", "duckdb.db.wal"}

# Parquet's `CompressionCodec` enum, from `parquet.thrift`.
PARQUET_CODECS = {
    0: "uncompressed",
    1: "snappy",
    2: "gzip",
    3: "lzo",
    4: "brotli",
    5: "lz4",
    6: "zstd",
    7: "lz4-raw",
}

# Thrift compact protocol type ids.
_T_STOP, _T_TRUE, _T_FALSE, _T_BYTE, _T_I16, _T_I32, _T_I64, _T_DOUBLE = range(8)
_T_BINARY, _T_LIST, _T_SET, _T_MAP, _T_STRUCT = range(8, 13)


class _CompactReader:
    """Read Thrift compact protocol values. Structs read as dicts keyed by field id."""

    def __init__(self, data: bytes) -> None:
        self.data = data
        self.pos = 0

    def byte(self) -> int:
        value = self.data[self.pos]
        self.pos += 1
        return value

    def varint(self) -> int:
        shift = result = 0
        while True:
            b = self.byte()
            result |= (b & 0x7F) << shift
            if not b & 0x80:
                return result
            shift += 7

    def zigzag(self) -> int:
        n = self.varint()
        return (n >> 1) ^ -(n & 1)

    def value(self, kind: int) -> object:
        if kind in (_T_TRUE, _T_FALSE):
            # A bool list element is a whole byte; a bool field carries its value in its type.
            return self.byte() == _T_TRUE
        if kind == _T_BYTE:
            return self.byte()
        if kind in (_T_I16, _T_I32, _T_I64):
            return self.zigzag()
        if kind == _T_DOUBLE:
            self.pos += 8
            return None
        if kind == _T_BINARY:
            length = self.varint()
            self.pos += length
            return None
        if kind in (_T_LIST, _T_SET):
            header = self.byte()
            size = header >> 4
            if size == 15:
                size = self.varint()
            return [self.value(header & 0x0F) for _ in range(size)]
        if kind == _T_MAP:
            size = self.varint()
            if size == 0:
                return {}
            types = self.byte()
            for _ in range(size):
                self.value(types >> 4)
                self.value(types & 0x0F)
            return None
        if kind == _T_STRUCT:
            return self.struct()
        raise ValueError(f"unknown thrift compact type {kind}")

    def struct(self) -> dict[int, object]:
        fields: dict[int, object] = {}
        field_id = 0
        while True:
            header = self.byte()
            kind = header & 0x0F
            if kind == _T_STOP:
                return fields
            delta = header >> 4
            field_id = field_id + delta if delta else self.zigzag()
            if kind in (_T_TRUE, _T_FALSE):
                fields[field_id] = kind == _T_TRUE
            else:
                fields[field_id] = self.value(kind)


def parquet_codecs(path: Path) -> set[str]:
    """The codecs of every column chunk in a Parquet file, read from its footer."""
    with open(path, "rb") as f:
        f.seek(-8, 2)
        length_bytes, magic = struct.unpack("<I4s", f.read(8))
        if magic != b"PAR1":
            raise ValueError("no PAR1 footer")
        f.seek(-8 - length_bytes, 2)
        metadata = _CompactReader(f.read(length_bytes)).struct()
    codecs = set()
    # FileMetaData.row_groups (4) -> RowGroup.columns (1) -> ColumnChunk.meta_data (3)
    # -> ColumnMetaData.codec (4).
    for row_group in metadata.get(4) or []:
        for column in row_group.get(1) or []:
            meta = column.get(3)
            if isinstance(meta, dict) and isinstance(meta.get(4), int):
                codecs.add(PARQUET_CODECS.get(meta[4], f"codec-{meta[4]}"))
    return codecs


def parquet_format_name(path: Path, directory: str = "parquet") -> str:
    """Name a Parquet file's format by its codec, as `vortex` and `vortex-compact` name Vortex's.

    `parquet` is zstd, the codec every Parquet file the benchmarks write themselves uses (zstd
    level 3). A file written with another codec, such as Snappy in a downloaded dataset, is
    `parquet-<codec>`. A file whose footer cannot be read, or that mixes codecs, keeps the name of
    its `directory` with a warning: `parquet`, as before codecs were recorded, or
    `parquet-snappy`, which `data-gen --formats parquet-snappy` writes.
    """
    try:
        codecs = parquet_codecs(path)
    except (OSError, ValueError, IndexError, struct.error) as error:
        print(f"Could not read the Parquet footer of {path}: {error}", file=sys.stderr)
        return directory
    if len(codecs) != 1:
        if codecs:
            print(f"{path} mixes Parquet codecs {sorted(codecs)}; recording it as {directory}", file=sys.stderr)
        return directory
    codec = next(iter(codecs))
    return "parquet" if codec == "zstd" else f"parquet-{codec}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture file sizes from benchmark data directories")
    parser.add_argument("data_dir", help="Data directory (e.g., vortex-bench/data)")
    parser.add_argument("--benchmark", required=True, help="Benchmark name (e.g., clickbench)")
    parser.add_argument(
        "--dataset-dir", required=True, help="The benchmark's directory under data_dir (e.g., clickbench_partitioned)"
    )
    parser.add_argument("--commit", required=True, help="Commit SHA")
    parser.add_argument("-o", "--output", required=True, help="Output JSONL file path")
    parser.add_argument(
        "--ingest-jsonl",
        help="Also write the sizes as `file_size` records for `post-ingest.py` to this JSONL file",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}", file=sys.stderr)
        sys.exit(1)

    benchmark_dir = data_dir / args.dataset_dir
    if not benchmark_dir.is_dir():
        print(f"Benchmark directory not found: {benchmark_dir}", file=sys.stderr)
        sys.exit(1)

    # Every on-disk format, named as `Format::name()` in `vortex-bench/src/lib.rs` names its
    # directory. The "vortex" CLI argument maps to the "vortex-file-compressed" directory.
    formats_to_capture = {
        "arrow-ipc",
        "csv",
        "duckdb",
        "lance",
        "litdata",
        "parquet",
        "parquet-snappy",
        "vortex-compact",
        "vortex-file-compressed",
        "vortex-spatial-native",
    }

    records = []

    # Walk subdirectories looking for format directories
    # Handle both direct format dirs (clickbench_partitioned/vortex-file-compressed/)
    # and scale factor subdirs (tpch/1.0/vortex-file-compressed/)
    for format_dir in benchmark_dir.rglob("*"):
        if not format_dir.is_dir():
            continue

        format_name = format_dir.name
        if format_name not in formats_to_capture:
            continue

        # Extract scale factor from path (e.g., "1.0" for tpch/1.0/vortex-file-compressed)
        # Default to "1.0" if no intermediate directory (e.g., clickbench)
        path_between = format_dir.relative_to(benchmark_dir).parent
        scale_factor = str(path_between) if str(path_between) != "." else "1.0"

        # Capture all files in this format directory
        for file_path in format_dir.rglob("*"):
            if not file_path.is_file():
                continue

            relative_path = file_path.relative_to(format_dir)
            # DuckDB keeps a catalog of views over the files in each format directory it queries
            # (`duckdb.db`, and its WAL while open). Only the `duckdb` format's database is data.
            if format_name != "duckdb" and relative_path.name in DUCKDB_CATALOG_FILES:
                continue

            size_bytes = file_path.stat().st_size
            recorded_format = (
                parquet_format_name(file_path, format_name)
                if format_name in ("parquet", "parquet-snappy") and file_path.suffix == ".parquet"
                else format_name
            )

            records.append(
                {
                    "metric": "file_size",
                    "unit": "bytes",
                    "value": size_bytes,
                    "commit_id": args.commit,
                    "file_size": {
                        "benchmark": args.benchmark,
                        "scale_factor": scale_factor,
                        "format": recorded_format,
                        "file": str(relative_path),
                    },
                }
            )

    # Sort for deterministic output
    records.sort(
        key=lambda r: (
            r["file_size"]["benchmark"],
            r["file_size"]["scale_factor"],
            r["file_size"]["format"],
            r["file_size"]["file"],
        )
    )

    # Write JSONL output
    with open(args.output, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    if args.ingest_jsonl:
        with open(args.ingest_jsonl, "w") as f:
            for record in records:
                f.write(
                    json.dumps(
                        {
                            "kind": "file_size",
                            "commit_sha": record["commit_id"],
                            **record["file_size"],
                            "value_bytes": record["value"],
                        }
                    )
                    + "\n"
                )

    print(f"Captured {len(records)} file sizes to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
