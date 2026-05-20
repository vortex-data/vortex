#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors
"""Single entry point that (re)creates the OnPair compression benchmark.

For every column in ``columns.COLUMNS`` and every ``bits × chunk × threshold``
cell this:

  1. builds the Rust ``onpair-chunk-bench`` binary (release),
  2. ensures the source parquet exists (generating TPC-H locally),
  3. compresses the sampled column into Vortex files (one OnPair dictionary per
     chunk) and verifies the string round-trip,
  4. aggregates the per-cell JSON into a markdown table + ``summary.json``.

The Rust binary parallelises chunk compression internally. Independent columns
are processed concurrently with ``--jobs``.

Usage::

    python benchmarks/onpair-bench/run.py                  # full default run
    python benchmarks/onpair-bench/run.py --sample-bytes 50_000_000  # quick
    python benchmarks/onpair-bench/run.py --bits 12 --chunk-mb 1,10  # subset
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from columns import COLUMNS, DATA_DIR, REPO_ROOT, SRC_DIR, Column

OUT_ROOT = DATA_DIR / "onpair-bench"
BIN = "onpair-chunk-bench"
MB = 1 << 20
GIB = 1 << 30


def available_cores() -> int:
    return os.cpu_count() or 1


def clean_outputs() -> None:
    """Remove generated OnPair benchmark output files.

    Source parquet caches under `onpair-bench-src` are intentionally preserved.
    """
    if OUT_ROOT.exists():
        print(f"==> removing {OUT_ROOT}", file=sys.stderr)
        shutil.rmtree(OUT_ROOT)
    else:
        print(f"==> nothing to clean at {OUT_ROOT}", file=sys.stderr)


def build_binary(release: bool, cuda: bool) -> Path:
    profile = ["--release"] if release else []
    features = ["--features", "cuda"] if cuda else []
    print(
        f"==> building {BIN} ({'release' if release else 'dev'}"
        f"{', cuda' if cuda else ''})",
        file=sys.stderr,
    )
    subprocess.run(
        ["cargo", "build", *profile, "-p", "vortex-bench", "--bin", BIN, *features],
        cwd=REPO_ROOT,
        check=True,
    )
    target = "release" if release else "debug"
    return REPO_ROOT / "target" / target / BIN


def download(url: str, dest: Path) -> None:
    """Stream `url` to `dest` (atomically via a .part file). Uses curl/wget if
    available, else urllib — so it works without extra deps."""
    import shutil
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"==> downloading {url}\n        -> {dest}", file=sys.stderr)
    if shutil.which("curl"):
        subprocess.run(["curl", "-L", "--fail", "-o", str(tmp), url], check=True)
    elif shutil.which("wget"):
        subprocess.run(["wget", "-O", str(tmp), url], check=True)
    else:
        import urllib.request
        with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f)
    tmp.rename(dest)


def text_to_parquet(src: Path, dest: Path, column: str) -> None:
    """Convert a newline-delimited text file to a one-column parquet file."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    print(f"==> converting {src} -> {dest}", file=sys.stderr)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(src, encoding="utf-8", errors="replace") as f:
        values = [line.rstrip("\n\r") for line in f]
    table = pa.table({column: pa.array(values, type=pa.string())})
    tmp = dest.with_suffix(dest.suffix + ".part")
    pq.write_table(table, tmp)
    tmp.rename(dest)


def ensure_parquet(binary: Path, col: Column) -> Path:
    path = col.parquet_path()
    if path.exists():
        return path
    if col.kind == "tpch":
        # Generates *all* TPC-H tables (one file each) into the sf dir; the Rust
        # side is idempotent so repeated calls for sibling columns are no-ops.
        out_dir = col.tpch_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"==> generating TPC-H sf={col.scale_factor} tables", file=sys.stderr)
        subprocess.run(
            [str(binary), "gen-tpch", "--sf", str(col.scale_factor), "--out-dir", str(out_dir)],
            cwd=REPO_ROOT,
            check=True,
        )
        return path
    if col.kind == "tpcds":
        # DuckDB dsdgen → one parquet per table under <tpcds_dir>/parquet/.
        out_dir = col.tpcds_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"==> generating TPC-DS sf={col.scale_factor} tables (duckdb dsdgen)", file=sys.stderr)
        subprocess.run(
            [str(binary), "gen-tpcds", "--sf", str(col.scale_factor), "--out-dir", str(out_dir)],
            cwd=REPO_ROOT,
            check=True,
        )
        return path
    if col.kind == "parquet" and col.url:
        download(col.url, col.cache_path())
        return col.cache_path()
    if col.kind == "text" and col.url:
        raw_path = SRC_DIR / col.dataset_id / "raw" / f"{col.column}.txt"
        if not raw_path.exists():
            download(col.url, raw_path)
        text_to_parquet(raw_path, col.cache_path(), col.column)
        return col.cache_path()
    raise FileNotFoundError(
        f"parquet for {col.dataset_id}/{col.column} not found at {path} "
        f"and no download url configured"
    )


_STRING_ARROW_TYPES = ("string", "utf8", "large_string", "large_utf8")


def column_is_string(parquet: Path, column: str) -> bool:
    """True iff `column` exists in `parquet` and is a (any-width) string type."""
    import pyarrow.parquet as pq

    try:
        field = pq.read_schema(parquet).field(column)
    except KeyError:
        return False
    return any(t in str(field.type).lower() for t in _STRING_ARROW_TYPES)


def run_column(binary: Path, col: Column, args) -> list[dict]:
    parquet = ensure_parquet(binary, col)
    chunk_bytes = ",".join(str(int(mb * MB)) for mb in args.chunk_mb)
    bits = ",".join(str(b) for b in args.bits)
    thresholds = ",".join(str(t) for t in args.threshold)
    print(f"==> running {col.dataset_id}/{col.column}", file=sys.stderr)
    proc = subprocess.run(
        [
            str(binary), "run",
            "--parquet", str(parquet),
            "--column", col.column,
            "--dataset-id", col.dataset_id,
            "--bits", bits,
            "--chunk-bytes", chunk_bytes,
            "--threshold", thresholds,
            "--sample-bytes", str(args.sample_bytes),
            "--file-target-bytes", str(int(args.file_target_mb * MB)),
            "--out-dir", str(OUT_ROOT),
            *(
                [
                    "--gpu-decode",
                    "--gpu-iters", str(args.gpu_iters),
                    *(["--gpu-validate"] if args.gpu_validate else []),
                ]
                if args.gpu_decode
                else []
            ),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        tail = "\n".join(proc.stderr.strip().splitlines()[-5:])
        print(f"!! {col.dataset_id}/{col.column} FAILED:\n{tail}", file=sys.stderr)
        return []
    return json.loads(proc.stdout)


def fmt_bytes(n: int) -> str:
    for unit, size in (("GiB", GIB), ("MiB", MB), ("KiB", 1 << 10)):
        if n >= size:
            return f"{n / size:.2f} {unit}"
    return f"{n} B"


def markdown_table(rows: list[dict]) -> str:
    headers = [
        "dataset", "column", "bits", "thr", "chunk", "rows", "uniq", "uniq%",
        "chunks", "sample", "in-mem", "on-disk", "str→codec×", "str→files×",
        "enc GiB/s", "dec GiB/s", "gpu auto", "gpu best", "ok", "onpair",
    ]
    lines = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join("---" for _ in headers) + " |"]
    for r in rows:
        uniq_pct = 100.0 * r["unique_count"] / r["rows"] if r["rows"] else 0.0
        gpu = r.get("gpu")
        gpu_auto = ""
        gpu_best = ""
        if gpu:
            gpu_auto = f"{gpu['auto_kernel']} {gpu['auto_decode_gib_s']:.1f}"
            gpu_best = f"{gpu['best_kernel']} {gpu['best_decode_gib_s']:.1f}"
            if gpu.get("validated"):
                status = "ok" if gpu.get("verified") else "bad"
                gpu_best = f"{gpu_best} ({status})"
        lines.append("| " + " | ".join([
            r["dataset_id"], r["column"], str(r["bits"]), f"{r['threshold']:.2f}",
            fmt_bytes(r["chunk_bytes"]), f"{r['rows']:,}", f"{r['unique_count']:,}",
            f"{uniq_pct:.1f}%", str(r["n_chunks"]),
            fmt_bytes(r["sample_bytes"]), fmt_bytes(r["in_memory_bytes"]),
            fmt_bytes(r["on_disk_bytes"]), f"{r['mem_ratio']:.2f}",
            f"{r['disk_ratio']:.2f}", f"{r['encode_gib_s']:.2f}",
            f"{r['decode_gib_s']:.2f}", gpu_auto, gpu_best, "✓" if r["verified"] else "✗",
            "✓" if r["onpair_only"] else "✗",
        ]) + " |")
    return "\n".join(lines)


def pivot_table(rows: list[dict]) -> str:
    """Per-column compression (str→codec×) across every param combo:
    dict width (bits) × block size (chunk)."""
    # Stable param-combo column order.
    combos = sorted({(r["bits"], r["chunk_bytes"]) for r in rows})

    def combo_label(bits, chunk):
        return f"b{bits}/{fmt_bytes(chunk).split()[0]}"

    val = {}  # (dataset,column) -> {combo -> ratio}
    uniq = {}
    for r in rows:
        k = (r["dataset_id"], r["column"])
        val.setdefault(k, {})[(r["bits"], r["chunk_bytes"])] = r["mem_ratio"]
        uniq[k] = 100.0 * r["unique_count"] / r["rows"] if r["rows"] else 0.0

    headers = ["dataset/column", "uniq%"] + [combo_label(*c) for c in combos]
    lines = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join("---" for _ in headers) + " |"]
    for k in sorted(val, key=lambda k: (k[0], k[1])):
        cells = [f"{val[k].get(c, float('nan')):.2f}" for c in combos]
        lines.append("| " + " | ".join([f"{k[0]}/{k[1]}", f"{uniq[k]:.1f}%"] + cells) + " |")
    return "\n".join(lines)


def consolidated_summary(rows: list[dict], args, pivot: str, full_table: str) -> str:
    """Everything in one place: params, datasets, verification, the per-column
    pivot, key findings, and the full per-cell table."""
    from collections import Counter

    datasets = Counter(r["dataset_id"] for r in rows)
    cols = len({(r["dataset_id"], r["column"]) for r in rows})
    fails = [
        r for r in rows
        if (
            not r["verified"]
            or not r["onpair_only"]
            or (r.get("gpu", {}).get("validated") and not r["gpu"].get("verified"))
        )
    ]

    # Best param per column (max str→codec×).
    best = {}
    for r in rows:
        k = (r["dataset_id"], r["column"])
        if k not in best or r["mem_ratio"] > best[k]["mem_ratio"]:
            best[k] = r

    lines = [
        "# OnPair chunked-array compression — benchmark summary",
        "",
        "Each string column is OnPair-compressed per chunk (one dictionary each), "
        "every OnPair child is BtrBlocks-compressed, and the chunks are written to "
        "real `.vortex` files that preserve the OnPair encoding. The offset children "
        "use the smaller of BtrBlocks-only and delta+BtrBlocks. All cells are round-trip verified.",
        "",
        "## Run parameters",
        f"- dict widths (bits): `{args.bits}`",
        f"- block sizes (uncompressed): `{[f'{m:g}MB' for m in args.chunk_mb]}`",
        f"- training threshold: `{args.threshold}`",
        f"- raw sample cap: `{args.sample_bytes:,}` bytes  |  file target: `{args.file_target_mb:g} MB`",
        f"- GPU kernel-only decode: `{'on' if args.gpu_decode else 'off'}`"
        + (f" ({args.gpu_iters} timed iterations)" if args.gpu_decode else ""),
        f"- GPU byte validation: `{'on' if args.gpu_validate else 'off'}`",
        "",
        "## Coverage",
        f"- **{len(rows)} cells**, **{cols} columns**, datasets: "
        + ", ".join(f"{k} ({v})" for k, v in sorted(datasets.items())),
        f"- round-trip + OnPair-only failures: **{len(fails)}**",
        "",
        "## str→codec× per column × (dict-width / block)",
        "",
        pivot,
        "",
        "## Best configuration per column",
        "",
        "| dataset/column | uniq% | best str→codec× | bits | block |",
        "| --- | --- | --- | --- | --- |",
    ]
    for k in sorted(best, key=lambda k: best[k]["mem_ratio"]):
        r = best[k]
        up = 100.0 * r["unique_count"] / r["rows"] if r["rows"] else 0.0
        lines.append(f"| {k[0]}/{k[1]} | {up:.1f}% | {r['mem_ratio']:.2f}× | "
                     f"{r['bits']} | {fmt_bytes(r['chunk_bytes'])} |")
    lines += ["", "## All cells", "", full_table, ""]
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bits", type=lambda s: [int(x) for x in s.split(",")],
                   default=[12, 16])
    p.add_argument("--chunk-mb", type=lambda s: [float(x) for x in s.split(",")],
                   default=[1, 10, 100, 1000], help="per-chunk uncompressed MB budgets")
    p.add_argument("--threshold", type=lambda s: [float(x) for x in s.split(",")],
                   default=[0.2])
    p.add_argument("--sample-bytes", type=int, default=1_000_000_000)
    p.add_argument("--file-target-mb", type=float, default=200.0)
    p.add_argument("--gpu-decode", action="store_true",
                   help="also time CUDA kernel-only OnPair decompression for all applicable kernels")
    p.add_argument("--gpu-iters", type=int, default=10,
                   help="timed CUDA iterations per kernel when --gpu-decode is set")
    p.add_argument("--gpu-validate", action="store_true",
                   help="copy GPU output back and compare every applicable kernel against CPU bytes")
    p.add_argument("--jobs", type=int, default=0,
                   help="columns to run concurrently (default: all available CPU cores)")
    p.add_argument("--dev", action="store_true", help="dev build instead of release")
    p.add_argument("--datasets", type=lambda s: {x.strip() for x in s.split(",")},
                   default=None,
                   help="only these dataset ids (comma-separated), e.g. tpch-sf10,fineweb")
    p.add_argument("--columns", type=lambda s: {x.strip() for x in s.split(",")},
                   default=None,
                   help="only these column names (comma-separated), e.g. l_comment,text")
    p.add_argument("--list", action="store_true",
                   help="list available dataset/column pairs and exit")
    p.add_argument("--clean", action="store_true",
                   help="delete generated OnPair benchmark .vortex files and summaries, then exit")
    args = p.parse_args()

    if args.clean:
        clean_outputs()
        return 0

    if args.gpu_validate and not args.gpu_decode:
        print("--gpu-validate requires --gpu-decode", file=sys.stderr)
        return 1

    if args.list:
        for c in COLUMNS:
            print(f"{c.dataset_id}\t{c.column}")
        return 0

    # Restrict to the requested datasets / columns (both filters are AND-ed).
    columns = [c for c in COLUMNS
               if (args.datasets is None or c.dataset_id in args.datasets)
               and (args.columns is None or c.column in args.columns)]
    if not columns:
        print("no columns match the given --datasets/--columns filters", file=sys.stderr)
        return 1

    binary = build_binary(release=not args.dev, cuda=args.gpu_decode)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    # Ensure each source parquet exists up front (sequentially) so concurrent
    # columns never race on generation, then keep only columns that are present
    # and string-typed.
    selected: list[Column] = []
    for col in columns:
        try:
            parquet = ensure_parquet(binary, col)
        except FileNotFoundError as e:
            # External datasets (ClickBench/FineWeb/book-reviews) aren't
            # auto-downloaded; skip any whose source parquet is absent so the
            # run still completes on whatever data is present (TPC-H always
            # generates locally).
            print(f"-- skip {col.dataset_id}/{col.column}: {e}", file=sys.stderr)
            continue
        if column_is_string(parquet, col.column):
            selected.append(col)
        else:
            print(f"-- skip {col.dataset_id}/{col.column} (missing or non-string)",
                  file=sys.stderr)
    print(f"==> {len(selected)}/{len(columns)} columns selected", file=sys.stderr)

    jobs = args.jobs if args.jobs > 0 else available_cores()
    print(f"==> running with {jobs} column worker(s)", file=sys.stderr)

    if jobs > 1:
        with ThreadPoolExecutor(max_workers=jobs) as pool:
            futs = {pool.submit(run_column, binary, c, args): c for c in selected}
            for fut in as_completed(futs):
                results.extend(fut.result())
    else:
        for col in selected:
            results.extend(run_column(binary, col, args))

    results.sort(key=lambda r: (r["dataset_id"], r["column"], r["bits"],
                                r["threshold"], r["chunk_bytes"]))

    summary_json = OUT_ROOT / "summary.json"
    summary_md = OUT_ROOT / "summary.md"
    pivot_md = OUT_ROOT / "summary_pivot.md"
    consolidated = OUT_ROOT / "SUMMARY.md"
    summary_json.write_text(json.dumps(results, indent=2))
    table = markdown_table(results)
    pivot = pivot_table(results)
    summary_md.write_text(table + "\n")
    pivot_md.write_text("# str→codec× per column × (dict-width / block)\n\n"
                        + pivot + "\n")
    consolidated.write_text(consolidated_summary(results, args, pivot, table))

    print("\n" + pivot)
    print(f"\nWrote {consolidated} (everything in one place)\n"
          f"Wrote {summary_json}\nWrote {summary_md}\nWrote {pivot_md}", file=sys.stderr)

    failures = [
        r for r in results
        if (
            not r["verified"]
            or not r["onpair_only"]
            or (r.get("gpu", {}).get("validated") and not r["gpu"].get("verified"))
        )
    ]
    if failures:
        print(f"\n{len(failures)} cell(s) FAILED round-trip / onpair-only / GPU validation check",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
