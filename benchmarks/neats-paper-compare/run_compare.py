#!/usr/bin/env python3
"""
Compare reference C++ NeaTS against our Rust impl on the same datasets.

Reads each CSV column, quantizes f64 → i64 by scaling by 10**k where k is chosen so the
column's smallest non-zero diff maps to ~1 integer unit (max k=6). Writes the resulting i64
array as raw binary, then runs `DecompressorSIMD <bin> <bpc>` capturing the reported numbers.

We mirror the same quantized data through our Rust path by writing a parallel CSV of f64
values that, after quantization, would match the i64 stream — i.e. we just use the original
f64 column. Our NeaTS handles f64 natively with epsilon=0 (lossless against the quantized
representation when k matches the data's decimal precision).
"""
import csv
import math
import os
import struct
import subprocess
import sys
from pathlib import Path

DATA_DIR = Path("/home/user/vortex/benchmarks/real-data")
TMP_DIR = Path("/tmp/neats-cmp")
DECOMP = Path("/tmp/neats-reference/build/DecompressorSIMD")

TMP_DIR.mkdir(parents=True, exist_ok=True)


def read_numeric_columns(csv_path: Path):
    """Yield (column_name, list_of_f64) for every column that's >=80% numeric."""
    with csv_path.open() as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return
        cols = [[] for _ in header]
        for row in reader:
            for i, cell in enumerate(row):
                if i >= len(cols):
                    break
                try:
                    cols[i].append(float(cell.strip().strip('"')))
                except ValueError:
                    cols[i].append(None)
        for i, name in enumerate(header):
            col = cols[i]
            valid = [v for v in col if v is not None]
            if len(valid) >= max(16, len(col) // 2):
                yield name.strip().strip('"'), valid


def pick_scale(values):
    """Pick 10**k so each value's quantization error is at most 1. Cap k at 6."""
    if not values:
        return 1.0
    # Use the smallest absolute non-zero value as the precision floor, but cap to avoid
    # exploding the range.
    abs_nonzero = [abs(v) for v in values if v != 0.0]
    if not abs_nonzero:
        return 1.0
    min_abs = min(abs_nonzero)
    # Find k such that min_abs * 10**k >= 1.
    if min_abs >= 1.0:
        k = 0
    else:
        k = min(6, int(math.ceil(-math.log10(min_abs))))
    return 10.0 ** k


def quantize_to_i64(values, scale):
    out = []
    for v in values:
        q = round(v * scale)
        # Clamp into i64 range
        q = max(-(2 ** 62), min(2 ** 62, q))
        out.append(q)
    return out


def write_binary_i64(path: Path, values):
    with path.open("wb") as f:
        for v in values:
            f.write(struct.pack("<q", v))


def run_decompressor(bin_path: Path, bpc: int):
    """Run DecompressorSIMD, return dict of reported metrics. Times out at 60s."""
    try:
        result = subprocess.run(
            [str(DECOMP), str(bin_path), str(bpc)],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except subprocess.TimeoutExpired:
        return None
    if result.returncode != 0:
        return None
    lines = [l.strip() for l in result.stdout.strip().splitlines() if l.strip()]
    if len(lines) < 2:
        return None
    # The output is:
    #   compressor,dataset,compressed_bit_size,compression ratio,compression_speed(MB/s),random_access_speed(MB/s),full_decompression_speed(MB/s),
    #   NeaTS,...
    #   Decompression speed: X MB/s
    header_idx = None
    data_idx = None
    for i, l in enumerate(lines):
        if l.startswith("compressor,"):
            header_idx = i
        elif l.startswith("NeaTS,"):
            data_idx = i
    if data_idx is None:
        return None
    parts = lines[data_idx].split(",")
    if len(parts) < 7:
        return None
    return {
        "compressed_bit_size": int(parts[2]),
        "compression_ratio": float(parts[3]),
        "compress_speed_MBs": float(parts[4]),
        "random_access_speed_MBs": float(parts[5]),
        "decompress_speed_MBs": float(parts[6]),
    }


def main():
    print(f"{'dataset/column':<40} {'rows':>8} {'scale':>8} {'bpc':>4} {'comp_bits':>10} {'ratio_paper':>12} {'comp_MBs':>10} {'decomp_MBs':>11}")
    csv_files = sorted(DATA_DIR.glob("*.csv"))
    for csv_path in csv_files:
        for name, values in read_numeric_columns(csv_path):
            if len(values) < 1000:
                continue
            scale = pick_scale(values)
            i64s = quantize_to_i64(values, scale)
            label = f"{csv_path.stem}/{name}"
            bin_path = TMP_DIR / f"{csv_path.stem}__{name.replace('/', '_')}.bin"
            write_binary_i64(bin_path, i64s)
            # Try one bpc value (the C++ binary needs a bpc choice; 32 is a reasonable
            # ceiling that lets the partitioner pick tight pieces; 16 if the range fits).
            int_range = max(i64s) - min(i64s) + 1
            bpc = min(32, max(8, int(math.ceil(math.log2(int_range))) - 4))
            metrics = run_decompressor(bin_path, bpc)
            if metrics is None:
                print(f"{label[:39]:<40} {len(values):>8} {scale:>8g} {bpc:>4} {'FAIL':>10}")
                continue
            raw_bits = len(values) * 64
            actual_ratio = raw_bits / max(metrics["compressed_bit_size"], 1)
            print(
                f"{label[:39]:<40} {len(values):>8} {scale:>8g} {bpc:>4} "
                f"{metrics['compressed_bit_size']:>10} {actual_ratio:>11.2f}x "
                f"{metrics['compress_speed_MBs']:>10.1f} {metrics['decompress_speed_MBs']:>11.1f}"
            )


if __name__ == "__main__":
    main()
