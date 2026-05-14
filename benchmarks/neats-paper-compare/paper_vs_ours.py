#!/usr/bin/env python3
"""
Head-to-head: reference C++ NeaTS vs our Rust NeaTS on the same i64-quantized data.

For each CSV column with >=1000 rows:
1. Quantize f64 -> i64 (as in run_compare.py).
2. Run reference C++ DecompressorSIMD on the i64 binary.
3. Reconstruct the same f64 values (by re-quantizing) and run our Rust neats-table on a
   single-column CSV that contains those same values.
4. Print a row with both ratios side by side.
"""
import csv
import math
import struct
import subprocess
import tempfile
from pathlib import Path

DATA_DIR = Path("/home/user/vortex/benchmarks/real-data")
TMP_DIR = Path("/tmp/neats-cmp")
TMP_DIR.mkdir(parents=True, exist_ok=True)
DECOMP = Path("/tmp/neats-reference/build/DecompressorSIMD")
NEATS_TABLE = Path("/home/user/vortex/target/release/neats-table")


def read_numeric_columns(csv_path):
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
    abs_nonzero = [abs(v) for v in values if v != 0.0]
    if not abs_nonzero:
        return 1.0
    min_abs = min(abs_nonzero)
    if min_abs >= 1.0:
        return 1.0
    k = min(6, int(math.ceil(-math.log10(min_abs))))
    return 10.0 ** k


def run_paper(bin_path: Path, bpc: int):
    try:
        r = subprocess.run([str(DECOMP), str(bin_path), str(bpc)],
                           capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        return None
    if r.returncode != 0:
        return None
    for line in r.stdout.splitlines():
        line = line.strip()
        if line.startswith("NeaTS,"):
            parts = line.split(",")
            if len(parts) >= 7:
                return {
                    "compressed_bits": int(parts[2]),
                    "compress_MBs": float(parts[4]),
                    "decompress_MBs": float(parts[6]),
                }
    return None


def run_ours(csv_path: Path):
    """Run our neats-table on the single-column CSV; parse the 'neats' (PCO default) and
    'pco_lossy' bytes from the row."""
    try:
        r = subprocess.run([str(NEATS_TABLE), str(csv_path)],
                           capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        return None
    out = r.stdout.splitlines()
    # Find the data row (after header)
    for line in out:
        if "|" in line and "neats_pco" not in line and "input" not in line and "##" not in line:
            return parse_neats_table_row(line)
    return None


def parse_neats_table_row(line):
    # Format: name rows | neats_pco_bytes ratio | pco_lossy_bytes ratio | ... 6 encoder groups
    # Each group is "{bytes:>10.0} {ratio:>5.2}x"
    # Strip the pipes and extract numbers.
    parts = [p.strip() for p in line.split("|")]
    if len(parts) < 7:
        return None
    # parts[0]: "name rows"
    # parts[1..6]: each "BYTES RATIOx"
    results = {}
    keys = ["neats_pco", "pco_lossy", "neats_ppb", "ppb_lossy", "btr", "pco"]
    for key, seg in zip(keys, parts[1:7]):
        toks = seg.split()
        if len(toks) >= 1:
            try:
                results[key] = int(toks[0])
            except ValueError:
                pass
    return results


def main():
    print(
        f"{'dataset/column':<40} {'rows':>8} | "
        f"{'paper_bits':>10} {'paper_ratio':>12} | "
        f"{'ours_pco':>10} {'ours_ratio':>11} | "
        f"{'ours_ppb':>10} {'ratio':>7}"
    )
    print("-" * 120)
    for csv_path in sorted(DATA_DIR.glob("*.csv")):
        for name, values in read_numeric_columns(csv_path):
            if len(values) < 1000:
                continue
            scale = pick_scale(values)
            i64s = [max(-(2**62), min(2**62, round(v * scale))) for v in values]
            # Reconstruct what our Rust NeaTS will see (the SAME quantized values, re-divided
            # by scale). That way both compressors see the same set of representable values.
            requantized_f64 = [q / scale for q in i64s]

            int_range = max(i64s) - min(i64s) + 1
            bpc = min(32, max(8, int(math.ceil(math.log2(max(int_range, 2)))) - 4))

            label = f"{csv_path.stem}/{name}"
            bin_path = TMP_DIR / f"{csv_path.stem}__{name.replace('/', '_')}.bin"
            if not bin_path.exists():
                with bin_path.open("wb") as f:
                    for v in i64s:
                        f.write(struct.pack("<q", v))

            paper = run_paper(bin_path, bpc)

            # Write requantized f64 as a single-column CSV for our impl.
            csv_tmp = TMP_DIR / f"{csv_path.stem}__{name.replace('/', '_')}.csv"
            with csv_tmp.open("w") as f:
                f.write(f"{name}\n")
                for v in requantized_f64:
                    f.write(f"{v}\n")
            ours = run_ours(csv_tmp)

            raw_bits = len(values) * 64
            paper_str = "FAIL"
            paper_ratio_str = ""
            if paper:
                paper_str = f"{paper['compressed_bits']}"
                paper_ratio_str = f"{raw_bits / paper['compressed_bits']:.2f}x"
            ours_pco_str = "FAIL"
            ours_pco_ratio = ""
            ours_ppb_str = "-"
            ours_ppb_ratio = ""
            if ours:
                if "neats_pco" in ours:
                    ours_pco_str = f"{ours['neats_pco']}"
                    ours_pco_ratio = f"{(raw_bits // 8) / max(ours['neats_pco'], 1):.2f}x"
                if "neats_ppb" in ours:
                    ours_ppb_str = f"{ours['neats_ppb']}"
                    ours_ppb_ratio = f"{(raw_bits // 8) / max(ours['neats_ppb'], 1):.2f}x"

            print(
                f"{label[:39]:<40} {len(values):>8} | "
                f"{paper_str:>10} {paper_ratio_str:>12} | "
                f"{ours_pco_str:>10} {ours_pco_ratio:>11} | "
                f"{ours_ppb_str:>10} {ours_ppb_ratio:>7}"
            )


if __name__ == "__main__":
    main()
