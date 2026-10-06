# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Report an end-to-end run of the model-driven compressor against the production compressor.

Every number here comes from real compressions in the Rust harness: serialized bytes, compression
time (including features, inference and verification) and decode time. Each chunk was handled by
a model trained without that chunk's source.

Usage: uv run --no-project --with pandas python e2e_report.py <eval-out-dir> [tag]
"""

import sys

import pandas as pd

KEY = ["source", "column", "chunk"]
BANDWIDTHS = {"s3": ("S3 ~100 MB/s", 1e8), "nvme": ("NVMe ~2 GB/s", 2e9), "mem": ("memory ~20 GB/s", 2e10)}


def main() -> None:
    out = sys.argv[1]
    tag = sys.argv[2] if len(sys.argv) > 2 else "stock"
    rows = pd.read_csv(f"{out}/rows-{tag}.csv")
    ok = rows[rows.ok == 1].set_index(KEY)
    prod = ok[ok.variant == "default"]
    size = ok[ok.variant == "model/runend+sparse"]
    mb = prod.canonical_bytes.sum() / 1e6
    print(f"chunks {len(prod)}, canonical {mb:.1f} MB")
    print(f"production: {prod.bytes.sum() / 1e6:.2f} MB (ratio {prod.canonical_bytes.sum() / prod.bytes.sum():.2f}x), "
          f"compress {mb / (prod.compress_ns.sum() / 1e9):.0f} MB/s, decode {mb / (prod.decode_ns_median.sum() / 1e9) / 1e3:.2f} GB/s")
    print("cost per chunk = compress time / reads + bytes / bandwidth + decode time; all measured\n")

    def total(df, bw, reads):
        return (df.compress_ns / 1e9 / reads + df.bytes / bw + df.decode_ns_median / 1e9).sum()

    print(f"{'bandwidth':10s} {'reads':>5s} {'size-model':>11s} {'model-driven':>13s} {'bytes':>7s} {'decode':>7s} "
          f"{'compress':>9s} {'tried':>6s} {'kept':>5s}  worst held-out source")
    for key, (label, bw) in BANDWIDTHS.items():
        for variant in sorted(v for v in ok.variant.unique() if v.startswith(f"model@{key}@")):
            reads = float(variant.split("@r")[1])
            model = ok[ok.variant == variant]
            base = total(prod, bw, reads)
            kept = model.tree.str.extract(r"kept=(\S+)")[0]
            tried = model.tree.str.extract(r"tried=(\d)")[0].eq("1")
            per_src = {s: total(d, bw, reads) / total(prod.loc[d.index], bw, reads) - 1
                       for s, d in model.groupby(level="source")}
            worst = max(per_src, key=per_src.get)
            print(f"{key:10s} {reads:>5.0f} {total(size, bw, reads) / base - 1:>+11.1%} {total(model, bw, reads) / base - 1:>+13.1%} "
                  f"{model.bytes.sum() / prod.bytes.sum() - 1:>+7.1%} {model.decode_ns_median.sum() / prod.decode_ns_median.sum() - 1:>+7.1%} "
                  f"{model.compress_ns.sum() / prod.compress_ns.sum():>8.2f}x {tried.mean():>6.0%} {kept.ne('production').mean():>5.0%}  "
                  f"{worst} {per_src[worst]:+.1%}")


if __name__ == "__main__":
    main()
