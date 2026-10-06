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
    print(f"production: {prod.bytes.sum() / 1e6:.2f} MB, compress {mb / (prod.compress_ns.sum() / 1e9):.0f} MB/s, "
          f"decode {mb / (prod.decode_ns_median.sum() / 1e9) / 1e3:.2f} GB/s\n")

    def line(label, df, bw):
        c = (df.bytes / bw + df.decode_ns_median / 1e9).sum()
        c0 = (prod.bytes / bw + prod.decode_ns_median / 1e9).sum()
        return (f"{label:24s} cost {c / c0 - 1:>+7.1%}  bytes {df.bytes.sum() / prod.bytes.sum() - 1:>+7.1%}  "
                f"decode {df.decode_ns_median.sum() / prod.decode_ns_median.sum() - 1:>+7.1%}  "
                f"compress time {df.compress_ns.sum() / prod.compress_ns.sum():.2f}x")

    for key, (label, bw) in BANDWIDTHS.items():
        model = ok[ok.variant == f"model@{key}"]
        print(f"## {label}")
        print(line("size-model thresholds", size, bw))
        print(line("model-driven compressor", model, bw))
        kept = model.tree.str.extract(r"kept=(\S+)")[0]
        tried = model.tree.str.extract(r"tried=(\d)")[0].eq("1")
        print(f"{'':24s} proposals tried on {tried.mean():.0%} of chunks, kept on {kept.ne('production').mean():.0%}; "
              f"kept: {kept[kept.ne('production')].value_counts().head(5).to_dict()}")
        per_src = model.groupby(level="source").apply(
            lambda d: (d.bytes / bw + d.decode_ns_median / 1e9).sum()
            / (prod.loc[d.index].bytes / bw + prod.loc[d.index].decode_ns_median / 1e9).sum() - 1)
        print(f"{'':24s} cost by held-out source: " + ", ".join(f"{s} {v:+.1%}" for s, v in per_src.items()))
        print()


if __name__ == "__main__":
    main()
