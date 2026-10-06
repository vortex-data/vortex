# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Scan time when I/O and decode overlap: max(total bytes / total bandwidth, total decode / cores).

Usage: uv run --no-project --with pandas --with numpy python scan_cost.py <dataset-dir>
"""
import sys
import numpy as np
import pandas as pd

KEY = ["source", "column", "chunk"]
rows = pd.read_csv(sys.argv[1] + "/rows-stock.csv", dtype={"chunk": str})
c = rows[(rows.ok == 1) & (rows.variant != "forced/pco")]
B = c.pivot_table(index=KEY, columns="variant", values="bytes", aggfunc="min")
D = c.pivot_table(index=KEY, columns="variant", values="decode_ns_median", aggfunc="min") / 1e9
names = list(B.columns)
Bv, Dv = B.to_numpy(), D.to_numpy()
ok = np.isfinite(Bv) & np.isfinite(Dv)
prod = names.index("default")
canon = c[c.variant == "default"].canonical_bytes.sum()


def scan(choice, bw, cores):
    b = Bv[np.arange(len(choice)), choice].sum()
    d = Dv[np.arange(len(choice)), choice].sum()
    return max(b / bw, d / cores), b, d


def best_pipelined(bw, cores, allowed):
    # Minimise max(B/bw, D/cores): per chunk pick argmin of a*B/bw + (1-a)*D/cores, search a.
    best = None
    for a in np.linspace(0, 1, 201):
        score = np.where(ok & allowed, a * Bv / bw + (1 - a) * Dv / cores, np.inf)
        choice = score.argmin(axis=1)
        t = scan(choice, bw, cores)
        if best is None or t[0] < best[0][0]:
            best = (t, a)
    return best


allowed_all = np.ones(len(names), dtype=bool)[None, :]
allowed_for = np.array([n in ("default", "forced/for") for n in names])[None, :]
print(f"corpus: {canon / 1e6:.0f} MB canonical, {len(B)} chunks\n")
print(f"{'setup':38s} {'production':>12s} {'bound by':>9s}   {'+FOR, balanced':>15s}   {'any candidate':>14s}   {'serial-sum rule':>16s}")
for label, bw, cores in [
    ("S3, 1 stream (~90 MB/s), 16 cores", 9e7, 16),
    ("S3, 64 streams (~3 GB/s), 16 cores", 3e9, 16),
    ("S3, 256 streams (~12 GB/s), 16 cores", 1.2e10, 16),
    ("S3, 256 streams (~12 GB/s), 64 cores", 1.2e10, 64),
    ("NVMe x4 (~25 GB/s), 32 cores", 2.5e10, 32),
    ("S3, 256 streams (~12 GB/s), 4 cores", 1.2e10, 4),
    ("S3, 256 streams (~12 GB/s), 8 cores", 1.2e10, 8),
    ("NVMe x4 (~25 GB/s), 8 cores", 2.5e10, 8),
    ("in memory (~100 GB/s), 8 cores", 1e11, 8),
    ("in memory (~100 GB/s), 1 core", 1e11, 1),
]:
    p = scan(np.full(len(B), prod), bw, cores)
    bound = "network" if p[1] / bw > p[2] / cores else "CPU"
    (tf, af) = best_pipelined(bw, cores, allowed_for)
    (ta, aa) = best_pipelined(bw, cores, allowed_all)
    # The serial rule: per chunk minimise bytes / (bw / cores) + decode, i.e. per-core bandwidth.
    serial = np.where(ok, Bv / (bw / cores) + Dv, np.inf).argmin(axis=1)
    ts = scan(serial, bw, cores)
    print(f"{label:38s} {p[0] * 1e3:>10.0f}ms {bound:>9s}   {tf[0] / p[0] - 1:>+15.1%}   {ta[0] / p[0] - 1:>+14.1%}   {ts[0] / p[0] - 1:>+16.1%}")
