# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Compare sampling estimators against production and the one-step oracle.

Usage: uv run --no-project --with pandas python estimators.py <out-dir> [tag]
"""

import sys

import pandas as pd

KEY = ["source", "column", "chunk"]


def main() -> None:
    out = sys.argv[1]
    tag = sys.argv[2] if len(sys.argv) > 2 else "stock"
    rows = pd.read_csv(f"{out}/rows-{tag}.csv")
    ok = rows[rows.ok == 1]
    piv = ok.pivot_table(index=KEY, columns="variant", values="bytes", aggfunc="min")
    tim = ok.pivot_table(index=KEY, columns="variant", values="compress_ns", aggfunc="min")
    dec = ok.pivot_table(index=KEY, columns="variant", values="decode_ns_median", aggfunc="min")
    trees = ok.pivot_table(index=KEY, columns="variant", values="tree", aggfunc="first")
    forced = [c for c in piv.columns if c.startswith("forced/") and c != "forced/pco"]
    oracle = piv[forced + ["default"]].min(axis=1)
    base = piv["default"].sum()
    gap = base - oracle.sum()
    print(f"chunks {len(piv)}; default {base / 1e6:.2f} MB; one-step oracle {oracle.sum() / 1e6:.2f} MB (headroom {gap / base:.2%})\n")
    print(f"{'variant':22s} {'bytes vs default':>16s} {'headroom captured':>18s} {'regret vs oracle':>17s} "
          f"{'compress time':>14s} {'decode time':>12s} {'same tree':>10s}")
    for v in ["default", "nosample"] + sorted(c for c in piv.columns if c.startswith("est/")):
        b = piv[v].fillna(piv["default"])
        print(f"{v:22s} {b.sum() / base - 1:>16.2%} {(base - b.sum()) / gap:>18.1%} {b.sum() / oracle.sum() - 1:>17.2%} "
              f"{tim[v].sum() / tim['default'].sum():>13.2f}x {dec[v].sum() / dec['default'].sum():>11.2f}x "
              f"{(trees[v] == trees['default']).mean():>10.1%}")
    print("\nper source, bytes vs default:")
    src = piv.groupby(level="source").sum()
    cols = ["est/prod", "est/zero-ok", "est/serialized", "est/strat-4x1k", "est/strat-64x64", "est/serialized-all"]
    print((src[cols].div(src["default"], axis=0) - 1).map(lambda x: f"{x:+.2%}").to_string())
    print("\nworst regressions of est/serialized vs default (KB):")
    diff = (piv["est/serialized"] - piv["default"]) / 1e3
    print(diff.groupby(level=["source", "column"]).sum().sort_values(ascending=False).head(6).round(1).to_string())


if __name__ == "__main__":
    main()
