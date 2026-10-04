# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Analyse compressor-feasibility CSVs and answer the four feasibility questions.

Usage: uv run --with pandas --with scikit-learn analyze.py <out-dir> [tag]
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text

KEY = ["source", "column", "chunk"]
PRODUCTION_EXCLUDED = {"forced/pco"}


def pct(x: float) -> str:
    return f"{100 * x:.2f}%"


def load(out: Path, tag: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = pd.read_csv(out / f"rows-{tag}.csv")
    feats = pd.read_csv(out / f"features-{tag}.csv")
    return rows, feats


def q1(rows: pd.DataFrame) -> pd.DataFrame:
    print("\n## Q1: headroom over the current compressor (one-step oracle: best forced root, children as today)\n")
    ok = rows[rows.ok == 1]
    default = ok[ok.variant == "default"].set_index(KEY)
    cands = ok[ok.variant.str.startswith("forced/") & ~ok.variant.isin(PRODUCTION_EXCLUDED)]
    cands = pd.concat([cands, ok[ok.variant == "default"]])
    best = cands.loc[cands.groupby(KEY).bytes.idxmin()].set_index(KEY)
    df = default[["bytes", "canonical_bytes", "root", "compress_ns", "decode_ns_median"]].join(
        best[["bytes", "variant", "decode_ns_median"]], rsuffix="_oracle"
    )
    df["gain"] = 1 - df.bytes_oracle / df.bytes
    tot_d, tot_o, tot_c = df.bytes.sum(), df.bytes_oracle.sum(), df.canonical_bytes.sum()
    print(f"chunks: {len(df)}; canonical {tot_c / 1e6:.1f} MB")
    print(f"default bytes: {tot_d / 1e6:.2f} MB (ratio {tot_c / tot_d:.2f}x)")
    print(f"oracle  bytes: {tot_o / 1e6:.2f} MB (ratio {tot_c / tot_o:.2f}x)")
    print(f"total headroom: {pct(1 - tot_o / tot_d)}")
    print(f"chunks where default is already optimal (within 0.1%): {pct((df.gain <= 0.001).mean())}")
    print(f"chunks with gain > 5%: {pct((df.gain > 0.05).mean())}, > 20%: {pct((df.gain > 0.2).mean())}")
    by_src = df.groupby(level="source")[["bytes", "bytes_oracle"]].sum()
    by_src["headroom"] = 1 - by_src.bytes_oracle / by_src.bytes
    print("\nper source:\n" + by_src.assign(headroom=by_src.headroom.map(pct)).to_string())
    col = df.groupby(level=["source", "column"])[["bytes", "bytes_oracle"]].sum()
    col["saved_kb"] = (col.bytes - col.bytes_oracle) / 1e3
    col["gain"] = (1 - col.bytes_oracle / col.bytes).map(pct)
    print("\ntop columns by bytes saved:\n" + col.sort_values("saved_kb", ascending=False).head(12).to_string())
    print("\nwinning root scheme when it beats default by >1%:")
    print(df[df.gain > 0.01].variant.value_counts().to_string())

    with_pco = ok[ok.variant.str.startswith("forced/") | (ok.variant == "default")]
    best_pco = with_pco.groupby(KEY).bytes.min()
    print(f"\nheadroom if Pco were allowed: {pct(1 - best_pco.sum() / tot_d)}")
    return df


def q1_time(rows: pd.DataFrame) -> None:
    print("\n## Q1b: where compression time goes (default vs closed-form estimates, no sampling)\n")
    ok = rows[rows.ok == 1]
    d = ok[ok.variant == "default"].set_index(KEY).compress_ns
    n = ok[ok.variant == "nosample"].set_index(KEY).compress_ns
    both = pd.concat([d, n], axis=1, keys=["default", "nosample"]).dropna()
    mb = ok[ok.variant == "default"].set_index(KEY).canonical_bytes.loc[both.index].sum() / 1e6
    print(f"default compress: {both.default.sum() / 1e9:.2f}s ({mb / (both.default.sum() / 1e9):.0f} MB/s)")
    print(f"nosample compress: {both.nosample.sum() / 1e9:.2f}s ({mb / (both.nosample.sum() / 1e9):.0f} MB/s)")
    print(f"time attributable to sampling/deferred estimates: {pct(1 - both.nosample.sum() / both.default.sum())}")


def q2(rows: pd.DataFrame, feats: pd.DataFrame, df: pd.DataFrame) -> None:
    print("\n## Q2: choosing without sampling\n")
    ok = rows[rows.ok == 1]
    default = ok[ok.variant == "default"].set_index(KEY)
    ns = ok[ok.variant == "nosample"].set_index(KEY)
    j = default[["bytes", "root"]].join(ns[["bytes", "root"]], rsuffix="_ns", how="inner")
    print(f"closed-form (no sampling) vs default: bytes {pct(j.bytes_ns.sum() / j.bytes.sum() - 1)} "
          f"({'worse' if j.bytes_ns.sum() > j.bytes.sum() else 'better'}), same root encoding on {pct((j.root == j.root_ns).mean())} of chunks")
    oracle = df.bytes_oracle.reindex(j.index)
    print(f"regret vs oracle: default {pct(j.bytes.sum() / oracle.sum() - 1)}, closed-form {pct(j.bytes_ns.sum() / oracle.sum() - 1)}")

    # A shallow tree on features only, predicting the best forced root; evaluated leave-one-source-out.
    cands = ok[ok.variant.str.startswith("forced/") & ~ok.variant.isin(PRODUCTION_EXCLUDED)]
    table = cands.pivot_table(index=KEY, columns="variant", values="bytes", aggfunc="min")
    table = table.join(default.bytes.rename("default"), how="inner")
    f = feats.set_index(KEY).reindex(table.index)
    X = f.drop(columns=[]).fillna(-1).replace([np.inf, -np.inf], -1)
    schemes = [c for c in table.columns if c.startswith("forced/")]
    filled = table[schemes].fillna(np.inf)
    label = filled.idxmin(axis=1)
    sources = table.index.get_level_values("source")
    columns = pd.Series([f"{a}/{b}" for a, b, _ in table.index], index=table.index)
    col_ids = columns.astype("category").cat.codes.to_numpy()
    folds = {
        "leave-one-source-out": sources.to_numpy(),
        "5-fold grouped by column": col_ids % 5,
    }
    oracle_total = filled.min(axis=1).sum()
    for split, groups in folds.items():
        for depth in (2, 3, 4, 6):
            chosen = pd.Series(index=table.index, dtype=float)
            for g in np.unique(groups):
                train, test = groups != g, groups == g
                clf = DecisionTreeClassifier(max_depth=depth, random_state=0)
                # Weight by how much a wrong choice can cost on that chunk.
                w = (filled[train].replace(np.inf, np.nan).max(axis=1) - filled[train].min(axis=1)).fillna(0) + 1
                clf.fit(X[train], label[train], sample_weight=w)
                pred = clf.predict(X[test])
                chosen[test] = [filled.loc[idx, p] for idx, p in zip(table.index[test], pred)]
            chosen = chosen.replace(np.inf, np.nan).fillna(table.default)
            print(f"tree depth {depth} ({split}): bytes vs default {pct(chosen.sum() / table.default.sum() - 1)}, "
                  f"regret vs oracle {pct(chosen.sum() / oracle_total - 1)}, matches oracle on {pct((chosen <= filled.min(axis=1) * 1.001).mean())}")
    clf = DecisionTreeClassifier(max_depth=3, random_state=0).fit(X, label)
    print("\ndepth-3 tree on all data:\n" + export_text(clf, feature_names=list(X.columns)))


def q3(rows: pd.DataFrame) -> None:
    print("\n## Q3: removing heuristic caps (sampling decides instead)\n")
    ok = rows[rows.ok == 1]
    default = ok[ok.variant == "default"].set_index(KEY)
    for v in sorted(ok.variant[ok.variant.str.startswith("capoff/")].unique()):
        c = ok[ok.variant == v].set_index(KEY)
        j = default[["bytes", "decode_ns_median", "compress_ns"]].join(c[["bytes", "decode_ns_median", "compress_ns"]], rsuffix="_v", how="inner")
        changed = (j.bytes_v != j.bytes).mean()
        print(f"{v:16s} bytes {pct(j.bytes_v.sum() / j.bytes.sum() - 1):>8s}  decode {pct(j.decode_ns_median_v.sum() / j.decode_ns_median.sum() - 1):>8s}  "
              f"compress(1 rep) {pct(j.compress_ns_v.sum() / j.compress_ns.sum() - 1):>8s}  chunks changed {pct(changed)}")


def q4(rows: pd.DataFrame) -> None:
    print("\n## Q4: combining decode time with size\n")
    ok = rows[rows.ok == 1]
    cands = ok[(ok.variant == "default") | (ok.variant.str.startswith("forced/") & ~ok.variant.isin(PRODUCTION_EXCLUDED))].copy()
    # The same tree reached by two variants is one candidate; keep its fastest timing.
    cands = cands.sort_values("decode_ns_median").drop_duplicates(KEY + ["tree"])
    cands["dec_s"] = cands.decode_ns_median / 1e9
    default = ok[ok.variant == "default"].set_index(KEY)
    d_dec = default.decode_ns_median / 1e9
    print(f"default decode throughput: {default.canonical_bytes.sum() / d_dec.sum() / 1e9:.2f} GB/s of canonical data")
    pco = ok[ok.variant == "forced/pco"].set_index(KEY)
    j = default.join(pco, rsuffix="_pco", how="inner")
    print(f"Pco at the root: bytes {pct(j.bytes_pco.sum() / j.bytes.sum() - 1)}, decode time {j.decode_ns_median_pco.sum() / j.decode_ns_median.sum():.1f}x default")
    smallest = cands.groupby(KEY).bytes.transform("min")
    near = cands[cands.bytes <= smallest * 1.01]
    fastest_near = near.loc[near.groupby(KEY).dec_s.idxmin()].set_index(KEY)
    small = cands.loc[cands.groupby(KEY).bytes.idxmin()].set_index(KEY)
    print(f"among candidates within 1% of the smallest, picking the fastest decoder saves "
          f"{pct(1 - fastest_near.dec_s.sum() / small.dec_s.sum())} decode time for {pct(fastest_near.bytes.sum() / small.bytes.sum() - 1)} bytes")
    for name, bw in [("S3 ~100 MB/s", 100e6), ("NVMe ~2 GB/s", 2e9), ("memory ~20 GB/s", 20e9)]:
        cands["cost"] = cands.bytes / bw + cands.dec_s
        win = cands.loc[cands.groupby(KEY).cost.idxmin()].set_index(KEY)
        win_bytes = win.loc[default.index].bytes
        cost_default = default.bytes / bw + d_dec
        cost_small = small.bytes / bw + small.dec_s
        changed = (win.loc[default.index].tree != default.tree).mean()
        print(f"{name:16s}: vs default -> cost {pct(win.cost.sum() / cost_default.sum() - 1)}, bytes {pct(win_bytes.sum() / default.bytes.sum() - 1)}, "
              f"decode {pct(win.dec_s.sum() / d_dec.sum() - 1)}, tree changes on {pct(changed)}; "
              f"size-only oracle cost {pct(cost_small.sum() / cost_default.sum() - 1)}")


def main() -> None:
    out = Path(sys.argv[1])
    tag = sys.argv[2] if len(sys.argv) > 2 else "stock"
    rows, feats = load(out, tag)
    errs = rows[rows.ok == 0].variant.value_counts()
    if len(errs):
        print("variants that failed on some chunks (infeasible forced schemes):\n" + errs.to_string())
    df = q1(rows)
    q1_time(rows)
    q2(rows, feats, df)
    q3(rows)
    q4(rows)


if __name__ == "__main__":
    main()
