# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Train the learned integer scheme selector and generate its Rust model.

Reads the tables `vx-selector gather` writes: `chunks.csv`, `sites.csv` (one row per integer
compression site, with the features `vortex_compressor::selector::int_features` computes) and
`candidates.csv` (the whole chunk's bytes and decode time with each scheme forced at that site and
production everywhere else).

The model is one shallow decision tree. Each row is one site; its label is the scheme with the
lowest cost, `bytes / bandwidth + decode seconds`, and its weight is the regret of guessing wrong,
so near-ties count for little and big chunks with big misses count for a lot. No one column may
carry more than `--column-cap` of the total weight. Each leaf then ranks schemes by their total
regret over the leaf's rows; the first is the pick, and the leaf's confidence is the weighted share
of rows where the pick is within `--tolerance` of the best.

Evaluation is leave-one-group-out: for each group (a dataset, or a TPC-H table), train on the rest
and score the held-out sites. A site is decided by the tree when its leaf is confident and its
features are in the training range; otherwise it falls back to estimating over the leaf's top-k,
emulated here as production's pick when production's pick is in the top-k, else the best of the
top-k. The real end-to-end numbers come from `vx-selector eval`.

Usage:
    python train_selector.py --data out/ [--bandwidth 2e9] [--export ../../vortex-compressor/src/selector/generated/int_v1.rs]
"""

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

FEATURES = [
    "log2_len",
    "ptype_bits",
    "signed",
    "null_frac",
    "bits_bp",
    "bits_for",
    "for_gain",
    "min_is_negative",
    "log2_avg_run",
    "distinct_frac",
    "bits_distinct",
    "top_frac",
    "depth",
    "parent",
    "child_index",
]


def load(data: Path, bandwidth: float):
    sites = pd.read_csv(data / "sites.csv", keep_default_na=False)
    cands = pd.read_csv(data / "candidates.csv", keep_default_na=False)
    # A forced scheme that fell back to something else is infeasible at that site.
    cands = cands[(cands.applied == cands.scheme) & (cands.scheme != "canonical")].copy()
    cands["cost"] = cands.bytes / bandwidth + cands.decode_ns / 1e9
    sites = sites.drop_duplicates(["chunk", "path"])
    sites = sites[sites.set_index(["chunk", "path"]).index.isin(cands.set_index(["chunk", "path"]).index.unique())]
    cost = cands.pivot_table(index=["chunk", "path"], columns="scheme", values="cost", aggfunc="first")
    cost = cost.reindex(pd.MultiIndex.from_frame(sites[["chunk", "path"]]))
    return sites.reset_index(drop=True), cost.reset_index(drop=True)


def weights(sites: pd.DataFrame, cost: pd.DataFrame, cap: float) -> np.ndarray:
    c = cost.to_numpy()
    best = np.nanmin(c, axis=1)
    worst = np.where(np.isnan(c), np.nan, c)
    regret = np.nanmean(worst - best[:, None], axis=1)
    w = np.nan_to_num(regret, nan=0.0)
    # Cap any one column's share of the total weight.
    total = w.sum()
    if total > 0:
        for col, idx in sites.groupby("column").indices.items():
            share = w[idx].sum() / total
            if share > cap:
                w[idx] *= cap / share
    return w + 1e-12


def regret_matrix(cost: pd.DataFrame) -> np.ndarray:
    c = cost.to_numpy()
    best = np.nanmin(c, axis=1)
    r = c - best[:, None]
    # An infeasible pick falls back to estimating; charge it the worst feasible regret.
    worst = np.nanmax(np.where(np.isnan(r), -np.inf, r), axis=1)
    return np.where(np.isnan(r), worst[:, None], r)


class Model:
    def __init__(self, depth: int, min_leaf: int, tolerance: float):
        self.depth, self.min_leaf, self.tolerance = depth, min_leaf, tolerance

    def fit(self, X: np.ndarray, cost: pd.DataFrame, w: np.ndarray):
        self.schemes = list(cost.columns)
        c = cost.to_numpy()
        label = np.nanargmin(np.where(np.isnan(c), np.inf, c), axis=1)
        self.tree = DecisionTreeClassifier(max_depth=self.depth, min_samples_leaf=self.min_leaf, random_state=0)
        self.tree.fit(X, label, sample_weight=w)
        R = regret_matrix(cost)
        best = np.nanmin(c, axis=1)
        leaf_of = self.tree.apply(X)
        self.leaves = {}
        for leaf in np.unique(leaf_of):
            rows = leaf_of == leaf
            total = (R[rows] * w[rows, None]).sum(axis=0)
            feasible = (~np.isnan(c[rows])).any(axis=0)
            order = [i for i in np.argsort(total) if feasible[i]]
            pick = order[0]
            ok = c[rows, pick] <= best[rows] * (1 + self.tolerance)
            ok &= ~np.isnan(c[rows, pick])
            confidence = float((w[rows] * ok).sum() / w[rows].sum())
            self.leaves[leaf] = (order, confidence)
        self.fmin, self.fmax = X.min(axis=0), X.max(axis=0)
        return self

    def decide(self, X: np.ndarray):
        leaf_of = self.tree.apply(X)
        in_range = ((X >= self.fmin) & (X <= self.fmax)).all(axis=1)
        return [(self.leaves[l][0], self.leaves[l][1], r) for l, r in zip(leaf_of, in_range)]


def score(model: Model, X, cost: pd.DataFrame, production: pd.Series, min_conf: float, k: int):
    """Total cost of the chosen schemes, and how many sites skipped estimation."""
    c = cost[model.schemes].to_numpy()
    chosen = np.empty(len(X))
    picked = 0
    for i, (order, conf, in_range) in enumerate(model.decide(X)):
        prod = production.iat[i]
        prod_idx = model.schemes.index(prod) if prod in model.schemes else None
        if conf >= min_conf and in_range:
            picked += 1
            j = order[0]
            chosen[i] = c[i, j] if not np.isnan(c[i, j]) else c[i, prod_idx] if prod_idx is not None else np.nanmin(c[i])
        else:
            top = order[:k] if k else list(range(len(model.schemes)))
            if prod_idx is not None and prod_idx in top:
                chosen[i] = c[i, prod_idx]
            else:
                vals = [c[i, j] for j in top if not np.isnan(c[i, j])]
                chosen[i] = min(vals) if vals else np.nanmin(c[i])
    return chosen, picked


def production_cost(cost: pd.DataFrame, production: pd.Series) -> np.ndarray:
    c = cost.to_numpy()
    cols = {s: i for i, s in enumerate(cost.columns)}
    out = np.array([c[i, cols[p]] if p in cols else np.nan for i, p in enumerate(production)])
    return np.where(np.isnan(out), np.nanmin(c, axis=1), out)


def evaluate(sites, cost, args):
    X = sites[FEATURES].to_numpy(dtype=float)
    groups = sites.group.to_numpy()
    w = weights(sites, cost, args.column_cap)
    prod = production_cost(cost, sites.applied)
    oracle = np.nanmin(cost.to_numpy(), axis=1)
    print(f"{len(sites)} sites, {len(np.unique(sites.chunk))} chunks, {len(np.unique(groups))} groups, "
          f"{cost.shape[1]} schemes")
    print(f"production vs oracle (one-step, per site): {prod.sum() / oracle.sum() - 1:+.2%}\n")
    print(f"{'held-out group':24s} {'sites':>6s} {'model vs prod':>14s} {'oracle vs prod':>15s} {'skipped est.':>13s}")
    tot_m = tot_p = tot_o = 0.0
    tot_pick = 0
    for g in np.unique(groups):
        test = groups == g
        if test.all():
            continue
        m = Model(args.depth, args.min_leaf, args.tolerance).fit(X[~test], cost[~test], w[~test])
        chosen, picked = score(m, X[test], cost[test], sites.applied[test], args.min_confidence, args.fallback_k)
        p, o = prod[test].sum(), oracle[test].sum()
        tot_m += chosen.sum(); tot_p += p; tot_o += o; tot_pick += picked
        print(f"{g:24s} {test.sum():>6d} {chosen.sum() / p - 1:>+14.2%} {o / p - 1:>+15.2%} {picked / test.sum():>13.0%}")
    print(f"{'all (leave-one-out)':24s} {len(sites):>6d} {tot_m / tot_p - 1:>+14.2%} {tot_o / tot_p - 1:>+15.2%} "
          f"{tot_pick / len(sites):>13.0%}")


def rust_f64(x: float) -> str:
    if np.isinf(x):
        return "f64::INFINITY" if x > 0 else "f64::NEG_INFINITY"
    r = repr(float(x))
    return r if ("." in r or "e" in r) else r + ".0"


def export(model: Model, path: Path, data_hash: str, args):
    t = model.tree.tree_
    leaf_ids = {}
    nodes = []
    for n in range(t.node_count):
        if t.children_left[n] == -1:
            leaf_ids[n] = len(leaf_ids)
            nodes.append(f"Node::Leaf({leaf_ids[n]})")
        else:
            nodes.append(
                f"Node::Split {{ feature: {t.feature[n]}, threshold: {rust_f64(t.threshold[n])}, "
                f"left: {t.children_left[n]}, right: {t.children_right[n]} }}"
            )
    leaves = []
    for n in sorted(leaf_ids, key=leaf_ids.get):
        order, conf = model.leaves[n]
        leaves.append(f"Leaf {{ ranked: &[{', '.join(str(i) for i in order)}], confidence: {rust_f64(conf)} }}")
    schemes = ", ".join(f'"{s}"' for s in model.schemes)
    body = f"""// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

// @generated by benchmarks/compressor-lab/train_selector.py. Do not edit by hand.
// depth {args.depth}, min leaf {args.min_leaf}, tolerance {args.tolerance}, bandwidth {args.bandwidth:g} B/s

use crate::selector::INT_FEATURE_NAMES;
use crate::selector::Leaf;
use crate::selector::Node;
use crate::selector::TreeModel;

/// The integer selector.
#[rustfmt::skip]
pub static INT_V1: TreeModel = TreeModel {{
    name: "int-v1",
    training_hash: "{data_hash}",
    schemes: &[{schemes}],
    feature_names: &INT_FEATURE_NAMES,
    feature_min: &[{', '.join(rust_f64(x) for x in model.fmin)}],
    feature_max: &[{', '.join(rust_f64(x) for x in model.fmax)}],
    nodes: &[
        {(',' + chr(10) + '        ').join(nodes)},
    ],
    leaves: &[
        {(',' + chr(10) + '        ').join(leaves)},
    ],
}};
"""
    path.write_text(body)
    print(f"\nwrote {path}: {t.node_count} nodes, {len(leaves)} leaves")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--bandwidth", type=float, default=2e9, help="bytes per second; cost = bytes/bw + decode s")
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--min-leaf", type=int, default=5)
    p.add_argument("--tolerance", type=float, default=0.01)
    p.add_argument("--min-confidence", type=float, default=0.9)
    p.add_argument("--fallback-k", type=int, default=3)
    p.add_argument("--column-cap", type=float, default=0.05)
    p.add_argument("--export", type=Path)
    p.add_argument("--holdout", action="append", default=[], help="group left out of the exported model")
    args = p.parse_args()

    sites, cost = load(args.data, args.bandwidth)
    unknown = set(FEATURES) - set(sites.columns)
    if unknown:
        sys.exit(f"sites.csv lacks features {sorted(unknown)}")
    evaluate(sites, cost, args)
    if args.export:
        keep = ~sites.group.isin(args.holdout).to_numpy()
        sites, cost = sites[keep].reset_index(drop=True), cost[keep].reset_index(drop=True)
        X = sites[FEATURES].to_numpy(dtype=float)
        w = weights(sites, cost, args.column_cap)
        model = Model(args.depth, args.min_leaf, args.tolerance).fit(X, cost, w)
        digest = hashlib.sha256((args.data / "candidates.csv").read_bytes()).hexdigest()[:16]
        export(model, args.export, digest, args)


if __name__ == "__main__":
    main()
