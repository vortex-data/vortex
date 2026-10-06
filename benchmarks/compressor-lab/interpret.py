# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""What the data says about the compressor's estimates, in rules a person can read.

1. Calibration: closed-form size estimates vs the size each scheme actually reaches at the root.
2. Caps vs bit-cost rules: when does RunEnd / Sparse / Dict beat plain width-based packing
   (BitPacking or FOR)? Today's caps against inequalities in bits.
3. A shallow tree distilled from the best root, to read its splits.

Usage: uv run --no-project --with pandas --with scikit-learn python interpret.py <dataset-dir>
"""

import sys

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text

KEY = ["source", "column", "chunk"]
WIN = 0.95  # a scheme "wins" when it is at least 5% smaller than width-based packing


def load(dataset: str):
    rows = pd.read_csv(f"{dataset}/rows-stock.csv", dtype={"chunk": str})
    feats = pd.read_csv(f"{dataset}/features-stock.csv", dtype={"chunk": str}).set_index(KEY)
    ok = rows[(rows.ok == 1) & (rows.variant != "forced/pco")]
    b = ok.pivot_table(index=KEY, columns="variant", values="bytes", aggfunc="min")
    b.columns = [c.replace("forced/", "").replace("model/runend+sparse", "sizemodel") for c in b.columns]
    f = feats.reindex(b.index)
    return b, f


def bits_width(f):
    """Bits per value plain packing needs: the narrower of BitPacking and FOR."""
    bp = f.bits_bp.where(f.bits_bp >= 0, f.ptype_bits)
    return np.minimum(bp, f.bits_for).clip(lower=1)


def calibration(b, f):
    print("## 1. Closed-form estimates vs actual size at the root (bits per value)\n")
    n = f.len * (1 - f.null_frac)
    pos = np.log2(f.len).clip(lower=1)
    runs = (n / f.avg_run.clip(lower=1)).clip(lower=1)
    est = {
        "bitpacking": n * f.bits_bp.where(f.bits_bp >= 0, f.ptype_bits),
        "for": n * f.bits_for,
        "runend": runs * (f.bits_for + pos),
        "sparse": (n * (1 - f.top1_frac)).clip(lower=1) * (f.bits_for + pos),
        "dict": f.distinct * f.ptype_bits + n * np.log2(f.distinct.clip(lower=2)),
    }
    print(f"{'scheme':12s} {'actual / estimate (median)':>27s} {'IQR':>15s} {'rank corr':>10s}  reading")
    notes = {
        "bitpacking": "exact up to patches and metadata",
        "for": "exact up to metadata",
        "runend": "children compress well: ends and values are cascaded",
        "sparse": "exception values and positions compress below their raw width",
        "dict": "codes are cascaded (RLE/bit-packing), so the formula overstates",
    }
    for scheme, e in est.items():
        if scheme not in b:
            continue
        ratio = (b[scheme] * 8 / e).replace([np.inf, -np.inf], np.nan).dropna()
        corr = pd.concat([b[scheme], e], axis=1).dropna().corr(method="spearman").iloc[0, 1]
        q1, q2, q3 = ratio.quantile([0.25, 0.5, 0.75])
        print(f"{scheme:12s} {q2:>27.2f} {q1:>7.2f}–{q3:<7.2f} {corr:>10.2f}  {notes[scheme]}")


def rule_report(name, wins, rule, b, scheme, width):
    tp = (rule & wins).sum()
    fp = (rule & ~wins).sum()
    fn = (~rule & wins).sum()
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    # Bytes if the rule chooses between the scheme and width packing, vs the better of the two.
    chosen = np.where(rule, b[scheme], width).sum()
    best = np.minimum(b[scheme].fillna(np.inf), width).sum()
    print(f"  {name:52s} precision {precision:4.0%}  recall {recall:4.0%}  bytes vs best {chosen / best - 1:+6.1%}")


def caps(b, f):
    print("\n## 2. Today's caps vs bit-cost rules: does the scheme beat width-based packing by 5%?\n")
    width = b[["bitpacking", "for"]].min(axis=1)
    w = bits_width(f)
    pos = np.log2(f.len).clip(lower=1)
    n_ok = b.index[width.notna()]

    for scheme in ["runend", "sparse", "dict"]:
        sub = b.loc[n_ok]
        ff = f.loc[n_ok]
        ww = w.loc[n_ok]
        wins = (sub[scheme] < WIN * width.loc[n_ok]).fillna(False)
        print(f"{scheme}: wins on {wins.mean():.0%} of {len(sub)} chunks")
        if scheme == "runend":
            rule_report("today: avg_run >= 4", wins, ff.avg_run >= 4, sub, scheme, width.loc[n_ok])
            rule_report("bit cost: avg_run > (bits_for + log2 len) / width", wins,
                        ff.avg_run > (ff.bits_for + pos.loc[n_ok]) / ww, sub, scheme, width.loc[n_ok])
            # Ends are themselves FOR-packed, so they cost far fewer than log2(len) bits; fit it.
            best = max(range(0, 17), key=lambda e: (
                np.where(ff.avg_run > (ff.bits_for + e) / ww, sub[scheme], width.loc[n_ok]).sum() * -1))
            rule_report(f"bit cost, fitted: avg_run > (bits_for + {best}) / width", wins,
                        ff.avg_run > (ff.bits_for + best) / ww, sub, scheme, width.loc[n_ok])
        elif scheme == "sparse":
            exc = 1 - ff.top1_frac
            rule_report("today: top value >= 90%", wins, ff.top1_frac >= 0.9, sub, scheme, width.loc[n_ok])
            rule_report("bit cost: exceptions < width / (bits_for + log2 len)", wins,
                        exc < ww / (ff.bits_for + pos.loc[n_ok]), sub, scheme, width.loc[n_ok])
            best = max(np.linspace(0.5, 4, 36), key=lambda k: (
                -np.where(exc < k * ww / (ff.bits_for + pos.loc[n_ok]), sub[scheme], width.loc[n_ok]).sum()))
            rule_report(f"bit cost, fitted: exceptions < {best:.1f} × width / (bits_for + log2 len)", wins,
                        exc < best * ww / (ff.bits_for + pos.loc[n_ok]), sub, scheme, width.loc[n_ok])
        else:
            rule_report("today: distinct <= 50%", wins, ff.distinct_frac <= 0.5, sub, scheme, width.loc[n_ok])
            dict_bits = ff.distinct_frac * ff.ptype_bits + np.log2(ff.distinct.clip(lower=2))
            rule_report("bit cost: distinct_frac × ptype + log2 distinct < width", wins,
                        dict_bits < ww, sub, scheme, width.loc[n_ok])
            best = max(np.linspace(0.3, 2, 35), key=lambda k: (
                -np.where(dict_bits < k * ww, sub[scheme], width.loc[n_ok]).sum()))
            rule_report(f"bit cost, fitted: dict bits < {best:.2f} × width", wins,
                        dict_bits < best * ww, sub, scheme, width.loc[n_ok])
        print()


def distilled(b, f):
    print("## 3. A depth-3 tree for the smallest root, held out by source\n")
    # Delta and Sequence fall back to production's tree when they don't apply, which ties them with
    # other schemes; keep the distinct schemes, ordered simplest first so ties go to the simpler one.
    cols = [c for c in ["bitpacking", "for", "zigzag", "runend", "sparse", "dict", "rle"] if c in b]
    label = b[cols].fillna(np.inf).idxmin(axis=1)
    X = f.fillna(-1).drop(columns=["len"])
    sources = b.index.get_level_values("source")
    chosen = pd.Series(index=b.index, dtype=float)
    for s in sources.unique():
        tr, te = sources != s, sources == s
        clf = DecisionTreeClassifier(max_depth=3, min_samples_leaf=20, random_state=0).fit(X[tr], label[tr])
        pick = clf.predict(X[te])
        chosen[te] = [b.loc[i, p] if np.isfinite(b.loc[i, p]) else b.loc[i, "default"] for i, p in zip(b.index[te], pick)]
    print(f"label shares: {label.value_counts(normalize=True).round(2).to_dict()}")
    print(f"bytes vs production: {chosen.sum() / b['default'].sum() - 1:+.1%}; "
          f"vs best of these roots: {chosen.sum() / b[cols].min(axis=1).sum() - 1:+.1%}")
    clf = DecisionTreeClassifier(max_depth=3, min_samples_leaf=20, random_state=0).fit(X, label)
    print(export_text(clf, feature_names=list(X.columns), decimals=2))


def main() -> None:
    b, f = load(sys.argv[1])
    calibration(b, f)
    caps(b, f)
    distilled(b, f)


if __name__ == "__main__":
    main()
