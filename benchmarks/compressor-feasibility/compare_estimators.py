# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Which features matter, and how the model's ratio estimates compare with the compressor's own.

Usage: uv run --no-project --with pandas --with scikit-learn python compare_estimators.py \
    <train-dir> <est-dir>

`<train-dir>` is the training run (`run_e2e.sh` writes it to `<work>/train`). `<est-dir>` is a
run with the `spy` and `forced/` variants and the model at a ratio-only bandwidth.
"""

import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance

import train

KEY = ["source", "column", "chunk"]
WIDTH = {"i8": 1, "u8": 1, "i16": 2, "u16": 2, "i32": 4, "u32": 4, "i64": 8, "u64": 8}


def importance(X, bytes_, dec, meta):
    print("## Feature importance (permutation, held-out columns, mean over candidates)\n")
    cols = pd.Series([f"{s}/{c}" for s, c, _ in X.index]).astype("category").cat.codes.to_numpy()
    test = cols % 5 == 0
    canon, length = meta.canonical_bytes.to_numpy(), meta.len.to_numpy()
    imp = {"bytes": [], "decode": []}
    for scheme in bytes_.columns:
        ok = bytes_[scheme].notna().to_numpy()
        tr, te = ~test & ok, test & ok
        if tr.sum() < 50 or te.sum() < 20:
            continue
        targets = {
            "bytes": np.log2(bytes_[scheme].to_numpy() / canon),
            "decode": np.log2(dec[scheme].to_numpy() / length),
        }
        for name, y in targets.items():
            m = HistGradientBoostingRegressor(max_depth=4, max_iter=40, learning_rate=0.2, random_state=0)
            m.fit(X[tr], y[tr])
            r = permutation_importance(m, X[te], y[te], n_repeats=3, random_state=0)
            imp[name].append(pd.Series(r.importances_mean, index=X.columns, name=scheme))
    for name, series in imp.items():
        table = pd.concat(series, axis=1)
        mean = table.mean(axis=1).sort_values(ascending=False)
        print(f"{name} model, top features (drop in R² when shuffled):")
        print("  " + ", ".join(f"{f} {v:.2f}" for f, v in mean.head(10).items()))
        print(f"  barely used (< 0.005): {', '.join(mean[mean < 0.005].index)}\n")


def estimator_accuracy(train_dir, est_dir, X, bytes_, dec, meta):
    print("## Ratio estimates vs the ratio each scheme actually achieves at the root\n")
    rows = pd.read_csv(f"{est_dir}/rows-stock.csv")
    est = pd.read_csv(f"{est_dir}/estimates-stock.csv")
    ok = rows[rows.ok == 1]
    forced = ok[ok.variant.str.startswith("forced/")].copy()
    forced["scheme"] = forced.variant.str.removeprefix("forced/")
    width = forced.ptype.map(WIDTH)
    # The compressor's estimates compare buffer bytes, so compare them with buffer-byte ratios.
    forced["actual_nb"] = forced.len * width / forced.nbytes.replace(0, np.nan)
    forced["actual_ser"] = forced.canonical_bytes / forced.bytes
    actual = forced.set_index(KEY + ["scheme"])

    # Model predictions, each from the model trained without that chunk's source.
    sources = X.index.get_level_values("source").to_numpy()
    schemes = [s for s in bytes_.columns if s not in ("production", "sizemodel")]
    pb, _ = train.cross_predict(sources, X, bytes_, dec, meta, schemes)
    model = (meta.canonical_bytes.to_numpy()[:, None] / pb).stack().rename("model_ratio")
    model.index.names = KEY + ["scheme"]

    est = est.set_index(KEY + ["scheme"])
    j = actual[["actual_nb", "actual_ser"]].join(est[["kind", "ratio"]], how="left").join(model, how="left")
    j = j[j.index.get_level_values("scheme") != "pco"]

    def err(e, a):
        d = np.abs(np.log2(e / a))
        d = d[np.isfinite(d)]
        return f"median error {np.median(d):5.0%}, within 10% {np.mean(d < np.log2(1.1)):5.0%}, n={len(d)}"

    print(f"{'estimate':38s} accuracy (|log2(estimate / actual)|)")
    for kind in ["closed_form", "sample", "callback"]:
        sub = j[j.kind == kind]
        if len(sub):
            print(f"  compressor, {kind:24s} {err(sub.ratio, sub.actual_nb)}")
    print(f"  {'compressor, any non-skip':36s} {err(j.ratio[j.kind != 'skip'], j.actual_nb[j.kind != 'skip'])}")
    print(f"  {'model, all candidates':36s} {err(j.model_ratio, j.actual_ser)}")
    both = j[j.kind.isin(['closed_form', 'sample', 'callback'])]
    print(f"  {'model, same cases as compressor':36s} {err(both.model_ratio, both.actual_ser)}")

    print("\nper scheme, median error (compressor vs model), and how often the compressor skips it:")
    for scheme, g in j.groupby(level="scheme"):
        nonskip = g[g.kind.isin(["closed_form", "sample", "callback"])]
        ce = np.nanmedian(np.abs(np.log2(nonskip.ratio / nonskip.actual_nb))) if len(nonskip) else np.nan
        me = np.nanmedian(np.abs(np.log2(g.model_ratio / g.actual_ser)))
        skipped = (g.kind == "skip").mean()
        print(f"  {scheme:12s} compressor {ce:6.0%}   model {me:6.0%}   skipped by compressor {skipped:5.0%}")

    # Skipped schemes that would have been the best root.
    best = forced[forced.scheme != "pco"].loc[lambda d: d.groupby(KEY).bytes.idxmin()].set_index(KEY + ["scheme"])
    jb = best.join(est[["kind"]], how="left")
    print(f"\nthe best root scheme was skipped outright by the compressor on {(jb.kind == 'skip').mean():.0%} of chunks")

    # Ranking: does each estimator order the schemes like reality?
    def rank_agreement(col, act):
        good = []
        for _, g in j.groupby(level=KEY):
            g = g[[col, act]].replace([np.inf, -np.inf], np.nan).dropna()
            if len(g) >= 3:
                good.append(g[col].rank().corr(g[act].rank()))
        return np.nanmean(good)

    print(f"rank correlation with actual ratios, per chunk: compressor {rank_agreement('ratio', 'actual_nb'):.2f}, "
          f"model {rank_agreement('model_ratio', 'actual_ser'):.2f}")


def model_accuracy(est_dir, X, bytes_, dec, meta):
    print("\n## Model ratio accuracy by split, and with the compressor's estimates as features\n")
    est = pd.read_csv(f"{est_dir}/estimates-stock.csv")
    est["value"] = np.where(est.kind == "skip", -1.0, np.log2(est.ratio.clip(upper=1e6)))
    cmp_feats = est.pivot_table(index=KEY, columns="scheme", values="value", aggfunc="first")
    cmp_feats = cmp_feats.add_prefix("cmp_").reindex(X.index).fillna(-1)
    Xc = X.join(cmp_feats)
    schemes = [s for s in bytes_.columns if s not in ("production", "sizemodel", "pco")]
    sources = X.index.get_level_values("source").to_numpy()
    cols = pd.Series([f"{s}/{c}" for s, c, _ in X.index]).astype("category").cat.codes.to_numpy()
    canon = meta.canonical_bytes.to_numpy()[:, None]
    actual = canon / bytes_[schemes].to_numpy()
    for split, groups in [("5-fold grouped by column", cols % 5), ("leave-one-source-out", sources)]:
        for label, feats in [("features", X), ("features + compressor estimates", Xc)]:
            pb, _ = train.cross_predict(groups, feats, bytes_, dec, meta, schemes)
            pred = canon / pb[schemes].to_numpy()
            d = np.abs(np.log2(pred / actual))
            d = d[np.isfinite(d)]
            # Does the predicted-best scheme match the actual best?
            best_pred = np.nanargmax(np.nan_to_num(pred, nan=-1), axis=1)
            true_bytes = np.nan_to_num(bytes_[schemes].to_numpy(), nan=np.inf)
            chosen = true_bytes[np.arange(len(true_bytes)), best_pred]
            regret = np.where(np.isfinite(chosen), chosen, meta.bytes.to_numpy()).sum() / true_bytes.min(axis=1).sum() - 1
            print(f"  {split:26s} {label:34s} median error {np.median(d):4.0%}, within 10% {np.mean(d < np.log2(1.1)):4.0%}, "
                  f"bytes regret of its pick {regret:+.1%}")
    prod_regret = meta.bytes.sum() / np.nan_to_num(bytes_[schemes].to_numpy(), nan=np.inf).min(axis=1).sum() - 1
    print(f"  {'production compressor (its own estimates)':61s} bytes regret {prod_regret:+.1%}")


def ratios(est_dir):
    print("\n## Compression ratio (canonical bytes / compressed bytes, real compressions)\n")
    nopco = pd.read_csv(f"{est_dir}/rows-nopco.csv")
    nopco = nopco[nopco.variant.str.startswith("model@")].assign(
        variant=lambda d: d.variant.str.replace("model@", "model-no-pco@"))
    rows = pd.concat([pd.read_csv(f"{est_dir}/rows-stock.csv"), nopco])
    ok = rows[rows.ok == 1]
    canon = ok[ok.variant == "default"].canonical_bytes.sum()
    forced = ok[ok.variant.str.startswith("forced/")]
    oracle = forced[forced.variant != "forced/pco"].groupby(KEY).bytes.min().sum()
    oracle_pco = forced.groupby(KEY).bytes.min().sum()
    lines = {
        "production (current thresholds)": ok[ok.variant == "default"].bytes.sum(),
        "model, ratio-only, production schemes": ok[ok.variant == "model-no-pco@ratio"].bytes.sum(),
        "best root (oracle), production schemes": oracle,
        "model, ratio-only, Pco allowed": ok[ok.variant == "model@ratio"].bytes.sum(),
        "best root (oracle), Pco allowed": oracle_pco,
    }
    base = lines["production (current thresholds)"]
    for name, b in lines.items():
        print(f"  {name:40s} {canon / b:5.2f}x   bytes {b / base - 1:+6.1%}")
    by_src = ok[ok.variant.isin(["default", "model-no-pco@ratio"])].groupby(["source", "variant"]).agg(
        c=("canonical_bytes", "sum"), b=("bytes", "sum"))
    r = (by_src.c / by_src.b).unstack()
    print("\n  per source (production schemes):\n" + r.rename(columns={"default": "production", "model-no-pco@ratio": "model"})
          .map(lambda x: f"{x:.2f}x").to_string())


def main() -> None:
    train_dir, est_dir = sys.argv[1], sys.argv[2]
    X, bytes_, dec, meta = train.load(train_dir, "stock")
    importance(X, bytes_, dec, meta)
    estimator_accuracy(train_dir, est_dir, X, bytes_, dec, meta)
    model_accuracy(est_dir, X, bytes_, dec, meta)
    ratios(est_dir)


if __name__ == "__main__":
    main()
