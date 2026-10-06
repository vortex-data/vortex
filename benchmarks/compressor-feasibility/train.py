# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Train a selector that trades compression ratio against decode time.

The selector proposes; the compressor verifies. The model's best alternative to the production
choice is compressed only when its predicted cost is at least `gate` lower, and kept only when
its real cost is lower, so the result is never worse than production.

For every root scheme, regressors predict log2(compressed bytes / canonical bytes) and
log2(decode ns per value) from chunk features, and a classifier predicts whether the scheme is
feasible. At inference the selector picks the feasible scheme with the lowest
`bytes / bandwidth + decode_time`, so bandwidth is a knob rather than a training choice.

Usage: uv run --no-project --with pandas --with scikit-learn python train.py <out-dir> [tag]
"""

import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

KEY = ["source", "column", "chunk"]
BANDWIDTHS = {
    "S3 100 MB/s": 100e6,
    "NVMe 2 GB/s": 2e9,
    "memory 20 GB/s": 20e9,
}


def load(out: str, tag: str):
    rows = pd.read_csv(f"{out}/rows-{tag}.csv")
    feats = pd.read_csv(f"{out}/features-{tag}.csv").set_index(KEY)
    # Candidates: each scheme forced at the root, plus whole policies we can already run.
    policies = {"default": "production", "model/runend+sparse": "sizemodel"}
    cand = rows[rows.variant.str.startswith("forced/") | rows.variant.isin(policies)].copy()
    cand["scheme"] = cand.variant.str.removeprefix("forced/").replace(policies)
    ok = cand[cand.ok == 1]
    bytes_ = ok.pivot_table(index=KEY, columns="scheme", values="bytes", aggfunc="min")
    dec = ok.pivot_table(index=KEY, columns="scheme", values="decode_ns_median", aggfunc="min")
    meta = rows[rows.variant == "default"].set_index(KEY)
    index = meta.index
    bytes_, dec = bytes_.reindex(index), dec.reindex(index)
    X = feats.reindex(index).replace([np.inf, -np.inf], np.nan).fillna(-1)
    X["log_len"] = np.log2(meta.len)
    X = add_estimates(X)
    X = X.drop(columns=["len"])
    return X, bytes_, dec, meta


def add_estimates(X: pd.DataFrame) -> pd.DataFrame:
    """Closed-form size estimates per scheme, as log2 of bits per value relative to the ptype."""
    n = X.len * (1 - X.null_frac)
    w = X.ptype_bits
    pos = np.log2(X.len).clip(lower=1)
    runs = (n / X.avg_run.clip(lower=1)).clip(lower=1)
    patches = (n * (1 - X.top1_frac)).clip(lower=1)
    est = {
        "bp": n * X.bits_bp.where(X.bits_bp >= 0, w),
        "for": n * X.bits_for,
        "zz": n * X.bits_zz,
        "delta": n * X.bits_delta,
        "dict": X.distinct * w + n * np.log2(X.distinct.clip(lower=2)),
        "runend": runs * (X.bits_for + pos),
        "sparse": patches * (X.bits_for + pos),
        "p90_patched": n * X.bits_p90 + n * 0.1 * (w + pos),
    }
    for name, bits in est.items():
        X[f"est_{name}"] = np.log2(bits.clip(lower=1) / (X.len * w))
    return X


def cost(bytes_, dec_ns, bw):
    return bytes_ / bw + dec_ns / 1e9


def fit_predict(X, bytes_, dec, meta, train, test):
    """Predicted bytes and decode ns for every scheme on the test rows (NaN when infeasible)."""
    pb = pd.DataFrame(index=X.index[test], columns=bytes_.columns, dtype=float)
    pd_ = pb.copy()
    canon = meta.canonical_bytes.to_numpy()
    length = meta.len.to_numpy()
    for scheme in bytes_.columns:
        feasible = bytes_[scheme].notna().to_numpy()
        tr = train & feasible
        if tr.sum() < 20:
            continue
        if feasible[train].all():
            p_ok = np.ones(test.sum())
        elif not feasible[train].any():
            continue
        else:
            clf = HistGradientBoostingClassifier(max_depth=4, random_state=0)
            clf.fit(X[train], feasible[train])
            p_ok = clf.predict_proba(X[test])[:, list(clf.classes_).index(True)]
        yb = np.log2(bytes_[scheme].to_numpy()[tr] / canon[tr])
        yd = np.log2(dec[scheme].to_numpy()[tr] / length[tr])
        rb = HistGradientBoostingRegressor(max_depth=4, random_state=0).fit(X[tr], yb)
        rd = HistGradientBoostingRegressor(max_depth=4, random_state=0).fit(X[tr], yd)
        b = 2 ** rb.predict(X[test]) * canon[test]
        d = 2 ** rd.predict(X[test]) * length[test]
        b[p_ok < 0.5] = np.nan
        pb[scheme], pd_[scheme] = b, d
    return pb, pd_


def evaluate(name, groups, X, bytes_, dec, meta, schemes):
    print(f"\n### {name}\n")
    B, D = bytes_[schemes], dec[schemes]
    preds_b = pd.DataFrame(index=X.index, columns=schemes, dtype=float)
    preds_d = preds_b.copy()
    for g in np.unique(groups):
        test = groups == g
        pb, pdd = fit_predict(X, B, D, meta, ~test, test)
        preds_b.loc[pb.index, schemes] = pb[schemes]
        preds_d.loc[pdd.index, schemes] = pdd[schemes]

    default_b = meta.bytes
    default_d = meta.decode_ns_median
    print(f"{'bandwidth':16s} {'model cost':>11s} {'oracle cost':>12s} {'size-only':>10s} "
          f"{'model bytes':>12s} {'model decode':>13s} {'infeasible picks':>17s}")
    for label, bw in BANDWIDTHS.items():
        true_cost = cost(B, D, bw)
        pred_cost = cost(preds_b, preds_d, bw)
        pick = pred_cost.idxmin(axis=1)
        valid = pick.notna()
        idx = np.arange(len(pick))
        chosen_cost = pd.Series(np.nan, index=X.index)
        chosen_b = chosen_cost.copy()
        chosen_d = chosen_cost.copy()
        cols = [B.columns.get_loc(p) if isinstance(p, str) else -1 for p in pick]
        tc, tb, td = true_cost.to_numpy(), B.to_numpy(), D.to_numpy()
        for i, c in zip(idx, cols):
            if c >= 0:
                chosen_cost.iloc[i], chosen_b.iloc[i], chosen_d.iloc[i] = tc[i, c], tb[i, c], td[i, c]
        infeasible = chosen_cost.isna() & valid
        # Fall back to the production tree when the picked scheme fails.
        dc = cost(default_b, default_d, bw)
        chosen_cost = chosen_cost.fillna(dc)
        chosen_b = chosen_b.fillna(default_b)
        chosen_d = chosen_d.fillna(default_d)
        oracle = true_cost.min(axis=1).fillna(dc)
        size_only = true_cost.to_numpy()[np.arange(len(B)), np.nan_to_num(B.fillna(np.inf).to_numpy().argmin(axis=1))]
        print(f"{label:16s} {chosen_cost.sum() / dc.sum() - 1:>+11.1%} {oracle.sum() / dc.sum() - 1:>+12.1%} "
              f"{np.nansum(size_only) / dc.sum() - 1:>+10.1%} {chosen_b.sum() / default_b.sum() - 1:>+12.1%} "
              f"{chosen_d.sum() / default_d.sum() - 1:>+13.1%} {infeasible.mean():>17.1%}")
    return preds_b, preds_d


def cross_predict(groups, X, bytes_, dec, meta, schemes):
    preds_b = pd.DataFrame(index=X.index, columns=schemes, dtype=float)
    preds_d = preds_b.copy()
    for g in np.unique(groups):
        test = groups == g
        pb, pdd = fit_predict(X, bytes_[schemes], dec[schemes], meta, ~test, test)
        preds_b.loc[pb.index, schemes] = pb[schemes]
        preds_d.loc[pdd.index, schemes] = pdd[schemes]
    return preds_b, preds_d


def evaluate_verify(name, groups, X, bytes_, dec, comp, meta, schemes):
    """The model proposes its top-k alternatives; the real cost decides against production."""
    print(f"\n### propose-and-verify, {name}\n")
    preds_b, preds_d = cross_predict(groups, X, bytes_, dec, meta, schemes)
    B, D, C = bytes_[schemes], dec[schemes], comp[schemes]
    pb_, pd_ = meta.bytes.to_numpy(), meta.decode_ns_median.to_numpy()
    base_comp = meta.compress_ns.sum()
    print(f"{'bandwidth':16s} {'k':>2s} {'gate':>5s} {'cost':>8s} {'oracle':>8s} {'bytes':>8s} {'decode':>8s} "
          f"{'tried':>7s} {'extra compress':>15s}")
    for label, bw in BANDWIDTHS.items():
        true_cost = cost(B, D, bw).to_numpy()
        pred_cost = cost(preds_b, preds_d, bw).to_numpy()
        prod_cost = pb_ / bw + pd_ / 1e9
        oracle = np.minimum(np.nan_to_num(true_cost, nan=np.inf).min(axis=1), prod_cost)
        # The model's estimate of production's own cost, for gating.
        pred_prod = cost(preds_b[["production"]], preds_d[["production"]], bw).to_numpy()[:, 0] \
            if "production" in schemes else prod_cost
        for k in (1,):
            for gate in (0.0, 0.1, 0.2):
                order = np.argsort(np.nan_to_num(pred_cost, nan=np.inf), axis=1)[:, :k]
                chosen = prod_cost.copy()
                chosen_b, chosen_d = pb_.astype(float).copy(), pd_.astype(float).copy()
                tried = np.zeros(len(B), dtype=bool)
                extra = 0.0
                for i in range(len(B)):
                    for j in order[i]:
                        if schemes[j] == "production" or not np.isfinite(pred_cost[i, j]):
                            continue
                        if pred_cost[i, j] > pred_prod[i] * (1 - gate):
                            continue
                        tried[i] = True
                        extra += np.nan_to_num(C.iat[i, j])
                        if np.isfinite(true_cost[i, j]) and true_cost[i, j] < chosen[i]:
                            chosen[i], chosen_b[i], chosen_d[i] = true_cost[i, j], B.iat[i, j], D.iat[i, j]
                print(f"{label:16s} {k:>2d} {gate:>5.0%} {chosen.sum() / prod_cost.sum() - 1:>+8.1%} "
                      f"{oracle.sum() / prod_cost.sum() - 1:>+8.1%} {chosen_b.sum() / pb_.sum() - 1:>+8.1%} "
                      f"{chosen_d.sum() / pd_.sum() - 1:>+8.1%} {tried.mean():>7.0%} {extra / base_comp:>+15.0%}")


def export_ensemble(est) -> dict:
    """A boosted ensemble as plain arrays: go left when x[feature] <= threshold."""
    trees = []
    for (pred,) in est._predictors:
        nodes = pred.nodes
        trees.append({
            "feature": nodes["feature_idx"].astype(int).tolist(),
            "threshold": nodes["num_threshold"].astype(float).tolist(),
            "left": nodes["left"].astype(int).tolist(),
            "right": nodes["right"].astype(int).tolist(),
            "leaf": nodes["is_leaf"].astype(bool).tolist(),
            "value": nodes["value"].astype(float).tolist(),
        })
    return {"baseline": float(np.ravel(est._baseline_prediction)[0]), "trees": trees}


def walk(ens: dict, x: np.ndarray) -> float:
    total = ens["baseline"]
    for t in ens["trees"]:
        node = 0
        while not t["leaf"][node]:
            node = t["left"][node] if x[t["feature"][node]] <= t["threshold"][node] else t["right"][node]
        total += t["value"][node]
    return total


def load_compress(out: str, tag: str, index) -> pd.DataFrame:
    """Compression ns per candidate, aligned with `load`'s index."""
    rows = pd.read_csv(f"{out}/rows-{tag}.csv")
    policies = {"default": "production", "model/runend+sparse": "sizemodel"}
    cand = rows[(rows.ok == 1) & (rows.variant.str.startswith("forced/") | rows.variant.isin(policies))].copy()
    cand["scheme"] = cand.variant.str.removeprefix("forced/").replace(policies)
    return cand.pivot_table(index=KEY, columns="scheme", values="compress_ns", aggfunc="min").reindex(index)


def export_model(X, bytes_, dec, meta, train, comp=None) -> dict:
    """Fits every candidate on the training rows and exports it for the Rust compressor."""
    canon = meta.canonical_bytes.to_numpy()
    length = meta.len.to_numpy()
    out = {"features": list(X.columns), "candidates": {}}
    for scheme in bytes_.columns:
        feasible = bytes_[scheme].notna().to_numpy()
        tr = train & feasible
        if tr.sum() < 20:
            continue
        feas = None
        if not feasible[train].all():
            clf = HistGradientBoostingClassifier(max_depth=4, max_iter=40, learning_rate=0.2, random_state=0).fit(
                X[train], feasible[train])
            if list(clf.classes_) != [False, True]:
                continue
            feas = export_ensemble(clf)
        rb = HistGradientBoostingRegressor(max_depth=4, max_iter=40, learning_rate=0.2, random_state=0).fit(
            X[tr], np.log2(bytes_[scheme].to_numpy()[tr] / canon[tr]))
        rd = HistGradientBoostingRegressor(max_depth=4, max_iter=40, learning_rate=0.2, random_state=0).fit(
            X[tr], np.log2(dec[scheme].to_numpy()[tr] / length[tr]))
        eb, ed = export_ensemble(rb), export_ensemble(rd)
        # Parity: the plain tree walk the Rust side does must reproduce scikit-learn.
        sample = X[tr].to_numpy()[:50]
        assert np.allclose([walk(eb, x) for x in sample], rb.predict(sample), atol=1e-9), scheme
        assert np.allclose([walk(ed, x) for x in sample], rd.predict(sample), atol=1e-9), scheme
        entry = {"feasible": feas, "bytes": eb, "decode": ed}
        if comp is not None and scheme in comp.columns:
            rc = HistGradientBoostingRegressor(max_depth=4, max_iter=40, learning_rate=0.2, random_state=0).fit(
                X[tr], np.log2(comp[scheme].to_numpy()[tr] / length[tr]))
            entry["compress"] = export_ensemble(rc)
        out["candidates"][scheme] = entry
    return out


def main() -> None:
    import json
    from pathlib import Path

    args = sys.argv[1:]
    export_dir = None
    if "--export" in args:
        i = args.index("--export")
        export_dir = Path(args[i + 1])
        del args[i:i + 2]
    out = args[0]
    tag = args[1] if len(args) > 1 else "stock"
    X, bytes_, dec, meta = load(out, tag)
    sources = X.index.get_level_values("source").to_numpy()

    if export_dir is not None:
        export_dir.mkdir(parents=True, exist_ok=True)
        # One model per held-out source (trained without it), plus one on everything.
        for held_out in [*np.unique(sources), "all"]:
            train = sources != held_out
            model = export_model(X, bytes_, dec, meta, train, load_compress(out, tag, X.index))
            (export_dir / f"{held_out}.json").write_text(json.dumps(model))
            print(f"exported {held_out}.json: {len(model['candidates'])} candidates, trained on {train.sum()} chunks")
        return

    rows = pd.read_csv(f"{out}/rows-{tag}.csv")
    policies = {"default": "production", "model/runend+sparse": "sizemodel"}
    cand = rows[(rows.ok == 1) & (rows.variant.str.startswith("forced/") | rows.variant.isin(policies))].copy()
    cand["scheme"] = cand.variant.str.removeprefix("forced/").replace(policies)
    comp = cand.pivot_table(index=KEY, columns="scheme", values="compress_ns", aggfunc="min").reindex(X.index)
    print(f"chunks {len(X)}, features {X.shape[1]}, candidates {list(bytes_.columns)}")
    print("Costs are relative to the production compressor at the same bandwidth (lower is better).")
    columns = pd.Series([f"{s}/{c}" for s, c, _ in X.index]).astype("category").cat.codes.to_numpy()
    schemes = [s for s in bytes_.columns]
    folds = {"5-fold grouped by column": columns % 5, "leave-one-source-out": sources}
    for split, groups in folds.items():
        evaluate_verify(split, groups, X, bytes_, dec, comp, meta, schemes)


if __name__ == "__main__":
    main()
