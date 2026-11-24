# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""The seven report sections. Each takes the scored rows and presets and returns a `Section`."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import polars as pl

from .cost import Presets, fmt_bandwidth, picks, summarise
from .load import FEATURE_TIERS, KEY, PRODUCTION, Dataset
from .report import Code, Section, Table
from .stats import saved_ci
from .style import SEQUENTIAL, color

#: Substring of the root encoding id that shows a forced scheme really is at the root.
ROOT_NEEDLE = {"bitpacking": "bitpacked", "for": ".for"}

TIER_NAMES = {
    0: "context",
    1: "compressor stats",
    2: "distinct",
    3: "one extra pass",
    4: "derived",
    5: "sample",
}


# ---------------------------------------------------------------------------------------------
# Shared helpers


def cost_matrix(scored: pl.DataFrame, bandwidth: float) -> tuple[pl.DataFrame, list[str], np.ndarray]:
    """Sites × variants matrix of cost (inf where a candidate is missing), plus compress ns."""
    c = scored.with_columns(cost=pl.col("bytes") / bandwidth + pl.col("decode_ns") / 1e9)
    wide = c.pivot(on="variant", index=KEY, values="cost", aggregate_function="min").sort(KEY)
    variants = [v for v in wide.columns if v not in KEY]
    variants = [PRODUCTION] + sorted(v for v in variants if v != PRODUCTION)
    m = wide.select(variants).to_numpy().astype(float)
    m[np.isnan(m)] = np.inf
    return wide.select(KEY), variants, m


def compress_matrix(scored: pl.DataFrame, keys: pl.DataFrame, variants: list[str]) -> np.ndarray:
    wide = scored.pivot(on="variant", index=KEY, values="compress_ns", aggregate_function="min")
    wide = keys.join(wide, on=KEY, how="left")
    m = wide.select([v if v in wide.columns else pl.lit(None).alias(v) for v in variants]).to_numpy()
    m = m.astype(float)
    return np.nan_to_num(m, nan=0.0)


def folds(keys: pl.DataFrame, k: int = 5) -> tuple[np.ndarray, str]:
    """Group labels for held-out scoring: by source when there are at least 3, else by column."""
    sources = keys["source"].to_numpy()
    if len(np.unique(sources)) >= 3:
        return sources, "leave-one-source-out"
    cols = (keys["source"] + "/" + keys["column"]).to_numpy()
    uniq = np.unique(cols)
    rng = np.random.default_rng(0)
    assign = dict(zip(uniq, rng.permutation(len(uniq)) % k))
    note = f"{k}-fold grouped by column (fewer than 3 sources, so leave-one-source-out is not possible)"
    return np.array([assign[c] for c in cols]), note


def greedy_set(m: np.ndarray, variants: list[str], k: int, rows: np.ndarray | None = None) -> list[int]:
    """Greedily add the candidate that most lowers the total of per-site minimum cost."""
    sub = m if rows is None else m[rows]
    chosen = [0]
    cur = sub[:, 0].copy()
    for _ in range(k):
        totals = np.minimum(cur[:, None], sub).sum(axis=0)
        totals[chosen] = np.inf
        j = int(np.argmin(totals))
        if not np.isfinite(totals[j]) or totals[j] >= cur.sum() * (1 - 1e-6):
            break
        chosen.append(j)
        cur = np.minimum(cur, sub[:, j])
    return chosen


def heat(z, x, y, title, fmt=".1%", zmin=0.0, zmax=None, colorbar="Cost saved") -> go.Figure:
    z = np.asarray(z, dtype=float)
    text = [[("" if np.isnan(v) else format(v, fmt)) for v in row] for row in z]
    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=x,
            y=y,
            text=text,
            texttemplate="%{text}",
            colorscale=SEQUENTIAL,
            zmin=zmin,
            zmax=zmax if zmax is not None else max(float(np.nanmax(z)) if z.size else 1.0, 1e-9),
            xgap=2,
            ygap=2,
            colorbar={"title": colorbar, "tickformat": fmt[1:] if fmt.startswith(".") else None},
            hovertemplate="%{y} · %{x}: %{text}<extra></extra>",
        )
    )
    fig.update_layout(title=title, height=max(260, 46 * len(y) + 120))
    fig.update_xaxes(showgrid=False, side="bottom")
    fig.update_yaxes(showgrid=False, autorange="reversed")
    return fig


# ---------------------------------------------------------------------------------------------
# 1. Data health


def health(ds: Dataset, scored: pl.DataFrame) -> Section:
    r = ds.rows
    by = ["source", "dtype"]
    t = (
        r.group_by(by)
        .agg(
            pl.col("chunk").n_unique().alias("chunks"),
            pl.struct(KEY).n_unique().alias("sites"),
            pl.col("variant").n_unique().alias("candidates"),
            pl.col("canonical_bytes").filter(pl.col("variant") == PRODUCTION).sum().alias("canonical MB"),
            (~pl.col("ok")).mean().alias("failed"),
            (pl.col("ok") & pl.col("decode_ns").is_null()).mean().alias("untimed"),
        )
        .sort(by)
    )
    scored_sites = (
        scored.select(list(dict.fromkeys(by + KEY))).unique().group_by(by).agg(pl.len().alias("scored sites"))
    )
    t = t.join(scored_sites, on=by, how="left").with_columns(pl.col("scored sites").fill_null(0))
    m = ds.manifest
    lead = (
        f"{r['chunk'].n_unique():,} chunks and {r.height:,} candidate rows. "
        f"{scored.select(KEY).unique().height:,} sites have production and at least one other candidate "
        f"compressed and timed, and only those are scored below."
    )
    if m:
        lead += (
            f" The manifest reports {m.get('chunks', '?')} complete chunks, {m.get('incomplete', '?')} incomplete, "
            f"{m.get('timed_chunks', '?')} timed, with up to {m.get('max_runs', '?')} timing runs pooled per chunk."
        )
    blocks: list = [
        Table(
            t,
            "Coverage by source and dtype",
            {
                "canonical MB": "mb",
                "failed": "pct",
                "untimed": "pct",
                "chunks": "int",
                "sites": "int",
                "scored sites": "int",
            },
        )
    ]
    srcs = t["source"].unique().sort().to_list()
    dts = t["dtype"].unique().sort().to_list()
    if len(srcs) * len(dts) > 1:
        z = [
            [
                (t.filter((pl.col("source") == s) & (pl.col("dtype") == d))["chunks"].sum() or np.nan)
                for d in dts
            ]
            for s in srcs
        ]
        blocks.append(heat(z, dts, srcs, "Chunks per source and dtype", fmt=",.0f", colorbar="Chunks"))

    noise = scored.filter(pl.col("decode_ns_min").is_not_null() & (pl.col("decode_ns") > 0)).with_columns(
        noise=1 - pl.col("decode_ns_min") / pl.col("decode_ns")
    )
    if noise.height:
        nt = (
            noise.group_by("dtype")
            .agg(
                pl.col("noise").median().alias("median"),
                pl.col("noise").quantile(0.9).alias("p90"),
                (pl.col("noise") > 0.1).mean().alias("over 10%"),
            )
            .sort("dtype")
        )
        blocks.append(
            Table(
                nt,
                "Decode timing spread, 1 − min / median, per candidate",
                {"median": "pct", "p90": "pct", "over 10%": "pct"},
            )
        )
    forced = (
        r.filter(pl.col("ok") & pl.col("variant").str.starts_with("forced/"))
        .with_columns(scheme=pl.col("variant").str.strip_prefix("forced/"))
        .with_columns(needle=pl.col("scheme").replace(ROOT_NEEDLE))
        .with_columns(at_root=pl.col("root").fill_null("").str.contains(pl.col("needle"), literal=True))
        .group_by("variant")
        .agg(
            pl.len().alias("compressed"),
            pl.col("at_root").mean().alias("scheme at root"),
            pl.col("root").filter(~pl.col("at_root")).mode().first().alias("usual root otherwise"),
        )
        .sort("scheme at root")
    )
    if forced.height:
        blocks.append(
            Table(
                forced,
                "Forced candidates: how often the forced scheme is really the root",
                {"compressed": "int", "scheme at root": "pct"},
            )
        )
        broken = forced.filter(pl.col("scheme at root") < 0.05)["variant"].to_list()
        if broken:
            blocks.append(
                f"Warning: {', '.join(broken)} almost never put their scheme at the root, so they measure a "
                "fallback rather than the scheme. Headroom from those candidates is understated."
            )
    errs = (
        r.filter(~pl.col("ok"))
        .with_columns(pl.col("error").fill_null("").str.slice(0, 90))
        .group_by(["variant", "error"])
        .agg(pl.len().alias("rows"))
        .sort("rows", descending=True)
        .head(10)
    )
    if errs.height:
        blocks.append(Table(errs, "Most common candidate failures", {"rows": "int"}))
    return Section("health", "Data health", lead, blocks)


# ---------------------------------------------------------------------------------------------
# 2. Headroom


def headroom(scored: pl.DataFrame, presets: Presets) -> Section:
    rows = []
    groups = scored.select(["source", "dtype"]).unique().sort(["source", "dtype"])
    gl = [f"{s} · {d}" for s, d in groups.iter_rows()]
    z = np.full((len(gl), len(presets.presets)), np.nan)
    for j, p in enumerate(presets):
        pk = picks(scored, p.bandwidth)
        s = summarise(pk, []).row(0, named=True)
        lo, hi = saved_ci(pk)
        rows.append(
            {
                "preset": p.name,
                "bandwidth": p.bw_label,
                "saved": s["saved"],
                "90% CI": f"{lo:.1%} – {hi:.1%}" if not np.isnan(lo) else "",
                "saved per chunk": s["saved_per_chunk"],
                "prod optimal": s["prod_optimal"],
                "ratio prod": s["prod_ratio"],
                "ratio oracle": s["best_ratio"],
                "decode GB/s prod": s["prod_decode_gbps"],
                "decode GB/s oracle": s["best_decode_gbps"],
                "compress time": s["compress_x"],
            }
        )
        g = summarise(pk, ["source", "dtype"])
        for i, (src, dt) in enumerate(groups.iter_rows()):
            v = g.filter((pl.col("source") == src) & (pl.col("dtype") == dt))["saved"]
            z[i, j] = v[0] if len(v) else np.nan
    t = pl.DataFrame(rows)
    main = t.filter(pl.col("preset") == presets.primary).row(0, named=True)
    lead = (
        f"At {presets.primary} ({main['bandwidth']}) the per-chunk oracle is {main['saved']:.1%} cheaper than "
        f"production, byte-weighted; production already picks the cheapest candidate at "
        f"{main['prod optimal']:.0%} of sites. Cost is bytes / bandwidth + decode time. "
        f"Compress time is the oracle's chosen encoding relative to production's, not scored."
    )
    blocks: list = [
        Table(
            t,
            "Oracle vs production per preset",
            {
                "saved": "pct",
                "saved per chunk": "pct",
                "prod optimal": "pct",
                "ratio prod": "x",
                "ratio oracle": "x",
                "decode GB/s prod": "gbps",
                "decode GB/s oracle": "gbps",
                "compress time": "x",
            },
        ),
        heat(
            z, [f"{p.name}<br>{p.bw_label}" for p in presets], gl, "Cost saved by the oracle, source × dtype"
        ),
    ]
    if presets.sweep:
        fig = go.Figure()
        series = [("all", scored)] + [
            (d, scored.filter(pl.col("dtype") == d)) for d in scored["dtype"].unique().sort().to_list()
        ]
        if len(series) == 2:
            series = series[:1]
        for i, (name, sub) in enumerate(series[:8]):
            ys = [summarise(picks(sub, bw), [])["saved"][0] for bw in presets.sweep]
            fig.add_trace(
                go.Scatter(
                    x=list(presets.sweep),
                    y=ys,
                    name=name,
                    mode="lines+markers",
                    line={"width": 2, "color": color(i)},
                    marker={"size": 8},
                    customdata=[fmt_bandwidth(b) for b in presets.sweep],
                    hovertemplate=name + " · %{customdata}: %{y:.1%}<extra></extra>",
                )
            )
        fig.update_layout(title="Oracle saving across bandwidths", height=380, showlegend=len(series) > 1)
        fig.update_xaxes(
            type="log",
            title="Bandwidth (bytes/s)",
            tickvals=list(presets.sweep),
            ticktext=[fmt_bandwidth(b) for b in presets.sweep],
        )
        fig.update_yaxes(title="Cost saved vs production", tickformat=".0%", rangemode="tozero")
        blocks.append(fig)
    return Section("headroom", "Headroom", lead, blocks)


# ---------------------------------------------------------------------------------------------
# 3. Regret


def regret(scored: pl.DataFrame, presets: Presets) -> Section:
    p = presets.main
    pk = picks(scored, p.bandwidth)
    total = pk["regret"].sum()
    fig = go.Figure()
    dts = pk["dtype"].unique().sort().to_list()
    hi = float(pk["regret_frac"].quantile(0.99) or 0.0)
    for i, d in enumerate(dts[:8]):
        v = np.sort(pk.filter(pl.col("dtype") == d)["regret_frac"].to_numpy())
        y = np.arange(1, len(v) + 1) / len(v)
        fig.add_trace(
            go.Scatter(
                x=v,
                y=y,
                name=d,
                mode="lines",
                line={"width": 2, "color": color(i), "shape": "hv"},
                hovertemplate=d + ": %{y:.0%} of sites have regret ≤ %{x:.1%}<extra></extra>",
            )
        )
    fig.update_layout(title=f"Regret of production per site at {p.name}", height=360, showlegend=len(dts) > 1)
    fig.update_xaxes(title="Production cost / oracle cost − 1", tickformat=".0%", range=[0, max(hi, 0.01)])
    fig.update_yaxes(title="Share of sites", tickformat=".0%", range=[0, 1.02])

    cols = (
        pk.group_by(["source", "column", "dtype"])
        .agg(
            pl.len().alias("sites"),
            pl.col("regret").sum().alias("regret_s"),
            (1 - pl.col("best_cost").sum() / pl.col("prod_cost").sum()).alias("saved"),
            pl.col("best_variant").filter(~pl.col("optimal")).mode().first().alias("usual oracle pick"),
            pl.col("prod_root").mode().first().alias("usual production root"),
        )
        .with_columns(share=pl.col("regret_s") / max(total, 1e-30))
        .sort("regret_s", descending=True)
        .head(20)
        .drop("regret_s")
    )
    top_share = cols["share"].sum()
    lead = (
        f"Regret is how much more production costs than the oracle at a site, at {p.name} ({p.bw_label}). "
        f"The 20 columns below hold {top_share:.0%} of all regret."
    )
    blocks: list = [
        fig,
        Table(cols, "Columns that lose the most", {"saved": "pct", "share": "pct", "sites": "int"}),
    ]
    if pk["depth"].n_unique() > 1:
        blocks.append(
            Table(
                summarise(pk, ["depth"]),
                "By cascade depth",
                {"saved": "pct", "saved_per_chunk": "pct", "prod_optimal": "pct"},
            )
        )
    return Section("regret", "Regret of today's compressor", lead, blocks)


# ---------------------------------------------------------------------------------------------
# 4. Why it misses


def _estimates(f: pl.DataFrame) -> dict[str, pl.Expr]:
    n = pl.col("len") * (1 - pl.col("null_frac"))
    pos = pl.max_horizontal(pl.col("len").log(2), pl.lit(1.0))
    bp = pl.when(pl.col("bits_bp") >= 0).then(pl.col("bits_bp")).otherwise(pl.col("ptype_bits"))
    runs = pl.max_horizontal(n / pl.max_horizontal(pl.col("avg_run"), pl.lit(1.0)), pl.lit(1.0))
    return {
        "bitpacking": n * bp,
        "for": n * pl.col("bits_for"),
        "runend": runs * (pl.col("bits_for") + pos),
        "sparse": pl.max_horizontal(n * (1 - pl.col("top1_frac")), pl.lit(1.0)) * (pl.col("bits_for") + pos),
        "dict": pl.col("distinct") * pl.col("ptype_bits")
        + n * pl.max_horizontal(pl.col("distinct"), pl.lit(2.0)).log(2),
    }


def misses(ds: Dataset, scored: pl.DataFrame, presets: Presets) -> Section:
    p = presets.main
    pk = picks(scored, p.bandwidth).filter(~pl.col("optimal"))
    blocks: list = []
    if pk.height == 0:
        return Section("misses", "Why it misses", f"Production is optimal at every site at {p.name}.", [])
    pk = pk.with_columns(pl.col("prod_root").fill_null("?"), pl.col("best_root").fill_null("?"))
    conf = pk.group_by(["prod_root", "best_root"]).agg(pl.col("regret").sum())
    pr = (
        conf.group_by("prod_root")
        .agg(pl.col("regret").sum())
        .sort("regret", descending=True)["prod_root"]
        .to_list()
    )
    br = (
        conf.group_by("best_root")
        .agg(pl.col("regret").sum())
        .sort("regret", descending=True)["best_root"]
        .to_list()
    )
    tot = conf["regret"].sum()
    z = np.full((len(pr), len(br)), np.nan)
    for a, b, v in conf.iter_rows():
        z[pr.index(a), br.index(b)] = v / tot
    blocks.append(
        heat(
            z,
            br,
            pr,
            f"Share of regret: production's root (rows) vs the oracle's root (columns), {p.name}",
            colorbar="Share",
        )
    )
    bv = (
        pk.group_by("best_variant")
        .agg(
            pl.len().alias("sites"),
            (pl.col("regret").sum() / tot).alias("share of regret"),
            pl.col("regret_frac").median().alias("median regret"),
        )
        .sort("share of regret", descending=True)
    )
    blocks.append(
        Table(
            bv,
            "Which candidate wins where production loses",
            {"share of regret": "pct", "median regret": "pct", "sites": "int"},
        )
    )

    need = {"len", "null_frac", "bits_bp", "ptype_bits", "bits_for", "avg_run", "top1_frac", "distinct"}
    if need <= set(ds.features.columns):
        est = ds.features.select(KEY + [e.alias(k) for k, e in _estimates(ds.features).items()])
        rows, fig = [], go.Figure()
        for i, (scheme, _) in enumerate(_estimates(ds.features).items()):
            # Only chunks where the forced scheme really is the root; otherwise it fell back.
            needle = ROOT_NEEDLE.get(scheme, scheme)
            actual = scored.filter(
                (pl.col("variant") == f"forced/{scheme}") & pl.col("root").str.contains(needle, literal=True)
            ).select(KEY + ["bytes"])
            j = actual.join(est.select(KEY + [scheme]), on=KEY).filter(pl.col(scheme) > 0)
            if j.height < 3:
                continue
            ratio = (j["bytes"] * 8 / j[scheme]).to_numpy()
            corr = j.select(pl.corr(pl.col("bytes"), pl.col(scheme), method="spearman")).item()
            q1, q2, q3 = np.quantile(ratio, [0.25, 0.5, 0.75])
            rows.append(
                {
                    "scheme": scheme,
                    "actual / estimate (median)": q2,
                    "IQR low": q1,
                    "IQR high": q3,
                    "rank corr": corr,
                    "chunks": j.height,
                }
            )
            fig.add_trace(
                go.Scatter(
                    x=j[scheme] / 8,
                    y=j["bytes"],
                    mode="markers",
                    name=scheme,
                    marker={
                        "size": 8,
                        "color": color(i),
                        "opacity": 0.7,
                        "line": {"width": 1, "color": "white"},
                    },
                    hovertemplate=scheme + ": estimate %{x:,.0f} B, actual %{y:,.0f} B<extra></extra>",
                )
            )
        if rows:
            lo = min(min(t.x) for t in fig.data)
            hi = max(max(t.x) for t in fig.data)
            fig.add_trace(
                go.Scatter(
                    x=[lo, hi],
                    y=[lo, hi],
                    mode="lines",
                    name="estimate = actual",
                    line={"color": "#9a9893", "width": 1, "dash": "dot"},
                    hoverinfo="skip",
                )
            )
            fig.update_layout(title="Closed-form size estimate vs size reached, forced root", height=420)
            fig.update_xaxes(type="log", title="Estimated bytes")
            fig.update_yaxes(type="log", title="Actual bytes")
            blocks.append(
                Table(
                    pl.DataFrame(rows),
                    "Calibration of closed-form estimates",
                    {
                        "actual / estimate (median)": "f2",
                        "IQR low": "f2",
                        "IQR high": "f2",
                        "rank corr": "f2",
                        "chunks": "int",
                    },
                )
            )
            blocks.append(fig)
    lead = (
        f"Where production is not optimal at {p.name}, which root it picked and which the oracle picked, "
        f"weighted by regret. A large off-diagonal cell is one heuristic to fix."
    )
    return Section("misses", "Why it misses", lead, blocks)


# ---------------------------------------------------------------------------------------------
# 5. Cheap policies: fixed trial sets


def trial_sets(scored: pl.DataFrame, presets: Presets, max_k: int = 5) -> tuple[Section, dict]:
    rows, fig, best_sets = [], go.Figure(), {}
    for pi, p in enumerate(presets):
        keys, variants, m = cost_matrix(scored, p.bandwidth)
        cm = compress_matrix(scored, keys, variants)
        base, base_c = m[:, 0].sum(), cm[:, 0].sum()
        groups, how = folds(keys)
        order = greedy_set(m, variants, max_k)
        xs, ys = [0], [0.0]
        for k in range(1, len(order)):
            sel = order[: k + 1]
            in_sample = 1 - np.min(m[:, sel], axis=1).sum() / base
            held = 0.0
            for g in np.unique(groups):
                te = groups == g
                s = greedy_set(m, variants, k, rows=~te)
                held += np.min(m[te][:, s], axis=1).sum()
            rows.append(
                {
                    "preset": p.name,
                    "trials": k,
                    "set (added in order)": ", ".join(variants[j] for j in sel[1:]),
                    "saved": in_sample,
                    "saved held out": 1 - held / base,
                    "compress time": cm[:, sel].sum() / base_c if base_c else float("nan"),
                }
            )
            xs.append(k)
            ys.append(in_sample)
        best_sets[p.name] = [variants[j] for j in order[: min(3, len(order))]]
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                name=p.name,
                mode="lines+markers",
                line={"width": 2, "color": color(pi)},
                marker={"size": 8},
                hovertemplate=p.name + ": %{x} trials save %{y:.1%}<extra></extra>",
            )
        )
    fig.update_layout(title="Cost saved by production plus k fixed trials, keeping the cheapest", height=380)
    fig.update_xaxes(title="Extra candidates tried (k)", dtick=1)
    fig.update_yaxes(title="Cost saved vs production", tickformat=".0%", rangemode="tozero")
    t = pl.DataFrame(rows) if rows else pl.DataFrame({"preset": []})
    lead = (
        "No model: compress with production plus a fixed set of candidates and keep the cheapest. Sets are "
        f"grown greedily. Held-out figures choose the set without the scored group ({how}). Compress time "
        "counts every trial, relative to production alone."
    )
    sec = Section(
        "trials",
        "Cheap policies: fixed trial sets",
        lead,
        [fig, Table(t, "Greedy trial sets", {"saved": "pct", "saved held out": "pct", "compress time": "x"})],
    )
    return sec, best_sets


# ---------------------------------------------------------------------------------------------
# 6. Features


def _fit_predict(X, y, w, groups, tr_fn):
    pred = np.empty(len(y), dtype=object)
    for g in np.unique(groups):
        te = groups == g
        if len(np.unique(y[~te])) < 2:
            pred[te] = y[~te][0] if (~te).any() else PRODUCTION
            continue
        model = tr_fn().fit(X[~te], y[~te], sample_weight=w[~te])
        pred[te] = model.predict(X[te])
    return pred


def features(ds: Dataset, scored: pl.DataFrame, presets: Presets) -> Section:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.tree import DecisionTreeClassifier, export_text

    p = presets.main
    if not ds.feature_names:
        return Section("features", "Features", "The dataset has no features table.", [])
    keys, variants, m = cost_matrix(scored, p.bandwidth)
    f = keys.join(ds.features, on=KEY, how="left")
    names = [n for n in ds.feature_names if f[n].null_count() < f.height]
    X = f.select(names).to_numpy().astype(float)
    X = np.nan_to_num(X, nan=-1.0)
    label_idx = np.argmin(m, axis=1)
    y = np.array([variants[i] for i in label_idx], dtype=object)
    w = m[:, 0] / m[:, 0].mean()
    groups, how = folds(keys)
    col = {v: i for i, v in enumerate(variants)}
    best, prod = m.min(axis=1).sum(), m[:, 0].sum()

    def score(pred) -> float:
        idx = np.array([col[v] for v in pred])
        c = m[np.arange(len(idx)), idx]
        c = np.where(np.isfinite(c), c, m[:, 0])
        return c.sum() / best - 1

    def hgb():
        return HistGradientBoostingClassifier(max_depth=4, max_iter=150, learning_rate=0.1, random_state=0)

    tiers = sorted({FEATURE_TIERS.get(n, 5) for n in names})
    curve = []
    for t in tiers:
        use = [i for i, n in enumerate(names) if FEATURE_TIERS.get(n, 5) <= t]
        pred = _fit_predict(X[:, use], y, w, groups, hgb)
        curve.append(
            {
                "tiers": f"0–{t}" if t else "0",
                "adds": TIER_NAMES.get(t, str(t)),
                "features": len(use),
                "regret left": score(pred),
                "saved vs production": 1 - ((score(pred) + 1) * best) / prod,
            }
        )
    prod_regret = prod / best - 1
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[c["tiers"] for c in curve],
            y=[c["regret left"] for c in curve],
            mode="lines+markers",
            name="model on these tiers",
            line={"width": 2, "color": color(0)},
            marker={"size": 8},
            hovertemplate="tiers %{x}: regret left %{y:.1%}<extra></extra>",
        )
    )
    fig.add_hline(
        y=prod_regret,
        line={"color": color(1), "width": 2, "dash": "dash"},
        annotation_text=f"production {prod_regret:.1%}",
        annotation_position="top left",
    )
    fig.update_layout(
        title=f"Regret left by a held-out model, by feature tiers allowed ({p.name})",
        height=380,
        showlegend=False,
    )
    fig.update_xaxes(title="Feature tiers allowed (cheapest first)", type="category")
    fig.update_yaxes(title="Cost / oracle cost − 1", tickformat=".1%", rangemode="tozero")

    # Permutation importance in cost terms, on held-out predictions with all features.
    base_pred = _fit_predict(X, y, w, groups, hgb)
    base_score = score(base_pred)
    rng = np.random.default_rng(0)
    imp = []
    models = {}
    for g in np.unique(groups):
        te = groups == g
        if len(np.unique(y[~te])) >= 2:
            models[g] = hgb().fit(X[~te], y[~te], sample_weight=w[~te])
    for i, n in enumerate(names):
        Xp = X.copy()
        Xp[:, i] = rng.permutation(Xp[:, i])
        pred = base_pred.copy()
        for g, model in models.items():
            te = groups == g
            pred[te] = model.predict(Xp[te])
        imp.append(
            {
                "feature": n,
                "tier": FEATURE_TIERS.get(n, 5),
                "regret added when shuffled": score(pred) - base_score,
            }
        )
    imp_t = pl.DataFrame(imp).sort("regret added when shuffled", descending=True).head(15)

    tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=max(5, len(y) // 100), random_state=0)
    tree.fit(X, y, sample_weight=w)
    text = export_text(tree, feature_names=names, decimals=2)

    labels = (
        pl.DataFrame({"oracle pick": y})
        .group_by("oracle pick")
        .agg(pl.len().alias("sites"))
        .sort("sites", descending=True)
    )
    lead = (
        f"Can cheap features predict the oracle's pick at {p.name}? Models are gradient-boosted trees, weighted by "
        f"production cost and scored on held-out groups ({how}). With all tiers the model leaves "
        f"{base_score:.1%} regret against production's {prod_regret:.1%}."
    )
    return Section(
        "features",
        "Features",
        lead,
        [
            fig,
            Table(
                pl.DataFrame(curve),
                "Regret left per tier budget",
                {"regret left": "pct", "saved vs production": "pct", "features": "int"},
            ),
            Table(imp_t, "Permutation importance, held out", {"regret added when shuffled": "spct"}),
            Table(labels, "Oracle labels", {"sites": "int"}),
            "A depth-3 tree on all features and all sites, to read its splits:",
            Code(text),
        ],
    )


# ---------------------------------------------------------------------------------------------
# 7. Frontier


def frontier(scored: pl.DataFrame, presets: Presets, best_sets: dict) -> Section:
    rows = []
    prod_pt = None
    for p in presets:
        pk = picks(scored, p.bandwidth)
        s = summarise(pk, []).row(0, named=True)
        prod_pt = (s["prod_ratio"], s["prod_decode_gbps"])
        rows.append(
            {
                "preset": p.name,
                "policy": "oracle",
                "ratio": s["best_ratio"],
                "decode GB/s": s["best_decode_gbps"],
                "saved": s["saved"],
                "compress time": s["compress_x"],
            }
        )
        trial = best_sets.get(p.name, [PRODUCTION])
        if len(trial) > 1:
            tk = picks(scored, p.bandwidth, allowed=trial)
            ts = summarise(tk, []).row(0, named=True)
            rows.append(
                {
                    "preset": p.name,
                    "policy": "production + up to 2 fixed trials",
                    "ratio": ts["best_ratio"],
                    "decode GB/s": ts["best_decode_gbps"],
                    "saved": ts["saved"],
                    "compress time": ts["compress_x"],
                }
            )
    t = pl.DataFrame(rows)
    fig = go.Figure()
    for i, pol in enumerate(t["policy"].unique(maintain_order=True).to_list()):
        sub = t.filter(pl.col("policy") == pol)
        fig.add_trace(
            go.Scatter(
                x=sub["ratio"],
                y=sub["decode GB/s"],
                mode="lines+markers+text",
                text=sub["preset"],
                textposition="top center",
                name=pol,
                line={"width": 2, "color": color(i)},
                marker={"size": 9},
                textfont={"color": "#52514e", "size": 11},
                hovertemplate=pol + " at %{text}: ratio %{x:.2f}×, decode %{y:.1f} GB/s<extra></extra>",
            )
        )
    if prod_pt:
        fig.add_trace(
            go.Scatter(
                x=[prod_pt[0]],
                y=[prod_pt[1]],
                mode="markers+text",
                text=["production"],
                textposition="bottom center",
                name="production",
                marker={"size": 12, "color": "#0b0b0b", "symbol": "diamond"},
                hovertemplate="production: ratio %{x:.2f}×, decode %{y:.1f} GB/s<extra></extra>",
            )
        )
    fig.update_layout(title="Compression ratio vs decode throughput, per preset", height=440)
    fig.update_xaxes(title="Compression ratio (canonical / compressed)")
    fig.update_yaxes(title="Decode throughput (GB/s of canonical data)", rangemode="tozero")
    lead = (
        "Each preset moves the oracle along the ratio–decode trade-off: low bandwidth buys ratio, high bandwidth "
        "buys decode speed. The trial-set policy uses the first trials from section 5."
    )
    if prod_pt:
        lead += f" Production sits at {prod_pt[0]:.2f}× and {prod_pt[1]:.1f} GB/s for every preset."
    return Section(
        "frontier",
        "Frontier",
        lead,
        [
            fig,
            Table(t, "Points", {"ratio": "x", "decode GB/s": "gbps", "saved": "pct", "compress time": "x"}),
        ],
    )
