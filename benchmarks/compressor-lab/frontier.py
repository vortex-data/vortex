# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""The ratio / decode-speed frontier: production vs the model-driven compressor vs the best root.

Cost per chunk = bytes / bandwidth + decode time. Low bandwidth weighs compression ratio, high
bandwidth weighs decode speed. Production and the model-driven compressor come from `vx-lab eval`
(measured in one process); the best root comes from the dataset's candidate measurements.

Usage: uv run --no-project --with pandas python frontier.py <eval.csv> <dataset-dir> [exclude ...]
"""

import re
import sys

import pandas as pd

KEY = ["source", "column", "chunk"]


def bandwidth(label: str) -> float:
    m = re.fullmatch(r"([\d.]+)(MB|GB)ps", label)
    if not m:
        raise ValueError(label)
    return float(m.group(1)) * (1e6 if m.group(2) == "MB" else 1e9)


def main() -> None:
    eval_csv, dataset = sys.argv[1], sys.argv[2]
    exclude = {f"forced/{e}" for e in sys.argv[3:]}
    ev = pd.read_csv(eval_csv, dtype={"chunk": str})
    rows = pd.read_csv(f"{dataset}/rows-stock.csv", dtype={"chunk": str})
    cand = rows[(rows.ok == 1) & ~rows.variant.isin(exclude)]

    prod = ev[ev.variant == "production"].set_index(KEY)
    canon = prod.canonical_bytes.sum()
    held = (prod.model == "held-out").mean()
    print(f"{len(prod)} chunks, {canon / 1e6:.0f} MB canonical; {held:.0%} scored by a model that never saw "
          f"their source")
    print(f"production: ratio {canon / prod.bytes.sum():.2f}x, decode {canon / prod.decode_ns.sum():.2f} GB/s\n")
    print("cost = bytes / bandwidth + decode time (compression time not counted)\n")
    print(f"{'bandwidth':>10}  {'model cost':>10} {'best root':>10}  {'ratio':>15} {'decode GB/s':>15}  "
          f"{'tried':>5} {'kept':>5}  {'compress':>8}  worst source")

    for variant in [v for v in ev.variant.unique() if v.startswith("model@")]:
        label = variant.split("@", 1)[1]
        bw = bandwidth(label)
        model = ev[ev.variant == variant].set_index(KEY)

        def cost(df):
            return df.bytes / bw + df.decode_ns / 1e9

        p, m = cost(prod), cost(model.loc[prod.index])
        # Best root from the dataset, relative to the dataset's own production measurement.
        c = cand.assign(cost=cand.bytes / bw + cand.decode_ns_median / 1e9)
        best = c.groupby(KEY).cost.min()
        dprod = c[c.variant == "default"].set_index(KEY).cost
        oracle = best.loc[dprod.index].sum() / dprod.sum() - 1
        per_src = (m.groupby(level="source").sum() / p.groupby(level="source").sum() - 1)
        worst = per_src.idxmax()
        print(f"{label:>10}  {m.sum() / p.sum() - 1:>+10.1%} {oracle:>+10.1%}  "
              f"{canon / prod.bytes.sum():>6.2f}→{canon / model.bytes.sum():<6.2f}x "
              f"{canon / prod.decode_ns.sum():>6.2f}→{canon / model.decode_ns.sum():<6.2f}  "
              f"{model.tried.mean():>5.0%} {model.kept.ne('production').mean():>5.0%}  "
              f"{model.compress_ns.sum() / prod.compress_ns.sum():>7.2f}x  {worst} {per_src[worst]:+.1%}")
        kept = model.kept[model.kept != "production"].value_counts().head(4)
        print(f"{'':>10}  kept: {', '.join(f'{k} {v}' for k, v in kept.items())}")


if __name__ == "__main__":
    main()
