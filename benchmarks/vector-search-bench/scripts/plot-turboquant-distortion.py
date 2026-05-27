# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "matplotlib",
# ]
# ///

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Sweep bits-vs-distortion for TurboQuant and plot the curves.

Calls `vector-search-bench distortion` for each (dataset, bits) combination, parses the
table from stdout, and plots reconstruction NMSE and squared cosine-error curves with
mean/median/max shown on a log-scaled y-axis.

Each `--dataset` value may optionally pin a train layout with a colon, e.g.
`--dataset cohere-small-100k:single`, for datasets that host more than one layout.

Usage:
  uv run benchmarks/vector-search-bench/scripts/plot-turboquant-distortion.py \\
      --dataset sift-small-500k
  uv run benchmarks/vector-search-bench/scripts/plot-turboquant-distortion.py \\
      --dataset sift-small-500k --dataset glove-small-100k --samples 8192
  uv run benchmarks/vector-search-bench/scripts/plot-turboquant-distortion.py \\
      --dataset cohere-small-100k:single --bits 1 2 3 4 5 6 7 8 \\
      --output /tmp/distortion.png
"""

import argparse
import math
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, NullLocator

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BINARY = REPO_ROOT / "target" / "release" / "vector-search-bench"

METRIC_NAMES = [
    "reconstruction NMSE mean",
    "reconstruction NMSE median",
    "reconstruction NMSE max",
    "decoded cosine sqerr mean",
    "decoded cosine sqerr median",
    "decoded cosine sqerr max",
]


@dataclass(frozen=True)
class DatasetTarget:
    """One dataset to sweep, with the layout the bench should use for it."""

    name: str
    layout: str | None  # `None` means let the bench auto-pick.


@dataclass
class Run:
    target: DatasetTarget
    dim: int
    bits: int
    values: dict[str, float]

    @property
    def dataset(self) -> str:
        return self.target.name


DIM_RE = re.compile(r"dim=(\d+)")


def parse_dataset_arg(spec: str, default_layout: str | None) -> DatasetTarget:
    """Split a `name[:layout]` CLI value. `default_layout` fills in when no `:` is given."""
    if ":" in spec:
        name, layout = spec.split(":", 1)
        return DatasetTarget(name=name, layout=layout or None)
    return DatasetTarget(name=spec, layout=default_layout)


def parse_dim(stdout: str) -> int:
    """Pull `dim=N` out of the `## ...` header line."""
    match = DIM_RE.search(stdout)
    if not match:
        raise RuntimeError(f"could not find dim=N in header:\n{stdout}")
    return int(match.group(1))


def parse_table(stdout: str) -> dict[str, float]:
    """Pull `metric -> value` rows out of the tabled stdout."""
    row_re = re.compile(r"│\s*(.+?)\s*│\s*([^│]+?)\s*│")
    values: dict[str, float] = {}
    for line in stdout.splitlines():
        match = row_re.match(line)
        if not match:
            continue
        metric, value = match.group(1).strip(), match.group(2).strip()
        if metric in METRIC_NAMES:
            values[metric] = float(value)
    missing = [m for m in METRIC_NAMES if m not in values]
    if missing:
        raise RuntimeError(f"could not parse metrics {missing} from:\n{stdout}")
    return values


def run_one(
    binary: Path,
    target: DatasetTarget,
    bits: int,
    samples: int,
    seed: int,
    rounds: int,
) -> Run:
    cmd = [
        str(binary),
        "distortion",
        "--dataset",
        target.name,
        "--bits",
        str(bits),
        "--samples",
        str(samples),
        "--seed",
        str(seed),
        "--rounds",
        str(rounds),
    ]
    if target.layout:
        cmd.extend(["--layout", target.layout])
    layout_tag = f" layout={target.layout}" if target.layout else ""
    print(f"  running {target.name}{layout_tag} @ bits={bits} ...", file=sys.stderr)
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return Run(
        target=target,
        dim=parse_dim(result.stdout),
        bits=bits,
        values=parse_table(result.stdout),
    )


# Refined small-b values from `main.tex` line 273-274 ("for b = 1, 2, 3, 4 we have
# D_mse approx 0.36, 0.117, 0.03, 0.009"). Tighter than the general sqrt(3)*pi/2 * 4^(-b)
# upper bound, which is what we fall back to for b >= 5.
_NMSE_UPPER_REFINED = {1: 0.36, 2: 0.117, 3: 0.03, 4: 0.009}


def nmse_bound_stage1(bits: int) -> float:
    """Paper's Stage-1 unit-norm reconstruction upper bound for TurboQuant_mse.

    From the Stage 1 theorem (`main.tex`, line 272): for a unit-norm vector `x` quantized
    to `b` bits per coordinate, `E[||x - x'||^2] <= (sqrt(3)*pi/2) * 4^(-b)`. TurboQuant
    internally normalizes each input before quantizing, so the bound applies to per-row
    NMSE = `||x - x'||^2 / ||x||^2 = ||unit(x) - unit(x')||^2` directly. For small `b`
    (1..=4) the paper gives tighter refined values; we splice those in.
    """
    if bits in _NMSE_UPPER_REFINED:
        return _NMSE_UPPER_REFINED[bits]
    return (math.sqrt(3.0) * math.pi / 2.0) / (4.0**bits)


def nmse_lower_bound(bits: int) -> float:
    """Paper's Shannon lower bound on Stage-1 unit-norm reconstruction.

    From `main.tex` line 297: `D_mse(Q) >= 1/4^b` for any randomized `b`-bit quantizer.
    Independent of dimension; applies to NMSE for the same reason as the upper bound.
    """
    return 1.0 / (4.0**bits)


def compression_ratio(bits: int, dim: int) -> float:
    """Theoretical TurboQuant compression ratio vs f32 storage.

    Per the `vortex_tensor::encodings::turboquant` module docs, each vector is stored
    as `padded_dim * bits / 8` bytes of quantized codes plus one f32 stored norm
    (4 bytes), where `padded_dim` is the next power of two at least `dim`. The ratio is
    nonlinear in `bits` because of POT padding and the per-vector norm overhead.
    """
    padded_dim = 1 << (dim - 1).bit_length() if dim > 1 else 1
    per_vector_bytes = padded_dim * bits / 8.0 + 4.0
    original_bytes = dim * 4.0
    return original_bytes / per_vector_bytes


def cosine_sqerr_lower_bound(bits: int, dim: int) -> float:
    """Paper's Shannon lower bound on Stage-2 squared inner-product distortion.

    From `main.tex` line 298: `D_prod(Q) >= ||y||^2 / d * 1/4^b` for any randomized
    `b`-bit quantizer. With unit probes (`||y||^2 = 1`) this is `1 / (d * 4^b)`.
    """
    return 1.0 / (dim * (4.0**bits))


DATASET_PALETTE = [
    "#1f77b4",  # blue
    "#d62728",  # red
    "#2ca02c",  # green
    "#9467bd",  # purple
    "#ff7f0e",  # orange
    "#17becf",  # teal
    "#e377c2",  # pink
    "#8c564b",  # brown
    "#7f7f7f",  # grey
    "#bcbd22",  # olive
]

STAT_STYLES = [
    # (metric_suffix, label, linestyle, linewidth, marker)
    ("mean", "mean", "-", 2.4, "o"),
    ("max", "max", ":", 1.4, None),
]


def plot(runs: list[Run], args: argparse.Namespace) -> None:
    by_dataset: dict[str, list[Run]] = {}
    for r in runs:
        by_dataset.setdefault(r.dataset, []).append(r)
    for ds_runs in by_dataset.values():
        ds_runs.sort(key=lambda r: r.bits)

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.titleweight": "semibold",
            "axes.labelsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.6,
            "legend.frameon": False,
        }
    )

    # GridSpec with a dedicated bottom strip for the caption so the long text gets a real
    # subplot rect: no clipping by `bbox_inches`, no overlap with axis labels, no reliance
    # on matplotlib's `wrap=True` heuristic. Plot row gets the lion's share so the bottom
    # caption strip doesn't dominate visually; legends are anchored above the axes via
    # `bbox_to_anchor` (see `add_legends`), and constrained_layout reserves space for them
    # inside the plot row.
    fig = plt.figure(figsize=(22, 9.5), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, height_ratios=[12, 1])
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    caption_ax = fig.add_subplot(gs[1, :])
    caption_ax.axis("off")
    fig.suptitle(
        f"TurboQuant distortion vs bits per coordinate"
        f"     (samples={args.samples:,}, seed={args.seed}, rounds={args.rounds})",
        fontsize=14,
        fontweight="semibold",
    )

    dataset_colors = {ds: DATASET_PALETTE[i % len(DATASET_PALETTE)] for i, ds in enumerate(by_dataset)}
    dataset_dims = {ds: ds_runs[0].dim for ds, ds_runs in by_dataset.items()}

    plot_panel(
        axes[0],
        by_dataset,
        dataset_colors,
        metric_prefix="reconstruction NMSE",
        title=r"Reconstruction NMSE   (per vector, $\|x - x^\prime\|^2 / \|x\|^2$)",
        ylabel=r"$\|x - x^\prime\|^2 / \|x\|^2$",
    )
    bits_axis = sorted({r.bits for r in runs})
    axes[0].plot(
        bits_axis,
        [nmse_bound_stage1(b) for b in bits_axis],
        color="#222222",
        linestyle=(0, (4, 2, 1, 2)),
        linewidth=1.6,
        zorder=0,
    )
    axes[0].plot(
        bits_axis,
        [nmse_lower_bound(b) for b in bits_axis],
        color="#222222",
        linestyle=(0, (1, 2)),
        linewidth=1.4,
        zorder=0,
    )

    plot_panel(
        axes[1],
        by_dataset,
        dataset_colors,
        metric_prefix="decoded cosine sqerr",
        title=r"Squared cosine error   $(\cos(y_i, x_i) - \cos(y_i, x_i^\prime))^2$",
        ylabel="squared error",
    )
    for dataset, ds_runs in by_dataset.items():
        color = dataset_colors[dataset]
        d = ds_runs[0].dim
        bits = sorted({r.bits for r in ds_runs})
        axes[1].plot(
            bits,
            [cosine_sqerr_lower_bound(b, d) for b in bits],
            color=color,
            linestyle=(0, (1, 2)),
            linewidth=1.0,
            alpha=0.5,
            zorder=0,
        )

    plot_compression_panel(axes[2], by_dataset, dataset_colors)

    add_legends(fig, axes, dataset_colors, dataset_dims)
    caption_ax.text(
        0.5,
        1.0,
        "NMSE upper bound uses the paper's refined small-b values for b<=4 and the "
        "smooth sqrt(3)*pi/2 * 4^(-b) general formula for b>=5.  Lower bounds are the "
        "Shannon information-theoretic floor for any randomized b-bit quantizer.  "
        "Vortex ships TurboQuant Stage 1 only, so no Stage-2 inner-product upper "
        "bound is drawn on the cosine panel.  Probe vectors y_i are sampled iid "
        "uniform on the unit sphere.  Compression ratio is theoretical "
        "(padded_dim * bits / 8 + 4 bytes per vector), excludes per-shard centroid "
        "tables and file metadata.",
        ha="center",
        va="top",
        fontsize=9,
        color="#555555",
        wrap=True,
        transform=caption_ax.transAxes,
    )

    if args.output:
        fig.savefig(args.output, dpi=140, bbox_inches="tight")
        print(f"saved {args.output}", file=sys.stderr)
    else:
        plt.show()


def plot_panel(
    ax,
    by_dataset: dict[str, list[Run]],
    dataset_colors: dict[str, str],
    metric_prefix: str,
    title: str,
    ylabel: str,
) -> None:
    for dataset, ds_runs in by_dataset.items():
        color = dataset_colors[dataset]
        bits = [r.bits for r in ds_runs]
        for stat_key, _label, linestyle, linewidth, marker in STAT_STYLES:
            metric = f"{metric_prefix} {stat_key}"
            ys = [r.values[metric] for r in ds_runs]
            ax.plot(
                bits,
                ys,
                color=color,
                linestyle=linestyle,
                linewidth=linewidth,
                marker=marker,
                markersize=6,
                markerfacecolor=color,
                markeredgecolor="white",
                markeredgewidth=0.8,
                alpha=0.95 if marker else 0.75,
            )
    ax.set_yscale("log")
    ax.set_xlabel("bits per coordinate")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, which="major", linewidth=0.7, alpha=0.45)
    ax.grid(True, which="minor", linewidth=0.4, alpha=0.22)
    ax.minorticks_on()
    # Only the integer bit-widths should get an x-axis line; suppress the in-between
    # minor ticks that `minorticks_on()` adds (the y-axis minors stay - they're useful
    # on the log scale).
    ax.xaxis.set_minor_locator(NullLocator())


def plot_compression_panel(
    ax,
    by_dataset: dict[str, list[Run]],
    dataset_colors: dict[str, str],
) -> None:
    bits_axis = sorted({r.bits for runs in by_dataset.values() for r in runs})
    for dataset, ds_runs in by_dataset.items():
        color = dataset_colors[dataset]
        d = ds_runs[0].dim
        padded = 1 << (d - 1).bit_length() if d > 1 else 1
        suffix = f"  (padded {padded})" if padded != d else "  (no padding)"
        ax.plot(
            bits_axis,
            [compression_ratio(b, d) for b in bits_axis],
            color=color,
            linestyle="-",
            linewidth=2.4,
            marker="o",
            markersize=6,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.8,
            label=f"{dataset}{suffix}",
        )
    ax.set_xlabel("bits per coordinate")
    ax.set_ylabel(r"ratio vs f32  (=  $4d \,/\, (\mathrm{padded}\!\cdot\! b/8 + 4)$)")
    ax.set_title("Compression ratio   (theoretical)")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, which="major", linewidth=0.7, alpha=0.45)
    ax.grid(True, which="minor", linewidth=0.4, alpha=0.22)
    ax.minorticks_on()
    ax.xaxis.set_minor_locator(NullLocator())
    ax.legend(
        title="dataset",
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        fontsize=9,
        title_fontsize=10,
    )


def add_legends(
    fig,
    axes,
    dataset_colors: dict[str, str],
    dataset_dims: dict[str, int],
) -> None:
    dataset_handles = [
        Line2D(
            [],
            [],
            color=color,
            linewidth=2.4,
            marker="o",
            markersize=6,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.8,
            label=f"{dataset}  (d = {dataset_dims[dataset]})",
        )
        for dataset, color in dataset_colors.items()
    ]
    stat_handles = [
        Line2D(
            [],
            [],
            color="#333333",
            linestyle=linestyle,
            linewidth=linewidth,
            marker=marker,
            markersize=6 if marker else 0,
            markerfacecolor="#333333",
            markeredgecolor="white",
            markeredgewidth=0.8,
            label=label,
        )
        for _, label, linestyle, linewidth, marker in STAT_STYLES
    ]
    nmse_upper_handle = Line2D(
        [],
        [],
        color="#222222",
        linestyle=(0, (4, 2, 1, 2)),
        linewidth=1.6,
        label=(
            r"upper bound:  "
            r"$D_{\mathrm{mse}} \leq \frac{\sqrt{3}\,\pi}{2}\, 4^{-b}$  (refined for $b\!\leq\!4$)"
        ),
    )
    nmse_lower_handle = Line2D(
        [],
        [],
        color="#222222",
        linestyle=(0, (1, 2)),
        linewidth=1.4,
        label=r"lower bound:  $D_{\mathrm{mse}} \geq 4^{-b}$",
    )
    cosine_lower_handle = Line2D(
        [],
        [],
        color="#444444",
        linestyle=(0, (1, 2)),
        linewidth=1.0,
        alpha=0.5,
        label=r"lower bound:  $D_{\mathrm{prod}} \geq \frac{1}{d}\, 4^{-b}$",
    )

    axes[0].legend(
        handles=dataset_handles + [nmse_upper_handle, nmse_lower_handle],
        title="dataset / bound",
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        fontsize=10,
        title_fontsize=10,
    )
    axes[1].legend(
        handles=stat_handles + [cosine_lower_handle],
        title="statistic / bound",
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        fontsize=10,
        title_fontsize=10,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--dataset",
        action="append",
        required=True,
        help=(
            "Dataset to sweep (repeat to compare multiple). Optionally suffix "
            "`:layout` to pin a specific train layout for that dataset, e.g. "
            "`--dataset cohere-small-100k:single`. If omitted, the bench picks "
            "the dataset's only layout, or errors if there are several."
        ),
    )
    parser.add_argument(
        "--layout",
        default=None,
        help=("Default train layout applied to any `--dataset` entry that doesn't pin its own with `:layout`."),
    )
    parser.add_argument("--samples", type=int, default=65536)
    parser.add_argument(
        "--bits",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 5, 6, 7, 8],
        help="Bit widths to sweep (default: 1..=8).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument(
        "--binary",
        type=Path,
        default=DEFAULT_BINARY,
        help=f"Path to vector-search-bench (default: {DEFAULT_BINARY}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="If set, save the chart to this path instead of opening a window.",
    )
    args = parser.parse_args()

    print("building vector-search-bench (release) ...", file=sys.stderr)
    subprocess.run(
        ["cargo", "build", "-p", "vector-search-bench", "--release"],
        cwd=REPO_ROOT,
        check=True,
    )

    if not args.binary.exists():
        sys.exit(f"binary not found at {args.binary} after build")

    targets = [parse_dataset_arg(spec, args.layout) for spec in args.dataset]

    runs: list[Run] = []
    for target in targets:
        layout_tag = f" (layout={target.layout})" if target.layout else ""
        print(
            f"sweeping {target.name}{layout_tag} over bits {args.bits} ...",
            file=sys.stderr,
        )
        for bits in args.bits:
            runs.append(
                run_one(
                    args.binary,
                    target,
                    bits,
                    args.samples,
                    args.seed,
                    args.rounds,
                )
            )

    print_summary(runs)
    plot(runs, args)


def print_summary(runs: list[Run]) -> None:
    print()
    print("Summary (one row per (dataset, bits)):")
    header = ["dataset", "dim", "bits"] + METRIC_NAMES
    widths = [max(len(h), 14) for h in header]
    print("  " + "  ".join(h.ljust(w) for h, w in zip(header, widths)))
    for r in runs:
        cells = [r.dataset, str(r.dim), str(r.bits)] + [f"{r.values[m]:.3e}" for m in METRIC_NAMES]
        print("  " + "  ".join(c.ljust(w) for c, w in zip(cells, widths)))


if __name__ == "__main__":
    main()
