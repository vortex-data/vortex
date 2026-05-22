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
table from stdout, and plots reconstruction NMSE and pairwise cosine-error curves with
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
from matplotlib.ticker import MaxNLocator

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BINARY = REPO_ROOT / "target" / "release" / "vector-search-bench"

METRIC_NAMES = [
    "reconstruction NMSE mean",
    "reconstruction NMSE median",
    "reconstruction NMSE max",
    "decoded cosine err mean",
    "decoded cosine err median",
    "decoded cosine err max",
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


def nmse_bound_stage1(bits: int) -> float:
    """Paper's NMSE upper bound for TurboQuant_mse (Stage 1).

    From the Stage 1 theorem (`main.tex`, line 272): for a unit-norm vector `x` quantized
    to `b` bits per coordinate, `E[||x - x'||^2] <= (sqrt(3)*pi/2) / 4^b`. Because `x` is
    unit-norm, `||x - x'||^2` equals the normalized squared error `||x - x'||^2 / ||x||^2`,
    so the bound applies to the `reconstruction NMSE mean` curve directly.
    """
    return (math.sqrt(3.0) * math.pi / 2.0) / (4.0**bits)


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


def cosine_bound(bits: int, dim: int) -> float:
    """Paper's Stage-2 inner-product bound, rendered as an absolute-error envelope.

    From the Stage 2 theorem (`main.tex`, line 288): for unit y and an `x` quantized via
    TurboQuant_prod (Stage 2, MSE + QJL residual), `E[|<y, x> - <y, x'>|^2] <=
    sqrt(3)*pi^2/d * 4^(-b)`. Taking sqrt gives an upper envelope on the RMS error per
    bit width, and by Jensen also on the mean abs error.

    Caveat: Vortex currently implements only Stage 1 (no QJL residual correction). The
    Stage 1 inner-product error is biased and can sit *above* this Stage-2 envelope.
    """
    return math.pi * (3.0**0.25) / math.sqrt(dim) / (2.0**bits)


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

    fig, axes = plt.subplots(1, 3, figsize=(20, 6.5), constrained_layout=True)
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
        title="Reconstruction NMSE   (per vector, normalized squared error)",
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

    plot_panel(
        axes[1],
        by_dataset,
        dataset_colors,
        metric_prefix="decoded cosine err",
        title=r"Pairwise cosine error   $|\cos(x_i, x_j) - \cos(x_i^\prime, x_j^\prime)|$",
        ylabel="absolute error",
    )
    for dataset, ds_runs in by_dataset.items():
        color = dataset_colors[dataset]
        d = ds_runs[0].dim
        bits = sorted({r.bits for r in ds_runs})
        axes[1].plot(
            bits,
            [cosine_bound(b, d) for b in bits],
            color=color,
            linestyle=(0, (4, 2, 1, 2)),
            linewidth=1.2,
            alpha=0.6,
            zorder=0,
        )

    plot_compression_panel(axes[2], by_dataset, dataset_colors)

    add_legends(fig, axes, dataset_colors, dataset_dims)
    fig.text(
        0.5,
        -0.015,
        "Cosine bound is the paper's Stage-2 (TurboQuant_prod, MSE + QJL residual) "
        "envelope; Vortex currently ships Stage 1 only, so empirical curves may sit "
        "above it.  Compression ratio is theoretical "
        "(padded_dim * bits / 8 + 4 bytes per vector), excludes per-shard centroid "
        "tables and file metadata.",
        ha="center",
        fontsize=9,
        color="#555555",
        wrap=True,
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
    ax.legend(
        title="dataset",
        loc="upper right",
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
    nmse_bound_handle_s1 = Line2D(
        [],
        [],
        color="#222222",
        linestyle=(0, (4, 2, 1, 2)),
        linewidth=1.6,
        label=r"paper bound:  $D_{\mathrm{mse}} \leq \frac{\sqrt{3}\,\pi}{2}\, 4^{-b}$",
    )
    cosine_bound_handle = Line2D(
        [],
        [],
        color="#444444",
        linestyle=(0, (4, 2, 1, 2)),
        linewidth=1.2,
        alpha=0.6,
        label=(
            r"paper Stage-2 bound:  "
            r"$\sqrt{D_{\mathrm{prod}}} \leq \frac{\pi\,3^{1/4}}{\sqrt{d}}\, 2^{-b}$"
        ),
    )

    axes[0].legend(
        handles=dataset_handles + [nmse_bound_handle_s1],
        title="dataset / bound",
        loc="upper right",
        fontsize=10,
        title_fontsize=10,
    )
    axes[1].legend(
        handles=stat_handles + [cosine_bound_handle],
        title="statistic / bound",
        loc="upper right",
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
