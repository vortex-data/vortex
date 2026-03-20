#!/usr/bin/env python3
"""
Plot FSST analysis results from CSV files.

Reads CSVs from encodings/fsst/data/ and generates matplotlib figures.

Usage:
    python encodings/fsst/scripts/plot_analysis.py
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
OUT_DIR = os.path.join(DATA_DIR, "plots")


def load_csv(name: str) -> pd.DataFrame | None:
    path = os.path.join(DATA_DIR, name)
    if not os.path.exists(path):
        print(f"Warning: {path} not found, skipping")
        return None
    return pd.read_csv(path)


def fig_compression_ratio(df: pd.DataFrame):
    """Bar chart: compression ratio by dataset type."""
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.Set2(np.linspace(0, 1, len(df)))
    bars = ax.bar(df["dataset"], df["compression_ratio"], color=colors, edgecolor="black", linewidth=0.5)

    # Add value labels
    for bar, val in zip(bars, df["compression_ratio"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f"{val:.2f}x", ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("Compression Ratio (higher = better)")
    ax.set_title("FSST Compression Ratio Across Data Types")
    ax.set_xticklabels(df["dataset"], rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "compression_ratio.png"), dpi=150)
    plt.close()
    print("  -> compression_ratio.png")


def fig_symbol_length_dist(df: pd.DataFrame):
    """Stacked bar chart: symbol length distribution per dataset."""
    fig, ax = plt.subplots(figsize=(14, 7))
    sym_cols = [f"sym_{i}" for i in range(1, 9)]
    datasets = df["dataset"].values

    bottom = np.zeros(len(datasets))
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, 8))

    for i, col in enumerate(sym_cols):
        if col in df.columns:
            vals = df[col].values.astype(float)
            ax.bar(datasets, vals, bottom=bottom, label=f"len={i+1}", color=colors[i],
                   edgecolor="black", linewidth=0.3)
            bottom += vals

    ax.set_ylabel("Number of Symbols")
    ax.set_title("FSST Symbol Length Distribution by Dataset")
    ax.set_xticklabels(datasets, rotation=45, ha="right")
    ax.legend(title="Symbol Length", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "symbol_length_dist.png"), dpi=150)
    plt.close()
    print("  -> symbol_length_dist.png")


def fig_escape_vs_compression(df: pd.DataFrame):
    """Scatter: escape rate vs compression ratio, sized by mean sym len."""
    fig, ax = plt.subplots(figsize=(10, 7))
    scatter = ax.scatter(
        df["escape_rate"] * 100,
        df["compression_ratio"],
        s=df["mean_sym_len"] * 80,
        c=df["entropy"],
        cmap="coolwarm",
        edgecolors="black",
        linewidth=0.5,
        alpha=0.8,
    )
    for _, row in df.iterrows():
        ax.annotate(row["dataset"], (row["escape_rate"] * 100, row["compression_ratio"]),
                     textcoords="offset points", xytext=(5, 5), fontsize=8)

    ax.set_xlabel("Escape Rate (%)")
    ax.set_ylabel("Compression Ratio")
    ax.set_title("Escape Rate vs Compression (color=entropy, size=mean sym len)")
    fig.colorbar(scatter, label="Shannon Entropy (bits)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "escape_vs_compression.png"), dpi=150)
    plt.close()
    print("  -> escape_vs_compression.png")


def fig_noise_sweep(df: pd.DataFrame):
    """Line charts: noise % vs compression metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    metrics = [
        ("compression_ratio", "Compression Ratio"),
        ("escape_rate", "Escape Rate"),
        ("mean_sym_len", "Mean Symbol Length"),
        ("entropy", "Shannon Entropy (bits)"),
    ]

    for ax, (col, label) in zip(axes.flat, metrics):
        ax.plot(df["noise_pct"] * 100, df[col], "o-", color="steelblue", markersize=4)
        ax.set_xlabel("Noise %")
        ax.set_ylabel(label)
        ax.set_title(f"Noise % vs {label}")
        ax.grid(alpha=0.3)

    fig.suptitle("FSST Behavior Under Increasing Noise", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "noise_sweep.png"), dpi=150)
    plt.close()
    print("  -> noise_sweep.png")


def fig_regex_speedup(df: pd.DataFrame):
    """Bar chart: regex fused DFA speedup over byte DFA."""
    fig, ax = plt.subplots(figsize=(14, 7))

    labels = [f"{row['dataset']}\n{row['pattern']}" for _, row in df.iterrows()]
    colors = ["steelblue" if row["dataset"] == "english_prose" else "coral"
              for _, row in df.iterrows()]

    bars = ax.bar(range(len(df)), df["speedup"], color=colors, edgecolor="black", linewidth=0.5)

    for bar, val in zip(bars, df["speedup"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.2f}x", ha="center", va="bottom", fontsize=8)

    ax.axhline(y=1.0, color="red", linestyle="--", alpha=0.5, label="Break-even")
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Speedup (byte DFA / fused DFA)")
    ax.set_title("Regex-over-FSST Speedup: Fused DFA vs Byte-level DFA")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "regex_speedup.png"), dpi=150)
    plt.close()
    print("  -> regex_speedup.png")


def fig_symlen_vs_speedup(df: pd.DataFrame):
    """Scatter: mean symbol length vs regex speedup."""
    fig, ax = plt.subplots(figsize=(10, 7))

    ax.scatter(df["mean_sym_len"], df["speedup"], s=100, c="steelblue",
               edgecolors="black", linewidth=0.5, alpha=0.8, zorder=5)

    for _, row in df.iterrows():
        ax.annotate(row["pattern"], (row["mean_sym_len"], row["speedup"]),
                     textcoords="offset points", xytext=(5, 5), fontsize=8)

    # Ideal line: speedup = mean_sym_len
    x = np.linspace(df["mean_sym_len"].min() * 0.9, df["mean_sym_len"].max() * 1.1, 50)
    ax.plot(x, x, "--", color="red", alpha=0.5, label="Ideal: speedup = mean_sym_len")
    ax.axhline(y=1.0, color="gray", linestyle=":", alpha=0.5)

    ax.set_xlabel("Mean Symbol Length (bytes)")
    ax.set_ylabel("Speedup (byte DFA / fused DFA)")
    ax.set_title("Mean Symbol Length vs Regex Speedup")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "symlen_vs_speedup.png"), dpi=150)
    plt.close()
    print("  -> symlen_vs_speedup.png")


def fig_effective_alphabet(df: pd.DataFrame):
    """Bar chart: effective alphabet size (2^entropy) by dataset."""
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.Paired(np.linspace(0, 1, len(df)))

    bars = ax.bar(df["dataset"], df["effective_alphabet"], color=colors,
                  edgecolor="black", linewidth=0.5)

    for bar, val in zip(bars, df["effective_alphabet"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.0f}", ha="center", va="bottom", fontsize=9)

    ax.axhline(y=256, color="red", linestyle="--", alpha=0.5, label="Max (256)")
    ax.set_ylabel("Effective Alphabet Size (2^entropy)")
    ax.set_title("FSST Effective Alphabet Size by Dataset")
    ax.set_xticklabels(df["dataset"], rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "effective_alphabet.png"), dpi=150)
    plt.close()
    print("  -> effective_alphabet.png")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Reading CSVs from {DATA_DIR}")
    print(f"Writing plots to {OUT_DIR}\n")

    # Compression analysis plots
    compress_df = load_csv("compress_analysis.csv")
    if compress_df is not None:
        print("Compression analysis plots:")
        fig_compression_ratio(compress_df)
        fig_symbol_length_dist(compress_df)
        fig_escape_vs_compression(compress_df)
        fig_effective_alphabet(compress_df)

    # Noise sweep plots
    noise_df = load_csv("noise_sweep.csv")
    if noise_df is not None:
        print("Noise sweep plots:")
        fig_noise_sweep(noise_df)

    # Regex benchmark plots
    regex_df = load_csv("regex_bench.csv")
    if regex_df is not None:
        print("Regex benchmark plots:")
        fig_regex_speedup(regex_df)
        fig_symlen_vs_speedup(regex_df)

    print(f"\nDone! All plots in {OUT_DIR}/")


if __name__ == "__main__":
    main()
