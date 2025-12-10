# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Output formatting for benchmark comparisons."""

from typing import Any

import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.text import Text

console = Console()


def _ratio_to_color(ratio: float, threshold: float = 0.10) -> str:
    """Convert ratio to a color name."""
    if pd.isna(ratio):
        return "dim"
    if ratio < (1.0 - threshold):
        return "green"
    if ratio > (1.0 + threshold):
        return "red"
    return "yellow"


def _format_ratio(ratio: float, threshold: float = 0.10) -> Text:
    """Format ratio with color coding."""
    if pd.isna(ratio):
        return Text("N/A", style="dim")

    color = _ratio_to_color(ratio, threshold)
    text = f"{ratio:.3f}x"

    if ratio < (1.0 - threshold):
        text += " \u2191"  # Up arrow (faster)
    elif ratio > (1.0 + threshold):
        text += " \u2193"  # Down arrow (slower)

    return Text(text, style=color)


def _format_time_ns(value: float) -> str:
    """Format nanoseconds in a human-readable way."""
    if pd.isna(value):
        return "N/A"
    if value < 1_000:
        return f"{value:.0f}ns"
    if value < 1_000_000:
        return f"{value / 1_000:.1f}\u03bcs"
    if value < 1_000_000_000:
        return f"{value / 1_000_000:.1f}ms"
    return f"{value / 1_000_000_000:.2f}s"


class BenchmarkReporter:
    """Formats comparison results for output."""

    def __init__(
        self,
        comparison_df: pd.DataFrame,
        stats: dict[str, Any] | None = None,
        threshold: float = 0.10,
    ):
        self.df = comparison_df
        self.stats = stats or {}
        self.threshold = threshold

    def to_rich_table(
        self,
        title: str | None = None,
        base_label: str = "base",
        target_label: str = "target",
    ) -> Table:
        """Generate a rich table for terminal output."""
        table = Table(title=title or "Benchmark Comparison")

        table.add_column("Query", style="cyan")
        table.add_column(base_label, justify="right")
        table.add_column(target_label, justify="right")
        table.add_column("Ratio", justify="right")

        for _, row in self.df.iterrows():
            query = str(row.get("query", ""))

            base_val = row.get("value_base", float("nan"))
            target_val = row.get("value_target", float("nan"))
            ratio = row.get("ratio", float("nan"))

            table.add_row(
                query,
                _format_time_ns(base_val),
                _format_time_ns(target_val),
                _format_ratio(ratio, self.threshold),
            )

        return table

    def to_markdown(
        self,
        base_label: str = "base",
        target_label: str = "target",
    ) -> str:
        """Generate markdown table (GitHub-compatible)."""
        lines = []

        # Header
        lines.append(f"| Query | {base_label} | {target_label} | Ratio |")
        lines.append("|-------|---------|--------|-------|")

        for _, row in self.df.iterrows():
            name = str(row.get("name", ""))
            if "/" in name:
                name = name.split("/")[0]

            base_val = row.get("value_base", float("nan"))
            target_val = row.get("value_target", float("nan"))
            ratio = row.get("ratio", float("nan"))

            ratio_str = f"{ratio:.3f}x" if not pd.isna(ratio) else "N/A"
            if not pd.isna(ratio):
                if ratio < (1.0 - self.threshold):
                    ratio_str += " \U0001f680"  # Rocket
                elif ratio > (1.0 + self.threshold):
                    ratio_str += " \U0001f6a8"  # Alarm

            lines.append(f"| {name} | {_format_time_ns(base_val)} | {_format_time_ns(target_val)} | {ratio_str} |")

        return "\n".join(lines)

    def summary(self) -> str:
        """Generate summary statistics."""
        lines = ["## Summary", ""]

        geomean = self.stats.get("geomean", float("nan"))
        if not pd.isna(geomean):
            if geomean < (1.0 - self.threshold):
                emoji = "\u2705"  # Green check
            elif geomean > (1.0 + self.threshold):
                emoji = "\u274c"  # Red X
            else:
                emoji = "\u2796"  # Neutral
            lines.append(f"- **Overall**: {geomean:.3f}x {emoji}")

        improvements = self.stats.get("improvements", 0)
        regressions = self.stats.get("regressions", 0)
        lines.append(f"- **Improvements**: {improvements}")
        lines.append(f"- **Regressions**: {regressions}")

        best_name = self.stats.get("best_name")
        best_ratio = self.stats.get("best_ratio")
        if best_name and not pd.isna(best_ratio):
            lines.append(f"- **Best**: {best_name} ({best_ratio:.3f}x)")

        worst_name = self.stats.get("worst_name")
        worst_ratio = self.stats.get("worst_ratio")
        if worst_name and not pd.isna(worst_ratio):
            lines.append(f"- **Worst**: {worst_name} ({worst_ratio:.3f}x)")

        return "\n".join(lines)

    def print_summary(self) -> None:
        """Print summary to console with rich formatting."""
        geomean = self.stats.get("geomean", float("nan"))
        improvements = self.stats.get("improvements", 0)
        regressions = self.stats.get("regressions", 0)

        console.print("\n[bold]Summary[/bold]")

        if not pd.isna(geomean):
            color = _ratio_to_color(geomean, self.threshold)
            console.print(f"  Overall: [{color}]{geomean:.3f}x[/{color}]")

        console.print(f"  Improvements: [green]{improvements}[/green]")
        console.print(f"  Regressions: [red]{regressions}[/red]")

        best_name = self.stats.get("best_name")
        best_ratio = self.stats.get("best_ratio")
        if best_name and not pd.isna(best_ratio):
            console.print(f"  Best: {best_name} ([green]{best_ratio:.3f}x[/green])")

        worst_name = self.stats.get("worst_name")
        worst_ratio = self.stats.get("worst_ratio")
        if worst_name and not pd.isna(worst_ratio):
            console.print(f"  Worst: {worst_name} ([red]{worst_ratio:.3f}x[/red])")
