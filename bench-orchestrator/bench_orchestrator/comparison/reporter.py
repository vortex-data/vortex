# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Output formatting for benchmark comparisons."""

from typing import TYPE_CHECKING

import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.text import Text

if TYPE_CHECKING:
    from .analyzer import PivotComparison

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


def pivot_comparison_table(
    pivot: "PivotComparison",
    threshold: float = 0.10,
    row_keys: list[str] | str = "query",
) -> Table:
    """Generate a rich table for pivot comparison."""
    table = Table()

    # Normalize row_keys to list
    if isinstance(row_keys, str):
        row_keys = [row_keys]

    # Row key columns
    for key in row_keys:
        table.add_column(key.replace("_", " ").title(), style="cyan")

    # Add column for each comparison target
    for col in pivot.columns:
        if col == pivot.baseline:
            table.add_column(f"{col} (base)", justify="right")
        else:
            table.add_column(col, justify="right")

    # Add rows
    for _, row in pivot.df.iterrows():
        cells: list[str | Text] = []

        # Add row key values
        for key in row_keys:
            val = row.get(key, "")
            if key == "query" and val != "":
                cells.append(str(int(val)))
            else:
                cells.append(str(val))

        for col in pivot.columns:
            value = row.get(col, float("nan"))
            ratio = row.get(f"{col}_ratio", float("nan"))

            if pd.isna(value):
                cells.append(Text("N/A", style="dim"))
            elif col == pivot.baseline:
                # Baseline: just show time
                cells.append(_format_time_ns(value))
            else:
                # Non-baseline: show time and ratio
                time_str = _format_time_ns(value)
                if pd.isna(ratio):
                    cells.append(Text(f"{time_str}", style="dim"))
                else:
                    color = _ratio_to_color(ratio, threshold)
                    ratio_str = f"{ratio:.2f}x"
                    cells.append(Text(f"{time_str} ({ratio_str})", style=color))

        table.add_row(*cells)

    return table
