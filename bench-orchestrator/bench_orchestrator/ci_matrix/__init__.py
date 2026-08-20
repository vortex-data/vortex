# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Declarative CI benchmark matrix resolution."""

from .catalog import CATALOG
from .render import render_matrix

MATRIX_PRESETS = CATALOG.presets
BENCHMARKS = CATALOG.benchmarks


def resolve_matrix(preset: str) -> list[dict[str, object]]:
    """Render a named preset from the canonical benchmark catalog."""
    return render_matrix(preset, CATALOG)


__all__ = ["BENCHMARKS", "MATRIX_PRESETS", "resolve_matrix"]
