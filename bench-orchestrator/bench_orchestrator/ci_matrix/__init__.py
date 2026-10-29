# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Declarative CI benchmark matrix resolution."""

from collections.abc import Collection

from .catalog import CATALOG
from .render import render_matrix

MATRIX_PRESETS = CATALOG.presets
BENCHMARKS = CATALOG.benchmarks


def resolve_matrix(preset: str, only: Collection[str] | None = None) -> list[dict[str, object]]:
    """Render a named preset from the canonical benchmark catalog, optionally limited to some IDs."""
    return render_matrix(preset, CATALOG, only)


__all__ = ["BENCHMARKS", "MATRIX_PRESETS", "resolve_matrix"]
