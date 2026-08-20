# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Render declarative benchmark coverage as GitHub Actions matrices."""

from ..config import Format
from .model import BenchmarkCase, Catalog, Coverage
from .validate import validate_catalog

_NOT_GENERATED = frozenset({Format.LANCE})
_FORMAT_ORDER = (
    Format.PARQUET,
    Format.VORTEX,
    Format.VORTEX_COMPACT,
    Format.VORTEX_SPATIAL_NATIVE,
    Format.DUCKDB,
    Format.LANCE,
)


def _data_formats(coverage: Coverage) -> list[Format]:
    if coverage.data_formats is not None:
        return list(coverage.data_formats)

    present = {target.format for target in coverage.targets}
    return [fmt for fmt in _FORMAT_ORDER if fmt in present and fmt not in _NOT_GENERATED]


def _matrix_entry(benchmark: BenchmarkCase, coverage: Coverage) -> dict[str, object]:
    entry: dict[str, object] = {
        "id": benchmark.id,
        "subcommand": benchmark.subcommand,
        "name": benchmark.name,
        "targets": [target.to_dict() for target in coverage.targets],
        "data_formats": [fmt.value for fmt in _data_formats(coverage)],
    }
    if benchmark.scale_factor is not None:
        entry["scale_factor"] = str(benchmark.scale_factor)
    if benchmark.iterations is not None:
        entry["iterations"] = str(benchmark.iterations)
    if benchmark.local_dir is not None:
        entry["local_dir"] = benchmark.local_dir
    if benchmark.remote_key is not None:
        entry["remote_key"] = benchmark.remote_key
    return entry


def render_matrix(preset: str, catalog: Catalog) -> list[dict[str, object]]:
    """Render one named catalog preset as GitHub Actions matrix entries."""
    if preset not in catalog.presets:
        known = ", ".join(catalog.presets)
        raise ValueError(f"Unknown matrix preset {preset!r}. Available: {known}")

    validate_catalog(catalog)
    return [
        _matrix_entry(benchmark, benchmark.runs[preset]) for benchmark in catalog.benchmarks if preset in benchmark.runs
    ]
