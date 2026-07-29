# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Validation for declarative CI benchmark catalogs."""

from ..config import Format, engines_for_benchmark
from .model import BenchmarkCase, Catalog, Coverage, Storage


def _validate_coverage(benchmark: BenchmarkCase, preset: str, coverage: Coverage) -> None:
    if len(coverage.targets) == 0:
        raise ValueError(f"Benchmark {benchmark.id!r} resolved to no runnable targets")

    for target in coverage.targets:
        if not target.is_supported():
            raise ValueError(f"Benchmark {benchmark.id!r} has unsupported target {target} in preset {preset!r}")
        if target.engine not in engines_for_benchmark(benchmark.benchmark):
            raise ValueError(f"Benchmark {benchmark.id!r} cannot run target {target} in preset {preset!r}")
        if benchmark.storage is Storage.S3 and target.format is Format.LANCE:
            raise ValueError(f"Benchmark {benchmark.id!r} cannot run Lance from S3 in preset {preset!r}")

    if coverage.data_formats is not None:
        if len(coverage.data_formats) == 0:
            raise ValueError(f"Benchmark {benchmark.id!r} generates no data formats in preset {preset!r}")
        if len(coverage.data_formats) != len(set(coverage.data_formats)):
            raise ValueError(f"Benchmark {benchmark.id!r} repeats a data format in preset {preset!r}")
        if Format.LANCE in coverage.data_formats:
            raise ValueError(f"Benchmark {benchmark.id!r} cannot generate Lance data in preset {preset!r}")


def validate_catalog(catalog: Catalog) -> None:
    """Validate catalog references and per-preset benchmark invariants."""
    if not catalog.presets:
        raise ValueError("Benchmark catalog defines no matrix presets")

    known_presets = set(catalog.presets)
    ids_by_preset: dict[str, set[str]] = {preset: set() for preset in catalog.presets}

    for benchmark in catalog.benchmarks:
        if not benchmark.runs:
            raise ValueError(f"Benchmark {benchmark.id!r} is not scheduled by any matrix preset")

        unknown_presets = set(benchmark.runs) - known_presets
        if unknown_presets:
            unknown = ", ".join(sorted(unknown_presets))
            raise ValueError(f"Benchmark {benchmark.id!r} references unknown matrix presets: {unknown}")

        if benchmark.storage is Storage.S3 and (benchmark.local_dir is None or benchmark.remote_key is None):
            raise ValueError(f"S3 benchmark {benchmark.id!r} must define local_dir and remote_key")

        for preset, coverage in benchmark.runs.items():
            if benchmark.id in ids_by_preset[preset]:
                raise ValueError(f"Duplicate benchmark ID {benchmark.id!r} in matrix preset {preset!r}")
            ids_by_preset[preset].add(benchmark.id)
            _validate_coverage(benchmark, preset, coverage)
