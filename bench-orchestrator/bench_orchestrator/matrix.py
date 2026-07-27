# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Render benchmark definitions as GitHub Actions matrices."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum

from .config import Benchmark, BenchmarkTarget, Engine, Format


class Storage(Enum):
    """Where a benchmark's data lives when it runs."""

    NVME = "nvme"
    S3 = "s3"

    @property
    def label(self) -> str:
        """Human-facing name used in benchmark display names."""
        return "NVME" if self is Storage.NVME else "S3"


class BenchmarkGroup(Enum):
    """A separately scheduled group of benchmark definitions."""

    REGULAR = "regular"
    NIGHTLY = "nightly"


def _dedupe(targets: Iterable[BenchmarkTarget]) -> tuple[BenchmarkTarget, ...]:
    """Normalize and de-duplicate targets, preserving first-seen order."""
    return tuple(dict.fromkeys(target.normalized() for target in targets))


@dataclass(frozen=True)
class TargetSet:
    """An ordered set of engine/format targets with small set algebra."""

    targets: tuple[BenchmarkTarget, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "targets", _dedupe(self.targets))

    def __or__(self, other: TargetSet) -> TargetSet:
        """Return the ordered union of two target sets."""
        return TargetSet(self.targets + other.targets)

    def only(self, *engines: Engine) -> TargetSet:
        """Restrict the target set to the given engines."""
        keep = set(engines)
        return TargetSet(tuple(target for target in self.targets if target.engine in keep))

    def formats(self) -> list[Format]:
        """Return referenced formats in first-seen order."""
        return list(dict.fromkeys(target.format for target in self.targets))

    def __iter__(self):
        return iter(self.targets)

    def __len__(self) -> int:
        return len(self.targets)


def targets(engine: Engine, *formats: Format) -> TargetSet:
    """Build targets for one engine across several formats."""
    return TargetSet(tuple(BenchmarkTarget(engine=engine, format=fmt) for fmt in formats))


def df(*formats: Format) -> TargetSet:
    """Build DataFusion targets across several formats."""
    return targets(Engine.DATAFUSION, *formats)


def duck(*formats: Format) -> TargetSet:
    """Build DuckDB targets across several formats."""
    return targets(Engine.DUCKDB, *formats)


STANDARD = df(Format.PARQUET, Format.VORTEX, Format.VORTEX_COMPACT) | duck(
    Format.PARQUET, Format.VORTEX, Format.VORTEX_COMPACT
)
DEFAULTS = df(Format.PARQUET, Format.VORTEX) | duck(Format.PARQUET, Format.VORTEX)
_NOT_GENERATED = frozenset({Format.LANCE})
_FORMAT_ORDER = (
    Format.PARQUET,
    Format.VORTEX,
    Format.VORTEX_COMPACT,
    Format.VORTEX_NATIVE,
    Format.DUCKDB,
    Format.LANCE,
)


@dataclass(frozen=True)
class BenchmarkDef:
    """A benchmark and the canonical target superset it can run."""

    id: str
    benchmark: Benchmark
    name: str
    targets: TargetSet
    storage: Storage = Storage.NVME
    scale_factor: float | int | None = None
    iterations: int | None = None
    group: BenchmarkGroup = BenchmarkGroup.REGULAR
    pr_targets: TargetSet | None = None
    run_in_pr: bool = True
    local_dir: str | None = None
    remote_key: str | None = None

    @property
    def subcommand(self) -> str:
        """Return the ``vx-bench`` subcommand for this benchmark."""
        return self.benchmark.value


def _default_targets(benchmark: BenchmarkDef) -> TargetSet:
    """Return the cheap default targets supported by a benchmark."""
    return TargetSet(tuple(target for target in benchmark.targets if target in DEFAULTS.targets))


def _pr_full_targets(benchmark: BenchmarkDef) -> TargetSet:
    """Return the full target set supported by pull-request runners."""
    if benchmark.pr_targets is not None:
        return benchmark.pr_targets
    return TargetSet(tuple(target for target in benchmark.targets if target.format is not Format.LANCE))


def _pr_targets(benchmark: BenchmarkDef) -> TargetSet:
    """Return the cheap PR lane."""
    return TargetSet(tuple(target for target in _pr_full_targets(benchmark) if target in DEFAULTS))


MATRIX_PRESETS = {
    "develop": "Every SQL benchmark at full target coverage.",
    "pr": "The quicker pull-request SQL benchmark matrix.",
    "pr-full": "Every SQL benchmark at full PR target coverage.",
    "nightly": "Large-scale SF=100 TPC-H on NVMe and S3 at default targets.",
}


def _include_benchmark(preset: str, benchmark: BenchmarkDef) -> bool:
    if preset == "nightly":
        return benchmark.group is BenchmarkGroup.NIGHTLY
    return benchmark.group is BenchmarkGroup.REGULAR and (preset != "pr" or benchmark.run_in_pr)


def _targets_for(preset: str, benchmark: BenchmarkDef) -> TargetSet:
    if preset == "develop":
        return benchmark.targets
    if preset == "pr-full":
        return _pr_full_targets(benchmark)
    if preset == "pr":
        return _pr_targets(benchmark)
    return _default_targets(benchmark)


def _valid_for_storage(target_set: TargetSet, storage: Storage) -> TargetSet:
    """Drop targets that are invalid for the storage backend."""
    if storage is Storage.S3:
        return TargetSet(tuple(target for target in target_set if target.format is not Format.LANCE))
    return target_set


def _data_formats(target_set: TargetSet) -> list[Format]:
    """Return data formats that the data-generation step must produce."""
    present = set(target_set.formats())
    return [fmt for fmt in _FORMAT_ORDER if fmt in present and fmt not in _NOT_GENERATED]


def _matrix_entry(benchmark: BenchmarkDef, run_targets: TargetSet, data_format_targets: TargetSet) -> dict[str, object]:
    """Build one GitHub Actions ``include`` entry."""
    entry: dict[str, object] = {
        "id": benchmark.id,
        "subcommand": benchmark.subcommand,
        "name": benchmark.name,
        "targets": [target.to_dict() for target in run_targets],
        "data_formats": [fmt.value for fmt in _data_formats(data_format_targets)],
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


def resolve_matrix(preset: str, benchmarks: Iterable[BenchmarkDef]) -> list[dict[str, object]]:
    """Render a named preset as GitHub Actions matrix entries."""
    if preset not in MATRIX_PRESETS:
        known = ", ".join(MATRIX_PRESETS)
        raise ValueError(f"Unknown matrix preset {preset!r}. Available: {known}")

    entries: list[dict[str, object]] = []
    seen_ids: set[str] = set()
    for benchmark in benchmarks:
        if not _include_benchmark(preset, benchmark):
            continue
        if benchmark.id in seen_ids:
            raise ValueError(f"Duplicate benchmark ID {benchmark.id!r} in matrix preset {preset!r}")
        seen_ids.add(benchmark.id)

        run_targets = _valid_for_storage(_targets_for(preset, benchmark), benchmark.storage)
        if len(run_targets) == 0:
            raise ValueError(f"Benchmark {benchmark.id!r} resolved to no runnable targets")
        data_format_targets = benchmark.targets if preset == "pr-full" else run_targets
        data_format_targets = _valid_for_storage(data_format_targets, benchmark.storage)
        entries.append(_matrix_entry(benchmark, run_targets, data_format_targets))
    return entries
