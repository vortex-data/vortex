# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Configuration types and enums for benchmark orchestration."""

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, TypeVar


class Engine(Enum):
    """Logical execution engines for benchmark results."""

    DUCKDB = "duckdb"
    DATAFUSION = "datafusion"
    LANCE = "lance"

    @property
    def binary_name(self) -> str:
        """Return the benchmark executable name for this engine."""
        return {
            Engine.DUCKDB: "duckdb-bench",
            Engine.DATAFUSION: "datafusion-bench",
            Engine.LANCE: "lance-bench",
        }[self]

    @property
    def package_name(self) -> str:
        """Return the Cargo package name that provides this engine's benchmark binary."""
        return {
            Engine.DUCKDB: "duckdb-bench",
            Engine.DATAFUSION: "datafusion-bench",
            Engine.LANCE: "lance-bench",
        }[self]


class Format(Enum):
    """Data formats for benchmarks."""

    PARQUET = "parquet"
    VORTEX = "vortex"
    VORTEX_COMPACT = "vortex-compact"
    VORTEX_SPATIAL_NATIVE = "vortex-spatial-native"
    DUCKDB = "duckdb"
    LANCE = "lance"


class Integration(Enum):
    """How an engine reaches Vortex files.

    The files on disk are the same for every integration; only the engine-side code path differs.
    """

    # The engine's established path: DataFusion's `FileFormat`/`FileOpener` behind a
    # `ListingTable`, and DuckDB's table function.
    DEFAULT = "default"
    # The Scan API (`DataSourceScan::split`/`open`): DataFusion's `VortexTable` over a
    # `MultiFileDataSource`.
    SCAN_API = "scan-api"


class Benchmark(Enum):
    """Available benchmark suites."""

    APPIAN = "appian"
    TPCH = "tpch"
    TPCH_NESTED = "tpch-nested"
    TPCDS = "tpcds"
    CLICKBENCH = "clickbench"
    CLICKBENCH_SORTED = "clickbench-sorted"
    FINEWEB = "fineweb"
    GHARCHIVE = "gh-archive"
    JOB = "job"
    POLARSIGNALS = "polarsignals"
    PUBLIC_BI = "public-bi"
    STATPOPGEN = "statpopgen"
    SPATIALBENCH = "spatialbench"
    VORTEX_QUERIES = "vortex"


# Engine to supported formats mapping.
ENGINE_FORMATS: dict[Engine, list[Format]] = {
    Engine.DATAFUSION: [
        Format.PARQUET,
        Format.VORTEX,
        Format.VORTEX_COMPACT,
        Format.LANCE,
    ],
    Engine.DUCKDB: [
        Format.PARQUET,
        Format.VORTEX,
        Format.VORTEX_COMPACT,
        Format.VORTEX_SPATIAL_NATIVE,
        Format.DUCKDB,
    ],
    Engine.LANCE: [Format.LANCE],
}

# Engines each benchmark can run on. Benchmarks default to *every* engine; list one here only to
# restrict it. SpatialBench's queries use DuckDB-specific `ST_*` spatial SQL that DataFusion has no
# functions for yet.
BENCHMARK_ENGINES: dict[Benchmark, frozenset[Engine]] = {
    Benchmark.SPATIALBENCH: frozenset({Engine.DUCKDB}),
}


# Benchmarks whose Vortex scans still run on the V1 scan. The orchestrator runs every other
# benchmark on the planner-driven V2 scan, overriding the benchmark binaries' V1 default.
# These suites write the list layout, which the V2 scan cannot read yet.
BENCHMARK_SCAN_V1: frozenset[Benchmark] = frozenset(
    {
        Benchmark.POLARSIGNALS,
        Benchmark.STATPOPGEN,
        Benchmark.TPCH_NESTED,
    }
)


# Engine/format pairs each non-default integration supports. A DuckDB Scan API target needs only an
# entry here plus its switch in `BenchmarkExecutor`, once DuckDB has a `DataSource` table function.
INTEGRATION_TARGETS: dict[Integration, dict[Engine, frozenset[Format]]] = {
    Integration.SCAN_API: {
        Engine.DATAFUSION: frozenset({Format.VORTEX, Format.VORTEX_COMPACT}),
    },
}


def uses_scan_v2(benchmark: Benchmark, integration: Integration = Integration.DEFAULT) -> bool:
    """Return whether `benchmark` runs its Vortex scans on the V2 scan through `integration`."""
    # `VortexTable` reaches only the V1 scan today: `MultiFileDataSource` opens `LayoutReader`s.
    # The orchestrator passes `--scan-v1 true` so the command says so. Once the V2 scan sits behind
    # `split`/`open`, Scan API targets should follow the suite's engine like the others.
    if integration is Integration.SCAN_API:
        return False
    return benchmark not in BENCHMARK_SCAN_V1


def engines_for_benchmark(benchmark: Benchmark) -> frozenset[Engine]:
    """Return the engines `benchmark` supports, defaulting to every engine when unrestricted."""
    return BENCHMARK_ENGINES.get(benchmark, frozenset(Engine))


T = TypeVar("T")


def _unique_preserve_order(values: Iterable[T]) -> list[T]:
    seen: set[T] = set()
    unique: list[T] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique.append(value)
    return unique


@dataclass(frozen=True)
class BenchmarkTarget:
    """An explicit engine/format benchmark target, reached through one integration."""

    engine: Engine
    format: Format
    integration: Integration = Integration.DEFAULT

    def normalized(self) -> "BenchmarkTarget":
        """Normalize legacy lance targets onto the logical CI identity."""
        if self.engine == Engine.LANCE and self.format == Format.LANCE:
            return BenchmarkTarget(engine=Engine.DATAFUSION, format=Format.LANCE, integration=self.integration)
        return self

    @property
    def backend(self) -> Engine:
        """Return the engine whose benchmark binary should execute this target."""
        target = self.normalized()
        if target.format == Format.LANCE:
            return Engine.LANCE
        if target.engine == Engine.DATAFUSION:
            return Engine.DATAFUSION
        if target.engine == Engine.DUCKDB:
            return Engine.DUCKDB
        raise ValueError(f"Unsupported benchmark target: {target}")

    def is_supported(self) -> bool:
        """Return whether the logical engine supports this format through this integration."""
        target = self.normalized()
        if target.format not in ENGINE_FORMATS.get(target.engine, []):
            return False
        if target.integration is Integration.DEFAULT:
            return True
        return target.format in INTEGRATION_TARGETS.get(target.integration, {}).get(target.engine, frozenset())

    def uses_scan_v2(self, benchmark: Benchmark) -> bool:
        """Return whether this target's Vortex scans run on the V2 scan in `benchmark`."""
        return uses_scan_v2(benchmark, self.integration)

    def result_format(self, format_name: str) -> str:
        """Return the format label results record for this target, given the binary's label.

        The benchmark binaries label results by engine and format alone, so a non-default
        integration would collide with the default target's rows. The integration is appended to
        the format, which keeps the row under its engine's Parquet controls in comparisons.
        """
        if self.integration is Integration.DEFAULT:
            return format_name
        return f"{format_name} [{self.integration.value}]"

    def to_dict(self) -> dict[str, str]:
        target = self.normalized()
        data = {"engine": target.engine.value, "format": target.format.value}
        if target.integration is not Integration.DEFAULT:
            data["integration"] = target.integration.value
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BenchmarkTarget":
        if not isinstance(data, dict):
            raise ValueError("Targets must be JSON objects with engine and format fields")
        try:
            target = cls(
                engine=Engine(data["engine"]),
                format=Format(data["format"]),
                integration=Integration(data.get("integration", Integration.DEFAULT.value)),
            )
        except KeyError as exc:
            raise ValueError("Targets must include engine and format fields") from exc
        except ValueError as exc:
            raise ValueError(f"Invalid benchmark target: {data}") from exc
        return target.normalized()

    def __str__(self) -> str:
        target = self.normalized()
        return f"{target.engine.value}:{target.result_format(target.format.value)}"


def parse_targets_json(value: str) -> list[BenchmarkTarget]:
    """Parse a JSON array of explicit benchmark targets."""
    try:
        raw_targets = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError("Targets JSON must be a valid JSON array") from exc

    if not isinstance(raw_targets, list):
        raise ValueError("Targets JSON must be an array")

    return _unique_preserve_order(BenchmarkTarget.from_dict(item) for item in raw_targets)


def parse_formats_json(value: str) -> list[Format]:
    """Parse a JSON array of format names."""
    try:
        raw_formats = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError("Formats JSON must be a valid JSON array") from exc

    if not isinstance(raw_formats, list):
        raise ValueError("Formats JSON must be an array")

    formats: list[Format] = []
    for item in raw_formats:
        if not isinstance(item, str):
            raise ValueError("Formats JSON entries must be strings")
        try:
            formats.append(Format(item))
        except ValueError as exc:
            raise ValueError(f"Invalid format: {item}") from exc
    return _unique_preserve_order(formats)


def resolve_axis_targets(
    engines: Iterable[Engine], formats: Iterable[Format], benchmark: Benchmark | None = None
) -> tuple[list[BenchmarkTarget], list[str]]:
    """Expand engine/format axes into supported explicit targets."""
    warnings: list[str] = []
    targets: list[BenchmarkTarget] = []

    for engine in engines:
        if benchmark is not None and engine not in engines_for_benchmark(benchmark):
            warnings.append(f"Benchmark {benchmark.value} does not support engine {engine.value}")
            continue
        for fmt in formats:
            target = BenchmarkTarget(engine=engine, format=fmt).normalized()
            if not target.is_supported():
                warnings.append(f"Format {fmt.value} is not supported by engine {engine.value}")
                continue
            targets.append(target)

    return _unique_preserve_order(targets), warnings


def group_targets_by_backend(targets: Iterable[BenchmarkTarget]) -> dict[Engine, list[BenchmarkTarget]]:
    """Group logical benchmark targets by the backend binary required to run them."""
    groups: dict[Engine, list[BenchmarkTarget]] = {}
    for target in _unique_preserve_order(target.normalized() for target in targets):
        groups.setdefault(target.backend, []).append(target)
    return groups


def validate_targets(
    targets: Iterable[BenchmarkTarget], options: dict[str, str], benchmark: Benchmark | None = None
) -> list[str]:
    """Validate explicit targets against benchmark runner constraints."""
    errors: list[str] = []

    normalized_targets = [target.normalized() for target in targets]
    for target in normalized_targets:
        if target.format not in ENGINE_FORMATS.get(target.engine, []):
            errors.append(f"Format {target.format.value} is not supported by engine {target.engine.value}")
        elif not target.is_supported():
            errors.append(
                f"Integration {target.integration.value} does not support {target.engine.value}:{target.format.value}"
            )
        if benchmark is not None and target.engine not in engines_for_benchmark(benchmark):
            errors.append(f"Benchmark {benchmark.value} does not support engine {target.engine.value}")

    if options.get("remote-data-dir") and any(target.format == Format.LANCE for target in normalized_targets):
        errors.append("Lance format is not supported for remote storage benchmarks.")

    return _unique_preserve_order(errors)


@dataclass
class RunConfig:
    """Configuration for a benchmark run."""

    benchmark: Benchmark
    targets: list[BenchmarkTarget]
    queries: list[int] | None = None
    exclude_queries: list[int] | None = None
    iterations: int = 5
    label: str | None = None
    options: dict[str, str] = field(default_factory=dict)
    track_memory: bool = False

    @property
    def engines(self) -> list[Engine]:
        return _unique_preserve_order(target.engine for target in self.targets)

    @property
    def formats(self) -> list[Format]:
        return _unique_preserve_order(target.format for target in self.targets)

    @property
    def backends(self) -> list[Engine]:
        return _unique_preserve_order(target.backend for target in self.targets)

    def validate(self) -> list[str]:
        """Validate the configuration and return any errors."""
        return validate_targets(self.targets, self.options, self.benchmark)


@dataclass
class BuildConfig:
    """Configuration for building benchmark binaries."""

    profile: str = "release_debug"
    rustflags: str = "-C target-cpu=native -C force-frame-pointers=yes"
    features: tuple[str, ...] = ()


def get_workspace_root() -> Path:
    """Find the workspace root by looking for Cargo.toml with [workspace]."""
    current = Path.cwd()
    for parent in [current, *current.parents]:
        cargo_toml = parent / "Cargo.toml"
        if cargo_toml.exists():
            content = cargo_toml.read_text()
            if "[workspace]" in content:
                return parent
    raise RuntimeError("Could not find workspace root (Cargo.toml with [workspace])")


def get_results_dir() -> Path:
    """Get the directory for storing benchmark results."""
    return get_workspace_root() / "target" / "vortex-bench" / "runs"
