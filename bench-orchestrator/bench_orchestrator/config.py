# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Configuration types and enums for benchmark orchestration."""

import json
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
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
    """How an engine reaches Vortex files. The files on disk are the same for every integration."""

    # File-at-a-time integration: DataFusion's `ListingTable` and `FileOpener`, DuckDB's
    # `MultiFileReader`.
    FILE = "file"
    # A `DataSource` over the whole table: DataFusion's `VortexTable`. DuckDB has no `DataSource`
    # table function yet.
    DATASOURCE = "datasource"


class ScanEngine(Enum):
    """The Vortex scan engine a target runs on."""

    # `LayoutReader`s.
    V1 = "v1"
    # Planner-driven scan plans.
    V2 = "v2"


class SizeFormat(Enum):
    """A format a benchmark writes only to record its file sizes. No query reads it.

    `vx-bench write-size-formats` writes Arrow IPC and Snappy Parquet with `data-gen --size-formats`,
    and Lance with `lance-bench --data-only`.
    """

    ARROW_IPC = "arrow-ipc"
    LANCE = "lance"
    # The base Parquet with Snappy in place of zstd level 3, and every other writer setting the same.
    PARQUET_SNAPPY = "parquet-snappy"


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


# Formats whose targets name an integration and a scan engine. Other formats (Parquet, DuckDB's own
# format, Lance) are controls and comparisons, with neither axis.
VORTEX_FORMATS: frozenset[Format] = frozenset({Format.VORTEX, Format.VORTEX_COMPACT, Format.VORTEX_SPATIAL_NATIVE})

# Benchmarks the V2 scan cannot run yet: they write the list layout, which V2 cannot read. This is a
# constraint, not a selector; a target names its scan engine, and V2 targets are rejected here.
BENCHMARKS_WITHOUT_SCAN_V2: frozenset[Benchmark] = frozenset(
    {
        Benchmark.POLARSIGNALS,
        Benchmark.STATPOPGEN,
        Benchmark.TPCH_NESTED,
    }
)

# The integration and scan engine combinations each engine can run on Vortex formats. Adding a
# combination here, plus its flag in `BenchmarkExecutor`, is all a new one needs.
SUPPORTED_SCANS: dict[Engine, dict[Integration, frozenset[ScanEngine]]] = {
    Engine.DATAFUSION: {
        Integration.FILE: frozenset({ScanEngine.V1, ScanEngine.V2}),
        # `VortexTable` reaches only V1: `MultiFileDataSource` opens `LayoutReader`s. V2 needs the V2
        # scan behind `split`/`open` (#11626).
        Integration.DATASOURCE: frozenset({ScanEngine.V1}),
    },
    Engine.DUCKDB: {
        Integration.FILE: frozenset({ScanEngine.V1, ScanEngine.V2}),
    },
}

# Every combination in a stable order: the suite's default engine comes first wherever it is used.
SCAN_COMBINATIONS: tuple[tuple[Integration, ScanEngine], ...] = (
    (Integration.FILE, ScanEngine.V2),
    (Integration.FILE, ScanEngine.V1),
    (Integration.DATASOURCE, ScanEngine.V2),
    (Integration.DATASOURCE, ScanEngine.V1),
)


def supports_scan_v2(benchmark: Benchmark) -> bool:
    """Return whether the V2 scan can run `benchmark`."""
    return benchmark not in BENCHMARKS_WITHOUT_SCAN_V2


def default_scan_engine(benchmark: Benchmark) -> ScanEngine:
    """Return the scan engine a Vortex target runs when it names none: V2 wherever V2 can run."""
    return ScanEngine.V2 if supports_scan_v2(benchmark) else ScanEngine.V1


def scan_support_error(engine: Engine, integration: Integration, scan: ScanEngine | None) -> str | None:
    """Explain why `engine` cannot scan Vortex through `integration` on `scan`, or return None."""
    integrations = SUPPORTED_SCANS.get(engine, {})
    if integration not in integrations:
        if integration is Integration.DATASOURCE and engine is Engine.DUCKDB:
            return "duckdb has no DataSource table function yet, so it has no datasource targets"
        return f"{engine.value} has no {integration.value} integration"
    if scan is not None and scan not in integrations[integration]:
        if engine is Engine.DATAFUSION and integration is Integration.DATASOURCE and scan is ScanEngine.V2:
            return (
                "datafusion datasource targets run only the v1 scan: MultiFileDataSource opens "
                "LayoutReaders, and v2 needs the V2 scan behind split/open (#11626)"
            )
        return f"{engine.value} {integration.value} targets cannot run the {scan.value} scan"
    return None


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
    """An explicit benchmark target.

    Vortex targets also name an integration and a scan engine. A Vortex target without a scan
    engine runs the suite's default, see `resolved`.
    """

    engine: Engine
    format: Format
    integration: Integration = Integration.FILE
    scan: ScanEngine | None = None

    def normalized(self) -> "BenchmarkTarget":
        """Normalize legacy lance targets onto the logical CI identity."""
        if self.engine == Engine.LANCE and self.format == Format.LANCE:
            return replace(self, engine=Engine.DATAFUSION)
        return self

    @property
    def is_vortex(self) -> bool:
        """Return whether this target names an integration and a scan engine."""
        return self.format in VORTEX_FORMATS

    def resolved(self, benchmark: Benchmark) -> "BenchmarkTarget":
        """Fill in the scan engine a Vortex target runs in `benchmark` when it names none."""
        target = self.normalized()
        if target.is_vortex and target.scan is None:
            return replace(target, scan=default_scan_engine(benchmark))
        return target

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

    def support_errors(self, benchmark: Benchmark | None = None) -> list[str]:
        """Explain why this target cannot run, in `benchmark` when given; empty when it can."""
        target = self.normalized()
        if target.format not in ENGINE_FORMATS.get(target.engine, []):
            return [f"Format {target.format.value} is not supported by engine {target.engine.value}"]
        if not target.is_vortex:
            if target.integration is not Integration.FILE or target.scan is not None:
                return [f"{target.engine.value}:{target.format.value} takes no integration or scan engine"]
            return []
        error = scan_support_error(target.engine, target.integration, target.scan)
        if error is not None:
            return [error]
        if benchmark is not None:
            resolved = target.resolved(benchmark)
            if resolved.scan is ScanEngine.V2 and not supports_scan_v2(benchmark):
                return [f"Benchmark {benchmark.value} cannot run the v2 scan yet (list layout): {resolved}"]
            error = scan_support_error(target.engine, target.integration, resolved.scan)
            if error is not None:
                return [error]
        return []

    def is_supported(self, benchmark: Benchmark | None = None) -> bool:
        """Return whether this target can run, in `benchmark` when given."""
        return not self.support_errors(benchmark)

    def result_format(self, format_name: str, benchmark: Benchmark) -> str:
        """Return the format identity results record for this target, given the binary's label.

        The benchmark binaries label results by engine and format alone, and the benchmarks
        database keys rows on them. The file integration on the suite's default engine keeps the
        bare format, which is the identity its history was recorded under; every other combination
        appends both axes, e.g. `vortex-file-compressed [datasource, v1]`. The axes are also
        recorded as fields of their own, see `runner.labels`.
        """
        target = self.resolved(benchmark)
        if not target.is_vortex:
            return format_name
        if target.integration is Integration.FILE and target.scan is default_scan_engine(benchmark):
            return format_name
        assert target.scan is not None
        return f"{format_name} [{target.integration.value}, {target.scan.value}]"

    def to_dict(self) -> dict[str, str]:
        target = self.normalized()
        data = {"engine": target.engine.value, "format": target.format.value}
        if target.is_vortex:
            data["integration"] = target.integration.value
            if target.scan is not None:
                data["scan"] = target.scan.value
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BenchmarkTarget":
        if not isinstance(data, dict):
            raise ValueError("Targets must be JSON objects with engine and format fields")
        try:
            target = cls(
                engine=Engine(data["engine"]),
                format=Format(data["format"]),
                integration=Integration(data.get("integration", Integration.FILE.value)),
                scan=ScanEngine(data["scan"]) if data.get("scan") is not None else None,
            )
        except KeyError as exc:
            raise ValueError("Targets must include engine and format fields") from exc
        except ValueError as exc:
            raise ValueError(f"Invalid benchmark target: {data}") from exc
        return target.normalized()

    def __str__(self) -> str:
        """Render the target as `engine:format`, naming both axes for Vortex targets.

        For example `datafusion:vortex [file, v2]`, or `datafusion:vortex [file]` before the scan
        engine is resolved.
        """
        target = self.normalized()
        label = f"{target.engine.value}:{target.format.value}"
        if not target.is_vortex:
            return label
        axes = [target.integration.value] + ([target.scan.value] if target.scan is not None else [])
        return f"{label} [{', '.join(axes)}]"


def parse_targets_json(value: str) -> list[BenchmarkTarget]:
    """Parse a JSON array of explicit benchmark targets."""
    try:
        raw_targets = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError("Targets JSON must be a valid JSON array") from exc

    if not isinstance(raw_targets, list):
        raise ValueError("Targets JSON must be an array")

    return _unique_preserve_order(BenchmarkTarget.from_dict(item) for item in raw_targets)


def parse_size_formats_json(value: str) -> list[SizeFormat]:
    """Parse a JSON array of size-only format names."""
    try:
        raw_formats = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError("Size formats JSON must be a valid JSON array") from exc
    if not isinstance(raw_formats, list) or not all(isinstance(item, str) for item in raw_formats):
        raise ValueError("Size formats JSON must be an array of strings")
    try:
        return _unique_preserve_order(SizeFormat(item) for item in raw_formats)
    except ValueError as exc:
        raise ValueError(f"Invalid size format in {raw_formats}") from exc


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
            if benchmark is not None:
                target = target.resolved(benchmark)
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
        errors.extend(target.support_errors(benchmark))
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
