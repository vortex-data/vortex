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
        """Return the cargo binary name for this engine when used to execute benchmarks."""
        return {
            Engine.DUCKDB: "duckdb-bench",
            Engine.DATAFUSION: "datafusion-bench",
            Engine.LANCE: "lance-bench",
        }[self]


class Format(Enum):
    """Data formats for benchmarks."""

    ARROW = "arrow"
    PARQUET = "parquet"
    VORTEX = "vortex"
    VORTEX_COMPACT = "vortex-compact"
    DUCKDB = "duckdb"
    LANCE = "lance"


class Benchmark(Enum):
    """Available benchmark suites."""

    APPIAN = "appian"
    TPCH = "tpch"
    TPCDS = "tpcds"
    CLICKBENCH = "clickbench"
    CLICKBENCH_SORTED = "clickbench-sorted"
    FINEWEB = "fineweb"
    GHARCHIVE = "gh-archive"
    POLARSIGNALS = "polarsignals"
    PUBLIC_BI = "public-bi"
    STATPOPGEN = "statpopgen"


# Engine to supported formats mapping.
ENGINE_FORMATS: dict[Engine, list[Format]] = {
    Engine.DATAFUSION: [
        Format.ARROW,
        Format.PARQUET,
        Format.VORTEX,
        Format.VORTEX_COMPACT,
        Format.LANCE,
    ],
    Engine.DUCKDB: [
        Format.PARQUET,
        Format.VORTEX,
        Format.VORTEX_COMPACT,
        Format.DUCKDB,
    ],
    Engine.LANCE: [Format.LANCE],
}

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
    """An explicit engine/format benchmark target."""

    engine: Engine
    format: Format

    def normalized(self) -> "BenchmarkTarget":
        """Normalize legacy lance targets onto the logical CI identity."""
        if self.engine == Engine.LANCE and self.format == Format.LANCE:
            return BenchmarkTarget(engine=Engine.DATAFUSION, format=Format.LANCE)
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
        """Return whether the logical engine supports this format."""
        target = self.normalized()
        return target.format in ENGINE_FORMATS.get(target.engine, [])

    def to_dict(self) -> dict[str, str]:
        target = self.normalized()
        return {"engine": target.engine.value, "format": target.format.value}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BenchmarkTarget":
        if not isinstance(data, dict):
            raise ValueError("Targets must be JSON objects with engine and format fields")
        try:
            target = cls(
                engine=Engine(data["engine"]),
                format=Format(data["format"]),
            )
        except KeyError as exc:
            raise ValueError("Targets must include engine and format fields") from exc
        except ValueError as exc:
            raise ValueError(f"Invalid benchmark target: {data}") from exc
        return target.normalized()

    def __str__(self) -> str:
        target = self.normalized()
        return f"{target.engine.value}:{target.format.value}"


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
    engines: Iterable[Engine], formats: Iterable[Format]
) -> tuple[list[BenchmarkTarget], list[str]]:
    """Expand engine/format axes into supported explicit targets."""
    warnings: list[str] = []
    targets: list[BenchmarkTarget] = []

    for engine in engines:
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


def validate_targets(targets: Iterable[BenchmarkTarget], options: dict[str, str]) -> list[str]:
    """Validate explicit targets against benchmark runner constraints."""
    errors: list[str] = []

    normalized_targets = [target.normalized() for target in targets]
    for target in normalized_targets:
        if not target.is_supported():
            errors.append(f"Format {target.format.value} is not supported by engine {target.engine.value}")

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
        return validate_targets(self.targets, self.options)


@dataclass
class BuildConfig:
    """Configuration for building benchmark binaries."""

    profile: str = "release_debug"
    rustflags: str = "-C target-cpu=native -C force-frame-pointers=yes"
    features: tuple[str, ...] = ("unstable_encodings",)


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
