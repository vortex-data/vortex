# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Configuration types and enums for benchmark orchestration."""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class Engine(str, Enum):
    """Execution engines for benchmarks."""

    DUCKDB = "duckdb"
    DATAFUSION = "datafusion"
    LANCE = "lance"

    @property
    def binary_name(self) -> str:
        """Return the cargo binary name for this engine."""
        return {
            Engine.DUCKDB: "duckdb-bench",
            Engine.DATAFUSION: "datafusion-bench",
            Engine.LANCE: "lance-bench",
        }[self]


class Format(str, Enum):
    """Data formats for benchmarks."""

    PARQUET = "parquet"
    VORTEX = "vortex"
    VORTEX_COMPACT = "vortex-compact"
    DUCKDB = "duckdb"
    LANCE = "lance"


class Benchmark(str, Enum):
    """Available benchmark suites."""

    TPCH = "tpch"
    TPCDS = "tpcds"
    CLICKBENCH = "clickbench"
    FINEWEB = "fineweb"
    GHARCHIVE = "gh-archive"
    PUBLIC_BI = "public-bi"
    STATPOPGEN = "statpopgen"


# Engine to supported formats mapping
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
        Format.DUCKDB,
    ],
    Engine.LANCE: [Format.LANCE],
}


@dataclass
class RunConfig:
    """Configuration for a benchmark run."""

    benchmark: Benchmark
    engines: list[Engine]
    formats: list[Format]
    queries: list[int] | None = None
    exclude_queries: list[int] | None = None
    iterations: int = 5
    label: str | None = None
    options: dict[str, str] = field(default_factory=dict)
    track_memory: bool = False

    def validate(self) -> list[str]:
        """Validate the configuration and return any warnings."""
        warnings = []
        for engine in self.engines:
            supported = ENGINE_FORMATS.get(engine, [])
            for fmt in self.formats:
                if fmt not in supported:
                    warnings.append(f"Format {fmt.value} is not supported by engine {engine.value}")
        return warnings


@dataclass
class BuildConfig:
    """Configuration for building benchmark binaries."""

    profile: str = "release_debug"
    rustflags: str = "-C target-cpu=native -C force-frame-pointers=yes"


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
