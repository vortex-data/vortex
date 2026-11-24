# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Data model for declarative CI benchmark coverage."""

from collections.abc import Mapping
from dataclasses import dataclass, replace
from enum import Enum

from ..config import SCAN_COMBINATIONS, Benchmark, BenchmarkTarget, Format, Integration, ScanEngine, SizeFormat
from .targets import TargetSet


class Storage(Enum):
    """Where a benchmark's data lives when it runs."""

    NVME = "nvme"
    S3 = "s3"


class ScanCoverage(Enum):
    """Which integration and scan engine combinations a coverage runs for its Vortex targets."""

    # Each Vortex target as declared: the file integration on the suite's default scan engine,
    # unless the target names others.
    DEFAULT = "default"
    # The declared targets plus every other valid combination for the `vortex` format. Each
    # combination runs as a matrix entry of its own, see `render`.
    ALL = "all"


@dataclass(frozen=True)
class Coverage:
    """Targets and generated formats for one benchmark in one preset."""

    targets: TargetSet
    data_formats: tuple[Format, ...] | None = None
    scans: ScanCoverage = ScanCoverage.DEFAULT
    # Formats written after the queries only to record their file sizes. No query reads them.
    size_formats: tuple[SizeFormat, ...] = ()
    # Integration and scan engine combinations that `ScanCoverage.ALL` does not add.
    skip_scans: tuple[tuple[Integration, ScanEngine], ...] = ()

    def resolved_targets(self, benchmark: Benchmark) -> TargetSet:
        """Return the targets this coverage runs in `benchmark`, with every scan engine named.

        Combinations the suite cannot run, such as v2 on a list-layout suite, are left out of the
        added ones; a declared target the suite cannot run stays in and fails validation.
        """
        resolved = [target.resolved(benchmark) for target in self.targets]
        if self.scans is ScanCoverage.DEFAULT:
            return TargetSet(tuple(resolved))

        alternates: list[BenchmarkTarget] = []
        for target in resolved:
            if target.format is not Format.VORTEX:
                continue
            for integration, scan in SCAN_COMBINATIONS:
                if (integration, scan) in self.skip_scans:
                    continue
                candidate = replace(target, integration=integration, scan=scan)
                if candidate not in resolved and candidate.is_supported(benchmark):
                    alternates.append(candidate)
        return TargetSet(tuple(resolved) + tuple(alternates))


@dataclass(frozen=True)
class BenchmarkCase:
    """One concrete benchmark case and the presets that schedule it."""

    id: str
    benchmark: Benchmark
    name: str
    runs: Mapping[str, Coverage]
    storage: Storage = Storage.NVME
    scale_factor: float | int | None = None
    iterations: int | None = None
    local_dir: str | None = None
    remote_key: str | None = None
    dataset_dir: str | None = None

    @property
    def data_dir_name(self) -> str:
        """Return the directory under ``vortex-bench/data`` holding this benchmark's data."""
        return self.dataset_dir or self.subcommand

    @property
    def subcommand(self) -> str:
        """Return the ``vx-bench`` subcommand for this benchmark."""
        return self.benchmark.value


@dataclass(frozen=True)
class Catalog:
    """Named presets and the benchmark cases available to them."""

    presets: Mapping[str, str]
    benchmarks: tuple[BenchmarkCase, ...]
