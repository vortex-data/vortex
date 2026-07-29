# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Data model for declarative CI benchmark coverage."""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum

from ..config import Benchmark, Format
from .targets import TargetSet


class Storage(Enum):
    """Where a benchmark's data lives when it runs."""

    NVME = "nvme"
    S3 = "s3"


@dataclass(frozen=True)
class Coverage:
    """Targets and generated formats for one benchmark in one preset."""

    targets: TargetSet
    data_formats: tuple[Format, ...] | None = None


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

    @property
    def subcommand(self) -> str:
        """Return the ``vx-bench`` subcommand for this benchmark."""
        return self.benchmark.value


@dataclass(frozen=True)
class Catalog:
    """Named presets and the benchmark cases available to them."""

    presets: Mapping[str, str]
    benchmarks: tuple[BenchmarkCase, ...]
