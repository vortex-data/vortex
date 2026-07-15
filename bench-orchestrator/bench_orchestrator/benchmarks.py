# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""The SQL benchmarks and profiles run by CI.

This is the file to edit when changing benchmark coverage:

* ``BENCHMARKS`` declares which benchmark suites exist and every target they support.
* ``PROFILES`` declares how much of that coverage each CI workflow runs.

Matrix rendering lives in :mod:`bench_orchestrator.matrix` and should not need to change when a
benchmark is added or removed.
"""

from .config import Benchmark, Engine, Format
from .matrix import STANDARD, BenchmarkDef, Profile, Storage, all_targets, defaults, df, duck


def _tpch(scale_factor: float | int, storage: Storage, *, iterations: int | None = 10) -> BenchmarkDef:
    # Nightly historically uses the same matrix IDs as SF=1. The profiles are disjoint, so the
    # duplicate IDs never occur in one resolved matrix and historical result series stay intact.
    suffix = "" if scale_factor in {1, 100} else f"-{int(scale_factor)}"
    if storage is Storage.NVME:
        target_set = df(
            Format.ARROW,
            Format.PARQUET,
            Format.VORTEX,
            Format.VORTEX_COMPACT,
            Format.LANCE,
        ) | duck(
            Format.PARQUET,
            Format.VORTEX,
            Format.VORTEX_COMPACT,
            Format.DUCKDB,
        )
        local_dir = None
        remote_key = None
    else:
        target_set = STANDARD
        local_dir = f"vortex-bench/data/tpch/{scale_factor:.1f}"
        remote_key = f"tpch/{scale_factor:.1f}/"

    name = f"TPC-H on {storage.label}" if scale_factor == 100 else f"TPC-H SF={scale_factor:g} on {storage.label}"
    return BenchmarkDef(
        id=f"tpch-{storage.value}{suffix}",
        benchmark=Benchmark.TPCH,
        name=name,
        targets=target_set,
        storage=storage,
        scale_factor=scale_factor,
        iterations=iterations,
        nightly=scale_factor == 100,
        local_dir=local_dir,
        remote_key=remote_key,
    )


def _clickbench(benchmark: Benchmark, name: str) -> BenchmarkDef:
    return BenchmarkDef(
        id=f"{benchmark.value}-nvme",
        benchmark=benchmark,
        name=name,
        targets=df(
            Format.PARQUET,
            Format.VORTEX,
            Format.VORTEX_COMPACT,
            Format.LANCE,
        )
        | duck(
            Format.PARQUET,
            Format.VORTEX,
            Format.VORTEX_COMPACT,
            Format.DUCKDB,
        ),
    )


def _fineweb(storage: Storage) -> BenchmarkDef:
    if storage is Storage.NVME:
        return BenchmarkDef(
            id="fineweb",
            benchmark=Benchmark.FINEWEB,
            name="FineWeb NVMe",
            targets=STANDARD,
            scale_factor=100,
        )
    return BenchmarkDef(
        id="fineweb-s3",
        benchmark=Benchmark.FINEWEB,
        name="FineWeb S3",
        targets=STANDARD,
        storage=Storage.S3,
        scale_factor=100,
        local_dir="vortex-bench/data/fineweb",
        remote_key="fineweb/",
    )


# ==================================================================================================
# BENCHMARK DEFINITIONS — ADD OR REMOVE SQL BENCHMARKS HERE
# ==================================================================================================

BENCHMARKS: list[BenchmarkDef] = [
    _clickbench(Benchmark.CLICKBENCH, "Clickbench on NVME"),
    _clickbench(Benchmark.CLICKBENCH_SORTED, "Clickbench Sorted on NVME"),
    _tpch(1.0, Storage.NVME),
    _tpch(1.0, Storage.S3),
    _tpch(10.0, Storage.NVME),
    _tpch(10.0, Storage.S3),
    _tpch(100, Storage.NVME, iterations=None),  # nightly only
    _tpch(100.0, Storage.S3, iterations=None),  # nightly only
    BenchmarkDef(
        id="tpcds-nvme",
        benchmark=Benchmark.TPCDS,
        name="TPC-DS SF=1 on NVME",
        targets=STANDARD | duck(Format.DUCKDB),
        scale_factor=1.0,
    ),
    BenchmarkDef(
        id="statpopgen",
        benchmark=Benchmark.STATPOPGEN,
        name="Statistical and Population Genetics",
        targets=STANDARD.only(Engine.DUCKDB),
        scale_factor=100,
        local_dir="vortex-bench/data/statpopgen",
    ),
    _fineweb(Storage.NVME),
    _fineweb(Storage.S3),
    BenchmarkDef(
        id="polarsignals",
        benchmark=Benchmark.POLARSIGNALS,
        name="PolarSignals Profiling",
        targets=df(Format.VORTEX),
        scale_factor=1,
    ),
    BenchmarkDef(
        id="appian-nvme",
        benchmark=Benchmark.APPIAN,
        name="Appian on NVME",
        targets=STANDARD | duck(Format.DUCKDB),
        iterations=10,
    ),
]


# ==================================================================================================
# CI PROFILES — CHOOSE WHICH DECLARED COVERAGE EACH WORKFLOW RUNS HERE
# ==================================================================================================

PROFILES: dict[str, Profile] = {
    "develop": Profile(
        targets=all_targets,
        description="Every SQL benchmark at full target coverage; runs on each merge to develop.",
    ),
    "pr": Profile(
        targets=defaults,
        description="Every SQL benchmark at default targets (datafusion,duckdb × parquet,vortex).",
    ),
    "nightly": Profile(
        nightly=True,
        targets=defaults,
        description="Large-scale SF=100 TPC-H on NVMe and S3 at default targets.",
    ),
}
