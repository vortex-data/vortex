# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Canonical declaration of what SQL benchmarks run in each CI preset.

Edit this file when changing CI benchmark coverage. The other modules in this package only model,
validate, and render these declarations.
"""

from dataclasses import replace

from ..config import Benchmark, Engine, Format
from .model import BenchmarkCase, Catalog, Coverage, Storage
from .render import merge_coverage
from .targets import df, duck, targets

PRESETS = {
    "develop": "Every regular SQL benchmark at full target coverage.",
    "pr": "The quicker pull-request SQL benchmark matrix (action/bench-sql).",
    "pr-compact": "Pull-request SQL benchmarks for Vortex Compact plus Parquet controls.",
    "pr-all": "The union of the pr, pr-compact and pr-full matrices (action/bench-all).",
    "pr-full": "Every regular SQL benchmark at full PR target coverage (action/bench-sql-extended).",
    "nightly": "Large-scale SF=100 TPC-H on NVMe and S3, and SF=10/100 TPC-DS on NVMe and S3, at default targets.",
}

# Reusable target coverage

DEFAULT_TARGETS = df(Format.PARQUET, Format.VORTEX) | duck(Format.PARQUET, Format.VORTEX)
STANDARD_TARGETS = df(Format.PARQUET, Format.VORTEX, Format.VORTEX_COMPACT) | duck(
    Format.PARQUET,
    Format.VORTEX,
    Format.VORTEX_COMPACT,
)
FULL_LOCAL_TARGETS = df(
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
FULL_PR_TARGETS = df(
    Format.PARQUET,
    Format.VORTEX,
    Format.VORTEX_COMPACT,
) | duck(
    Format.PARQUET,
    Format.VORTEX,
    Format.VORTEX_COMPACT,
    Format.DUCKDB,
)
DEFAULT_WITH_DUCKDB_TARGETS = DEFAULT_TARGETS | duck(Format.DUCKDB)
STANDARD_WITH_DUCKDB_TARGETS = STANDARD_TARGETS | duck(Format.DUCKDB)
DATAFUSION_VORTEX_TARGETS = df(Format.VORTEX)
COMPACT_TARGETS = df(Format.PARQUET, Format.VORTEX_COMPACT) | duck(
    Format.PARQUET,
    Format.VORTEX_COMPACT,
)

DEFAULT = Coverage(DEFAULT_TARGETS)
STANDARD = Coverage(STANDARD_TARGETS)
FULL_LOCAL = Coverage(FULL_LOCAL_TARGETS)
FULL_PR = Coverage(
    FULL_PR_TARGETS,
    data_formats=(Format.PARQUET, Format.VORTEX, Format.VORTEX_COMPACT, Format.DUCKDB),
)
DEFAULT_WITH_DUCKDB_PR_FULL = Coverage(
    DEFAULT_WITH_DUCKDB_TARGETS,
    data_formats=(Format.PARQUET, Format.VORTEX, Format.VORTEX_COMPACT, Format.DUCKDB),
)
STANDARD_WITH_DUCKDB = Coverage(STANDARD_WITH_DUCKDB_TARGETS)
DATAFUSION_VORTEX = Coverage(DATAFUSION_VORTEX_TARGETS)
COMPACT = Coverage(COMPACT_TARGETS)
JOB_DUCKDB = Coverage(duck(Format.PARQUET, Format.VORTEX, Format.DUCKDB))

# Concrete benchmark cases

# TPC-DS SF=100 runs one engine/format target per job, each on its own machine, so a single slow
# target cannot push the others past the job timeout and lose their results.
TPCDS_SF100_TARGETS = (
    (Engine.DATAFUSION, Format.PARQUET, "DataFusion", "Parquet"),
    (Engine.DATAFUSION, Format.VORTEX, "DataFusion", "Vortex"),
    (Engine.DUCKDB, Format.PARQUET, "DuckDB", "Parquet"),
    (Engine.DUCKDB, Format.VORTEX, "DuckDB", "Vortex"),
)
TPCDS_SF100_CASES = tuple(
    BenchmarkCase(
        id=f"tpcds-{storage.value}-100-{engine.value}-{fmt.value}",
        benchmark=Benchmark.TPCDS,
        name=f"TPC-DS SF=100 on {storage.value.upper()} ({engine_name}, {fmt_name})",
        storage=storage,
        scale_factor=100.0,
        iterations=3,
        local_dir="vortex-bench/data/tpcds/100.0" if storage is Storage.S3 else None,
        # A distinct key per job: each job clears and re-uploads its own copy of the data.
        remote_key=f"tpcds/100.0-{engine.value}-{fmt.value}" if storage is Storage.S3 else None,
        runs={"nightly": Coverage(targets(engine, fmt))},
    )
    for storage in (Storage.NVME, Storage.S3)
    for engine, fmt, engine_name, fmt_name in TPCDS_SF100_TARGETS
)

BENCHMARKS = (
    BenchmarkCase(
        id="clickbench-nvme",
        benchmark=Benchmark.CLICKBENCH,
        name="Clickbench on NVME",
        dataset_dir="clickbench_partitioned",
        runs={
            "pr": DEFAULT,
            "pr-compact": COMPACT,
            "pr-full": DEFAULT_WITH_DUCKDB_PR_FULL,
            "develop": FULL_LOCAL,
        },
    ),
    BenchmarkCase(
        id="clickbench-sorted-nvme",
        benchmark=Benchmark.CLICKBENCH_SORTED,
        name="Clickbench Sorted on NVME",
        runs={
            "pr-compact": COMPACT,
            "pr-full": DEFAULT_WITH_DUCKDB_PR_FULL,
            "develop": FULL_LOCAL,
        },
    ),
    BenchmarkCase(
        id="tpch-nvme",
        benchmark=Benchmark.TPCH,
        name="TPC-H SF=1 on NVME",
        scale_factor=1.0,
        iterations=10,
        runs={
            "pr": DEFAULT,
            "pr-compact": COMPACT,
            "pr-full": FULL_PR,
            "develop": FULL_LOCAL,
        },
    ),
    BenchmarkCase(
        id="tpch-s3",
        benchmark=Benchmark.TPCH,
        name="TPC-H SF=1 on S3",
        storage=Storage.S3,
        scale_factor=1.0,
        iterations=10,
        local_dir="vortex-bench/data/tpch/1.0",
        remote_key="tpch/1.0",
        runs={
            "pr": DEFAULT,
            "pr-compact": COMPACT,
            "pr-full": STANDARD,
            "develop": STANDARD,
        },
    ),
    BenchmarkCase(
        id="tpch-nvme-10",
        benchmark=Benchmark.TPCH,
        name="TPC-H SF=10 on NVME",
        scale_factor=10.0,
        iterations=10,
        runs={
            "pr": DEFAULT,
            "pr-compact": COMPACT,
            "pr-full": FULL_PR,
            "develop": FULL_LOCAL,
        },
    ),
    BenchmarkCase(
        id="tpch-s3-10",
        benchmark=Benchmark.TPCH,
        name="TPC-H SF=10 on S3",
        storage=Storage.S3,
        scale_factor=10.0,
        iterations=10,
        local_dir="vortex-bench/data/tpch/10.0",
        remote_key="tpch/10.0",
        runs={
            "pr-compact": COMPACT,
            "pr-full": STANDARD,
            "develop": STANDARD,
        },
    ),
    BenchmarkCase(
        id="tpch-nvme",
        benchmark=Benchmark.TPCH,
        name="TPC-H on NVME",
        scale_factor=100,
        runs={"nightly": DEFAULT},
    ),
    BenchmarkCase(
        id="tpch-s3",
        benchmark=Benchmark.TPCH,
        name="TPC-H on S3",
        storage=Storage.S3,
        scale_factor=100.0,
        local_dir="vortex-bench/data/tpch/100.0",
        remote_key="tpch/100.0",
        runs={"nightly": DEFAULT},
    ),
    BenchmarkCase(
        id="tpcds-nvme",
        benchmark=Benchmark.TPCDS,
        name="TPC-DS SF=1 on NVME",
        scale_factor=1.0,
        runs={
            "pr": DEFAULT,
            "pr-compact": COMPACT,
            "pr-full": STANDARD_WITH_DUCKDB,
            "develop": STANDARD_WITH_DUCKDB,
        },
    ),
    BenchmarkCase(
        id="tpcds-nvme-10",
        benchmark=Benchmark.TPCDS,
        name="TPC-DS SF=10 on NVME",
        scale_factor=10.0,
        runs={"nightly": DEFAULT},
    ),
    BenchmarkCase(
        id="tpcds-s3-10",
        benchmark=Benchmark.TPCDS,
        name="TPC-DS SF=10 on S3",
        storage=Storage.S3,
        scale_factor=10.0,
        local_dir="vortex-bench/data/tpcds/10.0",
        remote_key="tpcds/10.0",
        runs={"nightly": DEFAULT},
    ),
    *TPCDS_SF100_CASES,
    BenchmarkCase(
        id="statpopgen",
        benchmark=Benchmark.STATPOPGEN,
        name="Statistical and Population Genetics",
        scale_factor=100,
        local_dir="vortex-bench/data/statpopgen",
        runs={
            "pr-compact": COMPACT,
            "pr-full": STANDARD,
            "develop": STANDARD,
        },
    ),
    BenchmarkCase(
        id="fineweb",
        benchmark=Benchmark.FINEWEB,
        name="FineWeb NVMe",
        scale_factor=100,
        runs={
            "pr": DEFAULT,
            "pr-compact": COMPACT,
            "pr-full": STANDARD,
            "develop": STANDARD,
        },
    ),
    BenchmarkCase(
        id="fineweb-s3",
        benchmark=Benchmark.FINEWEB,
        name="FineWeb S3",
        storage=Storage.S3,
        scale_factor=100,
        local_dir="vortex-bench/data/fineweb",
        remote_key="fineweb",
        runs={
            "pr-compact": COMPACT,
            "pr-full": STANDARD,
            "develop": STANDARD,
        },
    ),
    BenchmarkCase(
        id="polarsignals",
        benchmark=Benchmark.POLARSIGNALS,
        name="PolarSignals Profiling",
        scale_factor=1,
        runs={
            "pr": DATAFUSION_VORTEX,
            "pr-full": DATAFUSION_VORTEX,
            "develop": DATAFUSION_VORTEX,
        },
    ),
    BenchmarkCase(
        id="appian-nvme",
        benchmark=Benchmark.APPIAN,
        name="Appian on NVME",
        iterations=10,
        runs={
            "pr-compact": COMPACT,
            "pr-full": DEFAULT_WITH_DUCKDB_PR_FULL,
            "develop": STANDARD_WITH_DUCKDB,
        },
    ),
    BenchmarkCase(
        id="job-nvme",
        benchmark=Benchmark.JOB,
        name="JOB on NVME",
        # DuckDB only: DataFusion spends minutes per pass on the 17x queries' join orders.
        runs={"pr-full": JOB_DUCKDB},
    ),
    BenchmarkCase(
        id="vortex-queries",
        benchmark=Benchmark.VORTEX_QUERIES,
        name="Vortex queries",
        iterations=100,
        runs={
            "pr-full": DEFAULT,
            "develop": DEFAULT,
        },
    ),
)

PR_ALL_SOURCES = ("pr", "pr-compact", "pr-full")


def _with_pr_all(benchmark: BenchmarkCase) -> BenchmarkCase:
    sources = [benchmark.runs[preset] for preset in PR_ALL_SOURCES if preset in benchmark.runs]
    if not sources:
        return benchmark
    return replace(benchmark, runs={**benchmark.runs, "pr-all": merge_coverage(sources)})


CATALOG = Catalog(presets=PRESETS, benchmarks=tuple(map(_with_pr_all, BENCHMARKS)))
