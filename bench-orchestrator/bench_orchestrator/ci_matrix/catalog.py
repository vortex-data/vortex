# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Canonical declaration of what SQL benchmarks run in each CI preset.

Edit this file when changing CI benchmark coverage. The other modules in this package only model,
validate, and render these declarations.
"""

from ..config import Benchmark, Engine, Format
from .model import BenchmarkCase, Catalog, Coverage, Storage
from .targets import df, duck

PRESETS = {
    "develop": "Every regular SQL benchmark at full target coverage.",
    "pr": "The quicker pull-request SQL benchmark matrix.",
    "pr-full": "Every regular SQL benchmark at full PR target coverage.",
    "nightly": "Large-scale SF=100 TPC-H on NVMe and S3 at default targets.",
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
DUCKDB_DEFAULT_TARGETS = DEFAULT_TARGETS.only(Engine.DUCKDB)
DUCKDB_STANDARD_TARGETS = STANDARD_TARGETS.only(Engine.DUCKDB)
DATAFUSION_VORTEX_TARGETS = df(Format.VORTEX)

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
DUCKDB_DEFAULT = Coverage(DUCKDB_DEFAULT_TARGETS)
DUCKDB_STANDARD = Coverage(DUCKDB_STANDARD_TARGETS)
DATAFUSION_VORTEX = Coverage(DATAFUSION_VORTEX_TARGETS)

# Concrete benchmark cases

BENCHMARKS = (
    BenchmarkCase(
        id="clickbench-nvme",
        benchmark=Benchmark.CLICKBENCH,
        name="Clickbench on NVME",
        runs={
            "pr": DEFAULT,
            "pr-full": DEFAULT_WITH_DUCKDB_PR_FULL,
            "develop": FULL_LOCAL,
        },
    ),
    BenchmarkCase(
        id="clickbench-sorted-nvme",
        benchmark=Benchmark.CLICKBENCH_SORTED,
        name="Clickbench Sorted on NVME",
        runs={
            "pr": DEFAULT,
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
            "pr-full": STANDARD_WITH_DUCKDB,
            "develop": STANDARD_WITH_DUCKDB,
        },
    ),
    BenchmarkCase(
        id="statpopgen",
        benchmark=Benchmark.STATPOPGEN,
        name="Statistical and Population Genetics",
        scale_factor=100,
        local_dir="vortex-bench/data/statpopgen",
        runs={
            "pr": DUCKDB_DEFAULT,
            "pr-full": DUCKDB_STANDARD,
            "develop": DUCKDB_STANDARD,
        },
    ),
    BenchmarkCase(
        id="fineweb",
        benchmark=Benchmark.FINEWEB,
        name="FineWeb NVMe",
        scale_factor=100,
        runs={
            "pr": DEFAULT,
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
            "pr": DEFAULT,
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
            "pr-full": DEFAULT_WITH_DUCKDB_PR_FULL,
            "develop": STANDARD_WITH_DUCKDB,
        },
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

CATALOG = Catalog(presets=PRESETS, benchmarks=BENCHMARKS)
