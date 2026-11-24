# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Canonical declaration of what SQL benchmarks run in each CI preset.

Edit this file when changing CI benchmark coverage. The other modules in this package only model,
validate, and render these declarations.
"""

from dataclasses import replace

from ..config import Benchmark, BenchmarkTarget, Engine, Format, Integration, ScanEngine, SizeFormat
from .model import BenchmarkCase, Catalog, Coverage, ScanCoverage, Storage
from .render import merge_coverage
from .targets import TargetSet, df, duck

PRESETS = {
    "develop": (
        "Every regular SQL benchmark at full target coverage, and every Vortex scan on ClickBench and "
        "TPC-H SF=1 on NVMe."
    ),
    "pr": "The quicker pull-request SQL benchmark matrix (action/bench-sql).",
    "pr-compact": "Pull-request SQL benchmarks for Vortex Compact plus Parquet controls.",
    "pr-scan": (
        "ClickBench and TPC-H SF=1 on NVMe with every Vortex integration and scan engine, one machine "
        "per combination (action/bench-scan)."
    ),
    "pr-all": "The union of the pr, pr-compact and pr-full matrices (action/bench-all).",
    "pr-full": ("Every regular SQL benchmark at full PR target coverage (action/bench-sql-extended)."),
    "nightly": (
        "SF=100 TPC-H and nested TPC-H, and SF=10/100 TPC-DS, on NVMe and S3, with every Vortex scan "
        "except DataFusion datasource v1."
    ),
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

# Vortex targets name an integration (`file`, `datasource`) and a scan engine (`v1`, `v2`); see
# `BenchmarkTarget`. Declared targets run the file integration on the suite's default engine.
# `ScanCoverage.ALL` adds every other valid combination for the `vortex` format: DataFusion file-v1 and
# datasource-v1 beside file-v2, and DuckDB file-v1 beside file-v2, on V2 suites. Each combination runs
# as a matrix entry, and so a machine, of its own, so covering them all adds machines, not wall time.


def with_all_scans(coverage: Coverage, skip: tuple[tuple[Integration, ScanEngine], ...] = ()) -> Coverage:
    """Run every valid integration and scan engine combination for the `vortex` format, except `skip`."""
    return replace(coverage, scans=ScanCoverage.ALL, skip_scans=skip)


# Nightly leaves out DataFusion's datasource integration on the v1 scan, the legacy code path. At
# SF=10 and SF=100 it hangs on TPC-DS q62 until the job times out, and the runner shuts down during
# TPC-H q18. Nightly run 44843664966 lost six jobs to it.
NIGHTLY_SKIP_SCANS = ((Integration.DATASOURCE, ScanEngine.V1),)


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
# TPC-DS SF1 is the SQL data that bench.vortex.dev's File Sizes summary ranks, so develop also
# records the sizes of Lance, Arrow IPC, and Snappy Parquet there, the formats compress-bench also
# records. It queries Lance on DataFusion, and writes Arrow IPC and Snappy Parquet only for their
# sizes. All three are small and fast at SF1.
STANDARD_WITH_DUCKDB_AND_SIZE_FORMATS = Coverage(
    STANDARD_WITH_DUCKDB_TARGETS | df(Format.LANCE),
    size_formats=(SizeFormat.ARROW_IPC, SizeFormat.PARQUET_SNAPPY),
)
DATAFUSION_VORTEX = Coverage(DATAFUSION_VORTEX_TARGETS)
COMPACT = Coverage(COMPACT_TARGETS)
JOB_DUCKDB = Coverage(duck(Format.PARQUET, Format.VORTEX, Format.DUCKDB))

# Concrete benchmark cases

# TPC-DS SF=100 runs one target per job, each on its own machine, so a single slow target cannot
# push the others past the job timeout and lose their results. It leaves out `NIGHTLY_SKIP_SCANS`.
TPCDS_SF100_TARGETS = (
    (BenchmarkTarget(Engine.DATAFUSION, Format.PARQUET), "DataFusion, Parquet"),
    (BenchmarkTarget(Engine.DATAFUSION, Format.VORTEX), "DataFusion, Vortex"),
    (BenchmarkTarget(Engine.DUCKDB, Format.PARQUET), "DuckDB, Parquet"),
    (BenchmarkTarget(Engine.DUCKDB, Format.VORTEX), "DuckDB, Vortex"),
    (
        BenchmarkTarget(Engine.DATAFUSION, Format.VORTEX, Integration.FILE, ScanEngine.V1),
        "DataFusion, Vortex, file v1",
    ),
    (
        BenchmarkTarget(Engine.DUCKDB, Format.VORTEX, Integration.FILE, ScanEngine.V1),
        "DuckDB, Vortex, file v1",
    ),
)


def _tpcds_sf100_job(target: BenchmarkTarget) -> str:
    """Name a TPC-DS SF=100 job after its target; default-scan targets keep their original names."""
    job = f"{target.engine.value}-{target.format.value}"
    if target.scan is not None:
        job += f"-{target.integration.value}-{target.scan.value}"
    return job


TPCDS_SF100_CASES = tuple(
    BenchmarkCase(
        id=f"tpcds-{storage.value}-100-{_tpcds_sf100_job(target)}",
        benchmark=Benchmark.TPCDS,
        name=f"TPC-DS SF=100 on {storage.value.upper()} ({label})",
        storage=storage,
        scale_factor=100.0,
        iterations=3,
        local_dir="vortex-bench/data/tpcds/100.0" if storage is Storage.S3 else None,
        # A distinct key per job: each job clears and re-uploads its own copy of the data.
        remote_key=f"tpcds/100.0-{_tpcds_sf100_job(target)}" if storage is Storage.S3 else None,
        runs={"nightly": Coverage(TargetSet((target,)))},
    )
    for storage in (Storage.NVME, Storage.S3)
    for target, label in TPCDS_SF100_TARGETS
)

BENCHMARKS = (
    BenchmarkCase(
        id="clickbench-nvme",
        benchmark=Benchmark.CLICKBENCH,
        name="Clickbench on NVME",
        dataset_dir="clickbench_partitioned",
        runs={
            "pr": DEFAULT,
            "pr-scan": with_all_scans(DEFAULT),
            "pr-compact": COMPACT,
            "pr-full": DEFAULT_WITH_DUCKDB_PR_FULL,
            "develop": with_all_scans(FULL_LOCAL),
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
            "pr-scan": with_all_scans(DEFAULT),
            "pr-compact": COMPACT,
            "pr-full": FULL_PR,
            "develop": with_all_scans(FULL_LOCAL),
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
            "develop": STANDARD_WITH_DUCKDB_AND_SIZE_FORMATS,
        },
    ),
    BenchmarkCase(
        id="tpcds-nvme-10",
        benchmark=Benchmark.TPCDS,
        name="TPC-DS SF=10 on NVME",
        scale_factor=10.0,
        runs={
            "pr": DEFAULT,
            "pr-full": DEFAULT,
            "nightly": DEFAULT,
        },
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
        runs={
            "pr-full": JOB_DUCKDB,
            "develop": JOB_DUCKDB,
        },
    ),
    BenchmarkCase(
        id="tpch-nested-nvme",
        benchmark=Benchmark.TPCH_NESTED,
        name="Nested TPC-H SF=1 on NVME",
        scale_factor=1.0,
        iterations=10,
        runs={
            "pr-full": FULL_PR,
            "develop": FULL_LOCAL,
        },
    ),
    BenchmarkCase(
        id="tpch-nested-nvme",
        benchmark=Benchmark.TPCH_NESTED,
        name="Nested TPC-H on NVME",
        scale_factor=100,
        runs={"nightly": DEFAULT},
    ),
    BenchmarkCase(
        id="tpch-nested-s3",
        benchmark=Benchmark.TPCH_NESTED,
        name="Nested TPC-H on S3",
        storage=Storage.S3,
        scale_factor=100,
        # `scale_factor=100` generates the data under `100`, not `100.0`.
        local_dir="vortex-bench/data/tpch-nested/100/list",
        remote_key="tpch-nested/100/list",
        runs={"nightly": DEFAULT},
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

# Presets whose every benchmark runs every valid integration and scan engine combination, except
# the combinations each one skips. TPC-DS SF=100 lists its targets one per job already, see
# `TPCDS_SF100_TARGETS`. PR presets other than `pr-scan` keep the default scans, so the extra
# machines are opt-in.
ALL_SCAN_PRESETS = {"nightly": NIGHTLY_SKIP_SCANS}


def _with_all_scan_presets(benchmark: BenchmarkCase) -> BenchmarkCase:
    if benchmark in TPCDS_SF100_CASES:
        return benchmark
    runs = {
        preset: with_all_scans(coverage, ALL_SCAN_PRESETS[preset]) if preset in ALL_SCAN_PRESETS else coverage
        for preset, coverage in benchmark.runs.items()
    }
    return replace(benchmark, runs=runs)


PR_ALL_SOURCES = ("pr", "pr-compact", "pr-full")


def _with_pr_all(benchmark: BenchmarkCase) -> BenchmarkCase:
    sources = [benchmark.runs[preset] for preset in PR_ALL_SOURCES if preset in benchmark.runs]
    if not sources:
        return benchmark
    return replace(benchmark, runs={**benchmark.runs, "pr-all": merge_coverage(benchmark.benchmark, sources)})


CATALOG = Catalog(
    presets=PRESETS,
    benchmarks=tuple(_with_pr_all(_with_all_scan_presets(benchmark)) for benchmark in BENCHMARKS),
)
