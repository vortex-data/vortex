# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

from bench_orchestrator.config import (
    Benchmark,
    BenchmarkTarget,
    Engine,
    Format,
    group_targets_by_backend,
    parse_formats_json,
    parse_targets_json,
    resolve_axis_targets,
    validate_targets,
)


def test_parse_targets_json_normalizes_and_dedupes_lance_targets() -> None:
    targets = parse_targets_json('[{"engine":"lance","format":"lance"},{"engine":"datafusion","format":"lance"}]')

    assert targets == [BenchmarkTarget(engine=Engine.DATAFUSION, format=Format.LANCE)]


def test_parse_formats_json_accepts_ci_format_arrays() -> None:
    formats = parse_formats_json('["parquet","vortex","duckdb"]')

    assert formats == [Format.PARQUET, Format.VORTEX, Format.DUCKDB]


def test_resolve_axis_targets_filters_unsupported_combinations() -> None:
    targets, warnings = resolve_axis_targets(
        [Engine.DATAFUSION, Engine.DUCKDB],
        [Format.ARROW, Format.PARQUET],
    )

    assert targets == [
        BenchmarkTarget(engine=Engine.DATAFUSION, format=Format.ARROW),
        BenchmarkTarget(engine=Engine.DATAFUSION, format=Format.PARQUET),
        BenchmarkTarget(engine=Engine.DUCKDB, format=Format.PARQUET),
    ]
    assert warnings == ["Format arrow is not supported by engine duckdb"]


def test_resolve_axis_targets_skips_engines_a_benchmark_cannot_run() -> None:
    # SpatialBench is DuckDB-only (ST_* spatial SQL), so the DataFusion axis is dropped with a warning.
    targets, warnings = resolve_axis_targets(
        [Engine.DATAFUSION, Engine.DUCKDB],
        [Format.PARQUET, Format.VORTEX],
        Benchmark.SPATIALBENCH,
    )

    assert targets == [
        BenchmarkTarget(engine=Engine.DUCKDB, format=Format.PARQUET),
        BenchmarkTarget(engine=Engine.DUCKDB, format=Format.VORTEX),
    ]
    assert warnings == ["Benchmark spatialbench does not support engine datafusion"]


def test_validate_targets_rejects_engine_a_benchmark_cannot_run() -> None:
    errors = validate_targets(
        [BenchmarkTarget(engine=Engine.DATAFUSION, format=Format.PARQUET)],
        {},
        Benchmark.SPATIALBENCH,
    )

    assert errors == ["Benchmark spatialbench does not support engine datafusion"]


def test_validate_targets_rejects_remote_lance() -> None:
    errors = validate_targets(
        [BenchmarkTarget(engine=Engine.DATAFUSION, format=Format.LANCE)],
        {"remote-data-dir": "s3://benchmarks/tpch/"},
    )

    assert errors == ["Lance format is not supported for remote storage benchmarks."]


def test_group_targets_by_backend_routes_lance_to_lance_binary() -> None:
    groups = group_targets_by_backend(
        [
            BenchmarkTarget(engine=Engine.DATAFUSION, format=Format.PARQUET),
            BenchmarkTarget(engine=Engine.DATAFUSION, format=Format.LANCE),
            BenchmarkTarget(engine=Engine.DUCKDB, format=Format.VORTEX),
        ]
    )

    assert list(groups) == [
        Engine.DATAFUSION,
        Engine.LANCE,
        Engine.DUCKDB,
    ]
    assert groups[Engine.LANCE] == [BenchmarkTarget(engine=Engine.DATAFUSION, format=Format.LANCE)]
