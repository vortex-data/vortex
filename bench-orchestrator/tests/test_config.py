# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

import pytest
from bench_orchestrator.config import (
    BENCHMARK_SCAN_V1,
    Benchmark,
    BenchmarkTarget,
    Engine,
    Format,
    Integration,
    group_targets_by_backend,
    parse_formats_json,
    parse_targets_json,
    resolve_axis_targets,
    uses_scan_v2,
    validate_targets,
)

SCAN_API = BenchmarkTarget(engine=Engine.DATAFUSION, format=Format.VORTEX, integration=Integration.SCAN_API)


def test_parse_targets_json_normalizes_and_dedupes_lance_targets() -> None:
    targets = parse_targets_json('[{"engine":"lance","format":"lance"},{"engine":"datafusion","format":"lance"}]')

    assert targets == [BenchmarkTarget(engine=Engine.DATAFUSION, format=Format.LANCE)]


def test_parse_formats_json_accepts_ci_format_arrays() -> None:
    formats = parse_formats_json('["parquet","vortex","duckdb"]')

    assert formats == [Format.PARQUET, Format.VORTEX, Format.DUCKDB]


def test_parse_formats_json_accepts_vortex_spatial_native() -> None:
    formats = parse_formats_json('["parquet","vortex","vortex-spatial-native"]')

    assert formats == [Format.PARQUET, Format.VORTEX, Format.VORTEX_SPATIAL_NATIVE]


def test_resolve_axis_targets_offers_vortex_spatial_native_on_duckdb_only() -> None:
    # vortex-spatial-native is a DuckDB-only lane; the DataFusion axis is dropped as unsupported.
    targets, warnings = resolve_axis_targets(
        [Engine.DATAFUSION, Engine.DUCKDB],
        [Format.VORTEX_SPATIAL_NATIVE],
    )

    assert targets == [BenchmarkTarget(engine=Engine.DUCKDB, format=Format.VORTEX_SPATIAL_NATIVE)]
    assert warnings == ["Format vortex-spatial-native is not supported by engine datafusion"]


def test_resolve_axis_targets_filters_unsupported_combinations() -> None:
    targets, warnings = resolve_axis_targets(
        [Engine.DATAFUSION, Engine.DUCKDB],
        [Format.LANCE, Format.PARQUET],
    )

    assert targets == [
        BenchmarkTarget(engine=Engine.DATAFUSION, format=Format.LANCE),
        BenchmarkTarget(engine=Engine.DATAFUSION, format=Format.PARQUET),
        BenchmarkTarget(engine=Engine.DUCKDB, format=Format.PARQUET),
    ]
    assert warnings == ["Format lance is not supported by engine duckdb"]


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


def test_resolve_axis_targets_supports_spatialbench_three_lanes() -> None:
    # The single-command three-lane comparison: parquet, WKB vortex, and native-geometry vortex, all
    # on DuckDB.
    targets, warnings = resolve_axis_targets(
        [Engine.DUCKDB],
        [Format.PARQUET, Format.VORTEX, Format.VORTEX_SPATIAL_NATIVE],
        Benchmark.SPATIALBENCH,
    )

    assert targets == [
        BenchmarkTarget(engine=Engine.DUCKDB, format=Format.PARQUET),
        BenchmarkTarget(engine=Engine.DUCKDB, format=Format.VORTEX),
        BenchmarkTarget(engine=Engine.DUCKDB, format=Format.VORTEX_SPATIAL_NATIVE),
    ]
    assert warnings == []


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


def test_datafusion_package_and_binary_names_match() -> None:
    assert Engine.DATAFUSION.package_name == "datafusion-bench"
    assert Engine.DATAFUSION.binary_name == "datafusion-bench"


def test_default_integration_keeps_the_existing_target_identity() -> None:
    target = BenchmarkTarget(engine=Engine.DATAFUSION, format=Format.VORTEX)

    assert target.to_dict() == {"engine": "datafusion", "format": "vortex"}
    assert str(target) == "datafusion:vortex"
    assert target.result_format("vortex-file-compressed") == "vortex-file-compressed"


def test_scan_api_target_identity() -> None:
    assert SCAN_API.to_dict() == {"engine": "datafusion", "format": "vortex", "integration": "scan-api"}
    assert str(SCAN_API) == "datafusion:vortex [scan-api]"
    assert SCAN_API.result_format("vortex-file-compressed") == "vortex-file-compressed [scan-api]"
    assert parse_targets_json(
        '[{"engine":"datafusion","format":"vortex"},{"engine":"datafusion","format":"vortex","integration":"scan-api"}]'
    ) == [BenchmarkTarget(engine=Engine.DATAFUSION, format=Format.VORTEX), SCAN_API]


def test_parse_targets_json_rejects_unknown_integration() -> None:
    with pytest.raises(ValueError, match="Invalid benchmark target"):
        _ = parse_targets_json('[{"engine":"datafusion","format":"vortex","integration":"bogus"}]')


@pytest.mark.parametrize(
    ("engine", "fmt", "supported"),
    [
        (Engine.DATAFUSION, Format.VORTEX, True),
        (Engine.DATAFUSION, Format.VORTEX_COMPACT, True),
        (Engine.DATAFUSION, Format.PARQUET, False),
        # DuckDB has no `DataSource` table function yet.
        (Engine.DUCKDB, Format.VORTEX, False),
    ],
)
def test_scan_api_supported_targets(engine: Engine, fmt: Format, supported: bool) -> None:
    assert BenchmarkTarget(engine=engine, format=fmt, integration=Integration.SCAN_API).is_supported() is supported


def test_validate_targets_rejects_unsupported_integration() -> None:
    errors = validate_targets(
        [BenchmarkTarget(engine=Engine.DUCKDB, format=Format.VORTEX, integration=Integration.SCAN_API)], {}
    )

    assert errors == ["Integration scan-api does not support duckdb:vortex"]


@pytest.mark.parametrize("benchmark", list(Benchmark))
def test_scan_api_always_runs_the_v1_scan(benchmark: Benchmark) -> None:
    assert uses_scan_v2(benchmark) is (benchmark not in BENCHMARK_SCAN_V1)
    assert uses_scan_v2(benchmark, Integration.SCAN_API) is False
    assert SCAN_API.uses_scan_v2(benchmark) is False
