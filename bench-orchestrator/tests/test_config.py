# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

import pytest
from bench_orchestrator.config import (
    BENCHMARKS_WITHOUT_SCAN_V2,
    Benchmark,
    BenchmarkTarget,
    Engine,
    Format,
    Integration,
    ScanEngine,
    default_scan_engine,
    group_targets_by_backend,
    parse_formats_json,
    parse_targets_json,
    resolve_axis_targets,
    validate_targets,
)

DATASOURCE_V1 = BenchmarkTarget(Engine.DATAFUSION, Format.VORTEX, Integration.DATASOURCE, ScanEngine.V1)


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

    # Vortex targets resolve to the suite's default scan.
    assert targets == [
        BenchmarkTarget(engine=Engine.DUCKDB, format=Format.PARQUET),
        BenchmarkTarget(Engine.DUCKDB, Format.VORTEX, Integration.FILE, ScanEngine.V2),
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
        BenchmarkTarget(Engine.DUCKDB, Format.VORTEX, Integration.FILE, ScanEngine.V2),
        BenchmarkTarget(Engine.DUCKDB, Format.VORTEX_SPATIAL_NATIVE, Integration.FILE, ScanEngine.V2),
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


def test_scan_axes_name_the_target() -> None:
    assert BenchmarkTarget(Engine.DATAFUSION, Format.VORTEX).to_dict() == {
        "engine": "datafusion",
        "format": "vortex",
        "integration": "file",
    }
    assert DATASOURCE_V1.to_dict() == {
        "engine": "datafusion",
        "format": "vortex",
        "integration": "datasource",
        "scan": "v1",
    }
    assert str(DATASOURCE_V1) == "datafusion:vortex [datasource, v1]"
    assert str(BenchmarkTarget(Engine.DATAFUSION, Format.VORTEX)) == "datafusion:vortex [file]"
    assert str(BenchmarkTarget(Engine.DATAFUSION, Format.VORTEX).resolved(Benchmark.TPCH)) == (
        "datafusion:vortex [file, v2]"
    )
    assert str(BenchmarkTarget(Engine.DATAFUSION, Format.PARQUET).resolved(Benchmark.TPCH)) == "datafusion:parquet"
    assert parse_targets_json(
        '[{"engine":"datafusion","format":"vortex"},'
        '{"engine":"datafusion","format":"vortex","integration":"datasource","scan":"v1"}]'
    ) == [BenchmarkTarget(Engine.DATAFUSION, Format.VORTEX), DATASOURCE_V1]


@pytest.mark.parametrize(
    ("benchmark", "target", "result_format"),
    [
        # The file integration on the suite's default scan keeps the format its history is under.
        (Benchmark.TPCH, BenchmarkTarget(Engine.DATAFUSION, Format.VORTEX), "vortex-file-compressed"),
        (
            Benchmark.TPCH,
            BenchmarkTarget(Engine.DUCKDB, Format.VORTEX, Integration.FILE, ScanEngine.V2),
            "vortex-file-compressed",
        ),
        (Benchmark.TPCH_NESTED, BenchmarkTarget(Engine.DUCKDB, Format.VORTEX), "vortex-file-compressed"),
        (
            Benchmark.TPCH,
            BenchmarkTarget(Engine.DATAFUSION, Format.VORTEX, Integration.FILE, ScanEngine.V1),
            "vortex-file-compressed [file, v1]",
        ),
        (Benchmark.TPCH, DATASOURCE_V1, "vortex-file-compressed [datasource, v1]"),
        (Benchmark.TPCH_NESTED, DATASOURCE_V1, "vortex-file-compressed [datasource, v1]"),
        (Benchmark.TPCH, BenchmarkTarget(Engine.DATAFUSION, Format.PARQUET), "parquet"),
    ],
)
def test_result_format_keeps_history_for_the_default_scan(
    benchmark: Benchmark, target: BenchmarkTarget, result_format: str
) -> None:
    assert target.result_format(result_format.split(" [", 1)[0], benchmark) == result_format


def test_parse_targets_json_rejects_unknown_axes() -> None:
    with pytest.raises(ValueError, match="Invalid benchmark target"):
        _ = parse_targets_json('[{"engine":"datafusion","format":"vortex","integration":"scan-api"}]')
    with pytest.raises(ValueError, match="Invalid benchmark target"):
        _ = parse_targets_json('[{"engine":"datafusion","format":"vortex","scan":"v3"}]')


@pytest.mark.parametrize(
    ("engine", "integration", "scan", "error"),
    [
        (Engine.DATAFUSION, Integration.FILE, ScanEngine.V1, None),
        (Engine.DATAFUSION, Integration.FILE, ScanEngine.V2, None),
        (Engine.DATAFUSION, Integration.DATASOURCE, ScanEngine.V1, None),
        (Engine.DUCKDB, Integration.FILE, ScanEngine.V1, None),
        (Engine.DUCKDB, Integration.FILE, ScanEngine.V2, None),
        (Engine.DATAFUSION, Integration.DATASOURCE, ScanEngine.V2, "datasource targets run only the v1 scan"),
        (Engine.DUCKDB, Integration.DATASOURCE, ScanEngine.V1, "duckdb has no DataSource table function yet"),
        (Engine.DUCKDB, Integration.DATASOURCE, ScanEngine.V2, "duckdb has no DataSource table function yet"),
    ],
)
def test_scan_combinations(engine: Engine, integration: Integration, scan: ScanEngine, error: str | None) -> None:
    errors = validate_targets([BenchmarkTarget(engine, Format.VORTEX, integration, scan)], {})

    if error is None:
        assert errors == []
    else:
        (message,) = errors
        assert error in message


def test_non_vortex_targets_take_no_scan_axes() -> None:
    errors = validate_targets([BenchmarkTarget(Engine.DATAFUSION, Format.PARQUET, Integration.DATASOURCE)], {})

    assert errors == ["datafusion:parquet takes no integration or scan engine"]


@pytest.mark.parametrize("benchmark", list(Benchmark))
def test_v2_scan_is_the_default_where_the_suite_supports_it(benchmark: Benchmark) -> None:
    file_v2 = BenchmarkTarget(Engine.DATAFUSION, Format.VORTEX, Integration.FILE, ScanEngine.V2)
    expected = ScanEngine.V1 if benchmark in BENCHMARKS_WITHOUT_SCAN_V2 else ScanEngine.V2

    assert default_scan_engine(benchmark) is expected
    assert BenchmarkTarget(Engine.DATAFUSION, Format.VORTEX).resolved(benchmark).scan is expected
    assert file_v2.is_supported(benchmark) is (expected is ScanEngine.V2)


def test_validate_targets_rejects_v2_on_list_layouts() -> None:
    errors = validate_targets(
        [BenchmarkTarget(Engine.DUCKDB, Format.VORTEX, Integration.FILE, ScanEngine.V2)],
        {},
        benchmark=Benchmark.TPCH_NESTED,
    )

    assert errors == ["Benchmark tpch-nested cannot run the v2 scan yet (list layout): duckdb:vortex [file, v2]"]
