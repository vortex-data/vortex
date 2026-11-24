# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Tests for CI benchmark matrices."""

import json
from dataclasses import replace
from typing import cast

import pytest
from bench_orchestrator import cli as cli_module
from bench_orchestrator.ci_matrix import MATRIX_PRESETS, resolve_matrix
from bench_orchestrator.ci_matrix.catalog import CATALOG
from bench_orchestrator.ci_matrix.model import Catalog, Coverage
from bench_orchestrator.ci_matrix.render import render_matrix
from bench_orchestrator.ci_matrix.selection import parse_selection
from bench_orchestrator.ci_matrix.targets import TargetSet
from bench_orchestrator.config import (
    BenchmarkTarget,
    Engine,
    Format,
    Integration,
    ScanEngine,
    SizeFormat,
    parse_targets_json,
)
from typer.testing import CliRunner

runner = CliRunner()

REGULAR_IDS = (
    "clickbench-nvme",
    "clickbench-sorted-nvme",
    "tpch-nvme",
    "tpch-s3",
    "tpch-nvme-10",
    "tpch-s3-10",
    "tpcds-nvme",
    "statpopgen",
    "fineweb",
    "fineweb-s3",
    "polarsignals",
    "appian-nvme",
    "vortex-queries",
)
COMPACT_IDS = tuple(
    benchmark_id for benchmark_id in REGULAR_IDS if benchmark_id not in {"polarsignals", "vortex-queries"}
)
DEVELOP_IDS = (*REGULAR_IDS[:-1], "job-nvme", "tpch-nested-nvme", "vortex-queries")
# TPC-DS SF=10 on NVMe runs in the PR presets and nightly, not on every develop commit.
PR_IDS = (*REGULAR_IDS[:7], "tpcds-nvme-10", *REGULAR_IDS[7:])
PR_FULL_IDS = (*PR_IDS[:-1], "job-nvme", "tpch-nested-nvme", "vortex-queries")
# The benchmark cases each preset runs; a case covering every scan runs as several matrix entries.
EXPECTED_GROUPS = {
    "develop": DEVELOP_IDS,
    "pr": tuple(
        benchmark_id
        for benchmark_id in PR_IDS
        if benchmark_id
        not in {
            "clickbench-sorted-nvme",
            "tpch-s3-10",
            "statpopgen",
            "fineweb-s3",
            "appian-nvme",
            "vortex-queries",
        }
    ),
    "pr-scan": ("clickbench-nvme", "tpch-nvme"),
    "pr-compact": COMPACT_IDS,
    "pr-all": PR_FULL_IDS,
    "pr-full": PR_FULL_IDS,
    "nightly": (
        "tpch-nvme",
        "tpch-s3",
        "tpcds-nvme-10",
        "tpcds-s3-10",
        *(
            f"tpcds-{storage}-100-{target}"
            for storage in ("nvme", "s3")
            for target in (
                "datafusion-parquet",
                "datafusion-vortex",
                "duckdb-parquet",
                "duckdb-vortex",
                "datafusion-vortex-file-v1",
                "duckdb-vortex-file-v1",
            )
        ),
        "tpch-nested-nvme",
        "tpch-nested-s3",
    ),
}
SCAN_SUFFIXES = ("datafusion-file-v1", "datafusion-datasource-v1", "duckdb-file-v1")
# Nightly skips DataFusion datasource v1, see `NIGHTLY_SKIP_SCANS`.
NIGHTLY_SCAN_SUFFIXES = ("datafusion-file-v1", "duckdb-file-v1")


def _entries(preset: str) -> list[dict[str, object]]:
    return resolve_matrix(preset)


def _targets(entry: dict[str, object]) -> set[tuple[str, str]]:
    """Return an entry's targets as `(engine, format)`, with any scan axes appended to the format."""
    targets = cast("list[dict[str, str]]", entry["targets"])
    return {
        (
            target["engine"],
            target["format"] + (f" [{target['integration']}, {target['scan']}]" if "integration" in target else ""),
        )
        for target in targets
    }


def _groups(preset: str) -> dict[str, dict[str, object]]:
    """Merge each case's matrix entries back into one: its targets and data formats."""
    groups: dict[str, dict[str, object]] = {}
    for entry in _entries(preset):
        group = groups.setdefault(str(entry["group"]), {"ids": [], "targets": set(), "data_formats": set()})
        cast("list[str]", group["ids"]).append(str(entry["id"]))
        cast("set[tuple[str, str]]", group["targets"]).update(_targets(entry))
        cast("set[str]", group["data_formats"]).update(cast("list[str]", entry["data_formats"]))
    return groups


def _group_targets(preset: str) -> dict[str, set[tuple[str, str]]]:
    return {group: cast("set[tuple[str, str]]", value["targets"]) for group, value in _groups(preset).items()}


ALL_SCANS = {
    ("datafusion", "parquet"),
    ("datafusion", "vortex [file, v2]"),
    ("datafusion", "vortex [file, v1]"),
    ("datafusion", "vortex [datasource, v1]"),
    ("duckdb", "parquet"),
    ("duckdb", "vortex [file, v2]"),
    ("duckdb", "vortex [file, v1]"),
}
NIGHTLY_SCANS = ALL_SCANS - {("datafusion", "vortex [datasource, v1]")}
DEFAULT_SCANS = {
    ("datafusion", "parquet"),
    ("datafusion", "vortex [file, v2]"),
    ("duckdb", "parquet"),
    ("duckdb", "vortex [file, v2]"),
}


@pytest.mark.parametrize(("preset", "expected_groups"), EXPECTED_GROUPS.items())
def test_matrix_presets(preset: str, expected_groups: tuple[str, ...]) -> None:
    entries = _entries(preset)
    ids = [entry["id"] for entry in entries]

    assert tuple(_groups(preset)) == expected_groups
    assert len(ids) == len(set(ids))
    assert all(entry["targets"] for entry in entries)
    assert "${{" not in json.dumps(entries)


@pytest.mark.parametrize("preset", ["pr-scan", "develop", "nightly"])
def test_every_scan_runs_on_a_machine_of_its_own(preset: str) -> None:
    groups = _groups(preset)
    entries = {entry["id"]: entry for entry in _entries(preset)}
    suffixes = NIGHTLY_SCAN_SUFFIXES if preset == "nightly" else SCAN_SUFFIXES
    scans = NIGHTLY_SCANS if preset == "nightly" else ALL_SCANS

    assert groups["tpch-nvme"]["ids"] == ["tpch-nvme", *(f"tpch-nvme-{suffix}" for suffix in suffixes)]
    assert scans <= _group_targets(preset)["tpch-nvme"]
    # The default entry keeps the case's id and history and records its data; a scan's entry
    # generates only the Vortex files it reads.
    assert entries["tpch-nvme"]["records_data"] is True
    assert ("datafusion", "vortex [file, v1]") not in _targets(entries["tpch-nvme"])
    for suffix in suffixes:
        entry = entries[f"tpch-nvme-{suffix}"]
        assert entry["group"] == "tpch-nvme"
        assert len(_targets(entry)) == 1
        assert entry["data_formats"] == ["vortex"]
        assert entry["records_data"] is False


def test_pr_scan_runs_every_scan_on_clickbench_and_tpch_nvme() -> None:
    groups = _group_targets("pr-scan")

    assert groups == {"clickbench-nvme": ALL_SCANS, "tpch-nvme": ALL_SCANS}


def test_pr_runs_only_the_default_scans() -> None:
    entries = _entries("pr")

    assert all(entry["id"] == entry["group"] for entry in entries)
    assert _group_targets("pr")["tpch-nvme"] == DEFAULT_SCANS


def test_develop_runs_every_scan_where_pr_scan_compares_against_it() -> None:
    groups = _group_targets("develop")

    for benchmark_id in ("clickbench-nvme", "tpch-nvme"):
        assert ALL_SCANS <= groups[benchmark_id]
    assert ("datafusion", "vortex [file, v1]") not in groups["tpch-nvme-10"]


@pytest.mark.parametrize("preset", ["develop", "nightly", "pr-full", "pr-all"])
def test_every_s3_entry_has_its_own_remote_key(preset: str) -> None:
    keys = [entry["remote_key"] for entry in _entries(preset) if "remote_key" in entry]

    assert len(keys) == len(set(keys))


@pytest.mark.parametrize("preset", ["develop", "nightly", "pr-full", "pr-all"])
def test_only_s3_entries_generate_parquet_with_large_row_groups(preset: str) -> None:
    entries = _entries(preset)

    assert any("remote_key" in entry for entry in entries)
    for entry in entries:
        if "remote_key" in entry:
            assert entry["row_group_rows"] == "1048576"
        else:
            assert "row_group_rows" not in entry


def test_tpcds_sf100_runs_one_target_per_job() -> None:
    nightly = [
        entry for entry in _entries("nightly") if str(entry["id"]).startswith("tpcds-") and "-100-" in str(entry["id"])
    ]

    assert len(nightly) == 12
    assert all(len(_targets(entry)) == 1 for entry in nightly)
    assert {target for entry in nightly for target in _targets(entry)} == NIGHTLY_SCANS
    s3_keys = [entry["remote_key"] for entry in nightly if "remote_key" in entry]
    assert len(s3_keys) == len(set(s3_keys)) == 6


def test_nightly_runs_every_scan_except_datasource_v1() -> None:
    groups = _group_targets("nightly")

    assert groups["tpch-s3"] == NIGHTLY_SCANS
    # Nested TPC-H reads list layouts, which the v2 scan cannot read yet.
    assert groups["tpch-nested-s3"] == {
        ("datafusion", "parquet"),
        ("datafusion", "vortex [file, v1]"),
        ("duckdb", "parquet"),
        ("duckdb", "vortex [file, v1]"),
    }


def test_pr_target_selection() -> None:
    develop = _group_targets("develop")
    pr_compact = _groups("pr-compact")
    pr_full = _group_targets("pr-full")

    assert ("datafusion", "lance") in develop["tpch-nvme"]
    for preset in (pr_full, develop):
        assert {engine for engine, _format in preset["statpopgen"]} == {"datafusion", "duckdb"}
    assert all(("datafusion", "lance") not in targets for targets in pr_full.values())
    assert "vortex-compact" in cast("set[str]", _groups("pr-full")["clickbench-nvme"]["data_formats"])
    for group in pr_compact.values():
        targets = cast("set[tuple[str, str]]", group["targets"])
        assert {file_format.split(" [", 1)[0] for _engine, file_format in targets} == {"parquet", "vortex-compact"}
        assert group["data_formats"] == {"parquet", "vortex-compact"}


@pytest.mark.parametrize("preset", ["pr", "pr-compact", "pr-full", "pr-all"])
def test_only_pr_scan_adds_scans_to_pull_requests(preset: str) -> None:
    # The extra scans cost a machine each, so pull requests opt in with `action/bench-scan`.
    assert all(entry["id"] == entry["group"] for entry in _entries(preset))


def test_nightly_runs_every_valid_scan_except_datasource_v1() -> None:
    for benchmark_id, targets in _group_targets("nightly").items():
        if "-100-" in benchmark_id:
            continue
        engines = {engine for engine, fmt in targets if fmt.startswith("vortex [")}
        scans = {(engine, fmt) for engine, fmt in targets if fmt.startswith("vortex [")}
        expected = {(engine, "vortex [file, v1]") for engine in engines}
        if not benchmark_id.startswith("tpch-nested"):
            expected |= {(engine, "vortex [file, v2]") for engine in engines}
        assert scans == expected, benchmark_id


@pytest.mark.parametrize("preset", ["develop", "nightly", "pr-full", "pr-all"])
def test_s3_local_dir_names_the_generated_scale_factor(preset: str) -> None:
    # prepare-data writes under the scale factor exactly as passed, so `100` and `100.0` differ.
    for entry in _entries(preset):
        if "local_dir" not in entry or "scale_factor" not in entry:
            continue
        numbers = [part for part in str(entry["local_dir"]).split("/") if part.replace(".", "", 1).isdigit()]
        assert numbers in ([], [entry["scale_factor"]]), entry["id"]


def test_scan_targets_round_trip_through_targets_json() -> None:
    entries = {entry["id"]: entry for entry in _entries("pr-scan")}

    (target,) = parse_targets_json(json.dumps(entries["tpch-nvme-datafusion-datasource-v1"]["targets"]))
    default = parse_targets_json(json.dumps(entries["tpch-nvme"]["targets"]))

    assert target == BenchmarkTarget(Engine.DATAFUSION, Format.VORTEX, Integration.DATASOURCE, ScanEngine.V1)
    assert BenchmarkTarget(Engine.DATAFUSION, Format.VORTEX, Integration.FILE, ScanEngine.V2) in default
    assert BenchmarkTarget(Engine.DATAFUSION, Format.PARQUET) in default


def test_pr_all_is_union_of_pr_compact_and_full() -> None:
    sources = [_groups(preset) for preset in ("pr", "pr-compact", "pr-full")]
    pr_all = _groups("pr-all")

    assert set(pr_all) == set().union(*sources)
    for benchmark_id, group in pr_all.items():
        assert group["targets"] == set().union(*(p[benchmark_id]["targets"] for p in sources if benchmark_id in p))
        for preset in sources:
            if source := preset.get(benchmark_id):
                assert cast("set[str]", source["data_formats"]) <= cast("set[str]", group["data_formats"])


def test_pr_full_covers_pr() -> None:
    pr = _groups("pr")
    pr_full = _groups("pr-full")

    for benchmark_id, group in pr.items():
        full_group = pr_full[benchmark_id]
        assert cast("set[object]", group["targets"]) <= cast("set[object]", full_group["targets"])
        assert cast("set[str]", group["data_formats"]) <= cast("set[str]", full_group["data_formats"])


def test_resolver_rejects_empty_targets() -> None:
    benchmark = replace(
        CATALOG.benchmarks[0],
        id="empty",
        runs={"develop": Coverage(TargetSet())},
    )
    catalog = Catalog(presets=CATALOG.presets, benchmarks=(benchmark,))

    with pytest.raises(ValueError, match="Benchmark 'empty' resolved to no runnable targets"):
        _ = render_matrix("develop", catalog)


def test_resolver_rejects_duplicate_ids() -> None:
    benchmark = CATALOG.benchmarks[0]
    catalog = Catalog(
        presets=CATALOG.presets,
        benchmarks=(benchmark, replace(benchmark, name="Duplicate")),
    )

    with pytest.raises(ValueError, match="Duplicate benchmark ID 'clickbench-nvme'"):
        _ = render_matrix("develop", catalog)


def test_matrix_entries_name_their_data_directory() -> None:
    entries = {entry["id"]: entry for entry in render_matrix("develop", CATALOG)}

    assert entries["clickbench-nvme"]["dataset_dir"] == "clickbench_partitioned"
    assert entries["tpch-nvme"]["dataset_dir"] == "tpch"


@pytest.mark.parametrize("preset", tuple(MATRIX_PRESETS))
def test_only_develop_tpcds_sf1_writes_size_only_formats(preset: str) -> None:
    writers = {entry["id"]: entry["size_formats"] for entry in _entries(preset) if entry["size_formats"]}

    if preset == "develop":
        assert writers == {"tpcds-nvme": ["arrow-ipc", "parquet-snappy"]}
    else:
        assert writers == {}


def test_develop_tpcds_sf1_queries_lance() -> None:
    entry = {entry["id"]: entry for entry in _entries("develop")}["tpcds-nvme"]
    targets = {(target["engine"], target["format"]) for target in entry["targets"]}

    assert ("datafusion", "lance") in targets
    assert "lance" not in entry["data_formats"]


def test_resolver_rejects_a_size_only_format_that_a_target_reads() -> None:
    benchmark = replace(
        CATALOG.benchmarks[0],
        id="reads-lance",
        runs={
            "develop": Coverage(
                TargetSet((BenchmarkTarget(Engine.DATAFUSION, Format.LANCE),)),
                size_formats=(SizeFormat.LANCE,),
            )
        },
    )
    catalog = Catalog(presets=CATALOG.presets, benchmarks=(benchmark,))

    with pytest.raises(ValueError, match="reads 'lance'.*needs no size-only write"):
        _ = render_matrix("develop", catalog)


def test_matrix_command() -> None:
    result = runner.invoke(cli_module.app, ["matrix", "develop"])
    assert result.exit_code == 0
    assert json.loads(result.stdout) == _entries("develop")

    result = runner.invoke(cli_module.app, ["matrix", "does-not-exist"])
    assert result.exit_code == 1
    assert "Unknown matrix preset" in result.stdout


def test_preset_names_are_stable() -> None:
    assert set(MATRIX_PRESETS) == set(EXPECTED_GROUPS)


@pytest.mark.parametrize("preset", MATRIX_PRESETS)
def test_only_selects_each_benchmark_and_entry(preset: str) -> None:
    entries = _entries(preset)
    # A default entry shares its benchmark's ID, which selects every scan of the benchmark.
    for entry in (entry for entry in entries if entry["id"] != entry["group"]):
        assert resolve_matrix(preset, [str(entry["id"])]) == [entry]
    for group in {str(entry["group"]) for entry in entries}:
        assert resolve_matrix(preset, [group]) == [entry for entry in entries if entry["group"] == group]


def test_only_selects_every_scan_of_a_benchmark() -> None:
    selected = resolve_matrix("pr-scan", ["tpch-nvme"])
    assert len(selected) > 1
    assert {entry["group"] for entry in selected} == {"tpch-nvme"}


def test_only_keeps_preset_order_and_ignores_empty_selection() -> None:
    assert [entry["id"] for entry in resolve_matrix("pr-all", ["polarsignals", "tpch-nvme"])] == [
        "tpch-nvme",
        "polarsignals",
    ]
    assert resolve_matrix("pr-all", []) == _entries("pr-all")


def test_only_rejects_ids_outside_preset() -> None:
    with pytest.raises(ValueError, match=r"Benchmark IDs not in matrix preset 'pr': appian-nvme, nope\."):
        _ = resolve_matrix("pr", ["polarsignals", "appian-nvme", "nope"])


@pytest.mark.parametrize("only", ["polarsignals,tpch-nvme", " polarsignals  tpch-nvme ", "tpch-nvme, polarsignals"])
def test_matrix_command_only(only: str) -> None:
    result = runner.invoke(cli_module.app, ["matrix", "pr-all", "--only", only])
    assert result.exit_code == 0
    assert json.loads(result.stdout) == resolve_matrix("pr-all", ["tpch-nvme", "polarsignals"])


def test_matrix_command_only_edge_cases() -> None:
    result = runner.invoke(cli_module.app, ["matrix", "pr-all", "--only", ""])
    assert result.exit_code == 0
    assert json.loads(result.stdout) == _entries("pr-all")

    result = runner.invoke(cli_module.app, ["matrix", "pr-all", "--only", "does-not-exist"])
    assert result.exit_code == 1
    assert "Benchmark IDs not in matrix preset" in result.stdout


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("", {}),
        ("polarsignals", {"polarsignals": ()}),
        ("tpch-nvme:q1,2", {"tpch-nvme": (1, 2)}),
        ("tpch-nvme:q1-4", {"tpch-nvme": (1, 2, 3, 4)}),
        ("tpch-nvme:Q1-q3,7", {"tpch-nvme": (1, 2, 3, 7)}),
        ("polarsignals,tpch-nvme:3 appian-nvme", {"polarsignals": (), "tpch-nvme": (3,), "appian-nvme": ()}),
        ("tpch-nvme:1 tpch-nvme:2", {"tpch-nvme": (1, 2)}),
        ("tpch-nvme:1 tpch-nvme", {"tpch-nvme": ()}),
    ],
)
def test_parse_selection(value: str, expected: dict[str, tuple[int, ...]]) -> None:
    assert parse_selection(value) == expected


@pytest.mark.parametrize("value", ["tpch-nvme:", "tpch-nvme:qx", "tpch-nvme:q4-1", "tpch-nvme:0", ":q1"])
def test_parse_selection_rejects(value: str) -> None:
    with pytest.raises(ValueError, match="Invalid"):
        _ = parse_selection(value)


def test_only_passes_queries_to_every_scan_entry() -> None:
    selected = resolve_matrix("pr-scan", {"tpch-nvme": (1, 2, 3, 4), "clickbench-nvme": ()})
    tpch = [entry for entry in selected if entry["group"] == "tpch-nvme"]
    assert len(tpch) > 1
    assert {entry["queries"] for entry in tpch} == {"1,2,3,4"}
    assert all("queries" not in entry for entry in selected if entry["group"] == "clickbench-nvme")


def test_matrix_command_only_queries() -> None:
    result = runner.invoke(cli_module.app, ["matrix", "pr-all", "--only", "polarsignals tpch-nvme:q1-3"])
    assert result.exit_code == 0
    assert [(entry["id"], entry.get("queries")) for entry in json.loads(result.stdout)] == [
        ("tpch-nvme", "1,2,3"),
        ("polarsignals", None),
    ]
