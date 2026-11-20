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
from bench_orchestrator.ci_matrix.targets import TargetSet
from bench_orchestrator.config import BenchmarkTarget, Engine, Format, Integration, ScanEngine, parse_targets_json
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
DEVELOP_IDS = (
    "clickbench-nvme",
    "clickbench-nvme-scans",
    "clickbench-sorted-nvme",
    "tpch-nvme",
    "tpch-nvme-scans",
    *REGULAR_IDS[3:-1],
    "job-nvme",
    "tpch-nested-nvme",
    "vortex-queries",
)
# TPC-DS SF=10 on NVMe runs in the PR presets and nightly, not on every develop commit.
PR_IDS = (*REGULAR_IDS[:7], "tpcds-nvme-10", *REGULAR_IDS[7:])
PR_FULL_IDS = (*PR_IDS[:-1], "job-nvme", "tpch-nested-nvme", "vortex-queries")
EXPECTED_IDS = {
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
    "pr-compact": COMPACT_IDS,
    "pr-all": PR_FULL_IDS,
    "pr-full": PR_FULL_IDS,
    "nightly": (
        "tpch-nvme",
        "tpch-nvme-scans",
        "tpch-s3",
        "tpch-s3-scans",
        "tpcds-nvme-10",
        "tpcds-nvme-10-scans",
        "tpcds-s3-10",
        "tpcds-s3-10-scans",
        *(
            f"tpcds-{storage}-100-{target}"
            for storage in ("nvme", "s3")
            for target in (
                "datafusion-parquet",
                "datafusion-vortex",
                "duckdb-parquet",
                "duckdb-vortex",
                "datafusion-vortex-file-v1",
                "datafusion-vortex-datasource-v1",
                "duckdb-vortex-file-v1",
            )
        ),
        "tpch-nested-nvme",
        "tpch-nested-nvme-scans",
        "tpch-nested-s3",
        "tpch-nested-s3-scans",
    ),
}


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


ALL_SCANS = {
    ("datafusion", "parquet"),
    ("datafusion", "vortex [file, v2]"),
    ("datafusion", "vortex [file, v1]"),
    ("datafusion", "vortex [datasource, v1]"),
    ("duckdb", "parquet"),
    ("duckdb", "vortex [file, v2]"),
    ("duckdb", "vortex [file, v1]"),
}
OTHER_SCANS = {("datafusion", "vortex [datasource, v1]")}


@pytest.mark.parametrize(("preset", "expected_ids"), EXPECTED_IDS.items())
def test_matrix_presets(preset: str, expected_ids: tuple[str, ...]) -> None:
    entries = _entries(preset)
    ids = [entry["id"] for entry in entries]

    assert tuple(ids) == expected_ids
    assert len(ids) == len(set(ids))
    assert all(entry["targets"] for entry in entries)
    assert "${{" not in json.dumps(entries)


def test_tpcds_sf100_runs_one_target_per_job() -> None:
    nightly = [
        entry for entry in _entries("nightly") if str(entry["id"]).startswith("tpcds-") and "-100-" in str(entry["id"])
    ]

    assert len(nightly) == 14
    assert all(len(_targets(entry)) == 1 for entry in nightly)
    assert {target for entry in nightly for target in _targets(entry)} == ALL_SCANS
    s3_keys = [entry["remote_key"] for entry in nightly if "remote_key" in entry]
    assert len(s3_keys) == len(set(s3_keys)) == 7


def test_nightly_runs_every_scan_with_its_other_scans_in_their_own_jobs() -> None:
    nightly = {entry["id"]: entry for entry in _entries("nightly")}

    assert _targets(nightly["tpch-nvme"]) | _targets(nightly["tpch-nvme-scans"]) == ALL_SCANS
    assert {fmt for _engine, fmt in _targets(nightly["tpch-nvme-scans"])} == {
        "vortex [file, v1]",
        "vortex [datasource, v1]",
    }
    # Nested TPC-H reads list layouts, which the v2 scan cannot read yet.
    assert _targets(nightly["tpch-nested-s3-scans"]) == {("datafusion", "vortex [datasource, v1]")}
    assert nightly["tpch-s3-scans"]["remote_key"] != nightly["tpch-s3"]["remote_key"]
    # Every S3 job clears and re-uploads its own copy of the data.
    s3_keys = [entry["remote_key"] for entry in nightly.values() if "remote_key" in entry]
    assert len(s3_keys) == len(set(s3_keys))


def test_pr_target_selection() -> None:
    develop = {entry["id"]: entry for entry in _entries("develop")}
    pr = {entry["id"]: entry for entry in _entries("pr")}
    pr_compact = {entry["id"]: entry for entry in _entries("pr-compact")}
    pr_full = {entry["id"]: entry for entry in _entries("pr-full")}

    for benchmark_id in ("clickbench-nvme", "tpch-nvme"):
        assert _targets(pr[benchmark_id]) == ALL_SCANS
        # develop runs the other scans in a job of their own, so a failure cannot lose the defaults.
        assert ALL_SCANS <= _targets(develop[benchmark_id]) | _targets(develop[f"{benchmark_id}-scans"])
        assert OTHER_SCANS.isdisjoint(_targets(develop[benchmark_id]))
    assert _targets(pr["tpch-nvme-10"]) == {
        ("datafusion", "parquet"),
        ("datafusion", "vortex [file, v2]"),
        ("duckdb", "parquet"),
        ("duckdb", "vortex [file, v2]"),
    }
    assert ("datafusion", "lance") in _targets(develop["tpch-nvme"])
    for preset in (pr_full, develop):
        assert {engine for engine, _format in _targets(preset["statpopgen"])} == {"datafusion", "duckdb"}
    assert all(("datafusion", "lance") not in _targets(entry) for entry in pr_full.values())
    assert "vortex-compact" in cast("list[str]", pr_full["clickbench-nvme"]["data_formats"])
    for entry in pr_compact.values():
        targets = _targets(entry)
        assert {file_format.split(" [", 1)[0] for _engine, file_format in targets} == {"parquet", "vortex-compact"}
        assert set(cast("list[str]", entry["data_formats"])) == {"parquet", "vortex-compact"}


OTHER_SCAN_IDS = {
    "pr": {"clickbench-nvme", "tpch-nvme"},
    "develop": {"clickbench-nvme-scans", "tpch-nvme-scans"},
    "pr-compact": set(),
}


@pytest.mark.parametrize(("preset", "expected_ids"), OTHER_SCAN_IDS.items())
def test_other_scans_run_on_clickbench_and_tpch_nvme(preset: str, expected_ids: set[str]) -> None:
    entries = _entries(preset)

    assert {entry["id"] for entry in entries if OTHER_SCANS & _targets(entry)} == expected_ids


@pytest.mark.parametrize("preset", ["pr-full", "pr-all"])
def test_pr_full_runs_every_valid_scan(preset: str) -> None:
    for entry in _entries(preset):
        targets = cast("list[dict[str, str]]", entry["targets"])
        engines = {target["engine"] for target in targets if target["format"] == "vortex"}
        scans = {
            (target["engine"], target["integration"], target["scan"])
            for target in targets
            if target["format"] == "vortex"
        }
        expected = {(engine, "file", "v1") for engine in engines} | (
            {("datafusion", "datasource", "v1")} if "datafusion" in engines else set()
        )
        if entry["id"] not in {"statpopgen", "polarsignals", "tpch-nested-nvme"}:
            expected |= {(engine, "file", "v2") for engine in engines}
        assert scans == expected, entry["id"]


def test_scan_targets_round_trip_through_targets_json() -> None:
    (entry,) = [entry for entry in _entries("pr") if entry["id"] == "tpch-nvme"]

    targets = parse_targets_json(json.dumps(entry["targets"]))

    assert BenchmarkTarget(Engine.DATAFUSION, Format.VORTEX, Integration.DATASOURCE, ScanEngine.V1) in targets
    assert BenchmarkTarget(Engine.DATAFUSION, Format.VORTEX, Integration.FILE, ScanEngine.V2) in targets
    assert BenchmarkTarget(Engine.DATAFUSION, Format.PARQUET) in targets
    assert {"engine": "datafusion", "format": "parquet"} in cast("list[dict[str, str]]", entry["targets"])


def test_pr_all_is_union_of_pr_compact_and_full() -> None:
    sources = [{entry["id"]: entry for entry in _entries(preset)} for preset in ("pr", "pr-compact", "pr-full")]
    pr_all = {entry["id"]: entry for entry in _entries("pr-all")}

    assert set(pr_all) == set().union(*sources)
    for benchmark_id, entry in pr_all.items():
        assert _targets(entry) == set().union(*(_targets(p[benchmark_id]) for p in sources if benchmark_id in p))
        for preset in sources:
            if source := preset.get(benchmark_id):
                assert set(cast("list[str]", source["data_formats"])) <= set(cast("list[str]", entry["data_formats"]))


def test_pr_full_covers_pr() -> None:
    pr = {entry["id"]: entry for entry in _entries("pr")}
    pr_full = {entry["id"]: entry for entry in _entries("pr-full")}

    for benchmark_id, entry in pr.items():
        full_entry = pr_full[benchmark_id]
        assert _targets(entry) <= _targets(full_entry)
        assert set(cast("list[str]", entry["data_formats"])) <= set(cast("list[str]", full_entry["data_formats"]))


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


def test_matrix_command() -> None:
    result = runner.invoke(cli_module.app, ["matrix", "develop"])
    assert result.exit_code == 0
    assert json.loads(result.stdout) == _entries("develop")

    result = runner.invoke(cli_module.app, ["matrix", "does-not-exist"])
    assert result.exit_code == 1
    assert "Unknown matrix preset" in result.stdout


def test_preset_names_are_stable() -> None:
    assert set(MATRIX_PRESETS) == set(EXPECTED_IDS)
