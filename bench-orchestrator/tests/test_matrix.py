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
PR_FULL_IDS = (*REGULAR_IDS[:-1], "job-nvme", "vortex-queries")
EXPECTED_IDS = {
    "develop": REGULAR_IDS,
    "pr": tuple(
        benchmark_id
        for benchmark_id in REGULAR_IDS
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
        "tpch-s3",
        "tpcds-nvme-10",
        "tpcds-s3-10",
        *(
            f"tpcds-{storage}-100-{target}"
            for storage in ("nvme", "s3")
            for target in ("datafusion-parquet", "datafusion-vortex", "duckdb-parquet", "duckdb-vortex")
        ),
    ),
}


def _entries(preset: str) -> list[dict[str, object]]:
    return resolve_matrix(preset)


def _targets(entry: dict[str, object]) -> set[tuple[str, str]]:
    targets = cast("list[dict[str, str]]", entry["targets"])
    return {(target["engine"], target["format"]) for target in targets}


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

    assert len(nightly) == 8
    assert all(len(_targets(entry)) == 1 for entry in nightly)
    s3_keys = [entry["remote_key"] for entry in nightly if "remote_key" in entry]
    assert len(s3_keys) == len(set(s3_keys)) == 4


def test_pr_target_selection() -> None:
    develop = {entry["id"]: entry for entry in _entries("develop")}
    pr = {entry["id"]: entry for entry in _entries("pr")}
    pr_compact = {entry["id"]: entry for entry in _entries("pr-compact")}
    pr_full = {entry["id"]: entry for entry in _entries("pr-full")}

    assert _targets(pr["tpch-nvme"]) == {
        ("datafusion", "parquet"),
        ("datafusion", "vortex"),
        ("duckdb", "parquet"),
        ("duckdb", "vortex"),
    }
    assert ("datafusion", "lance") in _targets(develop["tpch-nvme"])
    for preset in (pr_full, develop):
        assert {engine for engine, _format in _targets(preset["statpopgen"])} == {"datafusion", "duckdb"}
    assert all(("datafusion", "lance") not in _targets(entry) for entry in pr_full.values())
    assert "vortex-compact" in cast("list[str]", pr_full["clickbench-nvme"]["data_formats"])
    for entry in pr_compact.values():
        targets = _targets(entry)
        assert {file_format for _engine, file_format in targets} == {"parquet", "vortex-compact"}
        assert set(cast("list[str]", entry["data_formats"])) == {"parquet", "vortex-compact"}


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


@pytest.mark.parametrize("preset", MATRIX_PRESETS)
def test_only_selects_each_benchmark(preset: str) -> None:
    entries = _entries(preset)
    for entry in entries:
        assert resolve_matrix(preset, [str(entry["id"])]) == [entry]


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
