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
EXPECTED_IDS = {
    "develop": REGULAR_IDS,
    "pr": tuple(
        benchmark_id
        for benchmark_id in REGULAR_IDS
        if benchmark_id not in {"tpch-s3-10", "appian-nvme", "vortex-queries"}
    ),
    "pr-compact": COMPACT_IDS,
    "pr-full": REGULAR_IDS,
    "nightly": ("tpch-nvme", "tpch-s3"),
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
    assert all(("datafusion", "lance") not in _targets(entry) for entry in pr_full.values())
    assert "vortex-compact" in cast("list[str]", pr_full["clickbench-nvme"]["data_formats"])
    for entry in pr_compact.values():
        targets = _targets(entry)
        assert {file_format for _engine, file_format in targets} == {"parquet", "vortex-compact"}
        assert set(cast("list[str]", entry["data_formats"])) == {"parquet", "vortex-compact"}


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


def test_matrix_command() -> None:
    result = runner.invoke(cli_module.app, ["matrix", "develop"])
    assert result.exit_code == 0
    assert json.loads(result.stdout) == _entries("develop")

    result = runner.invoke(cli_module.app, ["matrix", "does-not-exist"])
    assert result.exit_code == 1
    assert "Unknown matrix preset" in result.stdout


def test_preset_names_are_stable() -> None:
    assert set(MATRIX_PRESETS) == set(EXPECTED_IDS)
