# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Tests for CI benchmark matrices."""

import json
from dataclasses import replace
from typing import cast

import pytest
from bench_orchestrator import cli as cli_module
from bench_orchestrator.benchmarks import BENCHMARKS
from bench_orchestrator.matrix import MATRIX_PRESETS, TargetSet, resolve_matrix
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
EXPECTED_IDS = {
    "develop": REGULAR_IDS,
    "pr": tuple(
        benchmark_id
        for benchmark_id in REGULAR_IDS
        if benchmark_id not in {"tpch-s3-10", "appian-nvme", "vortex-queries"}
    ),
    "pr-full": REGULAR_IDS,
    "nightly": ("tpch-nvme", "tpch-s3"),
}


def _entries(preset: str) -> list[dict[str, object]]:
    return resolve_matrix(preset, BENCHMARKS)


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


def test_resolver_rejects_empty_targets() -> None:
    benchmark = replace(BENCHMARKS[0], id="empty", targets=TargetSet())

    with pytest.raises(ValueError, match="Benchmark 'empty' resolved to no runnable targets"):
        _ = resolve_matrix("develop", [benchmark])


def test_resolver_rejects_duplicate_ids() -> None:
    benchmark = BENCHMARKS[0]

    with pytest.raises(ValueError, match="Duplicate benchmark ID 'clickbench-nvme'"):
        _ = resolve_matrix("develop", [benchmark, replace(benchmark, name="Duplicate")])


def test_matrix_command() -> None:
    result = runner.invoke(cli_module.app, ["matrix", "develop"])
    assert result.exit_code == 0
    assert json.loads(result.stdout) == _entries("develop")

    result = runner.invoke(cli_module.app, ["matrix", "does-not-exist"])
    assert result.exit_code == 1
    assert "Unknown matrix preset" in result.stdout


def test_preset_names_are_stable() -> None:
    assert set(MATRIX_PRESETS) == set(EXPECTED_IDS)
