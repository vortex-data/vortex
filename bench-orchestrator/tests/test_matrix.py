# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Contract tests for the declarative CI benchmark matrix."""

import json
from typing import cast

from bench_orchestrator import cli as cli_module
from bench_orchestrator.benchmarks import BENCHMARKS, PROFILES
from bench_orchestrator.config import Benchmark, Engine, Format
from bench_orchestrator.matrix import (
    DEFAULTS,
    BenchmarkDef,
    Profile,
    Storage,
    all_targets,
    defaults,
    df,
    duck,
    resolve_matrix,
)
from typer.testing import CliRunner

runner = CliRunner()


def _targets(entry: dict[str, object]) -> list[dict[str, str]]:
    return cast("list[dict[str, str]]", entry["targets"])


def test_default_policy_only_narrows_declared_targets() -> None:
    benchmark = BenchmarkDef(
        id="duckdb-only",
        benchmark=Benchmark.TPCH,
        name="DuckDB only",
        targets=duck(Format.VORTEX, Format.DUCKDB),
    )

    assert set(defaults(benchmark)) == set(duck(Format.VORTEX))


def test_resolver_emits_the_fields_consumed_by_the_workflow() -> None:
    benchmark = BenchmarkDef(
        id="remote",
        benchmark=Benchmark.TPCH,
        name="Remote",
        targets=df(Format.ARROW, Format.PARQUET, Format.LANCE, Format.VORTEX) | duck(Format.DUCKDB),
        storage=Storage.S3,
        scale_factor=1,
        iterations=10,
        local_dir="data/tpch",
        remote_key="tpch/1.0/",
    )

    [entry] = resolve_matrix(Profile(targets=all_targets), [benchmark])

    assert entry == {
        "id": "remote",
        "subcommand": "tpch",
        "name": "Remote",
        "targets": [
            {"engine": "datafusion", "format": "arrow"},
            {"engine": "datafusion", "format": "parquet"},
            {"engine": "datafusion", "format": "vortex"},
            {"engine": "duckdb", "format": "duckdb"},
        ],
        "data_formats": ["parquet", "vortex", "duckdb"],
        "scale_factor": "1",
        "iterations": "10",
        "local_dir": "data/tpch",
        "remote_key": "tpch/1.0/",
    }


def test_ci_profiles_have_distinct_and_consistent_roles() -> None:
    assert set(PROFILES) == {"develop", "pr", "nightly"}
    regular_ids = {benchmark.id for benchmark in BENCHMARKS if not benchmark.nightly}
    nightly_ids = {benchmark.id for benchmark in BENCHMARKS if benchmark.nightly}
    develop = {entry["id"]: entry for entry in resolve_matrix(PROFILES["develop"], BENCHMARKS)}
    pr = {entry["id"]: entry for entry in resolve_matrix(PROFILES["pr"], BENCHMARKS)}
    nightly = {entry["id"]: entry for entry in resolve_matrix(PROFILES["nightly"], BENCHMARKS)}

    assert set(develop) == regular_ids
    assert set(pr) == regular_ids
    assert set(nightly) == nightly_ids
    assert len(develop) == len([benchmark for benchmark in BENCHMARKS if not benchmark.nightly])
    assert len(nightly) == len([benchmark for benchmark in BENCHMARKS if benchmark.nightly])

    default_targets = set(DEFAULTS)
    for entry in (*pr.values(), *nightly.values()):
        targets = {(Engine(target["engine"]), Format(target["format"])) for target in _targets(entry)}
        assert targets
        assert targets <= {(target.engine, target.format) for target in default_targets}

    tpch = develop["tpch-nvme"]
    assert [(target["engine"], target["format"]) for target in _targets(tpch)] == [
        ("datafusion", "arrow"),
        ("datafusion", "parquet"),
        ("datafusion", "vortex"),
        ("datafusion", "vortex-compact"),
        ("datafusion", "lance"),
        ("duckdb", "parquet"),
        ("duckdb", "vortex"),
        ("duckdb", "vortex-compact"),
        ("duckdb", "duckdb"),
    ]


def test_existing_display_and_scale_values_are_preserved() -> None:
    develop = {entry["id"]: entry for entry in resolve_matrix(PROFILES["develop"], BENCHMARKS)}
    nightly = {entry["id"]: entry for entry in resolve_matrix(PROFILES["nightly"], BENCHMARKS)}

    assert develop["tpch-nvme"]["scale_factor"] == "1.0"
    assert develop["statpopgen"]["scale_factor"] == "100"
    assert develop["polarsignals"]["scale_factor"] == "1"
    assert nightly["tpch-nvme"]["name"] == "TPC-H on NVME"
    assert nightly["tpch-nvme"]["scale_factor"] == "100"
    assert nightly["tpch-s3"]["name"] == "TPC-H on S3"
    assert nightly["tpch-s3"]["scale_factor"] == "100.0"


def test_matrix_command_emits_json_and_rejects_unknown_profiles() -> None:
    result = runner.invoke(cli_module.app, ["matrix", "develop"])
    assert result.exit_code == 0
    assert json.loads(result.stdout)

    result = runner.invoke(cli_module.app, ["matrix", "does-not-exist"])
    assert result.exit_code == 1
