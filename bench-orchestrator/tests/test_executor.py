# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

from pathlib import Path

from bench_orchestrator.config import Benchmark, ExecutionBackend, Format
from bench_orchestrator.runner.executor import BenchmarkExecutor


def test_build_command_adds_duckdb_cleanup_flag() -> None:
    executor = BenchmarkExecutor(Path("/tmp/duckdb-bench"), ExecutionBackend.DUCKDB)

    cmd = executor.build_command(
        benchmark=Benchmark.TPCH,
        formats=[Format.PARQUET, Format.VORTEX],
        iterations=7,
        options={"scale-factor": "1.0"},
    )

    assert cmd[:5] == [
        "/tmp/duckdb-bench",
        "tpch",
        "--display-format",
        "gh-json",
        "--iterations",
    ]
    assert "--formats" in cmd
    assert "parquet,vortex" in cmd
    assert "--delete-duckdb-database" in cmd
    assert "--opt" in cmd
    assert "scale-factor=1.0" in cmd


def test_build_command_omits_formats_for_lance_backend() -> None:
    executor = BenchmarkExecutor(Path("/tmp/lance-bench"), ExecutionBackend.LANCE)

    cmd = executor.build_command(
        benchmark=Benchmark.TPCH,
        formats=[Format.LANCE],
        queries=[1, 3],
    )

    assert cmd[0] == "/tmp/lance-bench"
    assert "--formats" not in cmd
    assert "--queries" in cmd
    assert "1,3" in cmd
