# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

import sys
import textwrap
from pathlib import Path

from bench_orchestrator.config import Benchmark, Engine, Format
from bench_orchestrator.runner.executor import BenchmarkExecutor


def test_build_command_adds_duckdb_cleanup_flag() -> None:
    executor = BenchmarkExecutor(Path("/tmp/duckdb-bench"), Engine.DUCKDB)

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
    executor = BenchmarkExecutor(Path("/tmp/lance-bench"), Engine.LANCE)

    cmd = executor.build_command(
        benchmark=Benchmark.TPCH,
        formats=[Format.LANCE],
        queries=[1, 3],
    )

    assert cmd[0] == "/tmp/lance-bench"
    assert "--formats" not in cmd
    assert "--queries" in cmd
    assert "1,3" in cmd


def test_run_streams_logs_without_counting_them(tmp_path: Path) -> None:
    script = tmp_path / "fake-bench.py"
    script.write_text(
        textwrap.dedent(
            f"""\
            #!{sys.executable}
            import sys

            print("preparing duckdb tables", file=sys.stderr, flush=True)
            print('{{"engine":"duckdb","format":"parquet","query":0}}', flush=True)
            print("finished query 0", file=sys.stderr, flush=True)
            """
        )
    )
    script.chmod(0o755)

    executor = BenchmarkExecutor(script, Engine.DUCKDB)
    streamed: list[str] = []

    results = executor.run(
        benchmark=Benchmark.CLICKBENCH,
        formats=[Format.PARQUET],
        iterations=1,
        on_result=streamed.append,
    )

    assert results == ['{"engine":"duckdb","format":"parquet","query":0}']
    assert streamed == results
