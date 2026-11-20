# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

import json
import os
import sys
import textwrap
from pathlib import Path

import pytest
from bench_orchestrator.config import BENCHMARK_SCAN_V1, Benchmark, Engine, Format, Integration
from bench_orchestrator.runner.executor import BenchmarkExecutor


def test_build_command_adds_duckdb_cleanup_flag() -> None:
    executor = BenchmarkExecutor(Path("/tmp/duckdb-bench"), Engine.DUCKDB)

    cmd = executor.build_command(
        benchmark=Benchmark.TPCH,
        formats=[Format.PARQUET, Format.VORTEX],
        iterations=7,
        options={"scale-factor": "1.0"},
    )

    assert cmd[:7] == [
        "/tmp/duckdb-bench",
        "tpch",
        "--display-format",
        "gh-json",
        "--iterations",
        "7",
        "--hide-progress-bar",
    ]
    assert cmd[cmd.index("--formats") + 1] == "parquet,vortex"
    assert "--delete-duckdb-database" in cmd
    assert cmd[cmd.index("--opt") + 1] == "scale-factor=1.0"


def test_build_command_serializes_vortex_spatial_native_format() -> None:
    executor = BenchmarkExecutor(Path("/tmp/duckdb-bench"), Engine.DUCKDB)

    cmd = executor.build_command(
        benchmark=Benchmark.SPATIALBENCH,
        formats=[Format.PARQUET, Format.VORTEX, Format.VORTEX_SPATIAL_NATIVE],
        iterations=1,
        options={"scale-factor": "1.0"},
    )

    assert cmd[cmd.index("--formats") + 1] == "parquet,vortex,vortex-spatial-native"


def test_scan_v1_benchmarks_are_explicit() -> None:
    assert BENCHMARK_SCAN_V1 == {Benchmark.POLARSIGNALS, Benchmark.STATPOPGEN, Benchmark.TPCH_NESTED}


@pytest.mark.parametrize("engine", [Engine.DATAFUSION, Engine.DUCKDB])
@pytest.mark.parametrize("benchmark", list(Benchmark))
def test_build_command_selects_scan_version(engine: Engine, benchmark: Benchmark) -> None:
    executor = BenchmarkExecutor(Path("/tmp/bench"), engine)

    cmd = executor.build_command(benchmark=benchmark, formats=[Format.VORTEX])

    expected = "true" if benchmark in BENCHMARK_SCAN_V1 else "false"
    assert cmd[cmd.index("--scan-v1") + 1] == expected


def test_build_command_omits_formats_for_lance_backend() -> None:
    executor = BenchmarkExecutor(Path("/tmp/lance-bench"), Engine.LANCE)

    cmd = executor.build_command(
        benchmark=Benchmark.TPCH,
        formats=[Format.LANCE],
        query=3,
    )

    assert cmd[0] == "/tmp/lance-bench"
    assert "--formats" not in cmd
    assert "--scan-v1" not in cmd
    assert cmd[cmd.index("--queries") + 1] == "3"


def test_build_command_includes_ingest_output_when_set() -> None:
    executor = BenchmarkExecutor(Path("/tmp/duckdb-bench"), Engine.DUCKDB)

    cmd = executor.build_command(
        benchmark=Benchmark.TPCH,
        formats=[Format.PARQUET],
        ingest_output=Path("results.ingest.jsonl"),
    )

    assert "--ingest-jsonl" in cmd
    flag_idx = cmd.index("--ingest-jsonl")
    assert cmd[flag_idx + 1] == "results.ingest.jsonl"


def test_build_command_omits_ingest_output_when_unset() -> None:
    executor = BenchmarkExecutor(Path("/tmp/duckdb-bench"), Engine.DUCKDB)

    cmd = executor.build_command(
        benchmark=Benchmark.TPCH,
        formats=[Format.PARQUET],
    )

    assert "--ingest-jsonl" not in cmd


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


@pytest.mark.parametrize("benchmark", [Benchmark.CLICKBENCH, Benchmark.TPCH])
def test_scan_api_runs_vortex_table_on_the_v1_scan(benchmark: Benchmark, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VORTEX_USE_SCAN_API", raising=False)
    executor = BenchmarkExecutor(Path("/tmp/datafusion-bench"), Engine.DATAFUSION)

    cmd = executor.build_command(benchmark=benchmark, formats=[Format.VORTEX], integration=Integration.SCAN_API)
    env = executor.build_env(Integration.SCAN_API)

    # `datafusion-bench` takes `VortexTable` only with both: the variable alone runs FileOpener.
    assert cmd[cmd.index("--scan-v1") + 1] == "true"
    assert env["VORTEX_USE_SCAN_API"] == "1"


def test_default_integration_clears_a_stray_scan_api_export(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VORTEX_USE_SCAN_API", "1")
    executor = BenchmarkExecutor(Path("/tmp/datafusion-bench"), Engine.DATAFUSION)

    env = executor.build_env()

    assert "VORTEX_USE_SCAN_API" not in env
    assert env["PATH"] == os.environ["PATH"]


def test_scan_api_is_rejected_by_backends_without_it() -> None:
    executor = BenchmarkExecutor(Path("/tmp/duckdb-bench"), Engine.DUCKDB)

    with pytest.raises(ValueError, match="duckdb cannot run the scan-api integration"):
        _ = executor.build_command(benchmark=Benchmark.TPCH, formats=[Format.VORTEX], integration=Integration.SCAN_API)


def test_run_passes_the_integration_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VORTEX_USE_SCAN_API", raising=False)
    script = tmp_path / "fake-bench.py"
    script.write_text(
        textwrap.dedent(
            f"""\
            #!{sys.executable}
            import json, os, sys

            print(json.dumps({{"scan_api": os.environ.get("VORTEX_USE_SCAN_API"), "args": sys.argv[1:]}}))
            """
        )
    )
    script.chmod(0o755)
    executor = BenchmarkExecutor(script, Engine.DATAFUSION)

    (line,) = executor.run(
        benchmark=Benchmark.TPCH, formats=[Format.VORTEX], iterations=1, integration=Integration.SCAN_API
    )

    record = json.loads(line)
    assert record["scan_api"] == "1"
    assert record["args"][record["args"].index("--scan-v1") + 1] == "true"
