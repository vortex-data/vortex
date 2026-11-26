# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
from bench_orchestrator.config import BENCHMARKS_WITHOUT_SCAN_V2, Benchmark, Engine, Format, Integration, ScanEngine
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


def test_benchmarks_without_scan_v2_are_explicit() -> None:
    assert BENCHMARKS_WITHOUT_SCAN_V2 == {Benchmark.POLARSIGNALS, Benchmark.STATPOPGEN, Benchmark.TPCH_NESTED}


@pytest.mark.parametrize("engine", [Engine.DATAFUSION, Engine.DUCKDB])
@pytest.mark.parametrize("benchmark", list(Benchmark))
def test_build_command_defaults_to_the_suites_scan(engine: Engine, benchmark: Benchmark) -> None:
    executor = BenchmarkExecutor(Path("/tmp/bench"), engine)

    cmd = executor.build_command(benchmark=benchmark, formats=[Format.VORTEX])

    expected = "true" if benchmark in BENCHMARKS_WITHOUT_SCAN_V2 else "false"
    assert cmd[cmd.index("--scan-v1") + 1] == expected


@pytest.mark.parametrize("engine", [Engine.DATAFUSION, Engine.DUCKDB])
@pytest.mark.parametrize(("scan", "scan_v1"), [(ScanEngine.V1, "true"), (ScanEngine.V2, "false")])
def test_build_command_passes_the_targets_scan(engine: Engine, scan: ScanEngine, scan_v1: str) -> None:
    executor = BenchmarkExecutor(Path("/tmp/bench"), engine)

    cmd = executor.build_command(benchmark=Benchmark.TPCH, formats=[Format.VORTEX], scan=scan)

    assert cmd[cmd.index("--scan-v1") + 1] == scan_v1


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


def test_write_data_writes_lance_without_running_a_query(monkeypatch) -> None:
    calls = []

    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)

    BenchmarkExecutor(Path("/tmp/lance-bench"), Engine.LANCE).write_data(Benchmark.TPCDS, {"scale-factor": "1.0"})
    BenchmarkExecutor(Path("/tmp/datafusion-bench"), Engine.DATAFUSION).write_data(Benchmark.TPCDS)

    assert calls == [["/tmp/lance-bench", "tpcds", "--data-only", "--opt", "scale-factor=1.0"]]


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


def test_datasource_integration_registers_vortex_table() -> None:
    executor = BenchmarkExecutor(Path("/tmp/datafusion-bench"), Engine.DATAFUSION)

    cmd = executor.build_command(
        benchmark=Benchmark.TPCH, formats=[Format.VORTEX], integration=Integration.DATASOURCE, scan=ScanEngine.V1
    )

    assert cmd[cmd.index("--scan-api") + 1] == "true"
    assert cmd[cmd.index("--scan-v1") + 1] == "true"


@pytest.mark.parametrize("engine", [Engine.DATAFUSION, Engine.DUCKDB])
def test_file_integration_omits_the_scan_api_flag(engine: Engine) -> None:
    executor = BenchmarkExecutor(Path("/tmp/bench"), engine)

    cmd = executor.build_command(benchmark=Benchmark.TPCH, formats=[Format.VORTEX])

    assert "--scan-api" not in cmd


def test_datasource_integration_is_rejected_by_duckdb() -> None:
    executor = BenchmarkExecutor(Path("/tmp/duckdb-bench"), Engine.DUCKDB)

    with pytest.raises(ValueError, match="duckdb cannot run the datasource integration"):
        _ = executor.build_command(
            benchmark=Benchmark.TPCH, formats=[Format.VORTEX], integration=Integration.DATASOURCE
        )
