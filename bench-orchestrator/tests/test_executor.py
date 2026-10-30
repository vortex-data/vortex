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


def test_build_command_enables_scan_v2_for_datafusion() -> None:
    executor = BenchmarkExecutor(Path("/tmp/datafusion-bench"), Engine.DATAFUSION)

    cmd = executor.build_command(benchmark=Benchmark.FINEWEB, formats=[Format.VORTEX])

    assert cmd[cmd.index("--scan-v2") + 1] == "true"


def test_build_command_omits_scan_v2_for_duckdb() -> None:
    executor = BenchmarkExecutor(Path("/tmp/duckdb-bench"), Engine.DUCKDB)

    cmd = executor.build_command(benchmark=Benchmark.TPCH, formats=[Format.VORTEX])

    assert "--scan-v2" not in cmd


def test_build_command_serializes_vortex_spatial_native_format() -> None:
    executor = BenchmarkExecutor(Path("/tmp/duckdb-bench"), Engine.DUCKDB)

    cmd = executor.build_command(
        benchmark=Benchmark.SPATIALBENCH,
        formats=[Format.PARQUET, Format.VORTEX, Format.VORTEX_SPATIAL_NATIVE],
        iterations=1,
        options={"scale-factor": "1.0"},
    )

    assert cmd[cmd.index("--formats") + 1] == "parquet,vortex,vortex-spatial-native"


def test_build_command_omits_formats_for_lance_backend() -> None:
    executor = BenchmarkExecutor(Path("/tmp/lance-bench"), Engine.LANCE)

    cmd = executor.build_command(
        benchmark=Benchmark.TPCH,
        formats=[Format.LANCE],
        query=3,
    )

    assert cmd[0] == "/tmp/lance-bench"
    assert "--formats" not in cmd
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
