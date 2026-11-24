# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

import json

from bench_orchestrator import cli as cli_module
from bench_orchestrator.runner.executor import BenchmarkExecutor
from bench_orchestrator.storage.store import ResultStore
from typer.testing import CliRunner

runner = CliRunner()


def test_prepare_data_uses_formats_json(tmp_path, monkeypatch) -> None:
    data_gen = tmp_path / "data-gen"
    data_gen.write_text("", encoding="utf-8")

    captured: dict[str, list[str]] = {}

    def fake_run(cmd: list[str], check: bool) -> None:
        assert check is True
        captured["cmd"] = cmd

    monkeypatch.setattr(cli_module.BenchmarkBuilder, "get_data_generator_path", lambda self: data_gen)
    monkeypatch.setattr(cli_module.subprocess, "run", fake_run)

    result = runner.invoke(
        cli_module.app,
        [
            "prepare-data",
            "tpch",
            "--formats-json",
            '["parquet","vortex"]',
            "--opt",
            "scale-factor=1.0",
        ],
    )

    assert result.exit_code == 0
    assert captured["cmd"] == [
        str(data_gen),
        "tpch",
        "--formats",
        "parquet,vortex",
        "--opt",
        "scale-factor=1.0",
    ]


def test_write_size_formats_runs_data_gen_and_lance_bench(tmp_path, monkeypatch) -> None:
    data_gen = tmp_path / "data-gen"
    lance_bench = tmp_path / "lance-bench"
    for binary in (data_gen, lance_bench):
        binary.write_text("", encoding="utf-8")

    commands: list[list[str]] = []

    def fake_run(cmd: list[str], check: bool) -> None:
        assert check is True
        commands.append(cmd)

    monkeypatch.setattr(cli_module.BenchmarkBuilder, "get_data_generator_path", lambda self: data_gen)
    monkeypatch.setattr(cli_module.BenchmarkBuilder, "get_binary_path", lambda self, engine: lance_bench)
    monkeypatch.setattr(cli_module.subprocess, "run", fake_run)

    result = runner.invoke(
        cli_module.app,
        [
            "write-size-formats",
            "tpcds",
            "--formats-json",
            '["arrow-ipc","lance"]',
            "--opt",
            "scale-factor=1.0",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert commands == [
        [str(data_gen), "tpcds", "--size-formats", "arrow-ipc", "--opt", "scale-factor=1.0"],
        [str(lance_bench), "tpcds", "--data-only", "--opt", "scale-factor=1.0"],
    ]


def test_write_size_formats_rejects_an_unknown_format() -> None:
    result = runner.invoke(cli_module.app, ["write-size-formats", "tpcds", "--formats-json", '["orc"]'])

    assert result.exit_code == 1
    assert "Invalid size format" in result.stdout


def test_run_writes_compatibility_results_output(tmp_path, monkeypatch) -> None:
    run_store = ResultStore(base_dir=tmp_path / "runs")
    output_path = tmp_path / "artifacts" / "results.json"
    binary_path = tmp_path / "datafusion-bench"
    binary_path.write_text("", encoding="utf-8")

    sample_line = json.dumps(
        {
            "name": "tpch_q1/datafusion:parquet",
            "storage": "nvme",
            "dataset": {"scale_factor": "1.0"},
            "unit": "ns",
            "value": 100,
            "all_runtimes": [95, 100, 105],
            "target": {"engine": "datafusion", "format": "parquet"},
            "commit_id": "deadbeef",
            "env_triple": {
                "architecture": "x86_64",
                "operating_system": "linux",
                "environment": "gnu",
            },
        }
    )

    monkeypatch.setattr(cli_module, "ResultStore", lambda: run_store)
    monkeypatch.setattr(cli_module.BenchmarkBuilder, "get_binary_path", lambda self, backend: binary_path)
    monkeypatch.setattr(cli_module, "drop_os_caches", lambda: None)
    monkeypatch.setattr(BenchmarkExecutor, "list_queries", lambda self, *args, **kwargs: [1])

    def fake_run(self, **kwargs: object) -> list[str]:
        kwargs["on_result"](sample_line)
        return [sample_line]

    monkeypatch.setattr(BenchmarkExecutor, "run", fake_run)

    result = runner.invoke(
        cli_module.app,
        [
            "run",
            "tpch",
            "--targets-json",
            '[{"engine":"datafusion","format":"parquet"}]',
            "--no-build",
            "--output",
            str(output_path),
        ],
    )

    assert result.exit_code == 0
    assert output_path.read_text(encoding="utf-8") == sample_line + "\n"

    run_dirs = [path for path in (tmp_path / "runs").iterdir() if path.is_dir() and path.name != "latest"]
    assert len(run_dirs) == 1

    results_path = run_dirs[0] / "results.jsonl"
    assert results_path.read_text(encoding="utf-8") == sample_line + "\n"

    metadata = json.loads((run_dirs[0] / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["targets"] == [{"engine": "datafusion", "format": "parquet"}]
    assert metadata["binaries"] == {"datafusion": str(binary_path)}


def test_run_combines_ingest_output_per_backend(tmp_path, monkeypatch) -> None:
    run_store = ResultStore(base_dir=tmp_path / "runs")
    output_path = tmp_path / "artifacts" / "results.ingest.jsonl"
    binary_paths = {
        cli_module.Engine.DATAFUSION: tmp_path / "datafusion-bench",
        cli_module.Engine.DUCKDB: tmp_path / "duckdb-bench",
    }
    for binary_path in binary_paths.values():
        binary_path.write_text("", encoding="utf-8")

    monkeypatch.setattr(cli_module, "ResultStore", lambda: run_store)
    monkeypatch.setattr(cli_module.BenchmarkBuilder, "get_binary_path", lambda self, backend: binary_paths[backend])
    monkeypatch.setattr(cli_module, "drop_os_caches", lambda: None)
    monkeypatch.setattr(BenchmarkExecutor, "list_queries", lambda self, *args, **kwargs: [1])

    seen_backend_paths = []

    def fake_run(self, **kwargs: object) -> list[str]:
        backend_output = kwargs["ingest_output"]
        assert backend_output is not None
        assert backend_output != output_path
        backend_output.write_text(f"{self.backend.value}-ingest\n", encoding="utf-8")
        seen_backend_paths.append(backend_output)
        return []

    monkeypatch.setattr(BenchmarkExecutor, "run", fake_run)

    result = runner.invoke(
        cli_module.app,
        [
            "run",
            "tpch",
            "--targets-json",
            '[{"engine":"datafusion","format":"parquet"},{"engine":"duckdb","format":"parquet"}]',
            "--no-build",
            "--ingest-jsonl",
            str(output_path),
        ],
    )

    assert result.exit_code == 0
    assert output_path.read_text(encoding="utf-8") == "datafusion-ingest\nduckdb-ingest\n"
    assert len(seen_backend_paths) == 2
    assert seen_backend_paths[0] != seen_backend_paths[1]


def test_run_writes_lance_data_before_the_first_cache_drop(tmp_path, monkeypatch) -> None:
    run_store = ResultStore(base_dir=tmp_path / "runs")
    binary_path = tmp_path / "lance-bench"
    binary_path.write_text("", encoding="utf-8")
    events = []

    monkeypatch.setattr(cli_module, "ResultStore", lambda: run_store)
    monkeypatch.setattr(cli_module.BenchmarkBuilder, "get_binary_path", lambda self, backend: binary_path)
    monkeypatch.setattr(cli_module, "drop_os_caches", lambda: events.append("drop caches"))
    monkeypatch.setattr(BenchmarkExecutor, "list_queries", lambda self, *args, **kwargs: [1, 2])
    monkeypatch.setattr(BenchmarkExecutor, "write_data", lambda self, *args, **kwargs: events.append("write data"))

    def fake_run(self, **kwargs: object) -> list[str]:
        events.append(f"query {kwargs['query']}")
        return []

    monkeypatch.setattr(BenchmarkExecutor, "run", fake_run)

    result = runner.invoke(
        cli_module.app,
        ["run", "tpcds", "--targets-json", '[{"engine":"datafusion","format":"lance"}]', "--no-build"],
    )

    assert result.exit_code == 0, result.output
    assert events == ["write data", "drop caches", "query 1", "drop caches", "query 2"]


def test_format_query_reading_summarizes_runtimes() -> None:
    line = json.dumps(
        {
            "name": "tpch_q01/datafusion:parquet",
            "unit": "ns",
            "value": 12_000_000,
            "all_runtimes": [11_500_000, 12_000_000, 14_250_000],
        }
    )
    assert cli_module.format_query_reading(["not json", line]) == "median 12.00 ms (min 11.50, max 14.25, n=3)"


def test_format_query_reading_without_timing() -> None:
    memory = json.dumps({"name": "tpch_q01 memory", "unit": "bytes", "value": 1024})
    assert cli_module.format_query_reading([memory]) == "no timing reported"


def test_run_keeps_each_scan_apart(tmp_path, monkeypatch) -> None:
    run_store = ResultStore(base_dir=tmp_path / "runs")
    output_path = tmp_path / "artifacts" / "results.json"
    ingest_path = tmp_path / "artifacts" / "results.ingest.jsonl"
    binary_path = tmp_path / "datafusion-bench"
    binary_path.write_text("", encoding="utf-8")

    monkeypatch.setattr(cli_module, "ResultStore", lambda: run_store)
    monkeypatch.setattr(cli_module.BenchmarkBuilder, "get_binary_path", lambda self, backend: binary_path)
    monkeypatch.setattr(cli_module, "drop_os_caches", lambda: None)
    monkeypatch.setattr(BenchmarkExecutor, "list_queries", lambda self, *args, **kwargs: [1])

    seen_scans = []

    def fake_run(self, **kwargs: object) -> list[str]:
        # The binary labels every scan identically; only the orchestrator knows the axes.
        seen_scans.append((kwargs["integration"].value, kwargs["scan"].value))
        line = json.dumps(
            {
                "name": "tpch_q01/datafusion:vortex-file-compressed",
                "unit": "ns",
                "value": 100,
                "all_runtimes": [100],
                "target": {"engine": "datafusion", "format": "vortex-file-compressed"},
            }
        )
        kwargs["on_result"](line)
        record = {"kind": "query_measurement", "engine": "datafusion", "format": "vortex-file-compressed"}
        kwargs["ingest_output"].write_text(json.dumps(record) + "\n", encoding="utf-8")
        return [line]

    monkeypatch.setattr(BenchmarkExecutor, "run", fake_run)

    result = runner.invoke(
        cli_module.app,
        [
            "run",
            "tpch",
            "--targets-json",
            '[{"engine":"datafusion","format":"vortex"},'
            '{"engine":"datafusion","format":"vortex","integration":"datasource","scan":"v1"}]',
            "--no-build",
            "--output",
            str(output_path),
            "--ingest-jsonl",
            str(ingest_path),
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert seen_scans == [("file", "v2"), ("datasource", "v1")]
    results = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert [record["name"] for record in results] == [
        "tpch_q01/datafusion:vortex-file-compressed",
        "tpch_q01/datafusion:vortex-file-compressed [datasource, v1]",
    ]
    assert [(record["integration"], record["scan_engine"]) for record in results] == [
        ("file", "v2"),
        ("datasource", "v1"),
    ]
    ingest = [json.loads(line) for line in ingest_path.read_text(encoding="utf-8").splitlines()]
    assert [record["format"] for record in ingest] == [
        "vortex-file-compressed",
        "vortex-file-compressed [datasource, v1]",
    ]
    assert [record["scan_engine"] for record in ingest] == ["v2", "v1"]
    assert "datafusion:vortex [file, v2]" in result.stdout
    assert "datafusion:vortex [datasource, v1]" in result.stdout
