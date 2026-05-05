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

    def fake_run(self, **kwargs):
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


def test_run_combines_gh_json_v3_output_per_backend(tmp_path, monkeypatch) -> None:
    run_store = ResultStore(base_dir=tmp_path / "runs")
    output_path = tmp_path / "artifacts" / "results.v3.jsonl"
    binary_paths = {
        cli_module.Engine.DATAFUSION: tmp_path / "datafusion-bench",
        cli_module.Engine.DUCKDB: tmp_path / "duckdb-bench",
    }
    for binary_path in binary_paths.values():
        binary_path.write_text("", encoding="utf-8")

    monkeypatch.setattr(cli_module, "ResultStore", lambda: run_store)
    monkeypatch.setattr(cli_module.BenchmarkBuilder, "get_binary_path", lambda self, backend: binary_paths[backend])

    seen_backend_paths = []

    def fake_run(self, **kwargs):
        backend_output = kwargs["gh_json_v3"]
        assert backend_output is not None
        assert backend_output != output_path
        backend_output.write_text(f"{self.backend.value}-v3\n", encoding="utf-8")
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
            "--gh-json-v3",
            str(output_path),
        ],
    )

    assert result.exit_code == 0
    assert output_path.read_text(encoding="utf-8") == "datafusion-v3\nduckdb-v3\n"
    assert len(seen_backend_paths) == 2
    assert seen_backend_paths[0] != seen_backend_paths[1]
