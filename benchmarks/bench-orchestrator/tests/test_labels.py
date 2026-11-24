# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

import json
from pathlib import Path

import pytest
from bench_orchestrator.config import Benchmark, BenchmarkTarget, Engine, Format, Integration, ScanEngine
from bench_orchestrator.runner.labels import label_ingest_file, label_result_line

FILE_DEFAULT = BenchmarkTarget(Engine.DATAFUSION, Format.VORTEX)
DATASOURCE_V1 = BenchmarkTarget(Engine.DATAFUSION, Format.VORTEX, Integration.DATASOURCE, ScanEngine.V1)
PARQUET = BenchmarkTarget(Engine.DATAFUSION, Format.PARQUET)


def _result_line(name: str, file_format: str = "vortex-file-compressed") -> str:
    return json.dumps(
        {
            "name": name,
            "storage": "nvme",
            "unit": "ns",
            "value": 10,
            "all_runtimes": [10, 11],
            "target": {"engine": "datafusion", "format": file_format},
        }
    )


@pytest.mark.parametrize(
    "name", ["tpch_q01/datafusion:vortex-file-compressed", "tpch_q01_memory/datafusion:vortex-file-compressed"]
)
def test_label_result_line_names_both_axes(name: str) -> None:
    record = json.loads(label_result_line(_result_line(name), DATASOURCE_V1, Benchmark.TPCH))

    assert record["name"] == name + " [datasource, v1]"
    assert record["target"] == {"engine": "datafusion", "format": "vortex-file-compressed [datasource, v1]"}
    assert (record["integration"], record["scan_engine"]) == ("datasource", "v1")
    assert record["all_runtimes"] == [10, 11]


@pytest.mark.parametrize(("benchmark", "scan"), [(Benchmark.TPCH, "v2"), (Benchmark.TPCH_NESTED, "v1")])
def test_label_result_line_keeps_the_default_identity(benchmark: Benchmark, scan: str) -> None:
    name = "tpch_q01/datafusion:vortex-file-compressed"

    record = json.loads(label_result_line(_result_line(name), FILE_DEFAULT, benchmark))

    assert record["name"] == name
    assert record["target"]["format"] == "vortex-file-compressed"
    assert (record["integration"], record["scan_engine"]) == ("file", scan)


def test_label_result_line_leaves_controls_and_logs_alone() -> None:
    line = _result_line("tpch_q01/datafusion:parquet", "parquet")

    assert label_result_line(line, PARQUET, Benchmark.TPCH) == line
    assert label_result_line("not json", DATASOURCE_V1, Benchmark.TPCH) == "not json"


def test_label_ingest_file_names_both_axes(tmp_path: Path) -> None:
    record = {"kind": "query_measurement", "engine": "datafusion", "format": "vortex-file-compressed", "query_idx": 1}
    path = tmp_path / "part.jsonl"
    path.write_text(json.dumps(record) + "\n")

    label_ingest_file(path, DATASOURCE_V1, Benchmark.TPCH)

    (line,) = path.read_text().splitlines()
    assert json.loads(line) == {
        **record,
        "format": "vortex-file-compressed [datasource, v1]",
        "integration": "datasource",
        "scan_engine": "v1",
    }


def test_label_ingest_file_leaves_controls_alone(tmp_path: Path) -> None:
    path = tmp_path / "part.jsonl"
    path.write_text('{"kind": "query_measurement", "format": "parquet"}\n')

    label_ingest_file(path, PARQUET, Benchmark.TPCH)

    assert path.read_text() == '{"kind": "query_measurement", "format": "parquet"}\n'
