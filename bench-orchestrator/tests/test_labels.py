# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

import json
from pathlib import Path

import pytest
from bench_orchestrator.config import BenchmarkTarget, Engine, Format, Integration
from bench_orchestrator.runner.labels import relabel_ingest_file, relabel_result_line

DEFAULT = BenchmarkTarget(engine=Engine.DATAFUSION, format=Format.VORTEX)
SCAN_API = BenchmarkTarget(engine=Engine.DATAFUSION, format=Format.VORTEX, integration=Integration.SCAN_API)


def _result_line(name: str) -> str:
    return json.dumps(
        {
            "name": name,
            "storage": "nvme",
            "unit": "ns",
            "value": 10,
            "all_runtimes": [10, 11],
            "target": {"engine": "datafusion", "format": "vortex-file-compressed"},
        }
    )


@pytest.mark.parametrize(
    "name", ["tpch_q01/datafusion:vortex-file-compressed", "tpch_q01_memory/datafusion:vortex-file-compressed"]
)
def test_relabel_result_line_marks_the_integration(name: str) -> None:
    record = json.loads(relabel_result_line(_result_line(name), SCAN_API))

    assert record["name"] == name + " [scan-api]"
    assert record["target"] == {"engine": "datafusion", "format": "vortex-file-compressed [scan-api]"}
    assert record["all_runtimes"] == [10, 11]


def test_relabel_result_line_leaves_default_targets_and_logs_alone() -> None:
    line = _result_line("tpch_q01/datafusion:vortex-file-compressed")

    assert relabel_result_line(line, DEFAULT) == line
    assert relabel_result_line("not json", SCAN_API) == "not json"


def test_relabel_ingest_file_marks_the_integration(tmp_path: Path) -> None:
    record = {"kind": "query_measurement", "engine": "datafusion", "format": "vortex-file-compressed", "query_idx": 1}
    path = tmp_path / "part.jsonl"
    path.write_text(json.dumps(record) + "\n")

    relabel_ingest_file(path, SCAN_API)

    (line,) = path.read_text().splitlines()
    assert json.loads(line) == {**record, "format": "vortex-file-compressed [scan-api]"}


def test_relabel_ingest_file_leaves_default_targets_alone(tmp_path: Path) -> None:
    path = tmp_path / "part.jsonl"
    path.write_text('{"kind": "query_measurement", "format": "vortex-file-compressed"}\n')

    relabel_ingest_file(path, DEFAULT)

    assert path.read_text() == '{"kind": "query_measurement", "format": "vortex-file-compressed"}\n'
