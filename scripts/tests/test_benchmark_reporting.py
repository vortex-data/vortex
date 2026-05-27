# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
COMPARE_SCRIPT = REPO_ROOT / "scripts" / "compare-benchmark-jsons.py"
CAPTURE_SCRIPT = REPO_ROOT / "scripts" / "capture-file-sizes.py"


def load_compare_module():
    spec = importlib.util.spec_from_file_location("compare_benchmark_jsons", COMPARE_SCRIPT)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def timing_row(name: str, base: int, pr: int) -> dict[str, object]:
    return {
        "name": name,
        "value_base": base,
        "value_pr": pr,
        "all_runtimes_base": [base, base, base],
        "all_runtimes_pr": [pr, pr, pr],
    }


def test_within_engine_analysis_uses_each_engines_own_parquet_control() -> None:
    compare = load_compare_module()
    rows = [
        timing_row("tpch_q01/datafusion:parquet", 100, 200),
        timing_row("tpch_q01/datafusion:vortex-file-compressed", 100, 180),
        timing_row("tpch_q01/duckdb:parquet", 100, 100),
        timing_row("tpch_q01/duckdb:vortex-file-compressed", 100, 120),
    ]
    df = pd.DataFrame(rows)
    df[["engine", "file_format", "query"]] = df["name"].apply(compare.extract_target_fields)

    analyses = compare.build_within_engine_statistical_analyses(df, threshold_pct=5)

    assert set(analyses) == {"datafusion", "duckdb"}
    assert compare.build_verdict(analyses["datafusion"])["impact"] == "-10.0%"
    assert compare.build_verdict(analyses["duckdb"])["impact"] == "+20.0%"


def file_size_record(commit: str, size: int) -> dict[str, object]:
    return {
        "metric": "file_size",
        "unit": "bytes",
        "value": size,
        "commit_id": commit,
        "file_size": {
            "benchmark": "tpch",
            "scale_factor": "10",
            "format": "vortex-file-compressed",
            "file": "part-0.vortex",
        },
    }


def test_file_size_report_reads_shared_benchmark_rows() -> None:
    compare = load_compare_module()

    report = compare.format_file_size_report(
        pd.DataFrame([file_size_record("base-sha", 100)]),
        pd.DataFrame([file_size_record("pr-sha", 125)]),
    )

    assert "<summary>File Size Changes (1 files changed, +25.0% overall, 1↑ 0↓)</summary>" in report
    assert "| part-0.vortex | 10 | vortex-file-compressed | 100 B | 125 B | +25 B | +25.0% |" in report


def test_capture_file_sizes_emits_shared_benchmark_rows(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    format_dir = data_dir / "tpch" / "10" / "vortex-file-compressed"
    format_dir.mkdir(parents=True)
    (format_dir / "part-0.vortex").write_bytes(b"x" * 42)
    output_path = tmp_path / "sizes.jsonl"

    result = subprocess.run(
        [
            sys.executable,
            str(CAPTURE_SCRIPT),
            str(data_dir),
            "--benchmark",
            "tpch",
            "--commit",
            "deadbeef",
            "-o",
            str(output_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    records = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert records == [
        {
            "metric": "file_size",
            "unit": "bytes",
            "value": 42,
            "commit_id": "deadbeef",
            "file_size": {
                "benchmark": "tpch",
                "scale_factor": "10",
                "format": "vortex-file-compressed",
                "file": "part-0.vortex",
            },
        }
    ]
