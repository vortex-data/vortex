# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Tests for `scripts/s3-query-report.py`."""

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def report() -> ModuleType:
    spec = importlib.util.spec_from_file_location("s3_query_report", REPO_ROOT / "scripts/s3-query-report.py")
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write(path: Path, runner: str, seconds: dict[int, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(
            {
                "name": f"tpch_q{q:02}/datafusion:vortex-file-compressed",
                "unit": "ns",
                "value": int(s * 1e9),
                "runner": runner,
            }
        )
        for q, s in seconds.items()
    ]
    path.write_text("\n".join(lines) + "\n")


def test_report_compares_queries(report: ModuleType, tmp_path: Path):
    write(tmp_path / "real/tpch.json", "real", {1: 2.0, 2: 1.0})
    write(tmp_path / "model/tpch.json", "simulated-model", {1: 2.2, 2: 0.9})
    md = report.render(report.load(tmp_path))
    assert "| q01 | 2.00 | 2.20 | +10% |" in md
    assert "| q02 | 1.00 | 0.90 | -10% |" in md
    assert "| **total** | **3.00** | **3.10** | **+3%** |" in md
    assert "geometric mean 0.99x real" in md


def test_failed_queries_are_left_out_of_totals(report: ModuleType, tmp_path: Path):
    write(tmp_path / "real/tpch.json", "real", {1: 2.0, 2: 1.0})
    write(tmp_path / "model/tpch.json", "simulated-model", {1: 2.0})
    md = report.render(report.load(tmp_path))
    assert "| q02 | 1.00 | failed | - |" in md
    assert "| **total** | **2.00** | **2.00** | **+0%** |" in md
    assert "1 completed on every store" in md
