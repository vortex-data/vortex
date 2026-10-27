# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Tests for `scripts/s3-scan-report.py`."""

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def report() -> ModuleType:
    spec = importlib.util.spec_from_file_location("s3_scan_report", REPO_ROOT / "scripts/s3-scan-report.py")
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write(path: Path, store: str, runs: dict[str, list[float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tables = [{"table": t, "files": 2, "bytes": 4 << 20, "rows": 1000, "runs_s": r} for t, r in runs.items()]
    path.write_text(json.dumps({"dataset": "tpch-sf10", "store": store, "iterations": 3, "tables": tables}))


def test_report_compares_simulated_with_real(report: ModuleType, tmp_path: Path):
    write(tmp_path / "real/tpch.json", "real", {"lineitem": [2.0, 1.0, 3.0], "orders": [1.0, 1.0, 1.0]})
    write(tmp_path / "model/tpch.json", "simulated-model", {"lineitem": [2.2, 2.2, 2.2], "orders": [0.9, 0.9, 0.9]})
    md = report.render(report.load(tmp_path))
    assert "## tpch-sf10" in md
    # Medians: lineitem 2.0 real vs 2.2 simulated, orders 1.0 vs 0.9; total 3.0 vs 3.1.
    assert "| lineitem | 2 | 4 | 1,000 | 2.00 | 2.20 | +10% |" in md
    assert "| orders | 2 | 4 | 1,000 | 1.00 | 0.90 | -10% |" in md
    assert "**3.00** | **3.10** | **+3%**" in md


def test_report_ignores_unrelated_json(report: ModuleType, tmp_path: Path):
    write(tmp_path / "real/tpch.json", "real", {"lineitem": [1.0]})
    (tmp_path / "other.json").write_text(json.dumps({"schema": "x"}))
    assert list(report.load(tmp_path)) == ["tpch-sf10"]
