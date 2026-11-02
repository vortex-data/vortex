# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

spec = importlib.util.spec_from_file_location(
    "codspeed_repeatability", Path(__file__).resolve().parents[1] / "codspeed_repeatability.py"
)
assert spec is not None and spec.loader is not None
repeatability = importlib.util.module_from_spec(spec)
spec.loader.exec_module(repeatability)


def report(host: str, values: list[float], binary: str = "same") -> dict[str, Any]:
    return {
        "host": host,
        "command": ["benchmark", "--bench"],
        "environment": {},
        "instrument": "walltime",
        "value_field": "stats.min_ns",
        "scale": 1e-9,
        "binary_sha256": binary,
        "runs": [{"kernel": value} for value in values],
    }


@pytest.mark.parametrize("slow,passes", [(1.04, True), (1.2, False)])
def test_qualifies_process_spread_across_hosts(slow, passes):
    reports = [report(str(host), [1.0, slow] * 5) for host in range(3)]
    result = repeatability.summarize(reports, 30, 3, False)
    assert result["passes"] is passes
    assert result["processes"] == 30


def test_rejects_insufficient_hosts_and_coverage_changes():
    first = report("host", [1.0] * 30)
    assert not repeatability.summarize([first], 30, 3, False)["passes"]
    second = report("other", [1.0])
    second["runs"] = [{"other_kernel": 1.0}]
    with pytest.raises(ValueError, match="coverage"):
        repeatability.summarize([first, second], 30, 3, False)


def test_requires_explicit_build_comparison():
    reports = [report("one", [1.0]), report("two", [1.0], "rebuilt")]
    with pytest.raises(ValueError, match="identity"):
        repeatability.summarize(reports, 2, 2, False)
    assert repeatability.summarize(reports, 2, 2, True)["passes"]


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf")])
def test_rejects_invalid_or_duplicate_measurements(tmp_path, value):
    path = tmp_path / "result.json"
    path.write_text(json.dumps({"uri": "kernel", "stats": {"min_ns": value}}))
    with pytest.raises(ValueError, match="invalid"):
        repeatability.read_measurements(tmp_path, "*.json", "stats.min_ns", 1e-9)
    path.write_text(json.dumps({"uri": "kernel", "stats": {"min_ns": 1000}}))
    assert repeatability.read_measurements(tmp_path, "*.json", "stats.min_ns", 1e-9) == pytest.approx({"kernel": 1e-6})
    (tmp_path / "duplicate.json").write_text(path.read_text())
    with pytest.raises(ValueError, match="duplicate"):
        repeatability.read_measurements(tmp_path, "*.json", "stats.min_ns", 1e-9)


def test_collect_runs_independent_commands_and_keeps_results(tmp_path):
    script = Path(__file__).resolve().parents[1] / "codspeed_repeatability.py"
    output = tmp_path / "collected"
    command = (
        "import os,json,pathlib; "
        "p=pathlib.Path(os.environ['CODSPEED_CARGO_WORKSPACE_ROOT'])/'value.json'; "
        "p.write_text(json.dumps({'uri':'kernel','value':20}))"
    )
    subprocess.run(
        [
            sys.executable,
            str(script),
            "collect",
            "--binary",
            sys.executable,
            "--instrument",
            "simulation",
            "--output",
            str(output),
            "--repeats",
            "2",
            "--result-glob",
            "*.json",
            "--value-field",
            "value",
            "--scale",
            "1",
            "--",
            sys.executable,
            "-c",
            command,
        ],
        check=True,
    )
    result = json.loads((output / "report.json").read_text())
    assert result["runs"] == [{"kernel": 20}, {"kernel": 20}]
    assert len(list(output.glob("process-*/value.json"))) == 2


@pytest.mark.parametrize(
    ("mode", "runner_mode", "expected"),
    [
        (None, None, 0),
        ("walltime", None, 0),
        ("simulation", None, 2),
        (None, "simulation", 2),
        (None, "walltime", 0),
        ("walltime", "simulation", 0),
    ],
)
def test_python_exclusions_only_apply_to_integer_scalar_scans(monkeypatch, mode, runner_mode, expected):
    # Collection does not need native Vortex or benchmark data. Stub only those imports.
    vortex = ModuleType("vortex")
    vortex.__dict__["VortexFile"] = object
    benchmark = ModuleType("benchmark")
    benchmark.__dict__["data"] = SimpleNamespace(COLUMN_SETS={"int": ("x",)})
    monkeypatch.setitem(sys.modules, "vortex", vortex)
    monkeypatch.setitem(sys.modules, "benchmark", benchmark)
    monkeypatch.delenv("CODSPEED_ENV", raising=False)
    monkeypatch.delenv("CODSPEED_RUNNER_MODE", raising=False)
    if runner_mode:
        monkeypatch.setenv("CODSPEED_ENV", "test")
        monkeypatch.setenv("CODSPEED_RUNNER_MODE", runner_mode)
    path = Path(__file__).resolve().parents[2] / "vortex-python" / "benchmark" / "conftest.py"
    spec = importlib.util.spec_from_file_location("benchmark_conftest", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    marked = []
    items = [
        SimpleNamespace(path=path.parent / file, name=name, add_marker=lambda marker: marked.append(marker))
        for file, name in [
            ("test_scalar_at.py", "test_scan_scalar_at[int]"),
            ("test_scalar_at.py", "test_repeated_scan_scalar_at[int]"),
            ("test_scalar_at.py", "test_scan_scalar_at[int_str]"),
            ("test_scan.py", "test_scan_scalar_at[int]"),
            ("unrelated/test_scalar_at.py", "test_scan_scalar_at[int]"),
        ]
    ]
    module.pytest_collection_modifyitems(
        SimpleNamespace(getoption=lambda name, **kwargs: True if name == "codspeed" else mode), items
    )
    assert len(marked) == expected


@pytest.mark.parametrize("field,value", [("command", ["different"]), ("environment", {"DIVAN_SAMPLE_SIZE": "16"})])
def test_rejects_mixed_measurement_configuration(field, value):
    first = report("first", [1.0])
    second = report("second", [1.0])
    second[field] = value
    with pytest.raises(ValueError, match=field):
        repeatability.summarize([first, second], 2, 2, False)
