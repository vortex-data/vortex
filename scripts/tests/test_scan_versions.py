# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest


@pytest.fixture
def comparison() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "compare-scan-versions.py"
    spec = importlib.util.spec_from_file_location("compare_scan_versions", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_excludes_startup_samples_instead_of_using_runner_value(comparison: ModuleType, tmp_path: Path):
    path = tmp_path / "query.jsonl"
    record = {
        "name": "tpch_q06/datafusion:vortex-file-compressed",
        "commit_id": "head-sha",
        "target": {"engine": "datafusion", "format": "vortex"},
        "unit": "ns",
        "value": 9999,
        "all_runtimes": [9000, 8000, 100, 110, 90],
    }
    path.write_text(json.dumps(record), encoding="utf-8")
    result = comparison.measurement(path, query=6, engine="datafusion", commit="head-sha", iterations=5, warmup=2)
    assert result["cold_ns"] == 9000
    assert result["warm_ns"] == [100, 110, 90]


@pytest.mark.parametrize(
    "field,value",
    [
        ("commit_id", "wrong-sha"),
        ("name", "tpch_q07/datafusion:vortex"),
        ("unit", "ms"),
        ("all_runtimes", [100, 0, 100]),
        ("all_runtimes", [100, float("nan"), 100]),
        ("all_runtimes", [100, 100]),
    ],
)
def test_rejects_wrong_or_incomplete_measurements(comparison: ModuleType, tmp_path: Path, field: str, value: object):
    path = tmp_path / "query.jsonl"
    record = {
        "name": "tpch_q06/datafusion:vortex-file-compressed",
        "commit_id": "head-sha",
        "target": {"engine": "datafusion", "format": "vortex"},
        "unit": "ns",
        "all_runtimes": [100, 100, 100],
    }
    record[field] = value
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError):
        comparison.measurement(path, query=6, engine="datafusion", commit="head-sha", iterations=3, warmup=1)


def test_missing_and_duplicate_variants_cannot_produce_complete_summary(comparison: ModuleType):
    rows = [
        {"engine": "duckdb", "query": 1, "round": 0, "variant": variant, "warm_ns": [100, 100]}
        for variant in comparison.VARIANTS
    ]
    with pytest.raises(ValueError, match="duplicate or missing"):
        comparison.summarize(rows[:-1], ["duckdb"], [1], 1)
    with pytest.raises(ValueError, match="duplicate or missing"):
        comparison.summarize(rows + [rows[0]], ["duckdb"], [1], 1)


def test_interval_retains_process_level_variation(comparison: ModuleType):
    top = [[[90.0] * 20, [100.0] * 20, [110.0] * 20]]
    bottom = [[[100.0] * 20] * 3]
    low, high = comparison.ratio_interval(top, bottom)
    assert low <= 0.9
    assert high >= 1.1


def test_rejects_mismatched_build_revision_and_executable(comparison: ModuleType, tmp_path: Path):
    binary = tmp_path / "duckdb-bench"
    binary.write_bytes(b"executable")
    build = {"commit": "base-sha", "binaries": {binary.name: hashlib.sha256(binary.read_bytes()).hexdigest()}}
    (tmp_path / "build-manifest.json").write_text(json.dumps(build), encoding="utf-8")
    assert comparison.binary_provenance(binary, "base-sha")["commit"] == "base-sha"
    with pytest.raises(ValueError, match="Wrong build revision"):
        comparison.binary_provenance(binary, "head-sha")
    binary.write_bytes(b"different executable")
    with pytest.raises(ValueError, match="Binary hash"):
        comparison.binary_provenance(binary, "base-sha")


def test_runs_each_engine_and_scan_version_with_the_expected_binary(
    comparison: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    program = f"""#!{sys.executable}
import json
import os
import subprocess
from pathlib import Path
import sys
if '--print-queries' in sys.argv:
    print('1\\n2')
    raise SystemExit()
def arg(name):
    return sys.argv[sys.argv.index(name) + 1]
if os.environ.get('CI') == 'true' and ('--runner' not in sys.argv or arg('--runner') == 'unknown'):
    raise SystemExit('benchmark runner must not be unknown in CI; pass --runner')
revision = Path(__file__).parent.name
engine = Path(__file__).name.removesuffix('-bench')
version = 1 if arg('--scan-v1') == 'true' else 2
duration = 150 if revision == 'base' else 100 if version == 2 else 200
samples = [9000, 8000] + [duration] * (int(arg('--iterations')) - 2)
record = dict(name=f'tpch_q{{int(arg("--queries")):02}}/{{engine}}:vortex-file-compressed',
              commit_id=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
              target=dict(engine=engine, format='vortex'),
              unit='ns', all_runtimes=samples)
Path(arg('-o')).write_text(json.dumps(record))
"""
    monkeypatch.setenv("CI", "true")
    for revision in ("base", "head"):
        directory = tmp_path / revision
        directory.mkdir()
        for engine in ("duckdb", "datafusion"):
            binary = directory / f"{engine}-bench"
            binary.write_text(program, encoding="utf-8")
            binary.chmod(0o755)
        build = {
            "commit": revision + "-sha",
            "binaries": {
                binary.name: hashlib.sha256(binary.read_bytes()).hexdigest() for binary in directory.iterdir()
            },
        }
        (directory / "build-manifest.json").write_text(json.dumps(build), encoding="utf-8")
    check_output = subprocess.check_output

    def with_elf_notes(args: list[str], *, text: bool, env: dict[str, str] | None = None) -> str:
        if args[0] == "readelf":
            return "Build ID: test-binary\n"
        result = check_output(args, text=text, env=env)
        assert isinstance(result, str)
        return result

    monkeypatch.setattr(subprocess, "check_output", with_elf_notes)
    output = tmp_path / "results"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare-scan-versions.py",
            "tpch",
            "--scale-factor",
            "10.0",
            "--base-dir",
            str(tmp_path / "base"),
            "--head-dir",
            str(tmp_path / "head"),
            "--base-sha",
            "base-sha",
            "--head-sha",
            "head-sha",
            "--rounds",
            "1",
            "--iterations",
            "4",
            "--warmup",
            "2",
            "--threads",
            "2",
            "--output-dir",
            str(output),
        ],
    )
    comparison.main()
    rows = json.loads((output / "measurements.json").read_text(encoding="utf-8"))
    assert len(rows) == 12
    for row in rows:
        revision, version = row["variant"].split("-")
        assert Path(row["command"][0]).parent.name == revision
        flag = row["command"].index("--scan-v1")
        assert row["command"][flag + 1] == str(version == "v1").lower()
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["complete"] is True
    for engine in ("duckdb", "datafusion"):
        comparisons = summary["engines"][engine]["comparisons"]
        assert comparisons["head_v2_vs_base_v2"]["ratio"] == pytest.approx(2 / 3)
        assert comparisons["head_v2_vs_head_v1"]["ratio"] == pytest.approx(0.5)
