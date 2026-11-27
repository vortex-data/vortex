# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Unit tests for the `compat.py` orchestrator."""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest


@pytest.fixture
def compat(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    script = Path(__file__).resolve().parents[1] / "scripts" / "compat.py"
    spec = importlib.util.spec_from_file_location("vortex_compat_orchestrator", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def _write_version_store(root: Path, versions: list[str], fixtures_by_version: dict[str, list[str]]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "versions.json").write_text(json.dumps(versions))
    for ver in versions:
        arrays_dir = root / f"v{ver}" / "arrays"
        arrays_dir.mkdir(parents=True, exist_ok=True)
        names = fixtures_by_version[ver]
        manifest = {
            "version": ver,
            "generated_at": "2026-01-01T00:00:00Z",
            "fixtures": [{"name": name, "description": name, "sha256": ""} for name in names],
        }
        (arrays_dir / "manifest.json").write_text(json.dumps(manifest))
        for name in names:
            (arrays_dir / name).write_bytes(b"stub")


def test_cmd_check_runs_all_versions_and_reports_all_failures(compat, monkeypatch, tmp_path, capsys):
    store_dir = tmp_path / "store"
    _write_version_store(
        store_dir,
        ["0.58.0", "0.59.0", "0.60.0", "0.61.0", "0.62.0"],
        {
            "0.58.0": ["ignored.vortex"],
            "0.59.0": ["ignored.vortex"],
            "0.60.0": ["a.vortex", "b.vortex"],
            "0.61.0": ["missing_file.vortex", "c.vortex"],
            "0.62.0": ["d.vortex"],
        },
    )
    # Use a non-object JSON manifest, a malformed manifest, and a missing file to verify all continue.
    (store_dir / "v0.58.0" / "arrays" / "manifest.json").write_text("[]")
    (store_dir / "v0.59.0" / "arrays" / "manifest.json").write_text("{not-json")
    (store_dir / "v0.61.0" / "arrays" / "missing_file.vortex").unlink()

    checked_versions: list[list[str]] = []

    def fake_rust_check(dir: Path, mode: str = "superset", profile: str = "release") -> dict[str, object]:
        files = sorted(p.name for p in dir.glob("*.vortex"))
        checked_versions.append(files)
        if files == ["a.vortex", "b.vortex"]:
            return {
                "passed": [],
                "failed": [
                    {"name": "a.vortex", "error": "err a"},
                    {"name": "b.vortex", "error": "err b"},
                ],
                "skipped": [],
            }
        if files == ["c.vortex"]:
            return {
                "passed": ["c.vortex"],
                "failed": [],
                "skipped": ["missing_file.vortex"],
            }
        return {
            "passed": [],
            "failed": [{"name": "d.vortex", "error": "err d"}],
            "skipped": [],
        }

    monkeypatch.setattr(compat, "_run_rust_check", fake_rust_check)

    args = argparse.Namespace(store=str(store_dir), versions=None, mode="all", profile="ci")
    with pytest.raises(SystemExit) as exc:
        compat.cmd_check(args)

    assert exc.value.code == 1
    assert checked_versions == [["a.vortex", "b.vortex"], ["c.vortex"], ["d.vortex"]]

    stderr = capsys.readouterr().err
    assert "v0.58.0/(manifest):" in stderr
    assert "v0.59.0/(manifest):" in stderr
    assert "v0.60.0/a.vortex: err a" in stderr
    assert "v0.60.0/b.vortex: err b" in stderr
    assert "v0.61.0/missing_file.vortex: not found at v0.61.0/arrays/missing_file.vortex" in stderr
    assert "v0.62.0/d.vortex: err d" in stderr


def test_parallel_download_collects_io_errors(compat, tmp_path):
    class FlakyStore(compat.Store):
        def read(self, key: str) -> bytes | None:
            if key.endswith("boom.vortex"):
                raise OSError("network timeout")
            return b"ok"

    failures = compat._parallel_download(
        FlakyStore(),
        [{"name": "boom.vortex"}, {"name": "good.vortex"}],
        "v0.62.0/arrays",
        tmp_path,
    )

    assert failures == [("boom.vortex", "download failed: network timeout")]
    assert (tmp_path / "good.vortex").read_bytes() == b"ok"


def test_run_rust_check_handles_invalid_json(compat, monkeypatch, tmp_path):
    monkeypatch.setattr(compat, "_build_compat_bin", lambda _profile: "/fake/vortex-compat")
    monkeypatch.setattr(
        compat.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess([], 1, stdout="not json", stderr=None),
    )

    result = compat._run_rust_check(tmp_path)
    assert result["passed"] == []
    assert result["skipped"] == []
    assert len(result["failed"]) == 1
    assert result["failed"][0]["name"] == "(all)"
    assert "invalid JSON" in result["failed"][0]["error"]


def test_load_accepted_edition_failures_validates_checked_in_file(compat):
    overrides, failures = compat._load_accepted_edition_failures(compat.DEFAULT_ACCEPTED_EDITION_FAILURES)
    assert isinstance(overrides, dict)
    assert isinstance(failures, dict)


def test_load_accepted_edition_failures_rejects_duplicates_and_bad_entries(compat, tmp_path):
    dup_path = tmp_path / "dup.json"
    entry = {
        "edition": "core2025.05.0",
        "reader_version": "0.36.0",
        "fixture": "fsst.vortex",
        "expected_causes": ["Expected 2 buffers, got 3"],
        "reason": "legacy 2-buffer format",
    }
    dup_path.write_text(json.dumps({"failures": [entry, entry]}))
    with pytest.raises(ValueError, match="duplicate"):
        compat._load_accepted_edition_failures(dup_path)

    bad_path = tmp_path / "bad.json"
    bad_path.write_text(json.dumps({"failures": [{**entry, "reason": "   "}]}))
    with pytest.raises(ValueError, match="reason"):
        compat._load_accepted_edition_failures(bad_path)

    bad_causes_path = tmp_path / "bad_causes.json"
    bad_causes_path.write_text(json.dumps({"failures": [{**entry, "expected_causes": []}]}))
    with pytest.raises(ValueError, match="expected_causes"):
        compat._load_accepted_edition_failures(bad_causes_path)


def test_evaluate_edition_results_enforces_shrink_only_and_expected_causes(compat):
    accepted = {
        ("core2025.05.0", "0.36.0", "known_fail.vortex"): {
            "reason": "known issue",
            "expected_causes": ["expected err"],
        },
        ("core2025.05.0", "0.36.0", "new_cause.vortex"): {
            "reason": "known fsst issue",
            "expected_causes": ["Expected 2 buffers, got 3"],
        },
        ("core2025.05.0", "0.36.0", "partial_fix.vortex"): {
            "reason": "two old issues",
            "expected_causes": ["Expected 2 buffers, got 3", "old fixed bug"],
        },
        ("core2025.05.0", "0.36.0", "fixed_now.vortex"): {
            "reason": "was failing",
            "expected_causes": ["gone"],
        },
        ("core2025.05.0", "0.36.0", "removed.vortex"): {
            "reason": "no longer written",
            "expected_causes": ["gone"],
        },
    }
    report = {
        "passed": ["ok.vortex", "fixed_now.vortex"],
        "failed": [
            {"name": "known_fail.vortex", "error": "ValueError: expected err", "causes": ["ValueError: expected err"]},
            {
                "name": "new_cause.vortex",
                "error": "ValueError: Expected 2 buffers, got 3; ValueError: brand new regression",
                "causes": ["ValueError: Expected 2 buffers, got 3", "ValueError: brand new regression"],
            },
            {
                "name": "partial_fix.vortex",
                "error": "ValueError: Expected 2 buffers, got 3",
                "causes": ["ValueError: Expected 2 buffers, got 3"],
            },
            {"name": "new_regression.vortex", "error": "boom", "causes": ["boom"]},
        ],
    }

    evaluated = compat._evaluate_edition_results(
        "core2025.05.0",
        "0.36.0",
        [
            "ok.vortex",
            "fixed_now.vortex",
            "known_fail.vortex",
            "new_cause.vortex",
            "partial_fix.vortex",
            "new_regression.vortex",
        ],
        report,
        accepted,
    )

    assert evaluated["passed"] == ["ok.vortex", "fixed_now.vortex"]
    assert [f["name"] for f in evaluated["accepted"]] == ["known_fail.vortex", "partial_fix.vortex"]
    assert [f["name"] for f in evaluated["unexpected"]] == ["new_cause.vortex", "new_regression.vortex"]
    assert "brand new regression" in evaluated["unexpected"][0]["error"]
    assert [(s["name"], s["status"]) for s in evaluated["stale"]] == [
        ("partial_fix.vortex", "stale expected_causes: old fixed bug"),
        ("fixed_now.vortex", "passed"),
        ("removed.vortex", "not written"),
    ]


def test_cmd_check_editions_applies_override_and_fails_on_stale_or_unexpected(compat, monkeypatch, tmp_path):
    accepted_path = tmp_path / "accepted.json"
    accepted_path.write_text(
        json.dumps(
            {
                "reader_overrides": {
                    "core2025.06.0": {
                        "reader_version": "0.42.0",
                        "reason": "0.40.0 lacks vortex.pco",
                    }
                },
                "failures": [
                    {
                        "edition": "core2025.06.0",
                        "reader_version": "0.42.0",
                        "fixture": "fsst.vortex",
                        "expected_causes": ["Expected 2 buffers, got 3"],
                        "reason": "known fsst issue",
                    }
                ],
            }
        )
    )

    monkeypatch.setattr(
        compat,
        "_run_rust_editions",
        lambda profile="release": [
            {"id": "core2025.06.0", "min_library_version": "0.40.0"},
            {"id": "core2026.08.3", "min_library_version": "0.85.0"},
        ],
    )

    def fake_generate(output: Path, profile="release", edition=None, exclude=None):
        output.mkdir(parents=True, exist_ok=True)
        (output / "fixtures.json").write_text(
            json.dumps(
                {
                    "edition": edition,
                    "fixtures": [
                        {"name": "primitives.vortex", "description": "p", "sha256": "a"},
                        {"name": "fsst.vortex", "description": "f", "sha256": "b"},
                    ],
                }
            )
        )

    dumped_expected: list[Path] = []
    monkeypatch.setattr(
        compat,
        "_run_rust_dump_expected",
        lambda output, profile="release", exclude=None: dumped_expected.append(output),
    )
    monkeypatch.setattr(compat, "_run_rust_generate", fake_generate)

    reader_calls: list[str] = []

    def fake_reader(reader_version, src_dir, expected_dir, python_version="3.12") -> dict[str, object]:
        reader_calls.append(reader_version)
        if reader_version == "0.42.0":
            return {
                "passed": ["primitives.vortex"],
                "failed": [{"name": "fsst.vortex", "error": "Expected 2 buffers, got 3", "causes": []}],
            }
        return {"passed": ["primitives.vortex", "fsst.vortex"], "failed": []}

    monkeypatch.setattr(compat, "_run_edition_reader", fake_reader)

    json_out = tmp_path / "report.json"
    args = argparse.Namespace(
        profile="ci",
        editions=None,
        exclude=None,
        accepted_failures=str(accepted_path),
        python="3.12",
        json_output=str(json_out),
    )
    compat.cmd_check_editions(args)

    assert len(dumped_expected) == 1
    assert reader_calls == ["0.42.0", "0.85.0"]
    report = json.loads(json_out.read_text())
    assert len(report) == 2
    assert report[0]["reader_version"] == "0.42.0"
    assert [f["name"] for f in report[0]["accepted"]] == ["fsst.vortex"]

    # Now make fsst.vortex pass on 0.42.0 -> shrink-only check must fail with exit 1.
    def fake_reader_fixed(reader_version, src_dir, expected_dir, python_version="3.12") -> dict[str, object]:
        return {"passed": ["primitives.vortex", "fsst.vortex"], "failed": []}

    monkeypatch.setattr(compat, "_run_edition_reader", fake_reader_fixed)
    with pytest.raises(SystemExit) as exc:
        compat.cmd_check_editions(args)
    assert exc.value.code == 1
