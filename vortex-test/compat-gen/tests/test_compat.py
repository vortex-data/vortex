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
def compat() -> ModuleType:
    script = Path(__file__).resolve().parents[1] / "scripts" / "compat.py"
    spec = importlib.util.spec_from_file_location("vortex_compat_orchestrator", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
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


@pytest.mark.parametrize(
    ("profile", "expected_dir"),
    [
        ("release", "release"),
        ("ci", "ci"),
        ("dev", "debug"),
        ("test", "debug"),
        ("bench", "release"),
    ],
)
def test_profile_dir_name_and_build_path(compat, monkeypatch, tmp_path, profile: str, expected_dir: str):
    assert compat._profile_dir_name(profile) == expected_dir

    compat._BUILT_BINS.clear()
    monkeypatch.delenv("VORTEX_COMPAT_BIN", raising=False)
    monkeypatch.chdir(tmp_path)

    build_calls: list[list[str]] = []

    def fake_run_cmd(cmd: list[str], check: bool = False, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
        build_calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    def fake_subprocess_run(cmd: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(cmd, 0, json.dumps({"target_directory": str(tmp_path / "target")}), "")

    monkeypatch.setattr(compat, "_run_cmd", fake_run_cmd)
    monkeypatch.setattr(compat.subprocess, "run", fake_subprocess_run)

    bin_path = compat._build_compat_bin(profile)
    assert bin_path == str(tmp_path / "target" / expected_dir / compat.CARGO_BIN)
    assert build_calls == [["cargo", "build", "-p", compat.CARGO_BIN, "--profile", profile]]

    # Second lookup in the same working directory reuses the built binary.
    assert compat._build_compat_bin(profile) == bin_path
    assert len(build_calls) == 1


def test_cli_accepts_profile_before_or_after_subcommand(compat, monkeypatch):
    seen_profiles: list[str] = []

    def fake_check(args: argparse.Namespace) -> None:
        seen_profiles.append(args.profile)

    monkeypatch.setattr(compat, "cmd_check", fake_check)

    for argv in [
        ["compat.py", "--profile", "ci", "check", "--mode", "all"],
        ["compat.py", "check", "--profile", "ci", "--mode", "all"],
    ]:
        monkeypatch.setattr(sys, "argv", argv)
        compat.main()

    assert seen_profiles == ["ci", "ci"]


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


def test_run_rust_check_handles_malformed_or_nonzero_output(compat, monkeypatch, tmp_path):
    monkeypatch.setattr(compat, "_build_compat_bin", lambda _profile="release": "fake-bin")

    monkeypatch.setattr(
        compat.subprocess,
        "run",
        lambda *_a, **_kw: subprocess.CompletedProcess(["fake-bin"], 1, "not-json", ""),
    )
    bad_json = compat._run_rust_check(tmp_path)
    assert bad_json["failed"][0]["name"] == "(all)"
    assert "invalid check output" in bad_json["failed"][0]["error"]

    monkeypatch.setattr(
        compat.subprocess,
        "run",
        lambda *_a, **_kw: subprocess.CompletedProcess(["fake-bin"], 1, "[]", ""),
    )
    non_dict = compat._run_rust_check(tmp_path)
    assert non_dict["failed"][0]["name"] == "(all)"
    assert "unexpected check output type list" in non_dict["failed"][0]["error"]

    empty_ok = json.dumps({"passed": [], "failed": [], "skipped": []})
    monkeypatch.setattr(
        compat.subprocess,
        "run",
        lambda *_a, **_kw: subprocess.CompletedProcess(["fake-bin"], 2, empty_ok, ""),
    )
    nonzero_empty_fail = compat._run_rust_check(tmp_path)
    assert nonzero_empty_fail["failed"] == [{"name": "(all)", "error": "check process exited with code 2"}]
