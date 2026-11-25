# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Unit tests for the `compat.py` orchestrator."""

from __future__ import annotations

import argparse
import importlib.util
import json
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
