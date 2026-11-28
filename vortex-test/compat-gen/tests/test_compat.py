# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Unit tests for the `compat.py` orchestrator."""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from http.client import IncompleteRead
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pyarrow as pa
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


def test_cmd_check_editions_enforces_skips_gen_failures_and_shrink_only(compat, monkeypatch, tmp_path):
    accepted_path = tmp_path / "accepted.json"
    accepted_path.write_text(
        json.dumps(
            {
                "expected_skips": {
                    "core2025.06.0": ["map.vortex"],
                },
                "failures": [
                    {
                        "edition": "core2025.06.0",
                        "reader_version": "0.42.0",
                        "fixture": "fsst.vortex",
                        "expected_causes": ["column c1: Expected 2 buffers, got 3"],
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
            {"id": "core2025.06.0", "min_library_version": "0.42.0"},
            {"id": "core2026.08.3", "min_library_version": "0.85.0"},
        ],
    )

    def fake_generate(output: Path, profile="release", edition=None, exclude=None, check=True) -> int:
        output.mkdir(parents=True, exist_ok=True)
        skipped = [{"name": "map.vortex", "reason": "not in edition"}] if edition == "core2025.06.0" else []
        (output / "fixtures.json").write_text(
            json.dumps(
                {
                    "edition": edition,
                    "fixtures": [
                        {"name": "primitives.vortex", "description": "p", "sha256": "a"},
                        {"name": "fsst.vortex", "description": "f", "sha256": "b"},
                    ],
                    "skipped": skipped,
                }
            )
        )
        return 0

    dumped_expected: list[Path] = []
    monkeypatch.setattr(
        compat,
        "_run_rust_dump_expected",
        lambda output, profile="release", exclude=None, check=True: (dumped_expected.append(output), 0)[1],
    )
    monkeypatch.setattr(compat, "_run_rust_generate", fake_generate)

    reader_calls: list[str] = []

    def fake_reader(reader_version, src_dir, expected_dir, python_version="3.12") -> dict[str, object]:
        reader_calls.append(reader_version)
        if reader_version == "0.42.0":
            return {
                "passed": ["primitives.vortex"],
                "failed": [
                    {
                        "name": "fsst.vortex",
                        "error": "column c1: ValueError: Expected 2 buffers, got 3",
                        "causes": ["column c1: ValueError: Expected 2 buffers, got 3"],
                    }
                ],
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

    # A new failing column with the same error message must NOT match column c1's expected_causes.
    def fake_reader_new_col(reader_version, src_dir, expected_dir, python_version="3.12") -> dict[str, object]:
        if reader_version == "0.42.0":
            return {
                "passed": ["primitives.vortex"],
                "failed": [
                    {
                        "name": "fsst.vortex",
                        "error": (
                            "column c1: ValueError: Expected 2 buffers, got 3; "
                            "column c2: ValueError: Expected 2 buffers, got 3"
                        ),
                        "causes": [
                            "column c1: ValueError: Expected 2 buffers, got 3",
                            "column c2: ValueError: Expected 2 buffers, got 3",
                        ],
                    }
                ],
            }
        return {"passed": ["primitives.vortex", "fsst.vortex"], "failed": []}

    monkeypatch.setattr(compat, "_run_edition_reader", fake_reader_new_col)
    with pytest.raises(SystemExit) as exc:
        compat.cmd_check_editions(args)
    assert exc.value.code == 1
    new_col_report = json.loads(json_out.read_text())
    assert [f["name"] for f in new_col_report[0]["unexpected"]] == ["fsst.vortex"]
    assert "column c2" in new_col_report[0]["unexpected"][0]["error"]

    # An unexpected skip in an edition must fail as unexpected, and a fixture write failure in
    # one edition must still check the remaining editions and report all failures.
    def fake_generate_with_fail_and_skip(
        output: Path, profile="release", edition=None, exclude=None, check=True
    ) -> int:
        output.mkdir(parents=True, exist_ok=True)
        if edition == "core2025.06.0":
            (output / "fixtures.json").write_text(
                json.dumps(
                    {
                        "edition": edition,
                        "fixtures": [{"name": "fsst.vortex", "description": "f", "sha256": "b"}],
                        "skipped": [
                            {"name": "map.vortex", "reason": "not in edition"},
                            {"name": "new_skip.vortex", "reason": "unexpected encoding"},
                        ],
                        "failed": [{"name": "primitives.vortex", "error": "disallowed encoding"}],
                    }
                )
            )
            return 1
        (output / "fixtures.json").write_text(
            json.dumps(
                {
                    "edition": edition,
                    "fixtures": [{"name": "primitives.vortex", "description": "p", "sha256": "a"}],
                }
            )
        )
        return 0

    monkeypatch.setattr(compat, "_run_rust_generate", fake_generate_with_fail_and_skip)
    monkeypatch.setattr(compat, "_run_edition_reader", fake_reader)
    with pytest.raises(SystemExit) as exc:
        compat.cmd_check_editions(args)
    assert exc.value.code == 1
    gen_fail_report = json.loads(json_out.read_text())
    assert len(gen_fail_report) == 2
    assert {f["name"] for f in gen_fail_report[0]["unexpected"]} == {"primitives.vortex", "new_skip.vortex"}
    assert gen_fail_report[1]["edition"] == "core2026.08.3"

    # Now make fsst.vortex pass on 0.42.0 -> shrink-only check must fail with exit 1.
    monkeypatch.setattr(compat, "_run_rust_generate", fake_generate)

    def fake_reader_fixed(reader_version, src_dir, expected_dir, python_version="3.12") -> dict[str, object]:
        return {"passed": ["primitives.vortex", "fsst.vortex"], "failed": []}

    monkeypatch.setattr(compat, "_run_edition_reader", fake_reader_fixed)
    with pytest.raises(SystemExit) as exc:
        compat.cmd_check_editions(args)
    assert exc.value.code == 1

    # Excluded fixtures must not be flagged as stale when omitted via --exclude.
    def fake_generate_excluded(output: Path, profile="release", edition=None, exclude=None, check=True) -> int:
        output.mkdir(parents=True, exist_ok=True)
        fixtures = [{"name": "primitives.vortex", "description": "p", "sha256": "a"}]
        if not (exclude and any(pat in "fsst.vortex" for pat in exclude)):
            fixtures.append({"name": "fsst.vortex", "description": "f", "sha256": "b"})
        skipped = [{"name": "map.vortex", "reason": "not in edition"}] if edition == "core2025.06.0" else []
        (output / "fixtures.json").write_text(
            json.dumps({"edition": edition, "fixtures": fixtures, "skipped": skipped})
        )
        return 0

    monkeypatch.setattr(compat, "_run_rust_generate", fake_generate_excluded)
    monkeypatch.setattr(
        compat,
        "_run_edition_reader",
        lambda reader_version, src_dir, expected_dir, python_version="3.12": {
            "passed": ["primitives.vortex"],
            "failed": [],
        },
    )
    args_excl = argparse.Namespace(
        profile="ci",
        editions=None,
        exclude="fsst",
        accepted_failures=str(accepted_path),
        python="3.12",
        json_output=None,
    )
    compat.cmd_check_editions(args_excl)

    # An accepted entry whose (edition, reader_version) is never checked must fail as stale.
    orphan_path = tmp_path / "orphan.json"
    orphan_path.write_text(
        json.dumps(
            {
                "expected_skips": {
                    "core2025.06.0": ["map.vortex"],
                },
                "failures": [
                    {
                        "edition": "core2025.06.0",
                        "reader_version": "0.99.0",
                        "fixture": "fsst.vortex",
                        "expected_causes": ["Expected 2 buffers, got 3"],
                        "reason": "stale reader version",
                    }
                ],
            }
        )
    )
    orphan_json = tmp_path / "orphan_report.json"
    args_orphan = argparse.Namespace(
        profile="ci",
        editions=None,
        exclude=None,
        accepted_failures=str(orphan_path),
        python="3.12",
        json_output=str(orphan_json),
    )
    with pytest.raises(SystemExit) as exc:
        compat.cmd_check_editions(args_orphan)
    assert exc.value.code == 1
    orphan_report = json.loads(orphan_json.read_text())
    orphan_rows = [r for r in orphan_report if r["reader_version"] == "0.99.0"]
    assert len(orphan_rows) == 1
    assert orphan_rows[0]["stale"] == [
        {
            "name": "fsst.vortex",
            "status": "unchecked edition/reader_version",
            "reason": "stale reader version",
        }
    ]


def test_incomplete_read_retried_and_collected(compat, monkeypatch, tmp_path, capsys):
    # S3Store.read retries transient IncompleteRead and succeeds on a later attempt.
    attempts = 0

    class _Resp:
        def __enter__(self) -> _Resp:
            return self

        def __exit__(self, *_exc: object) -> bool:
            return False

        def read(self) -> bytes:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise IncompleteRead(b"partial", 10)
            return b"complete"

    monkeypatch.setattr(compat, "urlopen", lambda _url, timeout=10: _Resp())
    monkeypatch.setattr(compat.time, "sleep", lambda _s: None)
    assert compat.S3Store("bucket").read("v0.62.0/arrays/a.vortex") == b"complete"
    assert attempts == 2

    # Exhausted IncompleteRead on a manifest or fixture is collected so later versions still run.
    class TruncatingStore(compat.Store):
        def list_versions(self) -> list[str]:
            return ["0.61.0", "0.62.0"]

        def display_name(self) -> str:
            return "s3://fake"

        def read(self, key: str) -> bytes | None:
            if key == "v0.61.0/arrays/manifest.json":
                raise IncompleteRead(b"{", 100)
            if key == "v0.62.0/arrays/manifest.json":
                return json.dumps(
                    {
                        "version": "0.62.0",
                        "generated_at": "2026-01-01T00:00:00Z",
                        "fixtures": [
                            {"name": "trunc.vortex", "description": "", "sha256": ""},
                            {"name": "ok.vortex", "description": "", "sha256": ""},
                        ],
                    }
                ).encode()
            if key.endswith("trunc.vortex"):
                raise IncompleteRead(b"partial", 50)
            return b"good"

    checked: list[list[str]] = []
    monkeypatch.setattr(compat, "_parse_store", lambda _spec: TruncatingStore())
    monkeypatch.setattr(
        compat,
        "_run_rust_check",
        lambda d, mode="superset", profile="release": (
            checked.append(sorted(p.name for p in d.glob("*.vortex"))),
            {"passed": ["ok.vortex"], "failed": [], "skipped": ["trunc.vortex"]},
        )[1],
    )

    args = argparse.Namespace(store="s3://fake", versions=None, mode="all", profile="ci")
    with pytest.raises(SystemExit) as exc:
        compat.cmd_check(args)
    assert exc.value.code == 1
    assert checked == [["ok.vortex"]]
    stderr = capsys.readouterr().err
    assert "v0.61.0/(manifest):" in stderr
    assert "v0.62.0/trunc.vortex: download failed:" in stderr


def test_evaluate_edition_results_skips_stale_on_whole_reader_failure(compat):
    accepted = {
        ("core2025.05.0", "0.36.0", "fsst.vortex"): {
            "expected_causes": ["Expected 2 buffers, got 3"],
            "reason": "known fsst issue",
        }
    }
    report = {
        "passed": [],
        "failed": [{"name": "(all)", "error": "uv wheel install failed"}],
    }
    res = compat._evaluate_edition_results(
        "core2025.05.0",
        "0.36.0",
        ["primitives.vortex", "fsst.vortex"],
        report,
        accepted,
    )
    assert res["stale"] == []
    assert [f["name"] for f in res["unexpected"]] == ["(all)"]


@pytest.fixture
def edition_reader(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.setitem(sys.modules, "vortex", SimpleNamespace())
    path = Path(__file__).resolve().parents[1] / "scripts" / "edition_reader.py"
    spec = importlib.util.spec_from_file_location("vortex_compat_edition_reader_under_test", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, mod)
    spec.loader.exec_module(mod)
    return mod


def test_edition_reader_map_keys_sorted_mismatch_still_checks_values(compat, edition_reader):
    expected_type = pa.map_(pa.field("key", pa.string_view(), nullable=False), pa.int64(), keys_sorted=True)
    actual_type = pa.map_(pa.field("key", pa.string_view(), nullable=False), pa.int64(), keys_sorted=False)
    expected = pa.table(
        {"sorted_attrs": pa.array([[("a", 1), ("b", 2)]], type=expected_type)},
        schema=pa.schema([pa.field("sorted_attrs", expected_type, nullable=False)]),
    )
    actual_same = pa.table(
        {"sorted_attrs": pa.array([[("a", 1), ("b", 2)]], type=actual_type)},
        schema=pa.schema([pa.field("sorted_attrs", actual_type, nullable=False)]),
    )
    actual_wrong = pa.table(
        {"sorted_attrs": pa.array([[("a", 9), ("b", 2)]], type=actual_type)},
        schema=pa.schema([pa.field("sorted_attrs", actual_type, nullable=False)]),
    )

    same_causes = edition_reader._tables_equal(actual_same, expected)
    assert len(same_causes) == 1
    assert same_causes[0].startswith("type mismatch in column sorted_attrs:")

    wrong_causes = edition_reader._tables_equal(actual_wrong, expected)
    assert any(c.startswith("type mismatch in column sorted_attrs:") for c in wrong_causes)
    assert any(c.startswith("value mismatch in column sorted_attrs") for c in wrong_causes)

    accepted = {
        ("core2026.08.2", "0.85.0", "map.vortex"): {
            "expected_causes": [same_causes[0]],
            "reason": "known keys_sorted drop on 0.85.0",
        }
    }
    res_same = compat._evaluate_edition_results(
        "core2026.08.2",
        "0.85.0",
        ["map.vortex"],
        {"passed": [], "failed": [{"name": "map.vortex", "error": same_causes[0], "causes": same_causes}]},
        accepted,
    )
    assert res_same["unexpected"] == []
    assert len(res_same["accepted"]) == 1

    res_wrong = compat._evaluate_edition_results(
        "core2026.08.2",
        "0.85.0",
        ["map.vortex"],
        {"passed": [], "failed": [{"name": "map.vortex", "error": "; ".join(wrong_causes), "causes": wrong_causes}]},
        accepted,
    )
    assert len(res_wrong["unexpected"]) == 1


def test_edition_reader_probe_column_causes_isolates_corrupt_expected_ipc(edition_reader, monkeypatch, tmp_path):
    corrupt_ipc = tmp_path / "broken.vortex.arrow"
    corrupt_ipc.write_bytes(b"not a valid arrow ipc file")
    monkeypatch.setattr(edition_reader, "_get_struct_columns", lambda _p: {"ok": True, "columns": ["a", "b"]})
    assert edition_reader._probe_column_causes(tmp_path / "broken.vortex", corrupt_ipc) == []

    monkeypatch.delattr(edition_reader.os, "fork", raising=False)
    assert edition_reader._probe_column_causes(tmp_path / "broken.vortex", corrupt_ipc) == []

    for exc_type in (KeyboardInterrupt, SystemExit):
        with pytest.raises(exc_type):
            edition_reader._run_isolated(lambda e=exc_type: (_ for _ in ()).throw(e()))
