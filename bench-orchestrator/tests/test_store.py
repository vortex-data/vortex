# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Tests for benchmark ResultStore run reservation and latest symlink publication."""

import json
from datetime import datetime
from pathlib import Path

import pytest
from bench_orchestrator.config import Benchmark, BenchmarkTarget, BuildConfig, Engine, Format, RunConfig
from bench_orchestrator.storage import store as store_module
from bench_orchestrator.storage.store import ResultStore


class _FrozenDatetime(datetime):
    """Datetime subclass returning a fixed timestamp from now()."""

    frozen_now = datetime(2026, 11, 21, 12, 0, 0)

    @classmethod
    def now(cls, tz: object = None) -> datetime:
        return cls.frozen_now


def _make_configs(label: str | None = "same-label") -> tuple[RunConfig, BuildConfig]:
    config = RunConfig(
        benchmark=Benchmark.TPCH,
        targets=[BenchmarkTarget(engine=Engine.DATAFUSION, format=Format.PARQUET)],
        label=label,
    )
    build_config = BuildConfig()
    return config, build_config


def _patch_store_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(store_module, "datetime", _FrozenDatetime)
    monkeypatch.setattr(store_module, "_get_git_info", lambda: ("deadbeef", "develop", False))


def test_same_second_runs_preserve_both_results_and_metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_store_env(monkeypatch)
    store = ResultStore(base_dir=tmp_path / "runs")
    config, build_config = _make_configs()

    with store.create_run(config, build_config) as ctx1:
        run_id_1 = ctx1.metadata.run_id
        ctx1.write_raw_json(json.dumps({"query": "q1", "run": 1}))

    with store.create_run(config, build_config) as ctx2:
        run_id_2 = ctx2.metadata.run_id
        ctx2.write_raw_json(json.dumps({"query": "q1", "run": 2}))

    assert run_id_1 != run_id_2
    assert run_id_1.startswith("2026-11-21T12-00-00_tpch_same-label")
    assert run_id_2.startswith("2026-11-21T12-00-00_tpch_same-label")

    meta1 = store.get_run(run_id_1)
    meta2 = store.get_run(run_id_2)
    assert meta1 is not None and meta1.run_id == run_id_1
    assert meta2 is not None and meta2.run_id == run_id_2

    df1 = store.load_results(run_id_1)
    df2 = store.load_results(run_id_2)
    assert df1["run"].tolist() == [1]
    assert df2["run"].tolist() == [2]

    latest = store.get_run("latest")
    assert latest is not None and latest.run_id == run_id_2


def test_failed_run_is_partial_and_does_not_overwrite_latest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_store_env(monkeypatch)
    store = ResultStore(base_dir=tmp_path / "runs")
    config, build_config = _make_configs()

    with store.create_run(config, build_config) as ok_ctx:
        ok_id = ok_ctx.metadata.run_id
        ok_ctx.write_raw_json(json.dumps({"status": "ok"}))

    failed_id: str | None = None
    with pytest.raises(RuntimeError, match="boom"):
        with store.create_run(config, build_config) as bad_ctx:
            failed_id = bad_ctx.metadata.run_id
            bad_ctx.write_raw_json(json.dumps({"status": "partial"}))
            raise RuntimeError("boom")

    assert failed_id is not None and failed_id != ok_id
    failed_meta = store.get_run(failed_id)
    assert failed_meta is not None
    assert failed_meta.partial is True
    assert store.load_results(failed_id)["status"].tolist() == ["partial"]

    latest = store.get_run("latest")
    assert latest is not None and latest.run_id == ok_id


def test_slash_label_runs_are_flat_and_found_by_label(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_store_env(monkeypatch)
    store = ResultStore(base_dir=tmp_path / "runs")
    config, build_config = _make_configs(label="feature/experiment")

    with store.create_run(config, build_config) as ctx1:
        run_id_1 = ctx1.metadata.run_id
        ctx1.write_raw_json(json.dumps({"run": 1}))

    with store.create_run(config, build_config) as ctx2:
        run_id_2 = ctx2.metadata.run_id
        ctx2.write_raw_json(json.dumps({"run": 2}))

    assert run_id_1 != run_id_2
    assert store.load_results(run_id_1)["run"].tolist() == [1]
    assert store.load_results(run_id_2)["run"].tolist() == [2]

    latest = store.get_run("latest")
    assert latest is not None
    assert latest.run_id == run_id_2

    by_label = store.get_run("feature/experiment")
    assert by_label is not None
    assert by_label.label == "feature/experiment"

    listed_ids = {r.run_id for r in store.list_runs()}
    assert listed_ids == {run_id_1, run_id_2}


def test_empty_store_latest_fallback_excludes_partial_runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_store_env(monkeypatch)
    store = ResultStore(base_dir=tmp_path / "runs")
    config, build_config = _make_configs(label="first-fail")

    partial_id = ""
    with pytest.raises(RuntimeError, match="boom"), store.create_run(config, build_config) as ctx:
        partial_id = ctx.metadata.run_id
        ctx.write_raw_json(json.dumps({"status": "partial"}))
        raise RuntimeError("boom")

    assert not (store.base_dir / "latest").exists()
    assert store.get_run("latest") is None

    partial_meta = store.get_run(partial_id)
    assert partial_meta is not None
    assert partial_meta.partial is True


def test_malformed_metadata_is_skipped_by_discovery_and_lookups(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_store_env(monkeypatch)
    store = ResultStore(base_dir=tmp_path / "runs")
    config, build_config = _make_configs(label="good-run")

    with store.create_run(config, build_config) as ctx:
        valid_id = ctx.metadata.run_id
        ctx.write_raw_json(json.dumps({"query": "q1", "value": 1}))

    # Place a malformed metadata.json (JSON null) and an offset-aware timestamp record beside the
    # valid naive record, then remove the latest symlink so discovery and fallback must handle both.
    bad_dir = store.base_dir / "2026-11-21T12-00-01_tpch_bad"
    bad_dir.mkdir()
    (bad_dir / "metadata.json").write_text("null\n", encoding="utf-8")

    aware_dir = store.base_dir / "2026-11-21T12-00-02_tpch_aware"
    aware_dir.mkdir()
    aware_data = json.loads((store.base_dir / valid_id / "metadata.json").read_text(encoding="utf-8"))
    aware_data["run_id"] = aware_dir.name
    aware_data["label"] = "aware-run"
    aware_data["timestamp"] = f"{aware_data['timestamp']}+00:00"
    (aware_dir / "metadata.json").write_text(json.dumps(aware_data), encoding="utf-8")

    (store.base_dir / "latest").unlink()

    runs = store.list_runs()
    assert [r.run_id for r in runs] == [valid_id]

    latest = store.get_run("latest")
    assert latest is not None and latest.run_id == valid_id

    by_label = store.get_run("good-run")
    assert by_label is not None and by_label.run_id == valid_id

    assert store.get_run(bad_dir.name) is None
    assert store.get_run(aware_dir.name) is None
    assert store.get_run("aware-run") is None
