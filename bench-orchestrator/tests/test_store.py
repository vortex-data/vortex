# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Tests for benchmark ResultStore run reservation and latest symlink publication."""

import json
import threading
from datetime import datetime
from pathlib import Path

import pytest
from bench_orchestrator.config import Benchmark, BenchmarkTarget, BuildConfig, Engine, Format, RunConfig
from bench_orchestrator.storage import store as store_module
from bench_orchestrator.storage.store import ResultStore, RunContext


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


def test_concurrent_stores_reserve_distinct_runs_and_publish_latest_in_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_store_env(monkeypatch)
    base_dir = tmp_path / "runs"
    store_a = ResultStore(base_dir=base_dir)
    store_b = ResultStore(base_dir=base_dir)
    config, build_config = _make_configs()

    start_barrier = threading.Barrier(2)
    first_done = threading.Event()
    results: dict[str, str] = {}
    errors: list[BaseException] = []

    def worker(name: str, store: ResultStore, wait_for_first: bool) -> None:
        try:
            start_barrier.wait(timeout=5)
            with store.create_run(config, build_config) as ctx:
                results[name] = ctx.metadata.run_id
                ctx.write_raw_json(json.dumps({"worker": name}))
                if wait_for_first:
                    assert first_done.wait(timeout=5)
            if not wait_for_first:
                first_done.set()
        except BaseException as exc:  # pragma: no cover - surfaced in assertion
            errors.append(exc)

    t1 = threading.Thread(target=worker, args=("first", store_a, False))
    t2 = threading.Thread(target=worker, args=("second", store_b, True))
    t1.start()
    t2.start()
    t1.join(timeout=10)
    t2.join(timeout=10)

    assert not errors
    assert results["first"] != results["second"]

    for name in ("first", "second"):
        run_id = results[name]
        meta = store_a.get_run(run_id)
        assert meta is not None and meta.run_id == run_id
        df = store_a.load_results(run_id)
        assert df["worker"].tolist() == [name]

    # second completed after first, so latest must point to second
    latest = store_a.get_run("latest")
    assert latest is not None and latest.run_id == results["second"]


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


def test_run_context_rejects_existing_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_store_env(monkeypatch)
    store = ResultStore(base_dir=tmp_path / "runs")
    config, build_config = _make_configs()

    with store.create_run(config, build_config) as ctx:
        existing_dir = ctx.run_dir
        existing_meta = ctx.metadata
        ctx.write_raw_json(json.dumps({"preserved": True}))

    with pytest.raises(FileExistsError):
        with RunContext(existing_dir, existing_meta):
            pass

    assert store.load_results(existing_meta.run_id)["preserved"].tolist() == [True]


def test_temporary_symlinks_excluded_from_run_discovery(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_store_env(monkeypatch)
    store = ResultStore(base_dir=tmp_path / "runs")
    config, build_config = _make_configs()

    with store.create_run(config, build_config) as ctx:
        run_id = ctx.metadata.run_id
        ctx.write_raw_json(json.dumps({"v": 1}))

    # Simulate a leftover temporary symlink next to latest
    (store.base_dir / ".latest.12345.tmp").symlink_to(run_id)

    runs = store.list_runs()
    assert [r.run_id for r in runs] == [run_id]


def test_slash_label_creates_intermediate_directories(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_store_env(monkeypatch)
    store = ResultStore(base_dir=tmp_path / "runs")
    config, build_config = _make_configs(label="feature/new-scan")

    with store.create_run(config, build_config) as ctx1:
        run_id_1 = ctx1.metadata.run_id
        ctx1.write_raw_json(json.dumps({"run": 1}))

    with store.create_run(config, build_config) as ctx2:
        run_id_2 = ctx2.metadata.run_id
        ctx2.write_raw_json(json.dumps({"run": 2}))

    assert run_id_1 != run_id_2
    assert store.load_results(run_id_1)["run"].tolist() == [1]
    assert store.load_results(run_id_2)["run"].tolist() == [2]
