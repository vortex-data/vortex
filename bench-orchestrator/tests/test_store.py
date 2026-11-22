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


def test_soft_partial_run_does_not_replace_latest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_store_env(monkeypatch)
    store = ResultStore(base_dir=tmp_path / "runs")
    config, build_config = _make_configs(label="soft-fail")

    with store.create_run(config, build_config) as ctx_ok:
        ok_id = ctx_ok.metadata.run_id
        ctx_ok.write_raw_json(json.dumps({"status": "ok"}))

    with store.create_run(config, build_config) as ctx_partial:
        partial_id = ctx_partial.metadata.run_id
        ctx_partial.write_raw_json(json.dumps({"status": "partial"}))
        ctx_partial.metadata.partial = True

    latest = store.get_run("latest")
    assert latest is not None
    assert latest.run_id == ok_id

    partial_meta = store.get_run(partial_id)
    assert partial_meta is not None
    assert partial_meta.partial is True


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


def test_same_second_parent_and_slash_child_do_not_nest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_store_env(monkeypatch)
    fixed = datetime(2026, 4, 1, 12, 0, 0)

    class FixedDatetime(datetime):
        @classmethod
        def now(cls, tz=None) -> datetime:  # type: ignore[override]
            return fixed

    monkeypatch.setattr("bench_orchestrator.storage.store.datetime", FixedDatetime)

    store = ResultStore(base_dir=tmp_path / "runs")
    parent_cfg, build_cfg = _make_configs(label="parent")
    child_cfg, _ = _make_configs(label="parent/child")

    with store.create_run(parent_cfg, build_cfg) as parent_ctx:
        parent_id = parent_ctx.metadata.run_id
        parent_dir = parent_ctx.run_dir
        parent_ctx.write_raw_json(json.dumps({"run": "parent"}))

    with store.create_run(child_cfg, build_cfg) as child_ctx:
        child_id = child_ctx.metadata.run_id
        child_dir = child_ctx.run_dir
        child_ctx.write_raw_json(json.dumps({"run": "child"}))

    assert parent_dir.parent == store.base_dir
    assert child_dir.parent == store.base_dir
    assert not child_dir.is_relative_to(parent_dir)

    assert store.delete_run(parent_id) is True
    assert store.get_run(parent_id) is None
    child_meta = store.get_run(child_id)
    assert child_meta is not None
    assert child_meta.label == "parent/child"
    assert store.get_run("parent/child") is not None


def test_delete_run_preserves_legacy_nested_descendant(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_store_env(monkeypatch)
    store = ResultStore(base_dir=tmp_path / "runs")
    parent_cfg, build_cfg = _make_configs(label="parent")

    with store.create_run(parent_cfg, build_cfg) as parent_ctx:
        parent_id = parent_ctx.metadata.run_id
        parent_dir = parent_ctx.run_dir
        parent_ctx.write_raw_json(json.dumps({"run": "parent"}))

    # Simulate an older store where a slash-labeled run was nested under parent_dir.
    legacy_child_id = f"{parent_id}/child"
    legacy_child_dir = parent_dir / "child"
    legacy_child_dir.mkdir(parents=True)
    child_meta = json.loads((parent_dir / "metadata.json").read_text())
    child_meta["run_id"] = legacy_child_id
    child_meta["label"] = "parent/child"
    (legacy_child_dir / "metadata.json").write_text(json.dumps(child_meta))
    (legacy_child_dir / "results.jsonl").write_text(json.dumps({"run": "child"}) + "\n")

    assert store.delete_run(parent_id) is True
    assert store.get_run(parent_id) is None

    remaining = store.get_run(legacy_child_id)
    assert remaining is not None
    assert remaining.run_id == legacy_child_id
    assert len(store.load_results(legacy_child_id)) == 1


def test_latest_symlink_pointing_to_partial_run_falls_back(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_store_env(monkeypatch)
    store = ResultStore(base_dir=tmp_path / "runs")
    config, build_config = _make_configs(label="upgrade")

    with store.create_run(config, build_config) as ctx_ok:
        ok_id = ctx_ok.metadata.run_id
        ctx_ok.write_raw_json(json.dumps({"status": "ok"}))

    with store.create_run(config, build_config) as ctx_partial:
        partial_id = ctx_partial.metadata.run_id
        ctx_partial.write_raw_json(json.dumps({"status": "partial"}))
        ctx_partial.metadata.partial = True

    # Simulate an existing store upgraded from older code where latest pointed to a partial run.
    latest_link = store.base_dir / "latest"
    latest_link.unlink()
    latest_link.symlink_to(partial_id)

    latest = store.get_run("latest")
    assert latest is not None
    assert latest.run_id == ok_id
    assert latest.partial is False


def test_latest_recovery_prefers_latest_completion_time_for_overlapping_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_store_env(monkeypatch)
    times = iter(
        [
            datetime(2026, 4, 1, 12, 0, 0),  # start A
            datetime(2026, 4, 1, 12, 0, 5),  # start B
            datetime(2026, 4, 1, 12, 0, 10),  # finish B
            datetime(2026, 4, 1, 12, 0, 20),  # finish A
            datetime(2026, 4, 1, 12, 0, 30),  # start partial
            datetime(2026, 4, 1, 12, 0, 35),  # finish partial
        ]
    )

    class SequencedDatetime(datetime):
        @classmethod
        def now(cls, tz=None) -> datetime:  # type: ignore[override]
            return next(times)

    monkeypatch.setattr("bench_orchestrator.storage.store.datetime", SequencedDatetime)

    store = ResultStore(base_dir=tmp_path / "runs")
    cfg_a, build_cfg = _make_configs(label="run-a")
    cfg_b, _ = _make_configs(label="run-b")
    cfg_partial, _ = _make_configs(label="run-partial")

    with store.create_run(cfg_a, build_cfg) as ctx_a:
        id_a = ctx_a.metadata.run_id
        ctx_a.write_raw_json(json.dumps({"run": "a"}))
        with store.create_run(cfg_b, build_cfg) as ctx_b:
            id_b = ctx_b.metadata.run_id
            ctx_b.write_raw_json(json.dumps({"run": "b"}))

    # Live latest points to A (completed last), while list_runs preserves start-time order (B before A).
    live_latest = store.get_run("latest")
    assert live_latest is not None
    assert live_latest.run_id == id_a
    listed_ids = [r.run_id for r in store.list_runs(limit=None)]
    assert listed_ids == [id_b, id_a]

    with store.create_run(cfg_partial, build_cfg) as ctx_partial:
        partial_id = ctx_partial.metadata.run_id
        ctx_partial.metadata.partial = True

    latest_link = store.base_dir / "latest"
    latest_link.unlink()
    latest_link.symlink_to(partial_id)

    recovered = store.get_run("latest")
    assert recovered is not None
    assert recovered.run_id == id_a


def test_latest_recovery_skips_malformed_timestamp_candidate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_store_env(monkeypatch)
    times = iter(
        [
            datetime(2026, 4, 1, 12, 0, 0),  # start first
            datetime(2026, 4, 1, 12, 0, 1),  # finish first
            datetime(2026, 4, 1, 12, 0, 5),  # start second
            datetime(2026, 4, 1, 12, 0, 6),  # finish second
        ]
    )

    class SequencedDatetime(datetime):
        @classmethod
        def now(cls, tz=None) -> datetime:  # type: ignore[override]
            return next(times)

    monkeypatch.setattr("bench_orchestrator.storage.store.datetime", SequencedDatetime)

    store = ResultStore(base_dir=tmp_path / "runs")
    cfg_one, build_cfg = _make_configs(label="first")
    cfg_two, _ = _make_configs(label="second")

    with store.create_run(cfg_one, build_cfg) as ctx_one:
        ctx_one.write_raw_json(json.dumps({"run": 1}))

    with store.create_run(cfg_two, build_cfg) as ctx_two:
        id_two = ctx_two.metadata.run_id
        ctx_two.write_raw_json(json.dumps({"run": 2}))

    # Add an unrelated historical run with a malformed timestamp.
    corrupt_dir = store.base_dir / "2026-01-01T00-00-00_tpch_corrupt"
    corrupt_dir.mkdir()
    corrupt_meta = json.loads((store.base_dir / id_two / "metadata.json").read_text())
    corrupt_meta["run_id"] = corrupt_dir.name
    corrupt_meta["timestamp"] = "not-a-valid-timestamp"
    (corrupt_dir / "metadata.json").write_text(json.dumps(corrupt_meta))

    # Force recovery by removing the latest symlink.
    (store.base_dir / "latest").unlink()

    recovered = store.get_run("latest")
    assert recovered is not None
    assert recovered.run_id == id_two
    assert len(store.list_runs(limit=None)) == 2
