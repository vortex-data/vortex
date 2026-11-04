# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Tests for the CodSpeed shard planner in `scripts/codspeed_shards.py`.

Run with:

    uv run --no-project --with pytest pytest scripts/tests/test_codspeed_shards.py
"""

import importlib.util
import json
import re
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / "scripts" / "codspeed_shards.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("codspeed_shards", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


shards = _load()


def _targets(shard: dict[str, object]) -> set[str]:
    return set(re.findall(r"--bench (\S+)", str(shard["build-args"])))


def test_every_target_lands_in_exactly_one_shard_under_the_cap():
    counts = shards.load_counts()
    plan = shards.plan(counts, durations=shards.load_durations())
    seen = [name for shard in plan for name in _targets(shard)]
    assert sorted(seen) == sorted({name for _, name in counts})
    assert len(seen) == len(set(seen))
    assert all(shard["benchmarks"] <= shards.CAP for shard in plan)
    assert sum(shard["benchmarks"] for shard in plan) == sum(counts.values())


def test_allocator_exclusion_never_shares_a_shard():
    counts = {
        ("vortex-array", "take_primitive"): 10,
        ("vortex-array", "dict_compress"): 10,
        ("vortex-array", "compare"): 10,
        ("vortex-buffer", "buffer"): 10,
    }
    plan = shards.plan(counts, cap=100)
    excluded = [s for s in plan if s["exclude-allocations"]]
    assert set().union(*(_targets(s) for s in excluded)) == {"take_primitive", "dict_compress"}
    assert all(not (_targets(s) & {"take_primitive", "dict_compress"}) for s in plan if not s["exclude-allocations"])


def test_runtime_balancing_splits_a_package_even_below_the_count_cap():
    counts = {("vortex-buffer", name): 1 for name in "abcd"}
    durations = {
        ("vortex-buffer", name): seconds for name, seconds in zip("abcd", [100.0, 90.0, 10.0, 1.0], strict=True)
    }
    plan = shards.plan(counts, cap=100, durations=durations, max_shards=2)
    assert [_targets(s) for s in plan] == [{"a", "d"}, {"b", "c"}]
    assert [s["seconds"] for s in plan] == [101, 100]


def test_spare_slots_go_to_the_slower_allocation_pool():
    counts = {("vortex-buffer", name): 1 for name in "abcd"}
    counts[("vortex-array", "take_x")] = 1
    durations = {target: 40.0 for target in counts}
    durations[("vortex-array", "take_x")] = 10.0
    plan = shards.plan(counts, durations=durations)
    assert len(plan) == shards.MAX_SHARDS
    assert sum(not s["exclude-allocations"] for s in plan) == 4
    assert [_targets(s) for s in plan if s["exclude-allocations"]] == [{"take_x"}]


def test_small_packages_stay_together_when_they_fit_the_runtime_budget():
    counts = {("vortex-buffer", name): 1 for name in "ab"}
    counts.update({("vortex-fastlanes", name): 1 for name in "cd"})
    durations = {target: 10.0 if target[0] == "vortex-buffer" else 20.0 for target in counts}
    plan = shards.plan(counts, durations=durations, max_shards=2)
    by_target = {name: s["shard"] for s in plan for name in _targets(s)}
    assert by_target["a"] == by_target["b"]
    assert by_target["c"] != by_target["d"]


def test_runtime_balancing_respects_the_count_cap():
    counts = {("vortex-buffer", name): 60 for name in "abc"}
    durations = {target: 1.0 for target in counts}
    plan = shards.plan(counts, cap=100, durations=durations, max_shards=3)
    assert len(plan) == 3
    assert all(s["benchmarks"] == 60 for s in plan)


def test_new_target_uses_the_runtime_estimate():
    (shard,) = shards.plan({("vortex-buffer", "new"): 1}, durations={})
    assert shard["seconds"] == shards.DEFAULT_SECONDS


def test_build_args_name_only_the_features_of_selected_packages():
    counts = {("vortex-buffer", "a"): 1, ("vortex-fsst", "b"): 1}
    (shard,) = shards.plan(counts, cap=100, max_shards=1)
    assert (
        shard["build-args"]
        == "--features vortex-fsst/_test-harness -p vortex-buffer -p vortex-fsst --bench a --bench b"
    )

    (shard,) = shards.plan({("vortex-buffer", "a"): 1}, cap=100)
    assert shard["build-args"] == "-p vortex-buffer --bench a"


def test_target_over_the_cap_is_rejected():
    with pytest.raises(SystemExit, match="split it below"):
        shards.plan({("vortex-buffer", "a"): 101}, cap=100)


def test_more_shards_than_slots_is_rejected():
    counts = {("vortex-buffer", f"t{i}"): 100 for i in range(shards.MAX_SHARDS + 1)}
    with pytest.raises(SystemExit, match="slots"):
        shards.plan(counts, cap=100)


def test_github_outputs_key_shards_by_slot():
    plan = shards.plan({("vortex-buffer", "a"): 1, ("vortex-array", "take_x"): 1}, cap=100)
    outputs = shards.github_outputs(plan)
    assert json.loads(outputs["active"]) == [1, 2]
    by_slot = json.loads(outputs["shards"])
    assert sorted(by_slot) == ["1", "2"]
    assert by_slot["2"]["exclude-allocations"] is True


def test_workflow_matrix_has_max_shards_slots():
    workflow = (REPO_ROOT / ".github" / "workflows" / "codspeed.yml").read_text()
    slots = re.search(r"^\s+shard: \[([\d, ]+)\]$", workflow, re.MULTILINE)
    assert slots is not None
    assert [int(s) for s in slots.group(1).split(",")] == list(range(1, shards.MAX_SHARDS + 1))


def test_plan_is_deterministic():
    counts = shards.load_counts()
    durations = shards.load_durations()
    assert json.dumps(shards.plan(counts, durations=durations)) == json.dumps(
        shards.plan(dict(reversed(list(counts.items()))), durations=dict(reversed(list(durations.items()))))
    )


def test_counts_file_covers_every_discovered_target():
    known = shards.load_counts()
    assert set(shards.discover_targets()) <= set(known), "regenerate scripts/codspeed_bench_counts.json"


def test_counts_from_logs(tmp_path: Path):
    log = tmp_path / "job.txt"
    log.write_text(
        "2026-01-01T00:00:00.0Z Running vortex-buffer buffer\n"
        "2026-01-01T00:00:00.0Z Measured: vortex-buffer/benches/buffer.rs::a\n"
        "2026-01-01T00:00:00.0Z Measured: vortex-buffer/benches/buffer.rs::b\n"
        "2026-01-01T00:00:00.0Z Done running buffer\n"
        "2026-01-01T00:00:00.0Z Running post-execution phase...\n"
        "2026-01-01T00:00:00.0Z Measured: ignored\n"
    )
    assert shards.counts_from_logs([log]) == {"vortex-buffer": {"buffer": 2}}


def test_durations_include_the_whole_target_and_ignore_incomplete_runs(tmp_path: Path):
    log = tmp_path / "job.txt"
    log.write_text(
        "2026-01-01T00:00:00.0000000Z Running vortex-buffer buffer\n"
        "2026-01-01T00:00:02.0000000Z Measured: vortex-buffer/benches/buffer.rs::a\n"
        "2026-01-01T00:00:05.5000000Z Done running buffer\n"
        "2026-01-01T00:00:06.0000000Z Running post-execution phase...\n"
        "2026-01-01T00:00:07.0000000Z Running vortex-array widen\n"
    )
    assert shards.durations_from_logs([log]) == {"vortex-buffer": {"buffer": 5.5}}


def test_durations_use_the_median_of_complete_runs(tmp_path: Path):
    logs = []
    for seconds in [2, 4, 30]:
        log = tmp_path / f"{seconds}.txt"
        log.write_text(
            "2026-01-01T00:00:00.0000000Z Running vortex-datetime-parts split_temporal\n"
            f"2026-01-01T00:00:{seconds:02}.0000000Z Done running split_temporal\n"
        )
        logs.append(log)
    assert shards.durations_from_logs(logs) == {"vortex-datetime-parts": {"split_temporal": 4.0}}
