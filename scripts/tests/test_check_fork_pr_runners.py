# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Tests for `scripts/check_fork_pr_runners.py`.

Run with:

    uv run --no-project --with pytest --with pyyaml pytest scripts/tests/test_check_fork_pr_runners.py
"""

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / "scripts" / "check_fork_pr_runners.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("check_fork_pr_runners", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


check = _load()

TRUSTED = "(github.event_name != 'pull_request' || github.event.pull_request.head.repo.full_name == github.repository)"


# The module is loaded from a path, so ty sees its functions as untyped.
def _workflows() -> dict[str, dict[object, object]]:
    return check.load_workflows(REPO_ROOT)  # ty: ignore[unsound-return-statement]


def _job(workflows: dict[str, dict[object, object]], workflow: str, job: str) -> dict[object, object]:
    return check.jobs(workflows[workflow])[job]  # ty: ignore[unsound-return-statement]


def _problems_for(workflows: dict[str, dict[object, object]], workflow: str, job: str) -> list[str]:
    return [problem for problem in check.check_workflows(workflows) if problem.startswith(f"{workflow}: job `{job}`")]


def test_repository_workflows_pass():
    assert check.check_workflows(_workflows()) == []


def test_fallback_runs_on_without_trusted_guard_is_rejected():
    workflows = _workflows()
    job = _job(workflows, "ci.yml", "rust-lint")
    runs_on = job["runs-on"]
    assert isinstance(runs_on, str) and f"&& {TRUSTED}" in runs_on
    job["runs-on"] = runs_on.replace(f"&& {TRUSTED}", "")
    problems = _problems_for(workflows, "ci.yml", "rust-lint")
    assert len(problems) == 1 and "self-hosted branch of `runs-on`" in problems[0]


def test_self_hosted_only_job_without_trusted_condition_is_rejected():
    workflows = _workflows()
    job = _job(workflows, "ci.yml", "rust-test-linux-arm64")
    assert isinstance(job["if"], str) and TRUSTED in job["if"]
    del job["if"]
    problems = _problems_for(workflows, "ci.yml", "rust-test-linux-arm64")
    assert len(problems) == 1 and "can only run on a self-hosted runner" in problems[0]


def test_trusted_condition_behind_a_top_level_or_is_rejected():
    workflows = _workflows()
    job = _job(workflows, "ci.yml", "rust-test-linux-arm64")
    job["if"] = f"github.event.pull_request.draft != true || {TRUSTED}"
    assert len(_problems_for(workflows, "ci.yml", "rust-test-linux-arm64")) == 1


def test_trusted_condition_in_job_if_also_guards_a_fallback_job():
    workflows = _workflows()
    job = _job(workflows, "ci.yml", "rust-lint")
    runs_on = job["runs-on"]
    assert isinstance(runs_on, str)
    job["runs-on"] = runs_on.replace(f"&& {TRUSTED}", "")
    job["if"] = f"github.event.pull_request.draft != true && {TRUSTED}"
    assert _problems_for(workflows, "ci.yml", "rust-lint") == []


def test_runner_from_unreadable_matrix_needs_trusted_condition():
    workflows = _workflows()
    job = _job(workflows, "duckdb-r2.yml", "mirror")
    assert job["runs-on"] == "${{ matrix.runner }}"
    del job["if"]
    problems = _problems_for(workflows, "duckdb-r2.yml", "mirror")
    assert len(problems) == 1 and "matrix the check cannot see" in problems[0]


def test_runs_on_action_step_needs_runner_guard():
    workflows = _workflows()
    job = _job(workflows, "ci.yml", "rust-lint")
    guarded = [step for step in check.steps(job) if str(step.get("uses", "")).startswith("runs-on/action")]
    assert len(guarded) == 1
    del guarded[0]["if"]
    problems = _problems_for(workflows, "ci.yml", "rust-lint")
    assert len(problems) == 1 and "runs-on/action" in problems[0]


def test_label_triggered_workflows_are_exempt_until_they_run_on_pushes():
    workflows = _workflows()
    dispatch = workflows["pr-bench-dispatch.yml"]
    assert check.triggers(dispatch)["pull_request"] == {"types": ["labeled"], "branches": ["develop"]}
    assert "pr-bench-runner.yml" not in check.fork_reachable(workflows)

    check.triggers(dispatch)["pull_request"]["types"].append("synchronize")
    assert "pr-bench-runner.yml" in check.fork_reachable(workflows)
    assert _problems_for(workflows, "pr-bench-runner.yml", "build")


def test_pull_request_target_is_rejected():
    workflows = _workflows()
    workflows["bridge.yml"] = {"on": {"pull_request_target": {}}, "jobs": {}}
    assert check.check_workflows(workflows) == [
        "bridge.yml: uses the `pull_request_target` event, which runs fork code with secrets"
    ]


def test_github_hosted_only_jobs_are_ignored():
    workflows = _workflows()
    job = _job(workflows, "ci.yml", "license-check-and-audit-check")
    assert job["runs-on"] == "ubuntu-latest" and "if" in job
    del job["if"]
    assert _problems_for(workflows, "ci.yml", "license-check-and-audit-check") == []


@pytest.mark.parametrize(
    ("expr", "op", "expected"),
    [
        ("a && (b || c) && d", "&&", ["a ", " (b || c) ", " d"]),
        ("a && (b || c) && d", "||", ["a && (b || c) && d"]),
        ("x == 'a || b' || y", "||", ["x == 'a || b' ", " y"]),
        ("f(a && b) && c", "&&", ["f(a && b) ", " c"]),
    ],
)
def test_split_top_level(expr: str, op: str, expected: list[str]):
    assert check.split_top_level(expr, op) == expected


def test_self_hosted_guards_parses_the_fallback_shape():
    runs_on = (
        "${{ github.repository == 'vortex-data/vortex' && format('runs-on={0}', github.run_id) || 'ubuntu-latest' }}"
    )
    assert check.self_hosted_guards(runs_on) == [
        "github.repository == 'vortex-data/vortex'",
        "format('runs-on={0}', github.run_id)",
    ]
    assert check.self_hosted_guards("runs-on=${{ github.run_id }}/runner=gpu") is None
    assert check.self_hosted_guards("${{ matrix.runner }}") is None
    assert check.self_hosted_guards("${{ cond && 'ubuntu-latest' || format('runs-on={0}', github.run_id) }}") is None
