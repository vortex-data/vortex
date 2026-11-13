# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Tests for the agent check runner in `scripts/agent_checks.py`.

Run with:

    uv run --no-project --with pytest pytest scripts/tests/test_agent_checks.py
"""

import datetime as dt
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / "scripts" / "agent_checks.py"
NOW = dt.datetime(2026, 11, 13, 12, 0, tzinfo=dt.UTC)


def load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("agent_checks", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["agent_checks"] = module
    spec.loader.exec_module(module)
    return module


ac = load_module()

CHECK_MD = """\
<!-- SPDX-License-Identifier: CC-BY-4.0 -->
+++
id = "bench"
title = "Bench check"
description = "Reviews benches."
include = ["**/benches/*.rs"]
exclude = ["skip/**"]
max_files_per_shard = 2
max_shards_per_run = 10
cadence_days = 7
labels = ["agent-check"]
+++

Review the benchmarks.
"""


def sh(root: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)


def write(root: Path, path: str, text: str) -> None:
    (root / path).parent.mkdir(parents=True, exist_ok=True)
    (root / path).write_text(text)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    sh(root, "init", "-q", "-b", "main")
    sh(root, "config", "user.email", "test@example.com")
    sh(root, "config", "user.name", "Test")
    write(root, ".agents/checks/bench/CHECK.md", CHECK_MD)
    write(root, "Cargo.toml", "[workspace]\nmembers = []\n")
    write(root, "alpha/Cargo.toml", '[package]\nname = "alpha"\n')
    write(root, "enc/beta/Cargo.toml", '[package]\nname = "beta"\n')
    write(root, "skip/Cargo.toml", '[package]\nname = "skip"\n')
    for name in ("a", "b", "c"):
        write(root, f"alpha/benches/{name}.rs", f"fn {name}() {{}}\n")
    write(root, "enc/beta/benches/x.rs", "fn x() {}\n")
    write(root, "enc/beta/src/lib.rs", "fn lib() {}\n")
    write(root, "skip/benches/s.rs", "fn s() {}\n")
    sh(root, "add", "-A")
    sh(root, "commit", "-qm", "init")
    return root


def plan(
    root: Path,
    state_dir: Path,
    prs: list[dict[str, Any]] | None = None,
    max_shards: int | None = None,
    force: bool = False,
) -> dict[str, Any]:
    checks = list(ac.load_checks(root).values())
    return ac.build_plan(checks, root, state_dir, now=NOW, prs=prs, max_shards=max_shards, force=force)


def shard_files(p) -> dict[str, list[str]]:
    return {s["id"]: [f["path"] for f in s["files"]] for s in p["checks"]["bench"]["shards"]}


class TestParse:
    def test_parses_frontmatter_and_body(self, repo: Path):
        check = ac.load_checks(repo)["bench"]
        assert check.include == ("**/benches/*.rs",)
        assert check.max_files_per_shard == 2
        assert check.body == "Review the benchmarks.\n"
        assert "Read" in check.tools

    @pytest.mark.parametrize(
        ("frontmatter", "error"),
        [
            ('id = "x"\ntitle = "t"\ndescription = "d"\ninclude = []\nbogus = 1', "unknown frontmatter keys"),
            ('id = "x"\ntitle = "t"\ndescription = "d"', "missing frontmatter keys"),
            ('id = "x"\ntitle = "t"\ndescription = "d"\ninclude = []\nmode = "sometimes"', "`mode` must be one of"),
            ('id = "x"\ntitle = "t"\ndescription = "d"\ninclude = []\nmax_turns = true', "`max_turns` must be int"),
            ('id = "X Y"\ntitle = "t"\ndescription = "d"\ninclude = []', "kebab-case"),
        ],
    )
    def test_rejects_invalid_frontmatter(self, frontmatter: str, error: str):
        with pytest.raises(ac.CheckError, match=error):
            ac.parse_check(f"+++\n{frontmatter}\n+++\nbody\n")

    def test_rejects_missing_frontmatter(self):
        with pytest.raises(ac.CheckError, match="missing `\\+\\+\\+`"):
            ac.parse_check("just a body\n")


@pytest.mark.parametrize(
    ("pattern", "path", "matches"),
    [
        ("**/benches/*.rs", "alpha/benches/a.rs", True),
        ("**/benches/*.rs", "enc/beta/benches/x.rs", True),
        ("**/benches/*.rs", "benches/x.rs", True),
        ("**/benches/*.rs", "alpha/benches/sub/a.rs", False),
        ("**/benches/**/*.rs", "alpha/benches/sub/a.rs", True),
        ("*.rs", "alpha/a.rs", False),
        ("**/{tests,benches}/*.rs", "a/tests/t.rs", True),
        ("**/{tests,benches}/*.rs", "a/src/t.rs", False),
    ],
)
def test_glob(pattern: str, path: str, matches: bool):
    assert bool(ac.glob_to_regex(pattern).match(path)) is matches


class TestPlan:
    def test_one_shard_per_crate_with_the_rest_deferred(self, repo: Path, tmp_path: Path):
        p = plan(repo, tmp_path / "state")
        # alpha has 3 pending files but a budget of 2: one shard, and c.rs waits for a later run.
        assert shard_files(p) == {
            "alpha": ["alpha/benches/a.rs", "alpha/benches/b.rs"],
            "beta": ["enc/beta/benches/x.rs"],
        }
        alpha = ac.find_shard(p, "bench", "alpha")
        assert alpha["deferred"] == 1
        assert alpha["branch"] == "agent-checks/bench/alpha"
        beta = ac.find_shard(p, "bench", "beta")
        assert (beta["crate"], beta["crate_dir"]) == ("beta", "enc/beta")
        assert ac.matrix(p)["include"] == [{"check": "bench", "shard": "alpha"}, {"check": "bench", "shard": "beta"}]

    def test_deferred_files_come_first_next_run(self, repo: Path, tmp_path: Path):
        state_dir = tmp_path / "state"
        p = plan(repo, state_dir)
        shard = ac.find_shard(p, "bench", "alpha")
        results = {"reviewed": ["alpha/benches/a.rs", "alpha/benches/b.rs"], "findings": []}
        out = ac.finalize(ac.load_checks(repo)["bench"], p, shard, results, repo, now=NOW)
        ac.record(state_dir, [out["state_update"]], now=NOW)
        assert shard_files(plan(repo, state_dir))["alpha"] == ["alpha/benches/c.rs"]

    def test_max_shards_and_busy_branches(self, repo: Path, tmp_path: Path):
        prs = [{"number": 1, "state": "OPEN", "headRefName": "agent-checks/bench/alpha", "body": ""}]
        assert list(shard_files(plan(repo, tmp_path / "state", prs=prs))) == ["beta"]
        assert list(shard_files(plan(repo, tmp_path / "state", max_shards=1))) == ["alpha"]

    def test_files_claimed_by_open_prs_are_skipped(self, repo: Path, tmp_path: Path):
        marker = '<!-- agent-check: {"check": "other", "shard": "x", "claims": ["alpha/benches/a.rs"]} -->'
        prs = [{"number": 1, "state": "OPEN", "headRefName": "agent-checks/other/alpha", "body": marker}]
        assert shard_files(plan(repo, tmp_path / "state", prs=prs))["alpha"] == [
            "alpha/benches/b.rs",
            "alpha/benches/c.rs",
        ]

    def test_crate_scope_claims_the_crate_across_checks(self, repo: Path, tmp_path: Path):
        marker = '<!-- agent-check: {"check": "other", "shard": "beta", "claims": ["enc/beta"]} -->'
        prs = [{"number": 1, "state": "OPEN", "headRefName": "agent-checks/other/beta", "body": marker}]
        assert "beta" not in shard_files(plan(repo, tmp_path / "state", prs=prs))

        write(
            repo,
            ".agents/checks/bench/CHECK.md",
            CHECK_MD.replace("max_files_per_shard", 'edit_scope = "crate"\nmax_files_per_shard'),
        )
        assert ac.find_shard(plan(repo, tmp_path / "state"), "bench", "beta")["edit_scope"] == ["enc/beta"]

    def test_reviewed_files_are_skipped_until_they_change(self, repo: Path, tmp_path: Path):
        state_dir = tmp_path / "state"
        p = plan(repo, state_dir)
        shard = ac.find_shard(p, "bench", "beta")
        results = {"reviewed": ["enc/beta/benches/x.rs"], "findings": [], "summary": ""}
        out = ac.finalize(ac.load_checks(repo)["bench"], p, shard, results, repo, now=NOW)
        ac.record(state_dir, [out["state_update"]], planned=["bench"], now=NOW)

        # Cadence: nothing runs again within cadence_days unless forced.
        assert shard_files(plan(repo, state_dir)) == {}
        assert "beta" not in shard_files(plan(repo, state_dir, force=True))

        write(repo, "enc/beta/benches/x.rs", "fn x() { changed() }\n")
        sh(repo, "commit", "-qam", "change")
        again = ac.find_shard(plan(repo, state_dir, force=True), "bench", "beta")
        assert again["files"][0]["status"] == "changed"
        assert again["files"][0]["last_commit"] == p["base_commit"]

    def test_version_bump_requeues_unchanged_files(self, repo: Path, tmp_path: Path):
        state_dir = tmp_path / "state"
        p = plan(repo, state_dir)
        check = ac.load_checks(repo)["bench"]
        shard = ac.find_shard(p, "bench", "beta")
        out = ac.finalize(check, p, shard, {"reviewed": ["enc/beta/benches/x.rs"], "findings": []}, repo, now=NOW)
        ac.record(state_dir, [out["state_update"]], now=NOW)

        write(repo, ".agents/checks/bench/CHECK.md", CHECK_MD.replace("cadence_days = 7", "version = 2"))
        status = ac.find_shard(plan(repo, state_dir), "bench", "beta")["files"][0]["status"]
        assert status == "rubric"

    def test_content_match_filters_files(self, repo: Path, tmp_path: Path):
        write(
            repo,
            ".agents/checks/bench/CHECK.md",
            CHECK_MD.replace('include = ["**/benches/*.rs"]', 'include = ["**/*.rs"]\ncontent_match = "fn (a|lib)"'),
        )
        files = [path for paths in shard_files(plan(repo, tmp_path / "state")).values() for path in paths]
        assert files == ["alpha/benches/a.rs", "enc/beta/src/lib.rs"]


class TestFinalize:
    def finalize(
        self, repo: Path, results, shard_id: str = "beta", revert_out_of_scope: bool = True, run_verify: bool = False
    ) -> dict[str, Any]:
        p = plan(repo, Path(repo) / "no-state")
        p["checks"]["bench"]["suppressed"] = {"bench:old": {"file": "enc/beta/benches/x.rs", "pr": 9}}
        shard = ac.find_shard(p, "bench", shard_id)
        check = ac.load_checks(repo)["bench"]
        return ac.finalize(
            check,
            p,
            shard,
            ac.validate_results(results),
            repo,
            now=NOW,
            revert_out_of_scope=revert_out_of_scope,
            run_verify=run_verify,
        )

    def finding(self, **overrides: object) -> dict[str, object]:
        base = {
            "file": "enc/beta/benches/x.rs",
            "line": 1,
            "severity": "medium",
            "category": "missing-black-box",
            "title": "Result is not black-boxed",
            "detail": "The compiler can drop the work.",
            "fingerprint": "bench:x",
            "fixed": True,
        }
        return {**base, **overrides}

    def test_filters_findings_and_scopes_edits(self, repo: Path):
        write(repo, "enc/beta/benches/x.rs", "fn x() { black_box(()) }\n")
        write(repo, "alpha/benches/a.rs", "fn a() { sneaky() }\n")
        results = {
            "reviewed": ["enc/beta/benches/x.rs", "alpha/benches/a.rs"],
            "findings": [
                self.finding(severity="low", fingerprint="bench:low", fixed=False),
                self.finding(),
                self.finding(fingerprint="bench:old"),
                self.finding(file="alpha/benches/a.rs", fingerprint="bench:a"),
                self.finding(severity="critical", fingerprint="bench:bad"),
            ],
            "summary": "Black-boxed a result.",
        }
        out = self.finalize(repo, results)

        assert [f["fingerprint"] for f in out["findings"]] == ["bench:x", "bench:low"]
        assert any("malformed finding 4" in n for n in out["notes"])
        assert out["edits"] == ["enc/beta/benches/x.rs"]
        assert (repo / "alpha/benches/a.rs").read_text() == "fn a() {}\n"
        record = out["state_update"]["files"]["enc/beta/benches/x.rs"]
        assert len(record["blobs"]) == 2  # pre-fix and post-fix blobs both count as reviewed
        assert list(out["state_update"]["files"]) == ["enc/beta/benches/x.rs"]
        assert "### Fixed in this PR" in out["body"]
        assert "### Not fixed" in out["body"]
        [marker] = ac.parse_markers(out["body"])
        assert marker["fingerprints"] == {"bench:x": "enc/beta/benches/x.rs", "bench:low": "enc/beta/benches/x.rs"}
        assert marker["claims"] == ["enc/beta/benches/x.rs"]

    def test_partial_results_keep_finished_files_and_revert_the_rest(self, repo: Path):
        # The agent finished a.rs, then hit its turn limit halfway through editing b.rs.
        write(repo, "alpha/benches/a.rs", "fn a() { fixed() }\n")
        write(repo, "alpha/benches/b.rs", "fn b() { half_do\n")
        results = {"reviewed": ["alpha/benches/a.rs"], "findings": []}
        out = self.finalize(repo, results, shard_id="alpha")
        assert out["edits"] == ["alpha/benches/a.rs"]
        assert (repo / "alpha/benches/b.rs").read_text() == "fn b() {}\n"
        assert list(out["state_update"]["files"]) == ["alpha/benches/a.rs"]

    @pytest.mark.parametrize(("command", "kept"), [("true {crate}", True), ("false {crate}", False)])
    def test_verify_failure_reverts_all_edits(self, repo: Path, command: str, kept: bool):
        write(
            repo,
            ".agents/checks/bench/CHECK.md",
            CHECK_MD.replace("cadence_days = 7", f'verify = ["{command}"]'),
        )
        write(repo, "enc/beta/benches/x.rs", "fn x() { broken( }\n")
        results = {"reviewed": ["enc/beta/benches/x.rs"], "findings": [self.finding()]}
        out = self.finalize(repo, results, run_verify=True)
        assert bool(out["edits"]) is kept
        assert out["findings"][0]["fixed"] is kept
        assert ((repo / "enc/beta/benches/x.rs").read_text() == "fn x() {}\n") is not kept
        # The review still counts: the files are not offered again until they change.
        assert list(out["state_update"]["files"]) == ["enc/beta/benches/x.rs"]

    @pytest.mark.parametrize(
        "results",
        [
            [],
            {"findings": []},
            {"reviewed": "x.rs", "findings": []},
            {"reviewed": [], "findings": [], "summary": 3},
        ],
    )
    def test_rejects_malformed_results(self, results):
        with pytest.raises(ac.CheckError):
            ac.validate_results(results)

    def test_drops_only_the_malformed_findings(self):
        results = {"reviewed": [], "findings": [self.finding(), self.finding(severity="critical"), {"file": "x"}]}
        validated = ac.validate_results(results)
        assert len(validated["findings"]) == 1
        assert len(validated["invalid"]) == 2


def test_closed_unmerged_prs_become_suppressions(tmp_path: Path):
    marker = '<!-- agent-check: {"check": "bench", "shard": "beta", "fingerprints": {"bench:x": "x.rs"}} -->'
    prs = [
        {"number": 1, "state": "CLOSED", "mergedAt": None, "body": marker, "headRefName": "a"},
        {"number": 2, "state": "MERGED", "mergedAt": "2026-11-01T00:00:00Z", "body": marker.replace("x", "y")},
    ]
    ac.record(tmp_path, [], prs=prs, now=NOW)
    state = json.loads((tmp_path / "bench.json").read_text())
    assert state["suppressed"] == {"bench:x": {"file": "x.rs", "pr": 1}}
    assert state["last_run"] is None


def test_prompt_lists_assignment_and_suppressions(repo: Path, tmp_path: Path):
    p = plan(repo, tmp_path / "state")
    p["checks"]["bench"]["suppressed"] = {"bench:old": {"file": "enc/beta/benches/x.rs", "pr": 9}}
    check = ac.load_checks(repo)["bench"]
    prompt = ac.render_prompt(check, p, ac.find_shard(p, "bench", "beta"), "/tmp/results.json")
    assert "Review the benchmarks." in prompt
    assert "`enc/beta/benches/x.rs` (1 lines): never reviewed" in prompt
    assert "`/tmp/results.json`" in prompt
    assert "Edits may touch only your assigned files." in prompt
    assert "`bench:old`" in prompt


def test_repository_checks_parse():
    checks = ac.load_checks(REPO_ROOT)
    assert checks, "expected at least one check under .agents/checks"
    for check in checks.values():
        assert check.include
