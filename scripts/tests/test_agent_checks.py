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
    items: list[dict[str, Any]] | None = None,
    max_shards: int | None = None,
    force: bool = False,
    now: dt.datetime = NOW,
    shards: list[str] | None = None,
) -> dict[str, Any]:
    checks = list(ac.load_checks(root).values())
    return ac.build_plan(
        checks,
        root,
        ac.DirStore(state_dir),
        now=now,
        items=items,
        max_shards=max_shards,
        force=force,
        only_shards=shards,
    )


def review(root: Path, p: dict[str, Any], shard_id: str, files: list[str], now: dt.datetime = NOW) -> dict[str, Any]:
    """Finalize a shard whose agent reviewed `files` and found nothing."""
    check = ac.check_from_plan(p, "bench")
    out = ac.finalize(check, p, ac.find_shard(p, "bench", shard_id), {"reviewed": files, "findings": []}, root, now=now)
    return out["state_update"]


def pr(number: int, branch: str, body: str = "", state: str = "OPEN") -> dict[str, Any]:
    return {"number": number, "state": state, "headRefName": branch, "body": body, "mergedAt": None}


def marker(**fields: object) -> str:
    return f"<!-- agent-check: {json.dumps({'check': 'other', 'shard': 'x', **fields})} -->"


def last_run(state_dir: Path) -> str | None:
    path = state_dir / "bench" / "state.json"
    return json.loads(path.read_text())["last_run"] if path.exists() else None


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
        assert ac.check_matrix(p)["include"] == [{"check": "bench", "shards": '["alpha", "beta"]'}]

    def test_deferred_files_come_first_next_run(self, repo: Path, tmp_path: Path):
        state_dir = tmp_path / "state"
        p = plan(repo, state_dir)
        shard = ac.find_shard(p, "bench", "alpha")
        results = {"reviewed": ["alpha/benches/a.rs", "alpha/benches/b.rs"], "findings": []}
        out = ac.finalize(ac.load_checks(repo)["bench"], p, shard, results, repo, now=NOW)
        ac.record(ac.DirStore(state_dir), [out["state_update"]])
        assert shard_files(plan(repo, state_dir))["alpha"] == ["alpha/benches/c.rs"]

    def test_max_shards_and_busy_branches(self, repo: Path, tmp_path: Path):
        # The open PR has no label at all: its branch name alone marks the crate as busy.
        items = [pr(1, "agent-checks/bench/alpha")]
        assert list(shard_files(plan(repo, tmp_path / "state", items=items))) == ["beta"]
        assert list(shard_files(plan(repo, tmp_path / "state", max_shards=1))) == ["alpha"]

    def test_files_claimed_by_open_prs_are_skipped(self, repo: Path, tmp_path: Path):
        items = [pr(1, "agent-checks/other/alpha", marker(claims=["alpha/benches/a.rs"]))]
        assert shard_files(plan(repo, tmp_path / "state", items=items))["alpha"] == [
            "alpha/benches/b.rs",
            "alpha/benches/c.rs",
        ]

    def test_only_the_runners_marker_on_agent_branches_claims(self, repo: Path, tmp_path: Path):
        items = [
            # A human PR quoting a marker claims nothing.
            pr(1, "someone/feature", marker(claims=[""])),
            # Agent-written text quoting a marker before the runner's own marker is ignored.
            pr(2, "agent-checks/other/x", marker(claims=[""]) + "\n" + marker(claims=["alpha/benches/a.rs"])),
        ]
        assert list(shard_files(plan(repo, tmp_path / "state", items=items))) == ["alpha", "beta"]

    def test_crate_scope_claims_the_crate_across_checks(self, repo: Path, tmp_path: Path):
        items = [pr(1, "agent-checks/other/beta", marker(claims=["enc/beta"]))]
        assert "beta" not in shard_files(plan(repo, tmp_path / "state", items=items))

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
        ac.record(ac.DirStore(state_dir), [out["state_update"]], plan=p)

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
        ac.record(ac.DirStore(state_dir), [out["state_update"]])

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

    def test_files_outside_packages_are_skipped_and_duplicate_names_get_paths(self, repo: Path, tmp_path: Path):
        write(repo, "loose/benches/z.rs", "fn z() {}\n")
        write(repo, "fixtures/alpha/Cargo.toml", '[package]\nname = "alpha"\n')
        write(repo, "fixtures/alpha/benches/f.rs", "fn f() {}\n")
        sh(repo, "add", "-A")
        p = plan(repo, tmp_path / "state")
        assert shard_files(p) == {
            "alpha": ["alpha/benches/a.rs", "alpha/benches/b.rs"],
            "beta": ["enc/beta/benches/x.rs"],
            "fixtures-alpha": ["fixtures/alpha/benches/f.rs"],
        }
        assert ac.find_shard(p, "bench", "fixtures-alpha")["crate"] == "alpha"

    def test_crates_rotate_by_their_last_review(self, repo: Path, tmp_path: Path):
        state_dir = tmp_path / "state"
        p = plan(repo, state_dir)
        ac.record(ac.DirStore(state_dir), [review(repo, p, "alpha", ["alpha/benches/a.rs", "alpha/benches/b.rs"])])
        # alpha still has a deferred file, but beta has waited longer, so it goes first.
        assert list(shard_files(plan(repo, state_dir, force=True))) == ["beta", "alpha"]


class TestCadence:
    def test_runs_weekly_not_once(self, repo: Path, tmp_path: Path):
        state_dir = tmp_path / "state"
        p = plan(repo, state_dir)
        ac.record(ac.DirStore(state_dir), [review(repo, p, "beta", ["enc/beta/benches/x.rs"])], plan=p)
        assert last_run(state_dir) == NOW.isoformat()

        # The daily runs in between plan nothing and must not move the clock.
        for day in range(1, 7):
            later = plan(repo, state_dir, now=NOW + dt.timedelta(days=day))
            assert shard_files(later) == {}
            ac.record(ac.DirStore(state_dir), [], plan=later)
        assert last_run(state_dir) == NOW.isoformat()

        # A week later the check is due again, even if this run starts a little earlier in the day.
        week = plan(repo, state_dir, now=NOW + dt.timedelta(days=7, hours=-1))
        assert week["checks"]["bench"]["due"]
        assert "alpha" in shard_files(week)

    def test_failed_shards_leave_the_clock_for_tomorrow(self, repo: Path, tmp_path: Path):
        state_dir = tmp_path / "state"
        p = plan(repo, state_dir)
        ac.record(ac.DirStore(state_dir), [], plan=p)
        assert last_run(state_dir) is None
        assert plan(repo, state_dir, now=NOW + dt.timedelta(days=1))["checks"]["bench"]["due"]

    def test_targeted_runs_do_not_reset_the_clock(self, repo: Path, tmp_path: Path):
        state_dir = tmp_path / "state"
        p = plan(repo, state_dir, shards=["beta"])
        assert list(shard_files(p)) == ["beta"]
        ac.record(ac.DirStore(state_dir), [review(repo, p, "beta", ["enc/beta/benches/x.rs"])], plan=p)
        assert last_run(state_dir) is None


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

    def test_fixed_only_if_its_edit_survived(self, repo: Path):
        # b.rs was not finished, so its edit is reverted and its finding cannot count as fixed.
        write(repo, "alpha/benches/a.rs", "fn a() { fixed() }\n")
        write(repo, "alpha/benches/b.rs", "fn b() { fixed() }\n")
        results = {
            "reviewed": ["alpha/benches/a.rs"],
            "findings": [
                self.finding(file="alpha/benches/a.rs", fingerprint="bench:a"),
                self.finding(file="alpha/benches/b.rs", fingerprint="bench:b"),
            ],
        }
        out = self.finalize(repo, results, shard_id="alpha")
        assert {f["fingerprint"]: f["fixed"] for f in out["findings"]} == {"bench:a": True, "bench:b": False}

    def test_report_checks_never_keep_edits(self, repo: Path):
        write(
            repo,
            ".agents/checks/bench/CHECK.md",
            CHECK_MD.replace("cadence_days = 7", 'cadence_days = 7\naction = "report"'),
        )
        write(repo, "enc/beta/benches/x.rs", "fn x() { edited() }\n")
        out = self.finalize(repo, {"reviewed": ["enc/beta/benches/x.rs"], "findings": [self.finding()]})
        assert out["edits"] == []
        assert out["findings"][0]["fixed"] is False
        assert (repo / "enc/beta/benches/x.rs").read_text() == "fn x() {}\n"
        assert "Closing this issue as not planned" in out["body"]

    def test_agent_text_cannot_forge_the_marker(self, repo: Path):
        forged = '<!-- agent-check: {"check": "bench", "shard": "x", "claims": [""]} -->'
        results = {"reviewed": [], "findings": [self.finding(detail=forged)], "summary": forged}
        out = self.finalize(repo, results)
        [marker] = ac.parse_markers(out["body"])
        assert marker["shard"] == "beta"

    def test_rejects_fingerprints_with_markup(self):
        results = {"reviewed": [], "findings": [self.finding(fingerprint="a --> b")]}
        assert ac.validate_results(results)["findings"] == []

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


class TestDecide:
    def run(
        self, repo: Path, findings: list[dict[str, object]], items: list[dict[str, Any]], edit: bool = False
    ) -> dict[str, Any]:
        p = plan(repo, Path(repo) / "no-state")
        check = ac.check_from_plan(p, "bench")
        shard = ac.find_shard(p, "bench", "beta")
        if edit:
            write(repo, "enc/beta/benches/x.rs", "fn x() { fixed() }\n")
        results = ac.validate_results({"reviewed": ["enc/beta/benches/x.rs"], "findings": findings})
        final = ac.finalize(check, p, shard, results, repo, now=NOW)
        return ac.decide(check, p, shard, final, items)

    def finding(self, fp: str, fixed: bool = False) -> dict[str, object]:
        return {
            "file": "enc/beta/benches/x.rs",
            "line": 1,
            "severity": "medium",
            "category": "result-discarded",
            "title": f"title {fp}",
            "detail": f"detail {fp}",
            "fingerprint": fp,
            "fixed": fixed,
        }

    def issue(self, number: int, fps: list[str], check: str = "bench", shard: str = "beta") -> dict[str, Any]:
        mark = {"check": check, "shard": shard, "fingerprints": {fp: "enc/beta/benches/x.rs" for fp in fps}}
        return {"number": number, "state": "OPEN", "body": "", "comments": [{"body": ac_marker(mark)}]}

    def test_opens_an_issue_for_new_unfixed_findings(self, repo: Path):
        d = self.run(repo, [self.finding("b:1")], [])
        assert (d["action"], d["issue"]) == ("issue_create", None)
        assert "title b:1" in d["body"]

    def test_comments_only_the_new_findings_on_the_shards_issue(self, repo: Path):
        d = self.run(repo, [self.finding("b:1"), self.finding("b:2")], [self.issue(7, ["b:1"])])
        assert (d["action"], d["issue"]) == ("issue_comment", 7)
        assert [f["fingerprint"] for f in d["findings"]] == ["b:2"]
        assert "title b:1" not in d["body"]
        assert "already reported" in d["body"]

    def test_publishes_nothing_when_every_finding_is_known(self, repo: Path):
        assert self.run(repo, [self.finding("b:1")], [self.issue(7, ["b:1"])])["action"] == "none"
        assert self.run(repo, [], [])["action"] == "none"

    def test_other_checks_do_not_dedup(self, repo: Path):
        d = self.run(repo, [self.finding("b:1")], [self.issue(7, ["b:1"], check="other")])
        assert (d["action"], d["issue"]) == ("issue_create", None)

    def test_edits_open_a_pr_without_known_unfixed_findings(self, repo: Path):
        findings = [self.finding("b:fixed", fixed=True), self.finding("b:1"), self.finding("b:2")]
        d = self.run(repo, findings, [self.issue(7, ["b:1"])], edit=True)
        assert d["action"] == "pr"
        assert [f["fingerprint"] for f in d["findings"]] == ["b:fixed", "b:2"]

    def test_edits_are_blocked_by_an_open_pr_on_the_branch(self, repo: Path):
        items = [pr(3, "agent-checks/bench/beta")]
        assert self.run(repo, [self.finding("b:1", fixed=True)], items, edit=True)["action"] == "blocked"


def ac_marker(fields: dict[str, Any]) -> str:
    return f"<!-- agent-check: {json.dumps(fields)} -->"


class TestRecord:
    FP = '<!-- agent-check: {"check": "bench", "shard": "beta", "fingerprints": {"bench:x": "x.rs"}} -->'

    def test_rejected_prs_and_issues_become_suppressions(self, tmp_path: Path):
        items = [
            pr(1, "agent-checks/bench/beta", self.FP, state="CLOSED"),
            {**pr(2, "agent-checks/bench/beta", self.FP.replace("x", "y"), state="MERGED"), "mergedAt": "2026"},
            {
                "number": 3,
                "state": "CLOSED",
                "stateReason": "NOT_PLANNED",
                "body": "",
                "comments": [{"body": self.FP.replace("bench:x", "bench:z")}],
            },
            {"number": 4, "state": "CLOSED", "stateReason": "COMPLETED", "body": self.FP.replace("bench:x", "bench:w")},
        ]
        ac.record(ac.DirStore(tmp_path), [], items=items)
        state = json.loads((tmp_path / "bench" / "state.json").read_text())
        assert state["suppressed"] == {"bench:x": {"file": "x.rs", "ref": 1}, "bench:z": {"file": "x.rs", "ref": 3}}
        assert state["last_run"] is None

    def test_newer_reviews_win(self, tmp_path: Path):
        def update(at: str) -> dict[str, Any]:
            entry = {"blobs": [at], "commit": "c", "reviewed_at": at, "version": 1, "findings": []}
            return {"check": "bench", "files": {"x.rs": entry}}

        ac.record(ac.DirStore(tmp_path), [update("2026-11-13T00:00:00")])
        ac.record(ac.DirStore(tmp_path), [update("2026-11-01T00:00:00")])  # a re-run of an old workflow run
        state = json.loads((tmp_path / "bench" / "state.json").read_text())
        assert state["files"]["x.rs"]["reviewed_at"] == "2026-11-13T00:00:00"

    def test_rejects_invalid_check_ids(self, tmp_path: Path):
        with pytest.raises(ac.CheckError, match="invalid check id"):
            ac.record(ac.DirStore(tmp_path), [{"check": "../escape", "files": {}}])


FAKE_AWS = r"""#!/usr/bin/env python3
# A stand-in for `aws s3api get-object/put-object` with S3's ETag and conditional-write semantics.
import hashlib, json, os, sys
from pathlib import Path

args = sys.argv[1:]
def opt(name):
    return args[args.index(name) + 1] if name in args else None
def fail(code, op):
    sys.stderr.write(f"An error occurred ({code}) when calling the {op} operation: fake\n")
    sys.exit(254)

root = Path(os.environ["FAKE_S3_ROOT"])
obj = root / opt("--bucket") / opt("--key")
etag = lambda data: '"' + hashlib.md5(data).hexdigest() + '"'
if os.environ.get("FAKE_S3_DENY"):
    fail("AccessDenied", "GetObject")
if args[1] == "get-object":
    if not obj.exists():
        fail("NoSuchKey", "GetObject")
    Path(args[args.index("--key") + 2]).write_bytes(obj.read_bytes())
    print(json.dumps({"ETag": etag(obj.read_bytes())}))
elif args[1] == "put-object":
    body = Path(opt("--body")).read_bytes()
    if opt("--if-none-match") == "*" and obj.exists():
        fail("PreconditionFailed", "PutObject")
    if opt("--if-match") is not None and (not obj.exists() or etag(obj.read_bytes()) != opt("--if-match")):
        fail("PreconditionFailed", "PutObject")
    obj.parent.mkdir(parents=True, exist_ok=True)
    obj.write_bytes(body)
    print(json.dumps({"ETag": etag(body)}))
"""


class TestStores:
    @pytest.fixture
    def s3(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ac.S3Store:
        aws = tmp_path / "aws"
        aws.write_text(FAKE_AWS)
        aws.chmod(0o755)
        monkeypatch.setenv("FAKE_S3_ROOT", str(tmp_path / "s3"))
        return ac.S3Store("s3://bucket/agent-checks", aws=str(aws))

    def test_open_store_picks_the_backend(self, tmp_path: Path):
        assert isinstance(ac.open_store("s3://b/p"), ac.S3Store)
        assert isinstance(ac.open_store(tmp_path), ac.DirStore)
        assert ac.S3Store("s3://b/p/").key("bench") == "p/bench/state.json"

    @pytest.mark.parametrize("backend", ["dir", "s3"])
    def test_writes_are_compare_and_swap(self, backend: str, tmp_path: Path, s3: ac.S3Store):
        store = s3 if backend == "s3" else ac.DirStore(tmp_path / "dir")
        state, token = store.read("bench")
        assert (state, token) == (ac.empty_state("bench"), None)
        store.write("bench", {**state, "last_run": "a"}, token)
        with pytest.raises(ac.StateConflict):
            store.write("bench", {**state, "last_run": "stale"}, token)  # created by someone else
        state, token = store.read("bench")
        store.write("bench", {**state, "last_run": "b"}, token)
        with pytest.raises(ac.StateConflict):
            store.write("bench", {**state, "last_run": "stale"}, token)  # changed since this read
        assert store.read("bench")[0]["last_run"] == "b"

    def test_access_denied_is_not_treated_as_empty_state(self, s3: ac.S3Store, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("FAKE_S3_DENY", "1")
        with pytest.raises(ac.CheckError, match="AccessDenied"):
            s3.read("bench")

    def test_concurrent_records_both_land(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        store = ac.DirStore(tmp_path)
        real_write = store.write
        raced = []

        def racing_write(check_id: str, state: dict[str, Any], token: str | None) -> None:
            if not raced:  # another shard records between this writer's read and its write
                raced.append(True)
                other, other_token = store.read(check_id)
                real_write(check_id, {**other, "files": {**other["files"], "other.rs": entry("t1")}}, other_token)
            real_write(check_id, state, token)

        monkeypatch.setattr(store, "write", racing_write)
        monkeypatch.setattr(ac.time, "sleep", lambda _: None)
        ac.record(store, [{"check": "bench", "files": {"mine.rs": entry("t2")}}])
        assert set(store.read("bench")[0]["files"]) == {"other.rs", "mine.rs"}


def entry(at: str) -> dict[str, Any]:
    return {"blobs": [at], "commit": "c", "reviewed_at": at, "version": 1, "findings": []}


def test_plan_carries_the_check_definition(repo: Path, tmp_path: Path):
    p = plan(repo, tmp_path / "state")
    assert ac.check_from_plan(p, "bench") == ac.load_checks(repo)["bench"]
    with pytest.raises(ac.CheckError, match="start a new run"):
        ac.check_from_plan(p, "bench", now=NOW + dt.timedelta(days=1))


def test_prompt_lists_assignment_and_suppressions(repo: Path, tmp_path: Path):
    p = plan(repo, tmp_path / "state")
    p["checks"]["bench"]["suppressed"] = {"bench:old": {"file": "enc/beta/benches/x.rs", "pr": 9}}
    check = ac.load_checks(repo)["bench"]
    prompt = ac.render_prompt(check, p, ac.find_shard(p, "bench", "beta"), "/tmp/results.json")
    assert "Review the benchmarks." in prompt
    assert prompt.startswith("Agent check `bench`, rubric version 1.")
    assert "`enc/beta/benches/x.rs` (1 lines): never reviewed" in prompt
    assert "`/tmp/results.json`" in prompt
    assert "Edits may touch only your assigned files." in prompt
    assert "`bench:old`" in prompt


def test_repository_checks_parse():
    checks = ac.load_checks(REPO_ROOT)
    assert checks, "expected at least one check under .agents/checks"
    for check in checks.values():
        assert check.include
