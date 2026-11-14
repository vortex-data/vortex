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
APP = "vortex-claude[bot]"


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
labels = ["agent-check"]
+++

Review the benchmarks.
"""


def sh(root: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)


def write(root: Path, path: str, text: str) -> None:
    (root / path).parent.mkdir(parents=True, exist_ok=True)
    (root / path).write_text(text)


def set_check(root: Path, extra: str) -> None:
    """Add frontmatter lines to the test check."""
    write(root, ".agents/checks/bench/CHECK.md", CHECK_MD.replace("+++\n\n", f"{extra}\n+++\n\n"))


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
    now: dt.datetime = NOW,
    crates: list[str] | None = None,
) -> dict[str, Any]:
    checks = list(ac.load_checks(root).values())
    store = ac.DirStore(state_dir)
    view = ac.GitHubView(items or [])
    return ac.build_plan(checks, root, store, now=now, github=view, max_shards=max_shards, only_shards=crates)


def shard_files(p: dict[str, Any]) -> dict[str, list[str]]:
    return {s["id"]: [f["path"] for f in s["files"]] for s in p["checks"]["bench"]["shards"]}


def finding(file: str = "enc/beta/benches/x.rs", **overrides: object) -> dict[str, object]:
    base = {
        "file": file,
        "line": 1,
        "item": "x",
        "severity": "medium",
        "category": "missing-black-box",
        "title": "Result is not black-boxed",
        "detail": "The compiler can drop the work.",
        "fixed": True,
    }
    return {**base, **overrides}


def finalize(
    repo: Path,
    results: dict[str, Any],
    crate: str = "beta",
    p: dict[str, Any] | None = None,
    run_verify: bool = False,
    format_cmd: tuple[str, ...] = (),
) -> dict[str, Any]:
    p = p or plan(repo, repo.parent / "no-state")
    check = ac.check_from_plan(p, "bench")
    shard = ac.find_shard(p, "bench", crate)
    return ac.finalize(
        check, p, shard, ac.validate_results(results), repo, now=NOW, run_verify=run_verify, format_cmd=format_cmd
    )


def review(root: Path, p: dict[str, Any], crate: str, files: list[str]) -> dict[str, Any]:
    """The state update of a crate whose agent reviewed `files` and found nothing."""
    return finalize(root, {"reviewed": files, "findings": []}, crate, p)["state_update"]


def pr(number: int, branch: str, body: str = "", state: str = "OPEN", **extra: object) -> dict[str, Any]:
    return {"number": number, "state": state, "headRefName": branch, "body": body, "mergedAt": None, **extra}


def marker(check: str = "other", **fields: object) -> str:
    return f"<!-- agent-check: {json.dumps({'check': check, 'shard': 'x', **fields})} -->"


class TestParse:
    def test_parses_frontmatter_and_body(self, repo: Path):
        check = ac.load_checks(repo)["bench"]
        assert check.include == ("**/benches/*.rs",)
        assert (check.max_files_per_shard, check.max_turns, check.fixes) == (2, 120, True)
        assert check.body == "Review the benchmarks.\n"
        assert "Bash(cargo clippy:*)" in check.tools("beta")

    @pytest.mark.parametrize(
        ("frontmatter", "error"),
        [
            ('id = "x"\ntitle = "t"\ndescription = "d"\ninclude = []\nmode = "full"', "unknown frontmatter keys"),
            ('id = "x"\ntitle = "t"\ndescription = "d"', "missing frontmatter keys"),
            ('id = "x"\ntitle = "t"\ndescription = "d"\ninclude = []\nfixes = "no"', "`fixes` must be bool"),
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
        assert (alpha["deferred"], alpha["branch"]) == (1, "agent-checks/bench/alpha")
        beta = ac.find_shard(p, "bench", "beta")
        assert (beta["crate"], beta["crate_dir"]) == ("beta", "enc/beta")
        assert ac.check_matrix(p)["include"] == [{"check": "bench", "shards": '["alpha", "beta"]'}]

    def test_deferred_files_come_first_next_run(self, repo: Path, tmp_path: Path):
        p = plan(repo, tmp_path / "state")
        update = review(repo, p, "alpha", ["alpha/benches/a.rs", "alpha/benches/b.rs"])
        ac.record(ac.DirStore(tmp_path / "state"), [update])
        assert shard_files(plan(repo, tmp_path / "state"))["alpha"] == ["alpha/benches/c.rs"]

    def test_says_why_crates_wait(self, repo: Path, tmp_path: Path):
        # The open PR has no label at all: its branch name alone marks the crate as busy.
        p = plan(repo, tmp_path / "state", items=[pr(1, "agent-checks/bench/alpha")])
        assert list(shard_files(p)) == ["beta"]
        busy = {"crate": "alpha", "pending": "3", "reason": "its agent PR is still open"}
        assert p["checks"]["bench"]["skipped"] == [busy]
        p = plan(repo, tmp_path / "state", max_shards=1)
        assert "crate budget" in p["checks"]["bench"]["skipped"][0]["reason"]
        table = ac.plan_summary(p)
        assert "| `bench` | `alpha` | 2, 1 deferred | review (new) |" in table
        assert "waits: this run's crate budget" in table

    def test_caps_open_prs_per_check(self, repo: Path, tmp_path: Path):
        set_check(repo, "max_open_prs = 1")
        p = plan(repo, tmp_path / "state", items=[pr(1, "agent-checks/bench/gone")])
        assert shard_files(p) == {}
        assert all("max_open_prs" in s["reason"] for s in p["checks"]["bench"]["skipped"])

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

    def test_reviewed_files_are_skipped_until_they_change(self, repo: Path, tmp_path: Path):
        state_dir = tmp_path / "state"
        p = plan(repo, state_dir)
        ac.record(ac.DirStore(state_dir), [review(repo, p, "beta", ["enc/beta/benches/x.rs"])])
        assert "beta" not in shard_files(plan(repo, state_dir))

        write(repo, "enc/beta/benches/x.rs", "fn x() { changed() }\n")
        sh(repo, "commit", "-qam", "change")
        again = ac.find_shard(plan(repo, state_dir), "bench", "beta")
        assert again["files"][0]["status"] == "changed"
        assert again["files"][0]["last_commit"] == p["base_commit"]

    def test_version_bump_requeues_unchanged_files(self, repo: Path, tmp_path: Path):
        state_dir = tmp_path / "state"
        p = plan(repo, state_dir)
        ac.record(ac.DirStore(state_dir), [review(repo, p, "beta", ["enc/beta/benches/x.rs"])])
        set_check(repo, "version = 2")
        assert ac.find_shard(plan(repo, state_dir), "bench", "beta")["files"][0]["status"] == "rubric"

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
        update = review(repo, p, "alpha", ["alpha/benches/a.rs", "alpha/benches/b.rs"])
        ac.record(ac.DirStore(state_dir), [update])
        # alpha still has a deferred file, but beta has waited longer, so it goes first.
        assert list(shard_files(plan(repo, state_dir))) == ["beta", "alpha"]

    def test_changed_crates_alternate_with_the_rotation(self, repo: Path, tmp_path: Path):
        state_dir = tmp_path / "state"
        p = plan(repo, state_dir)
        ac.record(ac.DirStore(state_dir), [review(repo, p, "beta", ["enc/beta/benches/x.rs"])])
        write(repo, "enc/beta/benches/x.rs", "fn x() { changed() }\n")
        sh(repo, "commit", "-qam", "change")
        write(repo, "gamma/Cargo.toml", '[package]\nname = "gamma"\n')
        write(repo, "gamma/benches/g.rs", "fn g() {}\n")
        sh(repo, "add", "-A")
        assert list(shard_files(plan(repo, state_dir))) == ["beta", "alpha", "gamma"]

    def test_changed_files_take_at_most_half_the_budget(self):
        files: list[dict[str, Any]] = [
            {"path": f"c{i}", "status": "changed", "last_reviewed": "t", "lines": 1} for i in range(4)
        ]
        files += [{"path": f"n{i}", "status": "new", "last_reviewed": None, "lines": 1} for i in range(4)]
        assert [f["path"] for f in ac.take_budget(files, 4, None)] == ["c0", "c1", "n0", "n1"]

    def test_unreadable_state_skips_only_that_check(self, repo: Path, tmp_path: Path):
        write(tmp_path, "state/bench/state.json", "[]")
        p = plan(repo, tmp_path / "state")
        assert p["checks"]["bench"]["shards"] == [] and "schema" in p["checks"]["bench"]["error"]

    def test_plan_carries_the_check_definition(self, repo: Path, tmp_path: Path):
        p = plan(repo, tmp_path / "state")
        assert ac.check_from_plan(p, "bench") == ac.load_checks(repo)["bench"]
        with pytest.raises(ac.CheckError, match="start a new run"):
            ac.check_from_plan(p, "bench", rerun_at=NOW + dt.timedelta(days=1))
        # A first attempt is never refused: a check's last crates may start hours after planning.
        assert ac.check_from_plan(p, "bench").id == "bench"


class TestPrompt:
    def test_lists_assignment_suppressions_and_verify(self, repo: Path, tmp_path: Path):
        set_check(repo, 'verify = ["cargo clippy -p {crate} -- -A {braces}"]')
        p = plan(repo, tmp_path / "state")
        p["checks"]["bench"]["suppressed"] = {"bench:old": {"file": "enc/beta/benches/x.rs", "ref": 9}}
        check = ac.check_from_plan(p, "bench")
        prompt = ac.render_prompt(check, p, ac.find_shard(p, "bench", "beta"), "/tmp/results.json")
        assert prompt.startswith("Agent check `bench`, rubric version 1.")
        assert "Review the benchmarks." in prompt
        assert "`enc/beta/benches/x.rs` (1 lines): never reviewed" in prompt
        assert "`/tmp/results.json`" in prompt
        assert "`bench:old`" in prompt
        assert "`cargo clippy -p beta -- -A {braces}`" in prompt
        assert "Bash(cargo clippy -p beta -- -A {braces})" in check.tools("beta")


class TestFinalize:
    def test_filters_findings_and_scopes_edits(self, repo: Path):
        write(repo, "enc/beta/benches/x.rs", "fn x() { black_box(()) }\n")
        write(repo, "alpha/benches/a.rs", "fn a() { sneaky() }\n")
        p = plan(repo, repo.parent / "no-state")
        p["checks"]["bench"]["suppressed"] = {"missing-black-box:enc/beta/benches/x.rs:old": {"file": "x"}}
        results = {
            "reviewed": ["enc/beta/benches/x.rs", "alpha/benches/a.rs"],
            "findings": [
                finding(severity="low", item="low", fixed=False),
                finding(),
                finding(item="old"),
                finding(file="alpha/benches/a.rs"),
                finding(severity="critical", item="bad"),
            ],
            "summary": "Black-boxed a result.",
        }
        out = finalize(repo, results, p=p)

        assert [f["fingerprint"] for f in out["findings"]] == [
            "missing-black-box:enc/beta/benches/x.rs:x",
            "missing-black-box:enc/beta/benches/x.rs:low",
        ]
        assert any("malformed finding 4" in n for n in out["notes"])
        assert out["edits"] == ["enc/beta/benches/x.rs"]
        assert (repo / "alpha/benches/a.rs").read_text() == "fn a() {}\n"
        record = out["state_update"]["files"]["enc/beta/benches/x.rs"]
        assert len(record["blobs"]) == 2  # pre-fix and post-fix blobs both count as reviewed
        assert list(out["state_update"]["files"]) == ["enc/beta/benches/x.rs"]
        assert (out["title"], out["issue_title"]) == ("Bench check fixes in beta", "Bench check findings in beta")

    def test_partial_results_keep_finished_files_and_revert_the_rest(self, repo: Path):
        # The agent finished a.rs, then hit its turn limit halfway through editing b.rs.
        write(repo, "alpha/benches/a.rs", "fn a() { fixed() }\n")
        write(repo, "alpha/benches/b.rs", "fn b() { half_do\n")
        out = finalize(repo, {"reviewed": ["alpha/benches/a.rs"], "findings": []}, "alpha")
        assert out["edits"] == ["alpha/benches/a.rs"]
        assert (repo / "alpha/benches/b.rs").read_text() == "fn b() {}\n"
        assert list(out["state_update"]["files"]) == ["alpha/benches/a.rs"]

    @pytest.mark.parametrize(("command", "kept"), [("true {crate}", True), ("false {crate}", False)])
    def test_verify_failure_reverts_all_edits(self, repo: Path, command: str, kept: bool):
        set_check(repo, f'verify = ["{command}"]')
        write(repo, "enc/beta/benches/x.rs", "fn x() { broken( }\n")
        out = finalize(repo, {"reviewed": ["enc/beta/benches/x.rs"], "findings": [finding()]}, run_verify=True)
        assert bool(out["edits"]) is kept
        assert out["findings"][0]["fixed"] is kept
        assert ((repo / "enc/beta/benches/x.rs").read_text() == "fn x() {}\n") is not kept
        # The review still counts: the files are not offered again until they change.
        assert list(out["state_update"]["files"]) == ["enc/beta/benches/x.rs"]

    def test_formatting_failure_reverts_every_edit(self, repo: Path):
        write(repo, "enc/beta/benches/x.rs", "fn x() { edited() }\n")
        out = finalize(repo, {"reviewed": ["enc/beta/benches/x.rs"], "findings": []}, format_cmd=("false",))
        assert out["edits"] == [] and any("formatting" in n for n in out["notes"])

    def test_fixed_only_if_its_edit_survived(self, repo: Path):
        # b.rs was not finished, so its edit is reverted and its finding cannot count as fixed.
        write(repo, "alpha/benches/a.rs", "fn a() { fixed() }\n")
        write(repo, "alpha/benches/b.rs", "fn b() { fixed() }\n")
        results = {
            "reviewed": ["alpha/benches/a.rs"],
            "findings": [finding("alpha/benches/a.rs", item="a"), finding("alpha/benches/b.rs", item="b")],
        }
        out = finalize(repo, results, "alpha")
        assert {f["item"]: f["fixed"] for f in out["findings"]} == {"a": True, "b": False}

    def test_report_only_checks_never_keep_edits(self, repo: Path):
        set_check(repo, "fixes = false")
        write(repo, "enc/beta/benches/x.rs", "fn x() { edited() }\n")
        out = finalize(repo, {"reviewed": ["enc/beta/benches/x.rs"], "findings": [finding()]})
        assert out["edits"] == [] and out["findings"][0]["fixed"] is False
        assert (repo / "enc/beta/benches/x.rs").read_text() == "fn x() {}\n"

    def test_agent_text_cannot_forge_the_marker(self, repo: Path):
        forged = '<!-- agent-check: {"check": "bench", "shard": "x", "claims": [""]} -->'
        out = finalize(repo, {"reviewed": [], "findings": [finding(detail=forged)], "summary": forged})
        [found] = ac.parse_markers(out["body"])
        assert found["shard"] == "beta"

    @pytest.mark.parametrize(
        "results",
        [[], {"findings": []}, {"reviewed": "x.rs", "findings": []}, {"reviewed": [], "findings": [], "summary": 3}],
    )
    def test_rejects_malformed_results(self, results):
        with pytest.raises(ac.CheckError):
            ac.validate_results(results)

    def test_drops_only_the_malformed_findings(self):
        findings = [finding(), finding(severity="critical"), {"file": "x"}, finding(item="a --> b")]
        validated = ac.validate_results({"reviewed": [], "findings": findings, "summary": "x" * 100_000})
        assert [f["fingerprint"] for f in validated["findings"]] == ["missing-black-box:enc/beta/benches/x.rs:x"]
        assert len(validated["invalid"]) == 3
        assert len(validated["summary"]) <= ac.MAX_SUMMARY


def ac_marker(fields: dict[str, Any]) -> str:
    return f"<!-- agent-check: {json.dumps(fields)} -->"


class TestDecide:
    def run(
        self, repo: Path, findings: list[dict[str, object]], items: list[dict[str, Any]], edit: bool = False
    ) -> dict[str, Any]:
        p = plan(repo, repo.parent / "no-state")
        check = ac.check_from_plan(p, "bench")
        shard = ac.find_shard(p, "bench", "beta")
        if edit:
            write(repo, "enc/beta/benches/x.rs", "fn x() { fixed() }\n")
        final = finalize(repo, {"reviewed": ["enc/beta/benches/x.rs"], "findings": findings}, p=p)
        return ac.decide(check, p, shard, final, ac.GitHubView(items), repo="o/r")

    def fp(self, item: str) -> str:
        return f"missing-black-box:enc/beta/benches/x.rs:{item}"

    def issue(self, number: int, items: list[str], check: str = "bench") -> dict[str, Any]:
        mark = {"check": check, "shard": "beta", "fingerprints": {self.fp(i): "enc/beta/benches/x.rs" for i in items}}
        return {"number": number, "state": "OPEN", "body": "", "comments": [{"body": ac_marker(mark)}]}

    def test_opens_an_issue_for_new_unfixed_findings(self, repo: Path):
        d = self.run(repo, [finding(fixed=False)], [])
        assert (d["pr"], d["issue"], d["issue_number"]) == ("none", "create", None)
        assert d["issue_title"] == "Bench check findings in beta"
        assert "[`enc/beta/benches/x.rs:1`](https://github.com/o/r/blob/" in d["issue_body"]
        assert "close it as not planned" in d["issue_body"]

    def test_comments_only_the_new_findings_on_the_shards_issue(self, repo: Path):
        findings = [finding(item="a", fixed=False), finding(item="b", fixed=False)]
        d = self.run(repo, findings, [self.issue(7, ["a"])])
        assert (d["issue"], d["issue_number"]) == ("comment", 7)
        assert [f["item"] for f in d["findings"]] == ["b"]
        assert "already reported" in d["issue_body"]

    def test_publishes_nothing_when_every_finding_is_known(self, repo: Path):
        d = self.run(repo, [finding(item="a", fixed=False)], [self.issue(7, ["a"])])
        assert (d["pr"], d["issue"]) == ("none", "none")

    def test_other_checks_do_not_dedup(self, repo: Path):
        d = self.run(repo, [finding(item="a", fixed=False)], [self.issue(7, ["a"], check="other")])
        assert (d["issue"], d["issue_number"]) == ("create", None)

    def test_prs_carry_fixes_and_issues_carry_new_unfixed_findings(self, repo: Path):
        findings = [finding(item="fixed"), finding(item="a", fixed=False), finding(item="b", fixed=False)]
        d = self.run(repo, findings, [self.issue(7, ["a"])], edit=True)
        assert (d["pr"], d["issue"], d["issue_number"]) == ("open", "comment", 7)
        assert d["pr_title"] == "Bench check fixes in beta"
        assert f"1 finding that this PR does not fix: see {ac.ISSUE_REF}." in d["pr_body"]
        assert ac.REJECTED_LABEL in d["pr_body"]
        assert ac.parse_markers(d["pr_body"])[0]["claims"] == ["enc/beta/benches/x.rs"]
        assert self.fp("b") in ac.parse_markers(d["issue_body"])[0]["fingerprints"]

    def test_edits_are_blocked_by_an_open_pr_on_the_branch(self, repo: Path):
        findings = [finding(), finding(item="b", fixed=False)]
        d = self.run(repo, findings, [pr(3, "agent-checks/bench/beta")], edit=True)
        assert (d["pr"], d["issue"]) == ("blocked", "none")


def entry(at: str) -> dict[str, Any]:
    return {"blobs": [at], "commit": "c", "reviewed_at": at, "version": 1, "findings": []}


class TestRecord:
    MARK = '<!-- agent-check: {"check": "bench", "shard": "beta", "fingerprints": {"b:x.rs:1": "x.rs"}} -->'
    REJECTED = [{"name": "agent-checks:rejected"}]

    def mark(self, n: int) -> str:
        return self.MARK.replace(":1", f":{n}")

    def test_only_explicit_rejections_become_suppressions(self, tmp_path: Path):
        items = [
            pr(1, "agent-checks/bench/beta", self.mark(1), state="CLOSED"),  # closed for any reason: kept
            pr(2, "agent-checks/bench/beta", self.mark(2), state="CLOSED", labels=self.REJECTED),
            {"number": 3, "state": "CLOSED", "stateReason": "NOT_PLANNED", "comments": [{"body": self.mark(3)}]},
            {"number": 4, "state": "CLOSED", "stateReason": "COMPLETED", "body": self.mark(4)},
        ]
        ac.record(ac.DirStore(tmp_path), [], github=ac.GitHubView(items))
        state = json.loads((tmp_path / "bench" / "state.json").read_text())
        assert state["suppressed"] == {"b:x.rs:2": {"file": "x.rs", "ref": 2}, "b:x.rs:3": {"file": "x.rs", "ref": 3}}

    def test_unsuppressed_findings_stay_unsuppressed(self, tmp_path: Path):
        store = ac.DirStore(tmp_path)
        view = ac.GitHubView([pr(4, "agent-checks/bench/beta", self.MARK, state="CLOSED", labels=self.REJECTED)])
        ac.record(store, [], github=view)
        assert ac.unsuppress(store, "bench", ["b:x.rs:1", "nope"]) == ["b:x.rs:1"]
        ac.record(store, [], github=view)
        assert store.read("bench")[0]["suppressed"] == {}

    def test_newer_reviews_win(self, tmp_path: Path):
        ac.record(ac.DirStore(tmp_path), [{"check": "bench", "files": {"x.rs": entry("2026-11-13T00:00:00")}}])
        ac.record(ac.DirStore(tmp_path), [{"check": "bench", "files": {"x.rs": entry("2026-11-01T00:00:00")}}])
        state = json.loads((tmp_path / "bench" / "state.json").read_text())
        assert state["files"]["x.rs"]["reviewed_at"] == "2026-11-13T00:00:00"

    def test_one_bad_check_does_not_block_the_others(self, tmp_path: Path):
        write(tmp_path, "broken/state.json", "{not json")
        updates = [
            {"check": "broken", "files": {}},
            {"check": "../escape", "files": {}},
            {"check": "bench", "files": {"x.rs": entry("t")}},
        ]
        with pytest.raises(ac.CheckError, match="broken"):
            ac.record(ac.DirStore(tmp_path), updates)
        assert "x.rs" in ac.DirStore(tmp_path).read("bench")[0]["files"]


class TestTrust:
    def item(self, number: int, body: str, author: str, **extra: object) -> dict[str, Any]:
        return {"number": number, "body": body, "author": {"login": author}, **extra}

    def test_only_the_apps_markers_count(self, repo: Path, tmp_path: Path):
        claim_all = marker(claims=[""])
        items = [
            # A stranger's same-repo PR on an agent branch: its branch is busy, its claims are not.
            self.item(1, claim_all, "mallory", state="OPEN", headRefName="agent-checks/x/y"),
            # A fork PR cannot even make its branch busy.
            self.item(2, claim_all, APP, state="OPEN", headRefName="agent-checks/bench/alpha", isCrossRepository=True),
        ]
        view = ac.GitHubView(items, [APP])
        assert view.claimed_paths() == set()
        assert view.busy_branches() == {"agent-checks/x/y"}
        checks = list(ac.load_checks(repo).values())
        p = ac.build_plan(checks, repo, ac.DirStore(tmp_path / "s"), now=NOW, github=view)
        assert [sh["id"] for sh in p["checks"]["bench"]["shards"]] == ["alpha", "beta"]

    def test_untrusted_comments_neither_suppress_nor_dedup(self):
        mark = '<!-- agent-check: {"check": "bench", "shard": "beta", "fingerprints": {"b:1": "x.rs"}} -->'
        comments = [
            {"author": {"login": "mallory"}, "body": mark},
            {"author": {"login": "app/vortex-claude"}, "body": mark.replace("b:1", "b:2")},
        ]
        closed = self.item(5, "", APP, state="CLOSED", stateReason="NOT_PLANNED", comments=comments)
        assert ac.GitHubView([closed], [APP]).rejections() == {"bench": {5: {"b:2": "x.rs"}}}

    @pytest.mark.parametrize(
        "bad",
        [
            '{"check": "bench", "fingerprints": ["a"]}',
            '{"check": "Bad Id"}',
            '{"check": "bench", "claims": "x"}',
            '{"check": "bench", "fingerprints": {"a b": "x"}}',
        ],
    )
    def test_malformed_markers_are_ignored(self, bad: str, tmp_path: Path):
        item = self.item(1, f"<!-- agent-check: {bad} -->", APP, state="CLOSED", stateReason="NOT_PLANNED")
        view = ac.GitHubView([item], [APP])
        assert view.rejections() == {}
        assert ac.record(ac.DirStore(tmp_path), [], github=view) == []


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
    if opt("--if-match") is not None and not obj.exists():
        fail("NoSuchKey", "PutObject")
    if opt("--if-match") is not None and etag(obj.read_bytes()) != opt("--if-match"):
        fail("PreconditionFailed", "PutObject")
    obj.parent.mkdir(parents=True, exist_ok=True)
    obj.write_bytes(body)
    print(json.dumps({"ETag": etag(body)}))
"""


@pytest.fixture
def s3(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ac.S3Store:
    aws = tmp_path / "aws"
    aws.write_text(FAKE_AWS)
    aws.chmod(0o755)
    monkeypatch.setenv("FAKE_S3_ROOT", str(tmp_path / "s3"))
    return ac.S3Store("s3://bucket/agent-checks", aws=str(aws))


class TestStores:
    def test_open_store_picks_the_backend(self, tmp_path: Path):
        assert isinstance(ac.open_store("s3://b/p"), ac.S3Store)
        assert isinstance(ac.open_store(tmp_path), ac.DirStore)
        assert ac.S3Store("s3://b/p/").key("bench") == "p/bench/state.json"

    @pytest.mark.parametrize("backend", ["dir", "s3"])
    def test_writes_are_compare_and_swap(self, backend: str, tmp_path: Path, s3: ac.S3Store):
        store = s3 if backend == "s3" else ac.DirStore(tmp_path / "dir")
        state, token = store.read("bench")
        assert (state, token) == (ac.empty_state("bench"), None)
        store.write("bench", {**state, "files": {"a": 1}}, token)
        with pytest.raises(ac.StateConflict):
            store.write("bench", {**state, "files": {"stale": 1}}, token)  # created by someone else
        state, token = store.read("bench")
        store.write("bench", {**state, "files": {"b": 1}}, token)
        with pytest.raises(ac.StateConflict):
            store.write("bench", {**state, "files": {"stale": 1}}, token)  # changed since this read
        assert store.read("bench")[0]["files"] == {"b": 1}

    def test_a_deleted_object_is_a_conflict(self, s3: ac.S3Store, tmp_path: Path):
        state, _ = s3.read("bench")
        s3.write("bench", state, None)
        _, token = s3.read("bench")
        (tmp_path / "s3" / "bucket" / s3.key("bench")).unlink()  # someone resets the check
        with pytest.raises(ac.StateConflict):
            s3.write("bench", state, token)

    def test_access_denied_is_not_treated_as_empty_state(self, s3: ac.S3Store, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("FAKE_S3_DENY", "1")
        with pytest.raises(ac.CheckError, match="AccessDenied"):
            s3.read("bench")

    def test_concurrent_records_both_land(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        store = ac.DirStore(tmp_path)
        real_write = store.write
        raced = []

        def racing_write(check_id: str, state: dict[str, Any], token: str | None) -> None:
            if not raced:  # another writer records between this writer's read and its write
                raced.append(True)
                other, other_token = store.read(check_id)
                real_write(check_id, {**other, "files": {**other["files"], "other.rs": entry("t1")}}, other_token)
            real_write(check_id, state, token)

        monkeypatch.setattr(store, "write", racing_write)
        monkeypatch.setattr(ac.time, "sleep", lambda _: None)
        ac.record(store, [{"check": "bench", "files": {"mine.rs": entry("t2")}}])
        assert set(store.read("bench")[0]["files"]) == {"other.rs", "mine.rs"}

    def test_s3_put_new_is_write_once(self, s3: ac.S3Store):
        assert s3.put_new("bench", "runs/r/x/results.json", b"1")
        assert not s3.put_new("bench", "runs/r/x/results.json", b"2")
        assert s3.get("bench", "runs/r/x/results.json")[0] == b"1"

    @pytest.mark.parametrize(
        ("check", "key"), [("bench", "../o/state.json"), ("../x", "state.json"), ("bench", "a//b")]
    )
    def test_rejects_keys_outside_the_checks_prefix(self, tmp_path: Path, check: str, key: str):
        with pytest.raises(ac.CheckError, match="invalid storage key"):
            ac.DirStore(tmp_path).get(check, key)


class TestArchive:
    def handed(self, tmp_path: Path, name: str, check: str, shard: str, files: dict[str, str]) -> Path:
        d = tmp_path / name
        d.mkdir()
        (d / "meta.json").write_text(json.dumps({"check": check, "shard": shard}))
        for file, text in files.items():
            (d / file).write_text(text)
        return d

    def test_keeps_a_write_once_record_per_check_and_crate(self, repo: Path, tmp_path: Path):
        p = plan(repo, tmp_path / "plan-state")
        store = ac.DirStore(tmp_path / "store")
        dirs = [
            self.handed(tmp_path, "agent", "bench", "beta", {"results.json": "{}", "patch.diff": "d", "junk.sh": "x"}),
            self.handed(tmp_path, "out", "bench", "beta", {"decision.json": "{}", "pr.md": "pr"}),
            self.handed(tmp_path, "forged", "bench", "not-planned", {"results.json": "{}"}),
        ]
        written = ac.archive(store, p, "123-1", dirs)
        key = ac.run_key(p, "123-1")
        assert key == "20261113T120000Z_123-1"
        assert sorted(written) == sorted(
            f"bench/runs/{key}/{name}"
            for name in ("plan.json", "beta/results.json", "beta/patch.diff", "beta/decision.json", "beta/pr.md")
        )
        assert store.get("bench", f"runs/{key}/beta/junk.sh")[0] is None

        # Re-running the record job writes nothing and changes nothing.
        (dirs[1] / "pr.md").write_text("changed")
        assert ac.archive(store, p, "123-1", dirs) == []
        assert store.get("bench", f"runs/{key}/beta/pr.md")[0] == b"pr"


class TestApplyPatch:
    def patch_of(self, repo: Path, changes: dict[str, str], symlink: str | None = None) -> Path:
        for path, text in changes.items():
            write(repo, path, text)
        if symlink:
            (repo / symlink).symlink_to("/etc/passwd")
        sh(repo, "add", "-A")
        diff = subprocess.run(["git", "-C", str(repo), "diff", "--cached", "--binary"], capture_output=True, check=True)
        sh(repo, "reset", "-q", "--hard")
        patch = repo.parent / "agent.patch"
        patch.write_bytes(diff.stdout)
        return patch

    def test_applies_only_the_crates_files(self, repo: Path):
        shard = ac.find_shard(plan(repo, repo.parent / "no-state"), "bench", "beta")
        patch = self.patch_of(
            repo,
            {"enc/beta/benches/x.rs": "fn x() { fixed() }\n", "alpha/benches/a.rs": "fn a() { sneaky() }\n"},
            symlink="enc/beta/benches/link.rs",
        )
        applied, rejected = ac.apply_patch(repo, patch, shard)
        assert applied == ["enc/beta/benches/x.rs"]
        assert sorted(rejected) == ["alpha/benches/a.rs", "enc/beta/benches/link.rs"]
        assert (repo / "enc/beta/benches/x.rs").read_text() == "fn x() { fixed() }\n"
        assert (repo / "alpha/benches/a.rs").read_text() == "fn a() {}\n"
        assert not (repo / "enc/beta/benches/link.rs").exists()

    def test_empty_patch_applies_nothing(self, repo: Path):
        shard = ac.find_shard(plan(repo, repo.parent / "no-state"), "bench", "beta")
        empty = repo.parent / "empty.patch"
        empty.write_text("")
        assert ac.apply_patch(repo, empty, shard) == ([], [])


def test_repository_checks_parse():
    checks = ac.load_checks(REPO_ROOT)
    assert checks, "expected at least one check under .agents/checks"
    for check in checks.values():
        assert check.include
