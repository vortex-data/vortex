#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Runner for periodic, agent-driven code checks.

A check is a directory `.agents/checks/<id>/` holding a `CHECK.md`: TOML frontmatter between `+++`
lines declares what the check covers and how it is sharded, and the Markdown body is the rubric
the agent follows. See `.agents/checks/README.md` for the format.

The runner is deliberately agent-agnostic. It plans work, renders a prompt, and validates what the
agent wrote back; the agent itself is launched by CI (`.github/workflows/agent-checks.yml`) or by
`run` for local use.

Subcommands:

    list                       List checks and their configuration.
    plan                       Pick shards that need review and write a plan JSON.
    prompt                     Render the agent prompt for one shard of a plan.
    finalize                   Validate an agent's results file and build the PR body and state update.
    record                     Merge state updates (and PR-derived suppressions) into the state dir.
    run                        Local convenience: plan, prompt, invoke `claude -p`, finalize, record.

State is one JSON file per check (`<state-dir>/<id>.json`). CI keeps it on the orphan branch
`agent-checks-state`; locally it defaults to `target/agent-checks-state`. For every reviewed file it
records the git blob hashes that were reviewed (before and after the agent's own fix), so unchanged
files are skipped and a merged fix does not trigger its own re-review.

TODO: replace the state branch with a store that supports atomic per-shard writes, so runs need no
branch-level serialization and a failed push cannot lose a run's reviews.

Run the tests with:

    uv run --no-project --with pytest pytest scripts/tests/test_agent_checks.py
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tomllib
from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKS_DIR = Path(".agents/checks")
DEFAULT_LOCAL_STATE = Path("target/agent-checks-state")
STATE_SCHEMA = 1
BRANCH_PREFIX = "agent-checks"
MARKER_RE = re.compile(r"<!--\s*agent-check:\s*(\{.*?\})\s*-->", re.DOTALL)
SEVERITIES = ("high", "medium", "low")
MAX_PLAN_AGE = dt.timedelta(hours=20)
CHECK_ID_RE = re.compile(r"[a-z0-9][a-z0-9-]*")
FINGERPRINT_RE = re.compile(r"[\w./:#@+=,-]{1,200}")
STATUS_RANK = {"changed": 0, "new": 1, "rubric": 2, "stale": 3, "full": 4}
# Tools every check gets. Checks add more with `allowed_tools`.
BASE_TOOLS = (
    "Read",
    "Write",
    "Edit",
    "Glob",
    "Grep",
    "Bash(git diff:*)",
    "Bash(git log:*)",
    "Bash(git show:*)",
    "Bash(git blame:*)",
    # Lets the agent undo its own edits; finalize reverts anything outside the shard regardless.
    "Bash(git checkout -- :*)",
    "Skill",
)


class CheckError(Exception):
    """A check definition, plan, or results file is invalid."""


# --------------------------------------------------------------------------------------------------
# Check definitions
# --------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Check:
    id: str
    title: str
    description: str
    body: str
    include: tuple[str, ...]
    exclude: tuple[str, ...] = ()
    content_match: str | None = None
    edit_scope: str = "files"
    max_files_per_shard: int = 10
    max_lines_per_shard: int | None = None
    max_shards_per_run: int = 4
    mode: str = "incremental"
    revisit_after_days: int | None = None
    cadence_days: int = 7
    version: int = 1
    action: str = "pr"
    labels: tuple[str, ...] = ()
    skills: tuple[str, ...] = ()
    allowed_tools: tuple[str, ...] = ()
    verify: tuple[str, ...] = ()
    max_findings: int = 20
    model: str | None = None
    max_turns: int = 80
    enabled: bool = True

    @property
    def tools(self) -> list[str]:
        return list(dict.fromkeys([*BASE_TOOLS, *self.allowed_tools]))


_FIELD_TYPES: dict[str, type | tuple[type, ...]] = {
    "id": str,
    "title": str,
    "description": str,
    "include": list,
    "exclude": list,
    "content_match": str,
    "edit_scope": str,
    "max_files_per_shard": int,
    "max_lines_per_shard": int,
    "max_shards_per_run": int,
    "mode": str,
    "revisit_after_days": int,
    "cadence_days": int,
    "version": int,
    "action": str,
    "labels": list,
    "skills": list,
    "allowed_tools": list,
    "verify": list,
    "max_findings": int,
    "model": str,
    "max_turns": int,
    "enabled": bool,
}
_CHOICES = {
    "edit_scope": ("files", "crate"),
    "mode": ("incremental", "full"),
    "action": ("pr", "report"),
}
_REQUIRED = ("id", "title", "description", "include")


def parse_check(text: str, source: str = "<check>") -> Check:
    """Parse a CHECK.md: `+++`-delimited TOML frontmatter followed by the Markdown rubric."""
    lines = text.splitlines(keepends=True)
    try:
        start = next(i for i, line in enumerate(lines) if line.strip() == "+++")
        end = next(i for i in range(start + 1, len(lines)) if lines[i].strip() == "+++")
    except StopIteration:
        raise CheckError(f"{source}: missing `+++` TOML frontmatter") from None
    if any(line.strip() and not line.lstrip().startswith("<!--") for line in lines[:start]):
        raise CheckError(f"{source}: only comments may precede the frontmatter")
    try:
        meta = tomllib.loads("".join(lines[start + 1 : end]))
    except tomllib.TOMLDecodeError as e:
        raise CheckError(f"{source}: invalid frontmatter: {e}") from None

    unknown = sorted(set(meta) - set(_FIELD_TYPES))
    if unknown:
        raise CheckError(f"{source}: unknown frontmatter keys {unknown}")
    missing = [k for k in _REQUIRED if k not in meta]
    if missing:
        raise CheckError(f"{source}: missing frontmatter keys {missing}")
    for key, value in meta.items():
        expected = _FIELD_TYPES[key]
        # bool is a subclass of int; keep `max_turns = true` from parsing.
        if not isinstance(value, expected) or (expected is int and isinstance(value, bool)):
            raise CheckError(f"{source}: `{key}` must be {getattr(expected, '__name__', expected)}")
        if isinstance(value, list) and not all(isinstance(v, str) for v in value):
            raise CheckError(f"{source}: `{key}` must be a list of strings")
        if key in _CHOICES and value not in _CHOICES[key]:
            raise CheckError(f"{source}: `{key}` must be one of {_CHOICES[key]}")
        if expected is int and value < 1:
            raise CheckError(f"{source}: `{key}` must be positive")
    if not CHECK_ID_RE.fullmatch(meta["id"]):
        raise CheckError(f"{source}: `id` must be kebab-case")
    if "content_match" in meta:
        try:
            re.compile(meta["content_match"])
        except re.error as e:
            raise CheckError(f"{source}: invalid `content_match`: {e}") from None

    for key in ("include", "exclude", "labels", "skills", "allowed_tools", "verify"):
        if key in meta:
            meta[key] = tuple(meta[key])
    return Check(body="".join(lines[end + 1 :]).strip() + "\n", **meta)


def load_checks(root: Path = REPO_ROOT) -> dict[str, Check]:
    checks: dict[str, Check] = {}
    for path in sorted((root / CHECKS_DIR).glob("*/CHECK.md")):
        check = parse_check(path.read_text(), str(path.relative_to(root)))
        if check.id != path.parent.name:
            raise CheckError(f"{path}: id `{check.id}` must match its directory name")
        checks[check.id] = check
    return checks


# --------------------------------------------------------------------------------------------------
# Repository inspection
# --------------------------------------------------------------------------------------------------


def glob_to_regex(pattern: str) -> re.Pattern[str]:
    """Translate a repo-relative glob (`*`, `?`, `**`, `{a,b}`) into an anchored regex."""
    out: list[str] = []
    i = 0
    while i < len(pattern):
        c = pattern[i]
        if pattern.startswith("**/", i):
            out.append("(?:.*/)?")
            i += 3
        elif pattern.startswith("**", i):
            out.append(".*")
            i += 2
        elif c == "*":
            out.append("[^/]*")
            i += 1
        elif c == "?":
            out.append("[^/]")
            i += 1
        elif c == "{":
            close = pattern.find("}", i)
            if close == -1:
                raise CheckError(f"unbalanced brace in glob {pattern!r}")
            alts = pattern[i + 1 : close].split(",")
            out.append("(?:" + "|".join(glob_to_regex(a).pattern[:-2] for a in alts) + ")")
            i = close + 1
        else:
            out.append(re.escape(c))
            i += 1
    return re.compile("".join(out) + r"\Z")


def git(root: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True, text=True).stdout


def tracked_blobs(root: Path) -> dict[str, str]:
    """Map every tracked path to its blob hash in the index."""
    blobs: dict[str, str] = {}
    for entry in git(root, "ls-files", "-s", "-z").split("\0"):
        if entry:
            meta, path = entry.split("\t", 1)
            blobs[path] = meta.split()[1]
    return blobs


def crate_roots(root: Path, tracked: dict[str, str]) -> dict[str, str]:
    """Map each Cargo package directory ("" for the root) to its package name."""
    roots: dict[str, str] = {}
    for path in tracked:
        if path == "Cargo.toml" or path.endswith("/Cargo.toml"):
            try:
                manifest = tomllib.loads((root / path).read_text())
            except (OSError, tomllib.TOMLDecodeError):
                continue
            name = manifest.get("package", {}).get("name")
            if isinstance(name, str):
                parent = str(Path(path).parent)
                roots["" if parent == "." else parent] = name
    return roots


def crate_of(path: str, roots: dict[str, str]) -> tuple[str, str] | None:
    """Return (package name, package dir) owning `path`, or None if no package does."""
    parent = Path(path).parent
    while True:
        key = "" if str(parent) == "." else str(parent)
        if key in roots:
            return roots[key], key
        if key == "":
            return None
        parent = parent.parent


def in_scope(check: Check, root: Path, tracked: dict[str, str]) -> dict[str, str]:
    include = [glob_to_regex(p) for p in check.include]
    exclude = [glob_to_regex(p) for p in check.exclude]
    content = re.compile(check.content_match, re.MULTILINE) if check.content_match else None
    selected: dict[str, str] = {}
    for path, blob in sorted(tracked.items()):
        if not any(r.match(path) for r in include) or any(r.match(path) for r in exclude):
            continue
        if content is not None:
            try:
                if not content.search((root / path).read_text(errors="replace")):
                    continue
            except OSError:
                continue
        selected[path] = blob
    return selected


def line_count(root: Path, path: str) -> int:
    try:
        with open(root / path, "rb") as f:
            return sum(1 for _ in f)
    except OSError:
        return 0


# --------------------------------------------------------------------------------------------------
# State
# --------------------------------------------------------------------------------------------------


def empty_state(check_id: str) -> dict[str, Any]:
    return {"schema": STATE_SCHEMA, "check": check_id, "last_run": None, "files": {}, "suppressed": {}}


def load_state(state_dir: Path, check_id: str) -> dict[str, Any]:
    path = state_dir / f"{check_id}.json"
    if not path.exists():
        return empty_state(check_id)
    state = json.loads(path.read_text())
    if state.get("schema") != STATE_SCHEMA:
        raise CheckError(f"{path}: unsupported state schema {state.get('schema')}")
    return state


def save_state(state_dir: Path, state: dict[str, Any]) -> None:
    state_dir.mkdir(parents=True, exist_ok=True)
    path = state_dir / f"{state['check']}.json"
    path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")


def is_pr(item: dict[str, Any]) -> bool:
    return "headRefName" in item


def item_markers(item: dict[str, Any]) -> list[dict[str, Any]]:
    """The runner's marker in a PR or issue body and in each issue comment.

    Only the last marker of each text counts: the runner appends its marker after any
    agent-written text, so an earlier one is quoted content, not the runner's.
    """
    texts = [item.get("body") or ""]
    texts += [c.get("body") or "" for c in item.get("comments") or [] if isinstance(c, dict)]
    markers = []
    for text in texts:
        if found := parse_markers(text):
            markers.append(found[-1])
    return markers


def suppressions_from_items(items: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    """Fingerprints that maintainers rejected, keyed by check id.

    `items` holds `gh pr list` and `gh issue list` JSON. Closing an agent PR without merging it, or
    closing a findings issue as "not planned", rejects every finding listed in it.
    """
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for item in items:
        if str(item.get("state", "")).upper() not in ("CLOSED", "MERGED"):
            continue
        if is_pr(item):
            if item.get("mergedAt") or str(item.get("state")).upper() == "MERGED":
                continue
        elif str(item.get("stateReason") or "").upper() != "NOT_PLANNED":
            continue
        for marker in item_markers(item):
            for fp, file in marker.get("fingerprints", {}).items():
                out.setdefault(marker["check"], {})[fp] = {"file": file, "ref": item.get("number")}
    return out


def parse_markers(body: str) -> list[dict[str, Any]]:
    markers = []
    for m in MARKER_RE.finditer(body):
        try:
            data = json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and isinstance(data.get("check"), str):
            markers.append(data)
    return markers


def _open_agent_prs(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # Matched by branch, not label: removing the label from an open PR must not make its branch
    # look free, or the next run would force-push over it.
    return [
        i
        for i in items
        if is_pr(i)
        and str(i.get("state", "")).upper() == "OPEN"
        and str(i["headRefName"]).startswith(BRANCH_PREFIX + "/")
    ]


def busy_branches(items: list[dict[str, Any]]) -> set[str]:
    return {str(pr["headRefName"]) for pr in _open_agent_prs(items)}


def claimed_paths(items: list[dict[str, Any]]) -> set[str]:
    """Files and directories that open agent PRs, of any check, may still change.

    No shard may review or edit them until the PR is merged or closed, so two PRs never touch the
    same file. Issues claim nothing: they carry no edits.
    """
    out: set[str] = set()
    for pr in _open_agent_prs(items):
        for marker in item_markers(pr):
            out.update(p for p in marker.get("claims", []) if isinstance(p, str))
    return out


def overlaps(paths: Collection[str], claimed: Collection[str]) -> bool:
    """Whether any path equals, contains, or lies inside a claimed path ("" claims everything)."""
    return any(_within(p, list(claimed)) or any(_within(c, [p]) for c in claimed) for p in paths)


# --------------------------------------------------------------------------------------------------
# Planning
# --------------------------------------------------------------------------------------------------


def shard_slug(shard_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", shard_id).strip("-") or "all"


def branch_name(check_id: str, shard_id: str) -> str:
    return f"{BRANCH_PREFIX}/{check_id}/{shard_slug(shard_id)}"


def file_status(check: Check, record: dict[str, Any] | None, blob: str, now: dt.datetime) -> str | None:
    """Why `blob` needs review, or None if it is up to date."""
    if check.mode == "full":
        return "full"
    if record is None:
        return "new"
    if blob not in record.get("blobs", []):
        return "changed"
    if record.get("version") != check.version:
        return "rubric"
    if check.revisit_after_days is not None:
        reviewed = dt.datetime.fromisoformat(record["reviewed_at"])
        if now - reviewed > dt.timedelta(days=check.revisit_after_days):
            return "stale"
    return None


def take_budget(files: list[dict[str, Any]], max_files: int, max_lines: int | None) -> list[dict[str, Any]]:
    """The longest prefix of `files` within the per-shard budget, and always at least one file."""
    taken: list[dict[str, Any]] = []
    lines = 0
    for f in files:
        if taken and (len(taken) >= max_files or (max_lines is not None and lines + f["lines"] > max_lines)):
            break
        taken.append(f)
        lines += f["lines"]
    return taken


def _file_priority(f: dict[str, Any]) -> tuple[int, str, str]:
    return STATUS_RANK[str(f["status"])], str(f["last_reviewed"] or ""), str(f["path"])


# Scheduled runs start at slightly different times each day; without slack a check whose last run
# started a few minutes late would wait an extra day.
CADENCE_SLACK = dt.timedelta(hours=12)


def is_due(check: Check, state: dict[str, Any], now: dt.datetime) -> bool:
    last_run = state.get("last_run")
    if not last_run:
        return True
    return now - dt.datetime.fromisoformat(last_run) >= dt.timedelta(days=check.cadence_days) - CADENCE_SLACK


def plan_check(
    check: Check,
    root: Path,
    state: dict[str, Any],
    *,
    now: dt.datetime,
    busy: Collection[str] = (),
    claimed: set[str] | None = None,
    only_shards: Collection[str] | None = None,
    max_shards: int | None = None,
) -> list[dict[str, Any]]:
    """Return the shards of `check` to review this run, highest priority first.

    A shard is one Cargo package and is never split: at most one agent works on a crate per check
    at a time, on a stable branch `agent-checks/<check>/<crate>`. When a crate has more pending
    files than one shard's budget, the most urgent are taken and the rest wait for later runs.
    Files outside any package are not reviewed.

    Crates with changed files come first. The rest rotate: the crate whose last review is oldest
    (never reviewed counts as oldest) goes next, so a large backlog in one crate cannot starve
    the others.

    `claimed` holds paths that open PRs or shards already planned this run may edit. Files inside
    it are not reviewed, a shard whose edit scope overlaps it waits for a later run, and the
    selected shards' scopes are added to it, so the scopes of selected shards are disjoint.
    """
    claimed = set() if claimed is None else claimed
    tracked = tracked_blobs(root)
    roots = crate_roots(root, tracked)
    names = list(roots.values())

    def shard_id(name: str, crate_dir: str) -> str:
        # Package names are not unique across the repository (test fixtures reuse them).
        return name if names.count(name) == 1 else shard_slug(crate_dir)

    last_reviewed: dict[str, str] = {}
    for path, record in state["files"].items():
        if owner := crate_of(path, roots):
            last_reviewed[owner[1]] = max(last_reviewed.get(owner[1], ""), str(record.get("reviewed_at", "")))

    groups: dict[str, dict[str, Any]] = {}
    for path, blob in in_scope(check, root, tracked).items():
        if _within(path, list(claimed)):
            continue
        owner = crate_of(path, roots)
        if owner is None:
            continue
        record = state["files"].get(path)
        status = file_status(check, record, blob, now)
        if status is None:
            continue
        name, crate_dir = owner
        group = groups.setdefault(crate_dir, {"crate": name, "files": []})
        group["files"].append(
            {
                "path": path,
                "blob": blob,
                "status": status,
                "last_commit": record.get("commit") if record else None,
                "last_reviewed": record.get("reviewed_at") if record else None,
                "lines": line_count(root, path),
            }
        )

    shards: list[dict[str, Any]] = []
    for crate_dir, group in groups.items():
        pending = sorted(group["files"], key=_file_priority)
        files = take_budget(pending, check.max_files_per_shard, check.max_lines_per_shard)
        sid = shard_id(group["crate"], crate_dir)
        shards.append(
            {
                "id": sid,
                "crate": group["crate"],
                "branch": branch_name(check.id, sid),
                "crate_dir": crate_dir,
                "edit_scope": [crate_dir] if check.edit_scope == "crate" else [f["path"] for f in files],
                "files": files,
                "deferred": len(pending) - len(files),
                "changed": any(f["status"] == "changed" for f in files),
            }
        )

    shards.sort(key=lambda sh: (not sh["changed"], last_reviewed.get(sh["crate_dir"], ""), sh["id"]))
    if only_shards:
        shards = [sh for sh in shards if sh["id"] in only_shards or sh["crate"] in only_shards]
    selected: list[dict[str, Any]] = []
    for shard in shards:
        if len(selected) >= (max_shards or check.max_shards_per_run):
            break
        if shard["branch"] in busy or overlaps(shard["edit_scope"], claimed):
            continue
        selected.append(shard)
        claimed.update(shard["edit_scope"])
    return selected


def build_plan(
    checks: list[Check],
    root: Path,
    state_dir: Path,
    *,
    now: dt.datetime,
    items: list[dict[str, Any]] | None = None,
    only_shards: Collection[str] | None = None,
    max_shards: int | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Plan a run. `items` is the PR and issue JSON from GitHub.

    A check is `due` when its cadence has passed. Only due checks get `last_run` stamped by
    `record`, so a forced or single-shard run does not reset the weekly clock.
    """
    items = items or []
    busy = busy_branches(items)
    # Shared across checks, so shards of different checks in one run never edit the same file.
    claimed = claimed_paths(items)
    new_suppressed = suppressions_from_items(items)
    plan: dict[str, Any] = {
        "base_commit": git(root, "rev-parse", "HEAD").strip(),
        "generated_at": now.isoformat(timespec="seconds"),
        "checks": {},
    }
    for check in checks:
        state = load_state(state_dir, check.id)
        due = is_due(check, state, now)
        suppressed = {**state.get("suppressed", {}), **new_suppressed.get(check.id, {})}
        shards = []
        if due or force:
            shards = plan_check(
                check,
                root,
                state,
                now=now,
                busy=busy,
                claimed=claimed,
                only_shards=only_shards,
                max_shards=max_shards,
            )
        plan["checks"][check.id] = {
            # The review jobs read the check from here, never from the working tree that the agent
            # can edit, and every job of the run sees the same definition.
            "definition": dataclasses.asdict(check),
            "due": due and not only_shards,
            "shards": shards,
            "suppressed": suppressed,
        }
    return plan


def check_from_plan(plan: dict[str, Any], check_id: str, *, now: dt.datetime | None = None) -> Check:
    if check_id not in plan.get("checks", {}):
        raise CheckError(f"check `{check_id}` is not in the plan")
    if now is not None and now - dt.datetime.fromisoformat(plan["generated_at"]) > MAX_PLAN_AGE:
        # A re-run of an old workflow run would review stale assignments and could reopen work.
        raise CheckError(f"the plan is from {plan['generated_at']}; start a new run instead of re-running this one")
    fields = dict(plan["checks"][check_id]["definition"])
    for key, value in fields.items():
        if isinstance(value, list):
            fields[key] = tuple(value)
    return Check(**fields)


def matrix(plan: dict[str, Any]) -> dict[str, list[dict[str, str]]]:
    return {"include": [{"check": cid, "shard": s["id"]} for cid, c in plan["checks"].items() for s in c["shards"]]}


def find_shard(plan: dict[str, Any], check_id: str, shard_id: str) -> dict[str, Any]:
    try:
        return next(s for s in plan["checks"][check_id]["shards"] if s["id"] == shard_id)
    except (KeyError, StopIteration):
        raise CheckError(f"shard `{shard_id}` of check `{check_id}` is not in the plan") from None


# --------------------------------------------------------------------------------------------------
# Prompt
# --------------------------------------------------------------------------------------------------

RESULTS_CONTRACT = """\
## Output contract

Write your results as JSON to `{results}`. Rewrite the whole file each time you finish a file, not
only at the end: a run cut short by a timeout or turn limit keeps what that file records. If the
file is missing or malformed, the runner records nothing as reviewed.

```json
{{
  "reviewed": ["<repo-relative path of every assigned file you fully reviewed>"],
  "findings": [
    {{
      "file": "<assigned repo-relative path>",
      "line": 123,
      "severity": "high | medium | low",
      "category": "<kebab-case category from the rubric>",
      "title": "<one line>",
      "detail": "<what is wrong, why it matters, and the fix you made or suggest>",
      "fingerprint": "<stable id, built as the rubric says; letters, digits and ./:#@+=,_- only>",
      "fixed": true
    }}
  ],
  "summary": "<Markdown for the pull request body: what you changed and why, in a few lines>"
}}
```

Rules the runner enforces:

- Only files assigned below may be reviewed or reported on. Findings about other files are dropped.
- {edit_rule} Edits elsewhere are reverted. Never edit `Cargo.lock`.
- Do not run `git commit`, `git push`, or open pull requests or issues: the runner does that.
- List a file in `reviewed` only if you reviewed all of it, and finished and checked any edits to
  it. Edits to files missing from `reviewed` are reverted. Files you skip are offered again later.
- `fixed: true` only for findings your working-tree edits resolve. Leave the rest `false`.
{verify_rule}
- Precision over recall. Zero findings is a good result when the code is fine.
- At most {max_findings} findings are kept, most severe first.
"""


def render_prompt(check: Check, plan: dict[str, Any], shard: dict[str, Any], results: str) -> str:
    base = plan["base_commit"]
    out = [f"# Agent check `{check.id}` (version {check.version}): {check.title}\n", check.body]
    if check.skills:
        names = ", ".join(f"`{s}`" for s in check.skills)
        out.append(f"\nLoad these skills before you start: {names}.\n")
    if check.edit_scope == "crate":
        dirs = ", ".join(f"`{d or '.'}/`" for d in shard["edit_scope"])
        edit_rule = f"Edits may touch only files inside {dirs}."
    else:
        edit_rule = (
            "Edits may touch only your assigned files. Other agents are editing other files at the same "
            "time, so report problems that need changes elsewhere instead of making them."
        )
    if check.verify:
        commands = "\n".join(f"  `{c.format(crate=shard['crate'])}`" for c in check.verify)
        verify_rule = (
            "- After you finish, the runner runs these commands. If any fails, for any reason, every edit in this\n"
            "  shard is discarded and its findings are reported unfixed, so run them yourself before you finish\n"
            "  and undo edits that make them fail:\n" + commands
        )
    else:
        verify_rule = ""
    out.append(
        "\n"
        + RESULTS_CONTRACT.format(
            results=results, edit_rule=edit_rule, verify_rule=verify_rule, max_findings=check.max_findings
        )
    )
    out.append(f"\n## Assignment\n\nShard `{shard['id']}`, base commit `{base}`. Files:\n")
    for f in shard["files"]:
        if f["status"] == "changed" and f["last_commit"]:
            note = f"changed since the last review; `git diff {f['last_commit'][:12]} -- {f['path']}` shows how"
        else:
            note = {
                "new": "never reviewed",
                "changed": "changed since the last review",
                "rubric": "unchanged, but the rubric changed since the last review",
                "stale": "unchanged; due for a periodic re-review",
                "full": "full review",
            }[f["status"]]
        out.append(f"- `{f['path']}` ({f['lines']} lines): {note}\n")
    paths = {f["path"] for f in shard["files"]}
    suppressed = {
        fp: info for fp, info in plan["checks"][check.id].get("suppressed", {}).items() if info.get("file") in paths
    }
    if suppressed:
        out.append("\nMaintainers rejected these findings before. Do not report them again:\n\n")
        out.extend(f"- `{fp}`\n" for fp in sorted(suppressed))
    return "".join(out)


# --------------------------------------------------------------------------------------------------
# Finalize
# --------------------------------------------------------------------------------------------------


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise CheckError(f"results: {msg}")


def validate_results(data: object) -> dict[str, Any]:
    _require(isinstance(data, dict), "top level must be an object")
    assert isinstance(data, dict)
    _require(isinstance(data.get("reviewed"), list), "`reviewed` must be a list")
    _require(all(isinstance(p, str) for p in data["reviewed"]), "`reviewed` must hold paths")
    _require(isinstance(data.get("findings"), list), "`findings` must be a list")
    _require(isinstance(data.get("summary", ""), str), "`summary` must be a string")
    # One malformed finding must not throw away the whole review: drop it and say so.
    valid, invalid = [], []
    for i, f in enumerate(data["findings"]):
        if problem := finding_problem(f):
            invalid.append(f"dropped malformed finding {i}: {problem}")
        else:
            valid.append(f)
    return {**data, "findings": valid, "invalid": invalid}


def finding_problem(f: object) -> str | None:
    if not isinstance(f, dict):
        return "not an object"
    for key in ("file", "severity", "category", "title", "detail", "fingerprint"):
        if not (isinstance(f.get(key), str) and f[key]):
            return f"needs string `{key}`"
    if f["severity"] not in SEVERITIES:
        return f"unknown severity {f['severity']!r}"
    if not FINGERPRINT_RE.fullmatch(f["fingerprint"]):
        return f"fingerprint {f['fingerprint'][:80]!r} has characters outside {FINGERPRINT_RE.pattern}"
    if not isinstance(f.get("line", 0), int) or isinstance(f.get("line", 0), bool):
        return "`line` must be an integer"
    if not isinstance(f.get("fixed", False), bool):
        return "`fixed` must be a boolean"
    return None


def working_tree_changes(root: Path) -> list[str]:
    out = git(root, "status", "--porcelain", "-z", "--untracked-files=all")
    paths: list[str] = []
    entries = out.split("\0")
    i = 0
    while i < len(entries):
        entry = entries[i]
        if entry:
            paths.append(entry[3:])
            if entry[0] in "RC":  # renames carry the old path as the next entry
                i += 1
        i += 1
    return paths


def _within(path: str, dirs: list[str]) -> bool:
    return any(d == "" or path == d or path.startswith(d + "/") for d in dirs)


def revert(root: Path, paths: list[str]) -> None:
    tracked = tracked_blobs(root)
    for path in paths:
        if path in tracked:
            git(root, "checkout", "--", path)
        else:
            target = root / path
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink(missing_ok=True)


def finalize(
    check: Check,
    plan: dict[str, Any],
    shard: dict[str, Any],
    results: dict[str, Any],
    root: Path,
    *,
    now: dt.datetime,
    run_url: str | None = None,
    revert_out_of_scope: bool = False,
    run_verify: bool = False,
) -> dict[str, Any]:
    assigned = {f["path"]: f for f in shard["files"]}
    suppressed = plan["checks"][check.id].get("suppressed", {})
    notes: list[str] = list(results.get("invalid", []))

    reviewed = [p for p in dict.fromkeys(results["reviewed"]) if p in assigned]
    if extra := sorted(set(results["reviewed"]) - set(assigned)):
        notes.append(f"ignored reviewed paths outside the shard: {extra}")

    findings = []
    for f in results["findings"]:
        if f["file"] not in assigned:
            notes.append(f"dropped finding outside the shard: {f['file']}")
        elif f["fingerprint"] in suppressed:
            notes.append(f"dropped suppressed finding: {f['fingerprint']}")
        else:
            findings.append(f)
    findings.sort(key=lambda f: (SEVERITIES.index(f["severity"]), f["file"], f.get("line", 0)))
    if len(findings) > check.max_findings:
        notes.append(f"kept {check.max_findings} of {len(findings)} findings")
        findings = findings[: check.max_findings]

    # With file scope, only files the agent finished may change: an edit to a file it did not list
    # as reviewed may be half done, for example when the run hit its time or turn limit.
    # A `report` check never commits edits.
    if check.action == "report":
        scope: list[str] = []
    elif check.edit_scope == "files":
        scope = reviewed
    else:
        scope = shard["edit_scope"]
    changed = working_tree_changes(root)
    allowed = [p for p in changed if _within(p, scope) and not p.endswith("Cargo.lock")]
    rejected = [p for p in changed if p not in allowed]
    if rejected:
        notes.append(f"edits outside the scope {'reverted' if revert_out_of_scope else 'excluded'}: {rejected}")
        if revert_out_of_scope:
            revert(root, rejected)
    edits = allowed

    if edits and run_verify and check.verify:
        failure = verify_edits(check, shard, root)
        if failure:
            notes.append(f"{failure}; all edits were reverted and their findings are reported unfixed")
            revert(root, edits)
            edits = []
    # A finding is fixed only if the edit to its file survived the checks above.
    findings = [{**f, "fixed": bool(f.get("fixed")) and f["file"] in edits} for f in findings]

    reviewed_at = now.isoformat(timespec="seconds")
    update_files = {}
    for path in reviewed:
        blobs = [assigned[path]["blob"]]
        if path in edits and (root / path).exists():
            blobs.append(git(root, "hash-object", path).strip())
        update_files[path] = {
            "blobs": blobs,
            "commit": plan["base_commit"],
            "reviewed_at": reviewed_at,
            "version": check.version,
            "findings": sorted(f["fingerprint"] for f in findings if f["file"] == path),
        }

    title = f"{check.title}: {shard['id']}"
    return {
        "check": check.id,
        "shard": shard["id"],
        "branch": shard["branch"],
        "title": title,
        "labels": list(check.labels),
        "edits": edits,
        "findings": findings,
        "notes": notes,
        "body": render_body(check, plan, shard, results.get("summary", ""), findings, edits, notes, run_url),
        "state_update": {"check": check.id, "run_at": reviewed_at, "files": update_files},
    }


def _safe(text: str) -> str:
    """Neutralize HTML comments in agent-written text so it cannot forge the runner's marker."""
    return text.replace("<!--", "&lt;!--").replace("-->", "--&gt;")


def verify_edits(check: Check, shard: dict[str, Any], root: Path) -> str | None:
    """Run the check's `verify` commands; return a description of the first failure, if any."""
    for template in check.verify:
        command = template.format(crate=shard["crate"])
        print(f"verify: {command}", file=sys.stderr, flush=True)
        try:
            result = subprocess.run(shlex.split(command), cwd=root, check=False)
        except OSError as e:
            return f"verification `{command}` could not start: {e}"
        if result.returncode != 0:
            return f"verification `{command}` failed with exit code {result.returncode}"
    return None


def render_body(
    check: Check,
    plan: dict[str, Any],
    shard: dict[str, Any],
    summary: str,
    findings: list[dict[str, Any]],
    edits: list[str],
    notes: list[str],
    run_url: str | None,
) -> str:
    out = [
        f"Automated `{check.id}` review of `{shard['id']}` at {plan['base_commit'][:12]}.\n\n",
        _safe(summary.strip()) + "\n\n" if summary.strip() else "",
    ]

    def bullet(f: dict[str, Any]) -> str:
        loc = f"{f['file']}:{f['line']}" if f.get("line") else f["file"]
        detail = _safe(f["detail"].strip())
        return f"- **{f['severity']}** `{loc}` ({_safe(f['category'])}): {_safe(f['title'])}\n  {detail}\n"

    fixed = [f for f in findings if f["fixed"]]
    open_ = [f for f in findings if not f["fixed"]]
    if fixed:
        out.append("### Fixed in this PR\n\n")
        out.extend(bullet(f) for f in fixed)
        out.append("\n")
    if open_:
        out.append("### Not fixed: needs a maintainer decision\n\n")
        out.extend(bullet(f) for f in open_)
        out.append("\n")
    if not findings:
        out.append("No findings.\n\n")
    if notes:
        out.append("<details><summary>Runner notes</summary>\n\n")
        out.extend(f"- {n}\n" for n in notes)
        out.append("\n</details>\n\n")
    if edits:
        out.append("Closing this PR without merging suppresses all its findings in future runs. ")
    else:
        out.append("Closing this issue as not planned suppresses all its findings in future runs. ")
    out.append(f"The rubric is `.agents/checks/{check.id}/CHECK.md`.\n")
    if run_url:
        out.append(f"\n[Agent run]({run_url})\n")
    marker = {
        "check": check.id,
        "shard": shard["id"],
        "files": sorted(f["path"] for f in shard["files"]),
        "claims": sorted({*shard["edit_scope"], *edits}),
        "fingerprints": {f["fingerprint"]: f["file"] for f in findings},
    }
    out.append(f"\n<!-- agent-check: {json.dumps(marker, sort_keys=True)} -->\n")
    return "".join(out)


# --------------------------------------------------------------------------------------------------
# Record
# --------------------------------------------------------------------------------------------------


def record(
    state_dir: Path,
    updates: list[dict[str, Any]],
    *,
    plan: dict[str, Any] | None = None,
    items: list[dict[str, Any]] | None = None,
) -> list[str]:
    """Merge state updates and suppressions into `state_dir`. Returns the touched check ids.

    A due check gets `last_run` set to the plan's start time if at least one of its shards was
    recorded, or if it had nothing to review. When every shard failed, the clock is left alone so
    the next daily run tries again.

    Merging is idempotent, and a file keeps whichever review is newer, so retrying a push or
    re-running an old run never rolls the state back.
    """
    suppressed = suppressions_from_items(items or [])
    plan_checks = (plan or {}).get("checks", {})
    updated: set[str] = {str(u["check"]) for u in updates}
    stamped = {cid for cid, c in plan_checks.items() if c.get("due") and (cid in updated or not c.get("shards"))}
    touched: list[str] = sorted(updated | stamped | set(suppressed))
    for check_id in touched:
        if not CHECK_ID_RE.fullmatch(check_id):
            raise CheckError(f"invalid check id {check_id!r} in a state update")
        state = load_state(state_dir, check_id)
        for update in updates:
            if update["check"] != check_id:
                continue
            for path, entry in update["files"].items():
                current = state["files"].get(path)
                if current is None or str(entry["reviewed_at"]) >= str(current.get("reviewed_at", "")):
                    state["files"][path] = entry
        state["suppressed"].update(suppressed.get(check_id, {}))
        if check_id in stamped:
            stamp = str((plan or {})["generated_at"])
            if not state.get("last_run") or stamp > state["last_run"]:
                state["last_run"] = stamp
        save_state(state_dir, state)
    return touched


# --------------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------------


def _now(args: argparse.Namespace) -> dt.datetime:
    if getattr(args, "now", None):
        return dt.datetime.fromisoformat(args.now)
    return dt.datetime.now(dt.UTC).replace(microsecond=0)


def _read_json(path: str) -> object:
    try:
        return json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as e:
        raise CheckError(f"cannot read {path}: {e}") from None


def _read_dict(path: str) -> dict[str, Any]:
    data = _read_json(path)
    if not isinstance(data, dict):
        raise CheckError(f"{path}: expected a JSON object")
    return data


def _read_items(path: str | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    data = _read_json(path)
    if not isinstance(data, list) or not all(isinstance(item, dict) for item in data):
        raise CheckError(f"{path}: expected a JSON list of `gh pr list` / `gh issue list` items")
    return data


def _state_dir(args: argparse.Namespace) -> Path:
    return Path(args.state_dir) if args.state_dir is not None else Path(args.root) / DEFAULT_LOCAL_STATE


def _select(checks: dict[str, Check], ids: list[str] | None) -> list[Check]:
    if ids:
        missing = [i for i in ids if i not in checks]
        if missing:
            raise CheckError(f"unknown checks {missing}; known: {sorted(checks)}")
        return [checks[i] for i in ids]
    return [c for c in checks.values() if c.enabled]


def _github_output(values: dict[str, str]) -> None:
    path = os.environ.get("GITHUB_OUTPUT")
    if not path:
        return
    with open(path, "a") as f:
        for key, value in values.items():
            if "\n" in value:
                f.write(f"{key}<<__AGENT_CHECKS_EOF__\n{value}\n__AGENT_CHECKS_EOF__\n")
            else:
                f.write(f"{key}={value}\n")


def cmd_list(args: argparse.Namespace) -> int:
    for check in load_checks(args.root).values():
        flag = "" if check.enabled else " (disabled)"
        print(
            f"{check.id}{flag}: {check.title}\n  edit_scope={check.edit_scope} mode={check.mode} "
            f"action={check.action} cadence={check.cadence_days}d "
            f"max_shards_per_run={check.max_shards_per_run} include={list(check.include)}"
        )
    return 0


def cmd_plan(args: argparse.Namespace) -> int:
    checks = _select(load_checks(args.root), args.check)
    plan = build_plan(
        checks,
        args.root,
        _state_dir(args),
        now=_now(args),
        items=_read_items(args.github),
        only_shards=args.shard,
        max_shards=args.max_shards,
        force=args.force,
    )
    text = json.dumps(plan, indent=2) + "\n"
    if args.out:
        Path(args.out).write_text(text)
    else:
        sys.stdout.write(text)
    m = matrix(plan)
    for entry in m["include"]:
        print(f"planned {entry['check']} / {entry['shard']}", file=sys.stderr)
    _github_output(
        {"matrix": json.dumps(m), "count": str(len(m["include"])), "checks": json.dumps(list(plan["checks"]))}
    )
    return 0


def cmd_prompt(args: argparse.Namespace) -> int:
    plan = _read_dict(args.plan)
    check = check_from_plan(plan, args.check, now=_now(args))
    shard = find_shard(plan, args.check, args.shard)
    prompt = render_prompt(check, plan, shard, args.results)
    if args.out:
        Path(args.out).write_text(prompt)
    else:
        sys.stdout.write(prompt)
    _github_output(
        {
            "allowed_tools": ",".join(check.tools),
            "model": check.model or "",
            "max_turns": str(check.max_turns),
            "action": check.action,
            "branch": shard["branch"],
            "prompt": prompt,
        }
    )
    return 0


def cmd_finalize(args: argparse.Namespace) -> int:
    plan = _read_dict(args.plan)
    check = check_from_plan(plan, args.check, now=_now(args))
    shard = find_shard(plan, args.check, args.shard)
    results = validate_results(_read_json(args.results))
    out = finalize(
        check,
        plan,
        shard,
        results,
        args.root,
        now=_now(args),
        run_url=args.run_url,
        revert_out_of_scope=args.revert_out_of_scope,
        run_verify=args.verify,
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "body.md").write_text(out["body"])
    (out_dir / "state-update.json").write_text(json.dumps(out["state_update"], indent=2) + "\n")
    (out_dir / "summary.json").write_text(json.dumps({k: v for k, v in out.items() if k != "body"}, indent=2) + "\n")
    for note in out["notes"]:
        print(f"note: {note}", file=sys.stderr)
    print(
        f"{check.id}/{shard['id']}: reviewed {len(out['state_update']['files'])}/{len(shard['files'])} files, "
        f"{len(out['findings'])} findings, {len(out['edits'])} edited files",
        file=sys.stderr,
    )
    _github_output(
        {
            "has_edits": str(bool(out["edits"])).lower(),
            "has_findings": str(bool(out["findings"])).lower(),
            "has_unfixed": str(any(not f["fixed"] for f in out["findings"])).lower(),
            "title": out["title"],
            "branch": out["branch"],
            "labels": ",".join(out["labels"]),
        }
    )
    return 0


def cmd_record(args: argparse.Namespace) -> int:
    updates = [_read_dict(p) for p in args.updates]
    plan = _read_dict(args.plan) if args.plan else None
    touched = record(args.state_dir, updates, plan=plan, items=_read_items(args.github))
    print(f"updated state for {touched}", file=sys.stderr)
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    """Run one check locally against the working tree, using the `claude` CLI."""
    root = Path(args.root)
    check = load_checks(root)[args.check]
    now = _now(args)
    work = root / "target" / "agent-checks" / check.id
    work.mkdir(parents=True, exist_ok=True)
    state_dir = _state_dir(args)
    plan = build_plan([check], root, state_dir, now=now, only_shards=args.shard, max_shards=args.max_shards, force=True)
    shards = plan["checks"][check.id]["shards"]
    if not shards:
        print("nothing to review", file=sys.stderr)
        return 0
    (work / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    if args.dry_run:
        for shard in shards:
            print(render_prompt(check, plan, shard, str(work / f"{shard_slug(shard['id'])}.results.json")))
        return 0
    if working_tree_changes(root):
        raise CheckError("`run` needs a clean working tree so it can tell the agent's edits apart")
    claude = shutil.which("claude")
    if claude is None:
        raise CheckError("the `claude` CLI is not on PATH")
    updates = []
    for shard in shards:
        results_path = work / f"{shard_slug(shard['id'])}.results.json"
        results_path.unlink(missing_ok=True)
        prompt = render_prompt(check, plan, shard, str(results_path))
        cmd = [claude, "-p", prompt, "--allowedTools", ",".join(check.tools), "--max-turns", str(check.max_turns)]
        if check.model:
            cmd += ["--model", check.model]
        print(f"running {check.id} / {shard['id']} ({len(shard['files'])} files)", file=sys.stderr)
        subprocess.run(cmd, cwd=root, check=False)
        out = finalize(
            check,
            plan,
            shard,
            validate_results(_read_json(str(results_path)) if results_path.exists() else None),
            root,
            now=now,
            run_verify=True,
        )
        body_path = work / f"{shard_slug(shard['id'])}.md"
        body_path.write_text(out["body"])
        print(f"wrote {body_path}; edits left in the working tree: {out['edits']}", file=sys.stderr)
        updates.append(out["state_update"])
        if len(shards) > 1 and working_tree_changes(root):
            print("stopping after the first shard with edits; review and commit them, then rerun", file=sys.stderr)
            break
    record(state_dir, updates, plan=plan)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=REPO_ROOT, help="repository root")
    parser.add_argument("--now", help="override the current time (ISO 8601), for tests")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="list checks").set_defaults(func=cmd_list)

    p = sub.add_parser("plan", help="choose shards to review")
    p.add_argument("--check", action="append", help="check id (repeatable; default: all enabled)")
    p.add_argument("--state-dir", type=Path, help="default: <root>/target/agent-checks-state")
    p.add_argument(
        "--github",
        help="JSON list of `gh pr list` / `gh issue list` items (number, state, body; PRs: headRefName, "
        "mergedAt; issues: stateReason, comments)",
    )
    p.add_argument("--shard", action="append", help="plan only this crate (repeatable)")
    p.add_argument("--max-shards", type=int, help="override max_shards_per_run")
    p.add_argument("--force", action="store_true", help="ignore cadence_days")
    p.add_argument("--out", help="write the plan here instead of stdout")
    p.set_defaults(func=cmd_plan)

    p = sub.add_parser("prompt", help="render the agent prompt for one shard")
    p.add_argument("--plan", required=True)
    p.add_argument("--check", required=True)
    p.add_argument("--shard", required=True)
    p.add_argument("--results", required=True, help="path the agent must write its results to")
    p.add_argument("--out")
    p.set_defaults(func=cmd_prompt)

    p = sub.add_parser("finalize", help="validate results and build the PR body and state update")
    p.add_argument("--plan", required=True)
    p.add_argument("--check", required=True)
    p.add_argument("--shard", required=True)
    p.add_argument("--results", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--run-url")
    p.add_argument("--revert-out-of-scope", action="store_true")
    p.add_argument("--verify", action="store_true", help="run the check's `verify` commands on edits")
    p.set_defaults(func=cmd_finalize)

    p = sub.add_parser("record", help="merge state updates into the state dir")
    p.add_argument("--state-dir", type=Path, required=True)
    p.add_argument("--plan", help="the run's plan; its due checks get `last_run` stamped")
    p.add_argument(
        "--github",
        help="JSON list of `gh pr list` / `gh issue list` items (number, state, body; PRs: headRefName, "
        "mergedAt; issues: stateReason, comments)",
    )
    p.add_argument("updates", nargs="*")
    p.set_defaults(func=cmd_record)

    p = sub.add_parser("run", help="run a check locally with the `claude` CLI")
    p.add_argument("check")
    p.add_argument("--shard", action="append", help="review only this crate (repeatable)")
    p.add_argument("--max-shards", type=int, default=1)
    p.add_argument("--state-dir", type=Path, help="default: <root>/target/agent-checks-state")
    p.add_argument("--dry-run", action="store_true", help="print prompts without running the agent")
    p.set_defaults(func=cmd_run)

    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except CheckError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
