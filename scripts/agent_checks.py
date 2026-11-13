#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Runner for periodic, agent-driven code checks.

A check is a directory `.agents/checks/<id>/` holding a `CHECK.md`: TOML frontmatter between `+++`
lines declares what the check covers and how it is sharded, and the Markdown body is the rubric
the agent follows. See `.agents/checks/README.md` for the format.

The runner is deliberately agent-agnostic. It plans work, renders a prompt, and validates what the
agent wrote back; the agent itself is launched by CI (`.github/workflows/agent-checks-shard.yml`,
called from `agent-checks.yml`) or by `run` for local use.

Subcommands:

    list                       List checks and their configuration.
    plan                       Pick shards that need review and write a plan JSON.
    prompt                     Render the agent prompt for one shard of a plan.
    apply                      Apply the agent's patch on a fresh checkout, limited to the shard's scope.
    finalize                   Validate the results, revert out-of-scope edits, format, verify.
    decide                     From live GitHub state, decide what a shard publishes and dedup findings.
    record                     Merge state updates and suppressions into the state store, with CAS.
    archive                    Keep a write-once record of a run under each check's `runs/` prefix.
    run                        Local convenience: plan, prompt, invoke `claude -p`, finalize, record.

Each check owns a storage prefix, `<store>/<check>/`: the mutable `state.json`, and write-once run
records under `runs/<run>/`. CI keeps it in S3 (`--state s3://bucket/prefix`); locally it defaults
to the directory `target/agent-checks-state`.
Writes are compare-and-swap (S3 conditional writes on the ETag), so concurrent writers never lose
each other's updates. CI records every published shard once, at the end of the run. For every
reviewed file the state keeps the git blob hashes that were reviewed (before and after the agent's
own fix), so unchanged files are skipped and a merged fix does not trigger its own re-review.

PR and issue bodies carry a hidden marker. Only markers by a trusted author (`--trusted-author`,
the App's login in CI) on same-repository items count; see `GitHubView`.

Run the tests with:

    uv run --no-project --with pytest pytest scripts/tests/test_agent_checks.py
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import fcntl
import glob
import hashlib
import itertools
import json
import os
import random
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import tomllib
from collections.abc import Callable, Collection, Sequence
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


def _parse_state(raw: bytes | None, check_id: str, where: str) -> dict[str, Any]:
    if raw is None:
        return empty_state(check_id)
    try:
        state = json.loads(raw)
    except json.JSONDecodeError as e:
        raise CheckError(f"{where}: corrupt state: {e}") from None
    if not isinstance(state, dict) or state.get("schema") != STATE_SCHEMA:
        raise CheckError(
            f"{where}: unsupported state schema {state.get('schema') if isinstance(state, dict) else state!r}"
        )
    return state


def _dump_state(state: dict[str, Any]) -> bytes:
    return (json.dumps(state, indent=2, sort_keys=True) + "\n").encode()


class StateConflict(Exception):
    """Another writer changed the state since it was read."""


class CheckStore:
    """Per-check storage: everything a check keeps lives under its own prefix, `<root>/<check>/`.

    Objects are bytes under a relative key. Two write modes cover everything the runner stores:
    - compare-and-swap (`put` with the token from `get`), for the mutable `state.json`;
    - write-once (`put_new`), for the immutable run records under `runs/`.

    `get` returns the object and a version token (None if it does not exist). `put` stores the
    object only if the stored version still matches the token, None meaning "must not exist yet",
    and raises `StateConflict` otherwise.
    """

    def get(self, check_id: str, key: str) -> tuple[bytes | None, str | None]:
        raise NotImplementedError

    def put(self, check_id: str, key: str, data: bytes, token: str | None) -> None:
        raise NotImplementedError

    def put_new(self, check_id: str, key: str, data: bytes) -> bool:
        """Write an object that must not exist yet. Returns False, writing nothing, if it does."""
        try:
            self.put(check_id, key, data, None)
            return True
        except StateConflict:
            return False

    def read(self, check_id: str) -> tuple[dict[str, Any], str | None]:
        raw, token = self.get(check_id, STATE_KEY)
        return _parse_state(raw, check_id, self.where(check_id, STATE_KEY)), token

    def write(self, check_id: str, state: dict[str, Any], token: str | None) -> None:
        self.put(check_id, STATE_KEY, _dump_state(state), token)

    def where(self, check_id: str, key: str) -> str:
        raise NotImplementedError


STATE_KEY = "state.json"
_KEY_RE = re.compile(r"[A-Za-z0-9._-]+(/[A-Za-z0-9._-]+)*")


def _check_key(check_id: str, key: str) -> None:
    if not CHECK_ID_RE.fullmatch(check_id) or not _KEY_RE.fullmatch(key) or ".." in key.split("/"):
        raise CheckError(f"invalid storage key {check_id!r}/{key!r}")


class DirStore(CheckStore):
    """`<dir>/<check>/<key>` on disk, for local runs and tests. The token is a content hash."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def where(self, check_id: str, key: str) -> str:
        return str(self.root / check_id / key)

    def get(self, check_id: str, key: str) -> tuple[bytes | None, str | None]:
        _check_key(check_id, key)
        path = self.root / check_id / key
        raw = path.read_bytes() if path.exists() else None
        return raw, hashlib.sha256(raw).hexdigest() if raw is not None else None

    def put(self, check_id: str, key: str, data: bytes, token: str | None) -> None:
        _check_key(check_id, key)
        path = self.root / check_id / key
        path.parent.mkdir(parents=True, exist_ok=True)
        (self.root / check_id).mkdir(parents=True, exist_ok=True)
        with open(self.root / check_id / ".lock", "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            current = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
            if current != token:
                raise StateConflict(str(path))
            tmp = path.with_name(path.name + ".tmp")
            tmp.write_bytes(data)
            os.replace(tmp, path)


class S3Store(CheckStore):
    """`s3://<bucket>/<prefix>/<check>/<key>`, written with S3 conditional writes.

    The token is the object's ETag. A write sends `If-Match: <etag>`, or `If-None-Match: *` when
    the object must not exist yet, so S3 itself rejects a write based on a stale read (HTTP 412,
    or 409 for a concurrent conditional write). Uses the AWS CLI so the runner needs no packages;
    credentials and region come from the environment.
    """

    def __init__(self, uri: str, aws: str = "aws") -> None:
        m = re.fullmatch(r"s3://([^/]+)/?(.*)", uri)
        if not m:
            raise CheckError(f"not an s3:// URI: {uri!r}")
        self.bucket, self.prefix, self.aws = m.group(1), m.group(2).strip("/"), aws

    def key(self, check_id: str, key: str = STATE_KEY) -> str:
        return "/".join(p for p in (self.prefix, check_id, key) if p)

    def where(self, check_id: str, key: str) -> str:
        return f"s3://{self.bucket}/{self.key(check_id, key)}"

    def _s3api(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run([self.aws, "s3api", *args, "--output", "json"], capture_output=True, text=True)

    def get(self, check_id: str, key: str) -> tuple[bytes | None, str | None]:
        _check_key(check_id, key)
        with tempfile.TemporaryDirectory() as tmp:
            body = Path(tmp) / "object"
            r = self._s3api("get-object", "--bucket", self.bucket, "--key", self.key(check_id, key), str(body))
            if r.returncode != 0:
                # Only a missing object means "not there". AccessDenied (for example without
                # s3:ListBucket, S3 answers 403 for a missing key) must not reset the state.
                if "NoSuchKey" in r.stderr:
                    return None, None
                raise CheckError(f"{self.where(check_id, key)}: get-object failed: {r.stderr.strip()}")
            return body.read_bytes(), str(json.loads(r.stdout)["ETag"])

    def put(self, check_id: str, key: str, data: bytes, token: str | None) -> None:
        _check_key(check_id, key)
        condition = ["--if-match", token] if token is not None else ["--if-none-match", "*"]
        content_type = "application/json" if key.endswith(".json") else "text/plain; charset=utf-8"
        with tempfile.TemporaryDirectory() as tmp:
            body = Path(tmp) / "object"
            body.write_bytes(data)
            r = self._s3api(
                "put-object",
                "--bucket",
                self.bucket,
                "--key",
                self.key(check_id, key),
                "--body",
                str(body),
                "--content-type",
                content_type,
                *condition,
            )
        if r.returncode != 0:
            # A conditional write against a version that no longer exists (412 or 409), or against an
            # object deleted since it was read (NoSuchKey), is a conflict: re-read and try again.
            conflicts = ("PreconditionFailed", "ConditionalRequestConflict", "NoSuchKey", "(412)", "(409)")
            if any(code in r.stderr for code in conflicts):
                raise StateConflict(self.where(check_id, key))
            raise CheckError(f"{self.where(check_id, key)}: put-object failed: {r.stderr.strip()}")


def open_store(spec: str | Path) -> CheckStore:
    """`s3://bucket/prefix` for S3, anything else is a local directory."""
    return S3Store(str(spec)) if str(spec).startswith("s3://") else DirStore(Path(spec))


def load_state(store: CheckStore, check_id: str) -> dict[str, Any]:
    return store.read(check_id)[0]


CAS_ATTEMPTS = 8


def update_state(store: CheckStore, check_id: str, change: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
    """Apply `change` to the stored state with compare-and-swap, retrying on conflicts."""
    for attempt in range(CAS_ATTEMPTS):
        state, token = store.read(check_id)
        before = _dump_state(state)
        change(state)
        if _dump_state(state) == before:
            return state
        try:
            store.write(check_id, state, token)
            return state
        except StateConflict:
            time.sleep(min(30.0, 0.5 * 2**attempt) * (0.5 + random.random()))
    raise CheckError(f"state for `{check_id}` kept changing underneath; gave up after {CAS_ATTEMPTS} attempts")


def is_pr(item: dict[str, Any]) -> bool:
    return "headRefName" in item


def _login(entity: object) -> str:
    """A GitHub login, normalized so `app/x`, `x[bot]` and `X` compare equal."""
    login = str(entity.get("login", "")) if isinstance(entity, dict) else ""
    login = login.lower().removeprefix("app/").removesuffix("[bot]")
    return login


def valid_marker(data: object) -> bool:
    """Whether `data` has the shape the runner writes. Anything else is ignored, never trusted."""
    if not isinstance(data, dict) or not isinstance(data.get("check"), str):
        return False
    if not CHECK_ID_RE.fullmatch(data["check"]) or not isinstance(data.get("shard", ""), str):
        return False
    for key in ("files", "claims"):
        if not isinstance(data.get(key, []), list) or not all(isinstance(p, str) for p in data.get(key, [])):
            return False
    fps = data.get("fingerprints", {})
    return isinstance(fps, dict) and all(
        isinstance(fp, str) and FINGERPRINT_RE.fullmatch(fp) and isinstance(f, str) for fp, f in fps.items()
    )


def parse_markers(body: str) -> list[dict[str, Any]]:
    markers = []
    for m in MARKER_RE.finditer(body):
        try:
            data = json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
        if valid_marker(data):
            markers.append(data)
    return markers


class GitHubView:
    """The PRs and issues a run sees, with only the runner's own markers trusted.

    `items` is `gh pr list` / `gh issue list` JSON. A marker counts only if it is the last one in
    a body or comment written by one of `trusted` (the App's login), on an issue or a PR from this
    repository, never a fork. Anyone can open a PR or comment, so without this a stranger could
    claim every file, hide findings, or suppress them. `trusted=None` trusts every author, for
    local use and tests that do not model authors.
    """

    def __init__(self, items: list[dict[str, Any]], trusted: Collection[str] | None = None) -> None:
        self.items = items
        self.trusted = None if trusted is None else {_login({"login": t}) for t in trusted if t}

    def _trusts(self, author: object) -> bool:
        return self.trusted is None or _login(author) in self.trusted

    def markers(self, item: dict[str, Any]) -> list[dict[str, Any]]:
        if is_pr(item) and item.get("isCrossRepository"):
            return []
        texts = []
        if self._trusts(item.get("author")):
            texts.append(item.get("body") or "")
        for comment in item.get("comments") or []:
            if isinstance(comment, dict) and self._trusts(comment.get("author")):
                texts.append(comment.get("body") or "")
        out = []
        for text in texts:
            # The runner appends its marker after any agent-written text, so only the last counts.
            if found := parse_markers(text):
                out.append(found[-1])
        return out

    @staticmethod
    def is_open(item: dict[str, Any]) -> bool:
        return str(item.get("state", "")).upper() == "OPEN"

    def open_agent_prs(self) -> list[dict[str, Any]]:
        # Matched by branch, not label: removing the label from an open PR must not make its branch
        # look free, or the next run would force-push over it. Fork PRs cannot block our branches.
        return [
            i
            for i in self.items
            if is_pr(i)
            and self.is_open(i)
            and not i.get("isCrossRepository")
            and str(i["headRefName"]).startswith(BRANCH_PREFIX + "/")
        ]

    def busy_branches(self) -> set[str]:
        return {str(pr["headRefName"]) for pr in self.open_agent_prs()}

    def claimed_paths(self) -> set[str]:
        """Files and directories that open agent PRs, of any check, may still change.

        No shard may review or edit them until the PR is merged or closed, so two PRs never touch
        the same file. Issues claim nothing: they carry no edits.
        """
        out: set[str] = set()
        for pr in self.open_agent_prs():
            for marker in self.markers(pr):
                out.update(marker.get("claims", []))
        return out

    def rejections(self) -> dict[str, dict[int, dict[str, str]]]:
        """Per check, the agent PRs closed unmerged and issues closed as not planned.

        Maps check id -> item number -> {fingerprint: file}. Closing either rejects every finding
        listed in it, comments included.
        """
        out: dict[str, dict[int, dict[str, str]]] = {}
        for item in self.items:
            if str(item.get("state", "")).upper() not in ("CLOSED", "MERGED"):
                continue
            if is_pr(item):
                if item.get("mergedAt") or str(item.get("state")).upper() == "MERGED":
                    continue
            elif str(item.get("stateReason") or "").upper() != "NOT_PLANNED":
                continue
            for marker in self.markers(item):
                fps = out.setdefault(marker["check"], {}).setdefault(int(item.get("number") or 0), {})
                fps.update(marker.get("fingerprints", {}))
        return out


def new_suppressions(state: dict[str, Any], rejected: dict[int, dict[str, str]]) -> dict[str, dict[str, Any]]:
    """Suppressions from rejected items that `state` has not processed yet.

    Each item is processed once, so a maintainer who deletes a suppression from the state does
    not see it come back on the next run.
    """
    done = set(state.get("suppression_refs", []))
    out: dict[str, dict[str, Any]] = {}
    for ref, fps in rejected.items():
        if ref not in done:
            for fp, file in fps.items():
                out[fp] = {"file": file, "ref": ref}
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
    """Files within the per-shard budget, and always at least one.

    Changed files may use at most half of the file budget while other files are waiting, so a
    crate that changes every week still works through its never-reviewed backlog.
    """
    changed = [f for f in files if f["status"] == "changed"]
    rest = [f for f in files if f["status"] != "changed"]
    if rest:
        changed = changed[: max(1, max_files // 2)]
    taken: list[dict[str, Any]] = []
    lines = 0
    for f in sorted(changed + rest, key=_file_priority):
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

    Slots alternate between crates with changed files and the rotation of all other crates, whose
    order is the oldest last review first (never reviewed counts as oldest). So neither a lot of
    churn nor one crate's large backlog can starve the others.

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

    def rotation(sh: dict[str, Any]) -> tuple[str, str]:
        return last_reviewed.get(str(sh["crate_dir"]), ""), str(sh["id"])

    changed = sorted((sh for sh in shards if sh["changed"]), key=rotation)
    others = sorted((sh for sh in shards if not sh["changed"]), key=rotation)
    shards = [sh for pair in itertools.zip_longest(changed, others) for sh in pair if sh is not None]
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
    store: CheckStore,
    *,
    now: dt.datetime,
    github: GitHubView | None = None,
    only_shards: Collection[str] | None = None,
    max_shards: int | None = None,
    force: bool = False,
    scheduled: bool = False,
) -> dict[str, Any]:
    """Plan a run.

    A check is planned when its cadence has passed, or with `force`. Its `stamp` flag, which tells
    `record` to set `last_run`, is set only for a due check on a `scheduled` run that covers all
    crates. Manual runs never move the weekly clock, so the schedule stays on its weekday.

    A check whose state cannot be read is left out with an `error`, and the other checks are
    planned as usual.
    """
    github = github or GitHubView([])
    busy = github.busy_branches()
    # Shared across checks, so shards of different checks in one run never edit the same file.
    claimed = github.claimed_paths()
    rejections = github.rejections()
    plan: dict[str, Any] = {
        "base_commit": git(root, "rev-parse", "HEAD").strip(),
        "generated_at": now.isoformat(timespec="seconds"),
        "checks": {},
    }
    for check in checks:
        try:
            state = load_state(store, check.id)
        except CheckError as e:
            plan["checks"][check.id] = {
                "definition": dataclasses.asdict(check),
                "stamp": False,
                "shards": [],
                "suppressed": {},
                "error": str(e),
            }
            continue
        due = is_due(check, state, now)
        suppressed = {**state.get("suppressed", {}), **new_suppressions(state, rejections.get(check.id, {}))}
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
            "stamp": due and scheduled and not only_shards,
            "shards": shards,
            "suppressed": suppressed,
        }
    return plan


def check_from_plan(plan: dict[str, Any], check_id: str, *, rerun_at: dt.datetime | None = None) -> Check:
    """The check's definition as planned. `rerun_at` is set when a job is being re-run.

    A re-run of an old workflow run would review stale assignments, so it is refused once the
    plan is older than MAX_PLAN_AGE. A first attempt is never refused: a check's shards run one
    after another, and the last ones may start many hours after planning.
    """
    if check_id not in plan.get("checks", {}):
        raise CheckError(f"check `{check_id}` is not in the plan")
    if rerun_at is not None and rerun_at - dt.datetime.fromisoformat(plan["generated_at"]) > MAX_PLAN_AGE:
        raise CheckError(f"the plan is from {plan['generated_at']}; start a new run instead of re-running this one")
    fields = dict(plan["checks"][check_id]["definition"])
    for key, value in fields.items():
        if isinstance(value, list):
            fields[key] = tuple(value)
    return Check(**fields)


def matrix(plan: dict[str, Any]) -> dict[str, list[dict[str, str]]]:
    return {"include": [{"check": cid, "shard": s["id"]} for cid, c in plan["checks"].items() for s in c["shards"]]}


def check_matrix(plan: dict[str, Any]) -> dict[str, list[dict[str, str]]]:
    """One entry per check that has shards; `shards` is a JSON list for that check's inner matrix."""
    return {
        "include": [
            {"check": cid, "shards": json.dumps([sh["id"] for sh in c["shards"]])}
            for cid, c in plan["checks"].items()
            if c["shards"]
        ]
    }


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
    out = [f"Agent check `{check.id}`, rubric version {check.version}.\n\n", check.body]
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
    clipped = [{**f, "title": _clip(f["title"], MAX_TITLE), "detail": _clip(f["detail"], MAX_DETAIL)} for f in valid]
    return {**data, "findings": clipped, "invalid": invalid, "summary": _clip(data.get("summary", ""), MAX_SUMMARY)}


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
    format_cmd: Sequence[str] = (),
) -> dict[str, Any]:
    """Turn an agent's results into the shard's outcome: findings, kept edits, and a state update.

    Edits are kept only inside the shard's scope and only for files the agent finished. Then they
    are formatted with `format_cmd` and checked with the check's `verify` commands; if either
    fails, every edit is reverted and the findings are reported unfixed.
    """
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

    # Only files the agent finished may change: an edit to an assigned file it did not list as
    # reviewed may be half done, for example when the run hit its time or turn limit. With crate
    # scope, other files in the crate may change too. A `report` check never commits edits.
    unfinished = set(assigned) - set(reviewed)
    if check.action == "report":
        scope: list[str] = []
    elif check.edit_scope == "files":
        scope = reviewed
    else:
        scope = shard["edit_scope"]
    changed = working_tree_changes(root)
    allowed = [p for p in changed if _within(p, scope) and p not in unfinished and not p.endswith("Cargo.lock")]
    rejected = [p for p in changed if p not in allowed]
    if rejected:
        notes.append(f"edits outside the scope {'reverted' if revert_out_of_scope else 'excluded'}: {rejected}")
        if revert_out_of_scope:
            revert(root, rejected)
    edits = allowed

    if edits and format_cmd:
        if failure := format_edits(format_cmd, edits, root):
            notes.append(f"{failure}; all edits were reverted and their findings are reported unfixed")
            revert(root, edits)
            edits = []
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
        "agent_summary": results.get("summary", ""),
        "run_url": run_url,
        "body": render_body(check, plan, shard, results.get("summary", ""), findings, edits, notes, run_url, "summary"),
        "state_update": {"check": check.id, "run_at": reviewed_at, "files": update_files},
    }


# --------------------------------------------------------------------------------------------------
# Decide
# --------------------------------------------------------------------------------------------------


def decide(
    check: Check, plan: dict[str, Any], shard: dict[str, Any], final: dict[str, Any], github: GitHubView
) -> dict[str, Any]:
    """Decide what one shard publishes, from live GitHub state fetched just before publishing.

    Every shard's publish job calls this on its own, so the outcome never depends on what the
    planner saw hours earlier or on other shards. PRs carry only fixes; unfixed findings always go
    to the shard's issue, so they outlive a merged PR. A finding whose fingerprint already appears
    in an open agent PR or issue of the same check is not reported again.

    Returns `pr` and `issue` decisions:
    - `pr`: `open` when there are edits and no open PR on the shard's branch; `blocked` when a PR
      is already open there (then nothing is published or recorded, and the files are offered
      again once that PR is resolved); otherwise `none`.
    - `issue`: `comment` on the shard's open issue or `create` one when there are new unfixed
      findings; otherwise `none`.
    """
    known: set[str] = set()
    shard_issue: int | None = None
    branch_open = shard["branch"] in github.busy_branches()
    for item in github.items:
        if not github.is_open(item):
            continue
        for marker in github.markers(item):
            if marker["check"] != check.id:
                continue
            known.update(marker.get("fingerprints", {}))
            if not is_pr(item) and marker.get("shard") == shard["id"] and shard_issue is None:
                shard_issue = int(item["number"])

    edits = list(final["edits"])
    notes = list(final["notes"])
    fixed = [f for f in final["findings"] if f["fixed"]]
    unfixed = [f for f in final["findings"] if not f["fixed"] and f["fingerprint"] not in known]
    if duplicates := len(final["findings"]) - len(fixed) - len(unfixed):
        notes.append(f"left out {duplicates} findings already reported in an open agent PR or issue")

    if edits:
        pr = "blocked" if branch_open else "open"
    else:
        pr = "none"
    if pr == "blocked" or not unfixed:
        issue = "none"
    else:
        issue = "comment" if shard_issue is not None else "create"
    summary, run_url = final.get("agent_summary", ""), final.get("run_url")
    return {
        "pr": pr,
        "issue": issue,
        "issue_number": shard_issue,
        "findings": fixed + unfixed,
        "pr_body": render_body(check, plan, shard, summary, fixed, edits, notes, run_url, "pr", unfixed=len(unfixed)),
        "issue_body": render_body(check, plan, shard, summary, unfixed, [], notes, run_url, "issue"),
    }


def apply_patch(root: Path, patch: Path, check: Check, shard: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Apply the agent's patch to a fresh checkout, but only for paths the shard may edit.

    The patch comes from the agent's job, so it is untrusted: anything outside the shard's scope,
    `Cargo.lock`, `.git` paths, symlinks and submodules are left out. Git hooks are disabled for the
    apply. Returns (applied, rejected) paths.
    """
    if not patch.exists() or patch.stat().st_size == 0:
        return [], []
    try:
        numstat = git(root, "apply", "--numstat", "-z", str(patch))
        summary = git(root, "apply", "--summary", str(patch))
    except subprocess.CalledProcessError as e:
        return [], [f"<unreadable patch: {e.stderr.strip()[:200]}>"]
    paths: list[str] = []
    fields = numstat.split("\0")
    i = 0
    while i < len(fields):
        entry = fields[i]
        if not entry:
            i += 1
            continue
        parts = entry.split("\t")
        if len(parts) == 3 and parts[2] == "" and i + 2 < len(fields):  # a rename: old and new path follow
            paths += [fields[i + 1], fields[i + 2]]
            i += 3
        else:
            paths.append(parts[-1])
            i += 1
    special = {line.split()[-1] for line in summary.splitlines() if " 120000 " in line or " 160000 " in line}
    scope = [str(shard["crate_dir"])] if check.edit_scope == "crate" else [str(f["path"]) for f in shard["files"]]
    allowed: list[str] = []
    rejected: list[str] = []
    for path in dict.fromkeys(paths):
        bad = path in special or path.endswith("Cargo.lock") or ".git" in Path(path).parts
        (rejected if bad or not _within(path, scope) else allowed).append(path)
    if allowed:
        includes = [f"--include={glob.escape(p)}" for p in allowed]
        try:
            git(root, "-c", "core.hooksPath=/dev/null", "apply", "--whitespace=nowarn", *includes, str(patch))
        except subprocess.CalledProcessError as e:
            return [], allowed + rejected + [f"<patch did not apply: {e.stderr.strip()[:200]}>"]
    return allowed, rejected


def _safe(text: str) -> str:
    """Neutralize HTML comments in agent-written text so it cannot forge the runner's marker."""
    return str(text).replace("<!--", "&lt;!--").replace("-->", "--&gt;")


MAX_TITLE, MAX_DETAIL, MAX_SUMMARY = 300, 4_000, 20_000
# GitHub rejects PR and issue bodies over 65,536 characters.
MAX_BODY = 60_000


def _clip(text: str, limit: int) -> str:
    text = str(text)
    return text if len(text) <= limit else text[: limit - 20].rstrip() + " … (truncated)"


def format_edits(format_cmd: Sequence[str], edits: list[str], root: Path) -> str | None:
    """Format the edited Rust files; return a description of the failure, if any."""
    files = [p for p in edits if p.endswith(".rs") and (root / p).exists()]
    if not files:
        return None
    try:
        result = subprocess.run([*format_cmd, *files], cwd=root, check=False)
    except OSError as e:
        return f"formatting with `{' '.join(format_cmd)}` could not start: {e}"
    if result.returncode != 0:
        return f"formatting with `{' '.join(format_cmd)}` failed with exit code {result.returncode}"
    return None


def verify_commands(check: Check, shard: dict[str, Any]) -> list[str]:
    # Plain substitution, not str.format, so commands may contain other braces.
    return [template.replace("{crate}", shard["crate"]) for template in check.verify]


def verify_edits(check: Check, shard: dict[str, Any], root: Path) -> str | None:
    """Run the check's `verify` commands; return a description of the first failure, if any."""
    for command in verify_commands(check, shard):
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
    kind: str,
    *,
    unfixed: int = 0,
) -> str:
    """The body of the shard's PR (`kind="pr"`), issue post (`"issue"`), or job summary (`"summary"`).

    Everything agent-written is neutralized and clipped, and the runner's marker comes last.
    """
    out = [
        f"Automated `{check.id}` review of `{shard['id']}` at {plan['base_commit'][:12]}.\n\n",
        _safe(summary.strip()) + "\n\n" if kind != "issue" and summary.strip() else "",
    ]

    def bullet(f: dict[str, Any]) -> str:
        loc = f"{f['file']}:{f['line']}" if f.get("line") else f["file"]
        detail = _safe(f["detail"].strip())
        return f"- **{f['severity']}** `{_safe(loc)}` ({_safe(f['category'])}): {_safe(f['title'])}\n  {detail}\n"

    fixed = [f for f in findings if f["fixed"]]
    open_ = [f for f in findings if not f["fixed"]]
    if fixed:
        out.append("### Fixed in this PR\n\n" if kind == "pr" else "### Fixed\n\n")
        out.extend(bullet(f) for f in fixed)
        out.append("\n")
    if open_:
        out.append("### Needs a maintainer decision\n\n")
        out.extend(bullet(f) for f in open_)
        out.append("\n")
    if unfixed:
        out.append(f"{unfixed} findings that this PR does not fix are on the `{shard['id']}` issue.\n\n")
    if not findings and not unfixed:
        out.append("No findings.\n\n")
    if notes:
        out.append("<details><summary>Runner notes</summary>\n\n")
        out.extend(f"- {_safe(_clip(n, 1_000))}\n" for n in notes)
        out.append("\n</details>\n\n")
    if kind == "pr":
        out.append("Closing this PR without merging suppresses the findings it lists in future runs. ")
    elif kind == "issue":
        out.append("Closing this issue as not planned suppresses the findings it lists in future runs. ")
    out.append(f"The rubric is `.agents/checks/{check.id}/CHECK.md`.\n")
    if run_url:
        out.append(f"\n[Agent run]({run_url})\n")
    text = _clip("".join(out), MAX_BODY)
    marker = {
        "check": check.id,
        "shard": shard["id"],
        "files": sorted(f["path"] for f in shard["files"]),
        "claims": sorted({*shard["edit_scope"], *edits}) if kind != "issue" else [],
        "fingerprints": {f["fingerprint"]: f["file"] for f in findings},
    }
    return text + f"\n<!-- agent-check: {json.dumps(marker, sort_keys=True)} -->\n"


# --------------------------------------------------------------------------------------------------
# Record
# --------------------------------------------------------------------------------------------------


def record(
    store: CheckStore,
    updates: list[dict[str, Any]],
    *,
    plan: dict[str, Any] | None = None,
    github: GitHubView | None = None,
) -> list[str]:
    """Merge state updates and suppressions into the store. Returns the touched check ids.

    Each check's document is updated with compare-and-swap, so concurrent writers never lose each
    other's updates. Checks are independent: if one check's state cannot be read or written, the
    others are still recorded, and a CheckError naming the failures is raised at the end.

    A check the plan marked `stamp` gets `last_run` set to the plan's start time if at least one
    of its shards was recorded, or if it had nothing to review. When every shard failed, the clock
    is left alone so the next scheduled run tries again.

    Each rejected PR or issue is turned into suppressions once and then remembered in
    `suppression_refs`, so a suppression a maintainer deletes stays deleted.

    Merging is idempotent, and a file keeps whichever review is newer, so retrying a write or
    re-running an old run never rolls the state back.
    """
    rejections = (github or GitHubView([])).rejections()
    plan_checks = (plan or {}).get("checks", {})
    updated: set[str] = {str(u["check"]) for u in updates}
    stamped = {cid for cid, c in plan_checks.items() if c.get("stamp") and (cid in updated or not c.get("shards"))}
    touched: list[str] = sorted(updated | stamped | set(rejections))
    failures: list[str] = []
    for check_id in touched:
        if not CHECK_ID_RE.fullmatch(check_id):
            failures.append(f"invalid check id {check_id!r} in a state update")
            continue

        def change(state: dict[str, Any], check_id: str = check_id) -> None:
            for update in updates:
                if update["check"] != check_id:
                    continue
                for path, entry in update["files"].items():
                    current = state["files"].get(path)
                    if current is None or str(entry["reviewed_at"]) >= str(current.get("reviewed_at", "")):
                        state["files"][path] = entry
            rejected = rejections.get(check_id, {})
            state["suppressed"].update(new_suppressions(state, rejected))
            refs = set(state.get("suppression_refs", [])) | set(rejected)
            if refs:
                state["suppression_refs"] = sorted(refs)
            if check_id in stamped:
                stamp = str((plan or {})["generated_at"])
                if not state.get("last_run") or stamp > state["last_run"]:
                    state["last_run"] = stamp

        try:
            update_state(store, check_id, change)
        except CheckError as e:
            failures.append(f"{check_id}: {e}")
    if failures:
        raise CheckError("could not record: " + "; ".join(failures))
    return touched


# --------------------------------------------------------------------------------------------------
# Run archive
# --------------------------------------------------------------------------------------------------

# What a shard keeps of each run, under `<check>/runs/<run>/<shard>/`. Anything else in a handed-over
# directory is ignored.
ARCHIVED_FILES = (
    "results.json",
    "patch.diff",
    "outcome",
    "summary.json",
    "decision.json",
    "body.md",
    "pr.md",
    "issue.md",
)
MAX_ARCHIVED_BYTES = 5 * 1024 * 1024


def run_key(plan: dict[str, Any], run_id: str) -> str:
    """`<plan time>_<run id>`, so a check's runs list in time order."""
    stamp = dt.datetime.fromisoformat(plan["generated_at"]).astimezone(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    key = f"{stamp}_{run_id}"
    if not _KEY_RE.fullmatch(key):
        raise CheckError(f"invalid run id {run_id!r}")
    return key


def archive(store: CheckStore, plan: dict[str, Any], run_id: str, dirs: Sequence[Path]) -> list[str]:
    """Keep a write-once record of the run under each check's `runs/<run>/` prefix.

    For every planned check this stores the check's part of the plan, and for every handed-over
    directory (one per shard and job, tagged by its `meta.json`) the files in ARCHIVED_FILES.
    Objects are created with `put_new`, so a re-run never overwrites history; existing objects are
    skipped. Directories naming a check or shard that is not in the plan are ignored. Returns the
    keys written.
    """
    key = run_key(plan, run_id)
    written: list[str] = []
    for check_id, entry in plan["checks"].items():
        if not entry.get("shards"):
            continue
        snapshot = {
            "base_commit": plan["base_commit"],
            "generated_at": plan["generated_at"],
            "version": entry["definition"]["version"],
            "shards": entry["shards"],
        }
        if store.put_new(check_id, f"runs/{key}/plan.json", (json.dumps(snapshot, indent=2) + "\n").encode()):
            written.append(f"{check_id}/runs/{key}/plan.json")
    for directory in dirs:
        try:
            meta = json.loads((directory / "meta.json").read_text())
            check_id, shard_id = str(meta["check"]), str(meta["shard"])
        except (OSError, ValueError, KeyError, TypeError):
            print(f"archive: skipping {directory}: no valid meta.json", file=sys.stderr)
            continue
        shards = {sh["id"] for sh in plan["checks"].get(check_id, {}).get("shards", [])}
        if shard_id not in shards:
            print(f"archive: skipping {directory}: {check_id}/{shard_id} is not in the plan", file=sys.stderr)
            continue
        for name in ARCHIVED_FILES:
            path = directory / name
            if not path.is_file() or path.is_symlink():
                continue
            if path.stat().st_size > MAX_ARCHIVED_BYTES:
                print(f"archive: skipping {path}: larger than {MAX_ARCHIVED_BYTES} bytes", file=sys.stderr)
                continue
            if store.put_new(check_id, f"runs/{key}/{shard_slug(shard_id)}/{name}", path.read_bytes()):
                written.append(f"{check_id}/runs/{key}/{shard_slug(shard_id)}/{name}")
    return written


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


def _github_view(args: argparse.Namespace) -> GitHubView:
    trusted = getattr(args, "trusted_author", None)
    return GitHubView(_read_items(getattr(args, "github", None)), trusted or None)


def _store(args: argparse.Namespace) -> CheckStore:
    return open_store(args.state if args.state is not None else Path(args.root) / DEFAULT_LOCAL_STATE)


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
        _store(args),
        now=_now(args),
        github=_github_view(args),
        only_shards=args.shard,
        max_shards=args.max_shards,
        force=args.force,
        scheduled=args.scheduled,
    )
    text = json.dumps(plan, indent=2) + "\n"
    if args.out:
        Path(args.out).write_text(text)
    else:
        sys.stdout.write(text)
    m = matrix(plan)
    for entry in m["include"]:
        print(f"planned {entry['check']} / {entry['shard']}", file=sys.stderr)
    for cid, c in plan["checks"].items():
        if "error" in c:
            # The other checks still run; the error shows on the run's summary page.
            print(f"::error::{cid} was not planned: {c['error']}", file=sys.stderr)
    _github_output(
        {
            "matrix": json.dumps(m),
            "count": str(len(m["include"])),
            "checks": json.dumps(list(plan["checks"])),
            "check_matrix": json.dumps(check_matrix(plan)),
        }
    )
    return 0


def cmd_prompt(args: argparse.Namespace) -> int:
    plan = _read_dict(args.plan)
    check = check_from_plan(plan, args.check, rerun_at=_now(args) if args.rerun else None)
    shard = find_shard(plan, args.check, args.shard)
    prompt = render_prompt(check, plan, shard, args.results)
    # The agent may run exactly the verify commands it is told the runner will run.
    tools = list(dict.fromkeys([*check.tools, *(f"Bash({c})" for c in verify_commands(check, shard))]))
    if args.out:
        Path(args.out).write_text(prompt)
    else:
        sys.stdout.write(prompt)
    _github_output(
        {
            "allowed_tools": ",".join(tools),
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
    check = check_from_plan(plan, args.check)
    shard = find_shard(plan, args.check, args.shard)
    if not Path(args.results).exists():
        raise CheckError(
            f"the agent wrote no results to {args.results}, so nothing is recorded and the shard is retried "
            "on the next run; the agent step's log says why"
        )
    results = validate_results(_read_json(args.results))
    if args.notes and Path(args.notes).exists():
        results["invalid"] = [*results["invalid"], *map(str, json.loads(Path(args.notes).read_text()))]
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
        format_cmd=shlex.split(args.format_cmd) if args.format_cmd else (),
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


def cmd_decide(args: argparse.Namespace) -> int:
    plan = _read_dict(args.plan)
    check = check_from_plan(plan, args.check)
    shard = find_shard(plan, args.check, args.shard)
    out_dir = Path(args.out_dir)
    final = _read_dict(str(out_dir / "summary.json"))
    decision = decide(check, plan, shard, final, _github_view(args))
    (out_dir / "pr.md").write_text(decision["pr_body"])
    (out_dir / "issue.md").write_text(decision["issue_body"])
    summary = {k: decision[k] for k in ("pr", "issue", "issue_number")}
    summary["fingerprints"] = sorted(f["fingerprint"] for f in decision["findings"])
    (out_dir / "decision.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        f"{check.id}/{shard['id']}: pr={decision['pr']} issue={decision['issue']} "
        f"({len(decision['findings'])} of {len(final['findings'])} findings are new or fixed)",
        file=sys.stderr,
    )
    _github_output(
        {"pr": decision["pr"], "issue": decision["issue"], "issue_number": str(decision["issue_number"] or "")}
    )
    return 0


def cmd_apply(args: argparse.Namespace) -> int:
    plan = _read_dict(args.plan)
    shard = find_shard(plan, args.check, args.shard)
    check = check_from_plan(plan, args.check)
    applied, rejected = apply_patch(Path(args.root), Path(args.patch), check, shard)
    print(f"applied {applied}; rejected {rejected}", file=sys.stderr)
    if rejected:
        Path(args.notes).write_text(json.dumps([f"patch paths outside the shard were not applied: {rejected}"]))
    return 0


def cmd_archive(args: argparse.Namespace) -> int:
    written = archive(_store(args), _read_dict(args.plan), args.run_id, [Path(d) for d in args.dirs])
    print(f"archived {len(written)} objects", file=sys.stderr)
    return 0


def cmd_record(args: argparse.Namespace) -> int:
    updates = [_read_dict(p) for p in args.updates]
    plan = _read_dict(args.plan) if args.plan else None
    touched = record(_store(args), updates, plan=plan, github=_github_view(args))
    print(f"updated state for {touched}", file=sys.stderr)
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    """Run one check locally against the working tree, using the `claude` CLI."""
    root = Path(args.root)
    check = load_checks(root)[args.check]
    now = _now(args)
    work = root / "target" / "agent-checks" / check.id
    work.mkdir(parents=True, exist_ok=True)
    store = _store(args)
    plan = build_plan([check], root, store, now=now, only_shards=args.shard, max_shards=args.max_shards, force=True)
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
        handed = work / shard_slug(shard["id"])
        handed.mkdir(exist_ok=True)
        (handed / "meta.json").write_text(json.dumps({"check": check.id, "shard": shard["id"]}))
        (handed / "body.md").write_text(out["body"])
        shutil.copy(results_path, handed / "results.json")
        archive(store, plan, f"local-{os.getpid()}", [handed])
        print(f"wrote {body_path}; edits left in the working tree: {out['edits']}", file=sys.stderr)
        updates.append(out["state_update"])
        if len(shards) > 1 and working_tree_changes(root):
            print("stopping after the first shard with edits; review and commit them, then rerun", file=sys.stderr)
            break
    record(store, updates, plan=plan)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=REPO_ROOT, help="repository root")
    parser.add_argument("--now", help="override the current time (ISO 8601), for tests")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="list checks").set_defaults(func=cmd_list)

    p = sub.add_parser("plan", help="choose shards to review")
    p.add_argument("--check", action="append", help="check id (repeatable; default: all enabled)")
    p.add_argument("--state", help="s3://bucket/prefix or a directory (default: <root>/target/agent-checks-state)")
    p.add_argument(
        "--github",
        help="JSON list of `gh pr list` / `gh issue list` items (number, state, body; PRs: headRefName, "
        "mergedAt; issues: stateReason, comments)",
    )
    p.add_argument("--shard", action="append", help="plan only this crate (repeatable)")
    p.add_argument("--scheduled", action="store_true", help="a scheduled run: due checks move their clock")
    p.add_argument(
        "--trusted-author",
        action="append",
        help="login whose markers count (repeatable; default: every author)",
    )
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
    p.add_argument("--rerun", action="store_true", help="a re-run attempt: refuse plans that are too old")
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
    p.add_argument("--format-cmd", help="formatter run on edited .rs files, e.g. `rustfmt --edition 2024`")
    p.add_argument("--notes", help="JSON list of extra notes for the body, such as from `apply`")
    p.set_defaults(func=cmd_finalize)

    p = sub.add_parser("decide", help="decide from live GitHub state what a finalized shard publishes")
    p.add_argument("--plan", required=True)
    p.add_argument("--check", required=True)
    p.add_argument("--shard", required=True)
    p.add_argument("--out-dir", required=True, help="finalize's output directory; publish.md is written here")
    p.add_argument("--github", required=True, help="the open PRs and issues, as `gh ... list --json` items")
    p.add_argument(
        "--trusted-author",
        action="append",
        help="login whose markers count (repeatable; default: every author)",
    )
    p.set_defaults(func=cmd_decide)

    p = sub.add_parser("apply", help="apply an agent's patch, limited to the shard's scope")
    p.add_argument("--plan", required=True)
    p.add_argument("--check", required=True)
    p.add_argument("--shard", required=True)
    p.add_argument("--patch", required=True)
    p.add_argument("--notes", required=True, help="where to write notes about rejected paths")
    p.set_defaults(func=cmd_apply)

    p = sub.add_parser("record", help="merge state updates and suppressions into the state store")
    p.add_argument("--state", help="s3://bucket/prefix or a directory (default: <root>/target/agent-checks-state)")
    p.add_argument("--plan", help="the run's plan; its checks marked `stamp` get `last_run` set")
    p.add_argument(
        "--github",
        help="JSON list of `gh pr list` / `gh issue list` items (number, state, body; PRs: headRefName, "
        "mergedAt; issues: stateReason, comments)",
    )
    p.add_argument("updates", nargs="*")
    p.add_argument(
        "--trusted-author",
        action="append",
        help="login whose markers count (repeatable; default: every author)",
    )
    p.set_defaults(func=cmd_record)

    p = sub.add_parser("archive", help="keep a write-once record of a run in each check's storage")
    p.add_argument("--state", help="s3://bucket/prefix or a directory (default: <root>/target/agent-checks-state)")
    p.add_argument("--plan", required=True)
    p.add_argument("--run-id", required=True, help="unique per run attempt, such as `<run id>-<attempt>`")
    p.add_argument("dirs", nargs="*", help="handed-over directories, each with a meta.json")
    p.set_defaults(func=cmd_archive)

    p = sub.add_parser("run", help="run a check locally with the `claude` CLI")
    p.add_argument("check")
    p.add_argument("--shard", action="append", help="review only this crate (repeatable)")
    p.add_argument("--max-shards", type=int, default=1)
    p.add_argument("--state", help="s3://bucket/prefix or a directory (default: <root>/target/agent-checks-state)")
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
