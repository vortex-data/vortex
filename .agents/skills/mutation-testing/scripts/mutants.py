#!/usr/bin/env python3
"""Drive cargo-mutants over one scope for the mutation-testing skill.

Subcommands, in the order the skill uses them:

  run        Test every mutant in a scope and record the survivors with source excerpts.
  survivors  Print a run's surviving mutants, grouped by function and ranked.
  probe      Check whether any test reaches a source line, by inserting a temporary panic.
  verify     Re-test a run's survivors against the current tree.
  kills      Show which tests catch which mutants: tests that catch none, duplicates, a minimal set.
  compare    Diff two runs of one scope, to prove a test change lost no caught mutant.
  report     Check the triage, removals and questions against the evidence, and write the report.

Every command that builds runs cargo in place in the working tree. Do not edit files while one is
running. Each checks that the tree it leaves matches the tree it started from.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import time
import tomllib
from pathlib import Path
from typing import Any

# Keep in step with CARGO_MUTANTS_VERSION in .github/workflows/mutants.yml.
CARGO_MUTANTS_VERSION = "27.1.0"
DEFAULT_MAX_MUTANTS = 500
SCHEMA_VERSION = 1

# cargo-mutants exit codes: 0 all caught, 2 some missed, 3 some timed out, 4 baseline failed.
EXIT_REPORTABLE = {0, 2, 3}
EXIT_BASELINE_FAILED = 4

DETECTED = {"CaughtMutant", "Timeout"}
STATUS_BY_SUMMARY = {
    "CaughtMutant": "caught",
    "MissedMutant": "missed",
    "Timeout": "timeout",
    "Unviable": "unviable",
}
VERDICTS = ("gap-fixed", "equivalent", "bug", "deferred")
PROBE_MARKER = "MUTANTS_PROBE_REACHED"
EXCERPT_MAX_LINES = 60
EXCERPT_CONTEXT = 3


class ToolError(Exception):
    """A failure to report to the agent without a traceback."""


# --- git and environment ---------------------------------------------------------------------


def git(root: Path, *args: str) -> str:
    result = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise ToolError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout


def repo_root() -> Path:
    out = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True, check=False)
    if out.returncode != 0:
        raise ToolError("run this from inside the Vortex git repository")
    return Path(out.stdout.strip())


def tree_fingerprint(root: Path) -> str:
    """A hash of every tracked change and untracked file, to prove a run restored the tree."""
    digest = hashlib.sha256()
    digest.update(git(root, "rev-parse", "HEAD").encode())
    digest.update(git(root, "diff", "HEAD", "--binary").encode())
    untracked = git(root, "ls-files", "--others", "--exclude-standard", "-z").split("\0")
    for name in sorted(filter(None, untracked)):
        digest.update(name.encode())
        digest.update(hashlib.sha256((root / name).read_bytes()).digest())
    return digest.hexdigest()


def cargo_env() -> dict[str, str]:
    """Build without sccache, which disables incremental compilation and slows every mutant."""
    env = os.environ.copy()
    env["RUSTC_WRAPPER"] = ""
    env["CARGO_INCREMENTAL"] = "1"
    return env


def cargo_mutants_version(root: Path) -> str:
    result = subprocess.run(
        ["cargo", "mutants", "--version"], cwd=root, capture_output=True, text=True, check=False, env=cargo_env()
    )
    if result.returncode != 0:
        raise ToolError(f"cargo-mutants is not installed: cargo install --locked cargo-mutants@{CARGO_MUTANTS_VERSION}")
    version = result.stdout.split()[-1] if result.stdout.split() else "unknown"
    if version != CARGO_MUTANTS_VERSION:
        print(
            f"warning: cargo-mutants {version} is installed; CI pins {CARGO_MUTANTS_VERSION}",
            file=sys.stderr,
        )
    return version


def prepare_out_dir(root: Path, out: Path) -> Path:
    """A fresh output directory that git ignores, so writing to it does not change the tree."""
    out = out if out.is_absolute() else Path.cwd() / out
    if out.exists() and any(out.iterdir()):
        raise ToolError(f"{out} is not empty; give each run its own output directory")
    try:
        relative = out.resolve().relative_to(root.resolve())
    except ValueError:
        relative = None
    if relative is not None:
        ignored = subprocess.run(["git", "check-ignore", "-q", str(relative)], cwd=root, check=False).returncode == 0
        if not ignored:
            raise ToolError(f"{out} is inside the repository but not ignored by git; use a path under target/")
    out.mkdir(parents=True, exist_ok=True)
    return out


# --- cargo-mutants ---------------------------------------------------------------------------


def selection_args(args: argparse.Namespace) -> list[str]:
    selection: list[str] = ["--package", str(args.package)]
    for glob in args.file or []:
        selection += ["--file", glob]
    for regex in args.re or []:
        selection += ["--re", regex]
    if args.in_diff:
        selection += ["--in-diff", str(Path(args.in_diff).resolve())]
    return selection + [str(arg) for arg in args.cargo_mutants_args or []]


def list_mutants(root: Path, selection: list[str]) -> list[dict[str, Any]]:
    result = subprocess.run(
        ["cargo", "mutants", "--list", "--json", *selection],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
        env=cargo_env(),
    )
    if result.returncode != 0:
        hint = ""
        if "failed to parse" in result.stderr:
            hint = (
                "\nhint: cargo-mutants could not parse a module. Mark its `mod` item with "
                "#[cfg_attr(mutants, mutants::skip)]; see the skill's Troubleshooting section."
            )
        raise ToolError(f"cargo mutants --list failed:\n{result.stderr.strip()[-2000:]}{hint}")
    return json.loads(result.stdout or "[]")


def run_cargo_mutants(root: Path, selection: list[str], out: Path) -> int:
    """Run cargo-mutants in place, streaming its output to the terminal and to a log file."""
    command = ["cargo", "mutants", *selection, "--in-place", "--caught", "--unviable", "--colors", "never"]
    # `--no-fail-fast` runs every test binary even after one fails, so each mutant's log names
    # every test that catches it, which the kill matrix needs.
    command += ["--output", str(out), "--", "--no-fail-fast"]
    print("$ " + shlex.join(command), flush=True)
    before = tree_fingerprint(root)
    with open(out / "cargo-mutants.log", "w") as log:
        process = subprocess.Popen(
            command, cwd=root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=cargo_env()
        )
        assert process.stdout is not None
        # Keep draining cargo-mutants even if our stdout goes away: exiting early would leave it
        # mutating the tree with nobody to wait for the restore.
        echo = True
        try:
            for line in process.stdout:
                log.write(line)
                if echo:
                    try:
                        sys.stdout.write(line)
                        sys.stdout.flush()
                    except BrokenPipeError:
                        echo = False
            status = process.wait()
        except KeyboardInterrupt:
            # cargo-mutants restores the mutated file when it is interrupted; wait for it.
            process.wait()
            raise
    if tree_fingerprint(root) != before:
        raise ToolError(
            "the working tree changed during the run: cargo-mutants did not restore a file, or something "
            "edited the tree. Inspect `git status` and `git diff` before doing anything else."
        )
    return status


def load_outcomes(mutants_out: Path) -> list[dict[str, Any]]:
    path = mutants_out / "outcomes.json"
    if not path.exists():
        raise ToolError(f"{path} does not exist; cargo-mutants did not finish")
    outcomes = json.loads(path.read_text())["outcomes"]
    return [o for o in outcomes if isinstance(o.get("scenario"), dict) and "Mutant" in o["scenario"]]


def describe(mutant: dict[str, Any]) -> str:
    """The mutant's name without its `path:line:col: ` prefix."""
    prefix = f"{mutant['file']}:{mutant['span']['start']['line']}:{mutant['span']['start']['column']}: "
    name = str(mutant["name"])
    return name[len(prefix) :] if name.startswith(prefix) else name


def rust_regex_escape(text: str) -> str:
    """Escape text for both Rust's regex crate and Python's re, which escape different sets."""
    return re.sub(r"([\\.+*?()|\[\]{}^$])", r"\\\1", text)


def name_pattern(mutant: dict[str, Any]) -> str:
    """A regex for this mutant's name that tolerates its line and column moving."""
    return f"^{rust_regex_escape(mutant['file'])}:\\d+:\\d+: {rust_regex_escape(describe(mutant))}$"


def exclusion_patterns(root: Path) -> list[str]:
    config = root / ".cargo" / "mutants.toml"
    if not config.exists():
        return []
    return list(tomllib.loads(config.read_text()).get("exclude_re", []))


def excluded_by(name: str, patterns: list[str]) -> str | None:
    for pattern in patterns:
        if re.search(pattern, name):
            return pattern
    return None


# --- survivors -------------------------------------------------------------------------------


def excerpt(source_lines: list[str], start: int, end: int, marked: set[int]) -> list[str]:
    """Numbered source lines for a function, or for the mutated lines when the function is long."""
    if end - start + 1 <= EXCERPT_MAX_LINES:
        wanted = range(start, end + 1)
    else:
        lines: set[int] = set()
        for line in marked:
            lines.update(range(max(start, line - EXCERPT_CONTEXT), min(end, line + EXCERPT_CONTEXT) + 1))
        wanted = sorted(lines)
    rendered = []
    previous = None
    for line in wanted:
        if line < 1 or line > len(source_lines):
            continue
        if previous is not None and line != previous + 1:
            rendered.append("     ...")
        marker = ">" if line in marked else " "
        rendered.append(f"{marker}{line:5} | {source_lines[line - 1]}")
        previous = line
    return rendered


def collect_survivors(root: Path, outcomes: list[dict[str, Any]]) -> dict[str, Any]:
    """Group missed mutants by function, rank the groups, and attach source excerpts."""
    groups: dict[tuple[str, str], dict[str, Any]] = {}
    timeouts = []
    for outcome in outcomes:
        mutant = outcome["scenario"]["Mutant"]
        if outcome["summary"] == "Timeout":
            timeouts.append(mutant["name"])
        if outcome["summary"] != "MissedMutant":
            continue
        function = mutant.get("function") or {}
        function_name = function.get("function_name", "<module>")
        key = (mutant["file"], function_name)
        group = groups.setdefault(
            key,
            {
                "file": mutant["file"],
                "function": function_name,
                "span": [
                    function.get("span", mutant["span"])["start"]["line"],
                    function.get("span", mutant["span"])["end"]["line"],
                ],
                "body_replaceable": False,
                "mutants": [],
            },
        )
        group["body_replaceable"] |= mutant.get("genre") == "FnValue"
        group["mutants"].append(
            {
                "name": mutant["name"],
                "line": mutant["span"]["start"]["line"],
                "column": mutant["span"]["start"]["column"],
                "description": describe(mutant),
                "genre": mutant.get("genre"),
                "pattern": name_pattern(mutant),
            }
        )

    for group in groups.values():
        source = (root / group["file"]).read_text().splitlines()
        marked = {m["line"] for m in group["mutants"]}
        group["excerpt"] = excerpt(source, group["span"][0], group["span"][1], marked)
        group["mutants"].sort(key=lambda m: (m["line"], m["column"], m["description"]))

    # A function whose whole body can be replaced is unreached or its result is unchecked: the
    # strongest signal. Larger clusters come next.
    ranked = sorted(groups.values(), key=lambda g: (not g["body_replaceable"], -len(g["mutants"]), g["file"]))
    return {"groups": ranked, "timeouts": sorted(timeouts)}


# --- kill matrix -----------------------------------------------------------------------------

# libtest prints `test <name> ... ok`; doctest names contain spaces, so match lazily.
TEST_LINE = re.compile(r"^test (.+?) \.\.\. (ok|FAILED|ignored)\b")
# cargo announces each test binary as `Running unittests src/lib.rs (target/...)`, or, with
# `--verbose` as cargo-mutants runs it, as `Running \`<target dir>/<name>-<hash>\``. Doctests follow
# `Doc-tests <crate>`. Build commands also start with `Running \``, but contain spaces.
BINARY_LINE = re.compile(r"^\s+Running (?:unittests )?([^`\s]\S*) \(")
VERBOSE_BINARY_LINE = re.compile(r"^\s+Running `([^`\s]+)`$")
DOCTEST_LINE = re.compile(r"^\s+Doc-tests (\S+)")
RSTEST_CASE = re.compile(r"^(case_\d+|[A-Za-z][A-Za-z0-9_]*_\d+(_|$))")


def test_results(log: str, lib_binary: str) -> dict[str, str]:
    """Each test's result in a cargo test log, keyed by test id: the name, prefixed by its binary
    unless it is a unit test of the library, whose binary is called `lib_binary`."""
    binary = "src/lib.rs"
    results: dict[str, str] = {}
    for line in log.splitlines():
        if match := BINARY_LINE.match(line):
            binary = match.group(1)
        elif match := VERBOSE_BINARY_LINE.match(line):
            name = re.sub(r"-[0-9a-f]{16}$", "", Path(match.group(1)).name)
            binary = "src/lib.rs" if name == lib_binary else f"tests/{name}"
        elif match := DOCTEST_LINE.match(line):
            binary = "doctest"
        elif match := TEST_LINE.match(line):
            name = match.group(1)
            results[name if binary == "src/lib.rs" else f"[{binary}] {name}"] = match.group(2)
    return results


def test_function(test_id: str) -> str:
    """The test function an rstest case belongs to, or the test itself."""
    parts = test_id.split("::")
    for index, part in enumerate(parts[1:], 1):
        if RSTEST_CASE.match(part):
            return "::".join(parts[:index])
    return test_id


def build_kills(mutants_out: Path, outcomes: list[dict[str, Any]]) -> dict[str, Any]:
    """Which tests fail under each caught mutant, from cargo-mutants' per-mutant logs."""
    packages = {o["scenario"]["Mutant"]["package"] for o in outcomes}
    lib_binary = packages.pop().replace("-", "_") if len(packages) == 1 else ""
    baseline_log = mutants_out / "log" / "baseline.log"
    baseline = test_results(baseline_log.read_text(), lib_binary) if baseline_log.exists() else {}
    kills: dict[str, list[str]] = {}
    unattributed = []
    for outcome in outcomes:
        if outcome["summary"] != "CaughtMutant":
            continue
        name = outcome["scenario"]["Mutant"]["name"]
        log = (mutants_out / outcome["log_path"]).read_text()
        failed = [t for t, r in test_results(log, lib_binary).items() if r == "FAILED"]
        if not failed:
            # Caught by a crash, an abort or a build of the tests, not by a named failing test.
            unattributed.append(name)
        for test in failed:
            kills.setdefault(test, []).append(name)
    return {
        "schema": SCHEMA_VERSION,
        "kind": "kills",
        "tests": sorted(t for t, r in baseline.items() if r == "ok"),
        "kills": {t: sorted(k) for t, k in sorted(kills.items())},
        "unattributed": sorted(unattributed),
    }


def merge_kills(matrices: list[dict[str, Any]]) -> dict[str, Any]:
    tests: set[str] = set()
    kills: dict[str, set[str]] = {}
    unattributed: set[str] = set()
    for matrix in matrices:
        tests |= set(matrix["tests"])
        unattributed |= set(matrix["unattributed"])
        for test, names in matrix["kills"].items():
            kills.setdefault(test, set()).update(names)
    return {
        "schema": SCHEMA_VERSION,
        "kind": "kills",
        "tests": sorted(tests),
        "kills": {t: sorted(k) for t, k in sorted(kills.items())},
        "unattributed": sorted(unattributed),
    }


def analyze_kills(matrix: dict[str, Any]) -> dict[str, Any]:
    """Tests that catch nothing, tests whose kills others cover, and a small set that catches all."""
    kills = {t: set(k) for t, k in matrix["kills"].items()}
    killed = set().union(*kills.values()) if kills else set()
    catchers: dict[str, set[str]] = {}
    for test, names in kills.items():
        for name in names:
            catchers.setdefault(name, set()).add(test)
    unique = {t: sorted(n for n in k if catchers[n] == {t}) for t, k in kills.items()}

    by_set: dict[frozenset[str], list[str]] = {}
    for test, names in kills.items():
        by_set.setdefault(frozenset(names), []).append(test)
    identical = [sorted(group) for group in by_set.values() if len(group) > 1]

    remaining = set(killed)
    cover = []
    while remaining:
        best = max(sorted(kills), key=lambda t: len(kills[t] & remaining))
        cover.append(best)
        remaining -= kills[best]

    return {
        "tests_run": len(matrix["tests"]),
        "killed_mutants": len(killed),
        "unattributed": len(matrix["unattributed"]),
        "no_kills": sorted(t for t in matrix["tests"] if t not in kills),
        "unique_kills": {t: n for t, n in sorted(unique.items()) if n},
        "identical": sorted(identical, key=lambda g: (-len(g), g)),
        "cover": cover,
    }


def counts(outcomes: list[dict[str, Any]]) -> dict[str, int]:
    tally = {status: 0 for status in STATUS_BY_SUMMARY.values()}
    for outcome in outcomes:
        status = STATUS_BY_SUMMARY.get(outcome["summary"])
        if status:
            tally[status] += 1
    return tally


def print_survivors(survivors: dict[str, Any]) -> None:
    groups = survivors["groups"]
    total = sum(len(g["mutants"]) for g in groups)
    print(f"{total} missed mutants in {len(groups)} functions, most suspicious first.\n")
    for index, group in enumerate(groups, 1):
        flag = "  [whole body replaceable: unreached or unchecked?]" if group["body_replaceable"] else ""
        print(f"## {index}. {group['file']} :: {group['function']}{flag}")
        print(f"   lines {group['span'][0]}-{group['span'][1]}, {len(group['mutants'])} missed")
        for mutant in group["mutants"]:
            print(f"   {mutant['line']}:{mutant['column']}  {mutant['description']}")
        print()
        for line in group["excerpt"]:
            print("   " + line)
        print()
    if survivors["timeouts"]:
        print(f"{len(survivors['timeouts'])} mutants timed out; they count as detected:")
        for name in survivors["timeouts"]:
            print(f"   {name}")


# --- subcommands -----------------------------------------------------------------------------


def cmd_run(args: argparse.Namespace) -> int:
    root = repo_root()
    version = cargo_mutants_version(root)
    out = prepare_out_dir(root, Path(args.out))
    selection = selection_args(args)

    listed = list_mutants(root, selection)
    print(f"{len(listed)} mutants in scope.", flush=True)
    if len(listed) > args.max_mutants:
        raise ToolError(
            f"the scope has {len(listed)} mutants, more than --max-mutants {args.max_mutants}. "
            "Narrow it with --file or --re, or ask the orchestrator for a smaller shard."
        )

    head = git(root, "rev-parse", "HEAD").strip()
    dirty = bool(git(root, "status", "--porcelain").strip())
    started = time.time()
    status = run_cargo_mutants(root, selection, out) if listed else 0
    finished = time.time()

    if listed and status == EXIT_BASELINE_FAILED:
        raise ToolError(
            "the unmutated tests fail (cargo-mutants exit 4), so no mutant can be judged. "
            f"See {out}/mutants.out/log/baseline.log. Report this; do not triage."
        )
    if listed and status not in EXIT_REPORTABLE:
        raise ToolError(f"cargo-mutants exited with status {status}; see {out}/cargo-mutants.log")

    outcomes = load_outcomes(out / "mutants.out") if listed else []
    survivors = collect_survivors(root, outcomes)
    run = {
        "schema": SCHEMA_VERSION,
        "kind": "run",
        "commit": head,
        "dirty": dirty,
        "package": args.package,
        "selection": selection,
        "cargo_mutants_version": version,
        "exit_status": status,
        "listed": len(listed),
        "counts": counts(outcomes),
        "seconds": round(finished - started, 1),
    }
    (out / "run.json").write_text(json.dumps(run, indent=2) + "\n")
    (out / "survivors.json").write_text(json.dumps(survivors, indent=2) + "\n")
    if listed:
        kills = build_kills(out / "mutants.out", outcomes)
        (out / "kills.json").write_text(json.dumps(kills, indent=2) + "\n")

    tally = run["counts"]
    print(
        f"\n{len(listed)} mutants in {run['seconds']}s: {tally['caught']} caught, {tally['missed']} missed, "
        f"{tally['timeout']} timed out, {tally['unviable']} unviable."
    )
    print(f"Survivors: python3 {sys.argv[0]} survivors {out}")
    return 0


def cmd_kills(args: argparse.Namespace) -> int:
    """Print the kill matrix of one run, or of several cargo-mutants outputs merged."""
    matrices = []
    for source in args.sources:
        path = Path(source)
        if (path / "kills.json").exists():
            matrices.append(json.loads((path / "kills.json").read_text()))
        else:
            # A bare `mutants.out` directory, such as a CI shard's artifact.
            mutants_out = path / "mutants.out" if (path / "mutants.out").is_dir() else path
            matrices.append(build_kills(mutants_out, load_outcomes(mutants_out)))
    matrix = merge_kills(matrices)
    if args.out:
        Path(args.out).write_text(json.dumps(matrix, indent=2) + "\n")
    analysis = analyze_kills(matrix)
    if args.json:
        print(json.dumps(analysis, indent=2))
        return 0

    print(
        f"{analysis['tests_run']} tests ran; {analysis['killed_mutants']} mutants were caught by a named test "
        f"and {analysis['unattributed']} only by a crash or abort.\n"
    )
    print(f"{len(analysis['cover'])} tests together catch every one of those mutants.\n")
    print(f"Tests that catch no mutant ({len(analysis['no_kills'])}): candidates to question, not to delete.")
    for test in analysis["no_kills"]:
        print(f"   {test}")
    print(f"\nGroups of tests that catch exactly the same mutants ({len(analysis['identical'])}):")
    for group in analysis["identical"][: args.limit]:
        functions = {test_function(t) for t in group}
        where = "cases of one test function" if len(functions) == 1 else f"{len(functions)} test functions"
        print(f"   {len(group)} tests, {where}, {len(matrix['kills'][group[0]])} mutants:")
        for test in group[:6]:
            print(f"      {test}")
        if len(group) > 6:
            print(f"      ... {len(group) - 6} more")
    if len(analysis["identical"]) > args.limit:
        print(f"   ... {len(analysis['identical']) - args.limit} more groups (--limit)")
    return 0


def compare_runs(before_dir: Path, after_dir: Path) -> dict[str, Any]:
    """Per-mutant outcome changes between two runs of one scope, matched by name pattern."""
    before_run = json.loads((before_dir / "run.json").read_text())
    after_run = json.loads((after_dir / "run.json").read_text())
    if before_run["selection"] != after_run["selection"]:
        raise ToolError("the two runs have different scopes; rerun `run` with the same arguments")

    def index(run_dir: Path) -> dict[str, list[tuple[int, str, str]]]:
        found: dict[str, list[tuple[int, str, str]]] = {}
        for outcome in load_outcomes(run_dir / "mutants.out"):
            mutant = outcome["scenario"]["Mutant"]
            status = STATUS_BY_SUMMARY.get(outcome["summary"], outcome["summary"])
            found.setdefault(name_pattern(mutant), []).append((mutant["span"]["start"]["line"], mutant["name"], status))
        return found

    before, after = index(before_dir), index(after_dir)
    changes: dict[str, list[dict[str, str]]] = {"regressed": [], "fixed": [], "vanished": [], "new": []}
    for pattern, entries in before.items():
        later = after.get(pattern, [])
        for line, name, status in entries:
            if not later:
                changes["vanished"].append({"name": name, "before": status})
                continue
            _, now_name, now = min(later, key=lambda e: abs(e[0] - line))
            if status in ("caught", "timeout") and now == "missed":
                changes["regressed"].append({"name": name, "now": now_name})
            elif status == "missed" and now in ("caught", "timeout"):
                changes["fixed"].append({"name": name, "now": now_name})
    for pattern, entries in after.items():
        if pattern not in before:
            changes["new"] += [{"name": name, "now": status} for _, name, status in entries]

    before_tests = set(json.loads((before_dir / "kills.json").read_text())["tests"])
    after_tests = set(json.loads((after_dir / "kills.json").read_text())["tests"])
    return {
        "schema": SCHEMA_VERSION,
        "kind": "compare",
        "before": {"commit": before_run["commit"], "counts": before_run["counts"]},
        "after": {"commit": after_run["commit"], "counts": after_run["counts"]},
        "changes": changes,
        "tests_removed": sorted(before_tests - after_tests),
        "tests_added": sorted(after_tests - before_tests),
    }


def cmd_compare(args: argparse.Namespace) -> int:
    before_dir, after_dir = Path(args.before), Path(args.after)
    comparison = compare_runs(before_dir, after_dir)
    (before_dir / "compare.json").write_text(json.dumps(comparison, indent=2) + "\n")
    changes = comparison["changes"]
    print(
        f"caught before {comparison['before']['counts']['caught']}, after {comparison['after']['counts']['caught']}; "
        f"missed before {comparison['before']['counts']['missed']}, after {comparison['after']['counts']['missed']}"
    )
    print(f"tests: {len(comparison['tests_removed'])} removed, {len(comparison['tests_added'])} added")
    for kind in ("regressed", "fixed", "new", "vanished"):
        print(f"{kind}: {len(changes[kind])}")
        for change in changes[kind] if kind == "regressed" else changes[kind][:5]:
            print(f"   {change['name']}")
    if changes["regressed"]:
        print("\nREGRESSED: tests that caught these mutants are gone. Restore them or replace their checks.")
        return 2
    return 0


def cmd_survivors(args: argparse.Namespace) -> int:
    survivors = json.loads((Path(args.run) / "survivors.json").read_text())
    if args.json:
        print(json.dumps(survivors, indent=2))
    else:
        print_survivors(survivors)
    return 0


def cmd_probe(args: argparse.Namespace) -> int:
    root = repo_root()
    path = root / args.file
    original = path.read_bytes()
    lines = original.decode().splitlines(keepends=True)
    if not 1 <= args.line <= len(lines):
        raise ToolError(f"{args.file} has {len(lines)} lines; --line {args.line} is out of range")
    target = lines[args.line - 1]
    indent = target[: len(target) - len(target.lstrip())]
    probe = f'{indent}panic!("{PROBE_MARKER} {args.file}:{args.line}");\n'

    command = ["cargo", "test", "--package", args.package]
    if args.lib:
        command.append("--lib")
    if args.filter:
        command.append(args.filter)
    command += ["--", "--test-threads", "1"]

    before = tree_fingerprint(root)
    try:
        path.write_text("".join(lines[: args.line - 1]) + probe + "".join(lines[args.line - 1 :]))
        print("$ " + shlex.join(command), flush=True)
        result = subprocess.run(command, cwd=root, capture_output=True, text=True, check=False, env=cargo_env())
    finally:
        path.write_bytes(original)
    if tree_fingerprint(root) != before:
        raise ToolError(f"{args.file} was not restored after the probe; inspect `git diff` now")

    output = result.stdout + result.stderr
    # Failing tests also print `error: test failed`, so only a compile failure counts here.
    if "could not compile" in output:
        errors = [line for line in output.splitlines() if line.startswith("error")][:5]
        print("probe did not compile; choose a line that starts a statement inside a function body:")
        print("\n".join("   " + line for line in errors))
        return 2

    # libtest prints `thread '<test>' ... panicked at <location>:` and the message on the next line.
    reached = []
    output_lines = output.splitlines()
    for index, line in enumerate(output_lines):
        match = re.match(r"thread '([^']+)'.* panicked at ", line)
        if not match:
            continue
        if any(PROBE_MARKER in following for following in output_lines[index : index + 3]):
            reached.append(match.group(1))
    if reached:
        print(f"REACHED by {len(reached)} tests:")
        for name in sorted(set(reached)):
            print(f"   {name}")
    else:
        print(f"NOT REACHED: no test in `{' '.join(command)}` executes {args.file}:{args.line}.")
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    root = repo_root()
    run_dir = Path(args.run)
    run = json.loads((run_dir / "run.json").read_text())
    survivors = json.loads((run_dir / "survivors.json").read_text())
    cargo_mutants_version(root)
    out = prepare_out_dir(root, Path(args.out) if args.out else run_dir / "verify")

    targets = [m for g in survivors["groups"] for m in g["mutants"]]
    patterns = exclusion_patterns(root)
    results = {m["name"]: {"status": "not-found", "excluded_by": excluded_by(m["name"], patterns)} for m in targets}

    to_test = [m for m in targets if results[m["name"]]["excluded_by"] is None]
    status = 0
    outcomes: list[dict[str, Any]] = []
    if to_test:
        selection: list[str] = ["--package", str(run["package"])]
        for pattern in sorted({m["pattern"] for m in to_test}):
            selection += ["--re", pattern]
        status = run_cargo_mutants(root, selection, out)
        if status == EXIT_BASELINE_FAILED:
            raise ToolError(f"the tests fail without any mutation; see {out}/mutants.out/log/baseline.log")
        if status not in EXIT_REPORTABLE:
            raise ToolError(f"cargo-mutants exited with status {status}; see {out}/cargo-mutants.log")
        outcomes = load_outcomes(out / "mutants.out")

    by_pattern: dict[str, list[dict[str, Any]]] = {}
    for outcome in outcomes:
        mutant = outcome["scenario"]["Mutant"]
        entry = {"name": mutant["name"], "line": mutant["span"]["start"]["line"], "summary": outcome["summary"]}
        by_pattern.setdefault(name_pattern(mutant), []).append(entry)

    for target in targets:
        result = results[target["name"]]
        if result["excluded_by"] is not None:
            result["status"] = "excluded"
            continue
        candidates = by_pattern.get(target["pattern"], [])
        if not candidates:
            continue
        # The same change can apply on several lines of a function; match the nearest one, since
        # edits above it can move it.
        match = min(candidates, key=lambda c: abs(c["line"] - target["line"]))
        result["status"] = STATUS_BY_SUMMARY.get(match["summary"], match["summary"])
        result["now"] = match["name"]

    verify = {
        "schema": SCHEMA_VERSION,
        "kind": "verify",
        "commit": git(root, "rev-parse", "HEAD").strip(),
        "exit_status": status,
        "results": results,
    }
    (out / "verify.json").write_text(json.dumps(verify, indent=2) + "\n")

    tally: dict[str, int] = {}
    for result in results.values():
        status_name = str(result["status"])
        tally[status_name] = tally.get(status_name, 0) + 1
    print("\nSurvivors now: " + ", ".join(f"{n} {s}" for s, n in sorted(tally.items())))
    for name, result in results.items():
        if result["status"] in ("missed", "not-found"):
            print(f"   {result['status']:9}  {name}")
    return 0


QUESTION_TOPICS = ("removal", "semantics", "equivalence", "bug", "other")
REMOVAL_KINDS = ("no-kills", "duplicate")


def statuses_from_compare(root: Path, survivor_names: list[str], comparison: dict[str, Any]) -> dict[str, Any]:
    """Survivor statuses from a full rerun, for when `verify` was not run."""
    fixed = {c["name"] for c in comparison["changes"]["fixed"]}
    vanished = {c["name"] for c in comparison["changes"]["vanished"]}
    patterns = exclusion_patterns(root)
    results = {}
    for name in survivor_names:
        if name in fixed:
            results[name] = {"status": "caught"}
        elif name in vanished:
            results[name] = {"status": "excluded" if excluded_by(name, patterns) else "not-found"}
        else:
            results[name] = {"status": "missed"}
    return results


def load_optional(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ToolError(f"{path} must hold a JSON object")
    return value


def cmd_report(args: argparse.Namespace) -> int:
    root = repo_root()
    run_dir = Path(args.run)
    run = json.loads((run_dir / "run.json").read_text())
    survivors = json.loads((run_dir / "survivors.json").read_text())
    triage = json.loads((Path(args.triage) if args.triage else run_dir / "triage.json").read_text())
    verify = load_optional(Path(args.verify) if args.verify else run_dir / "verify" / "verify.json")
    comparison = load_optional(Path(args.compare) if args.compare else run_dir / "compare.json")
    answers = load_optional(Path(args.answers) if args.answers else run_dir / "answers.json") or {}
    kills_path = Path(args.kills) if args.kills else run_dir / "kills.json"
    matrix = load_optional(kills_path)

    survivor_names = [m["name"] for g in survivors["groups"] for m in g["mutants"]]
    errors: list[str] = []
    if verify is not None:
        results = verify["results"]
    elif comparison is not None:
        results = statuses_from_compare(root, survivor_names, comparison)
    else:
        results = {}
        if survivor_names:
            errors.append("neither verify.json nor compare.json exists; run `verify` or rerun and `compare`")
    if comparison is not None and comparison["changes"]["regressed"]:
        for change in comparison["changes"]["regressed"]:
            errors.append(f"regressed: {change['name']} was caught before the change and is missed after it")

    questions = triage.get("questions", [])
    question_ids = [q.get("id") for q in questions]
    refs: dict[str, list[dict[str, Any]]] = {}
    for index, question in enumerate(questions):
        label = f"question {question.get('id') or index + 1}"
        if not question.get("id") or question_ids.count(question["id"]) > 1:
            errors.append(f"{label}: give each question a unique id, such as Q1")
        if question.get("topic") not in QUESTION_TOPICS:
            errors.append(f"{label}: topic must be one of {', '.join(QUESTION_TOPICS)}")
        if not str(question.get("question", "")).strip():
            errors.append(f"{label}: ask the question")
        options = question.get("options", [])
        if len(options) < 2:
            errors.append(f"{label}: offer at least two options")
        if question.get("default") not in options:
            errors.append(f"{label}: the default, what happens with no answer, must be one of the options")
        answer = answers.get(question.get("id"), {}).get("answer")
        if answer is not None and answer not in options:
            errors.append(f"{label}: the answer {answer!r} is not one of its options")
        for ref in question.get("refs", []):
            refs.setdefault(ref, []).append(question)

    def answered(question: dict[str, Any]) -> str | None:
        answer = answers.get(question["id"], {}).get("answer")
        return None if answer is None else str(answer)

    seen: dict[str, int] = {}
    entries = triage.get("entries", [])
    for index, entry in enumerate(entries):
        label = f"entry {index + 1}"
        verdict = entry.get("verdict")
        if verdict not in VERDICTS:
            errors.append(f"{label}: verdict must be one of {', '.join(VERDICTS)}, not {verdict!r}")
        if not str(entry.get("reason", "")).strip():
            errors.append(f"{label}: give a reason")
        if not entry.get("mutants"):
            errors.append(f"{label}: list the mutants it covers")
        if verdict == "gap-fixed" and not entry.get("tests"):
            errors.append(f"{label}: a gap-fixed entry names the tests that kill its mutants")
        if verdict == "gap-fixed" and not str(entry.get("oracle", "")).strip():
            errors.append(
                f"{label}: say where the new tests' expected values come from (`oracle`): a reference "
                "implementation or the input contract, never the code's current output"
            )
        if verdict in ("bug", "deferred") and not any(m in refs for m in entry.get("mutants", [])):
            errors.append(f"{label}: a {verdict} verdict is a decision for a person; add a question that refs it")
        for name in entry.get("mutants", []):
            seen[name] = seen.get(name, 0) + 1
            if name not in survivor_names:
                errors.append(f"{label}: {name!r} is not a survivor of this run")
                continue
            status = results.get(name, {}).get("status")
            if verdict == "gap-fixed" and status not in ("caught", "timeout"):
                errors.append(f"{label}: {name!r} is marked gap-fixed but is {status} after the change")
            if verdict == "equivalent" and status != "excluded":
                errors.append(f"{label}: {name!r} is marked equivalent but no exclude_re entry matches it")
            if verdict in ("bug", "deferred") and status in ("caught", "timeout"):
                errors.append(f"{label}: {name!r} is marked {verdict} but is now caught; mark it gap-fixed")
    for name in survivor_names:
        if name not in seen:
            errors.append(f"untriaged survivor: {name}")
        elif seen[name] > 1:
            errors.append(f"survivor triaged {seen[name]} times: {name}")

    removals = triage.get("removals", [])
    if removals and matrix is None:
        errors.append(f"{kills_path} does not exist; removals need a kill matrix as evidence")
    if removals and args.kills is None and run["selection"] != ["--package", run["package"]]:
        errors.append(
            "this run covers part of the crate, so a test that catches nothing here may catch mutants "
            "elsewhere. Pass --kills with a package-wide matrix (`kills <runs or CI shards> --out`)."
        )
    removed_tests = {t for r in removals for t in r.get("tests", [])}
    for index, removal in enumerate(removals):
        label = f"removal {index + 1}"
        kind = removal.get("kind")
        if kind not in REMOVAL_KINDS:
            errors.append(f"{label}: kind must be one of {', '.join(REMOVAL_KINDS)}")
        if not str(removal.get("reason", "")).strip():
            errors.append(f"{label}: give a reason")
        if not removal.get("tests"):
            errors.append(f"{label}: list the tests")
        for test in removal.get("tests", []):
            if matrix is None:
                break
            if test not in matrix["tests"]:
                errors.append(f"{label}: {test!r} did not run in the kill matrix's runs")
                continue
            caught = set(matrix["kills"].get(test, []))
            if kind == "no-kills" and caught:
                errors.append(f"{label}: {test!r} catches {len(caught)} mutants, so it is not a no-kills test")
            if kind == "duplicate":
                other = removal.get("duplicate_of")
                if not other or other in removed_tests:
                    errors.append(f"{label}: name a kept test in `duplicate_of`")
                elif not caught <= set(matrix["kills"].get(other, [])):
                    errors.append(f"{label}: {test!r} catches mutants that {other!r} does not")
            asked = [q for q in refs.get(test, []) if q.get("topic") == "removal"]
            decisions = {a for a in (answered(q) for q in asked) if a is not None}
            if removal.get("applied"):
                if comparison is None:
                    errors.append(f"{label}: an applied removal needs a full rerun and `compare` as proof")
                elif test not in comparison["tests_removed"]:
                    errors.append(f"{label}: {test!r} still runs after the change")
                if not asked:
                    errors.append(f"{label}: ask about removing {test!r}; a person confirms every removal")
                elif decisions and decisions != {"remove"}:
                    errors.append(f"{label}: {test!r} was answered {sorted(decisions)}; restore it")
            else:
                if not asked:
                    errors.append(f"{label}: a proposed removal needs a removal question that refs {test!r}")
                elif "remove" in decisions:
                    errors.append(f"{label}: removing {test!r} was approved; apply it, rerun and compare")

    if errors:
        print("report: the triage is incomplete or inconsistent:", file=sys.stderr)
        for error in errors:
            print(f"   - {error}", file=sys.stderr)
        return 1

    changed = sorted(
        set(filter(None, git(root, "diff", "--name-only", run["commit"]).splitlines()))
        | set(filter(None, git(root, "ls-files", "--others", "--exclude-standard").splitlines()))
    )
    verdicts = {verdict: 0 for verdict in VERDICTS}
    for entry in entries:
        verdicts[entry["verdict"]] += len(entry["mutants"])
    rendered_questions = [
        {**q, "answer": answered(q), "answered_by": answers.get(q["id"], {}).get("by")} for q in questions
    ]
    report = {
        "schema": SCHEMA_VERSION,
        "kind": "report",
        "package": run["package"],
        "selection": run["selection"],
        "run_commit": run["commit"],
        "run_dirty": run["dirty"],
        "before": run["counts"],
        "after": comparison["after"]["counts"] if comparison else None,
        "verdicts": verdicts,
        "timeouts": survivors["timeouts"],
        "entries": [
            {**entry, "status": {name: results.get(name, {}).get("status") for name in entry["mutants"]}}
            for entry in entries
        ],
        "removals": removals,
        "questions": rendered_questions,
        "open_questions": sum(1 for q in rendered_questions if q["answer"] is None),
        "files_changed": changed,
    }
    (run_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    markdown = render_markdown(report)
    (run_dir / "report.md").write_text(markdown)
    print(markdown)
    return 0


def details(summary: str, items: list[str], indent: str = "  ") -> list[str]:
    """A collapsed Markdown list, so long mutant names do not bury the decisions."""
    if not items:
        return []
    return [
        f"{indent}<details><summary>{summary}</summary>",
        "",
        *[f"{indent}- {item}" for item in items],
        "",
        f"{indent}</details>",
    ]


def render_markdown(report: dict[str, Any]) -> str:
    before = report["before"]
    lines = [
        f"## Mutation testing: {report['package']}",
        "",
        f"Scope: `cargo mutants {' '.join(report['selection'])}` at `{report['run_commit'][:12]}`"
        + (" (with uncommitted changes)" if report["run_dirty"] else ""),
        "",
        "| | Caught | Missed | Timed out | Unviable |",
        "|---|---:|---:|---:|---:|",
        f"| Before | {before['caught']} | {before['missed']} | {before['timeout']} | {before['unviable']} |",
    ]
    if report["after"]:
        after = report["after"]
        lines.append(f"| After | {after['caught']} | {after['missed']} | {after['timeout']} | {after['unviable']} |")
    lines.append("")

    added = [e for e in report["entries"] if e["verdict"] == "gap-fixed"]
    if added:
        lines += ["### Tests added", ""]
        for entry in added:
            lines.append(f"- {entry['reason']}")
            lines.append(f"  - expected values: {entry['oracle']}")
            for test in entry["tests"]:
                lines.append(f"  - test: `{test}`")
            count = len(entry["mutants"])
            lines.append(f"  - catches {count} survivor{'' if count == 1 else 's'}")
        lines.append("")
    applied = [r for r in report["removals"] if r.get("applied")]
    if applied:
        lines += ["### Tests removed", ""]
        for removal in applied:
            evidence = (
                "caught no mutant in the package"
                if removal["kind"] == "no-kills"
                else f"caught only mutants that `{removal['duplicate_of']}` also catches"
            )
            lines.append(f"- {removal['reason']} Evidence: {evidence}; the rerun shows no regressions.")
            lines += [f"  - `{test}`" for test in removal["tests"]]
        lines.append("")

    others = [e for e in report["entries"] if e["verdict"] != "gap-fixed"]
    if others:
        lines += ["### Other survivors", ""]
        for entry in others:
            lines.append(f"- **{entry['verdict']}**: {entry['reason']}")
            lines += details(
                f"{len(entry['mutants'])} mutant{'' if len(entry['mutants']) == 1 else 's'}",
                [f"`{n}` → {entry['status'][n]}" for n in entry["mutants"]],
            )
        lines.append("")

    questions = report["questions"]
    if questions:
        lines += [
            "### Questions for reviewers",
            "",
            "Each question is a decision the agent should not make alone. Tick one option, or reply "
            "`Q<n>: <option>`. An unanswered question takes its default.",
            "",
        ]
        for question in sorted(questions, key=lambda q: q["answer"] is not None):
            status = f"answered: **{question['answer']}**" if question["answer"] else "open"
            lines.append(f"**{question['id']}** ({question['topic']}, {status}): {question['question']}")
            if question.get("context"):
                lines += ["", f"> {question['context']}"]
            lines.append("")
            for option in question["options"]:
                marks = []
                if option == question.get("recommendation"):
                    marks.append("recommended")
                if option == question["default"]:
                    marks.append("default")
                tick = "x" if option == question["answer"] else " "
                suffix = f" ({', '.join(marks)})" if marks else ""
                lines.append(f"- [{tick}] {option}{suffix}")
            lines += ["", *details("About", [f"`{ref}`" for ref in question.get("refs", [])], indent="")]
            lines.append("")
    if report["timeouts"]:
        lines += ["### Timed out (counted as detected)", ""]
        lines += [f"- `{name}`" for name in report["timeouts"]]
        lines.append("")
    return "\n".join(lines)


# --- entry point -----------------------------------------------------------------------------


def add_scope_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--package", "-p", required=True, help="crate to mutate, e.g. vortex-fastlanes")
    parser.add_argument("--file", "-f", action="append", help="limit to source files matching this glob")
    parser.add_argument("--re", action="append", help="limit to mutants whose name matches this regex")
    parser.add_argument("--in-diff", help="limit to mutants in the lines this diff changes")
    parser.add_argument(
        "cargo_mutants_args",
        nargs="*",
        help="extra cargo-mutants selection arguments after `--`, such as --shard 1/4",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="test every mutant in a scope")
    add_scope_args(run)
    run.add_argument("--out", required=True, help="fresh output directory, e.g. target/mutants/<label>")
    run.add_argument("--max-mutants", type=int, default=DEFAULT_MAX_MUTANTS, help="refuse larger scopes")
    run.set_defaults(handler=cmd_run)

    survivors = commands.add_parser("survivors", help="print a run's survivors")
    survivors.add_argument("run", help="the run's output directory")
    survivors.add_argument("--json", action="store_true", help="print JSON instead of text")
    survivors.set_defaults(handler=cmd_survivors)

    probe = commands.add_parser("probe", help="check whether tests reach a line")
    probe.add_argument("--package", "-p", required=True)
    probe.add_argument("--file", required=True, help="repository-relative source file")
    probe.add_argument("--line", type=int, required=True, help="a line that starts a statement")
    probe.add_argument("--lib", action="store_true", help="run only the library's unit tests")
    probe.add_argument("--filter", help="run only tests whose name contains this")
    probe.set_defaults(handler=cmd_probe)

    verify = commands.add_parser("verify", help="re-test a run's survivors against the current tree")
    verify.add_argument("run", help="the run's output directory")
    verify.add_argument("--out", help="fresh output directory (default: <run>/verify)")
    verify.set_defaults(handler=cmd_verify)

    kills = commands.add_parser("kills", help="print the kill matrix: which tests catch which mutants")
    kills.add_argument("sources", nargs="+", help="run directories, or cargo-mutants mutants.out directories")
    kills.add_argument("--out", help="write the merged matrix here, for `report --kills`")
    kills.add_argument("--json", action="store_true", help="print the analysis as JSON")
    kills.add_argument("--limit", type=int, default=15, help="how many identical groups to print")
    kills.set_defaults(handler=cmd_kills)

    compare = commands.add_parser("compare", help="diff two runs of one scope; nonzero exit on regressions")
    compare.add_argument("before", help="the run before the change")
    compare.add_argument("after", help="a fresh run of the same scope after the change")
    compare.set_defaults(handler=cmd_compare)

    report = commands.add_parser("report", help="validate the triage and write report.json and report.md")
    report.add_argument("run", help="the run's output directory")
    report.add_argument("--triage", help="triage file (default: <run>/triage.json)")
    report.add_argument("--verify", help="verify.json (default: <run>/verify/verify.json)")
    report.add_argument("--compare", help="compare.json (default: <run>/compare.json)")
    report.add_argument("--answers", help="answers to the questions (default: <run>/answers.json)")
    report.add_argument("--kills", help="package-wide kill matrix for removals (default: <run>/kills.json)")
    report.set_defaults(handler=cmd_report)

    args = parser.parse_args()
    try:
        return int(args.handler(args))
    except ToolError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    except BrokenPipeError:
        # The reader stopped early, as with `| head`; that is not an error.
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        return 0


if __name__ == "__main__":
    sys.exit(main())
