#!/usr/bin/env python3
"""Drive cargo-mutants over one scope for the mutation-testing skill.

Subcommands, in the order the skill uses them:

  run        Test every mutant in a scope and record the survivors with source excerpts.
  survivors  Print a run's surviving mutants, grouped by function and ranked.
  probe      Check whether any test reaches a source line, by inserting a temporary panic.
  verify     Re-test a run's survivors against the current tree.
  report     Check a triage file against a run and its verification, and write the report.

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
    command += ["--output", str(out)]
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

    tally = run["counts"]
    print(
        f"\n{len(listed)} mutants in {run['seconds']}s: {tally['caught']} caught, {tally['missed']} missed, "
        f"{tally['timeout']} timed out, {tally['unviable']} unviable."
    )
    print(f"Survivors: python3 {sys.argv[0]} survivors {out}")
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


def cmd_report(args: argparse.Namespace) -> int:
    root = repo_root()
    run_dir = Path(args.run)
    run = json.loads((run_dir / "run.json").read_text())
    survivors = json.loads((run_dir / "survivors.json").read_text())
    triage_path = Path(args.triage) if args.triage else run_dir / "triage.json"
    verify_path = Path(args.verify) if args.verify else run_dir / "verify" / "verify.json"
    triage = json.loads(triage_path.read_text())
    verify = json.loads(verify_path.read_text()) if verify_path.exists() else None

    survivor_names = [m["name"] for g in survivors["groups"] for m in g["mutants"]]
    errors: list[str] = []
    if survivor_names and verify is None:
        errors.append(f"{verify_path} does not exist; run `verify` after writing tests and exclusions")
    results = verify["results"] if verify else {}

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
        for name in entry.get("mutants", []):
            seen[name] = seen.get(name, 0) + 1
            if name not in survivor_names:
                errors.append(f"{label}: {name!r} is not a survivor of this run")
                continue
            status = results.get(name, {}).get("status")
            if verdict == "gap-fixed" and status not in ("caught", "timeout"):
                errors.append(f"{label}: {name!r} is marked gap-fixed but verify found it {status}")
            if verdict == "equivalent" and status != "excluded":
                errors.append(f"{label}: {name!r} is marked equivalent but no exclude_re entry matches it")
            if verdict in ("bug", "deferred") and status in ("caught", "timeout"):
                errors.append(f"{label}: {name!r} is marked {verdict} but is now caught; mark it gap-fixed")
    for name in survivor_names:
        if name not in seen:
            errors.append(f"untriaged survivor: {name}")
        elif seen[name] > 1:
            errors.append(f"survivor triaged {seen[name]} times: {name}")

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
    report = {
        "schema": SCHEMA_VERSION,
        "kind": "report",
        "package": run["package"],
        "selection": run["selection"],
        "run_commit": run["commit"],
        "run_dirty": run["dirty"],
        "before": run["counts"],
        "verdicts": verdicts,
        "timeouts": survivors["timeouts"],
        "entries": [
            {**entry, "status": {name: results.get(name, {}).get("status") for name in entry["mutants"]}}
            for entry in entries
        ],
        "files_changed": changed,
    }
    (run_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    markdown = render_markdown(report)
    (run_dir / "report.md").write_text(markdown)
    print(markdown)
    return 0


def render_markdown(report: dict[str, Any]) -> str:
    before = report["before"]
    viable = before["caught"] + before["missed"] + before["timeout"]
    after_missed = sum(1 for entry in report["entries"] for status in entry["status"].values() if status == "missed")
    lines = [
        f"## Mutation testing: {report['package']}",
        "",
        f"Scope: `cargo mutants {' '.join(report['selection'])}` at `{report['run_commit'][:12]}`"
        + (" (with uncommitted changes)" if report["run_dirty"] else ""),
        "",
        "| | Viable | Caught | Missed | Timed out |",
        "|---|---:|---:|---:|---:|",
        f"| Before | {viable} | {before['caught']} | {before['missed']} | {before['timeout']} |",
        "",
        "| Verdict | Mutants |",
        "|---|---:|",
    ]
    lines += [f"| {verdict} | {count} |" for verdict, count in report["verdicts"].items()]
    lines += ["", f"Missed after this change, excluding equivalents: {after_missed}.", ""]
    for verdict in VERDICTS:
        entries = [e for e in report["entries"] if e["verdict"] == verdict]
        if not entries:
            continue
        lines += [f"### {verdict}", ""]
        for entry in entries:
            lines.append(f"- {entry['reason']}")
            for test in entry.get("tests", []):
                lines.append(f"  - test: `{test}`")
            for name in entry["mutants"]:
                lines.append(f"  - `{name}` → {entry['status'][name]}")
        lines.append("")
    if report["timeouts"]:
        lines += ["### Timed out (counted as detected)", ""]
        lines += [f"- `{name}`" for name in report["timeouts"]]
        lines.append("")
    if report["files_changed"]:
        lines += ["### Files changed", ""]
        lines += [f"- `{name}`" for name in report["files_changed"]]
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

    report = commands.add_parser("report", help="validate the triage and write report.json and report.md")
    report.add_argument("run", help="the run's output directory")
    report.add_argument("--triage", help="triage file (default: <run>/triage.json)")
    report.add_argument("--verify", help="verify.json (default: <run>/verify/verify.json)")
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
