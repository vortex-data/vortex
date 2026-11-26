# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Summarize ephyra JSON reports and file their failures as GitHub issues.

`summary TARGET REPORT [--work-dir D]` prints Markdown for `$GITHUB_STEP_SUMMARY`: throughput,
timing, coverage per worker when the build is instrumented, failures, health warnings and the
input distributions. `issues TARGET REPORT --repo R
--run-url U --commit C [--crash-dir D]` opens one issue per failure signature unless an open issue
already has that exact title, so a failure that recurs on every run until it is fixed files one
issue and no comments. Failures are merged across the workers' reports first. With `--crash-dir`,
each issue carries the saved plan that reproduces it.
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

Report = dict[str, Any]

MAX_TITLE = 200
MAX_MESSAGE = 6000


def load(path: str) -> list[Report]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def failures(report: Report) -> list[Report]:
    """Distinct failures, falling back to the single `failure` of older reports."""
    found = report.get("failures") or []
    if not found and report.get("failure"):
        found = [
            {
                "signature": report["failure"].get("message", ""),
                "count": 1,
                "failure": report["failure"],
            }
        ]
    return found


def replay_line(failure: Report) -> str | None:
    replay = failure.get("replay")
    return f"EPHYRA_REPLAY={replay['seed']}:{replay['iteration']}" if replay else None


def counterexample(failure: Report) -> str:
    example = failure.get("shrunk") or failure.get("original") or {}
    return example.get("debug", "")


def merged_failures(reports: list[Report]) -> list[tuple[Report, Report]]:
    """Each distinct signature across all workers' reports, with its total count.

    Pairs the first failure seen for a signature with the report it came from.
    """
    merged: dict[str, tuple[Report, Report]] = {}
    for report in reports:
        for found in failures(report):
            signature = found.get("signature", "unknown failure")
            if signature in merged:
                merged[signature][0]["count"] += found.get("count", 1)
            else:
                merged[signature] = ({**found, "count": found.get("count", 1)}, report)
    return list(merged.values())


def crash_plans(crash_dir: str | None) -> dict[str, str]:
    """The saved crash file contents for each signature in `crash_dir`."""
    plans: dict[str, str] = {}
    if not crash_dir or not Path(crash_dir).is_dir():
        return plans
    for path in sorted(Path(crash_dir).glob("*.json")):
        try:
            crash = json.loads(path.read_text())
            plans[crash["signature"]] = json.dumps(crash)
        except (OSError, ValueError, KeyError) as e:
            print(f"skipping {path}: {e}", file=sys.stderr)
    return plans


def seconds(duration: Report | None) -> float:
    return duration["secs"] + duration["nanos"] / 1e9 if duration else 0.0


def fmt_ms(duration: Report | None) -> str:
    return f"{seconds(duration) * 1000:.2f}"


def corpus_counts(work_dir: str | None) -> dict[int, tuple[str, str]]:
    """Corpus entries each worker replayed and added, from its log in `work_dir/N/output.log`."""
    counts: dict[int, tuple[str, str]] = {}
    if not work_dir or not Path(work_dir).is_dir():
        return counts
    for log in Path(work_dir).glob("*/output.log"):
        replayed = added = "-"
        text = log.read_text(errors="replace")
        if match := re.search(r"ephyra: replaying (\d+) corpus entries", text):
            replayed = match.group(1)
        if match := re.search(r"ephyra: added (\d+) corpus entries", text):
            added = match.group(1)
        if log.parent.name.isdigit():
            counts[int(log.parent.name)] = (replayed, added)
    return counts


def summary(args: argparse.Namespace) -> None:
    reports = load(args.report)
    tests = sum(r["tests"] for r in reports)
    wall = max((seconds(r["duration"]) for r in reports), default=0.0)
    statuses = sorted({r["status"] for r in reports})
    print(f"### `{args.target}` (ephyra): {', '.join(statuses)}")
    rate = f", {tests / wall:,.0f} tests/s" if wall else ""
    print(f"{tests:,} tests over {len(reports)} worker(s) in {wall:,.0f} s{rate}\n")

    corpus = corpus_counts(args.work_dir)
    print(
        "| worker | status | tests | tests/s | discarded | property median / p90 / max (ms) "
        "| corpus replayed / added | seed |"
    )
    print("|---|---|---|---|---|---|---|---|")
    for i, r in enumerate(reports):
        secs = seconds(r["duration"])
        prop = (r.get("timing") or {}).get("prop") or {}
        replayed, added = corpus.get(i, ("-", "-"))
        print(
            f"| {i} | {r['status']} | {r['tests']:,} | {r['tests'] / secs if secs else 0:,.0f} "
            f"| {r['discarded']:,} | {fmt_ms(prop.get('median'))} / {fmt_ms(prop.get('p90'))} / "
            f"{fmt_ms(prop.get('max'))} | {replayed} / {added} | `{r['seed']}` |"
        )

    covered = [r for r in reports if r.get("coverage")]
    if covered:
        print("\n#### Coverage\n")
        print(
            "| worker | crate | functions reached | regions hit | inputs with new coverage "
            "| last growth at input |"
        )
        print("|---|---|---|---|---|---|")
        for i, r in enumerate(reports):
            c = r.get("coverage")
            if not c:
                continue
            fn_pct = 100 * c["functions_reached"] / max(c["functions_total"], 1)
            ct_pct = 100 * c["counters_hit"] / max(c["counters_total"], 1)
            print(
                f"| {i} | `{c['filter'] or 'all'}` "
                f"| {c['functions_reached']:,} / {c['functions_total']:,} ({fn_pct:.1f}%) "
                f"| {c['counters_hit']:,} / {c['counters_total']:,} ({ct_pct:.1f}%) "
                f"| {c['novel_inputs']:,} | {c['last_novelty_at']:,} |"
            )
        for i, r in enumerate(reports):
            c = r.get("coverage")
            if not c or not (c.get("partial") or c.get("unreached")):
                continue
            print(f"\n<details><summary>Worker {i} (`{c['filter']}`): least covered</summary>\n")
            for name, hit, total in (c.get("partial") or [])[:15]:
                print(f"- `{name}`: {hit} / {total} regions")
            unreached = c.get("unreached") or []
            if unreached:
                shown = ", ".join(f"`{n}`" for n in unreached[:15])
                print(f"\nNever reached ({len(unreached):,}): {shown}")
            print("\n</details>")

    found_failures = merged_failures(reports)
    if found_failures:
        print("\n#### Failures\n")
    for found, _ in found_failures:
        failure = found.get("failure") or {}
        print(f"- {found['count']}x `{found.get('signature', '')}`")
        if replay := replay_line(failure):
            print(f"  - replay: `{replay}`")
        if example := counterexample(failure):
            print(f"  - counterexample: `{example[:2000]}`")

    health = sorted({(h["kind"], h["message"]) for r in reports for h in r.get("health") or []})
    if health:
        print("\n#### Health\n")
        for kind, message in health:
            print(f"- `{kind}`: {message}")

    features: dict[str, list[Report]] = {}
    for r in reports:
        for name, stats in (r.get("features") or {}).items():
            features.setdefault(name, []).append(stats)
    covers: dict[str, list[float]] = {}
    for r in reports:
        for name, share in (r.get("covers") or {}).items():
            covers.setdefault(name, []).append(share)
    if features or covers:
        print("\n#### Inputs\n")
        for name, stats in sorted(features.items()):
            median = sum(s["median"] for s in stats) / len(stats)
            p90 = sum(s["p90"] for s in stats) / len(stats)
            print(
                f"- `{name}`: min {min(s['min'] for s in stats):g}, median {median:g}, "
                f"p90 {p90:g}, max {max(s['max'] for s in stats):g} "
                "(median and p90 averaged over workers)"
            )
        for name, shares in sorted(covers.items()):
            print(f"- {name}: {100 * sum(shares) / len(shares):.1f}% of passing inputs")

    labels: dict[str, int] = {}
    for r in reports:
        for name, count in (r.get("labels") or {}).items():
            labels[name] = labels.get(name, 0) + count
    if labels:
        print("\n<details><summary>Labels</summary>\n")
        print("| label | count |\n|---|---|")
        for name, count in sorted(labels.items()):
            print(f"| {name} | {count:,} |")
        print("\n</details>")


def gh(*args: str) -> str:
    return subprocess.run(["gh", *args], check=True, capture_output=True, text=True).stdout


def issue_body(found: Report, plan: str | None, args: argparse.Namespace) -> str:
    failure = found.get("failure") or {}
    message = (failure.get("message") or "")[:MAX_MESSAGE]
    test = f"{args.target}_ephyra"
    lines = [
        f"The ephyra `{args.target}` search failed {found.get('count', 1)} time(s) "
        f"in [this run]({args.run_url}) at commit {args.commit}.",
        "",
        f"Signature: `{found.get('signature', '')}`",
        "",
        "Shrunk counterexample:",
        "```",
        counterexample(failure)[:MAX_MESSAGE],
        "```",
    ]
    if plan:
        lines += [
            "",
            "Reproduce: save this plan as `crash.json` and replay it.",
            "```json",
            plan[:MAX_MESSAGE],
            "```",
            "```bash",
            "VORTEX_FUZZ_PLAN=crash.json cargo test -p vortex-fuzz --features ephyra "
            f"--test {test} -- --ignored --exact {args.target}_plan --nocapture",
            "```",
        ]
    lines += [
        "",
        "<details><summary>Failure message</summary>",
        "",
        "```",
        message,
        "```",
        "</details>",
    ]
    return "\n".join(lines)


def issues(args: argparse.Namespace) -> None:
    prefix = f"ephyra {args.target}: "
    plans = crash_plans(args.crash_dir)
    for found, _ in merged_failures(load(args.report)):
        signature = found.get("signature", "unknown failure")
        title = (prefix + signature)[:MAX_TITLE]
        body = issue_body(found, plans.get(signature), args)
        existing = json.loads(
            gh(
                "issue",
                "list",
                "--repo",
                args.repo,
                "--state",
                "open",
                "--search",
                f'in:title "{prefix.strip()}"',
                "--json",
                "number,title",
                "--limit",
                "100",
            )
        )
        match = next((i for i in existing if i["title"] == title), None)
        if match:
            # A known bug recurs on every run until it is fixed; do not comment each time.
            print(f"Already open as #{match['number']}: {title}")
        else:
            url = gh(
                "issue", "create", "--repo", args.repo, "--title", title, "--body", body
            ).strip()
            print(f"Opened {url}: {title}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("summary")
    s.add_argument("target")
    s.add_argument("report")
    s.add_argument("--work-dir", help="worker directories, for corpus counts from their logs")
    s.set_defaults(func=summary)
    i = sub.add_parser("issues")
    i.add_argument("target")
    i.add_argument("report")
    i.add_argument("--repo", required=True)
    i.add_argument("--run-url", required=True)
    i.add_argument("--commit", required=True)
    i.add_argument("--crash-dir")
    i.set_defaults(func=issues)
    args = parser.parse_args()
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
