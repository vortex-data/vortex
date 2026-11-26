# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Summarize ephyra JSON reports and file their failures as GitHub issues.

`summary REPORT` prints Markdown for `$GITHUB_STEP_SUMMARY`. `issues REPORT --repo R --run-url U
--commit C` opens one issue per failure signature, or comments on the open issue whose title
already names it. Issues are matched by exact title, so a failure that keeps recurring collects
comments instead of duplicates.
"""

import argparse
import json
import subprocess
import sys
from typing import Any

Report = dict[str, Any]

TITLE_PREFIX = "ephyra array_ops: "
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


def summary(args: argparse.Namespace) -> None:
    reports = load(args.report)
    tests = sum(r["tests"] for r in reports)
    statuses = sorted({r["status"] for r in reports})
    print(f"### `array_ops` (ephyra): {', '.join(statuses)}")
    print(f"{tests} tests over {len(reports)} worker(s)\n")
    print("| worker | status | tests | discarded | seconds | seed |\n|---|---|---|---|---|---|")
    for i, r in enumerate(reports):
        print(
            f"| {i} | {r['status']} | {r['tests']} | {r['discarded']} "
            f"| {r['duration']['secs']} | `{r['seed']}` |"
        )
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
    labels: dict[str, int] = {}
    for r in reports:
        for name, count in (r.get("labels") or {}).items():
            labels[name] = labels.get(name, 0) + count
    if labels:
        print("\n| label | count |\n|---|---|")
        for name, count in sorted(labels.items()):
            print(f"| {name} | {count} |")


def gh(*args: str) -> str:
    return subprocess.run(["gh", *args], check=True, capture_output=True, text=True).stdout


def issue_body(found: Report, report: Report, args: argparse.Namespace) -> str:
    failure = found.get("failure") or {}
    replay = replay_line(failure) or f"EPHYRA_SEED={report['seed']}"
    message = (failure.get("message") or "")[:MAX_MESSAGE]
    lines = [
        f"The ephyra `array_ops` property failed {found.get('count', 1)} time(s) "
        f"in [this run]({args.run_url}) at commit {args.commit}.",
        "",
        f"Signature: `{found.get('signature', '')}`",
        "",
        "Shrunk counterexample:",
        "```",
        counterexample(failure)[:MAX_MESSAGE],
        "```",
        "",
        "Reproduce:",
        "```bash",
        f"{replay} cargo test --release -p vortex-fuzz --features ephyra "
        "--test array_ops_ephyra -- --nocapture",
        "```",
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
    for found, report in merged_failures(load(args.report)):
        title = (TITLE_PREFIX + found.get("signature", "unknown failure"))[:MAX_TITLE]
        body = issue_body(found, report, args)
        existing = json.loads(
            gh(
                "issue",
                "list",
                "--repo",
                args.repo,
                "--state",
                "open",
                "--search",
                f'in:title "{TITLE_PREFIX.strip()}"',
                "--json",
                "number,title",
                "--limit",
                "100",
            )
        )
        match = next((i for i in existing if i["title"] == title), None)
        if match:
            gh("issue", "comment", str(match["number"]), "--repo", args.repo, "--body", body)
            print(f"Commented on #{match['number']}: {title}")
        else:
            url = gh(
                "issue", "create", "--repo", args.repo, "--title", title, "--body", body
            ).strip()
            print(f"Opened {url}: {title}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("summary")
    s.add_argument("report")
    s.set_defaults(func=summary)
    i = sub.add_parser("issues")
    i.add_argument("report")
    i.add_argument("--repo", required=True)
    i.add_argument("--run-url", required=True)
    i.add_argument("--commit", required=True)
    i.set_defaults(func=issues)
    args = parser.parse_args()
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
