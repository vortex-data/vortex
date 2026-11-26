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


def summary(args: argparse.Namespace) -> None:
    for report in load(args.report):
        print(f"### `{report['name']}`: {report['status']}")
        print(
            f"{report['tests']} tests, {report['discarded']} discarded, "
            f"{report['duration']['secs']} s, seed `{report['seed']}`"
        )
        for found in failures(report):
            failure = found.get("failure") or {}
            print(f"- {found.get('count', 1)}x `{found.get('signature', '')}`")
            if replay := replay_line(failure):
                print(f"  - replay: `{replay}`")
            if example := counterexample(failure):
                print(f"  - counterexample: `{example[:2000]}`")
        labels = report.get("labels") or {}
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
    for report in load(args.report):
        for found in failures(report):
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
