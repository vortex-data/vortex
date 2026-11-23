"""Command line for the triage scanner and publisher.

Usage, from `.github/scripts`:
    python3 -m triage scan --repo vortex-data/vortex
    python3 -m triage publish --repo vortex-data/vortex
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter

from .github import Item, closed_state, scan
from .publish import MemoryStore, Publisher


def _table(items: list[Item]) -> str:
    kinds = Counter(item.kind for item in items)
    reasons = Counter(reason for item in items for reason in item.reasons)
    lines = [
        f"{len(items)} tracked items: {kinds['issue']} issues, {kinds['pr']} PRs. "
        + ", ".join(f"{count} {reason}" for reason, count in sorted(reasons.items())),
        "",
        "| # | Kind | Reasons | Author | Assignees | Last external | Last team | Title |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for item in items:
        cells = [
            f"[{item.number}]({item.url})",
            item.kind,
            ", ".join(item.reasons),
            item.author,
            ", ".join(item.assignees),
            (item.external_at or "")[:10],
            (item.team_at or "")[:10],
            item.title.replace("|", "\\|"),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


class _PrintOnly:
    """Stands in for Slack and prints each message instead of posting it."""

    def post(self, text: str, key: str, payload: dict[str, str], thread_ts: str | None) -> str:
        print(f"--- {key}\n{text}\n")
        return key

    def find(self, key: str, thread_ts: str | None, oldest: float) -> None:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(prog="triage", description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    for name, text in [
        ("scan", "List the open items that need tracking, as a Markdown table."),
        ("publish", "Print the Slack messages a first run would post. Posts nothing."),
    ]:
        command = commands.add_parser(name, help=text)
        command.add_argument("--repo", required=True, help="owner/name")
    args = parser.parse_args()

    items = scan(args.repo)
    if args.command == "scan":
        print(_table([item for item in items if item.reasons]))
        return 0

    # Posting for real needs a durable store, so that a later run finds the threads it started.
    publisher = Publisher(MemoryStore(), _PrintOnly(), lambda n: closed_state(args.repo, n))
    failures = publisher.sync(items)
    print("\n".join(failures), file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
