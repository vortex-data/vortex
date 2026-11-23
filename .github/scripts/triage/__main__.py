"""Command line for the triage scanner, Claude evaluator input, and Slack publisher.

Usage, from `.github/scripts`:
    python3 -m triage scan --repo vortex-data/vortex
    python3 -m triage publish --repo vortex-data/vortex
    python3 -m triage prepare --repo vortex-data/vortex --out /tmp/due.json
    python3 -m triage sync --repo vortex-data/vortex --channel C0CL8NL99GA --ping
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

from .evaluate import (
    Cadence,
    build_due_payload,
    due_items,
    load_verdicts,
    parse_slack_users,
)
from .github import Item, closed_state, scan
from .publish import MemoryStore, Messenger, Publisher, Store
from .slack import Slack, SlackStore


def _literal(text: str) -> str:
    """Wrap contributor-written text in a code span, so Markdown shows it and does not render it."""
    fence = "`" * (max((len(run) for run in re.findall("`+", text)), default=0) + 1)
    escaped = text.replace("|", "\\|")
    return f"{fence} {escaped} {fence}"


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
            _literal(item.title),
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

    def update(self, ts: str, text: str, key: str, payload: dict[str, str]) -> None:
        print(f"--- [check] {key} (ts={ts}) verdict={payload.get('verdict', '')}\n")


def _add_cadence_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--channel", default="", help="Slack channel ID")
    parser.add_argument(
        "--max-new", type=int, default=None, help="Cap new threads started in one run"
    )
    parser.add_argument("--issue-days", type=int, default=7, help="Days between issue reminders")
    parser.add_argument("--pr-weekdays", type=int, default=1, help="Weekdays between PR reminders")
    parser.add_argument("--tz", default="America/New_York", help="IANA timezone for business hours")
    parser.add_argument("--ignore-hours", action="store_true", help="Ignore business-hours window")


def _cadence(args: argparse.Namespace) -> Cadence:
    return Cadence(
        issue_days=args.issue_days,
        pr_weekdays=args.pr_weekdays,
        tz=args.tz,
        ignore_hours=args.ignore_hours,
    )


def _slack_store(channel: str) -> tuple[Store, Messenger]:
    token = os.environ.get("SLACK_BOT_TOKEN", "").strip()
    if token and channel:
        client = Slack(token, channel)
        return SlackStore(client), client
    return MemoryStore(), _PrintOnly()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="triage", description=(__doc__ or "").splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)

    scan_cmd = commands.add_parser("scan", help="List tracked open items as a Markdown table.")
    scan_cmd.add_argument("--repo", required=True, help="owner/name")

    publish_cmd = commands.add_parser(
        "publish", help="Print the Slack messages a first run would post. Posts nothing."
    )
    publish_cmd.add_argument("--repo", required=True, help="owner/name")

    prepare_cmd = commands.add_parser(
        "prepare", help="Write JSON of items due for a progress check (for Claude evaluation)."
    )
    prepare_cmd.add_argument("--repo", required=True, help="owner/name")
    prepare_cmd.add_argument("--out", required=True, help="Path to write due-items JSON")
    prepare_cmd.add_argument("--github-output", default="", help="Optional GITHUB_OUTPUT file")
    _add_cadence_args(prepare_cmd)

    sync_cmd = commands.add_parser(
        "sync", help="Sync Slack threads and send due reminders using Slack metadata as the store."
    )
    sync_cmd.add_argument("--repo", required=True, help="owner/name")
    sync_cmd.add_argument("--verdicts", default="", help="Optional JSON verdicts file from Claude")
    sync_cmd.add_argument("--ping", action="store_true", help="Send reminder pings for due items")
    sync_cmd.add_argument("--dry-run", action="store_true", help="Print messages without posting")
    _add_cadence_args(sync_cmd)

    args = parser.parse_args(argv)
    items = scan(args.repo)

    if args.command == "scan":
        print(_table([item for item in items if item.reasons]))
        return 0

    if args.command == "publish":
        publisher = Publisher(MemoryStore(), _PrintOnly(), lambda n: closed_state(args.repo, n))
        failures = publisher.sync(items)
        if failures:
            print("\n".join(failures), file=sys.stderr)
        return 1 if failures else 0

    if args.command == "prepare":
        now_ts = time.time()
        store, _ = _slack_store(args.channel)
        due = due_items(items, store.load(), now_ts, _cadence(args), max_new=args.max_new)
        payload = build_due_payload(due, now_ts)
        Path(args.out).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        if args.github_output:
            with Path(args.github_output).open("a", encoding="utf-8") as handle:
                handle.write(f"due_count={len(due)}\n")
        print(f"Prepared {len(due)} due items in {args.out}")
        return 0

    verdicts = load_verdicts(Path(args.verdicts)) if args.verdicts else {}
    slack_users = parse_slack_users(os.environ.get("SLACK_USERS_JSON", ""))
    oncall_slack_id = os.environ.get("ONCALL_SLACK_ID", "").strip() or None

    if args.dry_run:
        store, _ = _slack_store(args.channel)
        messenger: Messenger = _PrintOnly()
    else:
        token = os.environ.get("SLACK_BOT_TOKEN", "").strip()
        if not token or not args.channel:
            print(
                "SLACK_BOT_TOKEN and --channel are required unless --dry-run is set.",
                file=sys.stderr,
            )
            return 2
        client = Slack(token, args.channel)
        store, messenger = SlackStore(client), client

    publisher = Publisher(store, messenger, lambda n: closed_state(args.repo, n))
    failures = publisher.sync(
        items,
        max_new=args.max_new,
        ping=args.ping,
        verdicts=verdicts,
        cadence=_cadence(args),
        slack_users=slack_users,
        oncall_slack_id=oncall_slack_id,
    )
    if failures:
        print("\n".join(failures), file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
