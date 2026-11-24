"""Command line for the Vortex triage bot.

Usage, from `.github/scripts`:
    python3 -m triage scan --repo vortex-data/vortex
    python3 -m triage render --repo vortex-data/vortex --out /tmp/triage.json
    python3 -m triage sync --repo vortex-data/vortex --channel C0123 --pings --claude
    python3 -m triage purge --channel C0123

`sync` and `purge` read `SLACK_BOT_TOKEN`. `sync` also reads `SLACK_USERS_JSON` and
`ONCALL_SLACK_ID`, and with `--claude` it needs the `anthropic` package and the federation
variables that `claude.py` reads. The repository's Actions logs are public, so these commands print
counts only, never item details or verdicts. `scan` and `render` run locally and print everything.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from . import evaluate, render
from .github import Item, closed_state, rate_limit, scan
from .hours import in_run_hours, london_date
from .rules import Assessment, assess
from .slack import Message, Slack, SlackError
from .sync import Config, Run, load


def _literal(text: str) -> str:
    """Wrap contributor-written text in a code span, so Markdown shows it and does not render it."""
    fence = "`" * (max((len(run) for run in re.findall("`+", text)), default=0) + 1)
    escaped = text.replace("|", "\\|")
    return f"{fence} {escaped} {fence}"


def _assessed(items: list[Item], now: datetime) -> list[tuple[Item, Assessment]]:
    pairs = [(item, assess(item, now)) for item in items]
    return [(item, a) for item, a in pairs if a is not None]


def _table(pairs: list[tuple[Item, Assessment]]) -> str:
    rows = [render.Row(item, a, None, None) for item, a in pairs]
    sections = Counter(row.section.title for row in rows)
    lines = [
        f"{len(rows)} tracked items: "
        + ", ".join(f"{sections[s.title]} {s.title}" for s in render.SECTIONS),
        "",
        "| # | Category | Section | Owes | Score (h) | Why | Title |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    grouped = render.ordered(rows)
    for section in render.SECTIONS:
        for row in grouped[section]:
            a = row.assessment
            cells = [
                f"[{row.item.number}]({row.item.url})",
                a.category.value,
                section.title,
                f"{a.owes.value}{' (Claude)' if a.from_claude else ''}",
                f"{a.score / 3600:.0f}",
                _literal(a.owes_reason if section is render.WAITING else a.tier_reason),
                _literal(row.item.title),
            ]
            lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _slack(channel: str) -> Slack:
    token = os.environ.get("SLACK_BOT_TOKEN", "").strip()
    if not token or not channel:
        raise SystemExit("SLACK_BOT_TOKEN and --channel are required.")
    return Slack(token, channel)


def purge(slack: Slack) -> tuple[int, int, int]:
    """Delete every message that this bot posted in the channel, including thread replies.

    Returns how many messages were deleted, how many were kept because people wrote them, and how
    many Slack refused to delete.
    """
    identity = slack.identity()
    counts = Counter[str]()

    def ours(message: dict[str, object]) -> bool:
        return message.get("user") == identity.user_id or bool(
            identity.bot_id and message.get("bot_id") == identity.bot_id
        )

    def remove(message: dict[str, object]) -> None:
        if not ours(message):
            counts["kept"] += 1
            return
        try:
            slack.delete(str(message["ts"]))
        except SlackError as error:
            # Slack refuses to delete some messages that carry the bot's user ID, and one of them
            # must not stop the purge.
            if error.code != "cant_delete_message":
                raise
            counts["refused"] += 1
        else:
            counts["deleted"] += 1

    # Collect first, because deleting while paging would move the cursor.
    for message in list(slack.history()):
        if int(str(message.get("reply_count") or 0)):
            for reply in list(slack.replies(str(message["ts"]))):
                if reply["ts"] != message["ts"]:
                    remove(reply)
        # A deleted root that still has replies stays as a tombstone, which nobody wrote.
        if message.get("subtype") != "tombstone":
            remove(message)
    return counts["deleted"], counts["kept"], counts["refused"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="triage", description=(__doc__ or "").splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)

    scan_cmd = commands.add_parser("scan", help="Print the tracked items as a Markdown table.")
    scan_cmd.add_argument("--repo", required=True, help="owner/name")

    render_cmd = commands.add_parser(
        "render", help="Write the overview and cards as Block Kit JSON."
    )
    render_cmd.add_argument("--repo", required=True, help="owner/name")
    render_cmd.add_argument("--out", required=True, help="Path of the JSON file to write")

    sync_cmd = commands.add_parser("sync", help="Bring the Slack channel up to date.")
    sync_cmd.add_argument("--repo", required=True, help="owner/name")
    sync_cmd.add_argument("--channel", required=True, help="Slack channel ID")
    sync_cmd.add_argument("--pings", action="store_true", help="Send reminders that mention people")
    sync_cmd.add_argument("--claude", action="store_true", help="Ask Claude who owes the next move")
    sync_cmd.add_argument(
        "--run-hours-only",
        action="store_true",
        help="Do nothing outside the run hours, as a scheduled run",
    )

    purge_cmd = commands.add_parser(
        "purge", help="Delete every message this bot posted in the channel."
    )
    purge_cmd.add_argument("--channel", required=True, help="Slack channel ID")

    args = parser.parse_args(argv)
    now = datetime.now(UTC)

    if args.command == "purge":
        deleted, kept, refused = purge(_slack(args.channel))
        print(
            f"Deleted {deleted} bot messages. Kept {kept} messages that people wrote. "
            f"Slack refused to delete {refused}."
        )
        return 0

    if args.command == "sync" and args.run_hours_only and not in_run_hours(now):
        # The check comes before the scan, so a skipped run spends no GitHub or Slack requests.
        print("Outside the run hours, so this run does nothing.")
        return 0

    items = scan(args.repo)

    if args.command == "scan":
        print(_table(_assessed(items, now)))
        return 0

    if args.command == "render":
        pairs = _assessed(items, now)
        rows = [render.Row(item, a, None, None) for item, a in pairs]
        content = {
            "overview": render.overview(rows, london_date(now), now, "on-call", "Claude off"),
            "cards": [render.card(row, now) for row in rows],
            "instructions": render.instructions(),
        }
        text = json.dumps({k: _dump(v) for k, v in content.items()}, indent=2, ensure_ascii=False)
        Path(args.out).write_text(text + "\n", encoding="utf-8")
        print(f"Wrote the overview and {len(rows)} cards to {args.out}")
        return 0

    slack = _slack(args.channel)
    channel = load(slack.history())
    ask = None
    if args.claude:
        # Only this path needs the `anthropic` package, so `scan`, `render`, and `purge` run
        # without it.
        from . import claude

        ask = claude.ask
    verdicts, status = evaluate.judge(items, channel, now, ask)
    config = Config(
        oncall=os.environ.get("ONCALL_SLACK_ID", "").strip() or None,
        users=evaluate.parse_slack_users(os.environ.get("SLACK_USERS_JSON", "")),
        pings=args.pings,
        claude=status,
        verdicts=verdicts,
    )
    run = Run(slack, config, now, lambda number: closed_state(args.repo, number), channel)
    failures = run.execute(items)
    print(f"Synced {len(items)} open items and {len(verdicts)} new verdicts.")
    print(f"{len(failures)} failures.")
    try:
        remaining, limit = rate_limit()
        print(f"GitHub GraphQL budget: {remaining} of {limit} points left this hour.")
    except Exception as error:
        # Every workflow in the repository shares this budget. The line only informs, so it must
        # never fail a run.
        print(f"Could not read the GitHub GraphQL budget: {type(error).__name__}")
    if failures:
        # Item numbers and Slack error codes only, which are already public.
        print("\n".join(failures), file=sys.stderr)
    return 1 if failures else 0


def _dump(value: object) -> object:
    if isinstance(value, list):
        return [_dump(v) for v in value]
    if isinstance(value, Message):
        return {
            "text": value.text,
            "blocks": list(value.blocks),
            "attachments": list(value.attachments),
        }
    return value


if __name__ == "__main__":
    sys.exit(main())
