"""Reminder cadence, Slack mention resolution, and Claude progress verdicts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any
from zoneinfo import ZoneInfo

from .github import FUZZER, Item

if TYPE_CHECKING:
    from .publish import Thread

WAITING_ON_VORTEX = "waiting_on_vortex"
WAITING_ON_CONTRIBUTOR = "waiting_on_contributor"
_VALID_STATES = frozenset({WAITING_ON_VORTEX, WAITING_ON_CONTRIBUTOR})
_REASON_LIMIT = 200


@dataclass(frozen=True)
class Cadence:
    """When reminders are allowed and how often each kind of item is pinged."""

    issue_days: int = 7
    pr_weekdays: int = 1
    tz: str = "America/New_York"
    start_hour: int = 9
    end_hour: int = 18
    ignore_hours: bool = False


@dataclass(frozen=True)
class Verdict:
    """Whether Vortex currently owes a response or action on an item."""

    state: str
    reason: str
    source: str = "fallback"


def in_business_hours(now_ts: float, cadence: Cadence) -> bool:
    if cadence.ignore_hours:
        return True
    local = datetime.fromtimestamp(now_ts, tz=ZoneInfo(cadence.tz))
    return local.weekday() < 5 and cadence.start_hour <= local.hour < cadence.end_hour


def weekdays_between(start_ts: float, end_ts: float, tz_name: str) -> int:
    """Count calendar weekdays strictly after `start_ts` up to and including `end_ts`."""
    tz = ZoneInfo(tz_name)
    start_date = datetime.fromtimestamp(start_ts, tz=tz).date()
    end_date = datetime.fromtimestamp(end_ts, tz=tz).date()
    if end_date <= start_date:
        return 0
    count = 0
    day = start_date + timedelta(days=1)
    while day <= end_date:
        if day.weekday() < 5:
            count += 1
        day += timedelta(days=1)
    return count


def is_ping_due(
    item: Item,
    thread: Thread | None,
    now_ts: float,
    cadence: Cadence,
    *,
    new_external: bool = False,
) -> bool:
    """Return True when `item` is eligible for a progress check and reminder right now."""
    if not in_business_hours(now_ts, cadence):
        return False
    if not item.reasons and (thread is None or not thread.open):
        return False
    if thread is None:
        return True
    if new_external or (item.external_at and item.external_at > thread.external_at):
        return True
    last_at = max(thread.last_ping_at, thread.last_checked_at)
    if last_at <= 0:
        return True
    if item.kind == "pr":
        return weekdays_between(last_at, now_ts, cadence.tz) >= cadence.pr_weekdays
    return (now_ts - last_at) >= cadence.issue_days * 86400


def due_items(
    items: list[Item],
    threads: dict[int, Thread],
    now_ts: float,
    cadence: Cadence,
    max_new: int | None = None,
) -> list[Item]:
    """Return the open items that are due for a progress check on this run."""
    if not in_business_hours(now_ts, cadence):
        return []
    result: list[Item] = []
    started = 0
    for item in items:
        thread = threads.get(item.number)
        if thread is None or (thread.root_ts is None and not thread.pending):
            if not item.reasons or (max_new is not None and started >= max_new):
                continue
            started += 1
        if is_ping_due(item, thread, now_ts, cadence):
            result.append(item)
    return result


def fallback_verdict(item: Item) -> Verdict:
    """Deterministic verdict used when Claude is disabled or omitted an item."""
    if FUZZER in item.reasons:
        return Verdict(WAITING_ON_VORTEX, "Open fuzzer issue.")
    if not item.assignees:
        return Verdict(WAITING_ON_VORTEX, "No GitHub assignee yet.")
    if item.external_at and (not item.team_at or item.external_at > item.team_at):
        who = item.external_by or item.author
        return Verdict(
            WAITING_ON_VORTEX, f"Latest activity is from `{who}` with no team reply after it."
        )
    return Verdict(WAITING_ON_CONTRIBUTOR, "A teammate replied after the latest external activity.")


def build_due_payload(items: list[Item], now_ts: float) -> dict[str, Any]:
    """Serialize due items into a compact, read-only JSON payload for Claude."""
    return {
        "evaluated_at": datetime.fromtimestamp(now_ts, tz=ZoneInfo("UTC")).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "items": [
            {
                "number": item.number,
                "kind": item.kind,
                "title": item.title,
                "url": item.url,
                "author": item.author,
                "created_at": item.created_at,
                "assignees": list(item.assignees),
                "reasons": list(item.reasons),
                "external_at": item.external_at,
                "external_by": item.external_by,
                "team_at": item.team_at,
                "body": item.body,
                "recent": [
                    {
                        "at": comment.at,
                        "author": comment.author,
                        "role": "team" if comment.team else "external",
                        "kind": comment.kind,
                        "state": comment.state,
                        "body": comment.body,
                    }
                    for comment in item.recent
                ],
            }
            for item in items
        ],
    }


def _clean_reason(raw: object) -> str:
    text = " ".join(str(raw or "").split())
    if len(text) <= _REASON_LIMIT:
        return text
    return text[: _REASON_LIMIT - 3].rstrip() + "..."


def _add_verdict(verdicts: dict[int, Verdict], number: object, value: object) -> None:
    if isinstance(number, bool) or not isinstance(value, dict):
        return
    if isinstance(number, int):
        item_id = number
    elif isinstance(number, str) and number.isdigit():
        item_id = int(number)
    else:
        return
    if item_id <= 0:
        return
    state = value.get("state")
    reason = _clean_reason(value.get("reason"))
    if state not in _VALID_STATES or not reason:
        return
    verdicts[item_id] = Verdict(state=state, reason=reason, source="claude")


def load_verdicts(path: Path) -> dict[int, Verdict]:
    """Load and validate Claude's JSON verdicts file. Invalid entries are ignored."""
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(raw, dict):
        return {}
    verdicts: dict[int, Verdict] = {}
    entries = raw.get("verdicts")
    if isinstance(entries, list):
        for entry in entries:
            if isinstance(entry, dict):
                _add_verdict(verdicts, entry.get("number"), entry)
        return verdicts
    for key, value in raw.items():
        _add_verdict(verdicts, key, value)
    return verdicts


def parse_slack_users(raw: str) -> dict[str, str]:
    """Parse a JSON mapping of GitHub logins to Slack user IDs."""
    if not raw.strip():
        return {}
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("SLACK_USERS_JSON must be a JSON object")
    return {
        str(login).strip().lower(): str(slack_id).strip()
        for login, slack_id in data.items()
        if str(login).strip() and str(slack_id).strip()
    }


def mention_for(item: Item, slack_users: dict[str, str], oncall_slack_id: str | None) -> str:
    """Format who should be pinged in Slack for `item`."""
    mapped = [
        f"<@{slack_users[login.lower()]}>"
        for login in item.assignees
        if login.lower() in slack_users
    ]
    if mapped:
        return " ".join(mapped)
    if oncall_slack_id:
        if item.assignees:
            gh = ", ".join(f"`{login}`" for login in item.assignees)
            return f"<@{oncall_slack_id}> (GitHub assignee: {gh})"
        return f"<@{oncall_slack_id}> (unassigned)"
    if item.assignees:
        return ", ".join(f"`{login}`" for login in item.assignees)
    return "on-call (unassigned)"
