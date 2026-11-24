"""Choose the items that Claude judges, build its input, and read its verdicts.

Claude answers two questions per item: does Vortex or the contributor owe the next move, and how
urgent is the item? Whether an item is tracked at all always comes from the deterministic rules.

Claude looks at an item when something changed that it has not seen: the item is new to the bot,
it has new activity, or its snooze ended. An item that nobody is reminded about also gets a fresh
look every few working days, because no person looks at it either. Claude looks at an item at most
once in `COOLDOWN`, so a busy thread costs one request per hour, not one per run.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any, NamedTuple

from .github import Item
from .hours import working_days_between
from .rules import Assessment, Owes, Tier, Verdict, assess, verdict_applies
from .state import ItemState
from .sync import Channel, Judgment

MAX_CANDIDATES = 30
COOLDOWN = timedelta(hours=1)
RECHECK_WORKING_DAYS = 5
FAILED = "⚠️ Claude failed, rules only"
_RECENT_EVENTS = 10
_REASON_LIMIT = 200
_TIERS = {t.value for t in Tier}

# The JSON schema that Claude's structured output must follow.
SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "verdicts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "number": {"type": "integer"},
                    "owes": {"type": "string", "enum": [o.value for o in Owes]},
                    "reason": {"type": "string"},
                    "priority": {"type": "string", "enum": [t.value for t in Tier]},
                    "priority_reason": {"type": "string"},
                },
                "required": ["number", "owes", "reason", "priority", "priority_reason"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["verdicts"],
    "additionalProperties": False,
}


def _wants_look(state: ItemState | None, assessment: Assessment, now: datetime) -> bool:
    """Return whether Claude should look at an item, apart from the cool-down."""
    if state is None or not verdict_applies(state.verdict, assessment.last_activity_at):
        return True
    if state.asked_at < state.snooze_until <= now.timestamp():
        return True
    unreminded = assessment.tier is Tier.LOW or assessment.owes is Owes.CONTRIBUTOR
    asked = datetime.fromtimestamp(state.asked_at, UTC)
    return unreminded and working_days_between(asked, now) >= RECHECK_WORKING_DAYS


def candidates(items: list[Item], channel: Channel, now: datetime) -> list[tuple[Item, Assessment]]:
    """Return the items that Claude looks at in this run, the most urgent first.

    Each item comes with the rules' assessment alone, which Claude sees and may correct.
    """
    chosen = []
    for item in items:
        rules = assess(item, now)
        if rules is None:
            continue
        root = channel.roots.get(item.number)
        state = root.state if root else None
        if state is not None:
            if state.snoozed(now) or now.timestamp() - state.asked_at < COOLDOWN.total_seconds():
                continue
            current = assess(item, now, state.verdict, state.priority) or rules
        else:
            current = rules
        if _wants_look(state, current, now):
            chosen.append((item, rules, current))
    tiers = list(Tier)
    chosen.sort(key=lambda entry: (tiers.index(entry[2].tier), -entry[2].score))
    return [(item, rules) for item, rules, _ in chosen[:MAX_CANDIDATES]]


def held(
    items: list[Item], channel: Channel, now: datetime, asked: frozenset[int]
) -> frozenset[int]:
    """Return the items whose reminders wait until Claude sees their newest activity.

    Claude answered its last look at each of these items, so its next answer comes within the
    cool-down. Until then, the rules alone might remind someone about an item that Claude would
    not, such as one where a contributor only said thanks. An item whose last look failed is not
    held, so a broken Claude never stops reminders.
    """
    waiting = set()
    for item in items:
        root = channel.roots.get(item.number)
        rules = assess(item, now)
        if item.number in asked or root is None or rules is None:
            continue
        verdict = root.state.verdict
        if (
            verdict is not None
            and verdict.checked_at.timestamp() >= root.state.asked_at
            and not verdict_applies(verdict, rules.last_activity_at)
        ):
            waiting.add(item.number)
    return frozenset(waiting)


def payload(chosen: list[tuple[Item, Assessment]], now: datetime) -> dict[str, Any]:
    """Serialize the chosen items into Claude's input. Every text field is untrusted."""
    return {
        "evaluated_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "items": [
            {
                "number": item.number,
                "kind": item.kind,
                "category": assessment.category.value,
                "title": item.title,
                "url": item.url,
                "author": item.author,
                "labels": sorted(item.labels),
                "assignees": list(item.assignees),
                "draft": item.draft,
                "created_at": item.created_at,
                "body": item.body,
                "rule_says": {
                    "owes": assessment.owes.value,
                    "reason": assessment.owes_reason,
                    "priority": assessment.tier.value,
                    "priority_reason": assessment.tier_reason,
                },
                "activity": [
                    {
                        "at": event.at,
                        "actor": event.actor,
                        "role": "team" if event.team else "contributor",
                        "kind": event.kind,
                        "state": event.state,
                        "text": event.text,
                    }
                    for event in item.events[-_RECENT_EVENTS:]
                ],
            }
            for item, assessment in chosen
        ],
    }


class Sent(NamedTuple):
    """What Claude saw of an item: its latest activity and the rules' tier."""

    activity_at: str
    tier: Tier


def sent(chosen: list[tuple[Item, Assessment]]) -> dict[str, Sent]:
    """Map each chosen item to what Claude sees of it.

    A verdict is cached with the activity time, so new activity makes it stale.
    """
    return {
        str(item.number): Sent(assessment.last_activity_at, assessment.tier)
        for item, assessment in chosen
    }


def _clean(text: object) -> str:
    return " ".join(str(text or "").split())[:_REASON_LIMIT]


def load_verdicts(raw: str, seen: dict[str, Sent], now: datetime) -> dict[int, Verdict]:
    """Read Claude's structured output.

    Entries for items that were not sent, and malformed entries, are dropped, because a missing
    verdict falls back to the rules. A malformed priority keeps the rules' tier.
    """
    try:
        data = json.loads(raw) if raw.strip() else {}
    except ValueError:
        return {}
    entries = data.get("verdicts") if isinstance(data, dict) else None
    verdicts: dict[int, Verdict] = {}
    for entry in entries if isinstance(entries, list) else []:
        if not isinstance(entry, dict):
            continue
        number, owes, reason = entry.get("number"), entry.get("owes"), _clean(entry.get("reason"))
        if isinstance(number, bool) or not isinstance(number, int) or str(number) not in seen:
            continue
        if owes not in {o.value for o in Owes} or not reason:
            continue
        item = seen[str(number)]
        priority, tier_reason = entry.get("priority"), _clean(entry.get("priority_reason"))
        # Claude's tier is kept only when it differs from the rules' tier, so that the rules still
        # decide when they change their answer later.
        tier = Tier(priority) if priority in _TIERS and tier_reason else None
        verdicts[number] = Verdict(
            Owes(owes),
            reason,
            item.activity_at,
            now.astimezone(UTC),
            tier=tier if tier is not item.tier else None,
            tier_reason=tier_reason,
        )
    return verdicts


class ClaudeError(Exception):
    """Claude gave no usable answer. The message is safe to print in a public log."""


Ask = Callable[[dict[str, Any]], str]


def judge(items: list[Item], channel: Channel, now: datetime, ask: Ask | None) -> Judgment:
    """Ask Claude about this run's candidates.

    The result includes the status that the overview shows, so a broken setup is visible in Slack.
    `ask` sends the payload to Claude and returns its JSON answer, and None turns Claude off. A
    failure falls back to the rules.
    """
    if ask is None:
        return Judgment(on=False)
    chosen = candidates(items, channel, now)
    asked = frozenset(item.number for item, _ in chosen)
    waiting = held(items, channel, now, asked)
    if waiting:
        print(f"Reminders for {len(waiting)} items wait for Claude's next look.")
    if not chosen:
        return Judgment(held=waiting)
    try:
        raw = ask(payload(chosen, now))
    except Exception as error:
        # Claude is optional, so any failure falls back to the rules instead of stopping the run.
        # Only a `ClaudeError` message is safe to print, because another message can hold item text.
        reason = str(error) if isinstance(error, ClaudeError) else type(error).__name__
        print(f"Claude failed: {reason}")
        return Judgment(status=FAILED, asked=asked, held=waiting)
    verdicts = load_verdicts(raw, sent(chosen), now)
    print(f"Claude judged {len(verdicts)} of {len(chosen)} items.")
    status = f"Claude checked {len(verdicts)}" if verdicts else FAILED
    return Judgment(status=status, verdicts=verdicts, asked=asked, held=waiting)


def parse_slack_users(raw: str) -> dict[str, str]:
    """Parse a JSON object that maps GitHub logins to Slack user IDs."""
    if not raw.strip():
        return {}
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("SLACK_USERS_JSON must be a JSON object")
    return {
        str(login).strip().lower(): str(user).strip()
        for login, user in data.items()
        if str(login).strip() and str(user).strip()
    }
