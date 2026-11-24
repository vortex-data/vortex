"""Choose the items that Claude judges, build its input, and read its verdicts.

Claude answers one question per item: does Vortex or the contributor owe the next move? Everything
else, including whether an item is tracked at all, comes from the deterministic rules.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from .github import Item
from .rules import Assessment, Category, Owes, Tier, Verdict, assess
from .sync import Channel

MAX_CANDIDATES = 30
FAILED = "⚠️ Claude failed, rules only"
_RECENT_EVENTS = 10
_REASON_LIMIT = 200

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
                },
                "required": ["number", "owes", "reason"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["verdicts"],
    "additionalProperties": False,
}


def candidates(items: list[Item], channel: Channel, now: datetime) -> list[tuple[Item, Assessment]]:
    """Return the High and Normal items whose cached verdict is missing or no longer valid."""
    chosen = []
    for item in items:
        root = channel.roots.get(item.number)
        assessment = assess(item, now, root.state.verdict if root else None)
        if (
            assessment is None
            or assessment.tier is Tier.LOW
            or assessment.category is Category.FUZZER
        ):
            continue
        if not assessment.from_claude:
            chosen.append((item, assessment))
    tiers = list(Tier)
    chosen.sort(key=lambda pair: (tiers.index(pair[1].tier), -pair[1].score))
    return chosen[:MAX_CANDIDATES]


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
                "assignees": list(item.assignees),
                "draft": item.draft,
                "created_at": item.created_at,
                "body": item.body,
                "rule_says": {"owes": assessment.owes.value, "reason": assessment.owes_reason},
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


def activity(chosen: list[tuple[Item, Assessment]]) -> dict[str, str]:
    """Map each chosen item to the latest activity that Claude sees.

    A verdict is cached with this time, so new activity makes it stale.
    """
    return {str(item.number): assessment.last_activity_at for item, assessment in chosen}


def load_verdicts(raw: str, seen: dict[str, str], now: datetime) -> dict[int, Verdict]:
    """Read Claude's structured output.

    Entries for items that were not sent, and malformed entries, are dropped, because a missing
    verdict falls back to the rules.
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
        number, owes = entry.get("number"), entry.get("owes")
        reason = " ".join(str(entry.get("reason") or "").split())[:_REASON_LIMIT]
        if isinstance(number, bool) or not isinstance(number, int) or str(number) not in seen:
            continue
        if owes not in {o.value for o in Owes} or not reason:
            continue
        verdicts[number] = Verdict(Owes(owes), reason, seen[str(number)], now.astimezone(UTC))
    return verdicts


class ClaudeError(Exception):
    """Claude gave no usable answer. The message is safe to print in a public log."""


Ask = Callable[[dict[str, Any]], str]


def judge(
    items: list[Item], channel: Channel, now: datetime, ask: Ask | None
) -> tuple[dict[int, Verdict], str]:
    """Ask Claude about this run's candidates.

    Returns the new verdicts and the Claude status that the overview shows, so a broken setup is
    visible in Slack. `ask` sends the payload to Claude and returns its JSON answer, and None turns
    Claude off. A failure falls back to the rules.
    """
    if ask is None:
        return {}, "Claude off"
    chosen = candidates(items, channel, now)
    if not chosen:
        return {}, ""
    try:
        raw = ask(payload(chosen, now))
    except Exception as error:
        # Claude is optional, so any failure falls back to the rules instead of stopping the run.
        # Only a `ClaudeError` message is safe to print, because another message can hold item text.
        reason = str(error) if isinstance(error, ClaudeError) else type(error).__name__
        print(f"Claude failed: {reason}")
        return {}, FAILED
    verdicts = load_verdicts(raw, activity(chosen), now)
    print(f"Claude judged {len(verdicts)} of {len(chosen)} items.")
    return verdicts, f"Claude checked {len(verdicts)}" if verdicts else FAILED


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
