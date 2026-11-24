"""The state that the bot keeps in Slack message metadata.

Each item root carries its item's state, and the overview carries the overview's state. A human
reply is only a request: the bot applies it, confirms it in the thread, and records the result
here, so deleting replies changes nothing. The pilot used the event type `vortex_triage`, which
this version ignores.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from .rules import Owes, Verdict

ITEM_EVENT = "vortex_triage_item"
OVERVIEW_EVENT = "vortex_triage_overview"
HELP_EVENT = "vortex_triage_help"
# Marks every reply the bot posts, so a later run can find a reply by its key.
REPLY_EVENT = "vortex_triage_reply"

# Raise this when a change makes old metadata unreadable. Messages with another version are ignored.
SCHEMA = "1"


class Status(StrEnum):
    OPEN = "open"
    CLOSED = "closed"
    MERGED = "merged"
    DELETED = "deleted"
    # A teammate answered the external comment on a team item.
    ANSWERED = "answered"
    # The item is open but no longer qualifies, for example after its `fuzzer` label was removed.
    UNTRACKED = "untracked"

    @property
    def final(self) -> bool:
        """An item that is closed, merged, or deleted never changes again."""
        return self in (Status.CLOSED, Status.MERGED, Status.DELETED)


@dataclass(frozen=True)
class ItemState:
    number: int
    status: Status = Status.OPEN
    # Unix times. Zero means never.
    reminded_at: float = 0.0
    snoozed_at: float = 0.0
    snooze_until: float = 0.0
    snoozed_by: str = ""
    # The Slack timestamp of the newest reply that the bot has handled.
    read_until: str = ""
    # The newest contributor activity that the thread already reports.
    seen_external_at: str = ""
    # A hash of the rendered card, so the bot edits the root only when it changes.
    card: str = ""
    verdict: Verdict | None = None

    def snoozed(self, now: datetime) -> bool:
        return self.snooze_until > now.timestamp()

    def unsnoozed(self) -> ItemState:
        return replace(self, snoozed_at=0.0, snooze_until=0.0, snoozed_by="")

    def payload(self) -> dict[str, str]:
        out = {
            "schema": SCHEMA,
            "number": str(self.number),
            "status": self.status.value,
            "reminded_at": _num(self.reminded_at),
            "snoozed_at": _num(self.snoozed_at),
            "snooze_until": _num(self.snooze_until),
            "snoozed_by": self.snoozed_by,
            "read_until": self.read_until,
            "seen_external_at": self.seen_external_at,
            "card": self.card,
        }
        if self.verdict is not None:
            out |= {
                "verdict_owes": self.verdict.owes.value,
                "verdict_reason": self.verdict.reason,
                "verdict_activity_at": self.verdict.activity_at,
                "verdict_checked_at": _num(self.verdict.checked_at.timestamp()),
            }
        return out

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> ItemState | None:
        """Read state from metadata, or return None if it is from another schema or malformed."""
        if payload.get("schema") != SCHEMA:
            return None
        try:
            verdict = None
            if payload.get("verdict_owes"):
                verdict = Verdict(
                    owes=Owes(payload["verdict_owes"]),
                    reason=str(payload.get("verdict_reason", "")),
                    activity_at=str(payload.get("verdict_activity_at", "")),
                    checked_at=datetime.fromtimestamp(float(payload["verdict_checked_at"]), UTC),
                )
            return cls(
                number=int(payload["number"]),
                status=Status(payload.get("status", Status.OPEN.value)),
                reminded_at=float(payload.get("reminded_at") or 0),
                snoozed_at=float(payload.get("snoozed_at") or 0),
                snooze_until=float(payload.get("snooze_until") or 0),
                snoozed_by=str(payload.get("snoozed_by", "")),
                read_until=str(payload.get("read_until", "")),
                seen_external_at=str(payload.get("seen_external_at", "")),
                card=str(payload.get("card", "")),
                verdict=verdict,
            )
        except (KeyError, TypeError, ValueError):
            return None


@dataclass(frozen=True)
class OverviewState:
    # The London date, in ISO format, of the working day that this overview covers.
    day: str
    read_until: str = ""
    # The people who got a reminder on this day, comma separated, so a later reminder says "more".
    reminded: str = ""
    # True once a newer overview replaced this one.
    replaced: bool = False

    def payload(self) -> dict[str, str]:
        values = asdict(self)
        values["replaced"] = "1" if self.replaced else ""
        return {"schema": SCHEMA, **{k: str(v) for k, v in values.items()}}

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> OverviewState | None:
        if payload.get("schema") != SCHEMA or not payload.get("day"):
            return None
        names = {f.name for f in fields(cls)} - {"replaced"}
        values = {name: str(payload.get(name, "")) for name in names}
        return cls(**values, replaced=bool(payload.get("replaced")))


def _num(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".") if value else "0"


def metadata(event_type: str, payload: dict[str, str]) -> dict[str, Any]:
    return {"event_type": event_type, "event_payload": payload}


def read_metadata(message: dict[str, Any], event_type: str) -> dict[str, Any] | None:
    """Return the metadata payload of `message` if it has the given event type."""
    meta = message.get("metadata") or {}
    payload = meta.get("event_payload")
    if meta.get("event_type") != event_type or not isinstance(payload, dict):
        return None
    return payload
