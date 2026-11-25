"""Parse the commands that people send as replies in an item's Slack thread.

A reply is a command only if it starts with `!`, after any leading mentions, so ordinary
conversation in a thread never triggers the bot.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime

from .hours import add_working_days, add_working_hours, morning, working_days_between
from .rules import Tier

MAX_HOURS = 168
MAX_DAYS = 60

USAGE = (
    f"`!snooze 5h` (working hours, at most {MAX_HOURS}), "
    f"`!snooze 2d` (working days, at most {MAX_DAYS}), `!snooze YYYY-MM-DD` (until that "
    "morning), `!unsnooze`, "
    "or `!priority high`, `normal`, `low`, or `auto`"
)
# The word that hands the priority back to the bot.
AUTO = "auto"
_LEVELS = ", ".join(f"`{level}`" for level in (*(t.value for t in Tier), AUTO))

_MENTIONS = re.compile(r"^(?:<@[A-Z0-9]+>\s*)+")
_DURATION = re.compile(r"^(\d+)\s*([a-z]*)$")
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_UNITS = {
    "h": "h",
    "hr": "h",
    "hrs": "h",
    "hour": "h",
    "hours": "h",
    "d": "d",
    "day": "d",
    "days": "d",
}
_ECHO_LIMIT = 30


@dataclass(frozen=True)
class Snooze:
    amount: int
    # "h" for working hours or "d" for working days.
    unit: str


@dataclass(frozen=True)
class SnoozeUntil:
    # The snooze ends when the window opens on this London date, or on the next weekday after it.
    day: date


@dataclass(frozen=True)
class Unsnooze:
    pass


@dataclass(frozen=True)
class Priority:
    # The tier that the item keeps until someone changes it, or None to let the bot decide again.
    tier: Tier | None


@dataclass(frozen=True)
class Invalid:
    """A reply that starts with `!` but is not a valid command. `error` says what to fix."""

    error: str


Command = Snooze | SnoozeUntil | Unsnooze | Priority | Invalid


def _echo(text: str) -> str:
    """Quote what the person typed inside a code span, without letting it break the span."""
    cleaned = " ".join(text.replace("`", "'").split())
    if len(cleaned) > _ECHO_LIMIT:
        cleaned = cleaned[: _ECHO_LIMIT - 3] + "..."
    return f"`{cleaned}`"


def parse(text: str) -> Command | None:
    """Return the command in a reply, or None when the reply is not a command."""
    body = _MENTIONS.sub("", text.strip()).strip()
    if not body.startswith("!"):
        return None
    words = body[1:].split()
    if not words:
        return Invalid(f"A command needs a name after `!`. Use {USAGE}.")
    name, args = words[0].lower(), words[1:]
    if name == "snooze":
        return _snooze(" ".join(args))
    if name == "unsnooze":
        if args:
            return Invalid(f"`!unsnooze` takes nothing after it, but got {_echo(' '.join(args))}.")
        return Unsnooze()
    if name == "priority":
        return _priority(" ".join(args).lower())
    return Invalid(f"{_echo('!' + name)} is not a command. Use {USAGE}.")


def _priority(argument: str) -> Command:
    if not argument:
        return Invalid(f"`!priority` needs a level: {_LEVELS}, for example `!priority high`.")
    if argument == AUTO:
        return Priority(None)
    if argument not in {t.value for t in Tier}:
        return Invalid(f"{_echo(argument)} is not a priority. Use {_LEVELS}.")
    return Priority(Tier(argument))


def _snooze(argument: str) -> Command:
    if not argument:
        return Invalid(
            "`!snooze` needs a duration or a date, for example `!snooze 5h`, `!snooze 2d`, or "
            "`!snooze YYYY-MM-DD`."
        )
    if _DATE.match(argument):
        try:
            return SnoozeUntil(date.fromisoformat(argument))
        except ValueError:
            return Invalid(f"{_echo(argument)} is not a date in the calendar.")
    match = _DURATION.match(argument.lower())
    if match is None:
        return Invalid(
            f"{_echo(argument)} is not a duration or a date. Use a whole number and a unit, "
            "for example `!snooze 5h` or `!snooze 2d`, or a date as `YYYY-MM-DD`."
        )
    try:
        amount = int(match.group(1))
    except ValueError:
        return Invalid(f"{_echo(argument)} is too large to use as a snooze duration.")
    unit_text = match.group(2)
    if not unit_text:
        return Invalid(
            f"{_echo(argument)} has no unit. Add `h` for working hours or `d` for working days, "
            f"for example `!snooze {amount}h`."
        )
    unit = _UNITS.get(unit_text)
    if unit is None:
        return Invalid(
            f"{_echo(unit_text)} is not a unit. Use `h` for working hours or `d` for working days."
        )
    if amount == 0:
        return Invalid("A snooze must last at least `1h` or `1d`. Use `!unsnooze` to end a snooze.")
    if unit == "h" and amount > MAX_HOURS:
        return Invalid(
            f"`{amount}h` is more than the limit of {MAX_HOURS} working hours. "
            f"Use days for longer snoozes, up to `{MAX_DAYS}d`."
        )
    if unit == "d" and amount > MAX_DAYS:
        return Invalid(f"`{amount}d` is more than the limit of {MAX_DAYS} working days.")
    return Snooze(amount, unit)


def snooze_end(command: Snooze | SnoozeUntil, requested: datetime) -> datetime | Invalid:
    """Return when a snooze that someone sent at `requested` ends, or why it cannot."""
    if isinstance(command, Snooze):
        if command.unit == "h":
            return add_working_hours(requested, command.amount)
        return add_working_days(requested, command.amount)
    until = morning(command.day)
    label = f"`{command.day.isoformat()}`"
    if until <= requested:
        return Invalid(f"The working day of {label} has already started. Pick a later date.")
    if working_days_between(requested, until) > MAX_DAYS:
        return Invalid(f"{label} is more than {MAX_DAYS} working days away.")
    return until
