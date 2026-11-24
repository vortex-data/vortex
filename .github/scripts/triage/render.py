"""Build every Slack message that the bot posts, as Block Kit.

Cards and the overview show GitHub logins as plain text and never mention anyone, so editing them
every hour sends no notifications. Only reminders and command confirmations mention people.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

from .commands import MAX_DAYS, MAX_HOURS
from .github import COMMENT, OPENED, PUSH, READY, REVIEW, Event, Item
from .rules import BUG_LABEL, Assessment, Category, Owes, Tier, ago
from .slack import Message
from .state import ItemState, Status

# Slack allows 50 blocks in a message and 3,000 characters in a section. Keep a margin below both.
MAX_BLOCKS = 48
MAX_SECTION_CHARS = 2900
TITLE_LIMIT = 60


@dataclass(frozen=True)
class Section:
    key: str
    marker: str
    title: str
    color: str


HIGH = Section("high", "🔴", "High", "#E01E5A")
NORMAL = Section("normal", "🟡", "Normal", "#ECB22E")
LOW = Section("low", "⚪", "Low", "#A0A0A0")
WAITING = Section("waiting", "🔵", "Waiting on contributor", "#36C5F0")
SECTIONS = (HIGH, NORMAL, LOW, WAITING)
_BY_TIER = {Tier.HIGH: HIGH, Tier.NORMAL: NORMAL, Tier.LOW: LOW}
# When the overview does not fit, rows are cut from these sections first. High is never cut.
_CUT_ORDER = (LOW, WAITING, NORMAL)


@dataclass(frozen=True)
class Row:
    """One tracked item, with everything a message needs to show it."""

    item: Item
    assessment: Assessment
    state: ItemState | None
    # A link to the item's Slack thread, or None if the item has no root.
    thread_url: str | None

    @property
    def section(self) -> Section:
        if self.assessment.owes is Owes.CONTRIBUTOR:
            return WAITING
        return _BY_TIER[self.assessment.tier]


def escape(text: str) -> str:
    # Collapse newlines so they cannot break `<url|label>`, escape Slack's control characters, and
    # break up triple backticks so a contributor's title cannot start a code block.
    cleaned = " ".join(text.split()).replace("```", "` ` `")
    return cleaned.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _title(item: Item) -> str:
    title = " ".join(item.title.split())
    if len(title) > TITLE_LIMIT:
        title = title[: TITLE_LIMIT - 3].rstrip() + "..."
    return escape(title)


def _link(item: Item) -> str:
    return f"<{item.url}|#{item.number} {_title(item)}>"


def kind_emoji(item: Item, assessment: Assessment) -> str:
    if assessment.category is Category.FUZZER:
        return "🧪"
    if assessment.category is Category.EXTERNAL_COMMENT:
        return "💬"
    if item.kind == "pr":
        return "🔀"
    return "🐞" if BUG_LABEL in item.labels else "📝"


def _owner(item: Item) -> str:
    return ", ".join(item.assignees) if item.assignees else "unassigned"


def date_token(moment: datetime) -> str:
    """Format a time that Slack shows in each reader's own timezone."""
    fallback = moment.strftime("%a %d %b %H:%M UTC")
    return f"<!date^{int(moment.timestamp())}^{{date_short_pretty}} at {{time}}|{fallback}>"


def _wait(row: Row, now: datetime) -> str:
    state = row.state
    if state is not None and state.snoozed(now):
        return f"😴 until {date_token(datetime.fromtimestamp(state.snooze_until, now.tzinfo))}"
    since = row.assessment.owed_since
    return f"waiting {ago(since, now)}" if since else ""


def _parts(*parts: str) -> str:
    return " · ".join(part for part in parts if part)


def _thread(row: Row) -> str:
    return f"<{row.thread_url}|🧵>" if row.thread_url else ""


def _overview_line(row: Row, now: datetime) -> str:
    detail = (
        escape(row.assessment.owes_reason)
        if row.section is WAITING
        else _wait(row, now).removeprefix("waiting ")
    )
    return _parts(
        f"{kind_emoji(row.item, row.assessment)} {_link(row.item)}",
        _owner(row.item),
        detail,
        _thread(row),
    )


def ordered(rows: Sequence[Row]) -> dict[Section, list[Row]]:
    """Group rows by section. Each section lists the highest score first."""
    tiers = list(Tier)
    grouped: dict[Section, list[Row]] = {section: [] for section in SECTIONS}
    for row in rows:
        grouped[row.section].append(row)
    for section, members in grouped.items():
        if section is WAITING:
            members.sort(key=lambda r: (tiers.index(r.assessment.tier), -r.assessment.score))
        else:
            members.sort(key=lambda r: -r.assessment.score)
    return grouped


def _chunks(lines: list[str]) -> list[str]:
    """Join lines into section texts that each fit Slack's character limit."""
    chunks: list[str] = []
    current: list[str] = []
    size = 0
    for line in lines:
        if current and size + len(line) + 1 > MAX_SECTION_CHARS:
            chunks.append("\n".join(current))
            current = []
            size = 0
        current.append(line)
        size += len(line) + 1
    if current:
        chunks.append("\n".join(current))
    return chunks


def _mrkdwn(text: str) -> dict[str, Any]:
    return {"type": "section", "text": {"type": "mrkdwn", "text": text}}


def _context(text: str) -> dict[str, Any]:
    return {"type": "context", "elements": [{"type": "mrkdwn", "text": text}]}


def overview(rows: Sequence[Row], day: date, now: datetime, oncall: str, claude: str) -> Message:
    """Build the daily overview, which lists every tracked item in every section."""
    grouped = ordered(rows)
    counts = {section: len(members) for section, members in grouped.items()}
    label = day.strftime("%a %d %b").replace(" 0", " ")
    summary = _parts(
        f"On call {oncall}" if oncall else "No on-call set",
        *(f"{s.marker} {counts[s]} {s.title if s is not WAITING else 'waiting'}" for s in SECTIONS),
        claude,
        f"updated {date_token(now)}",
    )
    head = [
        {
            "type": "header",
            "text": {"type": "plain_text", "text": f"📋 Vortex triage · {label}", "emoji": True},
        },
        _context(summary),
    ]
    lines = {
        section: [_overview_line(row, now) for row in members]
        for section, members in grouped.items()
    }
    cut = {section: 0 for section in SECTIONS}

    def build() -> tuple[list[dict[str, Any]], int]:
        attachments, blocks = [], len(head)
        for section in SECTIONS:
            if not counts[section]:
                continue
            shown = lines[section][: len(lines[section]) - cut[section]]
            if cut[section]:
                shown.append(f"_+{cut[section]} more_")
            body = [_mrkdwn(f"*{section.marker} {section.title} · {counts[section]}*")]
            body += [_mrkdwn(chunk) for chunk in _chunks(shown)]
            blocks += len(body)
            attachments.append({"color": section.color, "blocks": body})
        return attachments, blocks

    attachments, blocks = build()
    for section in _CUT_ORDER:
        while blocks > MAX_BLOCKS and cut[section] < len(lines[section]):
            cut[section] += 1
            attachments, blocks = build()
    text = f"Vortex triage for {label}: " + ", ".join(
        f"{counts[s]} {s.title if s is not WAITING else 'waiting on contributor'}" for s in SECTIONS
    )
    return Message(text=text, blocks=tuple(head), attachments=tuple(attachments))


def replaced_overview(day: date, link: str) -> Message:
    label = day.strftime("%a %d %b").replace(" 0", " ")
    return Message(
        text=f"Vortex triage for {label}",
        blocks=(
            _context(f"📋 Vortex triage · {label} · replaced by <{link}|the newest overview>"),
        ),
    )


def card(row: Row, now: datetime) -> Message:
    """Build the two-line card that an open item's root shows."""
    item, assessment = row.item, row.assessment
    detail = _parts(
        f"by {item.author}",
        f"owner {_owner(item)}",
        _wait(row, now),
        # A reason can come from Claude, whose input contributors write, so it is escaped too.
        escape(assessment.owes_reason if row.section is WAITING else assessment.tier_reason),
    )
    return Message(
        text=f"#{item.number} {escape(item.title)}",
        blocks=(
            _mrkdwn(f"{row.section.marker} {kind_emoji(item, assessment)} *{_link(item)}*"),
            _context(detail),
        ),
    )


_ENDED = {
    Status.MERGED: ("✅", "Merged"),
    Status.CLOSED: ("✅", "Closed"),
    Status.DELETED: ("✅", "Deleted or transferred"),
    Status.ANSWERED: ("✅", "Answered by a teammate"),
    Status.UNTRACKED: ("➖", "No longer tracked"),
}


def ended_card(number: int, status: Status, title: str, url: str) -> Message:
    """Build the card of an item that the bot no longer follows."""
    marker, word = _ENDED[status]
    label = f"<{url}|#{number} {escape(title)}>" if url else f"#{number}"
    return Message(
        text=f"#{number} {word}", blocks=(_mrkdwn(f"{marker} *{label}*"), _context(word))
    )


def _count(n: int, singular: str, plural: str) -> str:
    return f"{n} {singular if n == 1 else plural}"


def activity(item: Item, events: Sequence[Event], snooze_ended: bool) -> Message:
    """Report new contributor activity in one line."""
    kinds = Counter(e.kind for e in events)
    words = [
        _count(kinds[COMMENT], "comment", "comments") if kinds[COMMENT] else "",
        _count(kinds[REVIEW], "review", "reviews") if kinds[REVIEW] else "",
        _count(kinds[PUSH], "push", "pushes") if kinds[PUSH] else "",
        "ready for review" if kinds[READY] else "",
        "opened" if kinds[OPENED] else "",
    ]
    actors = ", ".join(f"`{escape(actor)}`" for actor in dict.fromkeys(e.actor for e in events))
    what = ", ".join(word for word in words if word)
    text = f"💬 New activity from {actors}: {what} · <{item.url}|GitHub>"
    if snooze_ended:
        text += " · snooze ended"
    return Message(text=text)


def snoozed(until: datetime, user: str) -> Message:
    when = date_token(until)
    return Message(
        text=f"😴 Snoozed until {when} by <@{user}>. New contributor activity ends it early."
    )


def unsnoozed(user: str, was_snoozed: bool) -> Message:
    if was_snoozed:
        return Message(text=f"🔔 Snooze ended by <@{user}>.")
    return Message(text=f"🔔 <@{user}>, this item is not snoozed, so there is nothing to end.")


def invalid(error: str) -> Message:
    return Message(text=f"❓ {error}")


def wrong_thread() -> Message:
    return Message(
        text="❓ Commands work only in an item's thread. Open the item's 🧵 link and reply there."
    )


def reminder(user: str | None, rows: Sequence[Row], first: bool, now: datetime) -> Message:
    """Build one person's reminder, which lists their due items with links."""
    who = f"<@{user}>" if user else "On-call (not set)"
    count = _count(len(rows), "item waits", "items wait")
    header = (
        f"⏰ {who}, {count} on you today"
        if first
        else f"⏰ {who}, {_count(len(rows), 'more item is', 'more items are')} due"
    )
    lines = [
        _parts(
            f"{row.section.marker} {kind_emoji(row.item, row.assessment)} {_link(row.item)}",
            _wait(row, now).removeprefix("waiting "),
            _thread(row),
        )
        for row in rows
    ]
    footer = "Reply `!snooze 2d` in an item's thread to pause its reminders."
    # The header and the footer take two blocks. Rows that do not fit are cut from the end, which
    # holds the least urgent items.
    shown, cut = lines, 0
    while 2 + len(_chunks(shown)) > MAX_BLOCKS:
        cut += 1
        shown = [*lines[: len(lines) - cut], f"_+{cut} more_"]
    blocks = [_mrkdwn(header), *(_mrkdwn(chunk) for chunk in _chunks(shown)), _context(footer)]
    return Message(text=header.removeprefix("⏰ "), blocks=tuple(blocks))


def instructions() -> Message:
    """Build the instructions message that people pin in the channel."""
    text = "\n".join(
        [
            "*How Vortex triage works*",
            "Every hour, the bot reads GitHub for open external issues, external PRs, "
            "external comments on team items, and fuzzer issues. "
            "Each weekday it posts an overview and edits it every hour.",
            "",
            "*Sections*",
            "🔴 High: reminded every working day",
            "🟡 Normal: reminded every 5 working days",
            "⚪ Low: listed, never reminded",
            "🔵 Waiting on contributor: the contributor owes the next move",
            "",
            "*Owners*",
            "To take an item, assign yourself on GitHub. "
            "Reminders go to the assignees, or to on-call when nobody is assigned.",
            "",
            "*Commands*, as a reply in an item's 🧵 thread:",
            f"`!snooze 5h` pauses reminders for 5 working hours (at most {MAX_HOURS}).",
            f"`!snooze 2d` pauses reminders for 2 working days (at most {MAX_DAYS}).",
            "`!unsnooze` ends a snooze.",
            "New contributor activity ends a snooze early. The bot answers within about an hour.",
            "",
            "*Working time* runs on weekdays from 09:00 London to 18:00 New York.",
        ]
    )
    return Message(text="How Vortex triage works", blocks=(_mrkdwn(text),))
