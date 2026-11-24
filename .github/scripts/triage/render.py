"""Build every Slack message that the bot posts, as Block Kit.

Messages name the people in the Slack mapping with Slack mentions, and everyone else by GitHub
login. Slack does not notify the people that an edit mentions, so the bot posts each message with
plain names (`People.mention` off) and then edits the mentions in. Only reminders and command
replies mention people in the post itself, so they are the only messages that notify anyone.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import date, datetime
from typing import Any

from .commands import MAX_DAYS, MAX_HOURS
from .github import COMMENT, OPENED, PUSH, READY, REVIEW, Event, Item
from .rules import ALWAYS_VORTEX, BUG_LABEL, Assessment, Category, Owes, Tier, ago
from .slack import Message
from .state import ItemState, Status

# Slack allows 50 blocks in a message and 3,000 characters in a section. Keep a margin below both.
MAX_BLOCKS = 48
MAX_SECTION_CHARS = 2900
TITLE_LIMIT = 60
# Slack wraps text at the width of the reader's window, and its font gives letters different widths,
# so no character count fits every screen. In live tests, overview rows of about 75 characters fit
# on one line in a desktop window, and rows of about 95 wrapped. An item's first line keeps to this
# budget, and its title is cut to fit, but a title always keeps at least _MIN_TITLE_CHARS.
LINE_CHARS = 70
_MIN_TITLE_CHARS = 20
# A list shows each item on two lines, and the second line starts with two em spaces, about four
# normal spaces wide. A blank line keeps the entries apart.
_INDENT = "\u2003\u2003"
_ENTRY_GAP = "\n\n"
# The most item links that the daily summary lists for one person. A link to a Slack thread is about
# 80 characters, so this keeps one person's line well below the section limit.
_SUMMARY_LINKS = 25


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
# The overview lists these sections. Items that wait on the contributor go in a reply in its thread.
_OVERVIEW_SECTIONS = (HIGH, NORMAL, LOW)
# When the overview does not fit, rows are cut from these sections first. High is never cut.
_CUT_ORDER = (LOW, NORMAL)


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


def _title(item: Item, limit: int) -> str:
    title = " ".join(item.title.split())
    if len(title) > limit:
        title = title[: limit - 3].rstrip() + "..."
    return escape(title)


def _link(item: Item, limit: int = TITLE_LIMIT) -> str:
    return f"<{item.url}|#{item.number} {_title(item, limit)}>"


@dataclass(frozen=True)
class Kind:
    """The emoji in front of an item, and what it means in the instructions' key."""

    emoji: str
    meaning: str


_PR = Kind("🔀", "external PR")
_ISSUE = Kind("📝", "external issue")
_BUG = Kind("🐞", f"issue with the `{BUG_LABEL}` label, from outside or inside the team")
_FUZZER = Kind("🧪", "fuzzer issue")
_COMMENT = Kind("💬", "external comment on a team issue or PR")
KINDS = (_PR, _ISSUE, _BUG, _FUZZER, _COMMENT)


def kind_emoji(item: Item, assessment: Assessment) -> str:
    if assessment.category is Category.FUZZER:
        return _FUZZER.emoji
    if assessment.category is Category.EXTERNAL_COMMENT:
        return _COMMENT.emoji
    if item.kind == "pr":
        return _PR.emoji
    return (_BUG if BUG_LABEL in item.labels else _ISSUE).emoji


@dataclass(frozen=True)
class People:
    """How a message names people."""

    # Lowercase GitHub login to Slack user ID, for the people the bot may mention.
    users: Mapping[str, str] = field(default_factory=dict)
    # The Slack user ID of the on-call engineer, who owns every item without an assignee.
    oncall: str | None = None
    # Whether to name the people in `users` with Slack mentions instead of GitHub logins.
    mention: bool = False

    def plain(self) -> People:
        return replace(self, mention=False)

    def name(self, login: str) -> str:
        user = self.users.get(login.lower()) if self.mention else None
        return f"<@{user}>" if user else login

    def oncall_name(self) -> str:
        if not self.oncall:
            return ""
        if self.mention:
            return f"<@{self.oncall}>"
        logins = [login for login, user in self.users.items() if user == self.oncall]
        # Without a login, the Slack ID still says who is on call.
        return logins[0] if logins else self.oncall

    def owners(self, item: Item) -> str:
        if item.assignees:
            return ", ".join(self.name(login) for login in item.assignees)
        return f"{self.oncall_name()} (on call)" if self.oncall else "unassigned"


def _contributor(item: Item, assessment: Assessment) -> str:
    """Name the outside person who made the item need Vortex."""
    if assessment.category in ALWAYS_VORTEX:
        # A bot or a teammate opened the issue, so nobody outside the team is involved.
        return ""
    if assessment.category is Category.EXTERNAL_COMMENT:
        return next((e.actor for e in reversed(item.events) if not e.team), "")
    return item.author


def date_token(moment: datetime) -> str:
    """Format a time that Slack shows in each reader's own timezone."""
    fallback = moment.strftime("%a %d %b %H:%M UTC")
    return f"<!date^{int(moment.timestamp())}^{{date_short_pretty}} at {{time}}|{fallback}>"


def _snoozed(row: Row, now: datetime) -> bool:
    return row.state is not None and row.state.snoozed(now)


def _wait(row: Row, now: datetime) -> str:
    state = row.state
    if state is not None and state.snoozed(now):
        return f"😴 until {date_token(datetime.fromtimestamp(state.snooze_until, now.tzinfo))}"
    since = row.assessment.owed_since
    return f"waiting {ago(since, now)}" if since else ""


def _parts(*parts: str) -> str:
    return " · ".join(part for part in parts if part)


def _thread(row: Row) -> str:
    # A word in link color reads as a link, and an emoji alone does not.
    return f"<{row.thread_url}|thread>" if row.thread_url else ""


def _first_line(row: Row, people: People) -> str:
    emoji = [kind_emoji(row.item, row.assessment)]
    login = _contributor(row.item, row.assessment)
    # An emoji is about as wide as two letters, and a space follows each one. The width counts the
    # login even when a mention shows the person's Slack name, because the bot does not know that
    # name.
    used = 3 * len(emoji) + len(f"#{row.item.number} ") + (len(f" · {login}") if login else 0)
    title_chars = max(LINE_CHARS - used, _MIN_TITLE_CHARS)
    tail = f" · {people.name(login)}" if login else ""
    return f"{' '.join(emoji)} {_link(row.item, title_chars)}{tail}"


def _entry(row: Row, now: datetime, people: People) -> str:
    """Show an item on two lines: the item and its contributor, then the team's side of it."""
    detail = escape(row.assessment.owes_reason) if row.section is WAITING else _wait(row, now)
    second = _parts(people.owners(row.item), detail, _thread(row))
    return f"{_first_line(row, people)}\n{_INDENT}{second}"


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


def _chunks(parts: list[str], separator: str = "\n") -> list[str]:
    """Join parts into section texts that each fit Slack's character limit."""
    chunks: list[str] = []
    current: list[str] = []
    size = 0
    for part in parts:
        if current and size + len(part) + len(separator) > MAX_SECTION_CHARS:
            chunks.append(separator.join(current))
            current = []
            size = 0
        current.append(part)
        size += len(part) + len(separator)
    if current:
        chunks.append(separator.join(current))
    return chunks


def _mrkdwn(text: str) -> dict[str, Any]:
    return {"type": "section", "text": {"type": "mrkdwn", "text": text}}


def _list(text: str) -> dict[str, Any]:
    # Without `expand`, Slack may hide the end of a long section behind a "see more" link, and a
    # list that spans several sections gets one link in each.
    return {**_mrkdwn(text), "expand": True}


def _context(text: str) -> dict[str, Any]:
    return {"type": "context", "elements": [{"type": "mrkdwn", "text": text}]}


def _attachments(
    sections: Sequence[Section],
    lines: dict[Section, list[str]],
    head: int,
    cut_order: Sequence[Section],
) -> list[dict[str, Any]]:
    """Lay out one colored attachment per section, within Slack's block limit.

    `head` is the number of blocks outside the attachments. When the sections do not fit, rows are
    cut from the end of each section in `cut_order`, which holds the least urgent items.
    """
    cut = {section: 0 for section in sections}

    def build() -> tuple[list[dict[str, Any]], int]:
        attachments, blocks = [], head
        for section in sections:
            if not lines[section]:
                continue
            shown = lines[section][: len(lines[section]) - cut[section]]
            if cut[section]:
                shown.append(f"_+{cut[section]} more_")
            body = [_mrkdwn(f"*{section.marker} {section.title} · {len(lines[section])}*")]
            body += [_list(chunk) for chunk in _chunks(shown, _ENTRY_GAP)]
            blocks += len(body)
            attachments.append({"color": section.color, "blocks": body})
        return attachments, blocks

    attachments, blocks = build()
    for section in cut_order:
        while blocks > MAX_BLOCKS and cut[section] < len(lines[section]):
            cut[section] += 1
            attachments, blocks = build()
    return attachments


def _shown(rows: Sequence[Row], now: datetime) -> dict[Section, list[Row]]:
    # A snoozed item leaves the overview until its snooze ends, as a closed item does, so that the
    # overview lists only what needs attention. The summary still counts snoozed items.
    return ordered([row for row in rows if not _snoozed(row, now)])


def overview(rows: Sequence[Row], day: date, now: datetime, people: People, claude: str) -> Message:
    """Build the daily overview, which lists every tracked item that is not snoozed.

    Items that wait on the contributor are only counted here. `waiting` lists them.
    """
    snoozed = sum(_snoozed(row, now) for row in rows)
    grouped = _shown(rows, now)
    counts = {section: len(members) for section, members in grouped.items()}
    label = day.strftime("%a %d %b").replace(" 0", " ")
    summary = _parts(
        f"On call {people.oncall_name()}" if people.oncall else "No on-call set",
        *(f"{s.marker} {counts[s]} {s.title}" for s in _OVERVIEW_SECTIONS),
        f"{WAITING.marker} {counts[WAITING]} waiting, in the thread",
        f"😴 {snoozed} snoozed" if snoozed else "",
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
    lines = {s: [_entry(row, now, people) for row in grouped[s]] for s in _OVERVIEW_SECTIONS}
    attachments = _attachments(_OVERVIEW_SECTIONS, lines, len(head), _CUT_ORDER)
    text = f"Vortex triage for {label}: " + ", ".join(
        f"{counts[s]} {s.title if s is not WAITING else 'waiting on contributor'}" for s in SECTIONS
    )
    return Message(text=text, blocks=tuple(head), attachments=tuple(attachments))


def waiting(rows: Sequence[Row], now: datetime, people: People) -> Message:
    """Build the reply in the overview's thread that lists the items that wait on the contributor.

    These items need nothing from their owners yet, so the list names the owners without mentions.
    """
    members = _shown(rows, now)[WAITING]
    if not members:
        return Message(
            text="Nothing waits on a contributor",
            blocks=(_mrkdwn(f"{WAITING.marker} Nothing waits on a contributor."),),
        )
    plain = people.plain()
    lines = {WAITING: [_entry(row, now, plain) for row in members]}
    attachments = _attachments((WAITING,), lines, 0, (WAITING,))
    text = f"{_count(len(members), 'item waits', 'items wait')} on the contributor"
    return Message(text=text, attachments=tuple(attachments))


def replaced_overview(day: date, link: str) -> Message:
    label = day.strftime("%a %d %b").replace(" 0", " ")
    return Message(
        text=f"Vortex triage for {label}",
        blocks=(
            _context(f"📋 Vortex triage · {label} · replaced by <{link}|the newest overview>"),
        ),
    )


def card(row: Row, now: datetime, people: People) -> Message:
    """Build the two-line card that an open item's root shows."""
    item, assessment = row.item, row.assessment
    detail = _parts(
        f"by {people.name(item.author)}",
        f"owner {people.owners(item)}",
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


_DONE, _UNTRACKED = "✅", "➖"
_ENDED = {
    Status.MERGED: (_DONE, "Merged"),
    Status.CLOSED: (_DONE, "Closed"),
    Status.DELETED: (_DONE, "Deleted or transferred"),
    Status.ANSWERED: (_DONE, "Answered by a teammate"),
    Status.UNTRACKED: (_UNTRACKED, "No longer tracked"),
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


def activity(item: Item, events: Sequence[Event], snooze_ended: bool, people: People) -> Message:
    """Report new contributor activity in one line."""
    kinds = Counter(e.kind for e in events)
    words = [
        _count(kinds[COMMENT], "comment", "comments") if kinds[COMMENT] else "",
        _count(kinds[REVIEW], "review", "reviews") if kinds[REVIEW] else "",
        _count(kinds[PUSH], "push", "pushes") if kinds[PUSH] else "",
        "ready for review" if kinds[READY] else "",
        "opened" if kinds[OPENED] else "",
    ]
    names = [people.name(actor) for actor in dict.fromkeys(e.actor for e in events)]
    actors = ", ".join(name if name.startswith("<@") else f"`{escape(name)}`" for name in names)
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


def prioritized(tier: Tier | None, user: str, was_set: bool) -> Message:
    if tier is not None:
        section = _BY_TIER[tier]
        return Message(
            text=f"{section.marker} Priority set to {section.title} by <@{user}>. "
            "It stays until someone sends `!priority auto`."
        )
    if was_set:
        return Message(text=f"🔄 <@{user}> handed the priority back to the bot.")
    return Message(text=f"🔄 <@{user}>, the bot already sets this item's priority.")


def invalid(error: str) -> Message:
    return Message(text=f"❓ {error}")


def wrong_thread() -> Message:
    return Message(
        text="❓ Commands work only in an item's thread. Open its thread link in the overview."
    )


def _mention(user: str | None) -> str:
    return f"<@{user}>" if user else "On-call (not set)"


def reminder_summary(due: Mapping[str | None, Sequence[Row]]) -> Message:
    """Build the day's one reminder in the overview's thread, which points everyone to their items.

    It mentions each person with due items once and links each of their items to its Slack thread,
    where the item's own reminder waits.
    """
    lines = []
    for user, rows in due.items():
        links = [f"<{row.thread_url or row.item.url}|#{row.item.number}>" for row in rows]
        # One person's line is one section, which cannot be split, so a long list is cut short.
        if len(links) > _SUMMARY_LINKS:
            links = [*links[:_SUMMARY_LINKS], f"+{len(links) - _SUMMARY_LINKS} more"]
        lines.append(f"{_mention(user)} · {_count(len(rows), 'item', 'items')}: {', '.join(links)}")
    header = "⏰ *Due today.* Each item's reminder is in its own thread."
    footer = "Reply `!snooze 2d` in an item's thread to pause its reminders."
    # One line per person keeps the message far below the block limit for any team size.
    blocks = [_mrkdwn(header), *(_list(chunk) for chunk in _chunks(lines)), _context(footer)]
    text = "Due today: " + ", ".join(
        f"{_mention(user)} {_count(len(rows), 'item', 'items')}" for user, rows in due.items()
    )
    return Message(text=text, blocks=tuple(blocks))


def item_reminder(row: Row, owners: Sequence[str | None], on_call: bool, now: datetime) -> Message:
    """Build the reminder in an item's own thread, which mentions the people who own the item."""
    who = ", ".join(_mention(user) for user in owners)
    if on_call and all(owners):
        who += " (on call)"
    detail = _parts(
        f"{row.section.marker} {row.section.title}",
        _wait(row, now),
        escape(row.assessment.tier_reason),
    )
    text = f"⏰ {who}, this item waits on you · {detail}"
    footer = "Reply `!snooze 2d` here to pause its reminders for 2 working days."
    return Message(text=text, blocks=(_mrkdwn(text), _context(footer)))


def instructions() -> Message:
    """Build the instructions message that people pin in the channel."""
    text = "\n".join(
        [
            "*How Vortex triage works*",
            "Every 5 minutes on weekdays, from 08:00 London to 19:00 New York, the bot reads "
            "GitHub for open external issues, external PRs, external comments on team items, "
            "fuzzer issues, and bug issues that the team or a bot opened. Each weekday it posts an "
            "overview and keeps it up to date, and each item gets its own thread here.",
            "",
            "*Sections*",
            f"{HIGH.marker} {HIGH.title}: reminded every working day",
            f"{NORMAL.marker} {NORMAL.title}: reminded every 5 working days",
            f"{LOW.marker} {LOW.title}: listed, never reminded",
            f"{WAITING.marker} {WAITING.title}: the contributor owes the next move, listed in "
            "a reply in the overview's thread",
            "",
            "*Key*",
            *(f"{kind.emoji} {kind.meaning}" for kind in KINDS),
            "😴 snoozed until the time shown on the item's card",
            f"{_DONE} merged, closed, or answered by a teammate",
            f"{_UNTRACKED} no longer tracked",
            "Each item takes two lines. The first shows the item and the outside contributor, and "
            "the item's number and title open it on GitHub. The second shows the assignees (or "
            "on-call when nobody is assigned), how long Vortex has owed a reply, and a link to the "
            "item's thread here.",
            "",
            "*Owners*",
            "To take an item, assign yourself on GitHub. "
            "Each working day, one reminder in the overview's thread mentions everyone with due "
            "items, and each due item gets a reminder in its own thread. Reminders go to the "
            "assignees, or to on-call when nobody is assigned. Messages show "
            "teammates as Slack mentions, except in the overview's waiting section, but only "
            "reminders and command replies notify anyone.",
            "",
            "*Commands*, as a reply in an item's thread:",
            "`!snooze 5h` hides the item from the overview and pauses its reminders for 5 working "
            f"hours (at most {MAX_HOURS}).",
            f"`!snooze 2d` does the same for 2 working days (at most {MAX_DAYS}).",
            "`!unsnooze` ends a snooze.",
            "`!priority high`, `!priority normal`, or `!priority low` sets the item's priority "
            "until `!priority auto` hands it back to the bot.",
            "New contributor activity ends a snooze early. The bot answers at its next run.",
            "",
            "*Working time*, which snoozes and reminders count, runs on weekdays from 09:00 "
            "London to 18:00 New York.",
        ]
    )
    return Message(text="How Vortex triage works", blocks=(_mrkdwn(text),))
