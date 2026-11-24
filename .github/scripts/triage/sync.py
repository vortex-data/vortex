"""One run of the triage bot: read GitHub and Slack, then bring Slack up to date.

A run is safe to repeat. Every reply carries a key, and a run that dies after a post re-reads that
thread on the next run, because the post moved the thread's `latest_reply` past the stored
`read_until`. The next run then finds the key and does not post twice.
"""

from __future__ import annotations

import hashlib
import json
import traceback
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from functools import partial
from typing import Any, Protocol

from . import render
from .commands import Invalid, Priority, Snooze, SnoozeUntil, Unsnooze, parse, snooze_end
from .github import Ended, Item
from .hours import (
    in_window,
    london_date,
    window,
    working_days_between,
)
from .rules import Owes, Tier, Verdict, assess, parse_time
from .slack import Identity, Message, SlackError, SlackUnknown
from .state import (
    HELP_EVENT,
    ITEM_EVENT,
    OVERVIEW_EVENT,
    REPLY_EVENT,
    SCHEMA,
    ItemState,
    OverviewState,
    Status,
    metadata,
    read_metadata,
)

NORMAL_REMINDER_DAYS = 5


class SlackApi(Protocol):
    channel: str

    def identity(self) -> Identity: ...

    def history(self) -> Iterable[dict[str, Any]]: ...

    def replies(self, thread_ts: str) -> Iterable[dict[str, Any]]: ...

    def post(
        self, message: Message, metadata: dict[str, Any], thread_ts: str | None = None
    ) -> str: ...

    def update(self, ts: str, message: Message, metadata: dict[str, Any]) -> None: ...

    def delete(self, ts: str) -> None: ...


@dataclass(frozen=True)
class Judgment:
    """What this run learned from Claude."""

    # False when this run does not use Claude at all.
    on: bool = True
    # The outcome of this run's request, for example "Claude checked 12", or empty without one.
    status: str = ""
    # New verdicts, which replace the cached ones.
    verdicts: Mapping[int, Verdict] = field(default_factory=dict)
    # The items that this run asked Claude about, whether or not Claude answered.
    asked: frozenset[int] = frozenset()
    # The items whose reminders wait until Claude sees their newest activity.
    held: frozenset[int] = frozenset()


@dataclass(frozen=True)
class Config:
    # The Slack user ID of the on-call engineer, who owns every item without a known assignee.
    oncall: str | None
    # Lowercase GitHub login to Slack user ID, for the people the bot may mention.
    users: dict[str, str]
    pings: bool
    claude: Judgment = field(default_factory=Judgment)


@dataclass
class Thread:
    """A top-level message that the bot posted, and the replies that this run read from it."""

    ts: str
    latest_reply: str
    # The keys of the bot's replies that this run read or posted, with each reply's timestamp.
    keys: dict[str, str] = field(default_factory=dict)
    read: bool = False


@dataclass
class Root:
    thread: Thread
    # The state this run works on, and the state that Slack holds now.
    state: ItemState
    saved: ItemState


@dataclass
class Overview:
    thread: Thread
    state: OverviewState


@dataclass
class Channel:
    roots: dict[int, Root]
    # The newest overview, which is the only one that lists items.
    overview: Overview | None
    help: tuple[str, str] | None
    # Older overviews that do not say yet that a newer one replaced them.
    stale: list[Overview] = field(default_factory=list)


def _newer(a: str, b: str) -> bool:
    return float(a or 0) > float(b or 0)


def load(messages: Iterable[dict[str, Any]]) -> Channel:
    """Rebuild the bot's state from the channel's top-level messages."""
    roots: dict[int, Root] = {}
    overviews: list[Overview] = []
    help_message: tuple[str, str] | None = None
    for message in messages:
        ts, latest = str(message["ts"]), str(message.get("latest_reply") or "")
        if (payload := read_metadata(message, ITEM_EVENT)) is not None:
            state = ItemState.from_payload(payload)
            # Duplicate roots can only come from a lost race, so the newest one wins.
            if state is not None and (
                state.number not in roots or _newer(ts, roots[state.number].thread.ts)
            ):
                roots[state.number] = Root(Thread(ts, latest), state, state)
        elif (payload := read_metadata(message, OVERVIEW_EVENT)) is not None:
            overview_state = OverviewState.from_payload(payload)
            if overview_state is not None:
                overviews.append(Overview(Thread(ts, latest), overview_state))
        elif (payload := read_metadata(message, HELP_EVENT)) is not None and payload.get(
            "schema"
        ) == SCHEMA:
            help_message = (ts, str(payload.get("card", "")))
    overviews.sort(key=lambda o: float(o.thread.ts))
    stale = [o for o in overviews[:-1] if not o.state.replaced]
    return Channel(roots, overviews[-1] if overviews else None, help_message, stale)


def _hash(message: Message) -> str:
    content = json.dumps([message.text, message.blocks, message.attachments], sort_keys=True)
    return hashlib.sha256(content.encode()).hexdigest()[:16]


class Run:
    def __init__(
        self,
        slack: SlackApi,
        config: Config,
        now: datetime,
        ended: Callable[[int], Ended | None],
        channel: Channel,
    ) -> None:
        self.slack = slack
        self.config = config
        self.now = now
        self.ended = ended
        self.channel = channel
        self.failures: list[str] = []
        # Links to the reminders that this run posted or found, by item number.
        self.reminders: dict[int, str] = {}
        self.identity = slack.identity()
        # Posts use `quiet`, and the edits after them use `people`. See the `render` module.
        self.people = render.People(config.users, config.oncall, mention=True)
        self.quiet = self.people.plain()

    def _link(self, ts: str) -> str:
        """Link to a top-level message, which Slack shows in the channel."""
        return f"{self.identity.url}archives/{self.slack.channel}/p{ts.replace('.', '')}"

    def _reply_link(self, ts: str, thread_ts: str) -> str:
        """Link to a reply, which Slack opens in its thread, in the form of `chat.getPermalink`."""
        return f"{self._link(ts)}?thread_ts={thread_ts}&cid={self.slack.channel}"

    def _requests(self, thread: Thread, read_until: str) -> tuple[list[dict[str, Any]], str]:
        """Read a thread if it changed, and return the human replies after `read_until`.

        Also returns the newest reply timestamp that this run read, which becomes the new
        `read_until`. The bot's own replies are recorded as keys and never returned as requests.
        """
        if not _newer(thread.latest_reply, read_until):
            return [], read_until
        requests, newest = [], read_until
        for reply in self.slack.replies(thread.ts):
            ts = str(reply["ts"])
            if ts == thread.ts:
                continue
            if (payload := read_metadata(reply, REPLY_EVENT)) is not None:
                thread.keys[str(payload.get("key", ""))] = ts
            elif _newer(ts, read_until) and self._human(reply):
                requests.append(reply)
            if _newer(ts, newest):
                newest = ts
        thread.read = True
        return sorted(requests, key=lambda r: float(r["ts"])), newest

    def _human(self, message: dict[str, Any]) -> bool:
        # A reply that is also sent to the channel has the `thread_broadcast` subtype. Other
        # subtypes are joins, edits, and bot messages.
        return bool(
            not message.get("bot_id")
            and message.get("subtype") in (None, "thread_broadcast")
            and message.get("user") != self.identity.user_id
        )

    def _reply(self, thread: Thread, key: str, build: Callable[[render.People], Message]) -> None:
        """Post a reply once, with plain names, and then edit the mentions in."""
        if key in thread.keys:
            return
        data = metadata(REPLY_EVENT, {"key": key})
        quiet = build(self.quiet)
        ts = self.slack.post(quiet, data, thread.ts)
        # The key goes in before the edit, so a failed edit leaves plain names and no second post.
        thread.keys[key] = ts
        loud = build(self.people)
        if loud != quiet:
            self.slack.update(ts, loud, data)

    def execute(self, items: list[Item]) -> list[str]:
        """Bring Slack up to date with the open `items`, and return what failed."""
        self._attempt("instructions", self._instructions)
        rows: dict[int, render.Row] = {}
        open_numbers = {item.number for item in items}
        for item in items:
            self._attempt(f"#{item.number}", lambda: self._item(item, rows))
        for number, root in self.channel.roots.items():
            if number not in open_numbers and not root.state.status.final:
                self._attempt(f"#{number}", lambda: self._end(number, root))
        self._attempt("overview", lambda: self._overview(list(rows.values())))
        for number, row in rows.items():
            self._attempt(f"#{number}", lambda: self._save(row))
        return self.failures

    def _attempt(self, what: str, action: Callable[[], None]) -> None:
        # One failed item must not stop the rest of the run. The run still fails at the end, which
        # posts a failure note in the channel.
        try:
            action()
        except (SlackError, SlackUnknown) as error:
            self.failures.append(f"{what}: {error}")
        except Exception as error:
            # The message of an unexpected error can hold item text, and the log is public, so
            # report only the error type and where it was raised.
            frame = traceback.extract_tb(error.__traceback__)[-1]
            where = f"{frame.filename.rsplit('/', 1)[-1]}:{frame.lineno}"
            self.failures.append(f"{what}: {type(error).__name__} at {where}")

    def _instructions(self) -> None:
        message = render.instructions()
        payload = {"schema": SCHEMA, "card": _hash(message)}
        if self.channel.help is None:
            self.slack.post(message, metadata(HELP_EVENT, payload))
        elif self.channel.help[1] != payload["card"]:
            self.slack.update(self.channel.help[0], message, metadata(HELP_EVENT, payload))

    def _item(self, item: Item, rows: dict[int, render.Row]) -> None:
        root = self.channel.roots.get(item.number)
        previous = root.state.verdict if root else None
        # This run's verdict wins over the cached one, which `assess` drops if it is stale.
        verdict = self.config.claude.verdicts.get(item.number) or previous
        assessment = assess(item, self.now, verdict, root.state.priority if root else None)
        asked_at = root.state.asked_at if root else 0.0
        if item.number in self.config.claude.asked:
            asked_at = self.now.timestamp()
        if assessment is None:
            if root is not None and root.state.status is Status.OPEN:
                # A teammate answered the external comment on a team item. Otherwise the item
                # lost the label that made it qualify, or everyone outside the team who took part
                # has joined the team list.
                answered = item.author_team and any(not e.team for e in item.events)
                status = Status.ANSWERED if answered else Status.UNTRACKED
                state = root.state = root.saved = replace(root.state, status=status)
                self.slack.update(
                    root.thread.ts,
                    render.ended_card(item.number, status, item.title, item.url),
                    metadata(ITEM_EVENT, state.payload()),
                )
            return
        if root is None:
            # The row goes in first, so that a failed post leaves the item in the overview.
            row = rows[item.number] = render.Row(item, assessment, None, None)
            # Every item gets a root, so that people can snooze or reprioritize it in its thread.
            state = ItemState(
                item.number,
                seen_external_at=assessment.last_external_at or "",
                verdict=verdict,
                asked_at=asked_at,
            )
            # `_save` edits the mentions in later in this run.
            message = render.card(replace(row, state=state), self.now, self.quiet)
            state = replace(state, card=_hash(message))
            ts = self.slack.post(message, metadata(ITEM_EVENT, state.payload()))
            root = Root(Thread(ts, ""), state, state)
            self.channel.roots[item.number] = root
            rows[item.number] = render.Row(item, assessment, state, self._link(ts))
            self._claude_moved(root.thread, item, None, verdict, None)
            return

        state = replace(root.state, status=Status.OPEN, verdict=verdict, asked_at=asked_at)
        link = self._link(root.thread.ts)
        # A failed Slack call below must cost only this item's thread work, not its place in the
        # overview and the reminders. A retry on the next run cannot post twice, because every
        # reply carries a key.
        rows[item.number] = render.Row(item, assessment, state, link)
        requests, read_until = self._requests(root.thread, state.read_until)
        for request in requests:
            state = self._command(root.thread, state, request)
        state = replace(state, read_until=read_until)
        if state.priority != root.state.priority:
            # The category does not depend on the priority, so the item is still tracked.
            assessment = assess(item, self.now, verdict, state.priority) or assessment

        last_external = assessment.last_external_at
        if last_external and last_external > state.seen_external_at:
            new = [e for e in item.events if not e.team and e.at > state.seen_external_at]
            wake = state.snoozed(self.now) and any(
                parse_time(e.at).timestamp() > state.snoozed_at for e in new
            )
            self._reply(
                root.thread, f"activity:{last_external}", partial(render.activity, item, new, wake)
            )
            if wake:
                state = state.unsnoozed()
            state = replace(state, seen_external_at=last_external)
        # A failed note leaves the old verdict and ask time in Slack, so a later run asks Claude
        # again and retries the note.
        self._claude_moved(root.thread, item, previous, verdict, state.priority)
        root.state = state
        rows[item.number] = render.Row(item, assessment, state, link)

    def _claude_moved(
        self,
        thread: Thread,
        item: Item,
        previous: Verdict | None,
        verdict: Verdict | None,
        priority: Tier | None,
    ) -> None:
        """Say in the item's thread when a new verdict changes Claude's choice of tier.

        Each change gets a note, so that the thread shows every tier that Claude chose and why.
        """
        if verdict is None or item.number not in self.config.claude.verdicts:
            return
        rules = assess(item, self.now)
        old = previous.tier if previous is not None else None
        if rules is None or verdict.tier is old:
            return
        before, after = old or rules.tier, verdict.tier or rules.tier
        if before is after:
            return
        note = render.claude_moved(
            before, after, verdict.tier is None, verdict.tier_reason, priority
        )
        key = f"claude:{verdict.checked_at.timestamp():.0f}"
        self._reply(thread, key, lambda _people: note)

    def _command(self, thread: Thread, state: ItemState, request: dict[str, Any]) -> ItemState:
        command = parse(str(request.get("text") or ""))
        if command is None:
            return state
        user = str(request.get("user") or "")
        requested = datetime.fromtimestamp(float(request["ts"]), UTC)
        if isinstance(command, Snooze | SnoozeUntil):
            until = snooze_end(command, requested)
            if isinstance(until, Invalid):
                reply = render.invalid(until.error)
            else:
                state = replace(
                    state,
                    snoozed_at=requested.timestamp(),
                    snooze_until=until.timestamp(),
                    snoozed_by=user,
                )
                reply = render.snoozed(until, user)
        elif isinstance(command, Unsnooze):
            reply = render.unsnoozed(user, state.snoozed(self.now))
            state = state.unsnoozed()
        elif isinstance(command, Priority):
            reply = render.prioritized(command.tier, user, state.priority is not None)
            state = replace(state, priority=command.tier, priority_by=user if command.tier else "")
        else:
            assert isinstance(command, Invalid)
            reply = render.invalid(command.error)
        self._reply(thread, f"command:{request['ts']}", lambda _people: reply)
        return state

    def _end(self, number: int, root: Root) -> None:
        ended = self.ended(number)
        if ended is None:
            # The item reopened after the scan. The next run sees it as open.
            return
        status = Status(ended.state)
        root.state = root.saved = replace(root.state, status=status)
        self.slack.update(
            root.thread.ts,
            render.ended_card(number, status, ended.title, ended.url),
            metadata(ITEM_EVENT, root.state.payload()),
        )

    def _save(self, row: render.Row) -> None:
        """Write a root's card and state, if either differs from what Slack holds."""
        root = self.channel.roots.get(row.item.number)
        if root is None:
            return
        message = render.card(replace(row, state=root.state), self.now, self.people)
        state = replace(root.state, card=_hash(message))
        if state != root.saved:
            self.slack.update(root.thread.ts, message, metadata(ITEM_EVENT, state.payload()))
        root.state = root.saved = state

    def _overview(self, rows: list[render.Row]) -> None:
        today = london_date(self.now)
        bounds = window(today)
        current = self.channel.overview
        started = bounds is not None and self.now >= bounds[0]
        if current is None or (started and current.state.day != today.isoformat()):
            # The day's first overview keeps showing the outcome of the latest request to Claude.
            state = OverviewState(
                day=today.isoformat(), claude=current.state.claude if current else ""
            )
            # The edit at the end of this method brings in the mentions.
            content = render.overview(rows, today, self.now, self.quiet, self._claude(state))
            ts = self.slack.post(content, metadata(OVERVIEW_EVENT, state.payload()))
            if current is not None:
                self.channel.stale.append(current)
            current = self.channel.overview = Overview(Thread(ts, ""), state)
        # Each older overview gets its own attempt, so a failed edit neither stops the reminders
        # nor gets lost: the next run finds that overview still not marked and tries again.
        link = self._link(current.thread.ts)
        for old in self.channel.stale:
            self._attempt("old overview", lambda: self._retire(old, link))

        state = current.state
        requests, read_until = self._requests(current.thread, state.read_until)
        for request in requests:
            if parse(str(request.get("text") or "")) is not None:
                key = f"command:{request['ts']}"
                self._reply(current.thread, key, lambda _people: render.wrong_thread())
        state = replace(state, read_until=read_until)
        if self.config.claude.status:
            state = replace(state, claude=self.config.claude.status)
        # The list goes first in a new overview's thread, above the day's reminder summary.
        state = self._waiting(current.thread, state, rows)
        if self.config.pings and in_window(self.now) and state.day == today.isoformat():
            state = self._remind(current.thread, state, rows)

        day = datetime.fromisoformat(state.day).date()
        content = render.overview(rows, day, self.now, self.people, self._claude(state))
        self.slack.update(current.thread.ts, content, metadata(OVERVIEW_EVENT, state.payload()))
        current.state = state

    def _claude(self, state: OverviewState) -> str:
        return state.claude if self.config.claude.on else "Claude off"

    def _retire(self, old: Overview, link: str) -> None:
        """Replace an older overview's list with a link to the newest overview."""
        state = replace(old.state, replaced=True)
        day = datetime.fromisoformat(state.day).date()
        self.slack.update(
            old.thread.ts,
            render.replaced_overview(day, link),
            metadata(OVERVIEW_EVENT, state.payload()),
        )
        old.state = state

    def _waiting(
        self, thread: Thread, state: OverviewState, rows: list[render.Row]
    ) -> OverviewState:
        """Post or edit the reply in the overview's thread that lists what waits on contributors."""
        key = f"waiting:{state.day}"
        data = metadata(REPLY_EVENT, {"key": key})
        message = render.waiting(rows, self.now, self.people)
        # A run that posted the reply but stopped before it saved the overview finds the reply by
        # its key, so it never posts a second list.
        ts = state.waiting or thread.keys.get(key, "")
        card = _hash(message)
        if not ts:
            ts = self.slack.post(message, data, thread.ts)
            thread.keys[key] = ts
        elif card != state.waiting_card:
            self.slack.update(ts, message, data)
        return replace(state, waiting=ts, waiting_card=card)

    def _owners(self, item: Item) -> list[str | None]:
        users = [
            self.config.users[a.lower()] for a in item.assignees if a.lower() in self.config.users
        ]
        return list(dict.fromkeys(users)) or [self.config.oncall]

    def _snooze_ended(self, state: ItemState) -> bool:
        # A snooze that ran out reminds the owners once, whatever the item's tier and whoever owes
        # the next move, because the person who snoozed it chose to look at it again then.
        # `!unsnooze` and new activity clear the snooze instead, so they never get here.
        return state.reminded_at < state.snooze_until <= self.now.timestamp()

    def _due(self, row: render.Row) -> bool:
        state, assessment = row.state, row.assessment
        if state is None or state.snoozed(self.now):
            return False
        if self._snooze_ended(state):
            return True
        if row.item.number in self.config.claude.held:
            return False
        if assessment.owes is not Owes.VORTEX or assessment.tier is Tier.LOW:
            return False
        if not state.reminded_at:
            return True
        reminded = datetime.fromtimestamp(state.reminded_at, UTC)
        if london_date(reminded) == london_date(self.now):
            return False
        if assessment.tier is Tier.NORMAL:
            return working_days_between(reminded, self.now) >= NORMAL_REMINDER_DAYS
        return True

    def _remind(
        self, thread: Thread, state: OverviewState, rows: list[render.Row]
    ) -> OverviewState:
        """Remind the owners of each due item in the item's own thread.

        The first reminders of the day come after one summary in the overview's thread, which
        mentions everyone with due items. An item that becomes due later in the day gets only its
        own reminder.
        """
        grouped = render.ordered([row for row in rows if self._due(row)])
        due = [row for section in render.SECTIONS for row in grouped[section]]
        if not due:
            return state
        people: dict[str | None, list[render.Row]] = {}
        first = not state.reminded
        if first:
            for row in due:
                for person in self._owners(row.item):
                    people.setdefault(person, []).append(row)
            # A failed summary stops here, before any item is marked reminded, so the next run
            # posts the summary and the item reminders together.
            summary = render.reminder_summary(people)
            self._reply(thread, f"remind:{state.day}", lambda _people: summary)
            state = replace(state, reminded=",".join(sorted(p or "oncall" for p in people)))
        for row in due:
            self._attempt(f"#{row.item.number}", lambda: self._remind_item(row, state.day))
        if first:
            self._attempt("summary", lambda: self._link_summary(thread, state.day, people))
        return state

    def _link_summary(
        self, thread: Thread, day: str, people: dict[str | None, list[render.Row]]
    ) -> None:
        """Point each item in the day's summary at its reminder.

        A link to a reply opens the item's thread, but the reminders exist only after the summary,
        so the summary first links to the item roots. The edit changes only links, so it notifies
        nobody again.
        """
        linked = {
            person: [
                replace(row, thread_url=self.reminders.get(row.item.number, row.thread_url))
                for row in rows
            ]
            for person, rows in people.items()
        }
        key = f"remind:{day}"
        self.slack.update(
            thread.keys[key],
            render.reminder_summary(linked),
            metadata(REPLY_EVENT, {"key": key}),
        )

    def _remind_item(self, row: render.Row, day: str) -> None:
        root = self.channel.roots[row.item.number]
        if not root.thread.read and _newer(root.thread.latest_reply, root.state.read_until):
            # This run could not read the thread's new replies, which may hold today's reminder
            # from a run that stopped before it saved. The next run reads them and reminds.
            return
        owners = self._owners(row.item)
        on_call = not any(a.lower() in self.config.users for a in row.item.assignees)
        ended = self._snooze_ended(root.state)
        reminder = render.item_reminder(row, owners, on_call, ended, self.now)
        # The end of a snooze has its own key, because the item may already have today's reminder.
        key = f"snooze-end:{root.state.snooze_until:.0f}" if ended else f"remind:{day}"
        self._reply(root.thread, key, lambda _people: reminder)
        self.reminders[row.item.number] = self._reply_link(root.thread.keys[key], root.thread.ts)
        root.state = replace(root.state, reminded_at=self.now.timestamp())
