# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Builders for the triage bot tests, and a fake Slack channel that behaves like the real one.

The triage package lives in `.github/scripts`, outside every import root, so the tests load it by
path. The fake channel keeps exactly what the bot posted. A thread's `latest_reply` moves only when
someone replies, never on an edit, because the bot's crash recovery depends on that.
"""

import importlib
import sys
from collections.abc import Callable, Iterable, Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / ".github" / "scripts"))
github = importlib.import_module("triage.github")
rules = importlib.import_module("triage.rules")
hours = importlib.import_module("triage.hours")
commands = importlib.import_module("triage.commands")
state = importlib.import_module("triage.state")
slack = importlib.import_module("triage.slack")
render = importlib.import_module("triage.render")
sync = importlib.import_module("triage.sync")
evaluate = importlib.import_module("triage.evaluate")
cli = importlib.import_module("triage.__main__")

TEAMMATE = "gatesn"
CONTRIBUTOR = "alamb"
# A Tuesday, inside the working window (09:00 to 23:00 UTC in November).
NOW = datetime(2026, 11, 24, 12, 0, tzinfo=UTC)
BOT_USER, BOT_ID = "UBOT", "BBOT"
ONCALL = "UONCALL"


def iso(moment: datetime) -> str:
    return moment.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def ago(**delta: float) -> str:
    """Return the GitHub timestamp of a moment before `NOW`, for example `ago(days=2)`."""
    return iso(NOW - timedelta(**delta))


def event(at: str, actor: str = CONTRIBUTOR, kind: str = "comment", state: str = "") -> Any:
    return github.Event(at=at, actor=actor, team=actor in github.TEAM, kind=kind, state=state)


def item(
    number: int = 1,
    *,
    kind: str = "issue",
    author: str = CONTRIBUTOR,
    created: str | None = None,
    events: Iterable[Any] = (),
    labels: Iterable[str] = (),
    assignees: Iterable[str] = (),
    draft: bool = False,
    changed_lines: int = 1000,
    title: str = "A title",
    author_bot: bool = False,
) -> Any:
    """Build an open item. Its opening event comes first, then `events`."""
    created = created or ago(days=1)
    opened = () if author_bot else (event(created, author, "opened"),)
    return github.Item(
        number=number,
        kind=kind,
        title=title,
        url=f"https://github.com/o/r/issues/{number}",
        author=author,
        author_bot=author_bot,
        author_team=author in github.TEAM,
        created_at=created,
        labels=frozenset(labels),
        assignees=tuple(assignees),
        events=(*opened, *events),
        draft=draft,
        changed_lines=changed_lines,
    )


class FakeSlack:
    """An in-memory channel. Faults make the next call of a method fail, before or after it acts."""

    channel = "C1"

    def __init__(self) -> None:
        self.messages: dict[str, dict[str, Any]] = {}
        self.clock = NOW.timestamp()
        self.calls: list[str] = []
        # (method, when) pairs. `before` raises SlackError and changes nothing. `after` acts and
        # then raises SlackUnknown, like a timeout after Slack stored the message.
        self.faults: list[tuple[str, str]] = []
        # Every post as Slack received it, before any edit, which decides who gets notified.
        self.posted: list[dict[str, Any]] = []

    def identity(self) -> Any:
        return slack.Identity(BOT_USER, BOT_ID, "https://example.slack.com/")

    def _ts(self) -> str:
        self.clock += 0.001
        return f"{self.clock:.6f}"

    def _fault(self, method: str) -> str | None:
        for fault in self.faults:
            if fault[0] == method:
                self.faults.remove(fault)
                if fault[1] == "before":
                    raise slack.SlackError(f"{method}: fault", code="fault")
                return fault[1]
        return None

    def _store(self, message: dict[str, Any]) -> str:
        ts = str(message["ts"])
        self.messages[ts] = message
        if (root_ts := message.get("thread_ts")) and root_ts != ts:
            root = self.messages[root_ts]
            root["latest_reply"] = ts
            root["reply_count"] = root.get("reply_count", 0) + 1
        return ts

    def _top(self) -> list[dict[str, Any]]:
        top = [m for m in self.messages.values() if m.get("thread_ts", m["ts"]) == m["ts"]]
        return sorted(top, key=lambda m: -float(m["ts"]))

    def history(self) -> Iterator[dict[str, Any]]:
        self.calls.append("history")
        return iter([dict(m) for m in self._top()])

    def replies(self, thread_ts: str) -> Iterator[dict[str, Any]]:
        self.calls.append("replies")
        self._fault("replies")
        thread = [m for m in self.messages.values() if m.get("thread_ts") == thread_ts]
        root = self.messages[thread_ts]
        rest = sorted((m for m in thread if m["ts"] != thread_ts), key=lambda m: float(m["ts"]))
        return iter([dict(root), *(dict(m) for m in rest)])

    def post(self, message: Any, metadata: dict[str, Any], thread_ts: str | None = None) -> str:
        self.calls.append("post")
        when = self._fault("post")
        self.posted.append({**message.params(), "metadata": metadata, "thread_ts": thread_ts})
        ts = self._store(
            {
                "ts": self._ts(),
                "user": BOT_USER,
                "bot_id": BOT_ID,
                **message.params(),
                "metadata": metadata,
                **({"thread_ts": thread_ts} if thread_ts else {}),
            }
        )
        if when == "after":
            raise slack.SlackUnknown("post: timed out")
        return ts

    def update(self, ts: str, message: Any, metadata: dict[str, Any]) -> None:
        self.calls.append("update")
        when = self._fault("update")
        if ts not in self.messages:
            raise slack.SlackError("chat.update: message_not_found", code="message_not_found")
        self.messages[ts].update({**message.params(), "metadata": metadata})
        if when == "after":
            raise slack.SlackUnknown("update: timed out")

    def delete(self, ts: str) -> None:
        self.calls.append("delete")
        # The first live purge got `cant_delete_message` for one message with the bot's user ID, so
        # the fake refuses a system message the same way.
        if self.messages[ts].get("subtype") == "channel_join":
            raise slack.SlackError("chat.delete: cant_delete_message", code="cant_delete_message")
        self.messages.pop(ts)

    def say(self, thread_ts: str | None, text: str, at: datetime, user: str = "UALICE") -> str:
        """Post a person's message at a given time, in a thread or, without one, in the channel."""
        message = {"ts": f"{at.timestamp():.6f}", "user": user, "text": text}
        return self._store({**message, "thread_ts": thread_ts} if thread_ts else message)

    def top(self, event_type: str) -> list[dict[str, Any]]:
        """Return the bot's top-level messages of one metadata type, newest first."""
        return [m for m in self._top() if (m.get("metadata") or {}).get("event_type") == event_type]

    def root(self, number: int) -> dict[str, Any]:
        roots = [m for m in self.top(state.ITEM_EVENT) if m["metadata"]["event_payload"]["number"] == str(number)]
        assert len(roots) == 1, f"expected one root for #{number}, found {len(roots)}"
        return roots[0]

    def thread(self, ts: str) -> list[dict[str, Any]]:
        """Return the replies in a thread, oldest first, without the root."""
        return list(self.replies(ts))[1:]

    def bot_replies(self, ts: str) -> list[str]:
        return [m["text"] for m in self.thread(ts) if m.get("user") == BOT_USER]


def run(
    fake: FakeSlack,
    items: list[Any],
    now: datetime = NOW,
    *,
    pings: bool = False,
    verdicts: dict[int, Any] | None = None,
    users: dict[str, str] | None = None,
    ended: Callable[[int], Any] | None = None,
) -> Any:
    """Run the bot once against the fake channel, as a new process would."""
    fake.clock = max(fake.clock, now.timestamp())
    config = sync.Config(
        oncall=ONCALL,
        users={"alice": "UALICE"} if users is None else users,
        pings=pings,
        verdicts=verdicts or {},
    )
    channel = sync.load(fake.history())
    return sync.Run(fake, config, now, ended or (lambda _number: None), channel).execute(items)
