"""Keep one Slack thread per tracked item, and reply in it when the item changes."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any, Protocol

from .github import EXTERNAL_AUTHOR, EXTERNAL_COMMENT, FUZZER, Item
from .slack import SlackError, SlackUnknown

_KINDS = {"issue": "Issue", "pr": "PR"}
_REASONS = {
    EXTERNAL_AUTHOR: "External author",
    EXTERNAL_COMMENT: "External comment",
    FUZZER: "Fuzzer",
}


@dataclass(frozen=True)
class Thread:
    """What has been posted to Slack for one item."""

    created_at: float
    root_ts: str | None = None
    status: str = ""
    external_at: str = ""
    updates: int = 0
    # The key of a post that was started but not confirmed. It is saved before the post is sent,
    # so a run that dies in between is found by the next run instead of posting twice.
    pending: str | None = None

    @property
    def open(self) -> bool:
        return self.status.startswith("Open")


class Store(Protocol):
    """Durable thread state. Slack only displays it."""

    def load(self) -> dict[int, Thread]: ...

    def create(self, number: int, thread: Thread) -> bool:
        """Add a thread. Returns False if the item already has one."""
        ...

    def save(self, number: int, thread: Thread) -> None: ...


class Messenger(Protocol):
    def post(self, text: str, key: str, payload: dict[str, str], thread_ts: str | None) -> str: ...

    def find(self, key: str, thread_ts: str | None, oldest: float) -> dict[str, Any] | None: ...


class MemoryStore:
    """A store that lasts for one process, for dry runs and tests."""

    def __init__(self) -> None:
        self.threads: dict[int, Thread] = {}

    def load(self) -> dict[int, Thread]:
        return dict(self.threads)

    def create(self, number: int, thread: Thread) -> bool:
        if number in self.threads:
            return False
        self.threads[number] = thread
        return True

    def save(self, number: int, thread: Thread) -> None:
        self.threads[number] = thread


def _escape(text: str) -> str:
    # Collapse newlines so they cannot break `<url|label>`, escape Slack's control chars, and
    # break up triple backticks so a contributor title cannot start a multiline code block.
    cleaned = " ".join(text.split()).replace("```", "` ` `")
    return cleaned.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def status(item: Item) -> str:
    people = ", ".join(f"`{login}`" for login in item.assignees)
    return f"Open · GitHub assignees: {people}" if people else "Open · no GitHub assignee"


def root_text(item: Item) -> str:
    tags = " · ".join(_REASONS[reason] for reason in item.reasons)
    return (
        f"*{_KINDS[item.kind]} #{item.number}* · {tags}\n"
        f"<{item.url}|{_escape(item.title)}>\n"
        f"Opened by `{item.author}` on {item.created_at[:10]} · {status(item)}"
    )


def update_text(thread: Thread, new_status: str, item: Item | None) -> str | None:
    """Describe what changed since the last post, or return None if nothing did."""
    lines = []
    if new_status != thread.status:
        lines.append(f"GitHub update: {new_status}")
    if item and item.external_at and item.external_at > thread.external_at:
        lines.append(f"New activity from `{item.external_by}` on <{item.url}|GitHub>")
    return "\n".join(lines) or None


class Publisher:
    def __init__(
        self,
        store: Store,
        slack: Messenger,
        closed_state: Callable[[int], str | None],
        now: Callable[[], float] = time.time,
    ) -> None:
        self.store = store
        self.slack = slack
        self.closed_state = closed_state
        self.now = now

    def sync(self, items: list[Item], max_new: int | None = None) -> list[str]:
        """Bring Slack up to date with the open `items`. Returns the items that failed.

        `items` must hold every open item, so that a tracked item missing from it is known to be
        closed. At most `max_new` threads are started, which keeps a backfill in small batches.
        """
        threads = self.store.load()
        open_items = {item.number: item for item in items}
        failures = []
        started = 0
        for number in sorted(open_items.keys() | threads.keys()):
            item = open_items.get(number)
            thread = threads.get(number)
            can_start = max_new is None or started < max_new
            if thread is None or (thread.root_ts is None and not thread.pending):
                if not (item and item.reasons) or not can_start:
                    continue
            elif not item and not thread.open and not thread.pending:
                continue
            try:
                if self._sync_item(number, item, thread, can_start=can_start):
                    started += 1
            except (SlackError, SlackUnknown, RuntimeError) as error:
                failures.append(f"#{number}: {error}")
        return failures

    def _sync_item(
        self, number: int, item: Item | None, thread: Thread | None, *, can_start: bool
    ) -> bool:
        if thread is None:
            thread = Thread(created_at=self.now())
            if not self.store.create(number, thread):
                return False
        if thread.pending:
            thread = self._recover(number, thread)

        if item:
            new_status = status(item)
            external_at = max(thread.external_at, item.external_at or "")
        else:
            closed = self.closed_state(number)
            if closed is None:
                # Reopened after the scan. The next run sees it as an open item.
                return False
            new_status, external_at = closed.capitalize(), thread.external_at
        payload = {"status": new_status, "external_at": external_at}

        if thread.root_ts is None:
            if item is None or not item.reasons or not can_start:
                return False
            self._post(number, thread, f"{number}:root", root_text(item), payload)
            return True
        if text := update_text(thread, new_status, item):
            self._post(number, thread, f"{number}:update:{thread.updates + 1}", text, payload)
        return False

    def _post(
        self, number: int, thread: Thread, key: str, text: str, payload: dict[str, str]
    ) -> None:
        thread = replace(thread, pending=key)
        self.store.save(number, thread)
        try:
            ts = self.slack.post(text, key, payload, thread.root_ts)
        except SlackError as error:
            if error.code == "thread_not_found" and thread.root_ts is not None:
                # The parent thread was deleted in Slack; drop the stale ts so the next run
                # starts a fresh root instead of failing forever.
                self.store.save(number, Thread(created_at=self.now()))
                raise
            # Slack refused the post, so there is nothing to look for on the next run.
            self.store.save(number, replace(thread, pending=None))
            raise
        self.store.save(number, _posted(thread, ts, payload))

    def _recover(self, number: int, thread: Thread) -> Thread:
        """Settle a post that an earlier run started: adopt it if it reached Slack."""
        try:
            found = self.slack.find(thread.pending or "", thread.root_ts, thread.created_at - 60)
        except SlackError as error:
            if error.code == "thread_not_found" and thread.root_ts is not None:
                reset = Thread(created_at=self.now())
                self.store.save(number, reset)
                return reset
            raise
        thread = _posted(thread, found["ts"], found) if found else replace(thread, pending=None)
        self.store.save(number, thread)
        return thread


def _posted(thread: Thread, ts: str, payload: dict[str, Any]) -> Thread:
    if thread.root_ts is None:
        thread = replace(thread, root_ts=ts)
    else:
        thread = replace(thread, updates=thread.updates + 1)
    return replace(
        thread, status=payload["status"], external_at=payload["external_at"], pending=None
    )
