"""The few Slack Web API calls the triage publisher needs."""

from __future__ import annotations

import http.client
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .publish import Thread

# Marks the messages this tool posts, in Slack message metadata.
EVENT_TYPE = "vortex_triage"

# Slack documents these `ok: false` errors as cases where part of the operation may have succeeded.
_AMBIGUOUS_ERRORS = frozenset(
    {"fatal_error", "internal_error", "request_timeout", "service_unavailable"}
)


class SlackError(Exception):
    """Slack rejected a call, so nothing was posted."""

    def __init__(self, message: str, code: str | None = None) -> None:
        super().__init__(message)
        self.code = code


class SlackUnknown(Exception):
    """A call failed without an answer from Slack, so a message may or may not have been posted."""


class Slack:
    def __init__(self, token: str, channel: str, attempts: int = 3) -> None:
        self.token = token
        self.channel = channel
        self.attempts = attempts

    def _call(self, method: str, params: dict[str, object], post: bool) -> dict[str, Any]:
        url = f"https://slack.com/api/{method}"
        headers = {"Authorization": f"Bearer {self.token}"}
        data = None
        if post:
            data = json.dumps(params).encode()
            headers["Content-Type"] = "application/json; charset=utf-8"
        else:
            url += "?" + urllib.parse.urlencode(params)
        attempts = max(1, self.attempts)
        for attempt in range(attempts):
            request = urllib.request.Request(url, data=data, headers=headers)
            try:
                with urllib.request.urlopen(request, timeout=30) as response:
                    body = json.load(response)
            except urllib.error.HTTPError as error:
                # Slack rejects a rate-limited call before acting on it, so a retry is safe.
                if error.code == 429:
                    if attempt + 1 < attempts:
                        time.sleep(int(error.headers.get("Retry-After", "1")))
                        continue
                    raise SlackError(f"{method}: rate limited", code="rate_limited") from error
                if error.code >= 500:
                    raise SlackUnknown(f"{method}: HTTP {error.code}") from error
                raise SlackError(f"{method}: HTTP {error.code}") from error
            except (OSError, ValueError, http.client.HTTPException) as error:
                raise SlackUnknown(f"{method}: {error}") from error
            if not body.get("ok"):
                err = str(body.get("error") or "")
                if err in _AMBIGUOUS_ERRORS:
                    raise SlackUnknown(f"{method}: {err}")
                raise SlackError(f"{method}: {err}", code=err or None)
            return body
        raise AssertionError("unreachable")

    def post(self, text: str, key: str, payload: dict[str, str], thread_ts: str | None) -> str:
        """Post a message tagged with `key` and return its timestamp."""
        params: dict[str, object] = {
            "channel": self.channel,
            "text": text,
            "unfurl_links": False,
            "metadata": {"event_type": EVENT_TYPE, "event_payload": {"key": key, **payload}},
        }
        if thread_ts:
            params["thread_ts"] = thread_ts
        return self._call("chat.postMessage", params, post=True)["ts"]

    def update(self, ts: str, text: str, key: str, payload: dict[str, str]) -> None:
        """Update a message's metadata (and text) in place."""
        params: dict[str, object] = {
            "channel": self.channel,
            "ts": ts,
            "text": text,
            "metadata": {"event_type": EVENT_TYPE, "event_payload": {"key": key, **payload}},
        }
        self._call("chat.update", params, post=True)

    def messages(
        self, thread_ts: str | None = None, oldest: float = 0.0
    ) -> Iterator[dict[str, Any]]:
        """Iterate messages in the channel or in one thread, including Slack metadata."""
        method = "conversations.replies" if thread_ts else "conversations.history"
        params: dict[str, object] = {
            "channel": self.channel,
            "include_all_metadata": "true",
            "limit": 200,
        }
        if thread_ts:
            params["ts"] = thread_ts
        elif oldest > 0:
            params["oldest"] = f"{oldest:.6f}"
        while True:
            body = self._call(method, params, post=False)
            yield from body["messages"]
            cursor = body.get("response_metadata", {}).get("next_cursor")
            if not cursor:
                return
            params["cursor"] = cursor

    def find(self, key: str, thread_ts: str | None, oldest: float) -> dict[str, Any] | None:
        """Find a message posted with `key`, in a thread or in the channel since `oldest`.

        Returns the message's `ts` and the payload it was posted with, or None.
        """
        for message in self.messages(thread_ts, oldest):
            metadata = message.get("metadata") or {}
            payload = metadata.get("event_payload") or {}
            if metadata.get("event_type") == EVENT_TYPE and payload.get("key") == key:
                return {"ts": message["ts"], **payload}
        return None


def _triage_payload(message: dict[str, Any]) -> dict[str, str] | None:
    metadata = message.get("metadata") or {}
    if metadata.get("event_type") != EVENT_TYPE:
        return None
    payload = metadata.get("event_payload")
    return payload if isinstance(payload, dict) else None


def _parse_key(key: str) -> tuple[int, str, int] | None:
    parts = key.split(":")
    if len(parts) < 2 or not parts[0].isdigit():
        return None
    number = int(parts[0])
    kind = parts[1]
    if kind == "root" and len(parts) == 2:
        return number, "root", 0
    if kind in {"update", "ping"} and len(parts) == 3 and parts[2].isdigit():
        return number, kind, int(parts[2])
    return None


def _parse_float(value: object) -> float:
    if not isinstance(value, (int, float, str)):
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


class SlackStore:
    """Reconstruct thread state from Slack channel history and thread metadata.

    Requires no external database: every root message and bot reply carries `vortex_triage`
    metadata with the item number, status, latest external activity timestamp, and last ping time.
    """

    def __init__(self, slack: Slack, oldest: float = 0.0) -> None:
        self.slack = slack
        self.oldest = oldest
        self._threads: dict[int, Thread] | None = None

    def load(self) -> dict[int, Thread]:
        if self._threads is not None:
            return dict(self._threads)
        from .publish import Thread

        roots: dict[int, tuple[dict[str, Any], dict[str, str]]] = {}
        for message in self.slack.messages(thread_ts=None, oldest=self.oldest):
            payload = _triage_payload(message)
            if not payload:
                continue
            parsed = _parse_key(str(payload.get("key", "")))
            if not parsed or parsed[1] != "root":
                continue
            number = parsed[0]
            existing = roots.get(number)
            if existing is None or _parse_float(message.get("ts")) < _parse_float(
                existing[0].get("ts")
            ):
                roots[number] = (message, payload)

        threads: dict[int, Thread] = {}
        for number, (root_msg, root_payload) in roots.items():
            root_ts = str(root_msg["ts"])
            status = str(root_payload.get("status", ""))
            external_at = str(root_payload.get("external_at", ""))
            last_ping_at = _parse_float(root_payload.get("last_ping_at"))
            last_checked_at = _parse_float(root_payload.get("last_checked_at"))
            updates = 0
            pings = 0

            if int(root_msg.get("reply_count") or 0) > 0:
                replies = sorted(
                    self.slack.messages(thread_ts=root_ts),
                    key=lambda msg: _parse_float(msg.get("ts")),
                )
                for reply in replies:
                    if str(reply.get("ts")) == root_ts:
                        continue
                    payload = _triage_payload(reply)
                    if not payload:
                        continue
                    parsed = _parse_key(str(payload.get("key", "")))
                    if not parsed or parsed[0] != number:
                        continue
                    _, kind, seq = parsed
                    if kind == "update":
                        updates = max(updates, seq)
                    elif kind == "ping":
                        pings = max(pings, seq)
                        ping_ts = _parse_float(payload.get("last_ping_at")) or _parse_float(
                            reply.get("ts")
                        )
                        last_ping_at = max(last_ping_at, ping_ts)
                    if payload.get("status"):
                        status = str(payload["status"])
                    if payload.get("external_at"):
                        external_at = max(external_at, str(payload["external_at"]))

            threads[number] = Thread(
                created_at=_parse_float(root_ts),
                root_ts=root_ts,
                status=status,
                external_at=external_at,
                updates=updates,
                last_ping_at=last_ping_at,
                last_checked_at=last_checked_at,
                pings=pings,
            )

        self._threads = threads
        return dict(threads)

    def create(self, number: int, thread: Thread) -> bool:
        threads = self.load()
        if number in threads:
            return False
        assert self._threads is not None
        self._threads[number] = thread
        return True

    def save(self, number: int, thread: Thread) -> None:
        self.load()
        assert self._threads is not None
        self._threads[number] = thread
