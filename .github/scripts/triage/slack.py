"""The few Slack Web API calls that the triage bot needs."""

from __future__ import annotations

import http.client
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

# Slack documents these `ok: false` errors as cases where part of the operation may have succeeded.
_AMBIGUOUS_ERRORS = frozenset(
    {"fatal_error", "internal_error", "request_timeout", "service_unavailable"}
)


class SlackError(Exception):
    """Slack rejected a call, so nothing changed."""

    def __init__(self, message: str, code: str | None = None) -> None:
        super().__init__(message)
        self.code = code


class SlackUnknown(Exception):
    """A call failed without an answer from Slack, so it may or may not have taken effect."""


@dataclass(frozen=True)
class Identity:
    """Who the bot is, and the workspace URL that thread links start with."""

    user_id: str
    bot_id: str
    url: str


@dataclass(frozen=True)
class Message:
    """What the bot posts: fallback text for notifications, plus Block Kit content."""

    text: str
    blocks: tuple[dict[str, Any], ...] = ()
    attachments: tuple[dict[str, Any], ...] = ()

    def params(self) -> dict[str, Any]:
        params: dict[str, Any] = {"text": self.text}
        if self.blocks:
            params["blocks"] = list(self.blocks)
        # An empty list clears attachments on `chat.update`, which matters when content shrinks.
        params["attachments"] = list(self.attachments)
        return params


class Slack:
    def __init__(self, token: str, channel: str, attempts: int = 5) -> None:
        self.token = token
        self.channel = channel
        self.attempts = attempts

    def _call(self, method: str, params: dict[str, Any], post: bool) -> dict[str, Any]:
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

    def identity(self) -> Identity:
        body = self._call("auth.test", {}, post=True)
        return Identity(user_id=body["user_id"], bot_id=body.get("bot_id", ""), url=body["url"])

    def _paged(self, method: str, params: dict[str, Any]) -> Iterator[dict[str, Any]]:
        params = {"channel": self.channel, "include_all_metadata": "true", "limit": 200, **params}
        while True:
            body = self._call(method, params, post=False)
            yield from body["messages"]
            cursor = body.get("response_metadata", {}).get("next_cursor")
            if not cursor:
                return
            params["cursor"] = cursor

    def history(self) -> Iterator[dict[str, Any]]:
        """Iterate every top-level message in the channel, newest first, with metadata."""
        return self._paged("conversations.history", {})

    def replies(self, thread_ts: str) -> Iterator[dict[str, Any]]:
        """Iterate a thread's root and replies, oldest first, with metadata."""
        return self._paged("conversations.replies", {"ts": thread_ts})

    def post(self, message: Message, metadata: dict[str, Any], thread_ts: str | None = None) -> str:
        """Post a message and return its timestamp."""
        params = {"channel": self.channel, **message.params(), "metadata": metadata}
        # `chat.update` does not take these, so only a new post sets them.
        params |= {"unfurl_links": False, "unfurl_media": False}
        if thread_ts:
            params["thread_ts"] = thread_ts
        return str(self._call("chat.postMessage", params, post=True)["ts"])

    def update(self, ts: str, message: Message, metadata: dict[str, Any]) -> None:
        params = {"channel": self.channel, "ts": ts, **message.params(), "metadata": metadata}
        self._call("chat.update", params, post=True)

    def delete(self, ts: str) -> None:
        self._call("chat.delete", {"channel": self.channel, "ts": ts}, post=True)
