"""The few Slack Web API calls the triage publisher needs."""

from __future__ import annotations

import http.client
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterator
from typing import Any

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

    def _messages(self, thread_ts: str | None, oldest: float) -> Iterator[dict[str, Any]]:
        method = "conversations.replies" if thread_ts else "conversations.history"
        params: dict[str, object] = {
            "channel": self.channel,
            "include_all_metadata": "true",
            "limit": 200,
        }
        if thread_ts:
            params["ts"] = thread_ts
        else:
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
        for message in self._messages(thread_ts, oldest):
            metadata = message.get("metadata") or {}
            payload = metadata.get("event_payload") or {}
            if metadata.get("event_type") == EVENT_TYPE and payload.get("key") == key:
                return {"ts": message["ts"], **payload}
        return None
