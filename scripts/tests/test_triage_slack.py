# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Tests for the Slack Web API client in `.github/scripts/triage/slack.py`."""

import email.message
import http.client
import importlib
import io
import json
import urllib.error
from typing import Any

import pytest

support = importlib.import_module("triage_support")
slack = support.slack
HELLO = slack.Message("hi")


class Response:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def __enter__(self) -> io.BytesIO:
        return io.BytesIO(json.dumps(self.payload).encode())

    def __exit__(self, *_args: object) -> bool:
        return False


def http_error(code: int, retry_after: str | None = None) -> urllib.error.HTTPError:
    headers = email.message.Message()
    if retry_after is not None:
        headers["Retry-After"] = retry_after
    return urllib.error.HTTPError("https://slack.com", code, "error", headers, None)


@pytest.fixture
def client(monkeypatch) -> Any:
    monkeypatch.setattr(slack.time, "sleep", lambda _s: None)
    return slack.Slack("xoxb-test", "C123", attempts=2)


def serve(monkeypatch, *outcomes: object) -> list[Any]:
    """Answer each Slack call with the next outcome, and return the requests that were sent."""
    queue, sent = list(outcomes), []

    def urlopen(request: Any, timeout: float = 30) -> object:
        sent.append(request)
        outcome = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    monkeypatch.setattr(slack.urllib.request, "urlopen", urlopen)
    return sent


def test_post_sends_metadata_and_never_unfurls(client, monkeypatch):
    sent = serve(monkeypatch, Response({"ok": True, "ts": "1.0"}))
    assert client.post(HELLO, {"event_type": "e", "event_payload": {}}, "0.5") == "1.0"
    body = json.loads(sent[0].data)
    assert body["metadata"] == {"event_type": "e", "event_payload": {}}
    assert (body["thread_ts"], body["unfurl_links"], body["unfurl_media"]) == ("0.5", False, False)
    # An update always sends attachments, so shrinking content clears the old ones.
    client.update("1.0", HELLO, {})
    assert json.loads(sent[1].data)["attachments"] == []
    assert "unfurl_links" not in json.loads(sent[1].data)


def test_rate_limit_retries_then_gives_up(client, monkeypatch):
    serve(monkeypatch, http_error(429, "0"), Response({"ok": True, "ts": "1.0"}))
    assert client.post(HELLO, {}) == "1.0"
    serve(monkeypatch, http_error(429, "0"))
    with pytest.raises(slack.SlackError, match="rate limited") as error:
        client.post(HELLO, {})
    assert error.value.code == "rate_limited"


def test_posts_and_rate_limit_retries_are_paced(client, monkeypatch):
    now = [0.0]
    sleeps = []

    def monotonic() -> float:
        return now[0]

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        now[0] += seconds

    monkeypatch.setattr(slack.time, "monotonic", monotonic)
    monkeypatch.setattr(slack.time, "sleep", sleep)
    sent = serve(
        monkeypatch,
        http_error(429, "0"),
        Response({"ok": True, "ts": "1.0"}),
        Response({"ok": True, "ts": "2.0"}),
    )

    assert client.post(HELLO, {}) == "1.0"
    assert client.post(HELLO, {}, "1.0") == "2.0"
    assert len(sent) == 3
    assert sleeps == [0, 1, 1]


@pytest.mark.parametrize(
    "outcome",
    [
        http_error(502),
        http.client.IncompleteRead(b"partial"),
        TimeoutError("timed out"),
        Response({"ok": False, "error": "internal_error"}),
        Response({"ok": False, "error": "request_timeout"}),
    ],
)
def test_unclear_failures_are_unknown(client, monkeypatch, outcome):
    serve(monkeypatch, outcome)
    with pytest.raises(slack.SlackUnknown):
        client.post(HELLO, {})


def test_definite_failures_carry_the_slack_code(client, monkeypatch):
    serve(monkeypatch, Response({"ok": False, "error": "message_not_found"}))
    with pytest.raises(slack.SlackError) as error:
        client.update("1.0", HELLO, {})
    assert error.value.code == "message_not_found"


def test_history_follows_the_cursor_and_asks_for_metadata(client, monkeypatch):
    sent = serve(
        monkeypatch,
        Response({"ok": True, "messages": [{"ts": "2.0"}], "response_metadata": {"next_cursor": "c2"}}),
        Response({"ok": True, "messages": [{"ts": "1.0"}]}),
    )
    assert [m["ts"] for m in client.history()] == ["2.0", "1.0"]
    assert "include_all_metadata=true" in sent[0].full_url
    assert "cursor=c2" in sent[1].full_url


def test_identity(client, monkeypatch):
    serve(monkeypatch, Response({"ok": True, "user_id": "U1", "bot_id": "B1", "url": "https://x.slack.com/"}))
    assert client.identity() == slack.Identity("U1", "B1", "https://x.slack.com/")
