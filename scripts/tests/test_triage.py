# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Tests for the triage scanner and Slack publisher in `.github/scripts/triage`.

Run with:

    uv run --no-project --with pytest pytest scripts/tests/test_triage.py
"""

import email.message
import http.client
import importlib
import io
import json
import subprocess
import sys
import urllib.error
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / ".github" / "scripts"))
github = importlib.import_module("triage.github")
publish = importlib.import_module("triage.publish")
slack = importlib.import_module("triage.slack")
evaluate = importlib.import_module("triage.evaluate")
cli = importlib.import_module("triage.__main__")

TEAM = {"login": "gatesn", "__typename": "User"}
OUTSIDER = {"login": "alamb", "__typename": "User"}
BOT = {"login": "github-actions", "__typename": "Bot"}


def node(author, comments=(), reviews=None, labels=(), assignees=(), number=1) -> dict[str, object]:
    result = {
        "number": number,
        "title": "A <title> & more",
        "url": f"https://github.com/o/r/issues/{number}",
        "createdAt": "2026-01-01T00:00:00Z",
        "author": author,
        "labels": {"nodes": [{"name": name} for name in labels]},
        "assignees": {"nodes": [{"login": login} for login in assignees]},
        "comments": {"nodes": [{"at": at, "author": who} for at, who in comments]},
    }
    if reviews is not None:
        result["reviews"] = {"nodes": [{"at": at, "author": who} for at, who in reviews]}
    return result


@pytest.mark.parametrize(
    ("kind", "item", "reasons"),
    [
        ("issue", node(TEAM), ()),
        ("issue", node({"login": "GatesN", "__typename": "User"}), ()),
        ("issue", node(OUTSIDER), ("external_author",)),
        ("pr", node(OUTSIDER, reviews=[]), ("external_author",)),
        ("issue", node(None), ("external_author",)),
        ("issue", node(BOT), ()),
        ("issue", node({"login": "renovate[bot]", "__typename": "User"}), ()),
        ("issue", node(BOT, labels=["fuzzer"]), ("fuzzer",)),
        ("pr", node(BOT, labels=["fuzzer"], reviews=[]), ()),
        ("issue", node(TEAM, comments=[("2026-01-02T00:00:00Z", OUTSIDER)]), ("external_comment",)),
        (
            "issue",
            node(TEAM, comments=[("2026-01-02T00:00:00Z", OUTSIDER), ("2026-01-03T00:00:00Z", BOT)]),
            ("external_comment",),
        ),
        (
            "issue",
            node(TEAM, comments=[("2026-01-02T00:00:00Z", OUTSIDER), ("2026-01-03T00:00:00Z", TEAM)]),
            (),
        ),
        (
            "pr",
            node(
                OUTSIDER,
                comments=[("2026-01-02T00:00:00Z", OUTSIDER)],
                reviews=[("2026-01-03T00:00:00Z", TEAM)],
            ),
            ("external_author",),
        ),
        (
            "pr",
            node(TEAM, comments=[("2026-01-02T00:00:00Z", TEAM)], reviews=[("2026-01-03T00:00:00Z", OUTSIDER)]),
            ("external_comment",),
        ),
        # A review that is still a draft has no submission time and does not count.
        ("pr", node(TEAM, reviews=[(None, OUTSIDER)]), ()),
    ],
)
def test_classify_reasons(kind, item, reasons):
    assert github.classify(kind, item).reasons == reasons


def test_classify_tracks_latest_activity():
    item = github.classify(
        "pr",
        node(
            OUTSIDER,
            comments=[("2026-01-04T00:00:00Z", OUTSIDER), ("2026-01-02T00:00:00Z", TEAM)],
            reviews=[("2026-01-03T00:00:00Z", TEAM)],
            assignees=["myrrc", "0ax1"],
        ),
    )
    assert (item.kind, item.author, item.assignees) == ("pr", "alamb", ("0ax1", "myrrc"))
    assert (item.external_at, item.external_by) == ("2026-01-04T00:00:00Z", "alamb")
    assert item.team_at == "2026-01-03T00:00:00Z"


def test_scan_follows_pages_and_sorts():
    calls = []

    def query(text, variables) -> dict[str, object]:
        calls.append(dict(variables))
        kind = "pullRequests" if "pullRequests" in text else "issues"
        cursor = variables.get("cursor")
        numbers = {("issues", None): [5], ("issues", "next"): [3], ("pullRequests", None): [4]}[kind, cursor]
        more = (kind, cursor) == ("issues", None)
        nodes = [node(OUTSIDER, reviews=[] if kind == "pullRequests" else None, number=n) for n in numbers]
        page = {"pageInfo": {"hasNextPage": more, "endCursor": "next" if more else None}, "nodes": nodes}
        return {"repository": {"items": page}}

    items = github.scan("o/r", query)
    assert [(item.number, item.kind) for item in items] == [(3, "issue"), (4, "pr"), (5, "issue")]
    assert [call.get("cursor") for call in calls] == [None, "next", None]


def test_scan_table_shows_titles_as_literal_text():
    hostile = github.classify("issue", node(OUTSIDER))
    hostile = github.Item(**{**hostile.__dict__, "title": "[click](https://example.com) `a``b` | <img>"})
    row = cli._table([hostile]).splitlines()[-1]
    assert row.endswith("| ``` [click](https://example.com) `a``b` \\| <img> ``` |")


class FakeSlack:
    """Records posts like Slack would, and can fail before or after a post lands."""

    def __init__(self):
        self.messages = []
        self.updates = []
        self.fail = None

    def post(self, text, key, payload, thread_ts) -> str:
        if self.fail == "refused":
            raise slack.SlackError("channel_not_found")
        if self.fail == "lost":
            raise slack.SlackUnknown("timed out")
        ts = f"{len(self.messages) + 1}.0"
        self.messages.append({"ts": ts, "text": text, "key": key, "thread_ts": thread_ts, **payload})
        if self.fail == "landed":
            raise slack.SlackUnknown("timed out")
        return ts

    def find(self, key, thread_ts, oldest) -> dict[str, str] | None:
        for message in self.messages:
            if message["key"] == key and message["thread_ts"] == thread_ts:
                return {k: message[k] for k in ("ts", "key", "status", "external_at")}
        return None

    def update(self, ts, text, key, payload):
        self.updates.append({"ts": ts, "text": text, "key": key, **payload})


def item(
    number=1, kind="issue", reasons=("external_author",), assignees=(), external_at="2026-01-01T00:00:00Z"
) -> object:
    return github.Item(
        number=number,
        kind=kind,
        title="A <title> & more",
        url=f"https://github.com/o/r/issues/{number}",
        author="alamb",
        created_at="2026-01-01T00:00:00Z",
        assignees=assignees,
        reasons=reasons,
        external_at=external_at,
        external_by="alamb",
    )


@pytest.fixture
def setup() -> tuple[object, ...]:
    store, fake, closed = publish.MemoryStore(), FakeSlack(), {}
    return store, fake, closed, publish.Publisher(store, fake, closed.get, now=lambda: 1000.0)


def test_one_root_per_item_and_reruns_post_nothing(setup):
    store, fake, _, publisher = setup
    items = [item(1), item(2, kind="pr", reasons=("external_author", "external_comment")), item(3, reasons=())]

    assert publisher.sync(items) == []
    assert publisher.sync(items) == []

    assert [m["key"] for m in fake.messages] == ["1:root", "2:root"]
    assert fake.messages[0]["text"] == (
        "*Issue #1* · External author\n"
        "<https://github.com/o/r/issues/1|A &lt;title&gt; &amp; more>\n"
        "Opened by `alamb` on 2026-01-01 · Open · no GitHub assignee"
    )
    assert fake.messages[1]["text"].startswith("*PR #2* · External author · External comment\n")
    assert store.threads[1].root_ts == "1.0" and store.threads[1].pending is None


def test_changes_reply_in_the_thread_once(setup):
    store, fake, closed, publisher = setup
    publisher.sync([item()])

    changed = item(assignees=("gatesn",), external_at="2026-02-01T00:00:00Z")
    publisher.sync([changed])
    publisher.sync([changed])
    closed[1] = "merged"
    publisher.sync([])
    publisher.sync([])

    replies = fake.messages[1:]
    assert [(m["key"], m["thread_ts"]) for m in replies] == [("1:update:1", "1.0"), ("1:update:2", "1.0")]
    assert replies[0]["text"] == (
        "GitHub update: Open · GitHub assignees: `gatesn`\n"
        "New activity from `alamb` on <https://github.com/o/r/issues/1|GitHub>"
    )
    assert replies[1]["text"] == "GitHub update: Merged"
    assert not store.threads[1].open

    publisher.sync([item()])
    assert fake.messages[-1]["text"] == "GitHub update: Open · no GitHub assignee"
    assert len(fake.messages) == 4


def test_tracked_item_stays_tracked_without_a_reason(setup):
    _, fake, _, publisher = setup
    publisher.sync([item(reasons=("external_comment",))])
    publisher.sync([item(reasons=(), assignees=("gatesn",))])
    assert [m["key"] for m in fake.messages] == ["1:root", "1:update:1"]


@pytest.mark.parametrize("update", [False, True])
def test_post_that_landed_before_a_timeout_is_not_repeated(setup, update):
    store, fake, _, publisher = setup
    current = item()
    if update:
        publisher.sync([current])
        current = item(assignees=("gatesn",))

    fake.fail = "landed"
    assert publisher.sync([current]) == ["#1: timed out"]
    assert store.threads[1].pending is not None
    fake.fail = None
    assert publisher.sync([current]) == []

    assert len(fake.messages) == (2 if update else 1)
    assert store.threads[1].pending is None
    assert store.threads[1].status == publish.status(current)


def test_root_in_doubt_is_settled_even_if_the_item_no_longer_qualifies(setup):
    store, fake, _, publisher = setup
    fake.fail = "landed"
    publisher.sync([item()])
    fake.fail = None
    assert publisher.sync([item(reasons=())], max_new=0) == []
    assert len(fake.messages) == 1
    assert store.threads[1].root_ts == "1.0" and store.threads[1].pending is None


def test_post_that_was_lost_is_sent_on_the_next_run(setup):
    store, fake, _, publisher = setup
    fake.fail = "lost"
    assert publisher.sync([item()]) == ["#1: timed out"]
    fake.fail = None
    assert publisher.sync([item()]) == []
    assert [m["key"] for m in fake.messages] == ["1:root"]
    assert store.threads[1].root_ts == "1.0"


def test_refused_post_fails_the_item_and_continues(setup):
    store, fake, _, publisher = setup
    fake.fail = "refused"
    assert publisher.sync([item(1), item(2)]) == ["#1: channel_not_found", "#2: channel_not_found"]
    assert store.threads[1].pending is None and store.threads[1].root_ts is None
    fake.fail = None
    assert publisher.sync([item(1), item(2)]) == []
    assert [m["key"] for m in fake.messages] == ["1:root", "2:root"]


def test_backfill_starts_at_most_max_new_threads(setup):
    _, fake, _, publisher = setup
    items = [item(n) for n in (1, 2, 3)]
    publisher.sync(items, max_new=2)
    assert [m["key"] for m in fake.messages] == ["1:root", "2:root"]
    publisher.sync([item(1, assignees=("gatesn",)), *items[1:]], max_new=0)
    assert [m["key"] for m in fake.messages][2:] == ["1:update:1"]
    publisher.sync(items, max_new=2)
    assert "3:root" in [m["key"] for m in fake.messages]


def test_reopened_item_waits_for_the_next_scan(setup):
    _, fake, _, publisher = setup
    publisher.sync([item()])
    publisher.sync([])
    assert len(fake.messages) == 1


def test_classify_captures_recent_conversation_without_bots():
    raw = {
        **node(OUTSIDER),
        "body": "  Problem   description  ",
        "comments": {
            "nodes": [
                {"at": "2026-01-02T00:00:00Z", "author": BOT, "body": "bot noise"},
                {"at": "2026-01-03T00:00:00Z", "author": TEAM, "body": "Will look tomorrow"},
            ],
        },
        "reviews": {
            "nodes": [
                {"at": "2026-01-04T00:00:00Z", "author": OUTSIDER, "body": "Pushed a fix", "state": "COMMENTED"},
            ],
        },
    }

    classified = github.classify("pr", raw)
    assert classified.body == "Problem description"
    assert [(c.author, c.team, c.kind, c.body, c.state) for c in classified.recent] == [
        ("gatesn", True, "comment", "Will look tomorrow", ""),
        ("alamb", False, "review", "Pushed a fix", "COMMENTED"),
    ]


def test_slack_store_reconstructs_threads_updates_and_pings():
    class FakeClient:
        def messages(self, thread_ts=None, oldest=0.0) -> object:
            if thread_ts is None:
                return iter(
                    [
                        {"ts": "100.0", "text": "unrelated human chat"},
                        {
                            "ts": "200.0",
                            "reply_count": 2,
                            "metadata": {
                                "event_type": slack.EVENT_TYPE,
                                "event_payload": {
                                    "key": "42:root",
                                    "status": "Open · no GitHub assignee",
                                    "external_at": "2026-01-01T00:00:00Z",
                                },
                            },
                        },
                    ]
                )
            assert thread_ts == "200.0"
            return iter(
                [
                    {"ts": "200.0", "metadata": {"event_type": slack.EVENT_TYPE, "event_payload": {"key": "42:root"}}},
                    {
                        "ts": "250.0",
                        "metadata": {
                            "event_type": slack.EVENT_TYPE,
                            "event_payload": {
                                "key": "42:update:1",
                                "status": "Open · GitHub assignees: `gatesn`",
                                "external_at": "2026-01-02T00:00:00Z",
                            },
                        },
                    },
                    {
                        "ts": "300.0",
                        "metadata": {
                            "event_type": slack.EVENT_TYPE,
                            "event_payload": {
                                "key": "42:ping:1",
                                "status": "Open · GitHub assignees: `gatesn`",
                                "external_at": "2026-01-02T00:00:00Z",
                                "last_ping_at": "300",
                            },
                        },
                    },
                ]
            )

    loaded = slack.SlackStore(FakeClient()).load()
    assert loaded[42] == publish.Thread(
        created_at=200.0,
        root_ts="200.0",
        status="Open · GitHub assignees: `gatesn`",
        external_at="2026-01-02T00:00:00Z",
        updates=1,
        last_ping_at=300.0,
        pings=1,
    )


def test_cadence_business_hours_and_weekday_counting():
    from datetime import datetime
    from zoneinfo import ZoneInfo

    ny = ZoneInfo("America/New_York")
    cadence = evaluate.Cadence(issue_days=7, pr_weekdays=1, tz="America/New_York")

    fri_17 = datetime(2026, 1, 9, 17, 0, tzinfo=ny).timestamp()
    sat_12 = datetime(2026, 1, 10, 12, 0, tzinfo=ny).timestamp()
    mon_08 = datetime(2026, 1, 12, 8, 59, tzinfo=ny).timestamp()
    mon_09 = datetime(2026, 1, 12, 9, 0, tzinfo=ny).timestamp()
    mon_15 = datetime(2026, 1, 12, 15, 0, tzinfo=ny).timestamp()
    fri_next = datetime(2026, 1, 16, 17, 0, tzinfo=ny).timestamp()

    assert evaluate.in_business_hours(fri_17, cadence)
    assert not evaluate.in_business_hours(sat_12, cadence)
    assert not evaluate.in_business_hours(mon_08, cadence)
    assert evaluate.in_business_hours(mon_09, cadence)
    assert evaluate.weekdays_between(fri_17, mon_09, "America/New_York") == 1
    assert evaluate.weekdays_between(mon_09, mon_15, "America/New_York") == 0

    pr_item = item(1, kind="pr", external_at="2026-01-01T00:00:00Z")
    pr_thread = publish.Thread(
        created_at=fri_17,
        root_ts="1.0",
        status="Open",
        external_at="2026-01-01T00:00:00Z",
        last_ping_at=fri_17,
        pings=1,
    )
    assert not evaluate.is_ping_due(pr_item, pr_thread, sat_12, cadence)
    assert evaluate.is_ping_due(pr_item, pr_thread, mon_09, cadence)

    issue_item = item(2, kind="issue", external_at="2026-01-01T00:00:00Z")
    issue_thread = publish.Thread(
        created_at=fri_17,
        root_ts="1.0",
        status="Open",
        external_at="2026-01-01T00:00:00Z",
        last_ping_at=fri_17,
        pings=1,
    )
    assert not evaluate.is_ping_due(issue_item, issue_thread, mon_09, cadence)
    assert evaluate.is_ping_due(issue_item, issue_thread, fri_next, cadence)


def test_load_verdicts_and_fallback(tmp_path):
    verdicts_file = tmp_path / "verdicts.json"
    verdicts_file.write_text(
        '{"1": {"state": "waiting_on_vortex", "reason": "  No review\\nyet  "}, '
        '"2": {"state": "bogus", "reason": "ignored"}, '
        '"bad": {"state": "waiting_on_vortex", "reason": "ignored"}, '
        '"3": {"state": "waiting_on_contributor", "reason": "Asked for tests"}}',
        encoding="utf-8",
    )
    loaded = evaluate.load_verdicts(verdicts_file)
    assert set(loaded.keys()) == {1, 3}
    assert loaded[1] == evaluate.Verdict("waiting_on_vortex", "No review yet", "claude")
    assert loaded[3] == evaluate.Verdict("waiting_on_contributor", "Asked for tests", "claude")
    assert evaluate.load_verdicts(tmp_path / "missing.json") == {}

    structured_file = tmp_path / "structured.json"
    structured_file.write_text(
        '{"verdicts": ['
        '{"number": 10, "state": "waiting_on_vortex", "reason": "  Still needs review  "},'
        '{"number": true, "state": "waiting_on_vortex", "reason": "ignored"},'
        '{"number": 11, "state": "bogus", "reason": "ignored"},'
        '{"number": 12, "state": "waiting_on_contributor", "reason": "Waiting for tests"}'
        "]}",
        encoding="utf-8",
    )
    structured = evaluate.load_verdicts(structured_file)
    assert structured == {
        10: evaluate.Verdict("waiting_on_vortex", "Still needs review", "claude"),
        12: evaluate.Verdict("waiting_on_contributor", "Waiting for tests", "claude"),
    }

    assert evaluate.fallback_verdict(item(1, reasons=("fuzzer",))).state == evaluate.WAITING_ON_VORTEX
    assert evaluate.fallback_verdict(item(2, assignees=())).state == evaluate.WAITING_ON_VORTEX
    replied = github.Item(
        number=3,
        kind="pr",
        title="t",
        url="https://example.com/3",
        author="alamb",
        created_at="2026-01-01T00:00:00Z",
        assignees=("gatesn",),
        reasons=("external_author",),
        external_at="2026-01-01T00:00:00Z",
        external_by="alamb",
        team_at="2026-01-02T00:00:00Z",
    )
    assert evaluate.fallback_verdict(replied).state == evaluate.WAITING_ON_CONTRIBUTOR


def test_mention_for_and_parse_slack_users():
    users = evaluate.parse_slack_users('{"GatesN": "U111", "myrrc": "U222"}')
    assert evaluate.mention_for(item(1, assignees=("gatesn",)), users, "U999") == "<@U111>"
    assert (
        evaluate.mention_for(item(1, assignees=("someone",)), users, "U999") == "<@U999> (GitHub assignee: `someone`)"
    )
    assert evaluate.mention_for(item(1, assignees=()), users, "U999") == "<@U999> (unassigned)"
    assert evaluate.mention_for(item(1, assignees=()), users, None) == "on-call (unassigned)"


def test_publisher_pings_on_cadence_and_respects_claude_verdict(setup):
    store, fake, _, _ = setup
    now = [1_767_978_000.0]  # 2026-01-09 12:00 EST (Friday)
    publisher = publish.Publisher(store, fake, lambda _: None, now=lambda: now[0])
    cadence = evaluate.Cadence(issue_days=7, pr_weekdays=1, tz="America/New_York")
    users = {"gatesn": "U111"}

    pr_waiting = item(1, kind="pr", assignees=("gatesn",))
    pr_done = item(2, kind="pr", assignees=("gatesn",))
    verdicts = {
        1: evaluate.Verdict(evaluate.WAITING_ON_VORTEX, "Teammate said 'will look tomorrow' 4 days ago.", "claude"),
        2: evaluate.Verdict(evaluate.WAITING_ON_CONTRIBUTOR, "Requested unit tests from contributor.", "claude"),
    }

    assert publisher.sync([pr_waiting, pr_done], ping=True, verdicts=verdicts, cadence=cadence, slack_users=users) == []
    assert [m["key"] for m in fake.messages] == ["1:root", "1:ping:1", "2:root"]
    assert "<@U111>" in fake.messages[1]["text"]
    assert "will look tomorrow" in fake.messages[1]["text"]

    # Same afternoon: not due yet.
    now[0] += 3600
    assert publisher.sync([pr_waiting, pr_done], ping=True, verdicts=verdicts, cadence=cadence, slack_users=users) == []
    assert [m["key"] for m in fake.messages] == ["1:root", "1:ping:1", "2:root"]

    # Next Monday noon: 1 weekday elapsed, so PR #1 pings again.
    now[0] += 3 * 86400
    assert publisher.sync([pr_waiting, pr_done], ping=True, verdicts=verdicts, cadence=cadence, slack_users=users) == []
    assert [m["key"] for m in fake.messages] == ["1:root", "1:ping:1", "2:root", "1:ping:2"]


def test_in_doubt_root_retries_and_refusals_respect_max_new(setup):
    _, fake, _, publisher = setup
    items = [item(n) for n in (1, 2, 3, 4)]
    fake.fail = "lost"
    # Even when root posts time out (in doubt), max_new caps the run at 2 attempts.
    assert len(publisher.sync(items, max_new=2)) == 2
    fake.fail = None
    assert publisher.sync(items, max_new=2) == []
    assert [m["key"] for m in fake.messages] == ["1:root", "2:root"]


def test_failed_first_ping_still_counts_new_root_against_max_new(setup):
    _, fake, _, publisher = setup
    original_post = fake.post

    def fail_ping(text, key, payload, thread_ts) -> object:
        if ":ping:" in key:
            raise slack.SlackError("rate limited", code="rate_limited")
        return original_post(text, key, payload, thread_ts)

    fake.post = fail_ping
    cadence = evaluate.Cadence(ignore_hours=True)
    failures = publisher.sync([item(1), item(2)], max_new=1, ping=True, cadence=cadence)
    assert failures == ["#1: rate limited"]
    assert [m["key"] for m in fake.messages] == ["1:root"]


def test_deleted_slack_parent_resets_thread_on_recovery(setup):
    store, fake, _, publisher = setup
    publisher.sync([item(1)])
    fake.fail = "lost"
    publisher.sync([item(1, assignees=("gatesn",))])

    def missing_thread(key, thread_ts, oldest):
        raise slack.SlackError("conversations.replies: thread_not_found", code="thread_not_found")

    fake.fail = None
    fake.find = missing_thread
    assert publisher.sync([item(1, assignees=("gatesn",))]) == []
    assert [m["key"] for m in fake.messages] == ["1:root", "1:root"]
    assert store.threads[1].root_ts == "2.0"


def test_root_text_neutralizes_multiline_and_triple_backticks():
    bad_item = github.Item(
        number=1,
        kind="issue",
        title="Fix ```code```\nand <tag> & more",
        url="https://github.com/o/r/issues/1",
        author="alamb",
        created_at="2026-01-01T00:00:00Z",
        assignees=(),
        reasons=(github.EXTERNAL_AUTHOR,),
    )
    text = publish.root_text(bad_item)
    assert "```" not in text
    assert "\nand " not in text
    assert "&lt;tag&gt; &amp; more" in text


def test_closed_state_handles_open_closed_merged_and_deleted(monkeypatch):
    def run(payload) -> object:
        return github.closed_state("o/r", 1, query=lambda _q, _v: {"repository": {"item": payload}})

    assert run({"state": "OPEN"}) is None
    assert run({"state": "CLOSED"}) == "closed"
    assert run({"state": "MERGED"}) == "merged"
    assert run(None) == "deleted"

    not_found = json.dumps(
        {"data": {"repository": {"item": None}}, "errors": [{"type": "NOT_FOUND", "message": "missing"}]}
    )
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_a, **_kw: subprocess.CompletedProcess([], 1, stdout=not_found, stderr="not found"),
    )
    assert github.closed_state("o/r", 99999999) == "deleted"


def test_slack_client_errors_retries_and_find(monkeypatch):
    client = slack.Slack("xoxb-test", "C123", attempts=2)
    monkeypatch.setattr(slack.time, "sleep", lambda _s: None)

    def fake_response(payload: dict[str, object]) -> object:
        class Resp:
            def __enter__(self) -> io.BytesIO:
                return io.BytesIO(json.dumps(payload).encode())

            def __exit__(self, *_args: object) -> bool:
                return False

        return Resp()

    retry_headers = email.message.Message()
    retry_headers["Retry-After"] = "0"

    # 429 retries and then succeeds.
    outcomes: list[object] = [
        urllib.error.HTTPError("https://slack.com", 429, "Too Many Requests", retry_headers, None),
        fake_response({"ok": True, "ts": "123.0"}),
    ]

    def fake_urlopen(_req, timeout=30) -> object:
        outcome = outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    monkeypatch.setattr(slack.urllib.request, "urlopen", fake_urlopen)
    assert client.post("hi", "1:root", {"status": "Open"}, None) == "123.0"

    # Exhausted 429 raises rate_limited.
    monkeypatch.setattr(
        slack.urllib.request,
        "urlopen",
        lambda _req, timeout=30: (_ for _ in ()).throw(
            urllib.error.HTTPError("https://slack.com", 429, "Too Many Requests", retry_headers, None)
        ),
    )
    with pytest.raises(slack.SlackError, match="rate limited") as exc_info:
        client.post("hi", "1:root", {"status": "Open"}, None)
    assert exc_info.value.code == "rate_limited"

    # 5xx, IncompleteRead, timeout, and ambiguous ok:false errors raise SlackUnknown.
    for failure in [
        urllib.error.HTTPError("https://slack.com", 502, "Bad Gateway", email.message.Message(), None),
        http.client.IncompleteRead(b"partial"),
        TimeoutError("timed out"),
    ]:
        monkeypatch.setattr(
            slack.urllib.request,
            "urlopen",
            lambda _req, timeout=30, err=failure: (_ for _ in ()).throw(err),
        )
        with pytest.raises(slack.SlackUnknown):
            client.post("hi", "1:root", {"status": "Open"}, None)

    for code in ("fatal_error", "internal_error", "request_timeout", "service_unavailable"):
        monkeypatch.setattr(
            slack.urllib.request,
            "urlopen",
            lambda _req, timeout=30, err=code: fake_response({"ok": False, "error": err}),
        )
        with pytest.raises(slack.SlackUnknown):
            client.post("hi", "1:root", {"status": "Open"}, None)

    # Definite ok:false raises SlackError with its error code.
    monkeypatch.setattr(
        slack.urllib.request,
        "urlopen",
        lambda _req, timeout=30: fake_response({"ok": False, "error": "thread_not_found"}),
    )
    with pytest.raises(slack.SlackError) as exc_info:
        client.find("1:update:1", "100.0", 0.0)
    assert exc_info.value.code == "thread_not_found"

    # find follows pagination and returns the matching event payload.
    pages = iter(
        [
            fake_response({"ok": True, "messages": [{"ts": "1.0"}], "response_metadata": {"next_cursor": "c2"}}),
            fake_response(
                {
                    "ok": True,
                    "messages": [
                        {
                            "ts": "2.0",
                            "metadata": {
                                "event_type": slack.EVENT_TYPE,
                                "event_payload": {"key": "1:root", "status": "Open"},
                            },
                        }
                    ],
                }
            ),
        ]
    )
    monkeypatch.setattr(slack.urllib.request, "urlopen", lambda _req, timeout=30: next(pages))
    assert client.find("1:root", None, 0.0) == {"ts": "2.0", "key": "1:root", "status": "Open"}


def test_waiting_on_contributor_records_check_and_obeys_cadence():
    from datetime import datetime
    from zoneinfo import ZoneInfo

    ny = ZoneInfo("America/New_York")
    mon_10 = datetime(2026, 1, 12, 10, 0, tzinfo=ny).timestamp()
    mon_11 = datetime(2026, 1, 12, 11, 0, tzinfo=ny).timestamp()
    tue_10 = datetime(2026, 1, 13, 10, 0, tzinfo=ny).timestamp()

    now = [mon_10]
    store = publish.MemoryStore()
    fake = FakeSlack()
    pub = publish.Publisher(store, fake, lambda _n: None, now=lambda: now[0])
    cadence = evaluate.Cadence(issue_days=7, pr_weekdays=1, tz="America/New_York")

    pr = item(10, kind="pr", assignees=("gatesn",), external_at="2026-01-12T09:00:00Z")
    waiting = {10: evaluate.Verdict(evaluate.WAITING_ON_CONTRIBUTOR, "Asked for a test.", "claude")}

    # First run creates the root and records the check in root metadata without posting a ping.
    assert pub.sync([pr], ping=True, verdicts=waiting, cadence=cadence) == []
    assert [m["key"] for m in fake.messages] == ["10:root"]
    assert len(fake.updates) == 1
    assert fake.updates[0]["last_checked_at"] == str(int(mon_10))
    assert store.threads[10].last_checked_at == mon_10

    # One hour later, the item is not due for re-evaluation.
    now[0] = mon_11
    assert evaluate.due_items([pr], store.load(), mon_11, cadence) == []
    assert pub.sync([pr], ping=True, verdicts=waiting, cadence=cadence) == []
    assert len(fake.updates) == 1

    # New external activity makes it due immediately even within the same hour.
    pr_replied = item(10, kind="pr", assignees=("gatesn",), external_at="2026-01-12T10:30:00Z")
    assert [i.number for i in evaluate.due_items([pr_replied], store.load(), mon_11, cadence)] == [10]
    vortex_owes = {10: evaluate.Verdict(evaluate.WAITING_ON_VORTEX, "Contributor replied.", "claude")}
    assert pub.sync([pr_replied], ping=True, verdicts=vortex_owes, cadence=cadence) == []
    assert [m["key"] for m in fake.messages] == ["10:root", "10:update:1", "10:ping:1"]

    # On the next weekday, cadence makes it due again.
    assert [i.number for i in evaluate.due_items([pr_replied], store.load(), tue_10, cadence)] == [10]
