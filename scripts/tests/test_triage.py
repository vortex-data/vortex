# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Tests for the triage scanner and Slack publisher in `.github/scripts/triage`.

Run with:

    uv run --no-project --with pytest pytest scripts/tests/test_triage.py
"""

import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / ".github" / "scripts"))
github = importlib.import_module("triage.github")
publish = importlib.import_module("triage.publish")
slack = importlib.import_module("triage.slack")

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


class FakeSlack:
    """Records posts like Slack would, and can fail before or after a post lands."""

    def __init__(self):
        self.messages = []
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
