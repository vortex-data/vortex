# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Tests for the GitHub scan in `.github/scripts/triage/github.py`."""

import importlib
import json
import subprocess
from typing import Any

import pytest

support = importlib.import_module("triage_support")
github = support.github

TEAM = {"login": "gatesn", "__typename": "User"}
OUTSIDER = {"login": "alamb", "__typename": "User"}
BOT = {"login": "renovate", "__typename": "Bot"}


def node(author: dict[str, Any] | None = OUTSIDER, number: int = 1, **extra: Any) -> dict[str, Any]:
    return {
        "number": number,
        "title": "Title",
        "url": f"https://github.com/o/r/pull/{number}",
        "createdAt": "2026-11-01T00:00:00Z",
        "body": "Body",
        "author": author,
        "labels": {"nodes": [{"name": "bug"}]},
        "assignees": {"nodes": [{"login": "myrrc"}, {"login": "0ax1"}]},
        "comments": {"nodes": []},
        **extra,
    }


def pr(author: dict[str, Any] | None = OUTSIDER, reviews=(), timeline=(), **extra: Any) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "isDraft": False,
        "additions": 10,
        "deletions": 5,
        "reviews": {"nodes": list(reviews)},
        "timelineItems": {"nodes": list(timeline)},
    }
    return node(author, **{**fields, **extra})


def commit(at: str, user: dict[str, Any] | None, committer: dict[str, Any] | None = None) -> dict[str, Any]:
    fields = {"committedDate": at, "author": {"user": user}, "committer": {"user": committer or user}}
    return {"__typename": "PullRequestCommit", "commit": fields}


def test_parse_orders_human_events_and_drops_bots():
    item = github.parse(
        "issue",
        node(
            comments={
                "nodes": [
                    {"createdAt": "2026-11-03T00:00:00Z", "body": "later", "author": TEAM},
                    {"createdAt": "2026-11-02T00:00:00Z", "body": "first", "author": OUTSIDER},
                    {"createdAt": "2026-11-04T00:00:00Z", "body": "bump", "author": BOT},
                ]
            }
        ),
    )
    assert [(e.kind, e.actor, e.team) for e in item.events] == [
        ("opened", "alamb", False),
        ("comment", "alamb", False),
        ("comment", "gatesn", True),
    ]
    assert item.assignees == ("0ax1", "myrrc")
    assert item.labels == frozenset({"bug"})


def test_parse_pull_request_fields_and_pushes():
    item = github.parse(
        "pr",
        pr(
            reviews=[
                {"submittedAt": "2026-11-02T00:00:00Z", "state": "CHANGES_REQUESTED", "body": "", "author": TEAM},
                # A pending review has no submission time and nobody else can see it.
                {"submittedAt": None, "state": "PENDING", "body": "", "author": TEAM},
            ],
            timeline=[
                # A commit email without a GitHub account is credited to the PR author.
                commit("2026-11-03T00:00:00Z", None),
                {"__typename": "ReadyForReviewEvent", "createdAt": "2026-11-04T00:00:00Z", "actor": OUTSIDER},
            ],
            isDraft=True,
        ),
    )
    assert [(e.kind, e.actor, e.state) for e in item.events] == [
        ("opened", "alamb", ""),
        ("review", "gatesn", "CHANGES_REQUESTED"),
        ("push", "alamb", ""),
        ("ready", "alamb", ""),
    ]
    assert item.draft
    assert item.changed_lines == 15


def test_team_pushes_are_not_replies():
    # "Update branch" by a teammate, a teammate's force-push, or a teammate's rebase of the
    # contributor's commits must not count as an answer.
    item = github.parse(
        "pr",
        pr(
            timeline=[
                commit("2026-11-02T00:00:00Z", TEAM),
                {"__typename": "HeadRefForcePushedEvent", "createdAt": "2026-11-03T00:00:00Z", "actor": TEAM},
                commit("2026-11-04T00:00:00Z", OUTSIDER),
                commit("2026-11-05T00:00:00Z", OUTSIDER, committer=TEAM),
            ]
        ),
    )
    assert [(e.kind, e.actor) for e in item.events] == [("opened", "alamb"), ("push", "alamb")]


def test_deleted_author_is_ghost_and_bot_author_is_flagged():
    assert github.parse("issue", node(None)).author == "ghost"
    bot = github.parse("issue", node(BOT))
    assert bot.author_bot
    assert bot.events == ()


def test_scan_follows_pages_and_sorts():
    calls = []

    def query(text: str, variables: dict[str, Any]) -> dict[str, Any]:
        calls.append(variables.get("cursor"))
        kind = "pullRequests" if "pullRequests" in text else "issues"
        cursor = variables.get("cursor")
        numbers = {("issues", None): [5], ("issues", "next"): [3], ("pullRequests", None): [4]}[kind, cursor]
        more = (kind, cursor) == ("issues", None)
        nodes = [pr(number=n) if kind == "pullRequests" else node(number=n) for n in numbers]
        page = {"pageInfo": {"hasNextPage": more, "endCursor": "next" if more else None}, "nodes": nodes}
        return {"repository": {"items": page}}

    items = github.scan("o/r", query)
    assert [(item.number, item.kind) for item in items] == [(3, "issue"), (4, "pr"), (5, "issue")]
    assert calls == [None, "next", None]


def test_closed_state(monkeypatch):
    def run(payload: dict[str, Any] | None) -> Any:
        return github.closed_state("o/r", 1, query=lambda _q, _v: {"repository": {"item": payload}})

    assert run({"state": "OPEN", "title": "T", "url": "u"}) is None
    assert run({"state": "CLOSED", "title": "T", "url": "u"}) == github.Ended("closed", "T", "u")
    assert run({"state": "MERGED", "title": "T", "url": "u"}) == github.Ended("merged", "T", "u")
    assert run(None) == github.Ended("deleted")

    # A deleted issue makes `gh` fail with partial data and a NOT_FOUND error.
    not_found = json.dumps({"data": {"repository": {"item": None}}, "errors": [{"type": "NOT_FOUND"}]})
    monkeypatch.setattr(
        subprocess, "run", lambda *_a, **_kw: subprocess.CompletedProcess([], 1, stdout=not_found, stderr="")
    )
    assert github.closed_state("o/r", 99999999) == github.Ended("deleted")


def test_gh_graphql_retries_a_cut_response(monkeypatch):
    results = [
        subprocess.CompletedProcess([], 1, stdout="", stderr="unexpected end of JSON input"),
        subprocess.CompletedProcess([], 0, stdout=json.dumps({"data": {"ok": True}}), stderr=""),
    ]
    sleeps: list[float] = []
    monkeypatch.setattr(subprocess, "run", lambda *_a, **_kw: results.pop(0))
    monkeypatch.setattr(github.time, "sleep", sleeps.append)
    assert github.gh_graphql("query", {}) == {"ok": True}
    assert sleeps == [1]


def test_gh_graphql_gives_up_after_three_attempts(monkeypatch):
    failure = subprocess.CompletedProcess([], 1, stdout="", stderr="bad credentials")
    monkeypatch.setattr(subprocess, "run", lambda *_a, **_kw: failure)
    monkeypatch.setattr(github.time, "sleep", lambda _s: None)
    with pytest.raises(RuntimeError, match="bad credentials"):
        github.gh_graphql("query", {})
