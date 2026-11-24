# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Tests for choosing Claude's candidates and reading its verdicts in `.github/scripts/triage`."""

import importlib
import json
from datetime import timedelta
from typing import Any

import pytest

support = importlib.import_module("triage_support")
evaluate, rules, state, sync = support.evaluate, support.rules, support.state, support.sync
item, event, ago, NOW = support.item, support.event, support.ago, support.NOW
TEAMMATE = support.TEAMMATE
Tier = rules.Tier
EMPTY = sync.Channel({}, None, None)
HOUR = timedelta(hours=1)


def channel_with(number, verdict=None, **fields: Any) -> Any:
    item_state = state.ItemState(number, verdict=verdict, **fields)
    thread = sync.Thread("1.0", "")
    return sync.Channel({number: sync.Root(thread, item_state, item_state)}, None, None)


def chosen_numbers(items, channel, now=NOW) -> list[int]:
    return [target.number for target, _ in evaluate.candidates(items, channel, now)]


def test_new_items_of_every_tier_are_candidates_most_urgent_first():
    high = item(1, created=ago(days=1))
    normal = item(2, created=ago(days=40), events=[event(ago(days=30), TEAMMATE), event(ago(days=20))])
    low = item(3, created=ago(days=60))
    fuzzer = item(4, author="fuzz-bot", author_bot=True, labels=["fuzzer"])
    bug = item(5, author=TEAMMATE, labels=["bug"])
    untracked = item(6, author=TEAMMATE)
    assert chosen_numbers([normal, low, fuzzer, high, bug, untracked], EMPTY) == [4, 1, 2, 5, 3]


def test_candidates_are_capped():
    items = [item(n, created=ago(days=1)) for n in range(1, 50)]
    assert len(evaluate.candidates(items, EMPTY, NOW)) == evaluate.MAX_CANDIDATES


def test_claude_looks_again_after_new_activity_but_at_most_once_an_hour():
    target = item(1, created=ago(days=1))
    seen = rules.Verdict(rules.Owes.VORTEX, "new", target.events[-1].at, NOW - HOUR)
    channel = channel_with(1, seen, asked_at=(NOW - HOUR).timestamp())
    assert chosen_numbers([target], channel) == []
    newer = item(1, created=ago(days=1), events=[event(ago(minutes=5))])
    assert chosen_numbers([newer], channel) == [1]
    assert chosen_numbers([newer], channel, NOW - timedelta(minutes=1)) == []
    # A failed look also waits, so a broken Claude costs one request an hour.
    failed = channel_with(1, asked_at=(NOW - timedelta(minutes=30)).timestamp())
    assert chosen_numbers([target], failed) == []
    assert chosen_numbers([target], failed, NOW + timedelta(minutes=30)) == [1]


def test_claude_skips_snoozed_items_and_looks_when_a_snooze_ends():
    target = item(1, created=ago(days=1))
    seen = rules.Verdict(rules.Owes.VORTEX, "new", target.events[-1].at, NOW - 3 * HOUR)
    asked = (NOW - 3 * HOUR).timestamp()
    snoozed = channel_with(1, seen, asked_at=asked, snooze_until=(NOW + HOUR).timestamp())
    newer = item(1, created=ago(days=1), events=[event(ago(minutes=5))])
    assert chosen_numbers([newer], snoozed) == []
    ended = channel_with(1, seen, asked_at=asked, snooze_until=(NOW - HOUR).timestamp())
    assert chosen_numbers([target], ended) == [1]


@pytest.mark.parametrize(
    ("target", "looks_again"),
    [
        # Nobody is reminded about an item that waits on the contributor or a Low item.
        (item(1, created=ago(days=20), events=[event(ago(days=19), TEAMMATE)]), True),
        (item(1, created=ago(days=60)), True),
        # A High item reminds its owners, so they look at it instead.
        (item(1, kind="pr", created=ago(days=20)), False),
    ],
)
def test_claude_looks_again_at_unreminded_items_every_few_working_days(target, looks_again):
    assessment = rules.assess(target, NOW)
    assert assessment is not None
    seen = rules.Verdict(assessment.owes, "same", target.events[-1].at, NOW)
    four = channel_with(1, seen, asked_at=(NOW - timedelta(days=6)).timestamp())
    assert support.hours.working_days_between(NOW - timedelta(days=6), NOW) == 4
    assert chosen_numbers([target], four) == []
    five = channel_with(1, seen, asked_at=(NOW - timedelta(days=7)).timestamp())
    assert chosen_numbers([target], five) == ([1] if looks_again else [])


def test_reminders_wait_only_for_an_answered_look_at_older_activity():
    target = item(1, created=ago(days=1), events=[event(ago(minutes=5))])
    seen = rules.Verdict(rules.Owes.VORTEX, "new", ago(days=1), NOW - timedelta(minutes=30))
    answered = channel_with(1, seen, asked_at=(NOW - timedelta(minutes=30)).timestamp())
    assert evaluate.held([target], answered, NOW, frozenset()) == {1}
    # Claude looks at it in this run, so nothing waits.
    assert evaluate.held([target], answered, NOW, frozenset({1})) == set()
    # The last look failed, so the rules decide until Claude answers again.
    failed = channel_with(1, seen, asked_at=(NOW - timedelta(minutes=10)).timestamp())
    assert evaluate.held([target], failed, NOW, frozenset()) == set()
    current = rules.Verdict(rules.Owes.VORTEX, "new", ago(minutes=5), NOW - timedelta(minutes=1))
    assert evaluate.held([target], channel_with(1, current), NOW, frozenset()) == set()


def test_payload_marks_roles_and_rule_answer():
    target = item(1, created=ago(days=3), labels=["bug"], events=[event(ago(days=2), TEAMMATE)])
    payload = evaluate.payload(evaluate.candidates([target], EMPTY, NOW), NOW)
    entry = payload["items"][0]
    assert entry["rule_says"] == {
        "owes": "contributor",
        "reason": "gatesn replied 2d ago",
        "priority": "high",
        "priority_reason": "bug, active 3d ago",
    }
    assert entry["labels"] == ["bug"]
    assert [e["role"] for e in entry["activity"]] == ["contributor", "team"]


def test_load_verdicts_keeps_only_valid_entries_for_items_that_were_sent():
    seen = {"1": evaluate.Sent(ago(days=1), Tier.HIGH), "2": evaluate.Sent(ago(days=2), Tier.NORMAL)}
    raw = json.dumps(
        {
            "verdicts": [
                {
                    "number": 1,
                    "owes": "vortex",
                    "reason": "  placeholder\nreply  ",
                    "priority": "high",
                    "priority_reason": "a ready PR",
                },
                {"number": 2, "owes": "maybe", "reason": "bad state"},
                {"number": 3, "owes": "vortex", "reason": "not sent"},
                {"number": True, "owes": "vortex", "reason": "bool"},
                {"number": 2, "owes": "contributor", "reason": ""},
                "junk",
            ]
        }
    )
    verdicts = evaluate.load_verdicts(raw, seen, NOW)
    # Claude kept the rules' tier, so the verdict leaves the tier to the rules.
    assert verdicts == {1: rules.Verdict(rules.Owes.VORTEX, "placeholder reply", ago(days=1), NOW, None, "a ready PR")}
    assert evaluate.load_verdicts("not json", seen, NOW) == {}
    assert evaluate.load_verdicts("[]", seen, NOW) == {}


@pytest.mark.parametrize(
    ("priority", "reason", "tier"),
    [
        ("low", "duplicate of #3", Tier.LOW),
        ("normal", "duplicate of #3", None),
        # A malformed priority keeps the rules' tier, but not the rest of the verdict.
        ("urgent", "duplicate of #3", None),
        ("low", "", None),
    ],
)
def test_load_verdicts_keeps_a_tier_only_when_claude_changes_it(priority, reason, tier):
    seen = {"1": evaluate.Sent(ago(days=1), Tier.NORMAL)}
    entry = {"number": 1, "owes": "vortex", "reason": "r", "priority": priority, "priority_reason": reason}
    (verdict,) = evaluate.load_verdicts(json.dumps({"verdicts": [entry]}), seen, NOW).values()
    assert verdict.tier is tier


def test_judge_reports_its_status(capsys):
    target = item(1, created=ago(days=1))

    def answer(payload) -> str:
        number = payload["items"][0]["number"]
        entry = {"number": number, "owes": "vortex", "reason": "new", "priority": "high", "priority_reason": "new"}
        return json.dumps({"verdicts": [entry]})

    def broken(_payload):
        raise evaluate.ClaudeError("HTTP 401 authentication_error")

    def crashed(_payload):
        raise RuntimeError("secret title")

    assert evaluate.judge([target], EMPTY, NOW, None) == sync.Judgment(on=False)
    assert evaluate.judge([], EMPTY, NOW, answer) == sync.Judgment()
    judgment = evaluate.judge([target], EMPTY, NOW, answer)
    assert (list(judgment.verdicts), judgment.status, judgment.asked) == ([1], "Claude checked 1", {1})
    failed = sync.Judgment(status=evaluate.FAILED, asked=frozenset({1}))
    assert evaluate.judge([target], EMPTY, NOW, broken) == failed
    assert evaluate.judge([target], EMPTY, NOW, lambda _p: "{}") == failed
    # Any other error also falls back to the rules, and the public log gets only its type.
    assert evaluate.judge([target], EMPTY, NOW, crashed) == failed
    log = capsys.readouterr().out
    assert "Claude failed: HTTP 401 authentication_error" in log
    assert "Claude failed: RuntimeError" in log
    assert "secret" not in log


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("", {}),
        ('{"GatesN": "U1", " ": "U2", "x": ""}', {"gatesn": "U1"}),
    ],
)
def test_parse_slack_users(raw, expected):
    assert evaluate.parse_slack_users(raw) == expected


def test_parse_slack_users_rejects_a_list():
    with pytest.raises(ValueError, match="JSON object"):
        evaluate.parse_slack_users("[]")
