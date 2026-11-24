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
EMPTY = sync.Channel({}, None, None)


def channel_with(number, verdict) -> Any:
    item_state = state.ItemState(number, verdict=verdict)
    thread = sync.Thread("1.0", "")
    return sync.Channel({number: sync.Root(thread, item_state, item_state)}, None, None)


def test_candidates_skip_low_fuzzer_and_cached_items_and_rank_by_tier():
    high = item(1, created=ago(days=1))
    normal = item(2, created=ago(days=40), events=[event(ago(days=30), TEAMMATE), event(ago(days=20))])
    low = item(3, created=ago(days=60))
    fuzzer = item(4, author="fuzz-bot", author_bot=True, labels=["fuzzer"])
    chosen = evaluate.candidates([normal, low, fuzzer, high], EMPTY, NOW)
    assert [target.number for target, _ in chosen] == [1, 2]

    cached = rules.Verdict(rules.Owes.VORTEX, "still waiting", high.events[-1].at, NOW)
    assert [t.number for t, _ in evaluate.candidates([high], channel_with(1, cached), NOW)] == []


def test_candidates_are_capped():
    items = [item(n, created=ago(days=1)) for n in range(1, 50)]
    assert len(evaluate.candidates(items, EMPTY, NOW)) == evaluate.MAX_CANDIDATES


def test_payload_marks_roles_and_rule_answer():
    target = item(1, created=ago(days=3), events=[event(ago(days=2), TEAMMATE)])
    payload = evaluate.payload(evaluate.candidates([target], EMPTY, NOW), NOW)
    entry = payload["items"][0]
    assert entry["rule_says"] == {"owes": "contributor", "reason": "gatesn replied 2d ago"}
    assert [e["role"] for e in entry["activity"]] == ["contributor", "team"]


def test_load_verdicts_keeps_only_valid_entries_for_items_that_were_sent():
    seen = {"1": ago(days=1), "2": ago(days=2)}
    raw = json.dumps(
        {
            "verdicts": [
                {"number": 1, "owes": "vortex", "reason": "  placeholder\nreply  "},
                {"number": 2, "owes": "maybe", "reason": "bad state"},
                {"number": 3, "owes": "vortex", "reason": "not sent"},
                {"number": True, "owes": "vortex", "reason": "bool"},
                {"number": 2, "owes": "contributor", "reason": ""},
                "junk",
            ]
        }
    )
    verdicts = evaluate.load_verdicts(raw, seen, NOW)
    assert verdicts == {1: rules.Verdict(rules.Owes.VORTEX, "placeholder reply", ago(days=1), NOW)}
    assert evaluate.load_verdicts("not json", seen, NOW) == {}
    assert evaluate.load_verdicts("[]", seen, NOW) == {}


def test_judge_reports_its_status(capsys):
    target = item(1, created=ago(days=1))

    def answer(payload) -> str:
        number = payload["items"][0]["number"]
        return json.dumps({"verdicts": [{"number": number, "owes": "vortex", "reason": "new"}]})

    def broken(_payload):
        raise evaluate.ClaudeError("HTTP 401 authentication_error")

    def crashed(_payload):
        raise RuntimeError("secret title")

    assert evaluate.judge([target], EMPTY, NOW, None) == ({}, "Claude off")
    assert evaluate.judge([], EMPTY, NOW, answer) == ({}, "")
    verdicts, status = evaluate.judge([target], EMPTY, NOW, answer)
    assert (list(verdicts), status) == ([1], "Claude checked 1")
    assert evaluate.judge([target], EMPTY, NOW, broken) == ({}, evaluate.FAILED)
    assert evaluate.judge([target], EMPTY, NOW, lambda _p: "{}") == ({}, evaluate.FAILED)
    # Any other error also falls back to the rules, and the public log gets only its type.
    assert evaluate.judge([target], EMPTY, NOW, crashed) == ({}, evaluate.FAILED)
    log = capsys.readouterr().out
    assert "Claude failed: HTTP 401 authentication_error" in log
    assert "Claude failed: RuntimeError" in log
    assert "secret" not in log


def test_new_verdict_is_cached_until_new_activity():
    target = item(1, created=ago(days=1))
    verdict = rules.Verdict(rules.Owes.CONTRIBUTOR, "asked to wait", target.events[-1].at, NOW)
    channel = channel_with(1, verdict)
    assert evaluate.candidates([target], channel, NOW + timedelta(hours=1)) == []
    newer = item(1, created=ago(days=1), events=[event(ago(minutes=1))])
    assert len(evaluate.candidates([newer], channel, NOW)) == 1


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
