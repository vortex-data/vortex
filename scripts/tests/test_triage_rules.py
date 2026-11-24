# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Tests for the deterministic rules in `.github/scripts/triage/rules.py`."""

import importlib
from datetime import timedelta
from typing import Any

import pytest

support = importlib.import_module("triage_support")
rules = support.rules
item, event, ago, NOW = support.item, support.event, support.ago, support.NOW
TEAMMATE, CONTRIBUTOR = support.TEAMMATE, support.CONTRIBUTOR
Category, Owes, Tier = rules.Category, rules.Owes, rules.Tier
WORKING_DAY = rules.BUG_BOOST // 3


def assess(target, verdict=None) -> Any:
    result = rules.assess(target, NOW, verdict)
    assert result is not None
    return result


@pytest.mark.parametrize(
    ("target", "category"),
    [
        # The fuzzer label wins, even on an issue that a bot opened.
        (item(author="fuzz-bot", author_bot=True, labels=["fuzzer"]), Category.FUZZER),
        (item(author="renovate", author_bot=True), None),
        # A bug that a teammate or a bot opened is tracked, but another team or bot issue is not.
        (item(author=TEAMMATE, labels=["bug"]), Category.INTERNAL_BUG),
        (item(author="devin-ai-integration", author_bot=True, labels=["bug"]), Category.INTERNAL_BUG),
        (item(kind="pr", author=TEAMMATE, labels=["bug"]), None),
        (item(labels=["bug"]), Category.EXTERNAL_ISSUE),
        (item(), Category.EXTERNAL_ISSUE),
        (item(kind="pr"), Category.EXTERNAL_PR),
        (item(author=TEAMMATE), None),
        (item(author=TEAMMATE, events=[event(ago(days=1))]), Category.EXTERNAL_COMMENT),
        (item(author=TEAMMATE, events=[event(ago(days=2)), event(ago(days=1), TEAMMATE)]), None),
        (
            item(kind="pr", author=TEAMMATE, events=[event(ago(days=1), kind="review", state="COMMENTED")]),
            Category.EXTERNAL_COMMENT,
        ),
        # Only an issue can be a fuzzer issue.
        (item(kind="pr", author=TEAMMATE, labels=["fuzzer"]), None),
    ],
)
def test_categorize(target, category):
    assert rules.categorize(target) == category


def test_fuzzer_issues_always_wait_on_vortex():
    target = item(author="fuzz-bot", author_bot=True, labels=["fuzzer"], created=ago(days=3))
    result = assess(target)
    assert (result.owes, result.tier, result.owed_since) == (Owes.VORTEX, Tier.HIGH, NOW - timedelta(days=3))
    verdict = rules.Verdict(Owes.CONTRIBUTOR, "no", ago(days=1), NOW)
    assert not assess(target, verdict).from_claude


def test_new_item_waits_on_vortex_since_it_opened():
    result = assess(item(created=ago(days=2)))
    assert (result.owes, result.owes_reason) == (Owes.VORTEX, "no team reply yet")
    assert result.owed_since == NOW - timedelta(days=2)


def test_vortex_owes_since_the_first_contributor_event_after_the_last_team_event():
    target = item(
        created=ago(days=10),
        events=[event(ago(days=5), TEAMMATE), event(ago(days=3)), event(ago(days=1))],
    )
    result = assess(target)
    assert (result.owes, result.owes_reason) == (Owes.VORTEX, "alamb is waiting")
    assert result.owed_since == NOW - timedelta(days=3)


def test_team_reply_last_waits_on_the_contributor():
    result = assess(item(events=[event(ago(days=2), TEAMMATE)], created=ago(days=3)))
    assert (result.owes, result.owes_reason, result.owed_since) == (Owes.CONTRIBUTOR, "gatesn replied 2d ago", None)


def test_same_second_events_do_not_crash():
    # A team comment and a contributor comment share a timestamp, and the contributor's sorts last.
    target = item(created=ago(days=3), events=[event(ago(days=1), TEAMMATE), event(ago(days=1))])
    result = assess(target)
    assert (result.owes, result.owed_since) == (Owes.VORTEX, NOW - timedelta(days=1))


def test_approved_pr_waits_on_vortex_to_merge():
    target = item(kind="pr", created=ago(days=5), events=[event(ago(days=2), TEAMMATE, "review", "APPROVED")])
    result = assess(target)
    assert (result.owes, result.owes_reason) == (Owes.VORTEX, "approved by gatesn, not merged")
    assert result.owed_since == NOW - timedelta(days=2)


def test_changes_requested_waits_on_the_contributor_until_they_act():
    review = event(ago(days=4), TEAMMATE, "review", "CHANGES_REQUESTED")
    waiting = item(kind="pr", created=ago(days=6), events=[review, event(ago(days=3), TEAMMATE)])
    assert (assess(waiting).owes, assess(waiting).owes_reason) == (Owes.CONTRIBUTOR, "changes requested 4d ago")
    pushed = item(kind="pr", created=ago(days=6), events=[review, event(ago(days=1), kind="push")])
    assert (assess(pushed).owes, assess(pushed).owed_since) == (Owes.VORTEX, NOW - timedelta(days=1))


def test_draft_pushes_are_work_in_progress():
    pushes = [event(ago(days=2), kind="push"), event(ago(days=1), kind="push")]
    draft = item(kind="pr", draft=True, created=ago(days=3), events=pushes)
    assert (assess(draft).owes, assess(draft).owes_reason) == (Owes.CONTRIBUTOR, "draft in progress")
    asked = item(kind="pr", draft=True, created=ago(days=3), events=[*pushes, event(ago(hours=2))])
    assert assess(asked).owes is Owes.VORTEX


def test_wait_on_a_former_draft_starts_when_it_was_marked_ready():
    events = [event(ago(days=20), kind="push"), event(ago(days=2), kind="ready")]
    result = assess(item(kind="pr", created=ago(days=21), events=events))
    assert (result.owes, result.owed_since) == (Owes.VORTEX, NOW - timedelta(days=2))


@pytest.mark.parametrize(
    ("target", "tier", "reason"),
    [
        (item(kind="pr", created=ago(days=2)), Tier.HIGH, "new PR, no team reply yet"),
        # A draft is not ready for review, so it never reaches High, even when it is new. It also
        # never drops to Low, however old it is.
        (
            item(kind="pr", draft=True, created=ago(days=2), events=[event(ago(days=1))]),
            Tier.NORMAL,
            "draft PR, contributor active 1d ago",
        ),
        (
            item(kind="pr", draft=True, created=ago(days=200), events=[event(ago(days=90))]),
            Tier.NORMAL,
            "draft PR, contributor active 3mo ago",
        ),
        # A ready PR stays High, however long it has been quiet.
        (
            item(kind="pr", created=ago(days=20), events=[event(ago(days=10), TEAMMATE), event(ago(days=3))]),
            Tier.HIGH,
            "ready PR, contributor active 3d ago",
        ),
        (item(kind="pr", created=ago(days=90)), Tier.HIGH, "ready PR, contributor active 3mo ago"),
        (item(created=ago(days=1)), Tier.HIGH, "new issue, no team reply yet"),
        # So does an external bug issue.
        (
            item(labels=["bug"], created=ago(days=40), events=[event(ago(days=30), TEAMMATE), event(ago(days=20))]),
            Tier.HIGH,
            "bug, active 2w ago",
        ),
        (item(labels=["bug"], created=ago(days=200)), Tier.HIGH, "bug, active 6mo ago"),
        (item(author=TEAMMATE, labels=["bug"], created=ago(days=200)), Tier.NORMAL, "bug filed by the team"),
        (
            item(created=ago(days=40), events=[event(ago(days=30), TEAMMATE), event(ago(days=20))]),
            Tier.NORMAL,
            "issue active 2w ago",
        ),
        (item(created=ago(days=40)), Tier.LOW, "issue quiet for 1mo"),
        (item(author=TEAMMATE, events=[event(ago(days=3))]), Tier.NORMAL, "external comment 3d ago"),
        (item(author=TEAMMATE, events=[event(ago(days=20))]), Tier.LOW, "external comment 2w ago"),
    ],
)
def test_tiers(target, tier, reason):
    result = assess(target)
    assert (result.tier, result.tier_reason) == (tier, reason)


def test_bugs_that_the_team_filed_always_wait_on_vortex():
    target = item(author=TEAMMATE, labels=["bug"], created=ago(days=10), events=[event(ago(days=2), TEAMMATE)])
    result = assess(target)
    assert (result.owes, result.owes_reason, result.owed_since) == (Owes.VORTEX, "open bug", NOW - timedelta(days=10))
    # Claude does not judge them, so a verdict does not apply.
    verdict = rules.Verdict(Owes.CONTRIBUTOR, "asked to wait", ago(days=2), NOW)
    assert assess(target, verdict).owes is Owes.VORTEX


def test_valid_verdict_replaces_the_rule():
    target = item(created=ago(days=5), events=[event(ago(days=3)), event(ago(days=2), TEAMMATE)])
    assert assess(target).owes is Owes.CONTRIBUTOR
    # Claude found that the team's reply was only a placeholder.
    verdict = rules.Verdict(Owes.VORTEX, "placeholder reply", ago(days=2), NOW - timedelta(hours=1))
    result = assess(target, verdict)
    assert (result.owes, result.owes_reason, result.from_claude) == (Owes.VORTEX, "placeholder reply", True)
    # Vortex has owed the next move since the contributor last acted.
    assert result.owed_since == NOW - timedelta(days=3)


def test_contributor_verdict_clears_the_wait():
    target = item(created=ago(days=2))
    verdict = rules.Verdict(Owes.CONTRIBUTOR, "asked to wait", ago(days=2), NOW)
    result = assess(target, verdict)
    assert (result.owes, result.owed_since) == (Owes.CONTRIBUTOR, None)


def test_verdict_goes_stale_with_new_activity_or_age():
    target = item(created=ago(days=10), events=[event(ago(days=1))])
    stale = rules.Verdict(Owes.CONTRIBUTOR, "old", ago(days=10), NOW - timedelta(hours=1))
    assert not assess(target, stale).from_claude
    current = rules.Verdict(Owes.CONTRIBUTOR, "fresh", ago(days=1), NOW - timedelta(days=6))
    assert rules.working_days_between(current.checked_at, NOW) == 4
    assert assess(target, current).from_claude
    expired = rules.Verdict(Owes.CONTRIBUTOR, "old", ago(days=1), NOW - timedelta(days=7))
    assert rules.working_days_between(expired.checked_at, NOW) == 5
    assert not assess(target, expired).from_claude


def test_score_adds_boosts_to_the_working_time_owed():
    base = assess(item(created=ago(days=1))).score
    assert base == rules.working_seconds(NOW - timedelta(days=1), NOW)
    assert assess(item(created=ago(days=1), labels=["bug"])).score == base + 3 * WORKING_DAY
    small = item(kind="pr", created=ago(days=1), changed_lines=100)
    medium = item(kind="pr", created=ago(days=1), changed_lines=500)
    assert assess(small).score == base + 2 * WORKING_DAY
    assert assess(medium).score == base + WORKING_DAY
    assert assess(item(created=ago(days=3))).score > base
