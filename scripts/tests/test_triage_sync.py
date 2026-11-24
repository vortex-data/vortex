# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Tests for one run of the triage bot in `.github/scripts/triage/sync.py`.

Each call to `run` starts from the fake channel alone, as a new hourly job does, so these tests
also check that the state in Slack is enough to continue after a crash.
"""

import importlib
import json
from datetime import timedelta
from typing import Any

import pytest

support = importlib.import_module("triage_support")
github, rules, state, sync, render = support.github, support.rules, support.state, support.sync, support.render
item, event, ago, iso, run = support.item, support.event, support.ago, support.iso, support.run
NOW, TEAMMATE, FakeSlack = support.NOW, support.TEAMMATE, support.FakeSlack
HOUR = timedelta(hours=1)

HIGH = item(1, created=ago(days=1))
WAITING = item(2, created=ago(days=3), events=[event(ago(days=2), TEAMMATE)])
LOW = item(3, created=ago(days=60))


def payload(fake, number) -> Any:
    return fake.root(number)["metadata"]["event_payload"]


def card_text(fake, number) -> str:
    return json.dumps(fake.root(number)["blocks"], ensure_ascii=False)


def overview_ts(fake) -> Any:
    return fake.top(state.OVERVIEW_EVENT)[0]["ts"]


def test_first_run_posts_roots_and_a_quiet_rerun_reads_nothing():
    fake = FakeSlack()
    assert run(fake, [HIGH, WAITING, LOW]) == []
    assert [len(fake.top(t)) for t in (state.HELP_EVENT, state.OVERVIEW_EVENT)] == [1, 1]
    # High and Normal items get a root, whoever owes the next move. Low items do not.
    assert sorted(m["metadata"]["event_payload"]["number"] for m in fake.top(state.ITEM_EVENT)) == ["1", "2"]
    assert "🔴" in card_text(fake, 1) and "🔵" in card_text(fake, 2)

    fake.calls.clear()
    assert run(fake, [HIGH, WAITING, LOW], NOW + timedelta(minutes=5)) == []
    # Only the overview's "updated" time changes, and no thread is read.
    assert fake.calls == ["history", "update"]


def test_snooze_is_confirmed_once_and_shown_on_the_card():
    fake = FakeSlack()
    run(fake, [HIGH])
    root = fake.root(1)["ts"]
    fake.say(root, "<@UBOT> !snooze 2h", NOW + timedelta(minutes=10))
    fake.say(root, "thanks, looking now", NOW + timedelta(minutes=11))
    assert run(fake, [HIGH], NOW + HOUR) == []
    replies = fake.bot_replies(root)
    assert len(replies) == 1
    assert replies[0].startswith("😴 Snoozed until <!date^") and "<@UALICE>" in replies[0]
    saved = payload(fake, 1)
    assert float(saved["snooze_until"]) == (NOW + timedelta(hours=2, minutes=10)).timestamp()
    assert saved["snoozed_by"] == "UALICE"
    assert "😴 until" in card_text(fake, 1)

    run(fake, [HIGH], NOW + 2 * HOUR)
    assert len(fake.bot_replies(root)) == 1


def test_bad_commands_get_help_and_unsnooze_ends_a_snooze():
    fake = FakeSlack()
    run(fake, [HIGH])
    root = fake.root(1)["ts"]
    fake.say(root, "!snooze 5w", NOW + timedelta(minutes=1))
    fake.say(root, "!snooze 1d", NOW + timedelta(minutes=2))
    fake.say(root, "!unsnooze", NOW + timedelta(minutes=3))
    run(fake, [HIGH], NOW + HOUR)
    replies = fake.bot_replies(root)
    assert replies[0].startswith("❓ `w` is not a unit")
    assert replies[1].startswith("😴 Snoozed until")
    assert replies[2] == "🔔 Snooze ended by <@UALICE>."
    assert payload(fake, 1)["snooze_until"] == "0"


def test_commands_in_the_overview_thread_point_to_the_item_thread():
    fake = FakeSlack()
    run(fake, [HIGH])
    fake.say(overview_ts(fake), "!snooze 2d", NOW + timedelta(minutes=1))
    fake.say(overview_ts(fake), "nice overview", NOW + timedelta(minutes=2))
    run(fake, [HIGH], NOW + HOUR)
    assert fake.bot_replies(overview_ts(fake)) == [render.wrong_thread().text]


def test_reply_is_not_repeated_after_the_save_fails():
    fake = FakeSlack()
    run(fake, [HIGH])
    root = fake.root(1)["ts"]
    fake.say(root, "!snooze 1d", NOW + timedelta(minutes=10))
    # The confirmation lands, then the overview edit and the root's save both fail.
    fake.faults = [("update", "before"), ("update", "before")]
    assert run(fake, [HIGH], NOW + HOUR) == ["overview: update: fault", "#1: update: fault"]
    assert payload(fake, 1)["snooze_until"] == "0"

    assert run(fake, [HIGH], NOW + 2 * HOUR) == []
    assert len(fake.bot_replies(root)) == 1
    assert payload(fake, 1)["snooze_until"] != "0"


def test_reply_is_not_repeated_after_a_post_times_out():
    fake = FakeSlack()
    run(fake, [HIGH])
    root = fake.root(1)["ts"]
    fake.say(root, "!snooze 1d", NOW + timedelta(minutes=10))
    # Slack stores the confirmation, but the bot never hears back.
    fake.faults = [("post", "after")]
    assert run(fake, [HIGH], NOW + HOUR) == ["#1: post: timed out"]
    assert run(fake, [HIGH], NOW + 2 * HOUR) == []
    assert len(fake.bot_replies(root)) == 1
    assert payload(fake, 1)["snooze_until"] != "0"


def test_top_level_posts_are_not_repeated_after_a_timeout():
    fake = FakeSlack()
    fake.faults = [("post", "after"), ("post", "after")]
    assert run(fake, [HIGH]) == ["instructions: post: timed out", "#1: post: timed out"]
    assert run(fake, [HIGH], NOW + HOUR) == []
    assert len(fake.top(state.HELP_EVENT)) == 1
    fake.root(1)


def test_deleted_root_is_posted_again():
    fake = FakeSlack()
    run(fake, [HIGH])
    fake.delete(fake.root(1)["ts"])
    run(fake, [HIGH], NOW + HOUR)
    fake.root(1)


def test_new_activity_is_reported_once_and_ends_a_snooze_early():
    fake = FakeSlack()
    run(fake, [HIGH])
    root = fake.root(1)["ts"]
    fake.say(root, "!snooze 2d", NOW + timedelta(minutes=5))
    run(fake, [HIGH], NOW + HOUR)
    assert payload(fake, 1)["snooze_until"] != "0"

    commented = item(1, created=ago(days=1), events=[event(iso(NOW + timedelta(hours=1, minutes=30)))])
    run(fake, [commented], NOW + 2 * HOUR)
    replies = fake.bot_replies(root)
    assert replies[-1].startswith("💬 New activity from `alamb`: 1 comment")
    assert replies[-1].endswith("· snooze ended")
    assert payload(fake, 1)["snooze_until"] == "0"

    run(fake, [commented], NOW + 3 * HOUR)
    assert len(fake.bot_replies(root)) == len(replies)


def test_answered_item_leaves_and_returns_on_the_same_root():
    base = [event(ago(days=1))]
    asked = item(5, author=TEAMMATE, created=ago(days=5), events=base)
    fake = FakeSlack()
    run(fake, [asked])
    assert payload(fake, 5)["status"] == "open"

    answered = item(5, author=TEAMMATE, created=ago(days=5), events=[*base, event(ago(hours=1), TEAMMATE)])
    run(fake, [answered], NOW + HOUR)
    assert payload(fake, 5)["status"] == "answered"
    assert "Answered by a teammate" in card_text(fake, 5)
    assert "#5" not in json.dumps(fake.top(state.OVERVIEW_EVENT)[0]["attachments"])

    again = item(5, author=TEAMMATE, created=ago(days=5), events=[*answered.events[1:], event(iso(NOW + HOUR))])
    run(fake, [again], NOW + 2 * HOUR)
    assert payload(fake, 5)["status"] == "open"
    assert "💬" in card_text(fake, 5)


def test_closed_item_gets_a_final_card_and_is_left_alone():
    fake = FakeSlack()
    run(fake, [HIGH])
    merged = github.Ended("merged", "Final title", "https://github.com/o/r/pull/1")
    run(fake, [], NOW + HOUR, ended=lambda _n: merged)
    assert payload(fake, 1)["status"] == "merged"
    assert "✅" in card_text(fake, 1) and "Final title" in card_text(fake, 1)

    def unexpected(_number):
        raise AssertionError("a final item must not be looked up again")

    assert run(fake, [], NOW + 2 * HOUR, ended=unexpected) == []


def test_reopened_item_is_left_for_the_next_scan():
    fake = FakeSlack()
    run(fake, [HIGH])
    assert run(fake, [], NOW + HOUR, ended=lambda _n: None) == []
    assert payload(fake, 1)["status"] == "open"


def test_claude_verdict_is_cached_in_the_root():
    fake = FakeSlack()
    verdict = rules.Verdict(rules.Owes.CONTRIBUTOR, "asked to wait for a release", HIGH.events[-1].at, NOW)
    run(fake, [HIGH], verdicts={1: verdict})
    assert payload(fake, 1)["verdict_owes"] == "contributor"
    assert "🔵" in card_text(fake, 1) and "asked to wait for a release" in card_text(fake, 1)
    # A later run without Claude keeps using the cached verdict.
    run(fake, [HIGH], NOW + HOUR)
    assert "🔵" in card_text(fake, 1)


def test_reminders_group_items_per_person_and_follow_up_once():
    alice_new = item(1, created=ago(days=1), assignees=["alice"])
    # bob has no Slack ID, so alice alone owns this item.
    shared = item(2, created=ago(days=2), assignees=["alice", "bob"])
    unassigned = item(3, created=ago(days=1))
    items = [alice_new, shared, unassigned]
    fake = FakeSlack()
    run(fake, items, pings=True)
    reminders = fake.bot_replies(overview_ts(fake))
    assert sorted(r.split("\n")[0] for r in reminders) == [
        "<@UALICE>, 2 items wait on you today",
        "<@UONCALL>, 1 item waits on you today",
    ]

    run(fake, items, NOW + HOUR, pings=True)
    assert len(fake.bot_replies(overview_ts(fake))) == 2

    later = item(4, created=ago(minutes=30), assignees=["alice"])
    run(fake, [*items, later], NOW + 2 * HOUR, pings=True)
    reminders = fake.bot_replies(overview_ts(fake))
    assert reminders[-1] == "<@UALICE>, 1 more item is due"
    assert len(reminders) == 3


def test_next_day_replaces_the_overview_and_respects_cadence_and_snoozes():
    high = item(1, created=ago(days=1), assignees=["alice"])
    snoozed = item(2, created=ago(days=1), assignees=["alice"])
    normal = item(3, created=ago(days=40), events=[event(ago(days=30), TEAMMATE), event(ago(days=20))])
    items = [high, snoozed, normal]
    fake = FakeSlack()
    run(fake, items, pings=True)
    first = overview_ts(fake)
    fake.say(fake.root(2)["ts"], "!snooze 3d", NOW + timedelta(minutes=5))

    # Before Wednesday's window opens, Tuesday's overview stays and nobody is reminded.
    early = NOW + timedelta(hours=19)
    run(fake, items, early, pings=True)
    assert overview_ts(fake) == first

    tomorrow = NOW + timedelta(days=1)
    run(fake, items, tomorrow, pings=True)
    second = overview_ts(fake)
    assert second != first
    assert "replaced by" in json.dumps(fake.messages[first]["blocks"])
    # The High item is due every working day. The snoozed one and the Normal one are not.
    assert fake.bot_replies(second) == ["<@UALICE>, 1 item waits on you today"]


def test_old_overview_is_marked_replaced_after_a_failed_edit():
    fake = FakeSlack()
    run(fake, [HIGH])
    first = overview_ts(fake)
    tomorrow = NOW + timedelta(days=1)
    fake.faults = [("update", "before")]
    assert run(fake, [HIGH], tomorrow, pings=True) == ["old overview: update: fault"]
    # The failed edit does not stop the reminders, and the next run edits the old overview again.
    assert fake.bot_replies(overview_ts(fake)) == ["<@UONCALL>, 1 item waits on you today"]
    assert "replaced by" not in json.dumps(fake.messages[first]["blocks"])
    assert run(fake, [HIGH], tomorrow + HOUR) == []
    assert "replaced by" in json.dumps(fake.messages[first]["blocks"])


def test_failed_thread_read_keeps_the_item_in_the_overview_and_reminders():
    fake = FakeSlack()
    run(fake, [HIGH])
    fake.say(fake.root(1)["ts"], "looking into it", NOW + timedelta(minutes=10))
    fake.faults = [("replies", "before")]
    assert run(fake, [HIGH], NOW + HOUR, pings=True) == ["#1: replies: fault"]
    summary = fake.messages[overview_ts(fake)]["blocks"][1]["elements"][0]["text"]
    # On-call has no GitHub login in the users map, so the overview names their Slack ID.
    assert summary.startswith("On call UONCALL · 🔴 1 High")
    assert fake.bot_replies(overview_ts(fake)) == ["<@UONCALL>, 1 item waits on you today"]


def test_no_reminders_without_pings():
    fake = FakeSlack()
    run(fake, [HIGH], pings=False)
    assert fake.bot_replies(overview_ts(fake)) == []


def test_one_broken_item_does_not_stop_the_others(monkeypatch):
    original = render.card

    def card(row, now) -> Any:
        if row.item.number == 2:
            raise ValueError("secret title")
        return original(row, now)

    monkeypatch.setattr(render, "card", card)
    fake = FakeSlack()
    failures = run(fake, [HIGH, item(2, created=ago(days=1))])
    assert len(failures) == 1
    # The log is public, so an unexpected error reports its type and place, never its message.
    assert failures[0].startswith("#2: ValueError at test_triage_sync.py:")
    assert "secret" not in failures[0]
    fake.root(1)


def test_load_keeps_the_newest_duplicate_root():
    older = {"ts": "1.0", "metadata": state.metadata(state.ITEM_EVENT, state.ItemState(1).payload())}
    newer = {"ts": "2.0", "metadata": state.metadata(state.ITEM_EVENT, state.ItemState(1).payload())}
    pilot = {"ts": "3.0", "metadata": {"event_type": "vortex_triage", "event_payload": {"number": "1"}}}
    channel = sync.load([pilot, newer, older])
    assert channel.roots[1].thread.ts == "2.0"
    assert channel.overview is None and channel.help is None


@pytest.mark.parametrize("schema", ["0", "2", None])
def test_state_from_another_schema_is_ignored(schema):
    data = {**state.ItemState(1).payload(), "schema": schema}
    assert state.ItemState.from_payload(data) is None


def test_state_round_trips_through_metadata():
    verdict = rules.Verdict(rules.Owes.VORTEX, "reason", ago(days=1), NOW)
    original = state.ItemState(
        7,
        status=state.Status.ANSWERED,
        reminded_at=NOW.timestamp(),
        snoozed_at=1.5,
        snooze_until=2.25,
        snoozed_by="U1",
        read_until="3.000100",
        seen_external_at=ago(days=1),
        card="abc",
        verdict=verdict,
    )
    assert all(isinstance(v, str) for v in original.payload().values())
    assert state.ItemState.from_payload(original.payload()) == original
    overview = state.OverviewState("2026-11-24", "1.0", "UALICE,oncall", True)
    assert state.OverviewState.from_payload(overview.payload()) == overview


def test_purge_deletes_only_the_bots_messages():
    fake = FakeSlack()
    joined = fake.say(None, "joined", NOW - HOUR, user=support.BOT_USER)
    fake.messages[joined]["subtype"] = "channel_join"
    run(fake, [HIGH])
    root = fake.root(1)["ts"]
    fake.say(root, "!snooze 1d", NOW + timedelta(minutes=1))
    run(fake, [HIGH], NOW + HOUR)
    fake.say(None, "hello channel", NOW + 2 * HOUR)
    # A message that Slack refuses to delete is counted, and the purge goes on.
    assert support.cli.purge(fake) == (4, 2, 1)
    remaining = list(fake.messages.values())
    assert sorted(m["text"] for m in remaining) == ["!snooze 1d", "hello channel", "joined"]
