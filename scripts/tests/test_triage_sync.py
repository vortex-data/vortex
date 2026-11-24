# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Tests for one run of the triage bot in `.github/scripts/triage/sync.py`.

Each call to `run` starts from the fake channel alone, as each new scheduled job does, so these tests
also check that the state in Slack is enough to continue after a crash.
"""

import importlib
import json
import re
from datetime import UTC, datetime, timedelta
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


def overview_replies(fake) -> list[str]:
    """Return the bot's replies in the overview's thread, without the list of waiting items."""
    return [
        m["text"]
        for m in fake.thread(overview_ts(fake))
        if m.get("user") == support.BOT_USER and not m["metadata"]["event_payload"]["key"].startswith("waiting:")
    ]


def waiting_list(fake) -> str:
    replies = fake.thread(overview_ts(fake))
    (reply,) = [m for m in replies if m["metadata"]["event_payload"]["key"].startswith("waiting:")]
    return json.dumps(reply, ensure_ascii=False)


def test_first_run_posts_roots_and_a_quiet_rerun_reads_nothing():
    fake = FakeSlack()
    assert run(fake, [HIGH, WAITING, LOW]) == []
    assert [len(fake.top(t)) for t in (state.HELP_EVENT, state.OVERVIEW_EVENT)] == [1, 1]
    # Every item gets a root, whatever its tier and whoever owes the next move.
    assert sorted(m["metadata"]["event_payload"]["number"] for m in fake.top(state.ITEM_EVENT)) == ["1", "2", "3"]
    assert "🔴" in card_text(fake, 1) and "🔵" in card_text(fake, 2) and "⚪" in card_text(fake, 3)

    # The second run reads the overview's thread once, because the first run posted the list of
    # waiting items there.
    run(fake, [HIGH, WAITING, LOW], NOW + timedelta(minutes=5))
    fake.calls.clear()
    assert run(fake, [HIGH, WAITING, LOW], NOW + timedelta(minutes=10)) == []
    # Only the overview's "updated" time changes, and no thread is read.
    assert fake.calls == ["history", "update"]


def test_items_that_wait_on_the_contributor_are_listed_in_the_overview_thread():
    fake = FakeSlack()
    run(fake, [HIGH, WAITING])
    overview = json.dumps(fake.messages[overview_ts(fake)], ensure_ascii=False)
    assert "|#1 " in overview and "|#2 " not in overview
    assert "🔵 1 waiting, in the thread" in overview
    assert "|#2 " in waiting_list(fake) and "|#1 " not in waiting_list(fake)
    # The next day's overview gets its own list.
    first = overview_ts(fake)
    run(fake, [HIGH, WAITING], NOW + timedelta(days=1))
    assert overview_ts(fake) != first
    assert "|#2 " in waiting_list(fake)


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
    # The run that confirms the snooze also takes the item out of the overview.
    overview = json.dumps(fake.messages[overview_ts(fake)], ensure_ascii=False)
    assert "issues/1|" not in overview and "😴 1 snoozed" in overview

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


def test_priority_set_in_the_thread_holds_until_auto():
    target = item(1, created=ago(days=1), assignees=["alice"])
    fake = FakeSlack()
    run(fake, [target])
    root = fake.root(1)["ts"]
    fake.say(root, "!priority low", NOW + timedelta(minutes=1))
    run(fake, [target], NOW + HOUR, pings=True)
    assert fake.bot_replies(root) == [
        "⚪ Priority set to Low by <@UALICE>. It stays until someone sends `!priority auto`."
    ]
    assert (payload(fake, 1)["priority"], payload(fake, 1)["priority_by"]) == ("low", "UALICE")
    assert "⚪" in card_text(fake, 1) and "priority set in Slack" in card_text(fake, 1)
    # A Low item gets no reminder, even though the rules make it High.
    assert overview_replies(fake) == []

    fake.say(root, "!priority auto", NOW + HOUR + timedelta(minutes=1))
    fake.say(root, "!priority auto", NOW + HOUR + timedelta(minutes=2))
    run(fake, [target], NOW + 2 * HOUR, pings=True)
    assert fake.bot_replies(root)[1:3] == [
        "🔄 <@UALICE> handed the priority back to the bot.",
        "🔄 <@UALICE>, the bot already sets this item's priority.",
    ]
    assert payload(fake, 1)["priority"] == ""
    assert "🔴" in card_text(fake, 1)
    assert len(item_reminders(fake, 1)) == 1


def test_end_of_a_snooze_reminds_the_owners_once_whoever_owes_the_next_move():
    # A ready PR stays High however old it gets, so it is due every working day.
    owed = item(1, kind="pr", created=ago(days=1), assignees=["alice"])
    waiting = item(2, created=ago(days=3), events=[event(ago(days=2), TEAMMATE)], assignees=["alice"])
    fake = FakeSlack()
    run(fake, [owed, waiting], pings=True)
    for number in (1, 2):
        fake.say(fake.root(number)["ts"], "!snooze 2026-11-28", NOW + timedelta(minutes=number))
    run(fake, [owed, waiting], NOW + HOUR, pings=True)
    assert fake.bot_replies(fake.root(2)["ts"])[0].startswith("😴 Snoozed until <!date^")
    assert float(payload(fake, 2)["snooze_until"]) == datetime(2026, 11, 30, 9, tzinfo=UTC).timestamp()

    # Friday passes without reminders, and the snooze ends on Monday morning.
    friday = datetime(2026, 11, 27, 10, tzinfo=UTC)
    run(fake, [owed, waiting], friday, pings=True)
    assert [len(item_reminders(fake, n)) for n in (1, 2)] == [1, 0]
    monday = datetime(2026, 11, 30, 9, 5, tzinfo=UTC)
    run(fake, [owed, waiting], monday, pings=True)
    reminders = [item_reminders(fake, n) for n in (1, 2)]
    assert [len(r) for r in reminders] == [2, 1]
    assert "the snooze on this item ended · 🔴 High" in reminders[0][1]
    assert reminders[1][0].startswith("⏰ <@UALICE>, the snooze on this item ended · 🔵 Waiting on contributor")
    # The item that waits on the contributor gets no more reminders.
    run(fake, [owed, waiting], monday + 2 * HOUR, pings=True)
    run(fake, [owed, waiting], monday + timedelta(days=1), pings=True)
    assert [len(item_reminders(fake, n)) for n in (1, 2)] == [3, 1]


def test_end_of_a_snooze_reminds_even_after_the_days_reminder():
    target = item(1, kind="pr", created=ago(days=1), assignees=["alice"])
    fake = FakeSlack()
    run(fake, [target], pings=True)
    root = fake.root(1)["ts"]
    assert len(item_reminders(fake, 1)) == 1
    fake.say(root, "!snooze 1h", NOW + timedelta(minutes=5))
    run(fake, [target], NOW + timedelta(minutes=10), pings=True)
    # A reply just before the snooze ends makes the next run read the thread, which holds the
    # day's reminder.
    fake.say(root, "looking at it after lunch", NOW + timedelta(minutes=60))
    run(fake, [target], NOW + timedelta(minutes=70), pings=True)
    reminders = item_reminders(fake, 1)
    assert len(reminders) == 2
    assert "the snooze on this item ended" in reminders[1]


def test_commands_in_the_overview_thread_point_to_the_item_thread():
    fake = FakeSlack()
    run(fake, [HIGH])
    fake.say(overview_ts(fake), "!snooze 2d", NOW + timedelta(minutes=1))
    fake.say(overview_ts(fake), "nice overview", NOW + timedelta(minutes=2))
    run(fake, [HIGH], NOW + HOUR)
    assert overview_replies(fake) == [render.wrong_thread().text]


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


def test_item_whose_author_joins_the_team_is_no_longer_tracked():
    fake = FakeSlack()
    run(fake, [item(6, author="newcomer", created=ago(days=2))])
    assert payload(fake, 6)["status"] == "open"
    # The same pull request, after "newcomer" joined the team list. Nobody answered anything.
    joined = item(6, author=TEAMMATE, created=ago(days=2))
    run(fake, [joined], NOW + HOUR)
    assert payload(fake, 6)["status"] == "untracked"
    assert "No longer tracked" in card_text(fake, 6)


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
    run(fake, [HIGH], claude=sync.Judgment(verdicts={1: verdict}, asked=frozenset({1})))
    assert payload(fake, 1)["verdict_owes"] == "contributor"
    assert "🔵" in card_text(fake, 1) and "asked to wait for a release" in card_text(fake, 1)
    # A later run without Claude keeps using the cached verdict.
    run(fake, [HIGH], NOW + HOUR)
    assert "🔵" in card_text(fake, 1)


def claude_says(owes, priority="normal", reason="a reason") -> Any:
    """Return a fake Claude that gives every item the same answer."""

    def answer(sent) -> str:
        entries = [
            {"number": e["number"], "owes": owes, "reason": reason, "priority": priority, "priority_reason": reason}
            for e in sent["items"]
        ]
        return json.dumps({"verdicts": entries})

    return answer


def judged(fake, items, now, ask) -> Any:
    """Ask the fake Claude about the channel's state before a run, as the sync command does."""
    return support.evaluate.judge(items, sync.load(fake.history()), now, ask)


def overview_text(fake) -> str:
    return json.dumps(fake.messages[overview_ts(fake)]["blocks"], ensure_ascii=False)


def test_claude_priority_changes_get_one_note_each():
    target = item(1, kind="pr", created=ago(days=1))
    fake = FakeSlack()
    run(fake, [target], claude=judged(fake, [target], NOW, claude_says("vortex", "low", "superseded by #9")))
    root = fake.root(1)["ts"]
    assert fake.bot_replies(root) == ["🤖 Claude moved this item from 🔴 High to ⚪ Low: superseded by #9"]
    assert "⚪" in card_text(fake, 1) and "Claude: superseded by #9" in card_text(fake, 1)

    # The same answer after new activity changes nothing, so it gets no note.
    later = item(1, kind="pr", created=ago(days=1), events=[event(ago(minutes=-90))])
    at = NOW + 2 * HOUR
    run(fake, [later], at, claude=judged(fake, [later], at, claude_says("vortex", "low", "still superseded")))
    assert len(fake.bot_replies(root)) == 2  # The activity reply only.

    # A teammate's priority still wins, and Claude's change back to the rules gets a note.
    fake.say(root, "!priority normal", at + timedelta(minutes=1))
    newest = item(1, kind="pr", created=ago(days=1), events=[event(ago(minutes=-150))])
    at = NOW + 4 * HOUR
    run(fake, [newest], at, claude=judged(fake, [newest], at, claude_says("vortex", "high", "a ready PR")))
    assert fake.bot_replies(root)[-1] == (
        "🤖 Claude moved this item from ⚪ Low back to the rules' 🔴 High: a ready PR\n"
        "A teammate's `!priority` still decides."
    )
    assert "🟡" in card_text(fake, 1)


def test_reminders_wait_for_claude_to_see_new_activity():
    # A teammate asked something, and Claude agreed that the contributor owes the answer.
    asked = item(1, created=ago(days=3), events=[event(ago(hours=2), TEAMMATE)])
    fake = FakeSlack()
    run(fake, [asked], pings=True, claude=judged(fake, [asked], NOW, claude_says("contributor")))
    assert item_reminders(fake, 1) == []

    # The contributor only says thanks. The rules alone would remind, but Claude looked within the
    # hour, so the reminder waits for Claude's next look.
    thanks = item(1, created=ago(days=3), events=[event(ago(hours=2), TEAMMATE), event(ago(minutes=-10))])

    def unused(_sent) -> str:
        raise AssertionError("Claude must not be asked within the hour")

    at = NOW + timedelta(minutes=15)
    judgment = judged(fake, [thanks], at, unused)
    assert judgment.held == {1}
    run(fake, [thanks], at, pings=True, claude=judgment)
    assert item_reminders(fake, 1) == []

    # After the hour, Claude looks and finds that nothing waits on Vortex.
    at = NOW + timedelta(minutes=65)
    run(fake, [thanks], at, pings=True, claude=judged(fake, [thanks], at, claude_says("contributor")))
    assert item_reminders(fake, 1) == []
    assert "🔵" in card_text(fake, 1)

    # A failed look does not hold reminders, so a broken Claude never silences the bot.
    more = item(
        1, created=ago(days=3), events=[event(ago(hours=2), TEAMMATE), event(ago(minutes=-10)), event(ago(minutes=-70))]
    )
    at = NOW + timedelta(minutes=135)
    run(fake, [more], at, pings=True, claude=judged(fake, [more], at, lambda _sent: "{}"))
    assert len(item_reminders(fake, 1)) == 1


def test_overview_keeps_showing_a_claude_failure_until_claude_answers():
    fake = FakeSlack()
    run(fake, [HIGH], claude=judged(fake, [HIGH], NOW, lambda _sent: "{}"))
    assert "⚠️ Claude failed, rules only" in overview_text(fake)
    # Within the hour, Claude is not asked, and the overview still shows the failure.
    at = NOW + timedelta(minutes=10)
    run(fake, [HIGH], at, claude=judged(fake, [HIGH], at, claude_says("vortex")))
    assert "⚠️ Claude failed, rules only" in overview_text(fake)
    at = NOW + 2 * HOUR
    run(fake, [HIGH], at, claude=judged(fake, [HIGH], at, claude_says("vortex", "high")))
    assert "Claude checked 1" in overview_text(fake)
    # The next day's overview starts with the latest status, and a run without Claude says so.
    tomorrow = NOW + timedelta(days=1)
    run(fake, [HIGH], tomorrow, claude=judged(fake, [HIGH], tomorrow, None))
    assert "Claude off" in overview_text(fake)


def item_reminders(fake, number) -> list[str]:
    return [text for text in fake.bot_replies(fake.root(number)["ts"]) if text.startswith("⏰")]


def test_reminders_go_to_each_item_thread_after_one_summary():
    alice_new = item(1, created=ago(days=1), assignees=["alice"])
    # bob has no Slack ID, so alice alone owns this item.
    shared = item(2, created=ago(days=2), assignees=["alice", "bob"])
    unassigned = item(3, created=ago(days=1))
    items = [alice_new, shared, unassigned]
    fake = FakeSlack()
    run(fake, items, pings=True)
    assert overview_replies(fake) == ["Due today: <@UALICE> 2 items, <@UONCALL> 1 item"]
    assert item_reminders(fake, 1)[0].startswith("⏰ <@UALICE>, this item waits on you")
    assert item_reminders(fake, 2)[0].startswith("⏰ <@UALICE>, this item waits on you")
    assert item_reminders(fake, 3)[0].startswith("⏰ <@UONCALL> (on call), this item waits on you")
    # Each item in the summary links to its reminder, which opens the item's thread in Slack.
    assert all(reminder_link(fake, n) in summary_blocks(fake) for n in (1, 2, 3))

    run(fake, items, NOW + HOUR, pings=True)
    assert len(overview_replies(fake)) == 1
    assert [len(item_reminders(fake, n)) for n in (1, 2, 3)] == [1, 1, 1]

    # An item that becomes due later in the day gets only its own reminder.
    later = item(4, created=ago(minutes=30), assignees=["alice"])
    run(fake, [*items, later], NOW + 2 * HOUR, pings=True)
    assert len(overview_replies(fake)) == 1
    assert item_reminders(fake, 4)[0].startswith("⏰ <@UALICE>, this item waits on you")


def reminder_link(fake, number) -> str:
    root = fake.root(number)["ts"]
    reply = next(m["ts"] for m in fake.thread(root) if m["text"].startswith("⏰"))
    return f"<https://example.slack.com/archives/C1/p{reply.replace('.', '')}?thread_ts={root}&cid=C1|#{number}>"


def summary_blocks(fake) -> str:
    summary = next(m for m in fake.thread(overview_ts(fake)) if m["text"].startswith("Due today"))
    return json.dumps(summary["blocks"], ensure_ascii=False)


def test_summary_is_kept_up_to_date_without_new_posts():
    items = [item(1, created=ago(days=1), assignees=["alice"]), item(2, created=ago(days=1))]
    fake = FakeSlack()
    run(fake, items, pings=True)
    # A summary from an older version of the bot stored no links to the reminders.
    summary = next(m for m in fake.thread(overview_ts(fake)) if m["text"].startswith("Due today"))
    summary["blocks"] = []
    fake.messages[overview_ts(fake)]["metadata"]["event_payload"].pop("summary")
    for number in (1, 2):
        fake.root(number)["metadata"]["event_payload"].pop("reminder")
    posts = len(fake.posted)
    run(fake, items, NOW + timedelta(minutes=5))
    assert all(reminder_link(fake, n) in summary_blocks(fake) for n in (1, 2))

    # An item reminded later in the day joins the summary through an edit, which pings nobody.
    later = item(3, created=ago(minutes=30), assignees=["alice"])
    run(fake, [*items, later], NOW + HOUR, pings=True)
    assert reminder_link(fake, 3) in summary_blocks(fake)
    assert len(overview_replies(fake)) == 1
    # The new root and its reminder are the only new posts.
    assert len(fake.posted) == posts + 2


def test_every_assignee_with_a_slack_id_is_reminded():
    shared = item(1, created=ago(days=1), assignees=["Bob", "alice", "carol"])
    fake = FakeSlack()
    # carol has no Slack ID, so only alice and bob get the item. Logins match in any case.
    run(fake, [shared], pings=True, users={"alice": "UALICE", "bob": "UBOB"})
    assert overview_replies(fake) == ["Due today: <@UBOB> 1 item, <@UALICE> 1 item"]
    assert item_reminders(fake, 1)[0].startswith("⏰ <@UBOB>, <@UALICE>, this item waits on you")
    assert "<@UBOB>, <@UALICE>, carol" in json.dumps(fake.messages[overview_ts(fake)], ensure_ascii=False)


def test_posts_mention_only_their_recipients_and_edits_add_everyone_else():
    items = [item(1, created=ago(days=1), assignees=["alice", "bob"]), item(2, created=ago(days=1))]
    users = {"alice": "UALICE", "bob": "UBOB"}
    fake = FakeSlack()
    run(fake, items, pings=True, users=users)
    # Slack notifies the people that a post mentions, but not the people that an edit adds. Only
    # reminders mention people in the post: the summary mentions everyone with due items, and an
    # item's reminder mentions its owners.
    expected = {
        overview_ts(fake): {"UALICE", "UBOB", "UONCALL"},
        fake.root(1)["ts"]: {"UALICE", "UBOB"},
        fake.root(2)["ts"]: {"UONCALL"},
    }
    for posted in fake.posted:
        mentioned = set(re.findall(r"<@(\w+)>", json.dumps(posted)))
        reminder = str(posted["metadata"]["event_payload"].get("key", "")).startswith("remind:")
        assert mentioned == (expected[posted["thread_ts"]] if reminder else set())
    assert "<@UALICE>, <@UBOB> · waiting" in json.dumps(fake.root(1), ensure_ascii=False)
    assert "owner <@UONCALL> (on call)" in json.dumps(fake.root(2), ensure_ascii=False)
    overview = json.dumps(fake.messages[overview_ts(fake)], ensure_ascii=False)
    assert "On call <@UONCALL>" in overview and "<@UALICE>, <@UBOB> · waiting" in overview

    # The second run reads the reminders in the item threads, which moves each card's read marker.
    # After that, a run with nothing new edits only the overview, so no card flips between forms.
    run(fake, items, NOW + timedelta(minutes=5), pings=True, users=users)
    fake.calls.clear()
    run(fake, items, NOW + timedelta(minutes=10), pings=True, users=users)
    assert fake.calls.count("update") == 1


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
    assert overview_replies(fake) == ["Due today: <@UALICE> 1 item"]
    assert [len(item_reminders(fake, n)) for n in (1, 2, 3)] == [2, 1, 1]


def test_old_overview_is_marked_replaced_after_a_failed_edit():
    fake = FakeSlack()
    run(fake, [HIGH])
    first = overview_ts(fake)
    tomorrow = NOW + timedelta(days=1)
    fake.faults = [("update", "before")]
    assert run(fake, [HIGH], tomorrow, pings=True) == ["old overview: update: fault"]
    # The failed edit does not stop the reminders, and the next run edits the old overview again.
    assert overview_replies(fake) == ["Due today: <@UONCALL> 1 item"]
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
    assert summary.startswith("On call <@UONCALL> · 🔴 1 High")
    # The summary lists the item, but its own reminder waits until the next run can read its thread.
    assert overview_replies(fake) == ["Due today: <@UONCALL> 1 item"]
    assert item_reminders(fake, 1) == []
    assert run(fake, [HIGH], NOW + 2 * HOUR, pings=True) == []
    assert len(item_reminders(fake, 1)) == 1


def test_failed_summary_holds_back_the_item_reminders_until_the_next_run():
    fake = FakeSlack()
    run(fake, [HIGH])
    fake.faults = [("post", "before")]
    assert run(fake, [HIGH], NOW + HOUR, pings=True) == ["overview: post: fault"]
    assert item_reminders(fake, 1) == []
    assert run(fake, [HIGH], NOW + 2 * HOUR, pings=True) == []
    assert overview_replies(fake) == ["Due today: <@UONCALL> 1 item"]
    assert len(item_reminders(fake, 1)) == 1


def test_no_reminders_without_pings():
    fake = FakeSlack()
    run(fake, [HIGH], pings=False)
    assert overview_replies(fake) == []


def test_one_broken_item_does_not_stop_the_others(monkeypatch):
    original = render.card

    def card(row, now, people) -> Any:
        if row.item.number == 2:
            raise ValueError("secret title")
        return original(row, now, people)

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
    verdict = rules.Verdict(rules.Owes.VORTEX, "reason", ago(days=1), NOW, rules.Tier.LOW, "duplicate")
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
        asked_at=NOW.timestamp(),
        priority=rules.Tier.NORMAL,
        priority_by="U2",
    )
    assert all(isinstance(v, str) for v in original.payload().values())
    assert state.ItemState.from_payload(original.payload()) == original
    overview = state.OverviewState(
        "2026-11-24",
        read_until="1.0",
        reminded="UALICE,oncall",
        waiting="2.0",
        waiting_card="ab",
        claude="Claude checked 3",
        replaced=True,
    )
    assert state.OverviewState.from_payload(overview.payload()) == overview


def test_scheduled_sync_outside_the_run_hours_does_nothing(monkeypatch):
    def scan(repo):
        raise RuntimeError("scanned")

    monkeypatch.setattr(support.cli, "scan", scan)
    monkeypatch.setattr(support.cli, "in_run_hours", lambda now: False)
    args = ["sync", "--repo", "o/r", "--channel", "C1"]
    assert support.cli.main([*args, "--run-hours-only"]) == 0
    # A manual run does not pass the flag, so it syncs at any hour.
    with pytest.raises(RuntimeError, match="scanned"):
        support.cli.main(args)


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
    assert support.cli.purge(fake) == (5, 2, 1)
    remaining = list(fake.messages.values())
    assert sorted(m["text"] for m in remaining) == ["!snooze 1d", "hello channel", "joined"]
