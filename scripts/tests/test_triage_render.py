# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Tests for the Slack messages in `.github/scripts/triage/render.py`."""

import importlib
import json
from dataclasses import replace
from datetime import timedelta
from typing import Any

support = importlib.import_module("triage_support")
render, rules, state = support.render, support.rules, support.state
item, event, ago, NOW = support.item, support.event, support.ago, support.NOW
TEAMMATE = support.TEAMMATE
Category, Owes, Tier = rules.Category, rules.Owes, rules.Tier


def row(number, tier=Tier.HIGH, owes=Owes.VORTEX, score=0.0, title="A title", reason="why", snooze_until=0.0) -> Any:
    target = item(number, title=title)
    assessment = rules.Assessment(
        category=Category.EXTERNAL_ISSUE,
        tier=tier,
        tier_reason=reason,
        owes=owes,
        owes_reason=reason,
        owed_since=NOW - timedelta(days=2) if owes is Owes.VORTEX else None,
        last_activity_at=ago(days=2),
        last_external_at=ago(days=2),
        score=score,
    )
    item_state = state.ItemState(number, snooze_until=snooze_until)
    return render.Row(target, assessment, item_state, f"https://example.slack.com/archives/C1/p{number}")


def blocks_of(message) -> int:
    return len(message.blocks) + sum(len(a["blocks"]) for a in message.attachments)


def texts_of(message) -> list[str]:
    return [block["text"]["text"] for a in message.attachments for block in a["blocks"]]


def test_overview_groups_sections_in_order_with_colors():
    rows = [
        row(1, Tier.LOW, score=5),
        row(2, Tier.HIGH, score=1),
        row(3, Tier.HIGH, score=9),
        row(4, Tier.NORMAL, owes=Owes.CONTRIBUTOR),
        row(5, Tier.NORMAL),
    ]
    message = render.overview(rows, NOW.date(), NOW, "gatesn", "Claude checked 2")
    assert message.blocks[0]["text"]["text"] == "📋 Vortex triage · Tue 24 Nov"
    summary = message.blocks[1]["elements"][0]["text"]
    assert summary.startswith("On call gatesn · 🔴 2 High · 🟡 1 Normal · ⚪ 1 Low · 🔵 1 waiting · Claude checked 2")
    assert [a["color"] for a in message.attachments] == [s.color for s in render.SECTIONS]
    high = texts_of(message)[1]
    # The highest score comes first.
    assert high.index("|#3 ") < high.index("|#2 ")
    # Each item takes two lines, and a blank line separates the items.
    assert high.split("\n\n")[0] == (
        "📝 <https://github.com/o/r/issues/3|#3 A title> · alamb\n"
        "\u2003\u2003unassigned · waiting 2d · <https://example.slack.com/archives/C1/p3|thread>"
    )


def test_overview_hides_snoozed_items_and_counts_them():
    later = (NOW + timedelta(hours=5)).timestamp()
    rows = [row(1), row(2, snooze_until=later), row(3, owes=Owes.CONTRIBUTOR, snooze_until=later)]
    # A snooze that already ended does not hide its item.
    rows.append(row(4, snooze_until=(NOW - timedelta(hours=1)).timestamp()))
    message = render.overview(rows, NOW.date(), NOW, "", "")
    body = "\n".join(texts_of(message))
    assert "|#1 " in body and "|#4 " in body
    assert "|#2 " not in body and "|#3 " not in body
    assert (
        "🔴 2 High · 🟡 0 Normal · ⚪ 0 Low · 🔵 0 waiting · 😴 2 snoozed" in message.blocks[1]["elements"][0]["text"]
    )


def test_overview_fits_slack_limits_and_cuts_low_first():
    long = "x" * 200
    rows = [row(n, Tier.HIGH, title=long) for n in range(1, 41)]
    rows += [row(n, Tier.NORMAL, title=long) for n in range(100, 200)]
    rows += [row(n, Tier.LOW, title=long) for n in range(1000, 3000)]
    rows += [row(n, Tier.NORMAL, owes=Owes.CONTRIBUTOR, title=long) for n in range(5000, 5100)]
    message = render.overview(rows, NOW.date(), NOW, "", "")
    assert blocks_of(message) <= 50
    assert all(len(text) <= 3000 for text in texts_of(message))
    body = "\n".join(texts_of(message))
    # Every High row survives, and Low is cut before anything else.
    assert all(f"|#{n} " in body for n in range(1, 41))
    assert "_+" in body
    # Every list block asks Slack to show its whole text, without a "see more" link.
    assert all(b.get("expand") for a in message.attachments for b in a["blocks"][1:])
    low = next(a for a in message.attachments if a["color"] == render.LOW.color)
    assert "more_" in json.dumps(low)
    assert "⚪ 2000 Low" in message.blocks[1]["elements"][0]["text"]


def test_overview_entry_shows_the_full_title_and_names_the_commenter():
    long = row(1, title="y" * 100)
    text = render.overview([long], NOW.date(), NOW, "", "").attachments[0]["blocks"][1]["text"]["text"]
    assert "|#1 " + "y" * 100 + ">" in text
    # On a team item, the contributor is the outside person who commented last.
    team_item = item(2, author=TEAMMATE, events=[event(ago(days=3)), event(ago(days=2), "bob")])
    comment = row(2)
    comment = replace(
        comment, item=team_item, assessment=replace(comment.assessment, category=Category.EXTERNAL_COMMENT)
    )
    text = render.overview([comment], NOW.date(), NOW, "", "").attachments[0]["blocks"][1]["text"]["text"]
    assert "> · bob\n\u2003\u2003unassigned · waiting 2d · <" in text


def test_titles_and_reasons_are_escaped():
    hostile = row(7, owes=Owes.CONTRIBUTOR, title="Fix ```code```\nand <b> & more", reason="<!channel> look")
    card = render.card(hostile, NOW)
    text = json.dumps(card.blocks)
    assert "```" not in text
    assert "\\nand" not in text
    assert "&lt;b&gt; &amp; more" in text
    assert "&lt;!channel&gt; look" in text
    assert "<!channel>" not in json.dumps(render.overview([hostile], NOW.date(), NOW, "", "").attachments)


def test_long_titles_are_cut():
    card = render.card(row(8, title="y" * 100), NOW)
    assert "y" * 57 + "..." in card.blocks[0]["text"]["text"]
    assert "y" * 58 not in card.blocks[0]["text"]["text"]


def test_card_shows_the_snooze_and_who_owns_the_item():
    snoozed = row(9, snooze_until=(NOW + timedelta(hours=5)).timestamp())
    detail = render.card(snoozed, NOW).blocks[1]["elements"][0]["text"]
    assert detail.startswith("by alamb · owner unassigned · 😴 until <!date^")
    waiting = render.card(row(9), NOW).blocks[1]["elements"][0]["text"]
    assert "waiting 2d" in waiting


def test_ended_card():
    card = render.ended_card(5, state.Status.MERGED, "Title", "https://github.com/o/r/pull/5")
    assert card.blocks[0]["text"]["text"] == "✅ *<https://github.com/o/r/pull/5|#5 Title>*"
    assert render.ended_card(6, state.Status.DELETED, "", "").blocks[0]["text"]["text"] == "✅ *#6*"


def test_activity_summarizes_events_in_one_line():
    events = [event(ago(hours=3)), event(ago(hours=2), kind="push"), event(ago(hours=1), "kou", "comment")]
    text = render.activity(item(3), events, snooze_ended=True).text
    assert text.startswith("💬 New activity from `alamb`, `kou`: 2 comments, 1 push")
    assert text.endswith("· snooze ended")


def test_reminder_mentions_the_owner_once_and_lists_items():
    rows = [row(1, Tier.HIGH), row(2, Tier.NORMAL)]
    first = render.reminder("UALICE", rows, True, NOW)
    assert first.text == "<@UALICE>, 2 items wait on you today"
    # A reminder lists the same two-line entries as the overview, with each item's section marker.
    assert first.blocks[1]["text"]["text"].startswith(
        "🔴 📝 <https://github.com/o/r/issues/1|#1 A title> · alamb\n\u2003\u2003unassigned · waiting 2d"
    )
    later = render.reminder(None, rows[:1], False, NOW)
    assert later.text == "On-call (not set), 1 more item is due"
    assert "!snooze 2d" in json.dumps(first.blocks)


def test_long_reminder_is_cut_with_a_count_and_keeps_its_footer():
    rows = [row(n, title="x" * 200) for n in range(1, 1001)]
    message = render.reminder("UALICE", rows, True, NOW)
    assert len(message.blocks) <= render.MAX_BLOCKS
    assert "_+" in message.blocks[-2]["text"]["text"]
    assert "!snooze 2d" in json.dumps(message.blocks[-1])


def test_scan_table_shows_titles_as_literal_text():
    hostile = item(1, title="[click](https://example.com) `a``b` | <img>")
    table = support.cli._table(support.cli._assessed([hostile], NOW))
    assert table.splitlines()[-1].endswith("| ``` [click](https://example.com) `a``b` \\| <img> ``` |")


def test_instructions_state_the_limits_and_explain_every_kind():
    text = render.instructions().blocks[0]["text"]["text"]
    assert "at most 168" in text and "at most 60" in text
    assert all(f"{kind.emoji} {kind.meaning}" in text for kind in render.KINDS)
