# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Tests for the Slack messages in `.github/scripts/triage/render.py`."""

import importlib
import json
import re
from dataclasses import replace
from datetime import timedelta
from typing import Any

import pytest

support = importlib.import_module("triage_support")
render, rules, state = support.render, support.rules, support.state
item, event, ago, NOW = support.item, support.event, support.ago, support.NOW
TEAMMATE = support.TEAMMATE
PLAIN = render.People()
MENTIONS = render.People({"alice": "UALICE", "gatesn": "UGATESN"}, "UGATESN", mention=True)
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
    message = render.overview(rows, NOW.date(), NOW, MENTIONS.plain(), "Claude checked 2")
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
        "\u2003\u2003gatesn (on call) · waiting 2d · <https://example.slack.com/archives/C1/p3|thread>"
    )


def test_overview_mentions_nobody_for_items_that_wait_on_the_contributor():
    rows = [row(1), row(2, owes=Owes.CONTRIBUTOR)]
    rows = [replace(r, item=item(r.item.number, assignees=["alice"])) for r in rows]
    texts = texts_of(render.overview(rows, NOW.date(), NOW, MENTIONS, ""))
    assert "  <@UALICE> · waiting" in next(t for t in texts if "|#1 " in t)
    assert "  alice · why" in next(t for t in texts if "|#2 " in t)


def test_people_mention_mapped_logins_and_give_unassigned_items_to_on_call():
    assigned = item(1, assignees=["Alice", "bob"])
    assert MENTIONS.owners(assigned) == "<@UALICE>, bob"
    assert MENTIONS.owners(item(2)) == "<@UGATESN> (on call)"
    # A post names everyone by login, so it notifies nobody.
    assert MENTIONS.plain().owners(assigned) == "Alice, bob"
    assert MENTIONS.plain().owners(item(2)) == "gatesn (on call)"
    assert PLAIN.owners(item(2)) == "unassigned"


def test_overview_hides_snoozed_items_and_counts_them():
    later = (NOW + timedelta(hours=5)).timestamp()
    rows = [row(1), row(2, snooze_until=later), row(3, owes=Owes.CONTRIBUTOR, snooze_until=later)]
    # A snooze that already ended does not hide its item.
    rows.append(row(4, snooze_until=(NOW - timedelta(hours=1)).timestamp()))
    message = render.overview(rows, NOW.date(), NOW, PLAIN, "")
    body = "\n".join(texts_of(message))
    assert "|#1 " in body and "|#4 " in body
    assert "|#2 " not in body and "|#3 " not in body
    assert (
        "🔴 2 High · 🟡 0 Normal · ⚪ 0 Low · 🔵 0 waiting · 😴 2 snoozed" in message.blocks[1]["elements"][0]["text"]
    )


# Mentions are longer than logins, so the limits must hold for the edited overview as well.
@pytest.mark.parametrize("people", [PLAIN, MENTIONS], ids=["plain", "mentions"])
def test_overview_fits_slack_limits_and_cuts_low_first(people):
    long = "x" * 200
    rows = [row(n, Tier.HIGH, title=long) for n in range(1, 41)]
    rows += [row(n, Tier.NORMAL, title=long) for n in range(100, 200)]
    rows += [row(n, Tier.LOW, title=long) for n in range(1000, 3000)]
    rows += [row(n, Tier.NORMAL, owes=Owes.CONTRIBUTOR, title=long) for n in range(5000, 5100)]
    message = render.overview(rows, NOW.date(), NOW, people, "")
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


def first_line(target) -> str:
    """Return the first line of an overview entry as Slack shows it, without link targets."""
    text = render.overview([target], NOW.date(), NOW, PLAIN, "").attachments[0]["blocks"][1]["text"]["text"]
    return re.sub(r"<[^|>]*\|([^>]*)>", r"\1", text.split("\n")[0])


def test_overview_entry_cuts_the_title_to_fit_one_line_and_names_the_commenter():
    # The title is cut so that the line fills the budget exactly, and the emoji counts as two letters.
    line = first_line(row(1, title="y" * 100))
    assert line.endswith("... · alamb")
    assert len(line) + 1 == render.LINE_CHARS
    assert first_line(row(1)) == "📝 #1 A title · alamb"
    # A long login does not squeeze the title below its minimum.
    long_login = replace(row(1), item=item(1, title="y" * 100, author="z" * 39))
    assert "y" * 17 + "... · " in first_line(long_login)
    # On a team item, the contributor is the outside person who commented last.
    team_item = item(2, author=TEAMMATE, events=[event(ago(days=3)), event(ago(days=2), "bob")])
    comment = row(2)
    comment = replace(
        comment, item=team_item, assessment=replace(comment.assessment, category=Category.EXTERNAL_COMMENT)
    )
    text = render.overview([comment], NOW.date(), NOW, PLAIN, "").attachments[0]["blocks"][1]["text"]["text"]
    assert "> · bob\n\u2003\u2003unassigned · waiting 2d · <" in text


def test_titles_and_reasons_are_escaped():
    hostile = row(7, owes=Owes.CONTRIBUTOR, title="Fix ```code```\nand <b> & more", reason="<!channel> look")
    card = render.card(hostile, NOW, PLAIN)
    text = json.dumps(card.blocks)
    assert "```" not in text
    assert "\\nand" not in text
    assert "&lt;b&gt; &amp; more" in text
    assert "&lt;!channel&gt; look" in text
    assert "<!channel>" not in json.dumps(render.overview([hostile], NOW.date(), NOW, PLAIN, "").attachments)


def test_long_titles_are_cut():
    card = render.card(row(8, title="y" * 100), NOW, PLAIN)
    assert "y" * 57 + "..." in card.blocks[0]["text"]["text"]
    assert "y" * 58 not in card.blocks[0]["text"]["text"]


def test_card_shows_the_snooze_and_who_owns_the_item():
    snoozed = row(9, snooze_until=(NOW + timedelta(hours=5)).timestamp())
    detail = render.card(snoozed, NOW, PLAIN).blocks[1]["elements"][0]["text"]
    assert detail.startswith("by alamb · owner unassigned · 😴 until <!date^")
    waiting = render.card(row(9), NOW, PLAIN).blocks[1]["elements"][0]["text"]
    assert "waiting 2d" in waiting


def test_ended_card():
    card = render.ended_card(5, state.Status.MERGED, "Title", "https://github.com/o/r/pull/5")
    assert card.blocks[0]["text"]["text"] == "✅ *<https://github.com/o/r/pull/5|#5 Title>*"
    assert render.ended_card(6, state.Status.DELETED, "", "").blocks[0]["text"]["text"] == "✅ *#6*"


def test_activity_summarizes_events_in_one_line():
    events = [event(ago(hours=3)), event(ago(hours=2), kind="push"), event(ago(hours=1), "kou", "comment")]
    text = render.activity(item(3), events, True, PLAIN).text
    assert text.startswith("💬 New activity from `alamb`, `kou`: 2 comments, 1 push")
    assert text.endswith("· snooze ended")


def test_summary_mentions_each_person_once_and_links_their_item_threads():
    message = render.reminder_summary({"UALICE": [row(1), row(2)], None: [row(3)]})
    assert message.text == "Due today: <@UALICE> 2 items, On-call (not set) 1 item"
    assert (
        "<@UALICE> · 2 items: <https://example.slack.com/archives/C1/p1|#1>, "
        "<https://example.slack.com/archives/C1/p2|#2>"
    ) in json.dumps(message.blocks, ensure_ascii=False)
    # One person's line cannot be split across sections, so a long list is cut short.
    many = render.reminder_summary({"UALICE": [row(n) for n in range(1, 1001)]})
    assert all(len(b["text"]["text"]) <= 3000 for b in many.blocks if b["type"] == "section")
    assert "+975 more" in json.dumps(many.blocks)


def test_item_reminder_mentions_its_owners_and_marks_on_call():
    text = render.item_reminder(row(1), ["UALICE", "UBOB"], False, NOW).text
    assert text.startswith("⏰ <@UALICE>, <@UBOB>, this item waits on you · 🔴 High · waiting 2d")
    on_call = render.item_reminder(row(1), ["UONCALL"], True, NOW).text
    assert on_call.startswith("⏰ <@UONCALL> (on call), this item waits on you")
    assert render.item_reminder(row(1), [None], True, NOW).text.startswith("⏰ On-call (not set), this")


def test_scan_table_shows_titles_as_literal_text():
    hostile = item(1, title="[click](https://example.com) `a``b` | <img>")
    table = support.cli._table(support.cli._assessed([hostile], NOW))
    assert table.splitlines()[-1].endswith("| ``` [click](https://example.com) `a``b` \\| <img> ``` |")


def test_instructions_state_the_limits_and_explain_every_kind():
    text = render.instructions().blocks[0]["text"]["text"]
    assert "at most 168" in text and "at most 60" in text
    assert all(f"{kind.emoji} {kind.meaning}" in text for kind in render.KINDS)
