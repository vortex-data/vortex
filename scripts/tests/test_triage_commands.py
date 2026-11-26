# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Tests for the thread commands in `.github/scripts/triage/commands.py`."""

import importlib
from datetime import UTC, date, datetime, timedelta

import pytest

support = importlib.import_module("triage_support")
commands = support.commands
NOW = support.NOW


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("!snooze 5h", commands.Snooze(5, "h")),
        ("!snooze 2d", commands.Snooze(2, "d")),
        ("!SNOOZE 3 Hours", commands.Snooze(3, "h")),
        ("  !snooze 1day  ", commands.Snooze(1, "d")),
        ("<@U123> !snooze 4hrs", commands.Snooze(4, "h")),
        ("<@U123> <@U456>!snooze 168h", commands.Snooze(168, "h")),
        ("!snooze 60d", commands.Snooze(60, "d")),
        ("!unsnooze", commands.Unsnooze()),
        ("<@U123> !Unsnooze", commands.Unsnooze()),
        ("!priority high", commands.Priority(support.rules.Tier.HIGH)),
        ("<@U123> !Priority LOW", commands.Priority(support.rules.Tier.LOW)),
        ("!priority auto", commands.Priority(None)),
        ("!snooze 2026-12-03", commands.SnoozeUntil(date(2026, 12, 3))),
    ],
)
def test_valid_commands(text, expected):
    assert commands.parse(text) == expected


def test_oversized_snooze_duration_is_rejected_without_raising():
    command = commands.parse("!snooze " + "9" * 5000 + "d")
    assert isinstance(command, commands.Invalid)
    assert "too large" in command.error


@pytest.mark.parametrize(
    "text",
    ["snooze 5h", "Thanks! I will look at this", "", "<@U123> can you snooze this?", "the !snooze command"],
)
def test_conversation_is_not_a_command(text):
    assert commands.parse(text) is None


@pytest.mark.parametrize(
    ("text", "fragment"),
    [
        ("!", "needs a name after `!`"),
        ("!snoze 5h", "`!snoze` is not a command"),
        ("!unsnooze now", "takes nothing after it, but got `now`"),
        ("!snooze", "needs a duration"),
        ("!snooze soon", "`soon` is not a duration or a date"),
        ("!snooze 2026-02-30", "`2026-02-30` is not a date in the calendar"),
        ("!snooze 3/12", "`3/12` is not a duration or a date"),
        ("!snooze 1.5h", "`1.5h` is not a duration"),
        ("!snooze 5", "`5` has no unit"),
        ("!snooze 5w", "`w` is not a unit"),
        ("!snooze 0h", "at least `1h` or `1d`"),
        ("!snooze 169h", "more than the limit of 168 working hours"),
        ("!snooze 61d", "more than the limit of 60 working days"),
        ("!priority", "needs a level: `high`, `normal`, `low`, `auto`"),
        ("!priority urgent", "`urgent` is not a priority"),
        ("!priority high please", "`high please` is not a priority"),
    ],
)
def test_each_mistake_gets_a_specific_error(text, fragment):
    result = commands.parse(text)
    assert isinstance(result, commands.Invalid)
    assert fragment in result.error


def test_errors_quote_input_safely():
    result = commands.parse("!snooze `" + "x" * 100)
    assert isinstance(result, commands.Invalid)
    # The quoted input stays one short code span, whatever the person typed.
    assert result.error.startswith("`'" + "x" * 26 + "...` is not a duration")


@pytest.mark.parametrize(
    ("day", "end"),
    [
        # `NOW` is Tuesday 24 November at 12:00 UTC, so the window on a later day opens at 09:00 UTC.
        (date(2026, 11, 25), datetime(2026, 11, 25, 9, tzinfo=UTC)),
        # A weekend date ends the snooze on Monday morning.
        (date(2026, 11, 28), datetime(2026, 11, 30, 9, tzinfo=UTC)),
    ],
)
def test_date_snooze_ends_when_that_working_day_starts(day, end):
    assert commands.snooze_end(commands.SnoozeUntil(day), NOW) == end


@pytest.mark.parametrize(
    ("day", "requested", "fragment"),
    [
        (date(2026, 11, 24), NOW, "`2026-11-24` has already started"),
        (date(2026, 11, 20), NOW, "`2026-11-20` has already started"),
        (date(2027, 3, 1), NOW, "more than 60 working days away"),
    ],
)
def test_date_snooze_must_start_later_and_within_the_limit(day, requested, fragment):
    result = commands.snooze_end(commands.SnoozeUntil(day), requested)
    assert isinstance(result, commands.Invalid)
    assert fragment in result.error


def test_date_snooze_for_today_works_before_the_window_opens():
    early = datetime(2026, 11, 24, 7, tzinfo=UTC)
    assert commands.snooze_end(commands.SnoozeUntil(date(2026, 11, 24)), early) == early + timedelta(hours=2)
