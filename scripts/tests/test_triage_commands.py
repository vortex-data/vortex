# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Tests for the thread commands in `.github/scripts/triage/commands.py`."""

import importlib

import pytest

support = importlib.import_module("triage_support")
commands = support.commands


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
    ],
)
def test_valid_commands(text, expected):
    assert commands.parse(text) == expected


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
        ("!snooze soon", "`soon` is not a duration"),
        ("!snooze 1.5h", "`1.5h` is not a duration"),
        ("!snooze 5", "`5` has no unit"),
        ("!snooze 5w", "`w` is not a unit"),
        ("!snooze 0h", "at least `1h` or `1d`"),
        ("!snooze 169h", "more than the limit of 168 working hours"),
        ("!snooze 61d", "more than the limit of 60 working days"),
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
