# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Tests for the working-time math in `.github/scripts/triage/hours.py`."""

import importlib
import re
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest

support = importlib.import_module("triage_support")
hours = support.hours


def utc(*args: int) -> datetime:
    return datetime(*args, tzinfo=UTC)


@pytest.mark.parametrize(
    ("day", "start", "end"),
    [
        # Both cities on winter time.
        (date(2026, 11, 24), utc(2026, 11, 24, 9), utc(2026, 11, 24, 23)),
        # Both cities on summer time.
        (date(2026, 7, 14), utc(2026, 7, 14, 8), utc(2026, 7, 14, 22)),
        # New York changes on 8 March and London on 29 March, so the window is an hour shorter.
        (date(2026, 3, 10), utc(2026, 3, 10, 9), utc(2026, 3, 10, 22)),
        # London changes back on 25 October and New York on 1 November.
        (date(2026, 10, 27), utc(2026, 10, 27, 9), utc(2026, 10, 27, 22)),
    ],
)
def test_window_follows_both_daylight_saving_schedules(day, start, end):
    assert hours.window(day) == (start, end)


def test_weekends_have_no_window():
    assert hours.window(date(2026, 11, 28)) is None
    assert hours.window(date(2026, 11, 29)) is None
    assert not hours.in_window(utc(2026, 11, 28, 12))


def test_window_edges():
    assert hours.in_window(utc(2026, 11, 24, 9))
    assert not hours.in_window(utc(2026, 11, 24, 23))
    # 00:30 UTC on Wednesday is still Tuesday evening in New York, but Wednesday in London, so it
    # falls before Wednesday's window rather than inside Tuesday's.
    assert not hours.in_window(utc(2026, 11, 25, 0, 30))


def _field(text: str) -> range:
    first, last = text.split("-")
    return range(int(first), int(last) + 1)


def test_schedule_covers_the_run_hours_of_every_day():
    workflow = Path(__file__).resolve().parents[2] / ".github" / "workflows" / "triage-sync.yml"
    (cron,) = re.findall(r'cron: "([^"]+)"', workflow.read_text())
    _, hour_field, _, _, day_field = cron.split()
    scheduled_hours, scheduled_days = _field(hour_field), _field(day_field)
    # Two years cover each daylight saving change, and the weeks when only one country has changed.
    day = date(2026, 1, 1)
    while day < date(2028, 1, 1):
        bounds = hours.window(day)
        if bounds is not None:
            start, end = bounds[0] - hours.RUN_MARGIN, bounds[1] + hours.RUN_MARGIN
            last = end - timedelta(minutes=1)
            assert start.date() == last.date() == day
            assert day.isoweekday() in scheduled_days
            assert start.hour in scheduled_hours and last.hour in scheduled_hours
            assert hours.in_run_hours(start) and hours.in_run_hours(last)
            assert not hours.in_run_hours(start - timedelta(minutes=1))
            assert not hours.in_run_hours(end)
        day += timedelta(days=1)
    assert not hours.in_run_hours(utc(2026, 11, 28, 12))


def test_next_window_moment_skips_nights_and_weekends():
    assert hours.next_window_moment(utc(2026, 11, 24, 12)) == utc(2026, 11, 24, 12)
    assert hours.next_window_moment(utc(2026, 11, 24, 6)) == utc(2026, 11, 24, 9)
    assert hours.next_window_moment(utc(2026, 11, 27, 23, 30)) == utc(2026, 11, 30, 9)


def test_working_seconds_skip_nights_and_weekends():
    # Friday 22:00 to Monday 10:00 is one hour on Friday and one hour on Monday.
    assert hours.working_seconds(utc(2026, 11, 27, 22), utc(2026, 11, 30, 10)) == 2 * 3600
    assert hours.working_seconds(utc(2026, 11, 24, 12), utc(2026, 11, 24, 12)) == 0


@pytest.mark.parametrize(
    ("start", "amount", "end"),
    [
        (utc(2026, 11, 24, 12), 5, utc(2026, 11, 24, 17)),
        # One hour is left on Friday, so the rest runs on Monday.
        (utc(2026, 11, 27, 22), 2, utc(2026, 11, 30, 10)),
        # A request at night starts counting at the next window.
        (utc(2026, 11, 25, 2), 1, utc(2026, 11, 25, 10)),
        # The full window of 14 hours ends exactly at its close.
        (utc(2026, 11, 24, 9), 14, utc(2026, 11, 24, 23)),
        # 11 hours on the first day, 11 full days of 14 hours, then 3 hours on the 12th day.
        (utc(2026, 11, 24, 12), 168, utc(2026, 12, 10, 12)),
    ],
)
def test_add_working_hours(start, amount, end):
    assert hours.add_working_hours(start, amount) == end


@pytest.mark.parametrize(
    ("start", "amount", "end"),
    [
        (utc(2026, 11, 24, 12), 1, utc(2026, 11, 25, 12)),
        (utc(2026, 11, 27, 12), 1, utc(2026, 11, 30, 12)),
        (utc(2026, 11, 24, 12), 5, utc(2026, 12, 1, 12)),
        # A request on Saturday counts from Monday's window start.
        (utc(2026, 11, 28, 12), 1, utc(2026, 12, 1, 9)),
        # Friday's window is 14 hours and Monday's is 13, so 13.5 hours in clamps to the end.
        (utc(2026, 10, 23, 21, 30), 1, utc(2026, 10, 26, 22)),
    ],
)
def test_add_working_days(start, amount, end):
    assert hours.add_working_days(start, amount) == end


def test_working_days_between_counts_weekdays_after_the_start_day():
    assert hours.working_days_between(utc(2026, 11, 24, 22), utc(2026, 11, 24, 23)) == 0
    assert hours.working_days_between(utc(2026, 11, 27, 12), utc(2026, 11, 30, 9)) == 1
    assert hours.working_days_between(utc(2026, 11, 24, 12), utc(2026, 12, 1, 12)) == 5


def test_snooze_limits_fit_the_math():
    # The longest snoozes must not overflow or loop forever.
    start = utc(2026, 11, 24, 12)
    assert hours.add_working_hours(start, 168) > start
    assert hours.add_working_days(start, 60) - start < timedelta(days=90)
