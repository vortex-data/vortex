"""Working time, which sets reminder cadence, snooze length, and the order of items in a tier.

A working day is a weekday in London. Its window runs from 09:00 in London to 18:00 in New York on
the same calendar date. That is 13 or 14 hours, depending on how the two daylight saving schedules
line up, and the window never crosses midnight in London. Every function returns UTC times.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, time, timedelta
from zoneinfo import ZoneInfo

LONDON = ZoneInfo("Europe/London")
NEW_YORK = ZoneInfo("America/New_York")


def london_date(moment: datetime) -> date:
    """Return the London calendar date of `moment`, which names its working day."""
    return moment.astimezone(LONDON).date()


def window(day: date) -> tuple[datetime, datetime] | None:
    """Return the start and end of the working window on `day`, or None on a weekend."""
    if day.weekday() >= 5:
        return None
    start = datetime.combine(day, time(9), LONDON).astimezone(UTC)
    end = datetime.combine(day, time(18), NEW_YORK).astimezone(UTC)
    return start, end


def in_window(moment: datetime) -> bool:
    bounds = window(london_date(moment))
    return bounds is not None and bounds[0] <= moment < bounds[1]


def next_window_moment(moment: datetime) -> datetime:
    """Return `moment` if it is inside a window, else the start of the next window."""
    day = london_date(moment)
    while True:
        bounds = window(day)
        if bounds is not None and moment < bounds[1]:
            return max(moment.astimezone(UTC), bounds[0])
        day += timedelta(days=1)


def working_seconds(start: datetime, end: datetime) -> float:
    """Return the working time between two moments, in seconds."""
    total = 0.0
    day, last = london_date(start), london_date(end)
    while day <= last:
        bounds = window(day)
        if bounds is not None:
            overlap = min(end, bounds[1]) - max(start, bounds[0])
            total += max(overlap.total_seconds(), 0.0)
        day += timedelta(days=1)
    return total


def add_working_hours(start: datetime, hours: int) -> datetime:
    """Return the moment that is `hours` of working time after `start`."""
    remaining = timedelta(hours=hours)
    moment = next_window_moment(start)
    while True:
        bounds = window(london_date(moment))
        # `next_window_moment` always returns a moment inside a window.
        assert bounds is not None
        available = bounds[1] - moment
        if remaining <= available:
            return moment + remaining
        remaining -= available
        moment = next_window_moment(bounds[1])


def add_working_days(start: datetime, days: int) -> datetime:
    """Return the same point in the window `days` working days after `start`.

    A start outside a window counts from the start of the next window. Windows can differ in length
    by an hour, so the result is clamped to the end of its window.
    """
    moment = next_window_moment(start)
    day = london_date(moment)
    first = window(day)
    assert first is not None
    offset = moment - first[0]
    for _ in range(days):
        day += timedelta(days=1)
        while day.weekday() >= 5:
            day += timedelta(days=1)
    bounds = window(day)
    assert bounds is not None
    return min(bounds[0] + offset, bounds[1])


def working_days_between(start: datetime, end: datetime) -> int:
    """Count the working days after the day of `start`, up to and including the day of `end`."""
    count = 0
    day, last = london_date(start) + timedelta(days=1), london_date(end)
    while day <= last:
        if day.weekday() < 5:
            count += 1
        day += timedelta(days=1)
    return count
