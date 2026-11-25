"""Decide which items the bot tracks, who owes the next move, and how urgent each item is.

Whether an item is tracked is always deterministic, so a model can never hide an item. A valid
Claude verdict can replace the rules' answer on who owes the next move and the rules' tier. A
teammate's `!priority` replaces both tiers.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum

from .github import COMMENT, READY, REVIEW, Event, Item
from .hours import working_seconds

FUZZER_LABEL = "fuzzer"
BUG_LABEL = "bug"
APPROVED = "APPROVED"
CHANGES_REQUESTED = "CHANGES_REQUESTED"

# Tier thresholds, in calendar days since the latest contributor activity. Bug issues and PRs do
# not age out of their tier, because a teammate snoozes or reprioritizes one that matters less.
NEW_DAYS = 5
NORMAL_ISSUE_DAYS = 30
NORMAL_COMMENT_DAYS = 14

# Score boosts, in seconds of working time. A working day is 13 or 14 hours.
_WORKING_DAY = 14 * 3600
BUG_BOOST = 3 * _WORKING_DAY
SMALL_PR_LINES, SMALL_PR_BOOST = 100, 2 * _WORKING_DAY
MEDIUM_PR_LINES, MEDIUM_PR_BOOST = 500, 1 * _WORKING_DAY


class Category(StrEnum):
    FUZZER = "fuzzer"
    # An issue with the bug label that a teammate or a bot opened.
    INTERNAL_BUG = "internal_bug"
    EXTERNAL_PR = "external_pr"
    EXTERNAL_ISSUE = "external_issue"
    EXTERNAL_COMMENT = "external_comment"


class Tier(StrEnum):
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"


class Owes(StrEnum):
    VORTEX = "vortex"
    CONTRIBUTOR = "contributor"


@dataclass(frozen=True)
class Verdict:
    """Claude's answer to who owes the next move and how urgent the item is, for the activity it
    saw."""

    owes: Owes
    reason: str
    activity_at: str
    checked_at: datetime
    # Claude's tier, or None when Claude kept the rules' tier.
    tier: Tier | None = None
    # Why Claude chose its tier, or why it kept the rules' tier.
    tier_reason: str = ""


@dataclass(frozen=True)
class Assessment:
    category: Category
    tier: Tier
    # Why the item has its tier, for example "ready PR, contributor active 2d ago".
    tier_reason: str
    owes: Owes
    # Why that side owes the next move, for example "changes requested 2d ago".
    owes_reason: str
    # When Vortex began to owe the next move, or None when the contributor owes it.
    owed_since: datetime | None
    # The latest activity by anyone, which a cached verdict must match.
    last_activity_at: str
    # The latest activity by a contributor, or None if there is none.
    last_external_at: str | None
    # Orders items inside a tier, highest first. It is working time in seconds plus boosts.
    score: float
    from_claude: bool = False


def parse_time(timestamp: str) -> datetime:
    return datetime.fromisoformat(timestamp.replace("Z", "+00:00")).astimezone(UTC)


def ago(moment: datetime, now: datetime) -> str:
    """Format the calendar time since `moment` in the largest whole unit, for example `3d`."""
    seconds = max((now - moment).total_seconds(), 0.0)
    for unit, size in (
        ("y", 365 * 86400),
        ("mo", 30 * 86400),
        ("w", 7 * 86400),
        ("d", 86400),
        ("h", 3600),
    ):
        if seconds >= size:
            return f"{int(seconds // size)}{unit}"
    return f"{int(seconds // 60)}m"


def categorize(item: Item) -> Category | None:
    """Return the item's category, or None when the bot does not track it."""
    if item.kind == "issue" and FUZZER_LABEL in item.labels:
        return Category.FUZZER
    if item.kind == "issue" and BUG_LABEL in item.labels and (item.author_team or item.author_bot):
        return Category.INTERNAL_BUG
    if item.author_bot:
        return None
    if not item.author_team:
        return Category.EXTERNAL_PR if item.kind == "pr" else Category.EXTERNAL_ISSUE
    replies = [e for e in item.events if e.kind in (COMMENT, REVIEW)]
    if replies and not replies[-1].team:
        return Category.EXTERNAL_COMMENT
    return None


def _relevant(item: Item) -> list[Event]:
    # Opening a draft or pushing to it is work in progress, not a request for review, so only the
    # contributor's comments can make Vortex owe the next move on a draft.
    if item.kind == "pr" and item.draft:
        return [e for e in item.events if e.team or e.kind in (COMMENT, REVIEW)]
    return list(item.events)


def _owes(item: Item, category: Category, now: datetime) -> tuple[Owes, str, datetime | None]:
    if category is Category.FUZZER:
        return Owes.VORTEX, "open fuzzer issue", parse_time(item.created_at)
    if category is Category.INTERNAL_BUG:
        # Nobody outside the team takes the next step on a bug that the team or a bot filed.
        return Owes.VORTEX, "open bug", parse_time(item.created_at)
    events = _relevant(item)
    # Events are compared by position, not by time, because two events can share a timestamp.
    if item.kind == "pr" and not item.draft:
        decisive = [
            i
            for i, e in enumerate(events)
            if e.team and e.kind == REVIEW and e.state in (APPROVED, CHANGES_REQUESTED)
        ]
        if decisive:
            review = events[decisive[-1]]
            if all(e.team for e in events[decisive[-1] + 1 :]):
                if review.state == APPROVED:
                    return (
                        Owes.VORTEX,
                        f"approved by {review.actor}, not merged",
                        parse_time(review.at),
                    )
                return (
                    Owes.CONTRIBUTOR,
                    f"changes requested {ago(parse_time(review.at), now)} ago",
                    None,
                )
    if not events:
        # Only a draft can get here, because every other item has its opening event.
        return Owes.CONTRIBUTOR, "draft in progress", None
    last = events[-1]
    if last.team:
        return Owes.CONTRIBUTOR, f"{last.actor} replied {ago(parse_time(last.at), now)} ago", None
    team = [i for i, e in enumerate(events) if e.team]
    ready = [i for i, e in enumerate(events) if e.kind == READY]
    # The last event is a contributor's, so an event always follows the last team event. A pull
    # request that was a draft asked for nothing until it was marked ready, so the wait starts there
    # at the earliest.
    first = events[max(team[-1] + 1 if team else 0, ready[-1] if ready else 0)]
    reason = f"{last.actor} is waiting" if team else "no team reply yet"
    return Owes.VORTEX, reason, parse_time(first.at)


def _tier(
    item: Item, category: Category, last_external: datetime | None, now: datetime
) -> tuple[Tier, str]:
    if category is Category.FUZZER:
        return Tier.HIGH, "fuzzer issue"
    if category is Category.INTERNAL_BUG:
        return Tier.NORMAL, "bug filed by the team"
    age = (now - last_external).days if last_external else 10**6
    when = ago(last_external, now) + " ago" if last_external else "never"
    quiet = ago(last_external, now) if last_external else "ever"
    created = parse_time(item.created_at)
    is_new = (now - created).days < NEW_DAYS and not any(e.team for e in item.events)
    if category is Category.EXTERNAL_PR:
        if item.draft:
            # A draft is not ready for review, so it never reaches High.
            return Tier.NORMAL, f"draft PR, contributor active {when}"
        if is_new:
            return Tier.HIGH, "new PR, no team reply yet"
        return Tier.HIGH, f"ready PR, contributor active {when}"
    if category is Category.EXTERNAL_ISSUE:
        if is_new:
            return Tier.HIGH, "new issue, no team reply yet"
        if BUG_LABEL in item.labels:
            return Tier.HIGH, f"bug, active {when}"
        if age < NORMAL_ISSUE_DAYS:
            return Tier.NORMAL, f"issue active {when}"
        return Tier.LOW, f"issue quiet for {quiet}"
    if age < NORMAL_COMMENT_DAYS:
        return Tier.NORMAL, f"external comment {when}"
    return Tier.LOW, f"external comment {when}"


# Categories where nobody outside the team can owe the next move.
ALWAYS_VORTEX = (Category.FUZZER, Category.INTERNAL_BUG)


def verdict_context(item: Item) -> str:
    """Return the normalized PR/label context in which a verdict remains valid."""
    labels = tuple(sorted(" ".join(label.casefold().split()) for label in item.labels))
    return repr((item.draft, labels))


def verdict_applies(verdict: Verdict | None, last_activity_at: str) -> bool:
    """Return True if a cached verdict still applies, because nothing happened after Claude's look.

    A verdict does not expire with age. Claude looks again at an item that nobody is reminded about
    after a few working days, so time alone cannot leave a wrong verdict in place for long.
    """
    return verdict is not None and verdict.activity_at >= last_activity_at


def assess(
    item: Item,
    now: datetime,
    verdict: Verdict | None = None,
    priority: Tier | None = None,
    cached_context: str | None = None,
) -> Assessment | None:
    """Assess a tracked item, or return None when the bot does not track it.

    `priority` is the tier that a teammate chose in Slack, which replaces the rules' and Claude's
    tier.
    """
    category = categorize(item)
    if category is None:
        return None
    external = [e for e in item.events if not e.team]
    last_external = parse_time(external[-1].at) if external else None
    last_activity_at = item.events[-1].at if item.events else item.created_at

    owes, owes_reason, owed_since = _owes(item, category, now)
    valid = (
        verdict
        if verdict_applies(verdict, last_activity_at)
        and (cached_context is None or cached_context == verdict_context(item))
        else None
    )
    # A fuzzer issue or a team's bug always waits on Vortex, so Claude only judges its tier.
    from_claude = valid is not None and category not in ALWAYS_VORTEX
    if from_claude and valid is not None:
        owes, owes_reason = valid.owes, valid.reason
        if owes is Owes.CONTRIBUTOR:
            owed_since = None
        elif owed_since is None:
            # Claude found that the team's last reply did not move the item forward, so Vortex has
            # owed the next move since the contributor last acted.
            owed_since = last_external or parse_time(item.created_at)
    tier, tier_reason = _tier(item, category, last_external, now)
    if valid is not None and valid.tier is not None:
        tier, tier_reason = valid.tier, f"Claude: {valid.tier_reason}"
    if priority is not None:
        tier, tier_reason = priority, "priority set in Slack"

    if owed_since is not None:
        score = working_seconds(owed_since, now)
    else:
        team = [e for e in item.events if e.team]
        score = working_seconds(parse_time(team[-1].at), now) if team else 0.0
    if BUG_LABEL in item.labels:
        score += BUG_BOOST
    if item.kind == "pr" and item.changed_lines <= SMALL_PR_LINES:
        score += SMALL_PR_BOOST
    elif item.kind == "pr" and item.changed_lines <= MEDIUM_PR_LINES:
        score += MEDIUM_PR_BOOST

    return Assessment(
        category=category,
        tier=tier,
        tier_reason=tier_reason,
        owes=owes,
        owes_reason=owes_reason,
        owed_since=owed_since,
        last_activity_at=last_activity_at,
        last_external_at=external[-1].at if external else None,
        score=score,
        from_claude=from_claude,
    )
