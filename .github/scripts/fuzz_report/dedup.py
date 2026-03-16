"""Deduplication checks for fuzzer crashes."""

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from .extract import CrashInfo


@dataclass
class DedupResult:
    """Result of a deduplication check."""

    duplicate: bool
    check: str | None = None
    confidence: Literal["exact", "high", "medium"] | None = None
    issue_number: int | None = None
    issue_url: str | None = None
    issue_title: str | None = None
    reason: str = ""
    check_order: int | None = None
    # Debug details: what values were compared to produce this result
    debug: dict | None = None

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


def load_issues(issues_path: str | Path) -> list[dict]:
    """Load issues from JSON file."""
    path = Path(issues_path)
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return []


def check_seed_hash(seed_hash: str, issues: list[dict]) -> DedupResult:
    """Check if seed hash exists in any issue body."""
    if not seed_hash or seed_hash == "unknown":
        return DedupResult(duplicate=False, check="seed_hash", reason="No seed hash provided")

    for issue in issues:
        body = issue.get("body", "")
        if seed_hash in body:
            return DedupResult(
                duplicate=True,
                check="seed_hash",
                confidence="exact",
                issue_number=issue["number"],
                issue_url=issue.get("url"),
                issue_title=issue.get("title"),
                reason="Exact seed hash match - same crash input",
                debug={"seed_hash": seed_hash},
            )

    return DedupResult(
        duplicate=False,
        check="seed_hash",
        reason="No matching seed hash found",
        debug={"seed_hash": seed_hash},
    )


def check_panic_location(panic_location: str, issues: list[dict]) -> DedupResult:
    """Check if panic location exists in any issue body."""
    if not panic_location or panic_location == "unknown":
        return DedupResult(
            duplicate=False,
            check="panic_location",
            reason="No panic location provided",
            debug={"panic_location": panic_location or ""},
        )

    # Extract file:line pattern
    match = re.search(r"([^/]+\.rs:\d+)", panic_location)
    file_pattern = match.group(1) if match else panic_location

    for issue in issues:
        body = issue.get("body", "")
        if file_pattern.lower() in body.lower():
            return DedupResult(
                duplicate=True,
                check="panic_location",
                confidence="high",
                issue_number=issue["number"],
                issue_url=issue.get("url"),
                issue_title=issue.get("title"),
                reason=f"Same panic location (file:line): {file_pattern}",
                debug={
                    "panic_location": panic_location,
                    "file_pattern": file_pattern,
                    "matched_issue": issue["number"],
                },
            )

    return DedupResult(
        duplicate=False,
        check="panic_location",
        reason="No matching panic location found",
        debug={
            "panic_location": panic_location,
            "file_pattern": file_pattern,
        },
    )


def check_stack_trace(stack_hash: str, issues: list[dict]) -> DedupResult:
    """Check if stack trace hash exists in any issue body."""
    if not stack_hash or stack_hash == "unknown":
        return DedupResult(
            duplicate=False,
            check="stack_trace",
            reason="No stack hash provided",
            debug={"stack_hash": stack_hash or ""},
        )

    for issue in issues:
        body = issue.get("body", "")
        if stack_hash in body:
            return DedupResult(
                duplicate=True,
                check="stack_trace",
                confidence="high",
                issue_number=issue["number"],
                issue_url=issue.get("url"),
                issue_title=issue.get("title"),
                reason="Same stack trace (top 5 frames match)",
                debug={
                    "stack_hash": stack_hash,
                    "matched_issue": issue["number"],
                },
            )

    return DedupResult(
        duplicate=False,
        check="stack_trace",
        reason="No matching stack trace hash found",
        debug={"stack_hash": stack_hash},
    )


def check_error_pattern(message_hash: str, error_variant: str, issues: list[dict]) -> DedupResult:
    """Check if error pattern exists in any issue body."""
    if not message_hash:
        return DedupResult(
            duplicate=False,
            check="error_pattern",
            reason="No message hash provided",
            debug={"error_variant": error_variant or ""},
        )

    # First try: exact message hash match
    for issue in issues:
        body = issue.get("body", "")
        if message_hash in body:
            return DedupResult(
                duplicate=True,
                check="error_pattern",
                confidence="high",
                issue_number=issue["number"],
                issue_url=issue.get("url"),
                issue_title=issue.get("title"),
                reason="Same error pattern (normalized message match)",
                debug={
                    "message_hash": message_hash,
                    "error_variant": error_variant,
                    "matched_issue": issue["number"],
                },
            )

    return DedupResult(
        duplicate=False,
        check="error_pattern",
        reason="No matching error pattern found",
        debug={
            "message_hash": message_hash,
            "error_variant": error_variant,
        },
    )


def check_duplicate(crash_info: CrashInfo, issues_path: str | Path) -> DedupResult:
    """Run all deduplication checks in order. First match wins."""
    issues = load_issues(issues_path)

    # Summary of extracted values for debugging (attached to every result)
    extraction_summary = {
        "panic_location": crash_info.panic_location,
        "crash_location": crash_info.crash_location,
        "error_variant": crash_info.error_variant,
        "stack_frames_top5": crash_info.stack_frames[:5],
        "normalized_message": crash_info.normalized_message,
    }

    # Check 1: Seed hash (exact duplicate)
    result = check_seed_hash(crash_info.seed_hash, issues)
    if result.duplicate:
        result.check_order = 1
        result.debug = {**(result.debug or {}), "extraction": extraction_summary}
        return result

    # Check 2: Panic location (same crash site)
    result = check_panic_location(crash_info.panic_location, issues)
    if result.duplicate:
        result.check_order = 2
        result.debug = {**(result.debug or {}), "extraction": extraction_summary}
        return result

    # Check 3: Stack trace hash (same call path)
    result = check_stack_trace(crash_info.stack_trace_hash, issues)
    if result.duplicate:
        result.check_order = 3
        result.debug = {**(result.debug or {}), "extraction": extraction_summary}
        return result

    # Check 4: Error pattern (normalized message)
    result = check_error_pattern(crash_info.message_hash, crash_info.error_variant, issues)
    if result.duplicate:
        result.check_order = 4
        result.debug = {**(result.debug or {}), "extraction": extraction_summary}
        return result

    # No matches found
    return DedupResult(
        duplicate=False,
        reason="No duplicate detected by any check",
        debug={"extraction": extraction_summary},
    )
