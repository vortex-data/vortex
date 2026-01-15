"""Tests for dedup module."""

import json
import tempfile
from pathlib import Path

import pytest

from ..dedup import (
    DedupResult,
    check_duplicate,
    check_error_pattern,
    check_panic_location,
    check_seed_hash,
    check_stack_trace,
)
from ..extract import CrashInfo


# Test fixture: existing issues
EXISTING_ISSUES = [
    {
        "number": 100,
        "title": "Fuzzing Crash: IndexOutOfBounds in file_io",
        "body": """## Fuzzing Crash Report

**Seed Hash**: `aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa`
**Stack Hash**: `bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb`
**Message Hash**: `cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc`

**Panic Location**: `vortex-array/src/compute/slice.rs:142`
**Error Variant**: `IndexOutOfBounds`
""",
        "url": "https://github.com/example/repo/issues/100",
    },
    {
        "number": 101,
        "title": "Fuzzing Crash: ScalarMismatch in array_ops",
        "body": """## Fuzzing Crash Report

**Seed Hash**: `dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd`

**Error Variant**: `ScalarMismatch`
""",
        "url": "https://github.com/example/repo/issues/101",
    },
]


@pytest.fixture
def issues_file():
    """Create a temporary issues JSON file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(EXISTING_ISSUES, f)
        f.flush()
        yield f.name
    Path(f.name).unlink()


class TestCheckSeedHash:
    def test_match_found(self):
        result = check_seed_hash(
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            EXISTING_ISSUES,
        )
        assert result.duplicate is True
        assert result.check == "seed_hash"
        assert result.confidence == "exact"
        assert result.issue_number == 100

    def test_no_match(self):
        result = check_seed_hash(
            "0000000000000000000000000000000000000000000000000000000000000000",
            EXISTING_ISSUES,
        )
        assert result.duplicate is False
        assert result.check == "seed_hash"

    def test_unknown_hash(self):
        result = check_seed_hash("unknown", EXISTING_ISSUES)
        assert result.duplicate is False


class TestCheckPanicLocation:
    def test_match_found(self):
        result = check_panic_location(
            "vortex-array/src/compute/slice.rs:142",
            EXISTING_ISSUES,
        )
        assert result.duplicate is True
        assert result.check == "panic_location"
        assert result.confidence == "high"
        assert result.issue_number == 100

    def test_partial_match(self):
        # Should match on file:line pattern
        result = check_panic_location("slice.rs:142", EXISTING_ISSUES)
        assert result.duplicate is True

    def test_no_match(self):
        result = check_panic_location("other/file.rs:999", EXISTING_ISSUES)
        assert result.duplicate is False


class TestCheckStackTrace:
    def test_match_found(self):
        result = check_stack_trace(
            "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            EXISTING_ISSUES,
        )
        assert result.duplicate is True
        assert result.check == "stack_trace"
        assert result.confidence == "high"

    def test_no_match(self):
        result = check_stack_trace(
            "0000000000000000000000000000000000000000000000000000000000000000",
            EXISTING_ISSUES,
        )
        assert result.duplicate is False


class TestCheckErrorPattern:
    def test_message_hash_match(self):
        result = check_error_pattern(
            "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
            "IndexOutOfBounds",
            EXISTING_ISSUES,
        )
        assert result.duplicate is True
        assert result.confidence == "high"

    def test_variant_match(self):
        result = check_error_pattern(
            "nomatchhash",
            "ScalarMismatch",
            EXISTING_ISSUES,
        )
        assert result.duplicate is True
        assert result.confidence == "medium"
        assert result.issue_number == 101

    def test_no_match(self):
        result = check_error_pattern("nomatch", "UnknownVariant", EXISTING_ISSUES)
        assert result.duplicate is False


class TestCheckDuplicate:
    def test_seed_hash_match_first(self, issues_file):
        """Seed hash match should return immediately."""
        crash_info = CrashInfo(
            panic_location="other.rs:1",
            panic_message="other message",
            error_variant="Other",
            stack_frames=["other"],
            stack_trace_hash="xxx",
            normalized_message="other",
            message_hash="yyy",
            crash_type="crash",
            seed_hash="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        )

        result = check_duplicate(crash_info, issues_file)
        assert result.duplicate is True
        assert result.check == "seed_hash"
        assert result.check_order == 1

    def test_panic_location_match_second(self, issues_file):
        """Panic location should be checked after seed hash."""
        crash_info = CrashInfo(
            panic_location="vortex-array/src/compute/slice.rs:142",
            panic_message="other message",
            error_variant="Other",
            stack_frames=["other"],
            stack_trace_hash="xxx",
            normalized_message="other",
            message_hash="yyy",
            crash_type="crash",
            seed_hash="nomatch",
        )

        result = check_duplicate(crash_info, issues_file)
        assert result.duplicate is True
        assert result.check == "panic_location"
        assert result.check_order == 2

    def test_no_match(self, issues_file):
        """Should return no duplicate when nothing matches."""
        crash_info = CrashInfo(
            panic_location="brand/new/file.rs:999",
            panic_message="brand new message",
            error_variant="BrandNewError",
            stack_frames=["new"],
            stack_trace_hash="nomatch",
            normalized_message="brand new",
            message_hash="nomatch",
            crash_type="crash",
            seed_hash="nomatch",
        )

        result = check_duplicate(crash_info, issues_file)
        assert result.duplicate is False
