"""Tests for dedup module."""

import json
import tempfile
from pathlib import Path

import pytest

from fuzz_report.dedup import (
    check_duplicate,
    check_error_pattern,
    check_panic_location,
    check_seed_hash,
    check_stack_trace,
)
from fuzz_report.extract import CrashInfo

EXISTING_ISSUES = [
    {
        "number": 100,
        "title": "Fuzzing Crash: IndexOutOfBounds in file_io",
        "body": (
            "## Fuzzing Crash Report\n\n"
            "**Panic Location**: `vortex-array/src/compute/slice.rs:142`\n"
            "**Error Variant**: `IndexOutOfBounds`\n"
            "\n<!-- seed_hash:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa "
            "stack_hash:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb "
            "message_hash:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc -->"
        ),
        "url": "https://github.com/example/repo/issues/100",
    },
    {
        "number": 101,
        "title": "Fuzzing Crash: ScalarMismatch in array_ops",
        "body": (
            "## Fuzzing Crash Report\n\n"
            "**Error Variant**: `ScalarMismatch`\n"
            "\n<!-- seed_hash:dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd -->"
        ),
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


def _make_crash_info(**overrides) -> CrashInfo:
    """Helper to create a CrashInfo with defaults."""
    defaults = {
        "panic_location": "brand/new/file.rs:999",
        "crash_location": "brand/new/file.rs:999",
        "panic_message": "brand new message",
        "error_variant": "BrandNewError",
        "stack_frames": ["new"],
        "stack_trace_raw": "",
        "debug_output": "",
        "seed_hash": "nomatch",
        "stack_trace_hash": "nomatch",
        "normalized_message": "brand new",
        "message_hash": "nomatch",
        "crash_type": "crash",
    }
    defaults.update(overrides)
    return CrashInfo(**defaults)


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
        """Seed hash match should return immediately with check_order=1."""
        crash_info = _make_crash_info(
            seed_hash="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        )
        result = check_duplicate(crash_info, issues_file)
        assert result.duplicate is True
        assert result.check == "seed_hash"
        assert result.check_order == 1

    def test_panic_location_match_second(self, issues_file):
        """Panic location should be checked after seed hash."""
        crash_info = _make_crash_info(
            panic_location="vortex-array/src/compute/slice.rs:142",
        )
        result = check_duplicate(crash_info, issues_file)
        assert result.duplicate is True
        assert result.check == "panic_location"
        assert result.check_order == 2

    def test_stack_trace_match_third(self, issues_file):
        """Stack trace should be checked after panic location."""
        crash_info = _make_crash_info(
            stack_trace_hash="bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        )
        result = check_duplicate(crash_info, issues_file)
        assert result.duplicate is True
        assert result.check == "stack_trace"
        assert result.check_order == 3

    def test_error_pattern_match_fourth(self, issues_file):
        """Error pattern should be checked last."""
        crash_info = _make_crash_info(
            message_hash="cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
        )
        result = check_duplicate(crash_info, issues_file)
        assert result.duplicate is True
        assert result.check == "error_pattern"
        assert result.check_order == 4

    def test_no_match(self, issues_file):
        """Should return no duplicate when nothing matches."""
        crash_info = _make_crash_info()
        result = check_duplicate(crash_info, issues_file)
        assert result.duplicate is False

    def test_empty_issues(self, temp_dir):
        """Empty issues file should return no duplicate."""
        empty_file = temp_dir / "empty.json"
        empty_file.write_text("[]")
        crash_info = _make_crash_info()
        result = check_duplicate(crash_info, str(empty_file))
        assert result.duplicate is False

    def test_missing_issues_file(self, temp_dir):
        """Missing issues file should return no duplicate."""
        crash_info = _make_crash_info()
        result = check_duplicate(crash_info, str(temp_dir / "nonexistent.json"))
        assert result.duplicate is False
