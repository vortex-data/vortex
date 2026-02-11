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
from fuzz_report.extract import CrashInfo, extract_crash_info

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


# ---------------------------------------------------------------------------
# End-to-end tests using real crash logs from production fuzzer runs.
#
# These logs are taken from https://github.com/vortex-data/vortex/issues/6048
# where two completely different bugs were incorrectly matched because the
# extractor pointed at vortex-error boilerplate instead of the real crash site.
# ---------------------------------------------------------------------------

# Crash 1: mask/struct cast error — panics in run_fuzz_action because a
# vortex_expect call in the mask operation fails.
# NOTE: The `panicked at` line points at vortex-error/src/lib.rs (the
# vortex_expect macro site), NOT the real caller. This is the actual
# format from CI logs.
MASK_STRUCT_CAST_LOG = """\
thread '<unnamed>' panicked at vortex-error/src/lib.rs:310:33:
mask operation should succeed in fuzz test:
  Cannot add non-nullable field during struct cast
stack backtrace:
   0: __rustc::rust_begin_unwind
             at /rustc/18d13b5332916ffca8eadb9106d54b5b434e9978/library/std/src/panicking.rs:689:5
   1: core::panicking::panic_fmt
             at /rustc/18d13b5332916ffca8eadb9106d54b5b434e9978/library/core/src/panicking.rs:80:14
   2: panic_display<vortex_error::VortexError>
             at /rustc/18d13b5332916ffca8eadb9106d54b5b434e9978/library/core/src/panicking.rs:259:5
   3: {closure#1}<alloc::sync::Arc<dyn vortex_array::array::Array>, vortex_error::VortexError>
             at ./vortex-error/src/lib.rs:457:9
   4: unwrap_or_else<alloc::sync::Arc<dyn vortex_array::array::Array>, vortex_error::VortexError>
             at /rustc/18d13b5332916ffca8eadb9106d54b5b434e9978/library/core/src/result.rs:1622:23
   5: vortex_expect<alloc::sync::Arc<dyn vortex_array::array::Array>, vortex_error::VortexError>
             at ./vortex-error/src/lib.rs:310:14
   6: run_fuzz_action
             at ./fuzz/src/array/mod.rs:645:22
   7: __libfuzzer_sys_run
             at ./fuzz/fuzz_targets/array_ops.rs:14:11

==12345== ERROR: libFuzzer: deadly signal
"""

# Crash 2: decimal sum overflow — panics constructing a decimal Scalar
# because the computed sum doesn't fit the declared precision.
# NOTE: Same as crash 1, the `panicked at` line points at vortex-error,
# not the real caller. This is the actual format from CI logs.
DECIMAL_SUM_LOG = """\
thread '<unnamed>' panicked at vortex-error/src/lib.rs:310:33:
unable to construct a decimal Scalar:
  Incompatible dtype decimal(76,75) with value decimal256(51612137)
stack backtrace:
   0: __rustc::rust_begin_unwind
             at /rustc/9e79395f92bff6a8f536430e42a4beae69f60ff8/library/std/src/panicking.rs:689:5
   1: core::panicking::panic_fmt
             at /rustc/9e79395f92bff6a8f536430e42a4beae69f60ff8/library/core/src/panicking.rs:80:14
   2: panic_display<vortex_error::VortexError>
             at /rustc/9e79395f92bff6a8f536430e42a4beae69f60ff8/library/core/src/panicking.rs:259:5
   3: {closure#1}<vortex_scalar::scalar::Scalar, vortex_error::VortexError>
             at ./vortex-error/src/lib.rs:457:9
   4: unwrap_or_else<vortex_scalar::scalar::Scalar, vortex_error::VortexError>
             at /rustc/9e79395f92bff6a8f536430e42a4beae69f60ff8/library/core/src/result.rs:1622:23
   5: vortex_expect<vortex_scalar::scalar::Scalar, vortex_error::VortexError>
             at ./vortex-error/src/lib.rs:310:14
   6: decimal
             at ./vortex-scalar/src/constructor.rs:61:10
   7: sum
             at ./vortex-array/src/arrays/decimal/compute/sum.rs:57:32
   8: invoke<vortex_array::arrays::decimal::vtable::DecimalVTable>
             at ./vortex-array/src/compute/sum.rs:226:17

==12345== ERROR: libFuzzer: deadly signal
"""


def _build_issue_body(crash_info: CrashInfo) -> str:
    """Build a realistic issue body from extracted crash info.

    Mirrors the new_issue.md template: crash location, panic message,
    raw stack trace, and the hidden hash comment.
    """
    return (
        f"## Fuzzing Crash Report\n\n"
        f"**Crash Location**: `{crash_info.crash_location}`\n\n"
        f"**Error Message**:\n```\n{crash_info.panic_message}\n```\n\n"
        f"**Stack Trace**:\n```\n{crash_info.stack_trace_raw}\n```\n\n"
        f"- **Target**: `array_ops`\n"
        f"- **Error Variant**: `{crash_info.error_variant}`\n\n"
        f"<!-- seed_hash:{crash_info.seed_hash} "
        f"stack_hash:{crash_info.stack_trace_hash} "
        f"message_hash:{crash_info.message_hash} -->"
    )


class TestEndToEndDedup:
    """End-to-end tests: extract from real logs, build issue bodies, run dedup.

    Reproduces the false-match scenario from
    https://github.com/vortex-data/vortex/issues/6048 where two unrelated
    crashes (mask/struct-cast vs decimal/sum) were matched because the
    extractor pointed at vortex-error boilerplate.
    """

    @pytest.fixture
    def crash1_info(self, temp_dir):
        log_path = temp_dir / "mask_crash.log"
        log_path.write_text(MASK_STRUCT_CAST_LOG)
        return extract_crash_info(str(log_path))

    @pytest.fixture
    def crash2_info(self, temp_dir):
        log_path = temp_dir / "decimal_crash.log"
        log_path.write_text(DECIMAL_SUM_LOG)
        return extract_crash_info(str(log_path))

    def test_extraction_skips_boilerplate(self, crash1_info, crash2_info):
        """Both crashes should extract real locations, not vortex-error."""
        # Crash 1: mask/struct cast
        assert "vortex-error" not in crash1_info.panic_location
        assert "vortex-error" not in crash1_info.crash_location
        assert "mod.rs:645" in crash1_info.panic_location
        assert "run_fuzz_action" in crash1_info.crash_location

        # Crash 2: decimal/sum
        assert "vortex-error" not in crash2_info.panic_location
        assert "vortex-error" not in crash2_info.crash_location
        assert "constructor.rs:61" in crash2_info.panic_location
        assert "decimal" in crash2_info.crash_location

    def test_stack_frames_are_different(self, crash1_info, crash2_info):
        """The two crashes should produce entirely different stack frames."""
        assert crash1_info.stack_frames != crash2_info.stack_frames
        assert crash1_info.stack_trace_hash != crash2_info.stack_trace_hash

    def test_panic_locations_are_different(self, crash1_info, crash2_info):
        """The two crashes should have different panic locations."""
        assert crash1_info.panic_location != crash2_info.panic_location

    def test_no_high_confidence_match(self, crash1_info, crash2_info, temp_dir):
        """Crash 2 must NOT match an issue created from crash 1 at
        high or exact confidence. The old bug would match on
        'lib.rs:310' (panic_location check, high confidence).
        """
        issue_body = _build_issue_body(crash1_info)
        issues_path = temp_dir / "issues.json"
        issues_path.write_text(
            json.dumps(
                [
                    {
                        "number": 6048,
                        "title": "Fuzzing Crash: VortexError in array_ops",
                        "body": issue_body,
                        "url": "https://github.com/vortex-data/vortex/issues/6048",
                    },
                ]
            )
        )

        result = check_duplicate(crash2_info, str(issues_path))

        # Must not match on panic_location or stack_trace (the old false match)
        if result.duplicate:
            assert result.check != "panic_location", (
                f"False panic_location match! debug={result.debug}"
            )
            assert result.check != "stack_trace", f"False stack_trace match! debug={result.debug}"
            assert result.confidence != "exact", f"False exact match! debug={result.debug}"

    def test_same_crash_does_match(self, crash1_info, temp_dir):
        """A second occurrence of the SAME crash should still be detected."""
        issue_body = _build_issue_body(crash1_info)
        issues_path = temp_dir / "issues.json"
        issues_path.write_text(
            json.dumps(
                [
                    {
                        "number": 6048,
                        "title": "Fuzzing Crash: VortexError in array_ops",
                        "body": issue_body,
                        "url": "https://github.com/vortex-data/vortex/issues/6048",
                    },
                ]
            )
        )

        result = check_duplicate(crash1_info, str(issues_path))
        assert result.duplicate is True
        # Should match on panic_location or stack_trace (high confidence)
        assert result.check in ("panic_location", "stack_trace")
        assert result.confidence == "high"

    def test_debug_info_is_present(self, crash1_info, crash2_info, temp_dir):
        """Dedup results should include debug details for diagnosis."""
        issue_body = _build_issue_body(crash1_info)
        issues_path = temp_dir / "issues.json"
        issues_path.write_text(
            json.dumps(
                [
                    {
                        "number": 6048,
                        "title": "Fuzzing Crash: VortexError in array_ops",
                        "body": issue_body,
                        "url": "https://github.com/vortex-data/vortex/issues/6048",
                    },
                ]
            )
        )

        result = check_duplicate(crash2_info, str(issues_path))
        assert result.debug is not None
        assert "extraction" in result.debug
        assert "panic_location" in result.debug["extraction"]
        assert "crash_location" in result.debug["extraction"]
        assert "stack_frames_top5" in result.debug["extraction"]


# ── Real-world test from GitHub issue #6429 ──────────────────────────────
# Two different fuzz targets (file_io and array_ops) hit the same
# decimal Scalar construction bug via the same call path.  The dedup
# should match them correctly on the real crash site, not on the
# vortex-error/src/lib.rs boilerplate.

ISSUE_6429_FILE_IO_LOG = """\
thread '<unnamed>' panicked at vortex-error/src/lib.rs:310:33:
unable to construct a decimal Scalar:
  Incompatible dtype decimal(76,-74)? with value decimal256(-1699999)
stack backtrace:
   0: __rustc::rust_begin_unwind
             at /rustc/9e79395f92bff6a8f536430e42a4beae69f60ff8/library/std/src/panicking.rs:689:5
   1: core::panicking::panic_fmt
             at /rustc/9e79395f92bff6a8f536430e42a4beae69f60ff8/library/core/src/panicking.rs:80:14
   2: panic_display<vortex_error::VortexError>
             at /rustc/9e79395f92bff6a8f536430e42a4beae69f60ff8/library/core/src/panicking.rs:259:5
   3: {closure#1}<vortex_scalar::scalar::Scalar, vortex_error::VortexError>
             at ./vortex-error/src/lib.rs:457:9
   4: unwrap_or_else<vortex_scalar::scalar::Scalar, vortex_error::VortexError>
             at /rustc/9e79395f92bff6a8f536430e42a4beae69f60ff8/library/core/src/result.rs:1622:23
   5: vortex_expect<vortex_scalar::scalar::Scalar, vortex_error::VortexError>
             at ./vortex-error/src/lib.rs:310:14
   6: decimal
             at ./vortex-scalar/src/constructor.rs:61:10
   7: sum
             at ./vortex-array/src/arrays/decimal/compute/sum.rs:57:32
   8: invoke<vortex_array::arrays::decimal::vtable::DecimalVTable>
             at ./vortex-array/src/compute/sum.rs:226:17
   9: sum_impl
             at ./vortex-array/src/compute/sum.rs:250:38
  10: invoke
             at ./vortex-array/src/compute/sum.rs:146:26
  11: invoke
             at ./vortex-array/src/compute/mod.rs:144:34
  12: sum_with_accumulator
             at ./vortex-array/src/compute/sum.rs:53:10
  13: sum
             at ./vortex-array/src/compute/sum.rs:70:5
  14: compute_stat
             at ./vortex-array/src/stats/array.rs:157:22
  15: push_chunk
             at ./vortex-layout/src/layouts/zoned/zone_map.rs:211:49
  16: write
             at ./vortex-file/src/writer.rs:385:22
  17: __libfuzzer_sys_run
             at ./fuzz/fuzz_targets/file_io.rs:73:10

==12345== ERROR: libFuzzer: deadly signal
"""

ISSUE_6429_ARRAY_OPS_LOG = """\
thread '<unnamed>' panicked at vortex-error/src/lib.rs:310:33:
unable to construct a decimal Scalar:
  Incompatible dtype decimal(76,75)? with value decimal256(51612137)
stack backtrace:
   0: __rustc::rust_begin_unwind
             at /rustc/9e79395f92bff6a8f536430e42a4beae69f60ff8/library/std/src/panicking.rs:689:5
   1: core::panicking::panic_fmt
             at /rustc/9e79395f92bff6a8f536430e42a4beae69f60ff8/library/core/src/panicking.rs:80:14
   2: panic_display<vortex_error::VortexError>
             at /rustc/9e79395f92bff6a8f536430e42a4beae69f60ff8/library/core/src/panicking.rs:259:5
   3: {closure#1}<vortex_scalar::scalar::Scalar, vortex_error::VortexError>
             at ./vortex-error/src/lib.rs:457:9
   4: unwrap_or_else<vortex_scalar::scalar::Scalar, vortex_error::VortexError>
             at /rustc/9e79395f92bff6a8f536430e42a4beae69f60ff8/library/core/src/result.rs:1622:23
   5: vortex_expect<vortex_scalar::scalar::Scalar, vortex_error::VortexError>
             at ./vortex-error/src/lib.rs:310:14
   6: decimal
             at ./vortex-scalar/src/constructor.rs:61:10
   7: sum
             at ./vortex-array/src/arrays/decimal/compute/sum.rs:57:32
   8: invoke<vortex_array::arrays::decimal::vtable::DecimalVTable>
             at ./vortex-array/src/compute/sum.rs:226:17
   9: sum_impl
             at ./vortex-array/src/compute/sum.rs:250:38
  10: invoke
             at ./vortex-array/src/compute/sum.rs:146:26
  11: invoke
             at ./vortex-array/src/compute/mod.rs:144:34
  12: sum_with_accumulator
             at ./vortex-array/src/compute/sum.rs:53:10
  13: sum
             at ./vortex-array/src/compute/sum.rs:70:5
  14: sum_canonical_array
             at ./fuzz/src/array/sum.rs:12:5
  15: arbitrary
             at ./fuzz/src/array/mod.rs:313:38

==12345== ERROR: libFuzzer: deadly signal
"""


class TestIssue6429:
    """End-to-end test for GitHub issue #6429.

    Two targets (file_io, array_ops) crash in the same decimal Scalar
    constructor bug. The dedup should:
      - Extract the real crash site (constructor.rs:61), not vortex-error
      - Match them as duplicates on panic_location or stack_trace
      - Report the match reason referencing constructor.rs, not lib.rs
    """

    @pytest.fixture
    def file_io_info(self, temp_dir):
        p = temp_dir / "file_io.log"
        p.write_text(ISSUE_6429_FILE_IO_LOG)
        return extract_crash_info(str(p))

    @pytest.fixture
    def array_ops_info(self, temp_dir):
        p = temp_dir / "array_ops.log"
        p.write_text(ISSUE_6429_ARRAY_OPS_LOG)
        return extract_crash_info(str(p))

    def test_extraction_skips_vortex_error(self, file_io_info, array_ops_info):
        """Neither crash should reference vortex-error in extracted fields."""
        for info in (file_io_info, array_ops_info):
            assert "vortex-error" not in info.panic_location
            assert "vortex-error" not in info.crash_location
            assert "vortex_expect" not in info.crash_location
            assert "constructor.rs:61" in info.panic_location
            assert "decimal" in info.crash_location

    def test_no_noise_in_stack_frames(self, file_io_info, array_ops_info):
        """Stack frames should not contain any noise."""
        for info in (file_io_info, array_ops_info):
            assert all("vortex_expect" not in f for f in info.stack_frames)
            assert all("{closure" not in f for f in info.stack_frames)
            assert "decimal" in info.stack_frames

    def test_same_bug_matches_correctly(self, file_io_info, array_ops_info, temp_dir):
        """array_ops crash should match the file_io issue — same bug."""
        issue_body = _build_issue_body(file_io_info)
        issues_path = temp_dir / "issues.json"
        issues_path.write_text(
            json.dumps(
                [
                    {
                        "number": 6429,
                        "title": "Fuzzing Crash: VortexError in file_io",
                        "body": issue_body,
                        "url": "https://github.com/vortex-data/vortex/issues/6429",
                    }
                ]
            )
        )

        result = check_duplicate(array_ops_info, str(issues_path))
        assert result.duplicate is True
        assert result.confidence == "high"

    def test_match_reason_references_real_site(self, file_io_info, array_ops_info, temp_dir):
        """The match reason must reference the real crash, not boilerplate."""
        issue_body = _build_issue_body(file_io_info)
        issues_path = temp_dir / "issues.json"
        issues_path.write_text(
            json.dumps(
                [
                    {
                        "number": 6429,
                        "title": "Fuzzing Crash: VortexError in file_io",
                        "body": issue_body,
                        "url": "https://github.com/vortex-data/vortex/issues/6429",
                    }
                ]
            )
        )

        result = check_duplicate(array_ops_info, str(issues_path))
        assert "lib.rs:310" not in result.reason
        if result.check == "panic_location":
            assert "constructor.rs:61" in result.reason
