"""Tests for extract module."""

import tempfile
from pathlib import Path

import pytest

from fuzz_report.extract import (
    CrashInfo,
    extract_crash_info,
    extract_crash_location,
    extract_debug_output,
    extract_error_variant,
    extract_panic_location,
    extract_panic_message,
    extract_stack_frames,
    extract_stack_trace_raw,
    get_crash_type,
    normalize_message,
)


INDEX_BOUNDS_LOG = """
Running: cargo +nightly fuzz run file_io
INFO: Seed: 1705312847

thread 'main' panicked at vortex-array/src/compute/slice.rs:142:5:
index out of bounds: the len is 10 but the index is 15
stack backtrace:
   0:     0x7f1234567890 - std::panicking::begin_panic_handler
   1:     0x7f1234567891 - core::panicking::panic_fmt
   2:     0x7f1234567892 - vortex_array::compute::slice::slice_primitive
   3:     0x7f1234567893 - vortex_array::Array::slice

==12345== ERROR: libFuzzer: deadly signal
"""

SCALAR_MISMATCH_LOG = """
Running: cargo +nightly fuzz run array_ops

thread 'main' panicked at fuzz/src/array/compare.rs:89:5:
Scalar mismatch: expected Int(42), got Int(0) in step 2

ScalarMismatch {
    expected: Scalar::Int(42),
    actual: Scalar::Int(0),
}

stack backtrace:
   0:     0x7f9876543210 - std::panicking::begin_panic_handler
   1:     0x7f9876543211 - vortex_fuzz::error::VortexFuzzError::scalar_mismatch

==67890== ERROR: libFuzzer: deadly signal
"""

TIMEOUT_LOG = """
ALARM: working on the last Unit for 120 seconds

==22222== ERROR: libFuzzer: timeout after 120 seconds
"""

DEBUG_OUTPUT_LOG = """
Output of `std::fmt::Debug`:
Array { dtype: Int32, len: 10 }

thread 'main' panicked at vortex-array/src/compute/slice.rs:142:5:
index out of bounds: the len is 10 but the index is 15

==12345== ERROR: libFuzzer: deadly signal
"""

LIBFUZZER_FRAME_LOG = """
thread 'main' panicked at vortex-array/src/compute/slice.rs:42:5:
test panic
stack backtrace:
   #0 0x7f1234567890 in std::panicking::begin_panic_handler
   #1 0x7f1234567891 in vortex_array::compute::slice::slice_primitive

==12345== ERROR: libFuzzer: deadly signal
"""


class TestExtractPanicLocation:
    def test_standard_format(self):
        assert (
            extract_panic_location(INDEX_BOUNDS_LOG)
            == "vortex-array/src/compute/slice.rs:142"
        )

    def test_unknown_when_missing(self):
        assert extract_panic_location("no panic here") == "unknown"


class TestExtractCrashLocation:
    def test_with_vortex_frame(self):
        loc = extract_crash_location(LIBFUZZER_FRAME_LOG)
        assert "vortex" in loc

    def test_fallback_to_panic_location(self):
        # Log with panic but no stack frames in "#N 0x... in ..." format
        log = """thread 'main' panicked at vortex-array/src/compute/slice.rs:142:5:
index out of bounds
"""
        loc = extract_crash_location(log)
        assert "slice.rs:142" in loc


class TestExtractPanicMessage:
    def test_index_bounds(self):
        msg = extract_panic_message(INDEX_BOUNDS_LOG)
        assert "index out of bounds" in msg

    def test_scalar_mismatch(self):
        msg = extract_panic_message(SCALAR_MISMATCH_LOG)
        assert "mismatch" in msg.lower()

    def test_error_format(self):
        log = "==123== ERROR: libFuzzer: timeout after 120 seconds"
        msg = extract_panic_message(log)
        assert "timeout" in msg.lower()


class TestExtractErrorVariant:
    def test_index_out_of_bounds(self):
        assert extract_error_variant(INDEX_BOUNDS_LOG) == "IndexOutOfBounds"

    def test_scalar_mismatch(self):
        assert extract_error_variant(SCALAR_MISMATCH_LOG) == "ScalarMismatch"

    def test_timeout(self):
        assert extract_error_variant(TIMEOUT_LOG) == "Timeout"

    def test_unknown(self):
        assert extract_error_variant("some random log") == "unknown"


class TestExtractStackFrames:
    def test_dash_format(self):
        frames = extract_stack_frames(INDEX_BOUNDS_LOG)
        assert len(frames) > 0
        assert any("vortex" in f for f in frames)

    def test_in_format(self):
        frames = extract_stack_frames(LIBFUZZER_FRAME_LOG)
        assert len(frames) > 0
        assert any("vortex" in f for f in frames)

    def test_no_frames(self):
        frames = extract_stack_frames("no stack trace here")
        assert frames == ["unknown"]


class TestExtractStackTraceRaw:
    def test_extracts_backtrace(self):
        raw = extract_stack_trace_raw(INDEX_BOUNDS_LOG)
        assert "stack backtrace:" in raw
        assert "vortex_array" in raw

    def test_empty_when_missing(self):
        raw = extract_stack_trace_raw("no stack trace")
        assert raw == ""


class TestExtractDebugOutput:
    def test_extracts_debug(self):
        output = extract_debug_output(DEBUG_OUTPUT_LOG)
        assert "Array" in output
        assert "Int32" in output

    def test_empty_when_missing(self):
        output = extract_debug_output("no debug output")
        assert output == ""


class TestGetCrashType:
    @pytest.mark.parametrize(
        "filename,expected",
        [
            ("crash-abc123", "crash"),
            ("leak-def456", "leak"),
            ("timeout-ghi789", "timeout"),
            ("oom-jkl012", "oom"),
            ("unknown", "unknown"),
            ("", "unknown"),
        ],
    )
    def test_crash_types(self, filename: str, expected: str):
        assert get_crash_type(filename) == expected


class TestNormalizeMessage:
    def test_replaces_numbers(self):
        assert normalize_message("index 15 of len 10") == "index N of len N"

    def test_preserves_text(self):
        assert normalize_message("no numbers here") == "no numbers here"


class TestExtractCrashInfo:
    def test_full_extraction(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "fuzz_output.log"
            crash_path = Path(tmpdir) / "crash-abc123"

            log_path.write_text(INDEX_BOUNDS_LOG)
            crash_path.write_bytes(b"test seed data")

            info = extract_crash_info(str(log_path), str(crash_path))

            assert info.error_variant == "IndexOutOfBounds"
            assert info.crash_type == "crash"
            assert "index out of bounds" in info.panic_message
            assert info.seed_hash != "unknown"
            assert len(info.stack_trace_hash) == 64  # SHA256
            assert info.stack_trace_raw != ""
            assert info.crash_location != "unknown"

    def test_without_crash_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            f.write(INDEX_BOUNDS_LOG)
            f.flush()
            log_path = f.name

        try:
            info = extract_crash_info(log_path)
            assert info.seed_hash == "unknown"
        finally:
            Path(log_path).unlink()

    def test_serialization_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "fuzz_output.log"
            log_path.write_text(INDEX_BOUNDS_LOG)

            info = extract_crash_info(str(log_path))
            json_str = info.to_json()
            data = __import__("json").loads(json_str)

            # Should be able to reconstruct
            info2 = CrashInfo(**data)
            assert info2.panic_message == info.panic_message
            assert info2.error_variant == info.error_variant
