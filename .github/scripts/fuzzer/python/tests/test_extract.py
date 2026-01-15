"""Tests for extract module."""

import tempfile
from pathlib import Path

import pytest

from ..extract import (
    CrashInfo,
    extract_crash_info,
    extract_error_variant,
    extract_panic_location,
    extract_panic_message,
    extract_stack_frames,
    get_crash_type,
    normalize_message,
)


# Test fixtures
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


class TestExtractPanicLocation:
    def test_standard_format(self):
        assert (
            extract_panic_location(INDEX_BOUNDS_LOG)
            == "vortex-array/src/compute/slice.rs:142"
        )

    def test_unknown_when_missing(self):
        assert extract_panic_location("no panic here") == "unknown"


class TestExtractPanicMessage:
    def test_index_bounds(self):
        msg = extract_panic_message(INDEX_BOUNDS_LOG)
        assert "index out of bounds" in msg

    def test_scalar_mismatch(self):
        msg = extract_panic_message(SCALAR_MISMATCH_LOG)
        assert "mismatch" in msg.lower()


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
    def test_extracts_frames(self):
        frames = extract_stack_frames(INDEX_BOUNDS_LOG)
        assert len(frames) > 0
        assert any("vortex" in f for f in frames)


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
