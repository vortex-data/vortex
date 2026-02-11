"""Tests for extract module."""

import tempfile
from pathlib import Path

import pytest

from fuzz_report.extract import (
    NOISE_FRAME_PATHS,
    NOISE_FUNC_NAMES,
    NOISE_FUNC_PREFIXES,
    CrashInfo,
    _is_noise_frame,
    _is_noise_func,
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

RUST_BACKTRACE_WITH_ERROR_BOILERPLATE = """
thread '<unnamed>' panicked at vortex-error/src/lib.rs:310:33:
called `Result::unwrap()` on an `Err` value: VortexError
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
             at ./vortex-array/src/vtable/compute.rs:120:9

==12345== ERROR: libFuzzer: deadly signal
"""


class TestIsNoiseFrame:
    """Unit tests for the _is_noise_frame helper.

    Note: /rustc/ stdlib frames (rust_begin_unwind, panic_fmt, unwrap_or_else)
    are never passed to _is_noise_frame because the `at ./` regex already
    excludes them — they have `at /rustc/...` paths.

    _is_noise_frame handles the second layer: project-local frames that are
    still error-handling boilerplate, driven by the NOISE_FRAME_PATHS deny list.
    """

    def test_deny_list_is_not_empty(self):
        assert len(NOISE_FRAME_PATHS) > 0

    @pytest.mark.parametrize("path", NOISE_FRAME_PATHS)
    def test_all_deny_list_entries_are_noise(self, path: str):
        assert _is_noise_frame("some_func", f"{path}:1:1")

    def test_closure_in_vortex_error_is_noise(self):
        assert _is_noise_frame(
            "{closure#1}<vortex_scalar::scalar::Scalar, vortex_error::VortexError>",
            "vortex-error/src/lib.rs:457:9",
        )

    def test_bare_closure_is_noise(self):
        assert _is_noise_frame("{closure#0}", "some/other/path.rs:1:1")

    def test_real_frame_is_not_noise(self):
        assert not _is_noise_frame("decimal", "vortex-scalar/src/constructor.rs:61:10")

    def test_real_frame_with_generics_is_not_noise(self):
        assert not _is_noise_frame(
            "invoke<vortex_array::arrays::decimal::vtable::DecimalVTable>",
            "vortex-array/src/vtable/compute.rs:120:9",
        )


class TestIsNoiseFunc:
    """Unit tests for _is_noise_func — filters function names in stack formats
    that lack file paths (libfuzzer ``#N 0x… in func``, dash ``N: 0x… - func``).
    """

    def test_prefix_list_is_not_empty(self):
        assert len(NOISE_FUNC_PREFIXES) > 0

    @pytest.mark.parametrize("prefix", NOISE_FUNC_PREFIXES)
    def test_all_prefixes_are_noise(self, prefix: str):
        assert _is_noise_func(f"{prefix}some_function")

    def test_std_panicking_is_noise(self):
        assert _is_noise_func("std::panicking::begin_panic_handler")

    def test_core_panicking_is_noise(self):
        assert _is_noise_func("core::panicking::panic_fmt")

    def test_dunder_sanitizer_is_noise(self):
        assert _is_noise_func("__sanitizer_print_stack_trace")

    def test_name_list_is_not_empty(self):
        assert len(NOISE_FUNC_NAMES) > 0

    @pytest.mark.parametrize("name", sorted(NOISE_FUNC_NAMES))
    def test_all_exact_names_are_noise(self, name: str):
        assert _is_noise_func(name)

    def test_vortex_expect_is_noise(self):
        assert _is_noise_func("vortex_expect")

    def test_vortex_expect_with_generics_is_noise(self):
        assert _is_noise_func(
            "vortex_expect<vortex_scalar::scalar::Scalar, vortex_error::VortexError>"
        )

    def test_vortex_unwrap_is_noise(self):
        assert _is_noise_func("vortex_unwrap")

    def test_fuzzer_print_stack_trace_is_noise(self):
        assert _is_noise_func("fuzzer::PrintStackTrace")

    def test_fuzzer_prefix_is_noise(self):
        assert _is_noise_func("fuzzer::Fuzzer::ExecuteCallback")

    def test_vortex_func_is_not_noise(self):
        assert not _is_noise_func("vortex_array::compute::slice::slice_primitive")

    def test_fuzz_func_is_not_noise(self):
        assert not _is_noise_func("fuzz::array::run_fuzz_action")

    def test_plain_func_is_not_noise(self):
        assert not _is_noise_func("decimal")


class TestExtractPanicLocation:
    def test_standard_format(self):
        assert extract_panic_location(INDEX_BOUNDS_LOG) == "vortex-array/src/compute/slice.rs:142"

    def test_unknown_when_missing(self):
        assert extract_panic_location("no panic here") == "unknown"

    def test_panicked_at_noise_path_is_skipped(self):
        """vortex_expect panics report vortex-error/src/lib.rs as the
        `panicked at` location. This is the macro site, not the real crash.
        The extractor must skip it and find the real location from the
        stack trace instead.
        """
        # This is the ACTUAL format from CI logs — panicked at points at
        # vortex-error/src/lib.rs, not the real caller.
        log = """\
thread '<unnamed>' panicked at vortex-error/src/lib.rs:310:33:
unable to construct a decimal Scalar
stack backtrace:
   5: vortex_expect
             at ./vortex-error/src/lib.rs:310:14
   6: decimal
             at ./vortex-scalar/src/constructor.rs:61:10
"""
        loc = extract_panic_location(log)
        assert "lib.rs" not in loc
        assert "constructor.rs:61" in loc

    def test_fallback_skips_noise_paths(self):
        """When the `panicked at` line is absent, the fallback regex scans for
        vortex paths in the log. It must skip NOISE_FRAME_PATHS like
        vortex-error/src/lib.rs and return the real crash site instead.
        """
        # Log WITHOUT a `panicked at` line — only a stack trace
        log = """\
stack backtrace:
   5: vortex_expect
             at ./vortex-error/src/lib.rs:310:14
   6: decimal
             at ./vortex-scalar/src/constructor.rs:61:10
"""
        loc = extract_panic_location(log)
        assert "lib.rs" not in loc
        assert "constructor.rs:61" in loc


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

    def test_skips_vortex_error_boilerplate(self):
        """Two layers of noise filtering in the Rust backtrace format:

        Layer 1 (implicit via regex): Frames from /rustc/ stdlib paths like
        rust_begin_unwind, panic_fmt, unwrap_or_else are never matched because
        the regex requires `at ./` (project-local), not `at /rustc/`.

        Layer 2 (explicit via _is_noise_frame): Frames from ./vortex-error/src/lib.rs
        (vortex_expect, closures) ARE project-local but are still error-handling
        boilerplate, so they are explicitly skipped.
        """
        loc = extract_crash_location(RUST_BACKTRACE_WITH_ERROR_BOILERPLATE)
        # Layer 1: /rustc/ stdlib frames never matched
        assert "rust_begin_unwind" not in loc
        assert "panic_fmt" not in loc
        assert "unwrap_or_else" not in loc
        # Layer 2: ./vortex-error/src/lib.rs frames explicitly filtered
        assert "vortex_expect" not in loc
        # Result: the real crash site
        assert "decimal" in loc

    def test_skips_fuzzer_print_stack_trace(self):
        """libfuzzer inserts its own C++ frames like fuzzer::PrintStackTrace
        early in the crash handler stack.  These must be skipped.
        """
        log = """\
thread '<unnamed>' panicked at vortex-error/src/lib.rs:310:33:
unable to construct a decimal Scalar
stack backtrace:
   0: __rustc::rust_begin_unwind
             at /rustc/abc123/library/std/src/panicking.rs:689:5
   1: core::panicking::panic_fmt
             at /rustc/abc123/library/core/src/panicking.rs:80:14
   2: vortex_expect<vortex_scalar::scalar::Scalar, vortex_error::VortexError>
             at ./vortex-error/src/lib.rs:310:14
   3: decimal
             at ./vortex-scalar/src/constructor.rs:61:10

==12345== ERROR: libFuzzer: deadly signal
   #0 0x55e0a0 in fuzzer::PrintStackTrace()
   #1 0x55e0b0 in fuzzer::Fuzzer::CrashCallback()
   #2 0x7f0000 in vortex_scalar::scalar::Scalar::from
"""
        loc = extract_crash_location(log)
        assert "fuzzer::PrintStackTrace" not in loc
        assert "fuzzer::Fuzzer" not in loc
        assert "decimal" in loc
        assert "constructor.rs:61" in loc


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
        # std:: frames should be filtered out
        assert all(not f.startswith("std::") for f in frames)

    def test_in_format_non_vortex_crash(self):
        """Crashes in non-vortex code (e.g. fuzz/) should still be captured."""
        log = """\
stack backtrace:
   #0 0x7f1234567890 in std::panicking::begin_panic_handler
   #1 0x7f1234567891 in fuzz::array::run_fuzz_action
   #2 0x7f1234567892 in __libfuzzer_sys_run

==12345== ERROR: libFuzzer: deadly signal
"""
        frames = extract_stack_frames(log)
        assert "fuzz::array::run_fuzz_action" in frames
        assert all(not f.startswith("std::") for f in frames)
        assert all(not f.startswith("__") for f in frames)

    def test_no_frames(self):
        frames = extract_stack_frames("no stack trace here")
        assert frames == ["unknown"]

    def test_skips_vortex_error_boilerplate(self):
        """Two layers of noise filtering in the Rust backtrace format:

        Layer 1 (implicit via regex): Frames from /rustc/ stdlib paths like
        rust_begin_unwind, panic_fmt, unwrap_or_else are never matched because
        the regex requires `at ./` (project-local), not `at /rustc/`.

        Layer 2 (explicit via _is_noise_frame): Frames from ./vortex-error/src/lib.rs
        (vortex_expect, closures) ARE project-local but are still error-handling
        boilerplate, so they are explicitly skipped.
        """
        frames = extract_stack_frames(RUST_BACKTRACE_WITH_ERROR_BOILERPLATE)
        # Layer 1: /rustc/ stdlib frames never matched (at /rustc/... not at ./)
        assert all("rust_begin_unwind" not in f for f in frames)
        assert all("panic_fmt" not in f for f in frames)
        assert all("panic_display" not in f for f in frames)
        assert all("unwrap_or_else" not in f for f in frames)
        # Layer 2: ./vortex-error/src/lib.rs frames explicitly filtered
        assert all("vortex_expect" not in f for f in frames)
        assert all("{closure" not in f for f in frames)
        # Result: only the real crash frames remain
        assert "decimal" in frames
        assert "sum" in frames


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
