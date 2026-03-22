"""Pytest configuration and shared fixtures."""

import json
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def temp_dir():
    """Create a temporary directory."""
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def sample_log_content():
    """Sample fuzzer log content with index out of bounds crash."""
    return """
Running: cargo +nightly fuzz run file_io
INFO: Seed: 1705312847

Output of `std::fmt::Debug`:
Array { dtype: Int32, len: 10 }

thread 'main' panicked at vortex-array/src/compute/slice.rs:142:5:
index out of bounds: the len is 10 but the index is 15
stack backtrace:
   0:     0x7f1234567890 - std::panicking::begin_panic_handler
   1:     0x7f1234567891 - core::panicking::panic_fmt
   2:     0x7f1234567892 - vortex_array::compute::slice::slice_primitive
   3:     0x7f1234567893 - vortex_array::Array::slice

==12345== ERROR: libFuzzer: deadly signal
"""


@pytest.fixture
def sample_issues():
    """Sample existing issues for dedup testing."""
    return [
        {
            "number": 100,
            "title": "Fuzzing Crash: IndexOutOfBounds in file_io",
            "body": (
                "## Fuzzing Crash Report\n\n"
                "**Seed Hash**: `aaa`\n"
                "**Panic Location**: `slice.rs:142`\n"
                "**Error Variant**: `IndexOutOfBounds`\n"
                "\n<!-- seed_hash:aaa stack_hash:bbb message_hash:ccc -->"
            ),
            "url": "https://github.com/example/issues/100",
        },
    ]


@pytest.fixture
def issues_file(sample_issues, temp_dir):
    """Write sample issues to a temporary JSON file."""
    path = temp_dir / "issues.json"
    path.write_text(json.dumps(sample_issues))
    return path
