"""Pytest configuration and shared fixtures."""

import json
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def temp_dir():
    """Create a temporary directory."""
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def sample_log_content():
    """Sample fuzzer log content."""
    return """
Running: cargo +nightly fuzz run file_io
INFO: Seed: 1705312847

thread 'main' panicked at vortex-array/src/compute/slice.rs:142:5:
index out of bounds: the len is 10 but the index is 15
stack backtrace:
   0:     0x7f1234567890 - std::panicking::begin_panic_handler
   1:     0x7f1234567891 - vortex_array::compute::slice::slice_primitive

==12345== ERROR: libFuzzer: deadly signal
"""


@pytest.fixture
def sample_issues():
    """Sample existing issues."""
    return [
        {
            "number": 100,
            "title": "Fuzzing Crash: IndexOutOfBounds",
            "body": "**Seed Hash**: `aaa`\n**Panic Location**: `slice.rs:142`",
            "url": "https://github.com/example/issues/100",
        },
    ]
