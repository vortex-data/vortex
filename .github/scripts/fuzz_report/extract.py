"""Extract crash information from fuzzer output logs."""

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class CrashInfo:
    """Extracted crash information."""

    panic_location: str
    crash_location: str
    panic_message: str
    error_variant: str
    stack_frames: list[str]
    stack_trace_raw: str
    debug_output: str
    seed_hash: str
    stack_trace_hash: str
    normalized_message: str
    message_hash: str
    crash_type: str

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


def _is_noise_path(path: str) -> bool:
    """Return True if a file path is error-handling boilerplate.

    The `panicked at` line can point at vortex-error/src/lib.rs when
    vortex_expect/vortex_unwrap panics — that's the macro location, not
    the real crash site. This helper filters those out everywhere.
    """
    return any(prefix in path for prefix in NOISE_FRAME_PATHS)


def extract_panic_location(log_content: str) -> str:
    """Extract panic location (file:line) from log.

    Skips noise paths (NOISE_FRAME_PATHS) even when they appear in the
    `panicked at` line itself — e.g. vortex_expect panics report
    vortex-error/src/lib.rs as the location, not the actual caller.
    """
    # Look for "panicked at file:line:" pattern (newer Rust format)
    match = re.search(r"panicked at ([^:]+\.rs:\d+)", log_content)
    if match and not _is_noise_path(match.group(1)):
        return match.group(1)

    # Look for "panicked at 'msg', file:line" pattern (older Rust format)
    match = re.search(r"panicked at [^,]+, ([^:]+:\d+)", log_content)
    if match and not _is_noise_path(match.group(1)):
        return match.group(1)

    # Fallback: scan "at ./path:line" from stack trace, skipping noise.
    # The `at ./` prefix scopes to project-local paths; _is_noise_path()
    # further excludes boilerplate like vortex-error/src/lib.rs.
    for match in re.finditer(r"at \./([^:\s]+:\d+)", log_content):
        loc = match.group(1)
        if _is_noise_path(loc):
            continue
        return loc

    return "unknown"


def extract_crash_location(log_content: str) -> str:
    """Extract crash location as file:function_name from stack frames.

    Prefers the Rust backtrace format (``N: func at ./path``) because it has
    file paths that enable reliable noise filtering.  Falls back to the
    libfuzzer and dash formats which only have function names.
    """
    func_name = None

    # Best: "N: function_name\n  at ./path" format (Rust backtrace)
    # The `at ./` regex excludes /rustc/ stdlib frames; _is_noise_frame()
    # further excludes vortex-error boilerplate and closure wrappers.
    for m in re.finditer(r"\s+\d+:\s+(\S+)\n\s+at\s+\./([^\n]+)", log_content):
        name = m.group(1)
        path = m.group(2)
        if _is_noise_frame(name, path):
            continue
        func_name = re.sub(r"<.*", "", name)
        break

    # Fallback: "#N 0x... in func" format (libfuzzer), skip noise prefixes
    if not func_name:
        for m in re.finditer(r"#\d+\s+0x[a-f0-9]+\s+in\s+([^\s<(]+)", log_content):
            if not _is_noise_func(m.group(1)):
                func_name = m.group(1)
                break

    # Fallback: "N: 0x... - func" format (dash), skip noise prefixes
    if not func_name:
        for m in re.finditer(r"\d+:\s+0x[a-f0-9]+\s+-\s+([^\s<(]+)", log_content):
            if not _is_noise_func(m.group(1)):
                func_name = m.group(1)
                break

    if func_name:
        panic_loc = extract_panic_location(log_content)
        if panic_loc != "unknown":
            return f"{panic_loc}:{func_name}"
        return func_name

    panic_loc = extract_panic_location(log_content)
    if panic_loc != "unknown":
        return panic_loc

    return "unknown"


def extract_panic_message(log_content: str) -> str:
    """Extract panic/error message from log."""
    # Look for Rust panic format: "panicked at path/file.rs:line:col:\nmessage"
    # Terminators: blank line, "stack backtrace:", "Backtrace:", or numbered frame
    match = re.search(
        r"panicked at [^\n]+\.rs:\d+(?::\d+)?:\s*\n"
        r"(.+?)"
        r"(?:\n\n|\nstack backtrace:|\nBacktrace:|\n\s*\d+:\s+\S+|\n\w+\s*\{)",
        log_content,
        re.DOTALL,
    )
    if match:
        return match.group(1).strip()

    # Look for "panicked at 'message'" format (older Rust)
    match = re.search(r"panicked at '([^']+)'", log_content)
    if match:
        return match.group(1)

    # Look for ERROR: message
    match = re.search(r"ERROR: (.+)", log_content)
    if match:
        return match.group(1)

    # Look for assertion message
    match = re.search(r"assertion `?failed`?: (.+)", log_content)
    if match:
        return match.group(1)

    return "unknown"


def extract_error_variant(log_content: str) -> str:
    """Extract VortexFuzzError variant or panic type."""
    # Look for VortexFuzzError enum variants
    variants = [
        "ScalarMismatch",
        "SearchSortedError",
        "MinMaxMismatch",
        "ArrayNotEqual",
        "DTypeMismatch",
        "LengthMismatch",
        "VortexError",
    ]
    for variant in variants:
        if variant in log_content:
            return variant

    # Detect common panic types from message
    if "index out of bounds" in log_content:
        return "IndexOutOfBounds"
    if re.search(r"assertion.*failed", log_content):
        return "AssertionFailed"
    if "unwrap" in log_content and "None" in log_content:
        return "UnwrapNone"
    if "overflow" in log_content:
        return "Overflow"
    if "out of memory" in log_content.lower() or "OOM" in log_content:
        return "OutOfMemory"
    if "timeout" in log_content.lower():
        return "Timeout"
    if "SEGV" in log_content or "segfault" in log_content.lower():
        return "Segfault"

    return "unknown"


# Paths that are error-handling / panic infrastructure, not real crash sites.
# Frames from /rustc/ stdlib are already excluded by the `at ./` regex (they
# have `at /rustc/...` paths). This list covers project-local paths that still
# match `at ./` but are boilerplate. Add new entries here as needed.
NOISE_FRAME_PATHS = [
    "vortex-error/src/lib.rs",
]

# Function-name prefixes that are never the real crash site.
# Used for stack formats that lack file paths (libfuzzer, dash format).
NOISE_FUNC_PREFIXES = (
    "std::",
    "core::",
    "alloc::",
    "fuzzer::",  # libfuzzer C++ internals (e.g. fuzzer::PrintStackTrace)
    "__",  # sanitizer, fuzzer, and C runtime internals
)

# Exact function names (after stripping generics) that are error-handling
# boilerplate.  These supplement NOISE_FUNC_PREFIXES for cases where the
# function doesn't match a prefix but is still infrastructure.
NOISE_FUNC_NAMES = frozenset(
    {
        "vortex_expect",
        "vortex_unwrap",
        "panic_display",
        "rust_begin_unwind",
    }
)


def _is_noise_frame(func_name: str, path: str) -> bool:
    """Return True if this stack frame is panic/error-handling boilerplate.

    Two layers of noise are filtered:

    1. Frames from /rustc/ stdlib (rust_begin_unwind, panic_fmt, etc.) are
       already excluded by the `at ./` regex — they have `at /rustc/...` paths,
       so the regex never matches them.

    2. Frames whose path matches NOISE_FRAME_PATHS (via _is_noise_path).
       These are project-local but are still infrastructure (e.g. vortex_expect,
       vortex_unwrap in vortex-error/src/lib.rs).

    3. Closure wrappers like {closure#0} that appear in generic unwrap/expect
       call chains.
    """
    clean = re.sub(r"<.*", "", func_name)
    if clean.startswith("{"):
        return True
    if _is_noise_path(path):
        return True
    return False


def _is_noise_func(func_name: str) -> bool:
    """Return True if a function name is obviously infrastructure.

    Used for stack trace formats that lack file paths (libfuzzer ``#N 0x…
    in func``, dash ``N: 0x… - func``).  Checks both prefix-based rules
    (NOISE_FUNC_PREFIXES) and exact-name rules (NOISE_FUNC_NAMES).
    """
    if func_name.startswith(NOISE_FUNC_PREFIXES):
        return True
    # Strip generics for exact match (regex already strips them, but be safe)
    clean = re.sub(r"<.*", "", func_name)
    return clean in NOISE_FUNC_NAMES


def extract_stack_frames(log_content: str) -> list[str]:
    """Extract stack trace frames (function names only).

    Prioritizes the Rust-style backtrace (N: func_name at ./path) over
    the libfuzzer crash handler frames (#N 0x... in func).
    """
    frames = []

    # Best: "N: function_name\n  at ./path" (Rust backtrace, most informative)
    #
    # The `at ./` pattern provides the first layer of filtering: it only matches
    # project-local paths, so /rustc/ stdlib frames (rust_begin_unwind, panic_fmt,
    # unwrap_or_else, etc.) are never captured.
    #
    # _is_noise_frame() provides the second layer: it filters out project-local
    # frames that are still boilerplate (vortex-error/src/lib.rs, closures).
    for match in re.finditer(r"\s+\d+:\s+(\S+)\n\s+at\s+\./([^\n]+)", log_content):
        func = match.group(1)
        path = match.group(2)
        if _is_noise_frame(func, path):
            continue
        # Strip generic parameters like <...>
        func = re.sub(r"<.*", "", func)
        frames.append(func)

    # Fallback: "#N 0x... in function_name" (libfuzzer format, no paths)
    if not frames:
        for match in re.finditer(r"#\d+\s+0x[a-f0-9]+\s+in\s+([^\s<(]+)", log_content):
            func = match.group(1)
            if not _is_noise_func(func):
                frames.append(func)

    # Fallback: "N: 0x... - function_name" (dash format, no paths)
    if not frames:
        for match in re.finditer(r"\d+:\s+0x[a-f0-9]+\s+-\s+([^\s<(]+)", log_content):
            func = match.group(1)
            if not _is_noise_func(func):
                frames.append(func)

    return frames[:10] if frames else ["unknown"]


# Maximum number of lines to keep in raw stack traces.  Deep async/futures
# call chains produce 100+ frames with huge generic signatures that blow past
# token limits in issue bodies and Claude analysis.  The first ~40 lines
# always contain the crash site and immediate callers.
_MAX_RAW_TRACE_LINES = 40


def _truncate_trace(raw: str) -> str:
    """Truncate a raw stack trace to _MAX_RAW_TRACE_LINES."""
    lines = raw.splitlines()
    if len(lines) <= _MAX_RAW_TRACE_LINES:
        return raw
    kept = lines[:_MAX_RAW_TRACE_LINES]
    kept.append(f"   ... ({len(lines) - _MAX_RAW_TRACE_LINES} more frames truncated)")
    return "\n".join(kept)


def extract_stack_trace_raw(log_content: str) -> str:
    """Extract the raw stack trace section from the log.

    Truncated to ~40 lines to avoid enormous issue bodies and token-limit
    failures in downstream Claude analysis.
    """
    # Look for "stack backtrace:" section
    match = re.search(
        r"(stack backtrace:\n(?:.*\n)*?)(?:\n\n|==\d+==|note:)",
        log_content,
    )
    if match:
        return _truncate_trace(match.group(1).strip())

    # Look for "Backtrace:" section (vortex_error format)
    match = re.search(
        r"(Backtrace:\n(?:.*\n)*?)(?:\n\n|\nstack backtrace:|\n==\d+==)",
        log_content,
    )
    if match:
        return _truncate_trace(match.group(1).strip())

    # Look for numbered frame lines with addresses
    lines = []
    for line in log_content.splitlines():
        if re.match(r"\s*#?\d+[:\s]+0x[a-f0-9]+", line):
            lines.append(line)
    if lines:
        return _truncate_trace("\n".join(lines))

    return ""


def extract_debug_output(log_content: str) -> str:
    """Extract the debug output section from the log.

    There may be multiple "Output of `std::fmt::Debug`:" sections. The last one
    (at the end of the log, after the crash) is the most useful — it contains
    the full failing input with tab-indented output.
    """
    # Find all occurrences and take the last one (the post-crash debug dump)
    matches = list(
        re.finditer(
            r"Output of `std::fmt::Debug`:\s*\n(.*?)(?:\nReproduce with:|\n\n(?=[A-Z])|\Z)",
            log_content,
            re.DOTALL,
        )
    )
    if matches:
        # Prefer the last match (post-crash), strip tab indentation
        raw = matches[-1].group(1).strip()
        # Remove leading tab from each line
        lines = [line.lstrip("\t") for line in raw.splitlines()]
        return "\n".join(lines)
    return ""


def get_crash_type(crash_filename: str) -> str:
    """Determine crash type from filename."""
    name = Path(crash_filename).name if crash_filename else ""
    if name.startswith("crash-"):
        return "crash"
    if name.startswith("leak-"):
        return "leak"
    if name.startswith("timeout-"):
        return "timeout"
    if name.startswith("oom-"):
        return "oom"
    return "unknown"


def compute_hash(content: str | bytes) -> str:
    """Compute SHA256 hash."""
    if isinstance(content, str):
        content = content.encode()
    return hashlib.sha256(content).hexdigest()


def normalize_message(message: str) -> str:
    """Normalize message by replacing numbers with N."""
    return re.sub(r"\d+", "N", message)


def extract_crash_info(log_path: str | Path, crash_path: str | Path | None = None) -> CrashInfo:
    """Extract crash information from log file and optional crash seed."""
    log_content = Path(log_path).read_text()

    panic_location = extract_panic_location(log_content)
    crash_location = extract_crash_location(log_content)
    panic_message = extract_panic_message(log_content)
    error_variant = extract_error_variant(log_content)
    stack_frames = extract_stack_frames(log_content)
    stack_trace_raw = extract_stack_trace_raw(log_content)
    debug_output = extract_debug_output(log_content)

    # Compute hashes
    stack_trace_hash = compute_hash("\n".join(stack_frames[:5]))
    normalized_msg = normalize_message(panic_message)
    message_hash = compute_hash(normalized_msg)

    # Compute seed hash if crash file provided
    seed_hash = "unknown"
    if crash_path and Path(crash_path).exists():
        seed_hash = compute_hash(Path(crash_path).read_bytes())

    crash_type = get_crash_type(str(crash_path) if crash_path else "")

    return CrashInfo(
        panic_location=panic_location,
        crash_location=crash_location,
        panic_message=panic_message,
        error_variant=error_variant,
        stack_frames=stack_frames,
        stack_trace_raw=stack_trace_raw,
        debug_output=debug_output,
        seed_hash=seed_hash,
        stack_trace_hash=stack_trace_hash,
        normalized_message=normalized_msg,
        message_hash=message_hash,
        crash_type=crash_type,
    )
