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


def extract_panic_location(log_content: str) -> str:
    """Extract panic location (file:line) from log."""
    # Look for "panicked at file:line:" pattern (newer Rust format)
    match = re.search(r"panicked at ([^:]+\.rs:\d+)", log_content)
    if match:
        return match.group(1)

    # Look for "panicked at 'msg', file:line" pattern (older Rust format)
    match = re.search(r"panicked at [^,]+, ([^:]+:\d+)", log_content)
    if match:
        return match.group(1)

    # Extract from vortex path in log
    match = re.search(r"(vortex[^/]+/src/[^:]+:\d+)", log_content)
    if match:
        return match.group(1)

    return "unknown"


def extract_crash_location(log_content: str) -> str:
    """Extract crash location as file:function_name from stack frames."""
    # Look for first vortex frame in various stack trace formats
    # Format 1: "#N 0x... in function_name"
    # Format 2: "N: 0x... - function_name"
    # Format 3: "#N 0x... in function_name /path/file.rs:line"
    # Format 4: "N: function_name\n  at ./path/file.rs:line"
    func_name = None

    # Try "#N 0x... in vortex..." format
    match = re.search(r"#\d+\s+0x[a-f0-9]+\s+in\s+(vortex[^\s<(]+)", log_content)
    if match:
        func_name = match.group(1)

    # Try "N: 0x... - vortex..." format
    if not func_name:
        match = re.search(r"\d+:\s+0x[a-f0-9]+\s+-\s+(vortex[^\s<(]+)", log_content)
        if match:
            func_name = match.group(1)

    # Try "N: function_name\n  at ./path" format (Rust backtrace)
    # The `at ./` regex excludes /rustc/ stdlib frames; _is_noise_frame()
    # further excludes vortex-error boilerplate and closure wrappers.
    if not func_name:
        for m in re.finditer(r"\s+\d+:\s+(\S+)\n\s+at\s+\./([^\n]+)", log_content):
            name = m.group(1)
            path = m.group(2)
            if _is_noise_frame(name, path):
                continue
            func_name = re.sub(r"<.*", "", name)
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


def _is_noise_frame(func_name: str, path: str) -> bool:
    """Return True if this stack frame is panic/error-handling boilerplate.

    Two layers of noise are filtered:

    1. Frames from /rustc/ stdlib (rust_begin_unwind, panic_fmt, etc.) are
       already excluded by the `at ./` regex — they have `at /rustc/...` paths,
       so the regex never matches them.

    2. Frames whose path starts with an entry in NOISE_FRAME_PATHS. These are
       project-local but are still infrastructure (e.g. vortex_expect,
       vortex_unwrap in vortex-error/src/lib.rs).

    3. Closure wrappers like {closure#0} that appear in generic unwrap/expect
       call chains.
    """
    clean = re.sub(r"<.*", "", func_name)
    if clean.startswith("{"):
        return True
    if any(path.startswith(prefix) for prefix in NOISE_FRAME_PATHS):
        return True
    return False


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

    # Fallback: "#N 0x... in function_name"
    if not frames:
        for match in re.finditer(r"#\d+\s+0x[a-f0-9]+\s+in\s+([^\s<(]+)", log_content):
            func = match.group(1)
            if func.startswith(("vortex", "std", "core", "alloc")):
                frames.append(func)

    # Fallback: "N: 0x... - function_name"
    if not frames:
        for match in re.finditer(r"\d+:\s+0x[a-f0-9]+\s+-\s+([^\s<(]+)", log_content):
            func = match.group(1)
            if func.startswith(("vortex", "std", "core", "alloc")):
                frames.append(func)

    return frames[:10] if frames else ["unknown"]


def extract_stack_trace_raw(log_content: str) -> str:
    """Extract the raw stack trace section from the log."""
    # Look for "stack backtrace:" section
    match = re.search(
        r"(stack backtrace:\n(?:.*\n)*?)(?:\n\n|==\d+==|note:)",
        log_content,
    )
    if match:
        return match.group(1).strip()

    # Look for "Backtrace:" section (vortex_error format)
    match = re.search(
        r"(Backtrace:\n(?:.*\n)*?)(?:\n\n|\nstack backtrace:|\n==\d+==)",
        log_content,
    )
    if match:
        return match.group(1).strip()

    # Look for numbered frame lines with addresses
    lines = []
    for line in log_content.splitlines():
        if re.match(r"\s*#?\d+[:\s]+0x[a-f0-9]+", line):
            lines.append(line)
    if lines:
        return "\n".join(lines)

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
