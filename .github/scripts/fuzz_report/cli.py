#!/usr/bin/env python3
"""CLI for fuzzer crash reporting utilities."""

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

from .dedup import check_duplicate
from .extract import CrashInfo, extract_crash_info
from .template import render_template, render_template_to_file

TEMPLATES_DIR = Path(__file__).parent / "templates"

# Marker used to find/update the single recurrence-tracking comment.
_RECURRENCE_MARKER = "<!-- fuzzer-recurrence-tracker -->"
_RECURRENCE_COUNT_RE = r"<!-- fuzzer-recurrence-tracker count:(\d+) -->"
# Variables that must be set (non-empty) before creating or commenting on an issue.
REQUIRED_REPORT_VARIABLES = ["FUZZ_TARGET", "CRASH_FILE", "ARTIFACT_URL"]


def parse_var_arg(var_str: str) -> tuple[str, str]:
    """Parse a -v KEY=VALUE argument into (key, value)."""
    if "=" not in var_str:
        raise argparse.ArgumentTypeError(
            f"Invalid variable format: {var_str!r} (expected KEY=VALUE)"
        )
    key, _, value = var_str.partition("=")
    return key, value


def _write_github_output(key: str, value: str) -> None:
    """Write a key=value pair to GITHUB_OUTPUT if running in Actions."""
    output_file = os.environ.get("GITHUB_OUTPUT")
    if output_file:
        with open(output_file, "a") as f:
            f.write(f"{key}={value}\n")


def _run_gh(cmd: list[str], *, retries: int = 1, **kwargs) -> subprocess.CompletedProcess:
    """Run a gh CLI command with logging and retry on failure.

    Prints the command before execution and surfaces stderr on failure.
    Retries up to ``retries`` times (default 1) with a short back-off.
    """
    print(f"+ {shlex.join(cmd)}", file=sys.stderr)
    last_exc: subprocess.CalledProcessError | None = None
    for attempt in range(1 + retries):
        if attempt > 0:
            wait = 5 * attempt
            print(f"Retrying in {wait}s (attempt {attempt + 1}/{1 + retries})...", file=sys.stderr)
            time.sleep(wait)
        try:
            return subprocess.run(cmd, check=True, **kwargs)
        except subprocess.CalledProcessError as exc:
            last_exc = exc
            stderr_text = (
                exc.stderr if isinstance(exc.stderr, str) else (exc.stderr or b"").decode()
            )
            print(f"gh command failed (exit {exc.returncode}): {stderr_text}", file=sys.stderr)
    raise last_exc  # type: ignore[misc]


def _load_crash_info(path: str | Path) -> CrashInfo:
    """Load CrashInfo from a JSON file."""
    crash_data = json.loads(Path(path).read_text())
    return CrashInfo(**crash_data)


def _find_crash_file(crash_dir: str, crash_name: str) -> str | None:
    """Search for a crash file by name within a directory."""
    for path in Path(crash_dir).rglob(crash_name):
        return str(path)
    return None


def _truncate(text: str, max_chars: int) -> str:
    """Truncate text to max_chars, appending a note if truncated."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n... ({len(text) - max_chars} chars truncated)"


# GitHub issue body limit is 65536 chars. Reserve ~5k for the fixed template
# chrome (headings, summary table, reproduction steps, etc.) and split the
# remaining budget between the two variable-length fields.
_BODY_BUDGET = 60000
_TEMPLATE_OVERHEAD = 5000
_FIELD_BUDGET = _BODY_BUDGET - _TEMPLATE_OVERHEAD  # 55k split between the two


def _build_template_variables(
    crash_info: CrashInfo,
    var_args: list[tuple[str, str]] | None = None,
    claude_analysis: str = "",
) -> dict[str, str]:
    """Build template variables from crash info, CLI args, and Claude analysis."""
    variables = {}
    if var_args:
        for key, value in var_args:
            variables[key] = value

    panic_msg = crash_info.panic_message
    stack_trace = crash_info.stack_trace_raw

    # Truncate the two large fields so their combined size fits the budget.
    combined = len(panic_msg) + len(stack_trace)
    if combined > _FIELD_BUDGET:
        # Give panic_message up to half, stack_trace gets the rest.
        msg_limit = min(len(panic_msg), _FIELD_BUDGET // 2)
        trace_limit = _FIELD_BUDGET - msg_limit
        panic_msg = _truncate(panic_msg, msg_limit)
        stack_trace = _truncate(stack_trace, trace_limit)
        print(
            f"Warning: Truncated issue fields to fit body limit "
            f"(panic_message={msg_limit}, stack_trace={trace_limit})",
            file=sys.stderr,
        )

    # Auto-populate from crash info (don't override explicit -v args)
    auto_vars = {
        "PANIC_MESSAGE": panic_msg,
        "CRASH_LOCATION": crash_info.crash_location,
        "STACK_TRACE_RAW": stack_trace,
        "DEBUG_OUTPUT": crash_info.debug_output,
        "SEED_HASH": crash_info.seed_hash,
        "STACK_TRACE_HASH": crash_info.stack_trace_hash,
        "MESSAGE_HASH": crash_info.message_hash,
    }
    for k, v in auto_vars.items():
        if k not in variables:
            variables[k] = v

    if claude_analysis:
        variables.setdefault("CLAUDE_ANALYSIS", claude_analysis)

    return variables


def _determine_action(
    dedup_path: str | Path | None,
) -> tuple[str, dict | None]:
    """Determine action from dedup result. Returns (action, dedup_dict).

    Actions:
      create       – new issue
      skip         – exact duplicate, do nothing
      update_count – high-confidence duplicate, bump recurrence counter
      comment      – medium-confidence duplicate, post full comment
    """
    if not dedup_path or not Path(dedup_path).exists():
        return "create", None

    dedup = json.loads(Path(dedup_path).read_text())
    if not dedup.get("duplicate", False):
        return "create", dedup

    if dedup.get("confidence") == "exact":
        return "skip", dedup

    if dedup.get("confidence") == "high":
        return "update_count", dedup

    return "comment", dedup


def _render_recurrence_body(count: int) -> str:
    """Render the minimal recurrence-tracking comment body."""
    return (
        f"Seen **{count}** time{'s' if count != 1 else ''}\n\n"
        f"<!-- fuzzer-recurrence-tracker count:{count} -->"
    )


def _update_recurrence_count(repo: str, issue_number: int | str) -> int:
    """Find-or-create the recurrence comment, incrementing its count.

    Uses a compare-and-swap pattern: reads the current count from the
    existing comment (if any), increments it, and writes back.

    Returns the new count.
    """
    # List all comments on the issue
    result = _run_gh(
        [
            "gh",
            "api",
            f"repos/{repo}/issues/{issue_number}/comments",
            "--paginate",
            "--jq",
            f'.[] | select(.body | contains("{_RECURRENCE_MARKER}")) | {{id: .id, body: .body}}',
        ],
        capture_output=True,
        text=True,
    )

    existing_id = None
    current_count = 0

    for line in result.stdout.strip().splitlines():
        if not line:
            continue
        comment = json.loads(line)
        existing_id = comment["id"]
        m = re.search(_RECURRENCE_COUNT_RE, comment["body"])
        if m:
            current_count = int(m.group(1))
        break

    new_count = current_count + 1
    body = _render_recurrence_body(new_count)

    if existing_id:
        # Update existing comment (not atomic — race is acceptable since
        # fuzz CI jobs are serialized)
        _run_gh(
            [
                "gh",
                "api",
                f"repos/{repo}/issues/comments/{existing_id}",
                "-X",
                "PATCH",
                "-f",
                f"body={body}",
            ],
        )
    else:
        # Create new recurrence comment
        _run_gh(
            [
                "gh",
                "api",
                f"repos/{repo}/issues/{issue_number}/comments",
                "-f",
                f"body={body}",
            ],
        )

    return new_count


def cmd_extract(args: argparse.Namespace) -> int:
    """Extract crash info from log file."""
    if not Path(args.log_file).exists():
        print(f"Error: Log file not found: {args.log_file}", file=sys.stderr)
        return 1

    crash_file = args.crash_file
    if not crash_file and args.crash_dir and args.crash_name:
        crash_file = _find_crash_file(args.crash_dir, args.crash_name)

    crash_info = extract_crash_info(args.log_file, crash_file)

    if args.output:
        Path(args.output).write_text(crash_info.to_json())
    else:
        print(crash_info.to_json())

    return 0


def cmd_check_duplicate(args: argparse.Namespace) -> int:
    """Check if crash is a duplicate."""
    if not Path(args.crash_info).exists():
        print(f"Error: Crash info file not found: {args.crash_info}", file=sys.stderr)
        return 1

    try:
        crash_info = _load_crash_info(args.crash_info)
    except (json.JSONDecodeError, TypeError) as e:
        print(f"Error: Invalid crash info: {e}", file=sys.stderr)
        return 1

    result = check_duplicate(crash_info, args.issues)

    if args.output:
        Path(args.output).write_text(result.to_json())
    else:
        print(result.to_json())

    # Write key fields to GITHUB_OUTPUT for conditional steps
    _write_github_output("duplicate", str(result.duplicate).lower())
    if result.duplicate:
        _write_github_output("confidence", result.confidence or "")
        _write_github_output("issue_number", str(result.issue_number or ""))

    return 0


def _validate_required_variables(variables: dict[str, str]) -> list[str]:
    """Return the names of any required variables that are missing or empty."""
    missing = []
    for name in REQUIRED_REPORT_VARIABLES:
        val = variables.get(name, "")
        if not val or val == "(not set)":
            missing.append(name)
    return missing


def cmd_report(args: argparse.Namespace) -> int:
    """Create or comment on a GitHub issue based on crash + dedup results."""
    if not Path(args.crash_info).exists():
        print(f"Error: Crash info not found: {args.crash_info}", file=sys.stderr)
        return 1

    try:
        crash_info = _load_crash_info(args.crash_info)
    except (json.JSONDecodeError, TypeError) as e:
        print(f"Error: Invalid crash info: {e}", file=sys.stderr)
        return 1

    # Read Claude analysis if available
    claude_analysis = ""
    if args.claude_analysis and Path(args.claude_analysis).exists():
        claude_analysis = Path(args.claude_analysis).read_text().strip()

    action, dedup = _determine_action(args.dedup_result)
    variables = _build_template_variables(crash_info, args.var, claude_analysis)
    existing_issue = dedup.get("issue_number") if dedup else None

    # Validate required variables before creating/commenting (skip is fine without them)
    if action != "skip":
        missing = _validate_required_variables(variables)
        if missing:
            print(
                f"Error: Required variables not set: {', '.join(missing)}",
                file=sys.stderr,
            )
            _write_github_output("validation_failed", "true")
            _write_github_output("missing_variables", ", ".join(missing))
            return 1

    if action == "skip":
        print(f"Exact duplicate of #{existing_issue}, skipping.", file=sys.stderr)
        _write_github_output("issue_number", str(existing_issue))
        return 0

    if action == "update_count":
        new_count = _update_recurrence_count(args.repo, existing_issue)
        print(
            f"Updated recurrence count on #{existing_issue} to {new_count}",
            file=sys.stderr,
        )
        _write_github_output("issue_number", str(existing_issue))
        return 0

    if action == "comment":
        variables.setdefault("DEDUP_REASON", dedup.get("reason", ""))
        variables.setdefault("DEDUP_CONFIDENCE", dedup.get("confidence", ""))

        body = render_template(str(TEMPLATES_DIR / "related_comment.md"), variables, use_env=False)
        body_file = Path("comment_body.md")
        body_file.write_text(body)

        _run_gh(
            [
                "gh",
                "issue",
                "comment",
                str(existing_issue),
                "--repo",
                args.repo,
                "--body-file",
                str(body_file),
            ],
        )
        print(f"Commented on #{existing_issue}", file=sys.stderr)
        _write_github_output("issue_number", str(existing_issue))
    else:
        fuzz_target = variables.get("FUZZ_TARGET", "unknown")
        title = f"Fuzzing Crash: {crash_info.error_variant} in {fuzz_target}"

        body = render_template(str(TEMPLATES_DIR / "new_issue.md"), variables, use_env=False)
        body_file = Path("issue_body.md")
        body_file.write_text(body)

        print(f"Issue title: {title}", file=sys.stderr)
        print(f"Issue body size: {len(body)} chars", file=sys.stderr)
        print(f"Repo: {args.repo}", file=sys.stderr)

        result = _run_gh(
            [
                "gh",
                "issue",
                "create",
                "--repo",
                args.repo,
                "--title",
                title,
                "--label",
                "bug,fuzzer",
                "--body-file",
                str(body_file),
            ],
            capture_output=True,
            text=True,
        )
        issue_url = result.stdout.strip()
        issue_number = issue_url.rstrip("/").split("/")[-1]

        print(f"Created issue #{issue_number}: {issue_url}", file=sys.stderr)
        _write_github_output("issue_number", issue_number)

        # Post full debug output as a follow-up comment (collapsed).
        debug_output = variables.get("DEBUG_OUTPUT", "")
        if debug_output and debug_output != "(not set)":
            comment_body = (
                "<details>\n<summary>Debug Output</summary>\n\n"
                f"```\n{debug_output}\n```\n</details>"
            )
            # Truncate comment to GitHub's limit too.
            if len(comment_body) > _BODY_BUDGET:
                comment_body = (
                    comment_body[: _BODY_BUDGET - 50] + "\n```\n</details>\n\n*Truncated*"
                )
            comment_file = Path("debug_comment.md")
            comment_file.write_text(comment_body)
            try:
                _run_gh(
                    [
                        "gh",
                        "issue",
                        "comment",
                        issue_number,
                        "--repo",
                        args.repo,
                        "--body-file",
                        str(comment_file),
                    ],
                )
            except subprocess.CalledProcessError:
                print("Warning: failed to post debug output comment", file=sys.stderr)

    return 0


def cmd_dry_run(args: argparse.Namespace) -> int:
    """Run the full pipeline (extract -> dedup -> render) without creating issues."""
    if not Path(args.log_file).exists():
        print(f"Error: Log file not found: {args.log_file}", file=sys.stderr)
        return 1

    # Step 1: Extract
    crash_file = args.crash_file
    if not crash_file and args.crash_dir and args.crash_name:
        crash_file = _find_crash_file(args.crash_dir, args.crash_name)

    crash_info = extract_crash_info(args.log_file, crash_file)
    print("=== Crash Info ===", file=sys.stderr)
    print(f"  panic_location: {crash_info.panic_location}", file=sys.stderr)
    print(f"  crash_location: {crash_info.crash_location}", file=sys.stderr)
    print(f"  error_variant:  {crash_info.error_variant}", file=sys.stderr)
    print(f"  panic_message:  {crash_info.panic_message}", file=sys.stderr)
    print(f"  crash_type:     {crash_info.crash_type}", file=sys.stderr)
    print(f"  seed_hash:      {crash_info.seed_hash}", file=sys.stderr)
    print(f"  stack_frames:   {crash_info.stack_frames[:5]}", file=sys.stderr)
    print(file=sys.stderr)

    # Step 2: Dedup (if issues file provided)
    dedup_result = None
    if args.issues:
        if not Path(args.issues).exists():
            print(f"Error: Issues file not found: {args.issues}", file=sys.stderr)
            return 1
        dedup_result = check_duplicate(crash_info, args.issues)
        print("=== Dedup Result ===", file=sys.stderr)
        print(f"  duplicate:  {dedup_result.duplicate}", file=sys.stderr)
        if dedup_result.duplicate:
            print(f"  check:      {dedup_result.check}", file=sys.stderr)
            print(f"  confidence: {dedup_result.confidence}", file=sys.stderr)
            print(f"  issue:      #{dedup_result.issue_number}", file=sys.stderr)
            print(f"  reason:     {dedup_result.reason}", file=sys.stderr)
        if dedup_result.debug:
            print(f"  debug:      {json.dumps(dedup_result.debug, indent=4)}", file=sys.stderr)
        print(file=sys.stderr)

    # Write dedup to temp file so _determine_action can read it
    dedup_path = None
    if dedup_result:
        dedup_path = Path("/tmp/dedup_result.json")
        dedup_path.write_text(dedup_result.to_json())

    action, dedup = _determine_action(dedup_path)

    # Read Claude analysis if provided
    claude_analysis = ""
    if args.claude_analysis and Path(args.claude_analysis).exists():
        claude_analysis = Path(args.claude_analysis).read_text().strip()

    variables = _build_template_variables(crash_info, args.var, claude_analysis)
    existing_issue = dedup.get("issue_number") if dedup else None

    # Validate required variables (same check as real report)
    if action != "skip":
        missing = _validate_required_variables(variables)
        if missing:
            print(
                f"Warning: Required variables not set: {', '.join(missing)}",
                file=sys.stderr,
            )

    print(f"=== Action: {action.upper()} ===", file=sys.stderr)

    if action == "skip":
        print(
            f"(exact duplicate of #{existing_issue}, no issue/comment would be created)",
            file=sys.stderr,
        )
        return 0

    if action == "update_count":
        print(
            f"(would update recurrence count on #{existing_issue})",
            file=sys.stderr,
        )
        print(file=sys.stderr)
        print(_render_recurrence_body(1))
        return 0

    if action == "comment":
        template_path = TEMPLATES_DIR / "related_comment.md"
        variables.setdefault("DEDUP_REASON", dedup.get("reason", ""))
        variables.setdefault("DEDUP_CONFIDENCE", dedup.get("confidence", ""))
        print(f"(would comment on #{existing_issue})", file=sys.stderr)
    else:
        template_path = TEMPLATES_DIR / "new_issue.md"
        print("(would create new issue)", file=sys.stderr)

    print(file=sys.stderr)
    print(render_template(str(template_path), variables, use_env=False))
    return 0


def cmd_render(args: argparse.Namespace) -> int:
    """Render a template."""
    if not Path(args.template).exists():
        print(f"Error: Template file not found: {args.template}", file=sys.stderr)
        return 1

    variables = {}
    if args.var:
        for key, value in args.var:
            variables[key] = value

    if args.output:
        render_template_to_file(args.template, args.output, variables)
    else:
        print(render_template(args.template, variables))

    return 0


def _add_var_args(parser: argparse.ArgumentParser) -> None:
    """Add -v/--var argument to a parser."""
    parser.add_argument(
        "-v",
        "--var",
        action="append",
        type=parse_var_arg,
        metavar="KEY=VALUE",
        help="Set a template variable (can be repeated)",
    )


def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Fuzzer crash reporting utilities",
        prog="fuzz_report",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Extract command
    extract_parser = subparsers.add_parser("extract", help="Extract crash info from fuzzer log")
    extract_parser.add_argument("log_file", help="Path to fuzzer output log")
    extract_parser.add_argument("crash_file", nargs="?", help="Path to crash seed file (optional)")
    extract_parser.add_argument("--crash-dir", help="Directory to search for crash file")
    extract_parser.add_argument("--crash-name", help="Crash file name to find in --crash-dir")
    extract_parser.add_argument("-o", "--output", help="Output JSON file")
    extract_parser.set_defaults(func=cmd_extract)

    # Check-duplicate command
    dedup_parser = subparsers.add_parser(
        "check-duplicate", help="Check if crash is a duplicate of existing issues"
    )
    dedup_parser.add_argument("crash_info", help="Path to crash info JSON")
    dedup_parser.add_argument("issues", help="Path to issues JSON (from gh issue list)")
    dedup_parser.add_argument("-o", "--output", help="Output JSON file")
    dedup_parser.set_defaults(func=cmd_check_duplicate)

    # Report command (create/comment/skip on GitHub)
    report_parser = subparsers.add_parser("report", help="Create or comment on a GitHub issue")
    report_parser.add_argument("crash_info", help="Path to crash info JSON")
    report_parser.add_argument("--repo", required=True, help="GitHub repo (owner/name)")
    report_parser.add_argument("--dedup-result", help="Path to dedup result JSON")
    report_parser.add_argument("--claude-analysis", help="Path to Claude analysis text")
    _add_var_args(report_parser)
    report_parser.set_defaults(func=cmd_report)

    # Dry-run command
    dryrun_parser = subparsers.add_parser("dry-run", help="Full pipeline without creating issues")
    dryrun_parser.add_argument("log_file", help="Path to fuzzer output log")
    dryrun_parser.add_argument("crash_file", nargs="?", help="Path to crash seed file (optional)")
    dryrun_parser.add_argument("--crash-dir", help="Directory to search for crash file")
    dryrun_parser.add_argument("--crash-name", help="Crash file name to find in --crash-dir")
    dryrun_parser.add_argument("--issues", help="Path to issues JSON for dedup check")
    dryrun_parser.add_argument("--claude-analysis", help="Path to Claude analysis text")
    _add_var_args(dryrun_parser)
    dryrun_parser.set_defaults(func=cmd_dry_run)

    # Render command
    render_parser = subparsers.add_parser("render", help="Render a template with variables")
    render_parser.add_argument("template", help="Path to template file")
    render_parser.add_argument("-o", "--output", help="Output file")
    _add_var_args(render_parser)
    render_parser.set_defaults(func=cmd_render)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
