#!/usr/bin/env python3
"""CLI for fuzzer crash reporting utilities."""

import argparse
import json
import sys
from pathlib import Path

from .dedup import check_duplicate
from .extract import CrashInfo, extract_crash_info
from .template import render_template, render_template_to_file


def parse_var_arg(var_str: str) -> tuple[str, str]:
    """Parse a -v KEY=VALUE argument into (key, value)."""
    if "=" not in var_str:
        raise argparse.ArgumentTypeError(f"Invalid variable format: {var_str!r} (expected KEY=VALUE)")
    key, _, value = var_str.partition("=")
    return key, value


def cmd_extract(args: argparse.Namespace) -> int:
    """Extract crash info from log file."""
    if not Path(args.log_file).exists():
        print(f"Error: Log file not found: {args.log_file}", file=sys.stderr)
        return 1

    crash_info = extract_crash_info(args.log_file, args.crash_file)

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

    # Load crash info from JSON
    try:
        crash_data = json.loads(Path(args.crash_info).read_text())
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in crash info file: {e}", file=sys.stderr)
        return 1

    try:
        crash_info = CrashInfo(**crash_data)
    except TypeError as e:
        print(f"Error: Invalid crash info format: {e}", file=sys.stderr)
        return 1

    result = check_duplicate(crash_info, args.issues)

    if args.output:
        Path(args.output).write_text(result.to_json())
    else:
        print(result.to_json())

    return 0


def cmd_dry_run(args: argparse.Namespace) -> int:
    """Run the full pipeline (extract → dedup → render) without creating issues."""
    if not Path(args.log_file).exists():
        print(f"Error: Log file not found: {args.log_file}", file=sys.stderr)
        return 1

    # Step 1: Extract
    crash_info = extract_crash_info(args.log_file, args.crash_file)
    print("=== Crash Info ===", file=sys.stderr)
    print(f"  panic_location: {crash_info.panic_location}", file=sys.stderr)
    print(f"  crash_location: {crash_info.crash_location}", file=sys.stderr)
    print(f"  error_variant:  {crash_info.error_variant}", file=sys.stderr)
    print(f"  panic_message:  {crash_info.panic_message}", file=sys.stderr)
    print(f"  crash_type:     {crash_info.crash_type}", file=sys.stderr)
    print(f"  seed_hash:      {crash_info.seed_hash}", file=sys.stderr)
    print(file=sys.stderr)

    # Step 2: Dedup (if issues file provided)
    action = "create"
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
            if dedup_result.confidence == "exact":
                action = "skip"
            else:
                action = "comment"
        print(file=sys.stderr)

    print(f"=== Action: {action.upper()} ===", file=sys.stderr)

    # Step 3: Render the appropriate template
    templates_dir = Path(__file__).parent / "templates"

    # Build variables from -v args and crash info
    variables = {}
    if args.var:
        for key, value in args.var:
            variables[key] = value

    # Auto-populate from crash info if not overridden
    auto_vars = {
        "PANIC_MESSAGE": crash_info.panic_message,
        "CRASH_LOCATION": crash_info.crash_location,
        "STACK_TRACE_RAW": crash_info.stack_trace_raw,
        "DEBUG_OUTPUT": crash_info.debug_output,
        "SEED_HASH": crash_info.seed_hash,
        "STACK_TRACE_HASH": crash_info.stack_trace_hash,
        "MESSAGE_HASH": crash_info.message_hash,
    }
    for k, v in auto_vars.items():
        if k not in variables:
            variables[k] = v

    if action == "skip":
        print(
            f"(exact duplicate of #{dedup_result.issue_number}, no issue/comment would be created)",
            file=sys.stderr,
        )
        return 0

    if action == "comment":
        template_path = templates_dir / "related_comment.md"
        variables.setdefault("DEDUP_REASON", dedup_result.reason)
        variables.setdefault("DEDUP_CONFIDENCE", dedup_result.confidence)
        print(f"(would comment on #{dedup_result.issue_number})", file=sys.stderr)
    else:
        template_path = templates_dir / "new_issue.md"
        print("(would create new issue)", file=sys.stderr)

    print(file=sys.stderr)
    print(render_template(str(template_path), variables, use_env=False))
    return 0


def cmd_render(args: argparse.Namespace) -> int:
    """Render a template."""
    if not Path(args.template).exists():
        print(f"Error: Template file not found: {args.template}", file=sys.stderr)
        return 1

    # Build variables dict from -v args
    variables = {}
    if args.var:
        for key, value in args.var:
            variables[key] = value

    if args.output:
        render_template_to_file(args.template, args.output, variables)
    else:
        print(render_template(args.template, variables))

    return 0


def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Fuzzer crash reporting utilities",
        prog="fuzz_report",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Extract command
    extract_parser = subparsers.add_parser(
        "extract",
        help="Extract crash info from fuzzer log",
    )
    extract_parser.add_argument("log_file", help="Path to fuzzer output log")
    extract_parser.add_argument(
        "crash_file",
        nargs="?",
        help="Path to crash seed file (optional)",
    )
    extract_parser.add_argument(
        "-o",
        "--output",
        help="Output JSON file (default: stdout)",
    )
    extract_parser.set_defaults(func=cmd_extract)

    # Check-duplicate command
    dedup_parser = subparsers.add_parser(
        "check-duplicate",
        help="Check if crash is a duplicate of existing issues",
    )
    dedup_parser.add_argument("crash_info", help="Path to crash info JSON")
    dedup_parser.add_argument(
        "issues",
        help="Path to issues JSON (from gh issue list)",
    )
    dedup_parser.add_argument(
        "-o",
        "--output",
        help="Output JSON file (default: stdout)",
    )
    dedup_parser.set_defaults(func=cmd_check_duplicate)

    # Dry-run command
    dryrun_parser = subparsers.add_parser(
        "dry-run",
        help="Full pipeline (extract + dedup + render) without creating issues",
    )
    dryrun_parser.add_argument("log_file", help="Path to fuzzer output log")
    dryrun_parser.add_argument(
        "crash_file",
        nargs="?",
        help="Path to crash seed file (optional)",
    )
    dryrun_parser.add_argument(
        "--issues",
        help="Path to issues JSON for dedup check (optional)",
    )
    dryrun_parser.add_argument(
        "-v",
        "--var",
        action="append",
        type=parse_var_arg,
        metavar="KEY=VALUE",
        help="Set a template variable (can be repeated, e.g. -v FUZZ_TARGET=file_io)",
    )
    dryrun_parser.set_defaults(func=cmd_dry_run)

    # Render command
    render_parser = subparsers.add_parser(
        "render",
        help="Render a template with variables",
    )
    render_parser.add_argument("template", help="Path to template file")
    render_parser.add_argument(
        "-o",
        "--output",
        help="Output file (default: stdout)",
    )
    render_parser.add_argument(
        "-v",
        "--var",
        action="append",
        type=parse_var_arg,
        metavar="KEY=VALUE",
        help="Set a template variable (can be repeated)",
    )
    render_parser.set_defaults(func=cmd_render)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
