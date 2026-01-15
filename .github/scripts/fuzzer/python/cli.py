#!/usr/bin/env python3
"""CLI for fuzzer crash reporting utilities."""

import argparse
import json
import sys
from pathlib import Path

from .extract import extract_crash_info
from .dedup import check_duplicate
from .template import render_template, render_template_to_file


def cmd_extract(args: argparse.Namespace) -> int:
    """Extract crash info from log file."""
    crash_info = extract_crash_info(args.log_file, args.crash_file)

    if args.output:
        Path(args.output).write_text(crash_info.to_json())
    else:
        print(crash_info.to_json())

    return 0


def cmd_check_duplicate(args: argparse.Namespace) -> int:
    """Check if crash is a duplicate."""
    # Load crash info from JSON
    crash_data = json.loads(Path(args.crash_info).read_text())

    # Create CrashInfo object
    from .extract import CrashInfo

    crash_info = CrashInfo(**crash_data)

    result = check_duplicate(crash_info, args.issues)
    print(result.to_json())

    return 0


def cmd_render(args: argparse.Namespace) -> int:
    """Render a template."""
    if args.output:
        render_template_to_file(args.template, args.output)
    else:
        print(render_template(args.template))

    return 0


def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Fuzzer crash reporting utilities",
        prog="fuzzer-report",
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
    dedup_parser.set_defaults(func=cmd_check_duplicate)

    # Render command
    render_parser = subparsers.add_parser(
        "render",
        help="Render a template with environment variables",
    )
    render_parser.add_argument("template", help="Path to template file")
    render_parser.add_argument(
        "-o",
        "--output",
        help="Output file (default: stdout)",
    )
    render_parser.set_defaults(func=cmd_render)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
