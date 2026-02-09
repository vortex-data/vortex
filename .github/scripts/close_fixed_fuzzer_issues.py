#!/usr/bin/env python3
"""Retest open fuzzer issues and close those whose crashes no longer reproduce.

Usage:
    python3 close_fixed_fuzzer_issues.py --target file_io
    python3 close_fixed_fuzzer_issues.py --target file_io --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass

CLEANUP_MARKER = "Auto-checked by weekly fuzzer issue cleanup"


@dataclass
class FuzzerIssue:
    number: int
    title: str
    target: str
    crash_file: str
    artifact_url: str
    body: str


def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    """Run a command, printing it for visibility."""
    print(f"  $ {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, **kwargs)


def fetch_open_fuzzer_issues(repo: str) -> list[dict]:
    """Fetch all open issues with the 'fuzzer' label."""
    result = run(
        [
            "gh", "issue", "list",
            "--repo", repo,
            "--label", "fuzzer",
            "--state", "open",
            "--json", "number,title,body,url",
            "--limit", "200",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"ERROR: Failed to fetch issues: {result.stderr}", file=sys.stderr)
        sys.exit(1)
    return json.loads(result.stdout)


def parse_issue(issue: dict) -> FuzzerIssue | None:
    """Extract target, crash file, and artifact URL from an issue body."""
    body = issue.get("body", "")

    target_match = re.search(r"\*\*Target\*\*:\s*`([^`]+)`", body)
    crash_file_match = re.search(r"\*\*Crash File\*\*:\s*`([^`]+)`", body)
    artifact_url_match = re.search(r"\*\*Crash Artifact\*\*:\s*(https://\S+)", body)

    if not target_match or not crash_file_match:
        return None

    return FuzzerIssue(
        number=issue["number"],
        title=issue["title"],
        target=target_match.group(1),
        crash_file=crash_file_match.group(1),
        artifact_url=artifact_url_match.group(1) if artifact_url_match else "",
        body=body,
    )


def extract_run_id(artifact_url: str) -> str | None:
    """Extract the workflow run ID from an artifact URL like .../runs/12345/..."""
    match = re.search(r"runs/(\d+)", artifact_url)
    return match.group(1) if match else None


def has_cleanup_comment(repo: str, issue_number: int) -> bool:
    """Check if the issue already has an 'Artifact Unavailable' cleanup comment."""
    result = run(
        [
            "gh", "api",
            f"repos/{repo}/issues/{issue_number}/comments",
            "--jq", f'[.[] | select(.body | contains("{CLEANUP_MARKER}"))] | length',
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return False
    try:
        return int(result.stdout.strip()) > 0
    except ValueError:
        return False


def comment_artifact_unavailable(repo: str, issue_number: int, dry_run: bool) -> None:
    """Comment that the crash artifact is no longer available."""
    body = (
        f"## Artifact Unavailable\n\n"
        f"The crash artifact for this issue is no longer available "
        f"(artifacts expire after 30 days). The crash can no longer be "
        f"automatically retested.\n\n"
        f"If this issue is still relevant, please reproduce manually and "
        f"re-upload the crash file.\n\n"
        f"---\n*{CLEANUP_MARKER}*"
    )
    if dry_run:
        print(f"  [dry-run] Would comment 'Artifact Unavailable' on #{issue_number}")
        return
    run(
        ["gh", "issue", "comment", str(issue_number), "--repo", repo, "--body", body],
        check=True,
    )


def close_issue_as_fixed(repo: str, issue_number: int, target: str, dry_run: bool) -> None:
    """Close the issue with a comment explaining the crash no longer reproduces."""
    body = (
        f"## Crash No Longer Reproduces\n\n"
        f"This crash was retested against the latest `main` branch and "
        f"the fuzzer completed successfully (exit code 0).\n\n"
        f"The underlying bug appears to have been fixed. Closing this issue.\n\n"
        f"If the crash reappears, the fuzzer will automatically open a new issue.\n\n"
        f"---\n*{CLEANUP_MARKER}*"
    )
    if dry_run:
        print(f"  [dry-run] Would close #{issue_number} as fixed")
        return
    run(
        ["gh", "issue", "comment", str(issue_number), "--repo", repo, "--body", body],
        check=True,
    )
    run(
        ["gh", "issue", "close", str(issue_number), "--repo", repo, "--reason", "completed"],
        check=True,
    )


def build_fuzz_target(target: str) -> bool:
    """Build the fuzz target once. Returns True on success."""
    print(f"\nBuilding fuzz target: {target}")
    env = os.environ.copy()
    env["RUSTFLAGS"] = "--cfg vortex_nightly"
    result = run(
        ["cargo", "+nightly", "fuzz", "build", "--dev", "--sanitizer=none", target],
        env=env,
    )
    return result.returncode == 0


def retest_crash(target: str, crash_path: str, timeout_secs: int = 120) -> str:
    """Run the fuzz target with the crash file. Returns 'fixed', 'reproduces', or 'timeout'."""
    env = os.environ.copy()
    env["RUSTFLAGS"] = "--cfg vortex_nightly"
    try:
        result = run(
            [
                "cargo", "+nightly", "fuzz", "run",
                "--dev", "--sanitizer=none",
                target, crash_path,
                "--", "-runs=1", "-rss_limit_mb=0",
            ],
            env=env,
            timeout=timeout_secs,
        )
        if result.returncode == 0:
            return "fixed"
        else:
            return "reproduces"
    except subprocess.TimeoutExpired:
        return "timeout"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Retest open fuzzer issues and close fixed ones.",
    )
    parser.add_argument(
        "--target",
        required=True,
        help="Fuzz target to process (e.g., file_io)",
    )
    parser.add_argument(
        "--repo",
        default=os.environ.get("GITHUB_REPOSITORY", ""),
        help="GitHub repository (owner/name). Defaults to $GITHUB_REPOSITORY.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions without modifying issues.",
    )
    args = parser.parse_args()

    if not args.repo:
        print("ERROR: --repo is required (or set GITHUB_REPOSITORY)", file=sys.stderr)
        sys.exit(1)

    print(f"Processing fuzzer issues for target={args.target} in {args.repo}")
    if args.dry_run:
        print("DRY RUN: no issues will be modified\n")

    # 1. Fetch open fuzzer issues
    raw_issues = fetch_open_fuzzer_issues(args.repo)
    print(f"Found {len(raw_issues)} open fuzzer issue(s)")

    # 2. Parse and filter to matching target
    issues: list[FuzzerIssue] = []
    for raw in raw_issues:
        parsed = parse_issue(raw)
        if parsed and parsed.target == args.target:
            issues.append(parsed)

    print(f"Found {len(issues)} issue(s) matching target={args.target}\n")
    if not issues:
        print("Nothing to do.")
        return

    # 3. Build the fuzz target once
    if not build_fuzz_target(args.target):
        print("ERROR: Failed to build fuzz target", file=sys.stderr)
        sys.exit(1)
    print()

    # 4. Process each issue
    summary: dict[str, list[int]] = {
        "closed": [],
        "still_reproduces": [],
        "artifact_unavailable": [],
        "timeout": [],
        "error": [],
    }

    for issue in issues:
        print(f"--- Issue #{issue.number}: {issue.title}")

        # Extract run ID from artifact URL
        if not issue.artifact_url:
            print(f"  No artifact URL found in issue body")
            if not has_cleanup_comment(args.repo, issue.number):
                comment_artifact_unavailable(args.repo, issue.number, args.dry_run)
            else:
                print(f"  Already commented about artifact unavailability, skipping")
            summary["artifact_unavailable"].append(issue.number)
            continue

        run_id = extract_run_id(issue.artifact_url)
        if not run_id:
            print(f"  Could not extract run ID from: {issue.artifact_url}")
            summary["error"].append(issue.number)
            continue

        # Download artifact into a temp directory
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_name = f"{args.target}-crash-artifacts"
            dl_result = run(
                [
                    "gh", "run", "download", run_id,
                    "--name", artifact_name,
                    "--repo", args.repo,
                    "--dir", tmpdir,
                ],
                capture_output=True,
                text=True,
            )

            if dl_result.returncode != 0:
                print(f"  Artifact download failed: {dl_result.stderr.strip()}")
                if not has_cleanup_comment(args.repo, issue.number):
                    comment_artifact_unavailable(args.repo, issue.number, args.dry_run)
                else:
                    print(f"  Already commented about artifact unavailability, skipping")
                summary["artifact_unavailable"].append(issue.number)
                continue

            # Locate crash file
            crash_path = os.path.join(tmpdir, args.target, issue.crash_file)
            if not os.path.isfile(crash_path):
                # Try without target subdirectory (artifact structure may vary)
                crash_path = os.path.join(tmpdir, issue.crash_file)
                if not os.path.isfile(crash_path):
                    print(f"  Crash file not found: {issue.crash_file}")
                    summary["error"].append(issue.number)
                    continue

            # Retest
            print(f"  Retesting crash file: {issue.crash_file}")
            result = retest_crash(args.target, crash_path)

            if result == "fixed":
                print(f"  Crash NO LONGER reproduces — closing issue")
                close_issue_as_fixed(args.repo, issue.number, args.target, args.dry_run)
                summary["closed"].append(issue.number)
            elif result == "reproduces":
                print(f"  Crash STILL reproduces — leaving open")
                summary["still_reproduces"].append(issue.number)
            elif result == "timeout":
                print(f"  Retest TIMED OUT — skipping")
                summary["timeout"].append(issue.number)

        print()

    # 5. Print summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Closed (fixed):          {summary['closed'] or 'none'}")
    print(f"  Still reproduces:        {summary['still_reproduces'] or 'none'}")
    print(f"  Artifact unavailable:    {summary['artifact_unavailable'] or 'none'}")
    print(f"  Timeout:                 {summary['timeout'] or 'none'}")
    print(f"  Error:                   {summary['error'] or 'none'}")


if __name__ == "__main__":
    main()
