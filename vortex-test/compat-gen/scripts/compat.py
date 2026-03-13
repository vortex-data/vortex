#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Unified CLI for Vortex backward-compatibility testing.

Subcommands:
  add-version   Generate fixtures for a version and optionally upload to S3.
  check         Validate fixtures against the current reader.

Both subcommands support local directories and S3 as targets.

Examples:
  # Generate fixtures locally (dry-run, no S3)
  python3 compat.py add-version --version 0.63.0 --target local:/tmp/compat

  # Generate and upload to S3
  python3 compat.py add-version --version 0.63.0 --target s3

  # Validate all versions from S3
  python3 compat.py check --target s3

  # Validate specific versions from a local directory
  python3 compat.py check --target local:/tmp/compat --versions 0.62.0,0.63.0
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

S3_BUCKET = "vortex-compat-fixtures"
FIXTURES_URL = "https://vortex-compat-fixtures.s3.amazonaws.com"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def log(msg: str) -> None:
    print(msg, file=sys.stderr)


def run(cmd: list[str], *, check: bool = True, **kwargs) -> subprocess.CompletedProcess:
    log(f"  $ {' '.join(cmd)}")
    return subprocess.run(cmd, check=check, **kwargs)


def http_get(url: str) -> bytes | None:
    """Fetch *url* over HTTPS.  Returns None on 404/403, raises on other errors."""
    try:
        with urllib.request.urlopen(url) as resp:
            return resp.read()
    except urllib.error.HTTPError as exc:
        if exc.code in (403, 404):
            return None
        raise


def version_sort_key(v: str) -> list[int]:
    return list(map(int, v.split(".")))


# ---------------------------------------------------------------------------
# Target parsing
# ---------------------------------------------------------------------------


class Target:
    """Parsed --target value: either local:<path> or s3."""

    def __init__(self, spec: str):
        if spec == "s3":
            self.kind = "s3"
            self.path = None
        elif spec.startswith("local:"):
            self.kind = "local"
            self.path = os.path.abspath(spec[len("local:"):])
        else:
            raise argparse.ArgumentTypeError(
                f"invalid target '{spec}': use 's3' or 'local:<path>'"
            )

    def __repr__(self) -> str:
        if self.kind == "s3":
            return "s3"
        return f"local:{self.path}"

    @property
    def is_s3(self) -> bool:
        return self.kind == "s3"

    @property
    def is_local(self) -> bool:
        return self.kind == "local"


# ---------------------------------------------------------------------------
# S3 helpers
# ---------------------------------------------------------------------------


def head_etag(bucket: str, key: str) -> str | None:
    """Fetch the current ETag for an S3 object, or None if missing."""
    result = subprocess.run(
        [
            "aws", "s3api", "head-object",
            "--bucket", bucket,
            "--key", key,
            "--query", "ETag",
            "--output", "text",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    etag = result.stdout.strip()
    if not etag or etag == "null":
        return None
    return etag


def put_object(bucket: str, key: str, body: str, if_match: str | None) -> bool:
    """Upload a single object with optional ETag precondition."""
    cmd = [
        "aws", "s3api", "put-object",
        "--bucket", bucket,
        "--key", key,
        "--body", body,
    ]
    if if_match:
        cmd.extend(["--if-match", if_match])
    result = subprocess.run(cmd, capture_output=True)
    return result.returncode == 0


def upload_versions_json(local_path: str, max_retries: int = 5) -> None:
    """Upload versions.json with ETag-based optimistic locking + retry."""
    key = "versions.json"
    for attempt in range(1, max_retries + 1):
        etag = head_etag(S3_BUCKET, key)
        if put_object(S3_BUCKET, key, local_path, etag):
            log("  versions.json uploaded.")
            return

        if attempt == max_retries:
            break

        delay = min(2**attempt, 30)
        log(f"  versions.json upload failed (attempt {attempt}/{max_retries}), retrying in {delay}s...")
        time.sleep(delay)

    log(f"ERROR: versions.json upload failed after {max_retries} attempts")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Shared logic
# ---------------------------------------------------------------------------


def fetch_versions_from_s3() -> list[str]:
    """Fetch the current versions.json from S3 (public HTTP)."""
    data = http_get(f"{FIXTURES_URL}/versions.json")
    if data is None:
        return []
    return json.loads(data)


def fetch_versions_from_local(root: str) -> list[str]:
    """Discover versions from a local directory tree."""
    versions = []
    if not os.path.isdir(root):
        return versions
    for entry in os.listdir(root):
        if entry.startswith("v"):
            version = entry[1:]
            manifest = os.path.join(root, entry, "manifest.json")
            if os.path.isfile(manifest):
                versions.append(version)
    versions.sort(key=version_sort_key)
    return versions


def fetch_previous_manifest_from_s3(versions: list[str], current_version: str) -> dict | None:
    """Fetch the manifest.json for the latest version before *current_version*."""
    candidates = [v for v in versions if v != current_version]
    if not candidates:
        return None
    candidates.sort(key=version_sort_key)
    latest = candidates[-1]
    log(f"  previous version: {latest}")
    data = http_get(f"{FIXTURES_URL}/v{latest}/manifest.json")
    if data is None:
        return None
    return json.loads(data)


def fetch_previous_manifest_from_local(root: str, versions: list[str], current_version: str) -> dict | None:
    """Fetch the manifest.json for the latest local version before *current_version*."""
    candidates = [v for v in versions if v != current_version]
    if not candidates:
        return None
    candidates.sort(key=version_sort_key)
    latest = candidates[-1]
    log(f"  previous version: {latest}")
    manifest_path = os.path.join(root, f"v{latest}", "manifest.json")
    if not os.path.isfile(manifest_path):
        return None
    with open(manifest_path) as f:
        return json.load(f)


def normalize_manifest_fixtures(manifest: dict) -> list[dict]:
    """Handle old manifest format where fixtures was a list of strings."""
    entries = manifest.get("fixtures", [])
    normalized = []
    for entry in entries:
        if isinstance(entry, str):
            normalized.append({"name": entry, "since": "unknown"})
        else:
            normalized.append(entry)
    return normalized


def merge_manifest(
    generated_manifest_path: str,
    previous_manifest: dict | None,
    current_version: str,
) -> None:
    """Merge `since` values from the previous manifest into the generated one.

    Also enforces the additive-only rule: every fixture in the previous manifest
    must exist in the generated output.
    """
    with open(generated_manifest_path) as f:
        generated = json.load(f)

    if previous_manifest is None:
        return

    prev_fixtures = normalize_manifest_fixtures(previous_manifest)
    prev_by_name = {e["name"]: e for e in prev_fixtures}
    gen_by_name = {e["name"]: e for e in generated["fixtures"]}

    # Additive-only check.
    missing = sorted(set(prev_by_name) - set(gen_by_name))
    if missing:
        log(f"ERROR: fixtures removed since previous version: {missing}")
        log("Fixtures must never be removed — only added.")
        sys.exit(1)

    # Merge since values.
    for entry in generated["fixtures"]:
        name = entry["name"]
        if name in prev_by_name:
            entry["since"] = prev_by_name[name]["since"]
        else:
            entry["since"] = current_version

    with open(generated_manifest_path, "w") as f:
        json.dump(generated, f, indent=2)
        f.write("\n")

    log(f"  merged manifest: {len(prev_by_name)} existing, {len(gen_by_name) - len(prev_by_name)} new fixtures")


def build_fixtures(version: str, output_dir: str) -> None:
    """Run cargo to build and execute compat-gen."""
    run([
        "cargo", "run", "-p", "vortex-compat", "--release",
        "--bin", "compat-gen", "--",
        "--version", version,
        "--output", output_dir,
    ])


# ---------------------------------------------------------------------------
# add-version subcommand
# ---------------------------------------------------------------------------


def cmd_add_version(args: argparse.Namespace) -> None:
    target = Target(args.target)
    version = args.version

    # Determine output directory.
    if target.is_local:
        output_dir = os.path.join(target.path, f"v{version}")
        os.makedirs(output_dir, exist_ok=True)
        owns_tmp = False
    else:
        if args.output:
            output_dir = args.output
            os.makedirs(output_dir, exist_ok=True)
            owns_tmp = False
        else:
            tmp = tempfile.mkdtemp(prefix="compat-gen-")
            output_dir = os.path.join(tmp, "fixtures")
            os.makedirs(output_dir)
            owns_tmp = True

    try:
        # Step 1: Build + generate fixtures.
        if not args.skip_build:
            log(f"[1/4] Generating fixtures for v{version}...")
            build_fixtures(version, output_dir)
        else:
            log(f"[1/4] Skipping build (--skip-build), using {output_dir}")

        # Step 2: Fetch previous manifest and merge.
        log("[2/4] Fetching previous manifest...")
        if target.is_local:
            versions = fetch_versions_from_local(target.path)
            prev_manifest = fetch_previous_manifest_from_local(target.path, versions, version)
        else:
            versions = fetch_versions_from_s3()
            prev_manifest = fetch_previous_manifest_from_s3(versions, version)
        manifest_path = os.path.join(output_dir, "manifest.json")
        merge_manifest(manifest_path, prev_manifest, version)

        if target.is_local:
            # For local targets, update a local versions.json.
            log("[3/4] Local target — no S3 upload needed.")
            log("[4/4] Updating local versions.json...")
            if version not in versions:
                versions.append(version)
                versions.sort(key=version_sort_key)
            versions_path = os.path.join(target.path, "versions.json")
            with open(versions_path, "w") as f:
                json.dump(versions, f, indent=2)
                f.write("\n")
            log(f"\nDone: fixtures for v{version} written to {output_dir}")
            with open(manifest_path) as f:
                log(f"Manifest:\n{f.read()}")
            return

        if args.dry_run:
            log("[3/4] Dry run — skipping S3 upload.")
            log("[4/4] Dry run — skipping versions.json update.")
            log(f"\nGenerated fixtures in: {output_dir}")
            with open(manifest_path) as f:
                log(f"Manifest:\n{f.read()}")
            return

        # Step 3: Upload fixtures to S3.
        log(f"[3/4] Uploading fixtures to s3://{S3_BUCKET}/v{version}/...")
        run([
            "aws", "s3", "cp", output_dir,
            f"s3://{S3_BUCKET}/v{version}/",
            "--recursive",
        ])

        # Step 4: Update versions.json.
        log("[4/4] Updating versions.json...")
        if version not in versions:
            versions.append(version)
            versions.sort(key=version_sort_key)
        tmp_dir = os.path.dirname(output_dir) if owns_tmp else tempfile.mkdtemp()
        local_versions_path = os.path.join(tmp_dir, "versions.json")
        with open(local_versions_path, "w") as f:
            json.dump(versions, f, indent=2)
            f.write("\n")
        upload_versions_json(local_versions_path)

        log(f"\nDone: fixtures for v{version} uploaded.")
    finally:
        if owns_tmp and not args.dry_run and not (target.is_local):
            import shutil
            shutil.rmtree(os.path.dirname(output_dir), ignore_errors=True)


# ---------------------------------------------------------------------------
# check subcommand
# ---------------------------------------------------------------------------


def cmd_check(args: argparse.Namespace) -> None:
    target = Target(args.target)

    # Build the cargo command for compat-validate.
    cmd = [
        "cargo", "run", "-p", "vortex-compat", "--release",
        "--bin", "compat-validate", "--",
    ]

    if target.is_s3:
        cmd.extend(["--fixtures-url", FIXTURES_URL])
    else:
        cmd.extend(["--fixtures-dir", target.path])

    if args.versions:
        cmd.extend(["--versions", args.versions])

    run(cmd)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="compat",
        description="Unified CLI for Vortex backward-compatibility testing.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # -- add-version ---
    add_parser = subparsers.add_parser(
        "add-version",
        help="Generate fixtures for a new version and store them.",
        description=(
            "Build fixture files for a version, merge manifests, and store "
            "them to a local directory or upload to S3."
        ),
    )
    add_parser.add_argument(
        "--version",
        required=True,
        help='Version tag for this fixture set (e.g. "0.63.0").',
    )
    add_parser.add_argument(
        "--target",
        required=True,
        help=(
            'Where to store fixtures. Use "s3" for the shared bucket, '
            'or "local:<path>" for a local directory (e.g. "local:/tmp/compat").'
        ),
    )
    add_parser.add_argument(
        "--output",
        help="Override output directory for generated fixtures (S3 target only).",
    )
    add_parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Skip cargo build + compat-gen run (assumes output already populated).",
    )
    add_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate and merge manifest but skip S3 upload (S3 target only).",
    )

    # -- check ---
    check_parser = subparsers.add_parser(
        "check",
        help="Validate fixtures against the current reader.",
        description=(
            "Run compat-validate to check that stored fixture files from "
            "all (or specific) versions can still be read correctly."
        ),
    )
    check_parser.add_argument(
        "--target",
        required=True,
        help=(
            'Where to read fixtures from. Use "s3" for the shared bucket, '
            'or "local:<path>" for a local directory.'
        ),
    )
    check_parser.add_argument(
        "--versions",
        help="Comma-separated list of versions to test (default: all discovered versions).",
    )

    args = parser.parse_args()
    if args.command == "add-version":
        cmd_add_version(args)
    elif args.command == "check":
        cmd_check(args)


if __name__ == "__main__":
    main()
