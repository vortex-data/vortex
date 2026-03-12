#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Upload Vortex backward-compat fixtures to S3.

Wraps the full upload lifecycle:
  1. Build + run compat-gen to produce fixture files and a naive manifest
  2. Fetch the previous version's manifest from S3 (via public HTTP)
  3. Merge `since` values: keep old `since` for existing fixtures, current
     version for new ones
  4. Enforce additive-only: every fixture in the previous manifest must exist
     in the generated output
  5. Upload the output directory to S3
  6. Update versions.json with ETag-based optimistic locking

Requires only Python 3 stdlib + `aws` CLI on PATH.
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
    """Fetch *url* over HTTPS.  Returns None on 404, raises on other errors."""
    try:
        with urllib.request.urlopen(url) as resp:
            return resp.read()
    except urllib.error.HTTPError as exc:
        if exc.code == 404 or exc.code == 403:
            return None
        raise


def version_sort_key(v: str) -> list[int]:
    return list(map(int, v.split(".")))


# ---------------------------------------------------------------------------
# S3 helpers (reuse head_etag / put_object pattern from scripts/s3-upload.py)
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

        delay = min(2 ** attempt, 30)
        log(f"  versions.json upload failed (attempt {attempt}/{max_retries}), "
            f"retrying in {delay}s...")
        time.sleep(delay)

    log(f"ERROR: versions.json upload failed after {max_retries} attempts")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------


def fetch_versions() -> list[str]:
    """Fetch the current versions.json from S3 (public HTTP)."""
    data = http_get(f"{FIXTURES_URL}/versions.json")
    if data is None:
        return []
    return json.loads(data)


def fetch_previous_manifest(versions: list[str], current_version: str) -> dict | None:
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


def normalize_manifest_fixtures(manifest: dict) -> list[dict]:
    """Handle old manifest format where fixtures was a list of strings."""
    entries = manifest.get("fixtures", [])
    normalized = []
    for entry in entries:
        if isinstance(entry, str):
            # Old format: just a filename string — no `since` info
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
        # First upload — nothing to merge.
        return

    prev_fixtures = normalize_manifest_fixtures(previous_manifest)
    prev_by_name = {e["name"]: e for e in prev_fixtures}
    gen_by_name = {e["name"]: e for e in generated["fixtures"]}

    # Additive-only check: every previous fixture must still exist.
    missing = sorted(set(prev_by_name) - set(gen_by_name))
    if missing:
        log(f"ERROR: fixtures removed since previous version: {missing}")
        log("Fixtures must never be removed — only added.")
        sys.exit(1)

    # Merge: keep old `since` for existing fixtures, current version for new.
    for entry in generated["fixtures"]:
        name = entry["name"]
        if name in prev_by_name:
            entry["since"] = prev_by_name[name]["since"]
        else:
            entry["since"] = current_version

    with open(generated_manifest_path, "w") as f:
        json.dump(generated, f, indent=2)
        f.write("\n")

    log(f"  merged manifest: {len(prev_by_name)} existing, "
        f"{len(gen_by_name) - len(prev_by_name)} new fixtures")


def build_fixtures(version: str, output_dir: str) -> None:
    """Run cargo to build and execute compat-gen."""
    run([
        "cargo", "run", "-p", "vortex-compat", "--release", "--bin", "compat-gen",
        "--", "--version", version, "--output", output_dir,
    ])


def upload_fixtures(version: str, output_dir: str) -> None:
    """Upload the output directory to S3."""
    run([
        "aws", "s3", "cp", output_dir,
        f"s3://{S3_BUCKET}/v{version}/",
        "--recursive",
    ])


def update_versions(version: str, tmp_dir: str) -> None:
    """Append version to versions.json and upload with optimistic locking."""
    versions = fetch_versions()

    if version not in versions:
        versions.append(version)
        versions.sort(key=version_sort_key)

    local_path = os.path.join(tmp_dir, "versions.json")
    with open(local_path, "w") as f:
        json.dump(versions, f, indent=2)
        f.write("\n")

    upload_versions_json(local_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build, generate, and upload Vortex backward-compat fixtures.",
    )
    parser.add_argument(
        "--version", required=True,
        help='Version tag for this fixture set (e.g. "0.62.0").',
    )
    parser.add_argument(
        "--output",
        help="Output directory for generated fixtures (default: temp dir).",
    )
    parser.add_argument(
        "--skip-build", action="store_true",
        help="Skip cargo build + compat-gen run (assumes --output already populated).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Generate and merge manifest but skip S3 upload.",
    )
    args = parser.parse_args()

    # Resolve output directory.
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
            log(f"[1/4] Generating fixtures for v{args.version}...")
            build_fixtures(args.version, output_dir)
        else:
            log(f"[1/4] Skipping build (--skip-build), using {output_dir}")

        # Step 2: Fetch previous manifest and merge `since` values.
        log("[2/4] Fetching previous manifest...")
        versions = fetch_versions()
        prev_manifest = fetch_previous_manifest(versions, args.version)
        manifest_path = os.path.join(output_dir, "manifest.json")
        merge_manifest(manifest_path, prev_manifest, args.version)

        if args.dry_run:
            log("[3/4] Dry run — skipping S3 upload.")
            log("[4/4] Dry run — skipping versions.json update.")
            log(f"\nGenerated fixtures in: {output_dir}")
            with open(manifest_path) as f:
                log(f"Manifest:\n{f.read()}")
            return

        # Step 3: Upload fixtures to S3.
        log(f"[3/4] Uploading fixtures to s3://{S3_BUCKET}/v{args.version}/...")
        upload_fixtures(args.version, output_dir)

        # Step 4: Update versions.json.
        log("[4/4] Updating versions.json...")
        # Use the parent of output_dir for the temp versions.json file.
        tmp_dir = os.path.dirname(output_dir) if owns_tmp else tempfile.mkdtemp()
        update_versions(args.version, tmp_dir)

        log(f"\nDone: fixtures for v{args.version} uploaded.")
    finally:
        # Clean up temp dir if we created one.
        if owns_tmp and not args.dry_run:
            import shutil
            shutil.rmtree(os.path.dirname(output_dir), ignore_errors=True)


if __name__ == "__main__":
    main()
