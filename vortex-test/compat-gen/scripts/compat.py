#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""
Vortex backward-compatibility orchestrator.

Manages fixture versions in S3 (or local directories) by calling the thin
`vortex-compat` Rust binary for generation and checking.

Usage:
    python compat.py publish  --version 0.63.0 [--store s3://bucket] [--dry-run]
    python compat.py check    [--versions 0.62.0,0.63.0] [--store s3://bucket]
    python compat.py generate --version 0.63.0 --output ./my-fixtures
    python compat.py list     [--store s3://bucket] [--version 0.63.0]
    python compat.py validate-manifest [--store s3://bucket]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlopen

DEFAULT_STORE = "s3://vortex-compat-fixtures"
CARGO_BIN = "vortex-compat"


# ---------------------------------------------------------------------------
# Store abstraction
# ---------------------------------------------------------------------------


class Store:
    """Abstract base for fixture stores."""

    def read(self, key: str) -> bytes | None:
        raise NotImplementedError

    def write(self, key: str, data: bytes) -> None:
        raise NotImplementedError

    def write_file(self, key: str, local_path: Path) -> None:
        raise NotImplementedError

    def list_versions(self) -> list[str]:
        raise NotImplementedError

    def display_name(self) -> str:
        raise NotImplementedError


class LocalStore(Store):
    def __init__(self, root: Path):
        self.root = root

    def read(self, key: str) -> bytes | None:
        path = self.root / key
        if not path.exists():
            return None
        return path.read_bytes()

    def write(self, key: str, data: bytes) -> None:
        path = self.root / key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def write_file(self, key: str, local_path: Path) -> None:
        import shutil

        dest = self.root / key
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local_path, dest)

    def list_versions(self) -> list[str]:
        versions_data = self.read("versions.json")
        if versions_data:
            return json.loads(versions_data)
        if not self.root.exists():
            return []
        versions = []
        for entry in self.root.iterdir():
            if entry.is_dir() and entry.name.startswith("v"):
                manifest = entry / "manifest.json"
                if manifest.exists():
                    versions.append(entry.name[1:])  # strip 'v' prefix
        versions.sort(key=version_sort_key)
        return versions

    def display_name(self) -> str:
        return str(self.root)


class S3Store(Store):
    def __init__(self, bucket: str):
        self.bucket = bucket
        self.https_base = f"https://{bucket}.s3.amazonaws.com"

    def read(self, key: str) -> bytes | None:
        url = f"{self.https_base}/{key}"
        try:
            with urlopen(url) as resp:
                return resp.read()
        except HTTPError as e:
            if e.code in (403, 404):
                return None
            raise

    def write(self, key: str, data: bytes) -> None:
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(data)
            tmp_path = f.name
        try:
            self.write_file(key, Path(tmp_path))
        finally:
            os.unlink(tmp_path)

    def write_file(self, key: str, local_path: Path) -> None:
        _run_cmd(
            ["aws", "s3", "cp", str(local_path), f"s3://{self.bucket}/{key}"],
            check=True,
        )

    def list_versions(self) -> list[str]:
        data = self.read("versions.json")
        if data:
            return json.loads(data)
        return []

    def display_name(self) -> str:
        return f"s3://{self.bucket}"


def parse_store(spec: str) -> Store:
    if spec.startswith("s3://"):
        return S3Store(spec[5:])
    return LocalStore(Path(spec))


# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------


def read_manifest(store: Store, version: str) -> dict | None:
    data = store.read(f"v{version}/manifest.json")
    if data is None:
        return None
    return json.loads(data)


def merge_manifest(
    store: Store, fixtures_json: dict, version: str, prev_version: str | None
) -> dict:
    """Build a manifest for `version`, merging `since` from prev_version."""
    entries = []
    prev_since: dict[str, str] = {}

    if prev_version:
        prev_manifest = read_manifest(store, prev_version)
        if prev_manifest:
            prev_since = {e["name"]: e["since"] for e in prev_manifest["fixtures"]}

    for f in fixtures_json["fixtures"]:
        name = f["name"]
        since = prev_since.get(name, version)
        entries.append({"name": name, "description": f["description"], "since": since})

    # Additive-only check.
    current_names = {e["name"] for e in entries}
    missing = [n for n in prev_since if n not in current_names]
    if missing:
        print(
            f"ERROR: fixtures removed since v{prev_version}: {', '.join(missing)}",
            file=sys.stderr,
        )
        print("Fixtures must never be removed.", file=sys.stderr)
        sys.exit(1)

    return {
        "version": version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "fixtures": entries,
    }


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_generate(args: argparse.Namespace) -> None:
    """Generate fixtures locally, then write a proper manifest."""
    output = Path(args.output)
    run_rust_generate(output)

    # Read fixtures.json and write a versioned manifest.
    fixtures_json = json.loads((output / "fixtures.json").read_text())
    manifest = {
        "version": args.version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "fixtures": [
            {"name": f["name"], "description": f["description"], "since": args.version}
            for f in fixtures_json["fixtures"]
        ],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote manifest.json for v{args.version}", file=sys.stderr)


def cmd_publish(args: argparse.Namespace) -> None:
    """Generate fixtures and publish to a store."""
    store = parse_store(args.store)
    version = args.version

    with tempfile.TemporaryDirectory() as tmpdir:
        output = Path(tmpdir) / "fixtures"

        # Step 1: Generate.
        print(f"generating fixtures...", file=sys.stderr)
        run_rust_generate(output)

        # Step 2: Read fixtures.json.
        fixtures_json = json.loads((output / "fixtures.json").read_text())

        # Step 3: Find previous version and merge manifest.
        versions = store.list_versions()
        prev = None
        for v in versions:
            if v != version:
                prev = v
        if prev:
            print(f"previous version: {prev}", file=sys.stderr)

        manifest = merge_manifest(store, fixtures_json, version, prev)
        manifest_json = json.dumps(manifest, indent=2) + "\n"

        if args.dry_run:
            print("dry run — not uploading.", file=sys.stderr)
            print(manifest_json)
            return

        # Step 4: Upload fixture files.
        print(
            f"uploading {len(manifest['fixtures'])} fixtures to {store.display_name()}...",
            file=sys.stderr,
        )
        for entry in manifest["fixtures"]:
            name = entry["name"]
            local = output / name
            key = f"v{version}/{name}"
            store.write_file(key, local)
            print(f"  uploaded {name}", file=sys.stderr)

        # Step 5: Upload manifest.
        store.write(f"v{version}/manifest.json", manifest_json.encode())
        print("  uploaded manifest.json", file=sys.stderr)

        # Step 6: Update versions.json.
        if version not in versions:
            versions.append(version)
            versions.sort(key=version_sort_key)
        store.write("versions.json", (json.dumps(versions, indent=2) + "\n").encode())
        print("  updated versions.json", file=sys.stderr)

        print(
            f"\ndone: {len(manifest['fixtures'])} fixtures for v{version} "
            f"published to {store.display_name()}",
            file=sys.stderr,
        )


def cmd_check(args: argparse.Namespace) -> None:
    """Download fixtures from store and check with Rust binary."""
    store = parse_store(args.store)

    if args.versions:
        versions = [v.strip() for v in args.versions.split(",")]
    else:
        versions = store.list_versions()

    if not versions:
        print("no versions found", file=sys.stderr)
        return

    print(f"checking {len(versions)} version(s): {', '.join(versions)}", file=sys.stderr)

    total_passed = 0
    total_failed = 0
    total_skipped = 0
    all_failures: list[tuple[str, str, str]] = []

    for version in versions:
        manifest = read_manifest(store, version)
        if manifest is None:
            print(f"  v{version}: no manifest found, skipping", file=sys.stderr)
            continue

        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            # Download all fixture files.
            for entry in manifest["fixtures"]:
                name = entry["name"]
                data = store.read(f"v{version}/{name}")
                if data is None:
                    print(f"  v{version}: {name} not found in store", file=sys.stderr)
                    continue
                (tmppath / name).write_bytes(data)

            # Run Rust checker.
            result = run_rust_check(tmppath, mode="subset")

            passed = len(result.get("passed", []))
            failed_list = result.get("failed", [])
            skipped = len(result.get("skipped", []))
            total_passed += passed
            total_failed += len(failed_list)
            total_skipped += skipped

            if failed_list:
                print(
                    f"  v{version}: {passed} passed, {len(failed_list)} FAILED, "
                    f"{skipped} skipped",
                    file=sys.stderr,
                )
                for f in failed_list:
                    print(f"    FAIL {f['name']}: {f['error']}", file=sys.stderr)
                    all_failures.append((version, f["name"], f["error"]))
            else:
                print(
                    f"  v{version}: {passed} passed, {skipped} skipped",
                    file=sys.stderr,
                )

    print(
        f"\nresult: {total_passed} passed, {total_failed} failed, {total_skipped} skipped",
        file=sys.stderr,
    )
    if all_failures:
        sys.exit(1)


def cmd_list(args: argparse.Namespace) -> None:
    """List versions or show a version's manifest."""
    store = parse_store(args.store)

    if args.version:
        manifest = read_manifest(store, args.version)
        if manifest is None:
            print(f"no manifest found for v{args.version}", file=sys.stderr)
            sys.exit(1)
        print(json.dumps(manifest, indent=2))
    else:
        versions = store.list_versions()
        if not versions:
            print("(no versions)", file=sys.stderr)
        for v in versions:
            print(v)


def cmd_validate_manifest(args: argparse.Namespace) -> None:
    """Check that manifests are additive-only across all versions."""
    store = parse_store(args.store)
    versions = store.list_versions()

    if not versions:
        print("no versions found", file=sys.stderr)
        return

    print(f"validating {len(versions)} version(s)...", file=sys.stderr)

    prev_names: set[str] | None = None
    prev_version: str | None = None
    errors: list[str] = []

    for version in versions:
        manifest = read_manifest(store, version)
        if manifest is None:
            print(f"  v{version}: no manifest, skipping", file=sys.stderr)
            continue
        names = {e["name"] for e in manifest["fixtures"]}

        if prev_names is not None:
            missing = prev_names - names
            if missing:
                msg = f"v{version} missing from v{prev_version}: {', '.join(sorted(missing))}"
                print(f"  FAIL: {msg}", file=sys.stderr)
                errors.append(msg)
            else:
                new = len(names) - len(prev_names)
                extra = f" (+{new} new)" if new > 0 else ""
                print(
                    f"  v{prev_version} -> v{version}: ok ({len(names)} fixtures{extra})",
                    file=sys.stderr,
                )
        else:
            print(f"  v{version}: {len(names)} fixtures (first)", file=sys.stderr)

        prev_names = names
        prev_version = version

    if errors:
        print(f"\n{len(errors)} error(s)", file=sys.stderr)
        sys.exit(1)
    else:
        print("\nall manifests are additive-only.", file=sys.stderr)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def run_rust_generate(output: Path) -> None:
    """Run `vortex-compat generate --output <dir>`."""
    cmd = _cargo_run_cmd() + ["generate", "--output", str(output)]
    _run_cmd(cmd, check=True)


def run_rust_check(dir: Path, mode: str = "subset") -> dict:
    """Run `vortex-compat check --dir <dir> --mode <mode>` and parse JSON."""
    cmd = _cargo_run_cmd() + ["check", "--dir", str(dir), "--mode", mode]
    result = subprocess.run(cmd, capture_output=True, text=True)
    # stdout has JSON, stderr has progress.
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)

    if result.stdout.strip():
        return json.loads(result.stdout)

    if result.returncode != 0:
        return {"passed": [], "failed": [{"name": "(all)", "error": "check process failed"}], "skipped": []}
    return {"passed": [], "failed": [], "skipped": []}


def _cargo_run_cmd() -> list[str]:
    """Build the cargo run command for vortex-compat."""
    bin_path = os.environ.get("VORTEX_COMPAT_BIN")
    if bin_path:
        return [bin_path]
    return ["cargo", "run", "-p", CARGO_BIN, "--release", "--"]


def _run_cmd(cmd: list[str], check: bool = False) -> subprocess.CompletedProcess:
    print(f"  $ {' '.join(cmd)}", file=sys.stderr)
    return subprocess.run(cmd, check=check)


def version_sort_key(v: str) -> list[int]:
    parts = []
    for p in v.split("."):
        try:
            parts.append(int(p))
        except ValueError:
            parts.append(0)
    return parts


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="compat.py",
        description="Vortex backward-compatibility fixture orchestrator",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # generate
    p = sub.add_parser("generate", help="Generate fixtures locally")
    p.add_argument("--version", required=True, help="Version tag (e.g. 0.63.0)")
    p.add_argument("--output", required=True, help="Output directory")

    # publish
    p = sub.add_parser("publish", help="Generate and publish fixtures to a store")
    p.add_argument("--version", required=True, help="Version tag (e.g. 0.63.0)")
    p.add_argument("--store", default=DEFAULT_STORE, help="Store spec (local path or s3://bucket)")
    p.add_argument("--dry-run", action="store_true", help="Generate but don't upload")

    # check
    p = sub.add_parser("check", help="Validate fixtures from a store")
    p.add_argument("--store", default=DEFAULT_STORE, help="Store spec")
    p.add_argument("--versions", help="Comma-separated versions (default: all)")

    # list
    p = sub.add_parser("list", help="List versions or show a manifest")
    p.add_argument("--store", default=DEFAULT_STORE, help="Store spec")
    p.add_argument("--version", help="Show manifest for this version")

    # validate-manifest
    p = sub.add_parser("validate-manifest", help="Check additive-only property")
    p.add_argument("--store", default=DEFAULT_STORE, help="Store spec")

    args = parser.parse_args()

    commands = {
        "generate": cmd_generate,
        "publish": cmd_publish,
        "check": cmd_check,
        "list": cmd_list,
        "validate-manifest": cmd_validate_manifest,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
