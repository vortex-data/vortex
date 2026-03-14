#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""
Vortex backward-compatibility orchestrator.

Manages fixture versions in S3 (or local directories) by calling the thin
`vortex-compat` Rust binary for generation and checking.  The Rust binary
handles only two things: generating .vortex files and comparing them.
Everything else (versioning, S3 upload/download, manifest merging, worktree
management) lives here.

Quick start:
    # Generate + publish for HEAD (version auto-detected from latest tag)
    python compat.py publish

    # Publish from an older tag
    python compat.py publish --git-ref v0.62.0

    # Check all published versions against current code
    python compat.py check
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlopen

DEFAULT_STORE = "s3://vortex-compat-fixtures"
CARGO_BIN = "vortex-compat"

EPILOG = """\
environment variables:
  VORTEX_COMPAT_BIN    Path to a pre-built vortex-compat binary.
                       Skips `cargo run` when set.

store spec:
  Local path           --store /tmp/compat-store
  S3 bucket            --store s3://my-bucket

  Default: s3://vortex-compat-fixtures
  S3 reads are public HTTPS; writes need AWS credentials (env or IAM role).

version detection:
  The version is always derived from a git tag. By default, HEAD's nearest
  tag is used (via `git describe --tags --abbrev=0`). Use --git-ref to
  target a different ref (e.g. v0.62.0). The 'v' prefix is stripped to
  produce the version string (v0.63.0 -> 0.63.0).

examples:
  # Publish from HEAD (version from latest tag)
  python compat.py publish
  python compat.py publish --dry-run

  # Publish from an older tag via worktree
  python compat.py publish --git-ref v0.62.0

  # Generate locally without publishing
  python compat.py generate --output /tmp/fixtures
  python compat.py generate --output /tmp/fixtures --git-ref v0.62.0

  # Check all versions, or specific ones
  python compat.py check
  python compat.py check --versions 0.62.0,0.63.0

  # Inspect store contents
  python compat.py list
  python compat.py list --version 0.62.0

  # Validate additive-only manifest property
  python compat.py validate-manifest
"""


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
    """Fixture store backed by a local directory."""

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
        dest = self.root / key
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local_path, dest)

    def list_versions(self) -> list[str]:
        versions_data = self.read("versions.json")
        if versions_data:
            return json.loads(versions_data)
        if not self.root.exists():
            return []
        # Fall back to directory listing.
        versions = []
        for entry in self.root.iterdir():
            if entry.is_dir() and entry.name.startswith("v"):
                manifest = entry / "manifest.json"
                if manifest.exists():
                    versions.append(entry.name[1:])  # strip 'v' prefix
        versions.sort(key=_version_sort_key)
        return versions

    def display_name(self) -> str:
        return str(self.root)


class S3Store(Store):
    """Fixture store backed by an S3 bucket (public reads, aws cli writes)."""

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


def _parse_store(spec: str) -> Store:
    """Parse a store specification into a Store instance."""
    if spec.startswith("s3://"):
        return S3Store(spec[5:])
    return LocalStore(Path(spec))


# ---------------------------------------------------------------------------
# Version detection
# ---------------------------------------------------------------------------


def _version_from_ref(git_ref: str | None = None) -> str:
    """Derive a version string from a git ref.

    If git_ref is None, uses HEAD. Finds the nearest tag and strips the 'v' prefix.
    For example, tag 'v0.63.0' yields version '0.63.0'.
    """
    cmd = ["git", "describe", "--tags", "--abbrev=0"]
    if git_ref:
        cmd.append(git_ref)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(
            f"error: could not detect version from git"
            + (f" ref '{git_ref}'" if git_ref else "")
            + f": {result.stderr.strip()}",
            file=sys.stderr,
        )
        sys.exit(1)
    tag = result.stdout.strip()
    # Strip 'v' prefix if present.
    version = re.sub(r"^v", "", tag)
    _info(f"detected version {version} (from tag {tag})")
    return version


# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------


def _read_manifest(store: Store, version: str) -> dict | None:
    data = store.read(f"v{version}/manifest.json")
    if data is None:
        return None
    return json.loads(data)


def _merge_manifest(
    store: Store, fixtures_json: dict, version: str, prev_version: str | None
) -> dict:
    """Build a manifest for `version`, carrying forward `since` from prev_version."""
    entries = []
    prev_since: dict[str, str] = {}

    if prev_version:
        prev_manifest = _read_manifest(store, prev_version)
        if prev_manifest:
            prev_since = {e["name"]: e["since"] for e in prev_manifest["fixtures"]}

    for f in fixtures_json["fixtures"]:
        name = f["name"]
        since = prev_since.get(name, version)
        entries.append({"name": name, "description": f["description"], "since": since})

    # Additive-only enforcement.
    current_names = {e["name"] for e in entries}
    missing = [n for n in prev_since if n not in current_names]
    if missing:
        print(
            f"error: fixtures removed since v{prev_version}: {', '.join(missing)}",
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
# Worktree helpers
# ---------------------------------------------------------------------------


def _worktree_generate(git_ref: str, output_dir: Path) -> None:
    """Create a git worktree at `git_ref`, build vortex-compat, and generate fixtures."""
    repo_root = Path(
        subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    )

    worktree_dir = Path(tempfile.mkdtemp(prefix=f"vortex-compat-wt-{git_ref}-"))
    try:
        _info(f"creating worktree at {git_ref} in {worktree_dir}")
        _run_cmd(
            ["git", "-C", str(repo_root), "worktree", "add", str(worktree_dir), git_ref],
            check=True,
        )

        _info(f"building vortex-compat at {git_ref}...")
        _run_cmd(
            ["cargo", "build", "-p", CARGO_BIN, "--release"],
            check=True,
            cwd=worktree_dir,
        )

        bin_path = worktree_dir / "target" / "release" / CARGO_BIN
        if not bin_path.exists():
            print(f"error: binary not found at {bin_path}", file=sys.stderr)
            sys.exit(1)

        _info(f"generating fixtures with {git_ref} binary...")
        _run_cmd([str(bin_path), "generate", "--output", str(output_dir)], check=True)

    finally:
        _run_cmd(
            ["git", "-C", str(repo_root), "worktree", "remove", "--force", str(worktree_dir)],
            check=False,
        )
        if worktree_dir.exists():
            shutil.rmtree(worktree_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_generate(args: argparse.Namespace) -> None:
    """Generate fixtures locally, then write a proper manifest."""
    output = Path(args.output)
    git_ref = args.git_ref
    version = _version_from_ref(git_ref)

    if git_ref:
        _worktree_generate(git_ref, output)
    else:
        _run_rust_generate(output)

    # Read fixtures.json and write a versioned manifest.
    fixtures_json = json.loads((output / "fixtures.json").read_text())
    manifest = {
        "version": version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "fixtures": [
            {"name": f["name"], "description": f["description"], "since": version}
            for f in fixtures_json["fixtures"]
        ],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    _info(f"wrote manifest.json for v{version}")


def cmd_publish(args: argparse.Namespace) -> None:
    """Generate fixtures and publish to a store."""
    store = _parse_store(args.store)
    git_ref = args.git_ref
    version = _version_from_ref(git_ref)

    with tempfile.TemporaryDirectory() as tmpdir:
        output = Path(tmpdir) / "fixtures"

        _info("generating fixtures...")
        if git_ref:
            _worktree_generate(git_ref, output)
        else:
            _run_rust_generate(output)

        fixtures_json = json.loads((output / "fixtures.json").read_text())

        versions = store.list_versions()
        prev = _find_prev_version(versions, version)
        if prev:
            _info(f"previous version: {prev}")

        manifest = _merge_manifest(store, fixtures_json, version, prev)
        manifest_json = json.dumps(manifest, indent=2) + "\n"

        if args.dry_run:
            _info("dry run — not uploading.")
            print(manifest_json)
            return

        _info(
            f"uploading {len(manifest['fixtures'])} fixtures to {store.display_name()}..."
        )
        for entry in manifest["fixtures"]:
            name = entry["name"]
            local = output / name
            key = f"v{version}/{name}"
            store.write_file(key, local)
            _info(f"  uploaded {name}")

        store.write(f"v{version}/manifest.json", manifest_json.encode())
        _info("  uploaded manifest.json")

        if version not in versions:
            versions.append(version)
            versions.sort(key=_version_sort_key)
        store.write("versions.json", (json.dumps(versions, indent=2) + "\n").encode())
        _info("  updated versions.json")

        _info(
            f"\ndone: {len(manifest['fixtures'])} fixtures for v{version} "
            f"published to {store.display_name()}"
        )


def cmd_check(args: argparse.Namespace) -> None:
    """Download fixtures from store and check with Rust binary."""
    store = _parse_store(args.store)

    if args.versions:
        versions = [v.strip() for v in args.versions.split(",")]
    else:
        versions = store.list_versions()

    if not versions:
        _info("no versions found")
        return

    _info(f"checking {len(versions)} version(s): {', '.join(versions)}")

    total_passed = 0
    total_failed = 0
    total_skipped = 0
    all_failures: list[tuple[str, str, str]] = []

    for version in versions:
        manifest = _read_manifest(store, version)
        if manifest is None:
            _info(f"  v{version}: no manifest found, skipping")
            continue

        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            for entry in manifest["fixtures"]:
                name = entry["name"]
                data = store.read(f"v{version}/{name}")
                if data is None:
                    _info(f"  v{version}: {name} not found in store")
                    continue
                (tmppath / name).write_bytes(data)

            result = _run_rust_check(tmppath, mode="subset")

            passed = len(result.get("passed", []))
            failed_list = result.get("failed", [])
            skipped = len(result.get("skipped", []))
            total_passed += passed
            total_failed += len(failed_list)
            total_skipped += skipped

            if failed_list:
                _info(
                    f"  v{version}: {passed} passed, {len(failed_list)} FAILED, "
                    f"{skipped} skipped"
                )
                for f in failed_list:
                    _info(f"    FAIL {f['name']}: {f['error']}")
                    all_failures.append((version, f["name"], f["error"]))
            else:
                _info(f"  v{version}: {passed} passed, {skipped} skipped")

    _info(
        f"\nresult: {total_passed} passed, {total_failed} failed, {total_skipped} skipped"
    )
    if all_failures:
        sys.exit(1)


def cmd_list(args: argparse.Namespace) -> None:
    """List versions or show a version's manifest."""
    store = _parse_store(args.store)

    if args.version:
        manifest = _read_manifest(store, args.version)
        if manifest is None:
            print(f"no manifest found for v{args.version}", file=sys.stderr)
            sys.exit(1)
        print(json.dumps(manifest, indent=2))
    else:
        versions = store.list_versions()
        if not versions:
            _info("(no versions)")
        for v in versions:
            print(v)


def cmd_validate_manifest(args: argparse.Namespace) -> None:
    """Check that manifests are additive-only across all versions."""
    store = _parse_store(args.store)
    versions = store.list_versions()

    if not versions:
        _info("no versions found")
        return

    _info(f"validating {len(versions)} version(s)...")

    prev_names: set[str] | None = None
    prev_version: str | None = None
    errors: list[str] = []

    for version in versions:
        manifest = _read_manifest(store, version)
        if manifest is None:
            _info(f"  v{version}: no manifest, skipping")
            continue
        names = {e["name"] for e in manifest["fixtures"]}

        if prev_names is not None:
            missing = prev_names - names
            if missing:
                msg = f"v{version} missing from v{prev_version}: {', '.join(sorted(missing))}"
                _info(f"  FAIL: {msg}")
                errors.append(msg)
            else:
                new = len(names) - len(prev_names)
                extra = f" (+{new} new)" if new > 0 else ""
                _info(
                    f"  v{prev_version} -> v{version}: ok ({len(names)} fixtures{extra})"
                )
        else:
            _info(f"  v{version}: {len(names)} fixtures (first)")

        prev_names = names
        prev_version = version

    if errors:
        _info(f"\n{len(errors)} error(s)")
        sys.exit(1)
    else:
        _info("\nall manifests are additive-only.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_rust_generate(output: Path) -> None:
    """Run `vortex-compat generate --output <dir>`."""
    cmd = _cargo_run_cmd() + ["generate", "--output", str(output)]
    _run_cmd(cmd, check=True)


def _run_rust_check(dir: Path, mode: str = "subset") -> dict:
    """Run `vortex-compat check --dir <dir> --mode <mode>` and parse JSON stdout."""
    cmd = _cargo_run_cmd() + ["check", "--dir", str(dir), "--mode", mode]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)

    if result.stdout.strip():
        return json.loads(result.stdout)

    if result.returncode != 0:
        return {
            "passed": [],
            "failed": [{"name": "(all)", "error": "check process failed"}],
            "skipped": [],
        }
    return {"passed": [], "failed": [], "skipped": []}


def _cargo_run_cmd() -> list[str]:
    """Build the command to invoke vortex-compat (pre-built binary or cargo run)."""
    bin_path = os.environ.get("VORTEX_COMPAT_BIN")
    if bin_path:
        return [bin_path]
    return ["cargo", "run", "-p", CARGO_BIN, "--release", "--"]


def _run_cmd(
    cmd: list[str], check: bool = False, cwd: Path | None = None
) -> subprocess.CompletedProcess:
    _info(f"  $ {' '.join(cmd)}")
    return subprocess.run(cmd, check=check, cwd=cwd)


def _find_prev_version(versions: list[str], current: str) -> str | None:
    """Find the highest version strictly less than `current`."""
    current_key = _version_sort_key(current)
    prev = None
    for v in versions:
        if _version_sort_key(v) < current_key:
            prev = v
    return prev


def _version_sort_key(v: str) -> list[int]:
    parts = []
    for p in v.split("."):
        try:
            parts.append(int(p))
        except ValueError:
            parts.append(0)
    return parts


def _info(msg: str) -> None:
    print(msg, file=sys.stderr)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="compat.py",
        description="Vortex backward-compatibility fixture orchestrator",
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    # -- generate --
    p = sub.add_parser(
        "generate",
        help="Generate fixtures locally",
        description=(
            "Build all fixture .vortex files and write them to a directory.\n"
            "Version is auto-detected from the nearest git tag at HEAD\n"
            "(or at --git-ref if specified)."
        ),
        epilog=(
            "examples:\n"
            "  python compat.py generate --output ./out\n"
            "  python compat.py generate --output ./out --git-ref v0.62.0"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--output", required=True, help="Output directory")
    p.add_argument(
        "--git-ref",
        help="Git ref (tag/branch/SHA) to build from via worktree. "
        "Version is derived from the nearest tag at this ref.",
    )

    # -- publish --
    p = sub.add_parser(
        "publish",
        help="Generate and publish fixtures to a store",
        description=(
            "Generate fixture files, merge the manifest with the previous version,\n"
            "and upload everything to the store. Version is auto-detected from the\n"
            "nearest git tag at HEAD (or at --git-ref)."
        ),
        epilog=(
            "examples:\n"
            "  python compat.py publish\n"
            "  python compat.py publish --dry-run\n"
            "  python compat.py publish --git-ref v0.62.0\n"
            "  python compat.py publish --store /tmp/store"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--store", default=DEFAULT_STORE, help="Store spec (default: %(default)s)"
    )
    p.add_argument(
        "--git-ref",
        help="Git ref to build from via worktree (e.g. v0.62.0). "
        "Version is derived from the nearest tag at this ref.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate and show manifest, but don't upload",
    )

    # -- check --
    p = sub.add_parser(
        "check",
        help="Validate fixtures from a store against current code",
        description=(
            "Download fixtures for each version from the store, then use the\n"
            "current vortex-compat binary to verify they can still be read and\n"
            "match expectations."
        ),
        epilog=(
            "examples:\n"
            "  python compat.py check\n"
            "  python compat.py check --versions 0.62.0,0.63.0\n"
            "  python compat.py check --store /tmp/store"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--store", default=DEFAULT_STORE, help="Store spec (default: %(default)s)"
    )
    p.add_argument(
        "--versions",
        help="Comma-separated versions to check (default: all)",
    )

    # -- list --
    p = sub.add_parser(
        "list",
        help="List versions or show a version's manifest",
        description="Inspect the contents of a fixture store.",
        epilog=(
            "examples:\n"
            "  python compat.py list\n"
            "  python compat.py list --version 0.62.0\n"
            "  python compat.py list --store /tmp/store"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--store", default=DEFAULT_STORE, help="Store spec (default: %(default)s)"
    )
    p.add_argument("--version", help="Show manifest for this version")

    # -- validate-manifest --
    p = sub.add_parser(
        "validate-manifest",
        help="Check additive-only property across all versions",
        description=(
            "Verify that no fixtures were removed between consecutive versions.\n"
            "New fixtures are allowed; removals are errors."
        ),
        epilog=(
            "examples:\n"
            "  python compat.py validate-manifest\n"
            "  python compat.py validate-manifest --store /tmp/store"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--store", default=DEFAULT_STORE, help="Store spec (default: %(default)s)"
    )

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
