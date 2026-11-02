#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors
# /// script
# dependencies = ["jsonschema"]
# ///

"""
Vortex backward- and forward-compatibility orchestrator.

Manages fixture versions in S3 (or local directories) by calling the thin
`vortex-compat` Rust binary for generation and checking.  The Rust binary
handles only two things: generating .vortex files and comparing them.
Everything else (versioning, S3 upload/download, manifest merging) lives here.

Quick start:
    # Generate + publish for HEAD (version auto-detected from latest tag)
    uv run compat.py publish

    # Publish from an older tag
    uv run compat.py publish --git-ref v0.62.0

    # Check all published versions against current code
    uv run compat.py check

    # Check that released readers can read files the current writer produces
    uv run compat.py forward
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

import jsonschema

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
  uv run compat.py publish
  uv run compat.py publish --dry-run

  # Publish using an older tag for version detection
  uv run compat.py publish --git-ref v0.62.0

  # Add new fixtures to an existing version (hash-verified)
  uv run compat.py publish --update
  uv run compat.py publish --update --dry-run

  # Generate locally without publishing
  uv run compat.py generate --output /tmp/fixtures
  uv run compat.py generate --output /tmp/fixtures --git-ref v0.62.0

  # Check all versions, or specific ones
  uv run compat.py check
  uv run compat.py check --mode last
  uv run compat.py check --versions 0.62.0,0.63.0

  # Read files written by the current writer with old releases, per edition
  uv run compat.py forward
  uv run compat.py forward --mode last

  # Inspect store contents
  uv run compat.py list
  uv run compat.py list 0.62.0

  # Validate additive-only manifest property
  uv run compat.py validate-manifest
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

    def __init__(self, root: Path) -> None:
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
                manifest = entry / "arrays" / "manifest.json"
                if manifest.exists():
                    versions.append(entry.name[1:])  # strip 'v' prefix
        versions.sort(key=_version_sort_key)
        return versions

    def display_name(self) -> str:
        return str(self.root)


class S3Store(Store):
    """Fixture store backed by an S3 bucket (public reads, aws cli writes)."""

    def __init__(self, bucket: str) -> None:
        self.bucket = bucket
        self.https_base = f"https://{bucket}.s3.amazonaws.com"

    def read(self, key: str) -> bytes | None:
        url = f"{self.https_base}/{key}"
        for attempt in range(3):
            try:
                with urlopen(url, timeout=10) as resp:
                    return resp.read()
            except HTTPError as e:
                if e.code in (403, 404):
                    return None
                raise
            except (URLError, ConnectionError, TimeoutError):
                if attempt < 2:
                    time.sleep(1 * (attempt + 1))
                    continue
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
        dest = f"s3://{self.bucket}/{key}"
        _info(f"  {local_path.name} -> {dest}")
        subprocess.run(
            ["aws", "s3", "cp", str(local_path), dest],
            check=True,
        )

    def list_versions(self) -> list[str]:
        data = self.read("versions.json")
        if data:
            return json.loads(data)
        return []

    def display_name(self) -> str:
        return f"s3://{self.bucket}"


class DryRunStore(Store):
    """Wrapper that validates write credentials without actually uploading.

    Reads are delegated to the inner store. Writes use ``aws s3 cp --dryrun``
    (for S3 stores) so that IAM permissions are verified, but no data is
    written.  For local stores, writes are silently skipped.
    """

    def __init__(self, inner: Store) -> None:
        self.inner = inner

    def read(self, key: str) -> bytes | None:
        return self.inner.read(key)

    def write(self, key: str, data: bytes) -> None:
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(data)
            tmp_path = f.name
        try:
            self.write_file(key, Path(tmp_path))
        finally:
            os.unlink(tmp_path)

    def write_file(self, key: str, local_path: Path) -> None:
        if isinstance(self.inner, S3Store):
            dest = f"s3://{self.inner.bucket}/{key}"
            _info(f"  {local_path.name} -> {dest} (dry-run)")
            subprocess.run(
                ["aws", "s3", "cp", str(local_path), dest, "--dryrun"],
                check=True,
            )
        else:
            _info(f"  dry-run: would write {key}")

    def list_versions(self) -> list[str]:
        return self.inner.list_versions()

    def display_name(self) -> str:
        return self.inner.display_name()


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
        ref_msg = f" ref '{git_ref}'" if git_ref else ""
        print(
            f"error: could not detect version from git{ref_msg}: {result.stderr.strip()}",
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


MANIFEST_SCHEMA = {
    "type": "object",
    "required": ["version", "generated_at", "fixtures"],
    "properties": {
        "version": {"type": "string"},
        "generated_at": {"type": "string"},
        "fixtures": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["name"],
                "properties": {
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                    "sha256": {"type": "string"},
                },
            },
        },
    },
}


def _validate_manifest(manifest: dict, version: str) -> None:
    """Validate manifest against the JSON schema."""
    try:
        jsonschema.validate(manifest, MANIFEST_SCHEMA)
    except jsonschema.ValidationError as e:
        raise ValueError(
            f"v{version} manifest: {e.message} (at path: {'/'.join(str(p) for p in e.absolute_path)})"
        ) from e


def _read_manifest(store: Store, version: str) -> dict | None:
    # Try new path first, then legacy path.
    data = store.read(f"v{version}/arrays/manifest.json")
    prefix = f"v{version}/arrays"
    if data is None:
        data = store.read(f"v{version}/manifest.json")
        prefix = f"v{version}"
    if data is None:
        return None
    manifest = json.loads(data)

    # Upgrade legacy format: flat list of filenames -> new object format.
    if isinstance(manifest.get("fixtures"), list) and manifest["fixtures"] and isinstance(manifest["fixtures"][0], str):
        _info(f"  upgrading legacy manifest format for v{version}")
        manifest["fixtures"] = [{"name": n, "description": "", "sha256": ""} for n in manifest["fixtures"]]

    _validate_manifest(manifest, version)
    # Stash the prefix so callers know where to fetch fixture files.
    manifest["_prefix"] = prefix
    return manifest


def _merge_manifest(
    store: Store,
    fixtures_json: dict,
    version: str,
    prev_version: str | None,
) -> dict:
    """Build a manifest for `version`, using sha256 from Rust-generated fixtures.json."""
    entries = []
    prev_names: set[str] = set()

    if prev_version:
        prev_manifest = _read_manifest(store, prev_version)
        if prev_manifest:
            prev_names = {e["name"] for e in prev_manifest["fixtures"]}

    for f in fixtures_json["fixtures"]:
        entries.append({"name": f["name"], "description": f["description"], "sha256": f["sha256"]})

    # Additive-only enforcement.
    current_names = {e["name"] for e in entries}
    missing = [n for n in prev_names if n not in current_names]
    if missing:
        print(
            f"error: fixtures removed since v{prev_version}: {', '.join(missing)}",
            file=sys.stderr,
        )
        print("Fixtures must never be removed.", file=sys.stderr)
        sys.exit(1)

    return {
        "version": version,
        "generated_at": datetime.now(UTC).isoformat(),
        "fixtures": entries,
    }


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_generate(args: argparse.Namespace) -> None:
    """Generate fixtures locally, then write a proper manifest."""
    output = Path(args.output)
    version = _version_from_ref(args.git_ref)

    _run_rust_generate(output, profile=args.profile)

    # Read fixtures.json (with sha256 from Rust) and write a versioned manifest.
    fixtures_json = json.loads((output / "fixtures.json").read_text())
    entries = []
    for f in fixtures_json["fixtures"]:
        entries.append({"name": f["name"], "description": f["description"], "sha256": f["sha256"]})
    manifest = {
        "version": version,
        "generated_at": datetime.now(UTC).isoformat(),
        "fixtures": entries,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    _info(f"wrote manifest.json for v{version}")


def cmd_publish(args: argparse.Namespace) -> None:
    """Generate fixtures and publish to a store."""
    store = _parse_store(args.store)
    git_ref = args.git_ref
    version = _version_from_ref(git_ref)

    versions = store.list_versions()

    if args.update:
        _publish_update(store, version, versions, args)
    else:
        _publish_full(store, version, versions, args)


def _publish_full(
    store: Store,
    version: str,
    versions: list[str],
    args: argparse.Namespace,
) -> None:
    """Full publish: upload all fixtures for a new version."""
    if version in versions and not args.force:
        _info(f"error: v{version} already exists in {store.display_name()}")
        _info("use --force to overwrite an existing version")
        sys.exit(1)

    with tempfile.TemporaryDirectory() as tmpdir:
        output = Path(tmpdir) / "fixtures"

        _info("generating fixtures...")
        _run_rust_generate(output, profile=args.profile)

        fixtures_json = json.loads((output / "fixtures.json").read_text())

        prev = _find_prev_version(versions, version)
        if prev:
            _info(f"previous version: {prev}")

        manifest = _merge_manifest(store, fixtures_json, version, prev)
        manifest_json = json.dumps(manifest, indent=2) + "\n"

        write_store = _write_store(
            store,
            args,
            f"\nabout to upload {len(manifest['fixtures'])} fixtures for v{version} to {store.display_name()}",
        )

        _parallel_upload(
            write_store,
            [(f"v{version}/arrays/{e['name']}", output / e["name"]) for e in manifest["fixtures"]],
        )

        write_store.write(f"v{version}/arrays/manifest.json", manifest_json.encode())

        if version not in versions:
            versions.append(version)
            versions.sort(key=_version_sort_key)
        write_store.write("versions.json", (json.dumps(versions, indent=2) + "\n").encode())


def _publish_update(
    store: Store,
    version: str,
    versions: list[str],
    args: argparse.Namespace,
) -> None:
    """Incremental update: add new fixtures to an existing version (hash-verified)."""
    if version not in versions:
        _info(f"error: v{version} not found in {store.display_name()}, use publish without --update for new versions")
        sys.exit(1)

    existing_manifest = _read_manifest(store, version)
    if existing_manifest is None:
        _info(f"error: v{version} has no manifest in {store.display_name()}")
        sys.exit(1)

    prefix = existing_manifest.pop("_prefix")

    with tempfile.TemporaryDirectory() as tmpdir:
        output = Path(tmpdir) / "fixtures"

        _info("generating fixtures...")
        _run_rust_generate(output, profile=args.profile)

        fixtures_json = json.loads((output / "fixtures.json").read_text())

        # Compare each generated fixture against the store.
        new_fixtures: list[str] = []
        for f in fixtures_json["fixtures"]:
            name = f["name"]
            local_path = output / name
            local_hash = hashlib.sha256(local_path.read_bytes()).hexdigest()
            remote_data = store.read(f"{prefix}/{name}")

            if remote_data is not None:
                remote_hash = hashlib.sha256(remote_data).hexdigest()
                if local_hash != remote_hash:
                    _info(f"error: hash mismatch for {name}: local={local_hash[:12]} remote={remote_hash[:12]}")
                    sys.exit(1)
                else:
                    _info(f"  {name}: unchanged (sha256 match)")
            else:
                new_fixtures.append(name)
                _info(f"  {name}: NEW")

        if not new_fixtures:
            _info("no new fixtures to add")
            return

        write_store = _write_store(
            store,
            args,
            f"\nabout to upload {len(new_fixtures)} new fixture(s) for v{version} to {store.display_name()}",
        )

        new_fixture_names = set(new_fixtures)
        _parallel_upload(
            write_store,
            [(f"{prefix}/{name}", output / name) for name in new_fixtures],
        )

        # Merge manifest: keep existing entries, add new ones.
        new_entries = existing_manifest["fixtures"][:]
        for f in fixtures_json["fixtures"]:
            if f["name"] in new_fixture_names:
                new_entries.append({"name": f["name"], "description": f["description"], "sha256": f["sha256"]})

        updated_manifest = {
            "version": version,
            "generated_at": datetime.now(UTC).isoformat(),
            "fixtures": new_entries,
        }
        write_store.write(f"{prefix}/manifest.json", (json.dumps(updated_manifest, indent=2) + "\n").encode())


def cmd_check(args: argparse.Namespace) -> None:
    """Download fixtures from store and check with Rust binary."""
    store = _parse_store(args.store)

    if args.versions and args.mode != "all":
        print("error: --versions and --mode are mutually exclusive", file=sys.stderr)
        sys.exit(1)

    if args.versions:
        versions = [v.strip() for v in args.versions.split(",")]
    else:
        versions = store.list_versions()
        if args.mode == "last" and versions:
            versions = versions[-1:]

    if not versions:
        _info("no versions found in store")
        sys.exit(1)

    _info(f"found versions.json at {store.display_name()}/versions.json: {versions}")
    _info(f"checking {len(versions)} version(s): {', '.join(versions)}")

    total_passed = 0
    total_failed = 0
    total_skipped = 0
    all_failures: list[tuple[str, str, str]] = []

    for version in versions:
        _info(f"\n--- v{version} ---")
        manifest = _read_manifest(store, version)
        if manifest is None:
            _info(f"  v{version}: no manifest found at v{version}/arrays/manifest.json or v{version}/manifest.json")
            all_failures.append((version, "(manifest)", "manifest not found"))
            total_failed += 1
            continue

        prefix = manifest.pop("_prefix", f"v{version}/arrays")
        _info(f"  manifest: {prefix}/manifest.json ({len(manifest['fixtures'])} fixtures)")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            _info(f"  downloading {len(manifest['fixtures'])} fixtures...")
            download_failures = _parallel_download(store, manifest["fixtures"], prefix, tmppath)
            for name, error in download_failures:
                _info(f"  v{version}: {name} {error}")
                all_failures.append((version, name, error))
                total_failed += 1

            _info(f"  checking v{version}...")
            result = _run_rust_check(tmppath, mode="superset", profile=args.profile)

            passed = len(result.get("passed", []))
            failed_list = result.get("failed", [])
            skipped = len(result.get("skipped", []))
            total_passed += passed
            total_failed += len(failed_list)
            total_skipped += skipped

            if failed_list:
                _info(f"  v{version}: {passed} passed, {len(failed_list)} FAILED, {skipped} skipped")
                for f in failed_list:
                    _info(f"    FAIL {f['name']}: {f['error']}")
                    all_failures.append((version, f["name"], f["error"]))
            else:
                _info(f"  v{version}: {passed} passed, {skipped} skipped")

    _info(f"\nresult: {total_passed} passed, {total_failed} failed, {total_skipped} skipped")
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


def cmd_verify(args: argparse.Namespace) -> None:
    """Verify fixture file integrity against manifest sha256 hashes."""
    store = _parse_store(args.store)
    versions = store.list_versions()

    if not versions:
        _info("no versions found")
        sys.exit(1)

    _info(f"verifying {len(versions)} version(s) in {store.display_name()}...")

    total_ok = 0
    errors: list[str] = []

    for version in versions:
        _info(f"\n--- v{version} ---")
        manifest = _read_manifest(store, version)
        if manifest is None:
            msg = f"v{version}: manifest not found"
            _info(f"  FAIL: {msg}")
            errors.append(msg)
            continue

        prefix = manifest.pop("_prefix", f"v{version}/arrays")

        for entry in manifest["fixtures"]:
            name = entry["name"]
            expected_hash = entry.get("sha256")
            data = store.read(f"{prefix}/{name}")

            if data is None:
                msg = f"v{version}/{name}: file missing from store"
                _info(f"  FAIL: {msg}")
                errors.append(msg)
                continue

            if expected_hash is None:
                msg = f"v{version}/{name}: no sha256 in manifest"
                _info(f"  FAIL: {msg}")
                errors.append(msg)
                continue

            actual_hash = hashlib.sha256(data).hexdigest()
            if actual_hash != expected_hash:
                msg = f"v{version}/{name}: sha256 mismatch expected={expected_hash[:12]} actual={actual_hash[:12]}"
                _info(f"  FAIL: {msg}")
                errors.append(msg)
            else:
                _info(f"  {name}: ok ({len(data)} bytes)")
                total_ok += 1

    _info(f"\nresult: {total_ok} ok, {len(errors)} failed")
    if errors:
        for e in errors:
            _info(f"  {e}")
        sys.exit(1)
    else:
        _info("all fixtures verified.")


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
                _info(f"  v{prev_version} -> v{version}: ok ({len(names)} fixtures{extra})")
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
# Forward compatibility
# ---------------------------------------------------------------------------


PYPI_PACKAGE = "vortex-data"
PYPI_JSON_URL = f"https://pypi.org/pypi/{PYPI_PACKAGE}/json"
FORWARD_READER = Path(__file__).resolve().with_name("forward_reader.py")
# Packages added to every reader environment. Some old wheels import typing_extensions without
# declaring it, and substrait 0.24 moved the generated protobuf modules they import. Every release
# so far accepts substrait 0.23. `uv run` ignores UV_CONSTRAINT for `--with`, so the pin is a
# requirement rather than a constraint.
FORWARD_EXTRA_DEPS = ["pyarrow", "typing_extensions", "substrait<0.24"]


def _released_versions() -> list[str]:
    """Return every final (non-pre-release, non-yanked) release of the Python package."""
    with urlopen(PYPI_JSON_URL, timeout=30) as resp:
        releases = json.load(resp)["releases"]
    versions = [
        version
        for version, files in releases.items()
        if re.fullmatch(r"\d+\.\d+\.\d+", version) and any(not f.get("yanked") for f in files)
    ]
    versions.sort(key=_version_sort_key)
    return versions


def _run_forward_reader(version: str, editions: list[str], fixtures_dir: Path, python: str) -> dict:
    """Read the files of `editions` with the released `version`, in an isolated environment.

    The reader reports each file as it goes. If an old native reader aborts the process, the file
    it was reading fails with the crash output, and the reader restarts without it.
    """
    passed: list[dict] = []
    failed: list[dict] = []
    with tempfile.NamedTemporaryFile("w", suffix=".txt") as skip_file:
        skip: list[str] = []
        while True:
            skip_file.seek(0)
            skip_file.truncate()
            skip_file.write("\n".join(skip) + "\n")
            skip_file.flush()

            cmd = ["uv", "run", "--quiet", "--isolated", "--no-project", "--python", python]
            cmd += ["--with", f"{PYPI_PACKAGE}=={version}"]
            for dep in FORWARD_EXTRA_DEPS:
                cmd += ["--with", dep]
            cmd += ["python", "-I", str(FORWARD_READER), "--dir", str(fixtures_dir)]
            cmd += ["--editions", ",".join(editions), "--version", version, "--skip", skip_file.name]
            result = subprocess.run(cmd, capture_output=True, text=True)

            reading: str | None = None
            done = False
            for line in result.stdout.splitlines():
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "reading" in record:
                    reading = record["reading"]
                elif "done" in record:
                    done = True
                else:
                    reading = None
                    skip.append(f"{record['edition']}/{record['name']}")
                    (passed if record["error"] is None else failed).append(record)
            if done:
                break

            detail = _crash_detail(result)
            if reading is None:
                # The reader failed before reading a file, e.g. the environment did not install.
                failed.extend({"edition": e, "name": "(reader)", "error": detail} for e in editions)
                break
            edition, name = reading.split("/", 1)
            failed.append({"edition": edition, "name": name, "error": f"reader crashed: {detail}"})
            skip.append(reading)
    return {"version": version, "passed": passed, "failed": failed}


def _crash_detail(result: subprocess.CompletedProcess) -> str:
    """Summarize why a reader process stopped early, preferring a Rust panic message."""
    lines = result.stderr.strip().splitlines()
    for i, line in enumerate(lines):
        if "panicked at" in line:
            return " ".join(lines[i : i + 2]).strip()
    # Otherwise the end of the output, without native backtrace frames.
    frames = re.compile(r"^\s*(\d+:|at )")
    tail = [line for line in lines if not frames.match(line)][-5:]
    return "\n".join(tail) or f"exit code {result.returncode}"


def cmd_forward(args: argparse.Namespace) -> None:
    """Write files with the current writer per frozen edition, then read them with old releases."""
    with tempfile.TemporaryDirectory() as tmpdir:
        if args.dir:
            fixtures_dir = Path(args.dir)
        else:
            fixtures_dir = Path(tmpdir) / "forward"
            _info("generating forward-compatibility fixtures...")
            _run_rust_generate_forward(fixtures_dir, exclude=args.exclude, profile=args.profile)
        _check_forward(fixtures_dir, args)


def _check_forward(fixtures_dir: Path, args: argparse.Namespace) -> None:
    manifest = json.loads((fixtures_dir / "forward.json").read_text())
    editions = manifest["editions"]
    if args.editions:
        wanted = {e.strip() for e in args.editions.split(",")}
        unknown = wanted - {e["edition"] for e in editions}
        if unknown:
            _info(f"error: unknown or unfrozen editions: {', '.join(sorted(unknown))}")
            sys.exit(1)
        editions = [e for e in editions if e["edition"] in wanted]

    # A frozen edition the current writer cannot write anything with has no forward guarantee to
    # test; that is a failure in its own right, not a reason to skip it.
    failures: list[tuple[str, dict]] = []
    for edition in editions:
        unwritable = ", ".join(f["name"] for f in edition["unwritable"])
        if not edition["written"]:
            errors = sorted({f["error"] for f in edition["unwritable"]})
            _info(f"{edition['edition']}: FAILED, the current writer cannot write any fixture: {'; '.join(errors)}")
            failures.append(("(writer)", {"edition": edition["edition"], "name": "(all)", "error": errors[0]}))
            continue
        suffix = f" (not representable: {unwritable})" if unwritable else ""
        _info(
            f"{edition['edition']}: {len(edition['written'])} files, "
            f"readable from v{edition['min_library_version']}{suffix}"
        )
    editions = [e for e in editions if e["written"]]

    if args.versions:
        versions = [v.strip() for v in args.versions.split(",")]
    else:
        versions = _released_versions()
        oldest = min((_version_sort_key(e["min_library_version"]) for e in editions), default=None)
        versions = [v for v in versions if oldest is not None and _version_sort_key(v) >= oldest]
        if args.mode == "last":
            # The first and the latest release each edition covers.
            ends = {e["min_library_version"] for e in editions}
            if versions:
                ends.add(versions[-1])
            versions = [v for v in versions if v in ends]

    # Each release reads every edition whose min_library_version it has reached.
    plan: list[tuple[str, list[str]]] = []
    for version in versions:
        covered = [
            e["edition"] for e in editions if _version_sort_key(version) >= _version_sort_key(e["min_library_version"])
        ]
        if covered:
            plan.append((version, covered))
        else:
            _info(f"skipping v{version}: older than every edition's min_library_version")

    if not plan:
        _info("no released versions to check")
        _report_forward_failures(failures)
        sys.exit(1)

    _info(f"\nreading with {len(plan)} release(s) of {PYPI_PACKAGE}, {args.jobs} at a time...")
    results: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {
            pool.submit(_run_forward_reader, version, covered, fixtures_dir, args.python): version
            for version, covered in plan
        }
        for future in as_completed(futures):
            result = future.result()
            version = futures[future]
            results[version] = result
            status = "FAILED" if result["failed"] else "ok"
            _info(f"  v{version}: {len(result['passed'])} passed, {len(result['failed'])} failed {status}")

    _info("\nresults by edition:")
    for edition in editions:
        name = edition["edition"]
        bad_versions = []
        checked = 0
        for version, covered in plan:
            if name not in covered:
                continue
            checked += 1
            edition_failures = [f for f in results[version]["failed"] if f["edition"] == name]
            if edition_failures:
                bad_versions.append(version)
                failures.extend((version, f) for f in edition_failures)
        if bad_versions:
            _info(f"  {name}: FAILED on {len(bad_versions)}/{checked} release(s): {', '.join(bad_versions)}")
        else:
            _info(f"  {name}: ok on {checked} release(s)")

    if failures:
        _report_forward_failures(failures)
        sys.exit(1)
    _info("\nall files are readable by every release their edition covers.")


def _report_forward_failures(failures: list[tuple[str, dict]]) -> None:
    """Print `(version, failure)` pairs; the version is `(writer)` for write failures."""
    _info(f"\n{len(failures)} failure(s):")
    for version, f in sorted(failures, key=lambda vf: (_version_sort_key(vf[0]), vf[1]["edition"])):
        where = version if version.startswith("(") else f"v{version}"
        _info(f"  {where} {f['edition']}/{f['name']}: {f['error']}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_store(store: Store, args: argparse.Namespace, prompt: str) -> Store:
    """Return a DryRunStore wrapper or the real store, with optional confirmation."""
    if args.dry_run:
        return DryRunStore(store)
    if not args.yes:
        _info(prompt)
        answer = input("proceed? [y/N] ").strip().lower()
        if answer not in ("y", "yes"):
            _info("aborted")
            sys.exit(1)
    return store


def _parallel_upload(store: Store, items: list[tuple[str, Path]], max_workers: int = 8) -> None:
    """Upload files to the store in parallel."""
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(store.write_file, key, local): key for key, local in items}
        for future in as_completed(futures):
            future.result()


def _parallel_download(
    store: Store,
    fixtures: list[dict],
    prefix: str,
    dest: Path,
    max_workers: int = 8,
) -> list[tuple[str, str]]:
    """Download fixture files from the store in parallel.

    Returns a list of (name, error) for any failures.
    """
    failures: list[tuple[str, str]] = []
    total_bytes = 0

    def _download_one(entry: dict) -> tuple[str, bytes | None]:
        name = entry["name"]
        data = store.read(f"{prefix}/{name}")
        return name, data

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_download_one, entry): entry["name"] for entry in fixtures}
        for future in as_completed(futures):
            name, data = future.result()
            if data is None:
                failures.append((name, f"not found at {prefix}/{name}"))
            else:
                (dest / name).write_bytes(data)
                total_bytes += len(data)

    _info(f"  downloaded {len(fixtures) - len(failures)} fixtures ({total_bytes} bytes)")
    return failures


def _build_compat_bin(profile: str = "release") -> str:
    """Build vortex-compat and return the path to the binary.

    If VORTEX_COMPAT_BIN is set, skips the build and returns that path.
    Otherwise runs `cargo build` with visible output, then locates the binary.
    """
    bin_path = os.environ.get("VORTEX_COMPAT_BIN")
    if bin_path:
        return bin_path

    _info(f"building vortex-compat ({profile})...")
    _run_cmd(["cargo", "build", "-p", CARGO_BIN, "--profile", profile], check=True)

    # Ask cargo where the binary is.
    result = subprocess.run(
        ["cargo", "metadata", "--format-version=1", "--no-deps"],
        capture_output=True,
        text=True,
        check=True,
    )
    target_dir = json.loads(result.stdout)["target_directory"]
    # Cargo puts "dev" profile binaries in "debug/", all others in "<profile>/".
    dir_name = "debug" if profile == "dev" else profile
    bin_path = str(Path(target_dir) / dir_name / CARGO_BIN)
    return bin_path


def _run_rust_generate(output: Path, profile: str = "release") -> None:
    """Run `vortex-compat generate --output <dir>`."""
    bin_path = _build_compat_bin(profile)
    _run_cmd([bin_path, "generate", "--output", str(output)], check=True)


def _run_rust_generate_forward(output: Path, exclude: str | None = None, profile: str = "release") -> None:
    """Run `vortex-compat generate-forward --output <dir>`."""
    bin_path = _build_compat_bin(profile)
    cmd = [bin_path, "generate-forward", "--output", str(output)]
    if exclude:
        cmd += ["--exclude", exclude]
    _run_cmd(cmd, check=True)


def _run_rust_check(dir: Path, mode: str = "superset", profile: str = "release") -> dict:
    """Run `vortex-compat check --dir <dir> --mode <mode>` and parse JSON stdout."""
    bin_path = _build_compat_bin(profile)
    cmd = [bin_path, "check", "--dir", str(dir), "--mode", mode]
    _info(f"  $ {' '.join(cmd)}")
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=None, text=True)  # noqa: UP022

    if result.stdout.strip():
        return json.loads(result.stdout)

    if result.returncode != 0:
        return {
            "passed": [],
            "failed": [{"name": "(all)", "error": "check process failed"}],
            "skipped": [],
        }
    return {"passed": [], "failed": [], "skipped": []}


def _run_cmd(cmd: list[str], check: bool = False, cwd: Path | None = None) -> subprocess.CompletedProcess:
    _info(f"  $ {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False, cwd=cwd)
    if check and result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, cmd)
    return result


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
        description="Vortex backward- and forward-compatibility fixture orchestrator",
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--profile",
        default="release",
        help="Cargo build profile (default: release). Use 'dev' for faster builds.",
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    # -- generate --
    p = sub.add_parser(
        "generate",
        help="Generate fixtures locally",
        description=(
            "Build all fixture .vortex files using the current binary and write\n"
            "them to a directory. Version is auto-detected from the nearest git\n"
            "tag at HEAD (or at --git-ref if specified)."
        ),
        epilog=(
            "examples:\n"
            "  uv run compat.py generate --output ./out\n"
            "  uv run compat.py generate --output ./out --git-ref v0.62.0"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--output", required=True, help="Output directory")
    p.add_argument(
        "--git-ref",
        help="Git ref for version detection (e.g. v0.62.0). "
        "Version is derived from the nearest tag at this ref. "
        "Fixtures are always built with the current binary.",
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
            "  uv run compat.py publish\n"
            "  uv run compat.py publish --dry-run\n"
            "  uv run compat.py publish --git-ref v0.62.0\n"
            "  uv run compat.py publish --store /tmp/store"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--store", default=DEFAULT_STORE, help="Store spec (default: %(default)s)")
    p.add_argument(
        "--git-ref",
        help="Git ref for version detection (e.g. v0.62.0). "
        "Version is derived from the nearest tag at this ref. "
        "Fixtures are always built with the current binary.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate credentials and show what would be uploaded, without writing",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing version in the store",
    )
    p.add_argument(
        "--update",
        action="store_true",
        help="Incremental update: add new fixtures to an existing version "
        "(hash-verified, skips unchanged files, errors on mismatches)",
    )
    p.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="Skip confirmation prompt",
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
            "  uv run compat.py check\n"
            "  uv run compat.py check --mode last\n"
            "  uv run compat.py check --versions 0.62.0,0.63.0\n"
            "  uv run compat.py check --store /tmp/store"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--store", default=DEFAULT_STORE, help="Store spec (default: %(default)s)")
    p.add_argument(
        "--versions",
        help="Comma-separated versions to check (mutually exclusive with --mode)",
    )
    p.add_argument(
        "--mode",
        choices=["all", "last"],
        default="all",
        help="Which versions to check: 'all' (default) or 'last' (most recent only). "
        "Mutually exclusive with --versions.",
    )

    # -- forward --
    p = sub.add_parser(
        "forward",
        help="Check that old releases read files the current writer produces per edition",
        description=(
            "Write every fixture with the current writer once per frozen core edition,\n"
            "enabling only that edition, then read each file with every released\n"
            f"{PYPI_PACKAGE} wheel from the edition's min_library_version on. Each\n"
            "release runs in an isolated uv environment, so no old code is compiled."
        ),
        epilog=(
            "examples:\n"
            "  uv run compat.py forward\n"
            "  uv run compat.py forward --mode last\n"
            "  uv run compat.py forward --editions core2025.05.0 --versions 0.36.0,0.90.0\n"
            "  uv run compat.py forward --exclude clickbench --jobs 8\n"
            "  uv run compat.py forward --dir /tmp/forward  # reuse generate-forward output"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--mode",
        choices=["all", "last"],
        default="all",
        help="Which releases read the files: 'all' (default) from each edition's min_library_version "
        "on, or 'last': only each edition's min_library_version and the latest release. "
        "Ignored with --versions.",
    )
    p.add_argument("--versions", help="Comma-separated releases to read with, instead of --mode")
    p.add_argument("--editions", help="Comma-separated frozen editions to check (default: all)")
    p.add_argument("--exclude", help="Comma-separated fixture name substrings to exclude")
    p.add_argument(
        "--dir",
        help="Use an existing `vortex-compat generate-forward` output instead of generating one",
    )
    p.add_argument("--jobs", type=int, default=4, help="Releases to read with concurrently (default: %(default)s)")
    p.add_argument(
        "--python",
        default="3.12",
        help="Python version for the reader environments (default: %(default)s)",
    )

    # -- list --
    p = sub.add_parser(
        "list",
        help="List versions or show a version's manifest",
        description="Inspect the contents of a fixture store.",
        epilog=(
            "examples:\n"
            "  uv run compat.py list\n"
            "  uv run compat.py list 0.62.0\n"
            "  uv run compat.py list --store /tmp/store"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--store", default=DEFAULT_STORE, help="Store spec (default: %(default)s)")
    p.add_argument("version", nargs="?", help="Show manifest for this version")

    # -- verify --
    p = sub.add_parser(
        "verify",
        help="Verify fixture file integrity against manifest sha256 hashes",
        description=(
            "Download every fixture file for every version and verify its\n"
            "SHA-256 hash matches the manifest. Also checks that all files\n"
            "listed in manifests are present in the store."
        ),
        epilog=("examples:\n  uv run compat.py verify\n  uv run compat.py verify --store /tmp/store"),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--store", default=DEFAULT_STORE, help="Store spec (default: %(default)s)")

    # -- validate-manifest --
    p = sub.add_parser(
        "validate-manifest",
        help="Check additive-only property across all versions",
        description=(
            "Verify that no fixtures were removed between consecutive versions.\n"
            "New fixtures are allowed; removals are errors."
        ),
        epilog=(
            "examples:\n  uv run compat.py validate-manifest\n  uv run compat.py validate-manifest --store /tmp/store"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--store", default=DEFAULT_STORE, help="Store spec (default: %(default)s)")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    commands = {
        "generate": cmd_generate,
        "publish": cmd_publish,
        "check": cmd_check,
        "forward": cmd_forward,
        "list": cmd_list,
        "verify": cmd_verify,
        "validate-manifest": cmd_validate_manifest,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
