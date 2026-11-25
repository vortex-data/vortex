#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors
"""Extract and safely download fuzzer crash inputs from GitHub issue bodies.

Supports both automated reports from report-fuzz-crash.yml (Actions artifacts)
and manually filed maintainer issues (cargo fuzz run command + GitHub file
attachments under https://github.com/user-attachments/files/...).
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

MAX_ATTACHMENT_BYTES = 100 * 1024 * 1024  # 100 MiB

TARGET_RE = re.compile(r"^[a-z0-9_]+$")
CRASH_FILE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
CRASH_FILENAME_FIND_RE = re.compile(
    r"(?<![A-Za-z0-9._-])((?:crash|oom|timeout|leak)-[0-9a-fA-F]{16,})"
)
USER_ATTACHMENT_RE = re.compile(
    r"https://github\.com/user-attachments/files/[0-9]+/[A-Za-z0-9._-]+"
)
BOT_TARGET_RE = re.compile(r"\*\*Target\*\*:\s*`([^`\r\n]+)`")
BOT_CRASH_FILE_RE = re.compile(r"\*\*Crash File\*\*:\s*`([^`\r\n]+)`")
CARGO_FUZZ_RUN_RE = re.compile(
    r"cargo\b[^\r\n]*?\bfuzz\s+run\b(?:[^\r\n\\]|\\[^\r\n]|\\[ \t]*\r?\n)*",
)

# cargo-fuzz run options that consume a separate following argument.
CARGO_FUZZ_VALUE_FLAGS = frozenset(
    {
        "-s",
        "--sanitizer",
        "-F",
        "--features",
        "--fuzz-dir",
        "--target",
        "--target-dir",
        "--jobs",
        "-j",
        "--profile",
        "--manifest-path",
    }
)

ARCHIVE_SUFFIXES = (".tar.gz", ".tgz", ".tar", ".zip")
IGNORED_ATTACHMENT_SUFFIXES = (".log", ".txt", ".md", ".json", ".png", ".jpg", ".jpeg", ".gif")


@dataclass(frozen=True)
class CrashDetails:
    target: str
    crash_file: str
    artifact_id: str
    attachment_url: str


def parse_cargo_fuzz_run(body: str, fuzz_targets_dir: Path) -> tuple[str, str]:
    """Extract (target, crash_file) from a cargo fuzz run command in the body."""
    for match in CARGO_FUZZ_RUN_RE.finditer(body):
        cmd = re.sub(r"\\[ \t]*\r?\n", " ", match.group(0))
        try:
            tokens = shlex.split(cmd)
        except ValueError:
            tokens = cmd.split()

        try:
            run_idx = tokens.index("run")
        except ValueError:
            continue

        positionals: list[str] = []
        i = run_idx + 1
        while i < len(tokens):
            tok = tokens[i]
            if tok == "--":
                break
            if tok in CARGO_FUZZ_VALUE_FLAGS:
                i += 2
                continue
            if tok.startswith("-"):
                i += 1
                continue
            positionals.append(tok)
            i += 1

        if not positionals:
            continue

        target = positionals[0]
        if TARGET_RE.match(target) and (fuzz_targets_dir / f"{target}.rs").is_file():
            crash_file = Path(positionals[1]).name if len(positionals) >= 2 else ""
            return target, crash_file

    return "", ""


def select_user_attachment(body: str, crash_file: str) -> str:
    """Pick the most likely crash input attachment from GitHub user-attachments URLs."""
    urls = list(dict.fromkeys(USER_ATTACHMENT_RE.findall(body)))
    if not urls:
        return ""

    crash_kind = crash_file.split("-", 1)[0].lower() if "-" in crash_file else ""

    def score(url: str) -> tuple[int, int]:
        name = url.rsplit("/", 1)[-1]
        lower = name.lower()
        if name == crash_file:
            return (0, 0)
        if lower.endswith(ARCHIVE_SUFFIXES):
            if crash_file.lower() in lower or (crash_kind and crash_kind in lower):
                return (1, 0)
            return (2, 0)
        if lower.endswith(IGNORED_ATTACHMENT_SUFFIXES):
            return (4, 0)
        return (3, 0)

    ranked = sorted(enumerate(urls), key=lambda item: (score(item[1]), item[0]))
    best_url = ranked[0][1]
    if score(best_url)[0] >= 4:
        return ""
    return best_url


def extract_crash_details(
    body: str,
    repo: str,
    server_url: str,
    fuzz_targets_dir: Path,
) -> CrashDetails:
    """Extract and validate crash metadata from an issue body."""
    target_match = BOT_TARGET_RE.search(body)
    crash_file_match = BOT_CRASH_FILE_RE.search(body)

    cmd_target, cmd_crash_file = parse_cargo_fuzz_run(body, fuzz_targets_dir)

    target = target_match.group(1).strip() if target_match else cmd_target
    if not TARGET_RE.match(target) or not (fuzz_targets_dir / f"{target}.rs").is_file():
        raise ValueError(
            "The issue has no target, or the target is not a fuzz target of this repository"
        )

    crash_file = crash_file_match.group(1).strip() if crash_file_match else cmd_crash_file
    if not crash_file:
        fallback = CRASH_FILENAME_FIND_RE.search(body)
        if fallback:
            crash_file = fallback.group(1)
    if not CRASH_FILE_RE.match(crash_file):
        raise ValueError("The issue has no crash file name, or the name is not valid")

    prefix = f"{server_url.rstrip('/')}/{repo}/actions/runs/"
    artifact_re = re.compile(re.escape(prefix) + r"([0-9]+)/artifacts/([0-9]+)")
    artifact_match = artifact_re.search(body)

    artifact_id = ""
    attachment_url = ""
    if artifact_match:
        artifact_id = artifact_match.group(2)
    else:
        attachment_url = select_user_attachment(body, crash_file)
        if not attachment_url or not USER_ATTACHMENT_RE.fullmatch(attachment_url):
            raise ValueError(
                "The issue has no valid repository artifact URL or GitHub file attachment URL"
            )

    return CrashDetails(
        target=target,
        crash_file=crash_file,
        artifact_id=artifact_id,
        attachment_url=attachment_url,
    )


def _write_limited(src_stream: object, dst_path: Path, max_bytes: int) -> None:
    total = 0
    with dst_path.open("wb") as out:
        while True:
            chunk = src_stream.read(65536)  # type: ignore[attr-defined]
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                raise ValueError(f"Extracted crash file exceeds {max_bytes} bytes")
            out.write(chunk)


def _extract_from_zip(archive_path: Path, target: str, crash_file: str, dst_path: Path) -> None:
    preferred = (f"{target}/{crash_file}", crash_file)
    with zipfile.ZipFile(archive_path) as zf:
        chosen: zipfile.ZipInfo | None = None
        for name in preferred:
            for info in zf.infolist():
                if info.filename == name:
                    chosen = info
                    break
            if chosen is not None:
                break
        if chosen is None:
            for info in zf.infolist():
                if not info.is_dir() and Path(info.filename).name == crash_file:
                    chosen = info
                    break

        if chosen is None:
            raise ValueError(f"Crash file {crash_file} not found in zip archive")

        mode = (chosen.external_attr >> 16) & 0o170000
        if mode == stat.S_IFLNK:
            raise ValueError("Refusing to extract symlink from zip archive")
        if chosen.file_size > MAX_ATTACHMENT_BYTES:
            raise ValueError(f"Zip entry exceeds {MAX_ATTACHMENT_BYTES} bytes")

        with zf.open(chosen) as src:
            _write_limited(src, dst_path, MAX_ATTACHMENT_BYTES)


def _extract_from_tar(archive_path: Path, target: str, crash_file: str, dst_path: Path) -> None:
    preferred = (f"{target}/{crash_file}", crash_file)
    with tarfile.open(archive_path, "r:*") as tf:
        members = tf.getmembers()
        chosen: tarfile.TarInfo | None = None
        for name in preferred:
            for member in members:
                normalized = member.name.lstrip("./")
                if normalized == name:
                    chosen = member
                    break
            if chosen is not None:
                break
        if chosen is None:
            for member in members:
                if member.isfile() and Path(member.name).name == crash_file:
                    chosen = member
                    break

        if chosen is None:
            raise ValueError(f"Crash file {crash_file} not found in tar archive")
        if not chosen.isfile():
            raise ValueError("Refusing to extract non-regular file from tar archive")
        if chosen.size > MAX_ATTACHMENT_BYTES:
            raise ValueError(f"Tar entry exceeds {MAX_ATTACHMENT_BYTES} bytes")

        src = tf.extractfile(chosen)
        if src is None:
            raise ValueError("Could not read member from tar archive")
        with src:
            _write_limited(src, dst_path, MAX_ATTACHMENT_BYTES)


def download_crash_file(
    *,
    repo: str,
    target: str,
    crash_file: str,
    dest_dir: Path,
    artifact_id: str,
    attachment_url: str,
) -> Path:
    """Download and safely extract the crash file into dest_dir."""
    if not TARGET_RE.match(target):
        raise ValueError("Invalid fuzz target name")
    if not CRASH_FILE_RE.match(crash_file):
        raise ValueError("Invalid crash file name")

    dest_dir.mkdir(parents=True, exist_ok=True)
    dst_path = dest_dir / crash_file

    with tempfile.TemporaryDirectory(dir=dest_dir) as tmp:
        tmp_dir = Path(tmp)
        if artifact_id:
            if not artifact_id.isdigit():
                raise ValueError("Invalid artifact ID")
            zip_path = tmp_dir / "crash-artifact.zip"
            with zip_path.open("wb") as out:
                subprocess.run(
                    ["gh", "api", f"repos/{repo}/actions/artifacts/{artifact_id}/zip"],
                    stdout=out,
                    check=True,
                )
            _extract_from_zip(zip_path, target, crash_file, dst_path)
        elif attachment_url:
            if not USER_ATTACHMENT_RE.fullmatch(attachment_url):
                raise ValueError("Attachment URL is not a valid GitHub user-attachments URL")
            filename = attachment_url.rsplit("/", 1)[-1]
            downloaded = tmp_dir / filename
            subprocess.run(
                [
                    "curl",
                    "-fLsS",
                    "--max-time",
                    "60",
                    "--max-filesize",
                    str(MAX_ATTACHMENT_BYTES),
                    "-o",
                    str(downloaded),
                    attachment_url,
                ],
                check=True,
            )
            lower = filename.lower()
            if lower.endswith(".zip"):
                _extract_from_zip(downloaded, target, crash_file, dst_path)
            elif lower.endswith((".tgz", ".tar.gz", ".tar")):
                _extract_from_tar(downloaded, target, crash_file, dst_path)
            else:
                shutil.copyfile(downloaded, dst_path)
        else:
            raise ValueError("Either --artifact-id or --attachment-url must be provided")

    if not dst_path.is_file() or dst_path.is_symlink():
        raise ValueError(f"Extracted crash file {crash_file} is not a regular file")
    return dst_path


def _write_outputs(outputs: dict[str, str], github_output: str | None) -> None:
    lines = [f"{k}={v}\n" for k, v in outputs.items()]
    if github_output:
        with open(github_output, "a", encoding="utf-8") as f:
            f.writelines(lines)
    else:
        sys.stdout.writelines(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract or download fuzzer crash inputs.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract_p = subparsers.add_parser("extract", help="Extract crash metadata from an issue body.")
    extract_p.add_argument("--body-file", type=Path, required=True)
    extract_p.add_argument("--repo", required=True)
    extract_p.add_argument("--server-url", default="https://github.com")
    extract_p.add_argument("--fuzz-targets-dir", type=Path, default=Path("fuzz/fuzz_targets"))
    extract_p.add_argument("--github-output", default=os.environ.get("GITHUB_OUTPUT"))

    download_p = subparsers.add_parser("download", help="Download and extract the crash file.")
    download_p.add_argument("--repo", required=True)
    download_p.add_argument("--target", required=True)
    download_p.add_argument("--crash-file", required=True)
    download_p.add_argument("--dest-dir", type=Path, required=True)
    download_p.add_argument("--artifact-id", default="")
    download_p.add_argument("--attachment-url", default="")
    download_p.add_argument("--github-output", default=os.environ.get("GITHUB_OUTPUT"))

    args = parser.parse_args()
    try:
        if args.command == "extract":
            body = args.body_file.read_text(encoding="utf-8")
            details = extract_crash_details(
                body=body,
                repo=args.repo,
                server_url=args.server_url,
                fuzz_targets_dir=args.fuzz_targets_dir,
            )
            _write_outputs(
                {
                    "target": details.target,
                    "crash_file": details.crash_file,
                    "artifact_id": details.artifact_id,
                    "attachment_url": details.attachment_url,
                },
                args.github_output,
            )
            source = (
                f"artifact={details.artifact_id}"
                if details.artifact_id
                else f"attachment={details.attachment_url}"
            )
            print(f"Extracted: target={details.target}, crash_file={details.crash_file}, {source}")
        elif args.command == "download":
            path = download_crash_file(
                repo=args.repo,
                target=args.target,
                crash_file=args.crash_file,
                dest_dir=args.dest_dir,
                artifact_id=args.artifact_id,
                attachment_url=args.attachment_url,
            )
            _write_outputs({"crash_file_path": str(path)}, args.github_output)
            print(f"Downloaded crash file: {path}")
    except (
        ValueError,
        subprocess.CalledProcessError,
        OSError,
        zipfile.BadZipFile,
        tarfile.TarError,
    ) as exc:
        print(f"::error::{exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
