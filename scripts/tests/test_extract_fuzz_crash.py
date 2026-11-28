# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Tests for `.github/scripts/extract_fuzz_crash.py`."""

from __future__ import annotations

import importlib.util
import io
import sys
import tarfile
import zipfile
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / ".github" / "scripts" / "extract_fuzz_crash.py"
FUZZ_TARGETS_DIR = REPO_ROOT / "fuzz" / "fuzz_targets"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("extract_fuzz_crash", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


efc = _load_module()


def test_bot_report_format():
    body = """
**Target**: `arrow_roundtrip`
**Crash File**: `crash-1dd9df3d790282b93cdb20911333313b0be3b6a0`
https://github.com/vortex-data/vortex/actions/runs/123456/artifacts/789012
"""
    details = efc.extract_crash_details(body, "vortex-data/vortex", "https://github.com", FUZZ_TARGETS_DIR)
    assert details.target == "arrow_roundtrip"
    assert details.crash_file == "crash-1dd9df3d790282b93cdb20911333313b0be3b6a0"
    assert details.artifact_id == "789012"
    assert details.attachment_url == ""
    assert details.rss_limit_mb == 0


def test_manual_multiline_command_and_tgz_attachment():
    body = r"""
[oom.tgz](https://github.com/user-attachments/files/35422130/oom.tgz)
[fuzz_local.log](https://github.com/user-attachments/files/35422457/fuzz_local.log)

```
cargo +nightly-2026-11-13 fuzz run -s none \
file_corruption oom-2875f50e9973079ca12b5811c161ee90d69eb128
```
"""
    details = efc.extract_crash_details(body, "vortex-data/vortex", "https://github.com", FUZZ_TARGETS_DIR)
    assert details.target == "file_corruption"
    assert details.crash_file == "oom-2875f50e9973079ca12b5811c161ee90d69eb128"
    assert details.artifact_id == ""
    assert details.attachment_url == "https://github.com/user-attachments/files/35422130/oom.tgz"
    assert details.rss_limit_mb == 2048


def test_inline_backtick_command_and_placeholder_fallback():
    inline = (
        "Reproduce with `cargo +nightly fuzz run --sanitizer=none "
        "file_corruption ./crash-0123456789abcdef` and "
        "https://github.com/user-attachments/files/1/crash-0123456789abcdef"
    )
    d_inline = efc.extract_crash_details(inline, "vortex-data/vortex", "https://github.com", FUZZ_TARGETS_DIR)
    assert d_inline.target == "file_corruption"
    assert d_inline.crash_file == "crash-0123456789abcdef"

    placeholder = (
        "Run `cargo fuzz run file_corruption <crash_file>` on "
        "https://github.com/user-attachments/files/2/crash-abcdef0123456789"
    )
    d_placeholder = efc.extract_crash_details(placeholder, "vortex-data/vortex", "https://github.com", FUZZ_TARGETS_DIR)
    assert d_placeholder.crash_file == "crash-abcdef0123456789"


def test_prefers_crash_archive_over_log_archive_and_rejects_unrelated_file():
    body = """
cargo fuzz run file_corruption crash-0123456789abcdef
https://github.com/user-attachments/files/1/crash-logs.zip
https://github.com/user-attachments/files/2/crash-0123456789abcdef.tgz
"""
    details = efc.extract_crash_details(body, "vortex-data/vortex", "https://github.com", FUZZ_TARGETS_DIR)
    assert details.attachment_url == "https://github.com/user-attachments/files/2/crash-0123456789abcdef.tgz"

    unrelated = """
cargo fuzz run file_corruption crash-0123456789abcdef
https://github.com/user-attachments/files/3/notes.csv
"""
    with pytest.raises(ValueError, match="attachment URL"):
        efc.extract_crash_details(unrelated, "vortex-data/vortex", "https://github.com", FUZZ_TARGETS_DIR)


def test_rejects_untrusted_host():
    body = (
        "cargo fuzz run file_corruption crash-0123456789abcdef\nhttps://evil.example/user-attachments/files/1/crash.tgz"
    )
    with pytest.raises(ValueError, match="attachment URL"):
        efc.extract_crash_details(body, "vortex-data/vortex", "https://github.com", FUZZ_TARGETS_DIR)


def test_tar_extract_rejects_symlink_and_enforces_member_limit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    symlink_tar = tmp_path / "symlink.tgz"
    with tarfile.open(symlink_tar, "w:gz") as tf:
        ti = tarfile.TarInfo("crash-0123456789abcdef")
        ti.type = tarfile.SYMTYPE
        ti.linkname = "/etc/passwd"
        tf.addfile(ti)

    with pytest.raises(ValueError, match="non-regular file"):
        efc._extract_from_tar(symlink_tar, "file_corruption", "crash-0123456789abcdef", tmp_path / "out")

    many_tar = tmp_path / "many.tgz"
    with tarfile.open(many_tar, "w:gz") as tf:
        for i in range(5):
            data = b"x"
            ti = tarfile.TarInfo(f"file_{i}")
            ti.size = len(data)
            tf.addfile(ti, io.BytesIO(data))

    monkeypatch.setattr(efc, "MAX_ARCHIVE_MEMBERS", 3)
    with pytest.raises(ValueError, match="exceeds 3 entries"):
        efc._extract_from_tar(many_tar, "file_corruption", "crash-0123456789abcdef", tmp_path / "out")


def test_zip_extract_happy_path(tmp_path: Path):
    zpath = tmp_path / "crash.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.writestr("file_corruption/crash-0123456789abcdef", b"payload")

    out = tmp_path / "out"
    efc._extract_from_zip(zpath, "file_corruption", "crash-0123456789abcdef", out)
    assert out.read_bytes() == b"payload"


def _write_tar(path: Path, entries: dict[str, bytes]) -> None:
    with tarfile.open(path, "w:gz") as tf:
        for name, data in entries.items():
            ti = tarfile.TarInfo(name)
            ti.size = len(data)
            tf.addfile(ti, io.BytesIO(data))


def test_tar_extract_happy_path_and_precedence(tmp_path: Path):
    crash_file = "crash-0123456789abcdef"

    # Prefer ./<target>/<crash_file> over root <crash_file>.
    precedence_tar = tmp_path / "precedence.tgz"
    _write_tar(
        precedence_tar,
        {
            crash_file: b"root-payload",
            f"./file_corruption/{crash_file}": b"target-payload",
        },
    )
    out = tmp_path / "out-target"
    efc._extract_from_tar(precedence_tar, "file_corruption", crash_file, out)
    assert out.read_bytes() == b"target-payload"

    # Fall back to a nested directory basename match when neither target nor root matches.
    nested_tar = tmp_path / "nested.tgz"
    _write_tar(
        nested_tar,
        {
            f"artifacts/other_dir/{crash_file}": b"nested-payload",
        },
    )
    out_nested = tmp_path / "out-nested"
    efc._extract_from_tar(nested_tar, "file_corruption", crash_file, out_nested)
    assert out_nested.read_bytes() == b"nested-payload"


def test_select_user_attachment_fallback_archive_and_raw_extension():
    crash_file = "crash-0123456789abcdef"

    # Non-log archive beats log archive even when the log archive appears first.
    body_archives = (
        "https://github.com/user-attachments/files/1/fuzz_logs.tgz\n"
        "https://github.com/user-attachments/files/2/repro.tgz\n"
    )
    assert (
        efc.select_user_attachment(body_archives, crash_file) == "https://github.com/user-attachments/files/2/repro.tgz"
    )

    # Raw attachment with an extension is accepted (below an exact match), while .log is rejected.
    body_txt = f"https://github.com/user-attachments/files/3/{crash_file}.txt"
    assert efc.select_user_attachment(body_txt, crash_file) == body_txt

    body_exact_and_txt = (
        f"https://github.com/user-attachments/files/3/{crash_file}.txt\n"
        f"https://github.com/user-attachments/files/4/{crash_file}\n"
    )
    assert (
        efc.select_user_attachment(body_exact_and_txt, crash_file)
        == f"https://github.com/user-attachments/files/4/{crash_file}"
    )

    body_log = f"https://github.com/user-attachments/files/5/{crash_file}.log"
    assert efc.select_user_attachment(body_log, crash_file) == ""

    # Checksum-only attachment is rejected; crash-named archive beats suffixed non-archives.
    body_checksum_only = f"https://github.com/user-attachments/files/6/{crash_file}.sha256"
    assert efc.select_user_attachment(body_checksum_only, crash_file) == ""

    body_archive_and_checksum = (
        f"https://github.com/user-attachments/files/6/{crash_file}.sha256\n"
        f"https://github.com/user-attachments/files/7/{crash_file}.tgz\n"
    )
    assert (
        efc.select_user_attachment(body_archive_and_checksum, crash_file)
        == f"https://github.com/user-attachments/files/7/{crash_file}.tgz"
    )

    # Suffixed raw crash input beats a log archive containing the crash name.
    body_log_archive_and_raw = (
        f"https://github.com/user-attachments/files/8/{crash_file}-logs.tgz\n"
        f"https://github.com/user-attachments/files/9/{crash_file}.bin\n"
    )
    assert (
        efc.select_user_attachment(body_log_archive_and_raw, crash_file)
        == f"https://github.com/user-attachments/files/9/{crash_file}.bin"
    )


def test_download_crash_file_dispatch_and_raw_size_limit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    crash_file = "crash-0123456789abcdef"

    def fake_curl_factory(payload_writer: Callable[[Path], object]) -> Callable[..., None]:
        def fake_run(cmd: list[str], *, check: bool = False, **_kwargs: object) -> None:
            assert check is True
            assert cmd[0] == "curl"
            out_path = Path(cmd[cmd.index("-o") + 1])
            payload_writer(out_path)

        return fake_run

    # .tgz dispatch
    monkeypatch.setattr(
        efc.subprocess,
        "run",
        fake_curl_factory(lambda p: _write_tar(p, {f"file_corruption/{crash_file}": b"from-tgz"})),
    )
    tgz_out = efc.download_crash_file(
        repo="vortex-data/vortex",
        target="file_corruption",
        crash_file=crash_file,
        dest_dir=tmp_path / "tgz",
        artifact_id="",
        attachment_url="https://github.com/user-attachments/files/1/repro.tgz",
    )
    assert tgz_out.read_bytes() == b"from-tgz"

    # .zip dispatch
    def write_zip(p: Path) -> None:
        with zipfile.ZipFile(p, "w") as zf:
            zf.writestr(f"file_corruption/{crash_file}", b"from-zip")

    monkeypatch.setattr(efc.subprocess, "run", fake_curl_factory(write_zip))
    zip_out = efc.download_crash_file(
        repo="vortex-data/vortex",
        target="file_corruption",
        crash_file=crash_file,
        dest_dir=tmp_path / "zip",
        artifact_id="",
        attachment_url="https://github.com/user-attachments/files/2/repro.zip",
    )
    assert zip_out.read_bytes() == b"from-zip"

    # raw dispatch (with .txt extension stripped to crash_file)
    monkeypatch.setattr(efc.subprocess, "run", fake_curl_factory(lambda p: p.write_bytes(b"from-raw")))
    raw_out = efc.download_crash_file(
        repo="vortex-data/vortex",
        target="file_corruption",
        crash_file=crash_file,
        dest_dir=tmp_path / "raw",
        artifact_id="",
        attachment_url=f"https://github.com/user-attachments/files/3/{crash_file}.txt",
    )
    assert raw_out.name == crash_file
    assert raw_out.read_bytes() == b"from-raw"

    # raw attachment exceeding MAX_ATTACHMENT_BYTES is rejected even if curl did not enforce it
    monkeypatch.setattr(efc, "MAX_ATTACHMENT_BYTES", 4)
    monkeypatch.setattr(efc.subprocess, "run", fake_curl_factory(lambda p: p.write_bytes(b"12345")))
    with pytest.raises(ValueError, match="exceeds 4 bytes"):
        efc.download_crash_file(
            repo="vortex-data/vortex",
            target="file_corruption",
            crash_file=crash_file,
            dest_dir=tmp_path / "oversized",
            artifact_id="",
            attachment_url=f"https://github.com/user-attachments/files/4/{crash_file}.bin",
        )
