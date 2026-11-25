# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Tests for `.github/scripts/extract_fuzz_crash.py`."""

from __future__ import annotations

import importlib.util
import io
import sys
import tarfile
import zipfile
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
