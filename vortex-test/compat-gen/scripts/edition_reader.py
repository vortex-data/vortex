#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Isolated oldest-reader worker for `compat.py check-editions`.

Invoked inside a `uv run --with vortex-data==<version> --with pyarrow` environment.
Reads every fixture in `--src-dir` with the installed `vortex` wheel, compares
decoded values against HEAD's expected Arrow IPC files in `--expected-dir`, and
isolates crashes and per-column causes so every failure is deterministic.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import tempfile
from collections.abc import Callable
from pathlib import Path

import pyarrow as pa
import pyarrow.ipc as ipc

import vortex


def _values_equal(a: object, b: object) -> bool:
    if isinstance(a, float) and isinstance(b, float):
        if math.isnan(a) and math.isnan(b):
            return True
        return a == b
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_values_equal(x, y) for x, y in zip(a, b, strict=True))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_values_equal(a[k], b[k]) for k in a)
    return a == b


def _tables_equal(actual: pa.Table, expected: pa.Table) -> tuple[bool, str | None]:
    if actual.num_rows != expected.num_rows:
        return False, f"row count mismatch: actual={actual.num_rows}, expected={expected.num_rows}"
    if actual.num_columns != expected.num_columns:
        return (
            False,
            f"column count mismatch: actual={actual.num_columns}, expected={expected.num_columns}",
        )
    if actual.equals(expected):
        return True, None
    for i in range(actual.num_columns):
        ac = actual.column(i).combine_chunks()
        ec = expected.column(i).combine_chunks()
        if ac.equals(ec):
            continue
        if not _values_equal(ac.to_pylist(), ec.to_pylist()):
            field_name = actual.schema.field(i).name
            return (
                False,
                f"value mismatch in column {i} ({field_name}, actual_type={ac.type}, expected_type={ec.type})",
            )
    return True, None


_FIELD_WRAPPER_RE = re.compile(r"Failed to canonicalize field Field \{.*?\}: ")


def _clean_error(err: str) -> str:
    """Extract the concise first meaningful line from a Rust/Arrow/Python error."""
    lines = [line.strip() for line in err.splitlines() if line.strip()]
    for line in lines:
        if line in ("Backtrace:", "disabled backtrace"):
            continue
        cleaned = line.removesuffix(" Backtrace: disabled backtrace").strip()
        cleaned = _FIELD_WRAPPER_RE.sub("", cleaned)
        return cleaned
    return err.strip()


def _extract_panic_reason(stderr_txt: str, label: str) -> str:
    lines = [line.strip() for line in stderr_txt.splitlines() if line.strip()]
    for i, line in enumerate(lines):
        if "panicked at" in line and i + 1 < len(lines):
            return f"{label}: {lines[i + 1]}"
    if lines:
        return f"{label}: {lines[0]}"
    return label


def _run_isolated(fn: Callable[[], dict]) -> dict:
    """Run `fn()` in a forked child on POSIX so Rust FFI aborts cannot kill the worker."""
    if not hasattr(os, "fork"):
        return fn()

    with (
        tempfile.NamedTemporaryFile("w+", delete=False) as tf,
        tempfile.NamedTemporaryFile("w+", delete=False) as ef,
    ):
        out_path, err_path = tf.name, ef.name

    pid = os.fork()
    if pid == 0:
        err_fd = os.open(err_path, os.O_WRONLY | os.O_TRUNC)
        os.dup2(err_fd, 2)
        os.close(err_fd)
        try:
            res = fn()
        except Exception as e:
            res = {"ok": False, "error": _clean_error(f"{type(e).__name__}: {e}")}
        Path(out_path).write_text(json.dumps(res))
        os._exit(0)

    _, status = os.waitpid(pid, 0)
    try:
        if os.WIFSIGNALED(status):
            sig = os.WTERMSIG(status)
            stderr_txt = Path(err_path).read_text(errors="replace")
            return {"ok": False, "error": _extract_panic_reason(stderr_txt, f"signal {sig}")}
        if os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0:
            content = Path(out_path).read_text()
            if content.strip():
                return json.loads(content)
        stderr_txt = Path(err_path).read_text(errors="replace")
        return {
            "ok": False,
            "error": _extract_panic_reason(stderr_txt, f"exit {os.WEXITSTATUS(status)}"),
        }
    finally:
        for p in (out_path, err_path):
            try:
                os.unlink(p)
            except OSError:
                pass


def _probe_column_causes(path: Path) -> list[str]:
    """Probe each top-level struct column individually so every failure cause is reported."""
    col_names_res = _run_isolated(lambda: _get_struct_columns(path))
    if not col_names_res.get("ok"):
        return []
    columns = col_names_res.get("columns", [])
    if len(columns) <= 1:
        return []

    causes: set[str] = set()
    for col in columns:
        res = _run_isolated(lambda c=col: _read_single_column(path, c))
        if not res.get("ok") and res.get("error"):
            causes.add(res["error"])
    return sorted(causes)


def _get_struct_columns(path: Path) -> dict:
    vf = vortex.open(str(path))
    dtype = vf.dtype
    names_attr = getattr(dtype, "names", None)
    if names_attr is None:
        return {"ok": True, "columns": []}
    names = list(names_attr() if callable(names_attr) else names_attr)
    return {"ok": True, "columns": names}


def _read_single_column(path: Path, col: str) -> dict:
    vf = vortex.open(str(path))
    vf.scan(projection=[col]).read_all().to_arrow_array()
    return {"ok": True}


def _verify_file(path: Path, exp_path: Path) -> dict:
    def _do() -> dict:
        vf = vortex.open(str(path))
        actual = vf.to_arrow().read_all()
        with ipc.open_file(str(exp_path)) as reader:
            expected = reader.read_all()
        ok, reason = _tables_equal(actual, expected)
        if ok:
            return {"ok": True}
        return {"ok": False, "error": reason or "table mismatch"}

    res = _run_isolated(_do)
    if res.get("ok"):
        return res

    # Collect every distinct cause across columns so parallel scan races don't hide causes.
    col_causes = _probe_column_causes(path)
    if col_causes:
        return {
            "ok": False,
            "error": "; ".join(col_causes),
            "causes": col_causes,
        }
    err = res.get("error", "unknown error")
    return {"ok": False, "error": err, "causes": [err]}


def cmd_verify(src_dir: Path, expected_dir: Path) -> None:
    manifest = json.loads((src_dir / "fixtures.json").read_text())
    passed: list[str] = []
    failed: list[dict] = []
    for entry in manifest["fixtures"]:
        name = entry["name"]
        path = src_dir / name
        exp_path = expected_dir / f"{name}.arrow"
        if not exp_path.exists():
            failed.append(
                {
                    "name": name,
                    "error": f"expected Arrow IPC missing: {exp_path.name}",
                    "causes": [f"expected Arrow IPC missing: {exp_path.name}"],
                }
            )
            continue
        res = _verify_file(path, exp_path)
        if res.get("ok"):
            passed.append(name)
        else:
            failed.append(
                {
                    "name": name,
                    "error": res.get("error", "unknown error"),
                    "causes": res.get("causes", [res.get("error", "unknown error")]),
                }
            )
    print(json.dumps({"passed": passed, "failed": failed}))


def main() -> None:
    parser = argparse.ArgumentParser(description="Isolated oldest-reader worker for compat-gen")
    sub = parser.add_subparsers(dest="command", required=True)

    p_verify = sub.add_parser("verify")
    p_verify.add_argument("--src-dir", type=Path, required=True)
    p_verify.add_argument("--expected-dir", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "verify":
        cmd_verify(args.src_dir, args.expected_dir)


if __name__ == "__main__":
    main()
