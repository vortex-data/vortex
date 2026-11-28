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
import signal
import tempfile
import time
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
    if isinstance(a, tuple) and isinstance(b, tuple):
        return len(a) == len(b) and all(_values_equal(x, y) for x, y in zip(a, b, strict=True))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_values_equal(a[k], b[k]) for k in a)
    return a == b


def _normalize_field(field: pa.Field) -> pa.Field:
    """Normalize an Arrow field's representation type while preserving its name and nullability."""
    return pa.field(field.name, _normalize_type(field.type), nullable=field.nullable)


def _normalize_type(dtype: pa.DataType) -> pa.DataType:
    """Normalize Arrow representation variants across Vortex/PyArrow versions without erasing logical types."""
    if pa.types.is_string_view(dtype) or pa.types.is_large_string(dtype):
        return pa.string()
    if pa.types.is_binary_view(dtype) or pa.types.is_large_binary(dtype):
        return pa.binary()
    if pa.types.is_list_view(dtype) or pa.types.is_large_list(dtype) or pa.types.is_list(dtype):
        return pa.list_(pa.field("item", _normalize_type(dtype.value_type), nullable=dtype.value_field.nullable))
    if pa.types.is_fixed_size_list(dtype):
        return pa.list_(
            pa.field("item", _normalize_type(dtype.value_type), nullable=dtype.value_field.nullable),
            dtype.list_size,
        )
    if pa.types.is_struct(dtype):
        return pa.struct([_normalize_field(f) for f in dtype])
    if pa.types.is_map(dtype):
        return pa.map_(
            pa.field("key", _normalize_type(dtype.key_type), nullable=dtype.key_field.nullable),
            pa.field("value", _normalize_type(dtype.item_type), nullable=dtype.item_field.nullable),
            keys_sorted=dtype.keys_sorted,
        )
    if pa.types.is_dictionary(dtype):
        return _normalize_type(dtype.value_type)
    return dtype


def _compare_column_causes(
    col: str,
    actual_field: pa.Field,
    actual_col: pa.Array,
    expected_field: pa.Field,
    expected_col: pa.Array,
) -> list[str]:
    if len(actual_col) != len(expected_col):
        return [f"row count mismatch in column {col}: actual={len(actual_col)}, expected={len(expected_col)}"]
    causes: list[str] = []
    if _normalize_field(actual_field) != _normalize_field(expected_field):
        causes.append(f"type mismatch in column {col}: actual={actual_field}, expected={expected_field}")
    if not actual_col.equals(expected_col) and not _values_equal(actual_col.to_pylist(), expected_col.to_pylist()):
        causes.append(
            f"value mismatch in column {col} (actual_type={actual_col.type}, expected_type={expected_col.type})"
        )
    return causes


def _tables_equal(actual: pa.Table, expected: pa.Table) -> list[str]:
    if actual.num_rows != expected.num_rows:
        return [f"row count mismatch: actual={actual.num_rows}, expected={expected.num_rows}"]
    if actual.num_columns != expected.num_columns:
        return [f"column count mismatch: actual={actual.num_columns}, expected={expected.num_columns}"]
    if actual.schema.names != expected.schema.names:
        return [f"column names mismatch: actual={actual.schema.names}, expected={expected.schema.names}"]
    if actual.equals(expected):
        return []
    causes: list[str] = []
    for i in range(actual.num_columns):
        actual_field = actual.schema.field(i)
        expected_field = expected.schema.field(i)
        ac = actual.column(i).combine_chunks()
        ec = expected.column(i).combine_chunks()
        causes.extend(_compare_column_causes(actual_field.name, actual_field, ac, expected_field, ec))
    return causes


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


def _run_isolated(fn: Callable[[], dict], timeout_s: float = 120.0) -> dict:
    """Run `fn()` in a forked child on POSIX so Rust FFI aborts or hangs cannot kill the worker."""
    if not hasattr(os, "fork"):
        try:
            return fn()
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException as e:
            return {"ok": False, "error": _clean_error(f"{type(e).__name__}: {e}")}

    with (
        tempfile.NamedTemporaryFile("w+", delete=False) as tf,
        tempfile.NamedTemporaryFile("w+", delete=False) as ef,
    ):
        out_path, err_path = tf.name, ef.name

    pid = os.fork()
    if pid == 0:
        exit_code = 0
        try:
            err_fd = os.open(err_path, os.O_WRONLY | os.O_TRUNC)
            os.dup2(err_fd, 2)
            os.close(err_fd)
            try:
                res = fn()
            except BaseException as e:
                res = {"ok": False, "error": _clean_error(f"{type(e).__name__}: {e}")}
            Path(out_path).write_text(json.dumps(res))
        except BaseException:
            exit_code = 1
        finally:
            os._exit(exit_code)

    deadline = time.monotonic() + timeout_s
    status = 0
    timed_out = False
    while True:
        waited_pid, status = os.waitpid(pid, os.WNOHANG)
        if waited_pid == pid:
            break
        if time.monotonic() >= deadline:
            timed_out = True
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            os.waitpid(pid, 0)
            break
        time.sleep(0.02)

    try:
        if timed_out:
            return {"ok": False, "timed_out": True, "error": f"timed out after {timeout_s:.0f}s"}
        if os.WIFSIGNALED(status):
            sig = os.WTERMSIG(status)
            stderr_txt = Path(err_path).read_text(errors="replace")
            return {"ok": False, "error": _extract_panic_reason(stderr_txt, f"signal {sig}")}
        if os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0:
            content = Path(out_path).read_text()
            if content.strip():
                res = json.loads(content)
                if not res.get("ok") and "Runtime dropped task without completing it" in res.get("error", ""):
                    stderr_txt = Path(err_path).read_text(errors="replace")
                    if "panicked at" in stderr_txt:
                        res["error"] = _extract_panic_reason(stderr_txt, "PanicException")
                return res
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


def _get_expected_columns(exp_path: Path) -> dict:
    with ipc.open_file(str(exp_path)) as reader:
        return {"ok": True, "columns": list(reader.schema.names)}


def _probe_column_causes(path: Path, exp_path: Path) -> list[str]:
    """Probe each top-level struct column individually so every failure or mismatch cause is reported."""
    col_names_res = _run_isolated(lambda: _get_struct_columns(path))
    if not col_names_res.get("ok"):
        return []
    columns = col_names_res.get("columns", [])
    if not columns:
        return []

    exp_names_res = _run_isolated(lambda: _get_expected_columns(exp_path))
    if not exp_names_res.get("ok"):
        return []
    expected_names = exp_names_res.get("columns", [])

    causes: set[str] = set()
    if columns != expected_names:
        causes.add(f"column names mismatch: actual={columns}, expected={expected_names}")

    if len(columns) > 1 or causes:
        for col in columns:
            res = _run_isolated(lambda c=col: _read_single_column(path, exp_path, c))
            if not res.get("ok"):
                for cause in res.get("causes") or ([res["error"]] if res.get("error") else []):
                    if f"column {col}" not in cause:
                        cause = f"column {col}: {cause}"
                    causes.add(cause)
    return sorted(causes)


def _get_struct_columns(path: Path) -> dict:
    vf = vortex.open(str(path))
    dtype = vf.dtype
    names_attr = getattr(dtype, "names", None)
    if names_attr is None:
        return {"ok": True, "columns": []}
    names = list(names_attr() if callable(names_attr) else names_attr)
    return {"ok": True, "columns": names}


def _read_single_column(path: Path, exp_path: Path, col: str) -> dict:
    vf = vortex.open(str(path))
    actual_arr = vf.scan(projection=[col]).read_all().to_arrow_array()
    if isinstance(actual_arr, pa.ChunkedArray):
        actual_arr = actual_arr.combine_chunks()
    if pa.types.is_struct(actual_arr.type):
        actual_field = actual_arr.type.field(col)
        actual_col = actual_arr.field(col)
    else:
        actual_field = pa.field(col, actual_arr.type, nullable=True)
        actual_col = actual_arr
    with ipc.open_file(str(exp_path)) as reader:
        expected_table = reader.read_all()
    if col not in expected_table.schema.names:
        return {"ok": False, "error": f"unexpected column {col} missing from expected Arrow IPC"}
    expected_field = expected_table.schema.field(col)
    expected_col = expected_table.column(col).combine_chunks()
    causes = _compare_column_causes(col, actual_field, actual_col, expected_field, expected_col)
    if causes:
        return {"ok": False, "error": "; ".join(causes), "causes": causes}
    return {"ok": True}


def _strip_error_wrapper(msg: str) -> str:
    """Strip outer Python/signal wrappers so full-table and projected errors can be matched."""
    for prefix in (
        "ArrowInvalid: External error: ",
        "ArrowInvalid: ",
        "ValueError: ",
        "PanicException: ",
    ):
        if msg.startswith(prefix):
            return msg[len(prefix) :]
    if msg.startswith("signal ") and ": " in msg:
        return msg.split(": ", 1)[1]
    return msg


def _verify_file(path: Path, exp_path: Path) -> dict:
    def _do() -> dict:
        vf = vortex.open(str(path))
        actual = vf.to_arrow().read_all()
        with ipc.open_file(str(exp_path)) as reader:
            expected = reader.read_all()
        causes = _tables_equal(actual, expected)
        if not causes:
            return {"ok": True}
        return {"ok": False, "error": "; ".join(causes), "causes": causes}

    res = _run_isolated(_do)
    if res.get("ok"):
        return res

    err = res.get("error", "unknown error")
    full_causes = res.get("causes") or [err]
    if res.get("timed_out"):
        return {"ok": False, "error": err, "causes": full_causes}

    # Collect every distinct cause across columns so parallel scan races don't hide causes,
    # while preserving any full-table error not explained by single-column projections.
    col_causes = _probe_column_causes(path, exp_path)
    if col_causes:
        causes = set(col_causes)
        col_core_msgs = {
            _strip_error_wrapper(c.split(": ", 1)[1])
            if c.startswith("column ") and ": " in c
            else _strip_error_wrapper(c)
            for c in col_causes
        }
        for cause in full_causes:
            core = _strip_error_wrapper(cause)
            explained = (
                cause in causes
                or any(col_core and col_core in core for col_core in col_core_msgs)
                or (
                    cause.startswith("column count mismatch:")
                    and any(c.startswith("column names mismatch:") for c in col_causes)
                )
            )
            if not explained:
                causes.add(cause)
        sorted_causes = sorted(causes)
        return {
            "ok": False,
            "error": "; ".join(sorted_causes),
            "causes": sorted_causes,
        }
    return {"ok": False, "error": err, "causes": full_causes}


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
