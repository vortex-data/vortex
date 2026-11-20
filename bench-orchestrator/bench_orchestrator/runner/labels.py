# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Label benchmark binary output with the target's integration and scan engine.

The benchmark binaries label results by engine and format alone, so every Vortex target on the same
files would share one identity. Each result gets both axes as fields of their own (`integration`
and `scan_engine`), and its format identity names them too unless it is the file integration on
the suite's default engine, whose history keeps the bare format; see
`BenchmarkTarget.result_format`.
"""

import json
from pathlib import Path
from typing import Any

from ..config import Benchmark, BenchmarkTarget


def _axes(target: BenchmarkTarget, benchmark: Benchmark) -> dict[str, str]:
    resolved = target.resolved(benchmark)
    if not resolved.is_vortex or resolved.scan is None:
        return {}
    return {"integration": resolved.integration.value, "scan_engine": resolved.scan.value}


def _label_name(name: object, target: BenchmarkTarget, benchmark: Benchmark) -> object:
    # Names have the shape `{dataset}_qNN[_memory]/{engine}:{format}`.
    if isinstance(name, str) and "/" in name and ":" in name.rsplit("/", 1)[1]:
        prefix, name_format = name.rsplit(":", 1)
        return f"{prefix}:{target.result_format(name_format, benchmark)}"
    return name


def label_result_line(line: str, target: BenchmarkTarget, benchmark: Benchmark) -> str:
    """Label one results-JSON line (`--display-format gh-json`) for `target`."""
    axes = _axes(target, benchmark)
    if not axes or not line.lstrip().startswith("{"):
        return line
    record: dict[str, Any] = json.loads(line)

    record_target = record.get("target")
    if isinstance(record_target, dict) and isinstance(record_target.get("format"), str):
        record_target["format"] = target.result_format(record_target["format"], benchmark)
        record["name"] = _label_name(record.get("name"), target, benchmark)
        record.update(axes)

    return json.dumps(record, separators=(",", ":"))


def label_ingest_file(path: Path, target: BenchmarkTarget, benchmark: Benchmark) -> None:
    """Label the `--ingest-jsonl` query records in `path` for `target`, in place."""
    axes = _axes(target, benchmark)
    if not axes:
        return
    lines = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            if record.get("kind") == "query_measurement" and isinstance(record.get("format"), str):
                record["format"] = target.result_format(record["format"], benchmark)
                record.update(axes)
            line = json.dumps(record, separators=(",", ":"))
        lines.append(line)
    path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")
