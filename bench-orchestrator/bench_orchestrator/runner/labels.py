# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Relabel benchmark binary output with the target's integration.

The benchmark binaries label results by engine and format alone. Results of a non-default
integration, such as `datafusion:vortex [scan-api]`, would otherwise share the default target's
identity in result files, comparisons and the benchmarks database.
"""

import json
from pathlib import Path

from ..config import BenchmarkTarget, Integration


def relabel_result_line(line: str, target: BenchmarkTarget) -> str:
    """Relabel one results-JSON line (`--display-format gh-json`) for `target`."""
    if target.integration is Integration.DEFAULT or not line.lstrip().startswith("{"):
        return line
    record = json.loads(line)

    record_target = record.get("target")
    if isinstance(record_target, dict) and isinstance(record_target.get("format"), str):
        binary_format = record_target["format"]
        record_target["format"] = target.result_format(binary_format)
        # Names have the shape `{dataset}_qNN[_memory]/{engine}:{format}`, with the format as
        # the binary labels it, which can differ from the serialized `target.format`.
        name = record.get("name")
        if isinstance(name, str) and "/" in name and ":" in name.rsplit("/", 1)[1]:
            prefix, name_format = name.rsplit(":", 1)
            record["name"] = f"{prefix}:{target.result_format(name_format)}"

    return json.dumps(record, separators=(",", ":"))


def relabel_ingest_file(path: Path, target: BenchmarkTarget) -> None:
    """Relabel the `--ingest-jsonl` records in `path` for `target`, in place."""
    if target.integration is Integration.DEFAULT:
        return
    lines = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            if isinstance(record.get("format"), str):
                record["format"] = target.result_format(record["format"])
            line = json.dumps(record, separators=(",", ":"))
        lines.append(line)
    path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")
