# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Render declarative benchmark coverage as GitHub Actions matrices."""

from ..config import SCAN_COMBINATIONS, Benchmark, BenchmarkTarget, Engine, Format, Integration, ScanEngine, SizeFormat
from .model import BenchmarkCase, Catalog, Coverage, ScanCoverage, Storage
from .targets import TargetSet
from .validate import validate_catalog

_NOT_GENERATED = frozenset({Format.LANCE})
# vortex-bench's `S3_ROW_GROUP_ROWS`: parquet-rs's default.
S3_ROW_GROUP_ROWS = 1024 * 1024
_FORMAT_ORDER = (
    Format.PARQUET,
    Format.VORTEX,
    Format.VORTEX_COMPACT,
    Format.VORTEX_SPATIAL_NATIVE,
    Format.DUCKDB,
    Format.LANCE,
)


def _data_formats(coverage: Coverage, targets: TargetSet) -> list[Format]:
    if coverage.data_formats is not None:
        return list(coverage.data_formats)

    present = {target.format for target in targets}
    return [fmt for fmt in _FORMAT_ORDER if fmt in present and fmt not in _NOT_GENERATED]


def merge_coverage(benchmark: Benchmark, coverages: list[Coverage]) -> Coverage:
    """Union several coverages of `benchmark`: every resolved target and every generated format."""
    targets = TargetSet()
    formats: set[Format] = set()
    size_formats: set[SizeFormat] = set()
    for coverage in coverages:
        resolved = coverage.resolved_targets(benchmark)
        targets = targets | resolved
        formats.update(_data_formats(coverage, resolved))
        size_formats.update(coverage.size_formats)
    all_scans = [c for c in coverages if c.scans is ScanCoverage.ALL]
    scans = ScanCoverage.ALL if all_scans else ScanCoverage.DEFAULT
    return Coverage(
        targets,
        data_formats=tuple(fmt for fmt in _FORMAT_ORDER if fmt in formats),
        scans=scans,
        size_formats=tuple(fmt for fmt in SizeFormat if fmt in size_formats),
        # Skip a combination only when every coverage that adds scans skips it.
        skip_scans=tuple(
            combination
            for combination in SCAN_COMBINATIONS
            if all_scans and all(combination in c.skip_scans for c in all_scans)
        ),
    )


ScanGroup = tuple[Engine, Integration, ScanEngine]


def _scan_group(target: BenchmarkTarget, benchmark: Benchmark) -> ScanGroup | None:
    """Return the scan a target runs, or `None` for the default group: every target that keeps its
    bare format identity, which is Parquet and the like and the file integration on the suite's
    default scan engine."""
    resolved = target.resolved(benchmark)
    if not resolved.is_vortex or resolved.result_format("", benchmark) == "":
        return None
    assert resolved.scan is not None
    return (resolved.engine, resolved.integration, resolved.scan)


def _matrix_entries(benchmark: BenchmarkCase, coverage: Coverage) -> list[dict[str, object]]:
    """Render one case's coverage, one entry per scan when it covers every scan.

    Each entry runs on a machine of its own, so the extra scans add machines rather than wall time,
    and a failing scan cannot take another entry's results with it. The default group keeps the
    case's id, name and history; the other scans are named `{id}-{engine}-{integration}-{scan}`.
    Every entry of a case shares its `group`, which the PR comment collects into one table.
    """
    targets = coverage.resolved_targets(benchmark.benchmark)
    groups: dict[ScanGroup | None, list[BenchmarkTarget]] = {}
    for target in targets:
        group = _scan_group(target, benchmark.benchmark) if coverage.scans is ScanCoverage.ALL else None
        groups.setdefault(group, []).append(target)

    entries = []
    for group, group_targets in groups.items():
        entry = _matrix_entry(benchmark, coverage, TargetSet(tuple(group_targets)), group)
        entries.append(entry)
    return entries


def _matrix_entry(
    benchmark: BenchmarkCase, coverage: Coverage, targets: TargetSet, group: ScanGroup | None
) -> dict[str, object]:
    suffix = "" if group is None else "-" + "-".join(axis.value for axis in group)
    if group is None:
        data_formats = _data_formats(coverage, targets)
    else:
        # A scan's entry generates only the formats its targets read.
        data_formats = _data_formats(Coverage(targets), targets)
    entry: dict[str, object] = {
        "id": benchmark.id + suffix,
        "group": benchmark.id,
        "group_name": benchmark.name,
        "subcommand": benchmark.subcommand,
        "name": benchmark.name if group is None else f"{benchmark.name} ({' '.join(a.value for a in group)})",
        "dataset_dir": benchmark.data_dir_name,
        "targets": [target.to_dict() for target in targets],
        "data_formats": [fmt.value for fmt in data_formats],
        # File sizes and trees describe the data, which every entry of a case generates alike, so
        # only the default entry records them.
        "records_data": group is None,
        # Formats the default entry writes after its queries, only for their file sizes.
        "size_formats": [fmt.value for fmt in coverage.size_formats] if group is None else [],
    }
    if benchmark.scale_factor is not None:
        entry["scale_factor"] = str(benchmark.scale_factor)
    if benchmark.iterations is not None:
        entry["iterations"] = str(benchmark.iterations)
    if benchmark.local_dir is not None:
        entry["local_dir"] = benchmark.local_dir
    if benchmark.remote_key is not None:
        # A distinct key per entry: each S3 entry clears and re-uploads its own copy of the data.
        entry["remote_key"] = benchmark.remote_key + suffix
    if benchmark.storage is Storage.S3:
        # Generated Parquet that queries read from S3 has larger row groups than the default, so
        # it needs fewer requests to object storage. Suites that query published files ignore it.
        entry["row_group_rows"] = str(S3_ROW_GROUP_ROWS)
    return entry


def render_matrix(preset: str, catalog: Catalog) -> list[dict[str, object]]:
    """Render one named catalog preset as GitHub Actions matrix entries."""
    if preset not in catalog.presets:
        known = ", ".join(catalog.presets)
        raise ValueError(f"Unknown matrix preset {preset!r}. Available: {known}")

    validate_catalog(catalog)
    entries = [
        entry
        for benchmark in catalog.benchmarks
        if preset in benchmark.runs
        for entry in _matrix_entries(benchmark, benchmark.runs[preset])
    ]
    ids = [entry["id"] for entry in entries]
    duplicates = sorted({entry_id for entry_id in ids if ids.count(entry_id) > 1})
    if duplicates:
        raise ValueError(f"Preset {preset!r} renders duplicate matrix entry IDs: {duplicates}")
    return entries
