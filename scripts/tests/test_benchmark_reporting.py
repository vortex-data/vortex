# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
COMPARE_SCRIPT = REPO_ROOT / "scripts" / "compare-benchmark-jsons.py"
CAPTURE_SCRIPT = REPO_ROOT / "scripts" / "capture-file-sizes.py"
RANDOM_ACCESS_SPLIT_SCRIPT = REPO_ROOT / "scripts" / "random-access-split.py"
FETCH_BASELINE_SCRIPT = REPO_ROOT / "scripts" / "fetch-benchmark-baseline.py"


def load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_compare_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("compare_benchmark_jsons", COMPARE_SCRIPT)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_random_access_split_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("random_access_split", RANDOM_ACCESS_SPLIT_SCRIPT)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_random_access_merge_preserves_each_open_mode(tmp_path: Path) -> None:
    split = load_random_access_split_module()
    parts = tmp_path / "parts"
    parts.mkdir()
    cached = {
        "kind": "random_access_time",
        "dataset": "taxi",
        "format": "parquet",
        "open_mode": "cached",
    }
    reopen = cached | {"open_mode": "reopen"}
    (parts / "0.ingest.jsonl").write_text(f"{json.dumps(cached)}\n", encoding="utf-8")
    (parts / "1.ingest.jsonl").write_text(
        f"{json.dumps(cached)}\n{json.dumps(reopen)}\n",
        encoding="utf-8",
    )
    output = tmp_path / "results.ingest.jsonl"

    split.merge(str(parts / "*.ingest.jsonl"), split.ingest_identity, str(output))

    records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert records == [cached, reopen]


def timing_row(name: str, base: int, pr: int) -> dict[str, object]:
    return {
        "name": name,
        "hot_value_base": base,
        "hot_value_pr": pr,
        "hot_runtimes_base": [base, base, base],
        "hot_runtimes_pr": [pr, pr, pr],
    }


def stored_timing_row(
    commit: str,
    name: str,
    value: int,
    storage: str | None = None,
    dataset: dict[str, object] | None = None,
    engine: str = "datafusion",
    file_format: str = "parquet",
    cold: int | None = None,
) -> dict[str, object]:
    """Build a runner result row whose first run is the cold one.

    `value` stays the median across all runs, which is what the benchmark binary
    writes, so the reporter has to derive the hot median from `all_runtimes`.
    """

    row: dict[str, object] = {
        "name": name,
        "unit": "ns",
        "value": value,
        "all_runtimes": [value if cold is None else cold, value, value],
        "commit_id": commit,
        "target": {"engine": engine, "format": file_format},
    }
    if storage is not None:
        row["storage"] = storage
    if dataset is not None:
        row["dataset"] = dataset
    return row


def query_record(commit: str, query_idx: int, engine: str, file_format: str, value: int) -> dict[str, object]:
    return {
        "kind": "query_measurement",
        "commit_sha": commit,
        "dataset": "tpch",
        "dataset_variant": None,
        "scale_factor": "1",
        "query_idx": query_idx,
        "storage": "nvme",
        "engine": engine,
        "format": file_format,
        "value_ns": value,
        "all_runtimes_ns": [value, value, value],
    }


def random_access_record(commit: str, file_format: str, open_mode: str, value: int) -> dict[str, object]:
    return {
        "kind": "random_access_time",
        "commit_sha": commit,
        "dataset": "taxi/uniform",
        "format": file_format,
        "open_mode": open_mode,
        "value_ns": value,
        "all_runtimes_ns": [value, value, value],
    }


def stored_custom_row(
    commit: str,
    name: str,
    unit: str,
    value: float,
    engine: str = "vortex",
    file_format: str = "vortex",
) -> dict[str, object]:
    return {
        "name": name,
        "unit": unit,
        "value": value,
        "commit_id": commit,
        "target": {"engine": engine, "format": file_format},
    }


def git_repo_with_one_commit(path: Path) -> Path:
    """Create a throwaway repository the reporter can walk.

    The reporter resolves its baseline from `git rev-list HEAD`, so the tests
    need a repository of their own rather than whatever checkout they happen to
    run inside — a contributor's shallow clone must not change the result.
    """

    path.mkdir(parents=True)
    identity = ["-c", "user.name=Vortex Test", "-c", "user.email=test@example.com"]
    subprocess.run(["git", "init", "--quiet"], cwd=path, check=True)
    (path / "README.md").write_text("benchmark reporting fixture\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=path, check=True)
    subprocess.run(["git", *identity, "commit", "--quiet", "-m", "fixture"], cwd=path, check=True)
    return path


def render_report(
    tmp_path: Path,
    base_rows: list[dict[str, object]],
    pr_rows: list[dict[str, object]],
    benchmark_name: str,
    pr_records: list[dict[str, object]] | None = None,
) -> str:
    base_path = tmp_path / "base.jsonl"
    pr_path = tmp_path / "pr.jsonl"
    base_path.write_text("".join(f"{json.dumps(row)}\n" for row in base_rows), encoding="utf-8")
    pr_path.write_text("".join(f"{json.dumps(row)}\n" for row in pr_rows), encoding="utf-8")
    command = [sys.executable, str(COMPARE_SCRIPT), str(base_path), str(pr_path), benchmark_name]
    if pr_records is not None:
        records_path = tmp_path / "pr.ingest.jsonl"
        records_path.write_text("".join(f"{json.dumps(row)}\n" for row in pr_records), encoding="utf-8")
        command += ["--pr-records", str(records_path)]

    result = subprocess.run(command, check=False, capture_output=True, text=True)

    assert result.returncode == 0, result.stderr
    return result.stdout


def markdown_row(report: str, name: str) -> list[str]:
    line = next(line for line in report.splitlines() if line.startswith(f"| {name} "))
    return [cell.strip() for cell in line.strip("|").split("|")]


def test_within_engine_analysis_uses_each_engines_own_parquet_control() -> None:
    compare = load_compare_module()
    rows = [
        timing_row("tpch_q01/datafusion:parquet", 100, 200),
        timing_row("tpch_q01/datafusion:vortex-file-compressed", 100, 180),
        timing_row("tpch_q01/duckdb:parquet", 100, 100),
        timing_row("tpch_q01/duckdb:vortex-file-compressed", 100, 120),
    ]
    df = pd.DataFrame(rows)
    df[["engine", "file_format", "query"]] = df["name"].apply(compare.extract_target_fields)

    analyses = compare.build_within_engine_statistical_analyses(df, threshold_pct=5)

    assert set(analyses) == {"datafusion", "duckdb"}
    assert compare.build_verdict(analyses["datafusion"])["impact"] == "-10.0%"
    assert compare.build_verdict(analyses["duckdb"])["impact"] == "+20.0%"


def test_random_access_attribution_excludes_competitor_rows() -> None:
    compare = load_compare_module()
    rows = [
        timing_row("random-access/taxi/correlated/parquet-tokio-local-disk", 100, 110),
        timing_row("random-access/taxi/correlated/vortex-tokio-local-disk", 100, 99),
        timing_row("random-access/taxi/correlated/lance-tokio-local-disk", 100, 200),
        timing_row("random-access/taxi/correlated/litdata-tokio-local-disk", 100, 300),
        timing_row("random-access/taxi/uniform/parquet-tokio-local-disk", 100, 110),
        timing_row("random-access/taxi/uniform/vortex-tokio-local-disk", 100, 99),
        timing_row("random-access/taxi/uniform/lance-tokio-local-disk", 100, 200),
        timing_row("random-access/taxi/uniform/litdata-tokio-local-disk", 100, 300),
    ]
    df = pd.DataFrame(rows)
    df[["engine", "file_format", "query"]] = df["name"].apply(compare.extract_target_fields)

    analyses = compare.build_within_engine_statistical_analyses(df, threshold_pct=5)

    assert df.loc[df["file_format"].isin(["lance", "litdata"]), "query"].isna().all()
    assert set(analyses) == {"random-access"}
    assert set(analyses["random-access"]["detail_df"]["file_format"]) == {
        "parquet",
        "vortex-file-compressed",
    }
    assert compare.build_verdict(analyses["random-access"])["status"] == "Likely improvement"


def test_random_access_report_keeps_competitor_details_without_attribution(tmp_path: Path) -> None:
    names_and_values = [
        ("random-access/taxi/correlated/parquet-tokio-local-disk", 110),
        ("random-access/taxi/correlated/vortex-tokio-local-disk", 99),
        ("random-access/taxi/correlated/lance-tokio-local-disk", 200),
        ("random-access/taxi/correlated/litdata-tokio-local-disk", 300),
        ("random-access/taxi/uniform/parquet-tokio-local-disk", 110),
        ("random-access/taxi/uniform/vortex-tokio-local-disk", 99),
        ("random-access/taxi/uniform/lance-tokio-local-disk", 200),
        ("random-access/taxi/uniform/litdata-tokio-local-disk", 300),
    ]
    base_rows = [stored_timing_row("base-sha", name, 100) for name, _pr_value in names_and_values]
    pr_rows = [stored_timing_row("pr-sha", name, pr_value) for name, pr_value in names_and_values]

    report = render_report(tmp_path, base_rows, pr_rows, "Random Access")

    assert "**Attributed Vortex impact**: -10.0%" in report
    assert "<summary>random-access / lance / ns " in report
    assert "<summary>random-access / litdata / ns " in report


def test_comparison_report_groups_by_target_and_unit(tmp_path: Path) -> None:
    base_rows = [
        stored_custom_row("base-sha", "timing/fixture", "ms", 12.5),
        stored_custom_row("base-sha", "size/fixture", "%", 45.25),
        stored_custom_row("base-sha", "ratio/fixture", "ratio", 0.75),
        stored_custom_row("base-sha", "parquet timing/fixture", "ms", 20.0, file_format="parquet"),
    ]
    pr_rows = [
        stored_custom_row("pr-sha", "timing/fixture", "ms", 11.75),
        stored_custom_row("pr-sha", "new timing/fixture", "ms", 3.125),
        stored_custom_row("pr-sha", "size/fixture", "%", 44.5),
        stored_custom_row("pr-sha", "ratio/fixture", "ratio", 0.8),
        stored_custom_row("pr-sha", "parquet timing/fixture", "ms", 21.0, file_format="parquet"),
    ]

    report = render_report(tmp_path, base_rows, pr_rows, "Mixed metrics")

    assert "unknown / unknown" not in report
    assert "How to read Verdict and Engines" not in report
    # These measurements report a single number, so there is no cold run to split out.
    assert "cold" not in report
    assert "hot" not in report
    assert "<summary>vortex / vortex-file-compressed / ms " in report
    assert "<summary>vortex / vortex-file-compressed / % " in report
    assert "<summary>vortex / vortex-file-compressed / ratio " in report
    assert "<summary>vortex / parquet / ms " in report
    assert markdown_row(report, "timing/fixture") == ["timing/fixture", "11.75 / 12.5 / -6.0%"]
    assert markdown_row(report, "size/fixture") == ["size/fixture", "44.5 / 45.25 / -1.7%"]
    assert markdown_row(report, "ratio/fixture") == ["ratio/fixture", "0.8 / 0.75 / +6.7%"]
    assert markdown_row(report, "new timing/fixture") == [
        "new timing/fixture",
        "3.125 / — / no baseline",
    ]


def test_comparison_report_handles_missing_benchmark_baseline(tmp_path: Path) -> None:
    pr_rows = [stored_custom_row("pr-sha", "new timing/fixture", "ms", 3.125)]

    report = render_report(tmp_path, [], pr_rows, "New benchmark")

    assert "No baseline is available for this benchmark yet" in report
    assert "vs base `none`" in report
    assert markdown_row(report, "new timing/fixture") == [
        "new timing/fixture",
        "3.125 / — / no baseline",
    ]


def test_comparison_report_handles_mixed_query_types_in_baseline(tmp_path: Path) -> None:
    base_rows = [
        stored_custom_row("base-sha", "random-access/fixture", "ms", 12.5),
        stored_timing_row("base-sha", "tpch_q01/datafusion:parquet", 100),
    ]
    pr_rows = [stored_custom_row("pr-sha", "random-access/fixture", "ms", 13.0)]

    report = render_report(tmp_path, base_rows, pr_rows, "Random Access")

    assert markdown_row(report, "random-access/fixture") == [
        "random-access/fixture",
        "13 / 12.5 / +4.0%",
    ]


def test_comparison_report_retains_sql_analysis(tmp_path: Path) -> None:
    targets = [
        ("parquet", "parquet", 100, 105),
        ("vortex-file-compressed", "vortex", 80, 70),
    ]
    base_rows = [
        stored_timing_row("base-sha", f"tpch_q01/datafusion:{name}", base, file_format=target)
        for name, target, base, _pr in targets
    ]
    pr_rows = [
        stored_timing_row("pr-sha", f"tpch_q01/datafusion:{name}", pr, file_format=target)
        for name, target, _base, pr in targets
    ]

    report = render_report(tmp_path, base_rows, pr_rows, "TPC-H")

    assert "**Verdict**:" in report
    assert "**Vortex (geomean)**: hot " in report
    assert "**Parquet (geomean)**: hot " in report
    assert "How to read Verdict and Engines" in report
    assert "<summary>datafusion / vortex-file-compressed / ns " in report
    assert "<summary>datafusion / parquet / ns " in report
    assert "unknown / unknown" not in report


def test_large_hot_medians_render_without_scientific_notation() -> None:
    compare = load_compare_module()

    # A median over two runs can land on a half nanosecond; at this magnitude "%g"
    # would switch to scientific notation and stop matching the baseline column.
    assert compare.format_measurement_value(1478968300.5) == "1478968300"
    assert compare.format_measurement_value(8228437.5) == "8228438"
    assert compare.format_measurement_value(12.5) == "12.5"
    assert compare.format_measurement_value(100) == "100"


def test_cold_and_hot_runtimes_split_the_first_run_from_the_rest() -> None:
    compare = load_compare_module()

    assert compare.cold_runtime([7, 9, 11]) == 7
    assert compare.hot_runtimes([7, 9, 11]) == [9, 11]
    assert compare.hot_runtime([7, 9, 11], 9) == 10


def test_hot_runtime_falls_back_to_the_reported_value_without_later_runs() -> None:
    compare = load_compare_module()

    assert math.isnan(compare.cold_runtime(None))
    assert compare.hot_runtimes(None) == []
    assert compare.hot_runtime(None, 42) == 42
    assert compare.hot_runtime([7], 7) == 7
    assert math.isnan(compare.hot_runtime(None, None))


def test_only_changes_past_the_suite_threshold_are_marked(tmp_path: Path) -> None:
    # TPC-H runs at a 10% threshold, so 9% is noise here and 11% is not.
    base_rows = [
        stored_timing_row("base-sha", "tpch_q01/datafusion:vortex-file-compressed", 100, file_format="vortex"),
        stored_timing_row("base-sha", "tpch_q02/datafusion:vortex-file-compressed", 100, file_format="vortex"),
        stored_timing_row("base-sha", "tpch_q03/datafusion:vortex-file-compressed", 100, file_format="vortex"),
    ]
    pr_rows = [
        stored_timing_row("pr-sha", "tpch_q01/datafusion:vortex-file-compressed", 109, file_format="vortex"),
        stored_timing_row("pr-sha", "tpch_q02/datafusion:vortex-file-compressed", 111, file_format="vortex"),
        stored_timing_row("pr-sha", "tpch_q03/datafusion:vortex-file-compressed", 89, file_format="vortex"),
    ]

    report = render_report(tmp_path, base_rows, pr_rows, "TPC-H")

    assert markdown_row(report, "tpch_q01/datafusion:vortex-file-compressed")[1] == "109 / 100 / +9.0%"
    assert markdown_row(report, "tpch_q02/datafusion:vortex-file-compressed")[1] == "111 / 100 / +11.0% 🔴"
    assert markdown_row(report, "tpch_q03/datafusion:vortex-file-compressed")[1] == "89 / 100 / -11.0% 🟢"


def test_report_splits_cold_and_hot_runs(tmp_path: Path) -> None:
    base_rows = [
        stored_timing_row("base-sha", "tpch_q01/datafusion:parquet", 100, cold=1000),
        stored_timing_row(
            "base-sha",
            "tpch_q01/datafusion:vortex-file-compressed",
            100,
            file_format="vortex",
            cold=1000,
        ),
    ]
    pr_rows = [
        stored_timing_row("pr-sha", "tpch_q01/datafusion:parquet", 100, cold=1000),
        stored_timing_row(
            "pr-sha",
            "tpch_q01/datafusion:vortex-file-compressed",
            50,
            file_format="vortex",
            cold=2000,
        ),
    ]

    report = render_report(tmp_path, base_rows, pr_rows, "TPC-H")

    # The marks sit on the numbers that moved, not on the benchmark name.
    assert markdown_row(report, "tpch_q01/datafusion:vortex-file-compressed") == [
        "tpch_q01/datafusion:vortex-file-compressed",
        "50 / 100 / -50.0% 🟢",
        "2000 / 1000 / +100.0% 🔴",
        "0.03 / 0.10 / -75.0%",
    ]
    assert "**Attributed Vortex impact**: -50.0%" in report
    assert "**Vortex (geomean)**: hot 0.500x ✅ · cold 2.000x ❌" in report
    assert "**Parquet (geomean)**: hot 1.000x ➖ · cold 1.000x ➖" in report
    assert "**Commits**: PR `pr-sha` vs base " in report


def test_verdict_ignores_a_cold_start_only_regression(tmp_path: Path) -> None:
    base_rows = [
        stored_timing_row("base-sha", "tpch_q01/datafusion:parquet", 100, cold=100),
        stored_timing_row(
            "base-sha",
            "tpch_q01/datafusion:vortex-file-compressed",
            100,
            file_format="vortex",
            cold=100,
        ),
    ]
    pr_rows = [
        stored_timing_row("pr-sha", "tpch_q01/datafusion:parquet", 100, cold=100),
        stored_timing_row(
            "pr-sha",
            "tpch_q01/datafusion:vortex-file-compressed",
            100,
            file_format="vortex",
            cold=400,
        ),
    ]

    report = render_report(tmp_path, base_rows, pr_rows, "TPC-H")

    assert "**Attributed Vortex impact**: +0.0%" in report
    assert "**Vortex (geomean)**: hot 1.000x ➖ · cold 4.000x ❌" in report
    assert markdown_row(report, "tpch_q01/datafusion:vortex-file-compressed") == [
        "tpch_q01/datafusion:vortex-file-compressed",
        "100 / 100 / +0.0%",
        "400 / 100 / +300.0% 🔴",
        "0.25 / 1.00 / -75.0%",
    ]


def file_size_record(commit: str, size: int) -> dict[str, object]:
    return file_size_record_for(commit, size, "tpch", "10", "vortex-file-compressed", "part-0.vortex")


def file_size_record_for(
    commit: str,
    size: int,
    benchmark: str,
    scale_factor: str,
    file_format: str,
    file_name: str,
) -> dict[str, object]:
    return {
        "metric": "file_size",
        "unit": "bytes",
        "value": size,
        "commit_id": commit,
        "file_size": {
            "benchmark": benchmark,
            "scale_factor": scale_factor,
            "format": file_format,
            "file": file_name,
        },
    }


def test_file_size_report_reads_shared_benchmark_rows() -> None:
    compare = load_compare_module()

    report = compare.format_file_size_report(
        pd.DataFrame([file_size_record("base-sha", 100)]),
        pd.DataFrame([file_size_record("pr-sha", 125)]),
    )

    assert "<summary>File Size Changes (1 files changed, +25.0% overall, 1↑ 0↓)</summary>" in report
    assert "| part-0.vortex | 10 | vortex-file-compressed | 100 B | 125 B | +25 B | +25.0% |" in report


def test_file_size_report_ignores_file_identities_with_a_zero_byte_side() -> None:
    compare = load_compare_module()

    report = compare.format_file_size_report(
        pd.DataFrame(
            [
                file_size_record_for("base-sha", 100, "tpch", "10", "vortex-file-compressed", "head-empty"),
                file_size_record_for("base-sha", 0, "tpch", "10", "vortex-file-compressed", "base-empty"),
                file_size_record_for("base-sha", 100, "tpch", "10", "vortex-file-compressed", "changed"),
            ]
        ),
        pd.DataFrame(
            [
                file_size_record_for("pr-sha", 0, "tpch", "10", "vortex-file-compressed", "head-empty"),
                file_size_record_for("pr-sha", 125, "tpch", "10", "vortex-file-compressed", "base-empty"),
                file_size_record_for("pr-sha", 125, "tpch", "10", "vortex-file-compressed", "changed"),
            ]
        ),
    )

    assert "<summary>File Size Changes (1 files changed, +25.0% overall, 1↑ 0↓)</summary>" in report
    assert "head-empty" not in report
    assert "base-empty" not in report


def test_file_size_report_ignores_baseline_rows_outside_pr_scope() -> None:
    compare = load_compare_module()

    report = compare.format_file_size_report(
        pd.DataFrame(
            [
                file_size_record_for("base-sha", 100, "tpch", "10.0", "vortex-file-compressed", "part-0.vortex"),
                file_size_record_for("base-sha", 200, "tpch", "1.0", "vortex-file-compressed", "part-0.vortex"),
                file_size_record_for("base-sha", 300, "clickbench", "1.0", "vortex-compact", "hits_0.vortex"),
            ]
        ),
        pd.DataFrame(
            [
                file_size_record_for("pr-sha", 125, "tpch", "10.0", "vortex-file-compressed", "part-0.vortex"),
            ]
        ),
    )

    assert "<summary>File Size Changes (1 files changed, +25.0% overall, 1↑ 0↓)</summary>" in report
    assert "hits_0.vortex" not in report
    assert "| part-0.vortex | 1.0 |" not in report


def test_file_size_report_omits_formats_the_pr_run_skipped() -> None:
    compare = load_compare_module()

    report = compare.format_file_size_report(
        pd.DataFrame(
            [
                file_size_record_for("base-sha", 100, "tpch", "10", "vortex-file-compressed", "part-0.vortex"),
                file_size_record_for("base-sha", 80, "tpch", "10", "vortex-compact", "part-0.vortex"),
                file_size_record_for("base-sha", 5, "tpch", "10", "vortex-compact", "duckdb.db"),
            ]
        ),
        pd.DataFrame(
            [
                file_size_record_for("pr-sha", 125, "tpch", "10", "vortex-file-compressed", "part-0.vortex"),
            ]
        ),
    )

    assert "<summary>File Size Changes (1 files changed, +25.0% overall, 1↑ 0↓)</summary>" in report
    assert "vortex-compact" not in report
    assert "-100.0%" not in report


def test_capture_file_sizes_emits_shared_benchmark_rows(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    format_dir = data_dir / "tpch" / "10" / "vortex-file-compressed"
    format_dir.mkdir(parents=True)
    (format_dir / "part-0.vortex").write_bytes(b"x" * 42)
    output_path = tmp_path / "sizes.jsonl"

    result = subprocess.run(
        [
            sys.executable,
            str(CAPTURE_SCRIPT),
            str(data_dir),
            "--benchmark",
            "tpch",
            "--dataset-dir",
            "tpch",
            "--commit",
            "deadbeef",
            "-o",
            str(output_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    records = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert records == [
        {
            "metric": "file_size",
            "unit": "bytes",
            "value": 42,
            "commit_id": "deadbeef",
            "file_size": {
                "benchmark": "tpch",
                "scale_factor": "10",
                "format": "vortex-file-compressed",
                "file": "part-0.vortex",
            },
        }
    ]


def test_baseline_fetch_refuses_a_shallow_checkout(tmp_path: Path) -> None:
    """A shallow clone must fail loudly, not render a baseline-free report.

    `git rev-list HEAD` reaches only the tip of a shallow clone, so every
    recorded commit looks unreachable. Reporting that as "no baseline yet"
    hides a real comparison behind a report that merely looks new.
    """

    origin = git_repo_with_one_commit(tmp_path / "origin")
    shallow = tmp_path / "shallow"
    subprocess.run(
        ["git", "clone", "--quiet", "--depth", "1", "--no-local", f"file://{origin}", str(shallow)],
        check=True,
        capture_output=True,
    )
    pr_path = tmp_path / "pr.ingest.jsonl"
    pr_path.write_text(f"{json.dumps(query_record('pr', 1, 'datafusion', 'parquet', 105))}\n", encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(FETCH_BASELINE_SCRIPT), str(pr_path), str(tmp_path / "base.jsonl"), "--postgres", ""],
        cwd=shallow,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "shallow checkout" in result.stderr
    assert "fetch-depth: 0" in result.stderr


def test_baseline_fetch_groups_distinct_pr_identities_by_kind() -> None:
    fetch = load_module("fetch_benchmark_baseline", FETCH_BASELINE_SCRIPT)
    records = [
        query_record("pr", 1, "datafusion", "parquet", 100),
        query_record("pr", 1, "datafusion", "parquet", 100),
        random_access_record("pr", "parquet", "cached", 100),
        {"kind": "vector_search_run", "dataset": "cohere"},
    ]

    assert fetch.identities_by_kind(records) == {
        "query_measurement": [("tpch", None, "1", 1, "nvme", "datafusion", "parquet")],
        "random_access_time": [("taxi/uniform", "parquet", "cached")],
    }


def test_baseline_fetch_matches_absent_dims_as_equal() -> None:
    fetch = load_module("fetch_benchmark_baseline", FETCH_BASELINE_SCRIPT)

    join, arrays = fetch._identity_join("compression_size", [("taxi", None, "parquet"), ("taxi", "gpu", "lance")])

    assert "unnest(%s::text[], %s::text[], %s::text[]) AS p(dataset, dataset_variant, format)" in join
    assert "t.dataset_variant IS NOT DISTINCT FROM p.dataset_variant" in join
    assert arrays == [["taxi", "taxi"], [None, "gpu"], ["parquet", "lance"]]


def test_report_compares_ingest_records(tmp_path: Path) -> None:
    targets = [("parquet", 100, 105), ("vortex-file-compressed", 80, 70)]
    base_records = [query_record("base-sha", 1, "datafusion", target, base) for target, base, _pr in targets]
    base_records += [random_access_record("base-sha", target, "cached", base) for target, base, _pr in targets]
    pr_records = [query_record("pr-sha", 1, "datafusion", target, pr) for target, _base, pr in targets]
    pr_records += [random_access_record("pr-sha", target, "cached", pr) for target, _base, pr in targets]
    pr_records.append(random_access_record("pr-sha", "lance", "cached", 300))
    pr_results = [
        stored_timing_row("pr-sha", "ignored when records exist", 1) | {"doc": "benchmarks/tpch.md"},
        file_size_record("pr-sha", 125),
    ]

    report = render_report(tmp_path, base_records, pr_results, "TPC-H", pr_records)

    assert "vs base `base-sha`" in report
    assert "**Verdict**:" in report
    assert "**Engines**: DataFusion " in report
    assert "random-access " in report
    assert "ignored when records exist" not in report
    assert markdown_row(report, "tpch_q01/datafusion:vortex-file-compressed")[1] == "70 / 80 / -12.5% 🟢"
    assert markdown_row(report, "random-access/taxi/uniform/parquet-cached")[1] == "105 / 100 / +5.0%"
    assert markdown_row(report, "random-access/taxi/uniform/lance-cached")[1] == "300 / — / no baseline"
    assert "<summary>random-access / lance / ns " in report
    assert "File Size" not in report


def test_report_says_when_the_baseline_could_not_be_read(tmp_path: Path) -> None:
    pr_records = [query_record("pr-sha", 1, "datafusion", "parquet", 100)]
    records_path = tmp_path / "pr.ingest.jsonl"
    records_path.write_text(f"{json.dumps(pr_records[0])}\n", encoding="utf-8")
    results_path = tmp_path / "results.json"
    results_path.write_text(f"{json.dumps(stored_timing_row('pr-sha', 'unused', 1))}\n", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(COMPARE_SCRIPT),
            str(tmp_path / "missing.jsonl"),
            str(results_path),
            "TPC-H",
            "--pr-records",
            str(records_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "The baseline could not be read from the benchmarks database" in result.stdout
    assert "No baseline is available for this benchmark yet" not in result.stdout


def test_query_records_keep_scale_factors_apart() -> None:
    compare = load_compare_module()
    sf1 = compare.ingest_record_row(query_record("sha", 1, "duckdb", "parquet", 5))
    sf10 = compare.ingest_record_row(query_record("sha", 1, "duckdb", "parquet", 5) | {"scale_factor": "10"})

    assert sf1 is not None and sf10 is not None
    assert sf1["name"] == sf10["name"] == "tpch_q01/duckdb:parquet"
    assert compare.dataset_key(sf1["dataset"]) != compare.dataset_key(sf10["dataset"])


def test_report_converts_compression_records() -> None:
    compare = load_compare_module()
    common = {"commit_sha": "sha", "dataset": "taxi", "dataset_variant": None, "format": "parquet"}

    encode = compare.ingest_record_row(
        common | {"kind": "compression_time", "op": "encode", "value_ns": 5, "all_runtimes_ns": [6, 5]}
    )
    decode = compare.ingest_record_row(
        common
        | {
            "kind": "compression_time",
            "dataset_variant": "gpu",
            "op": "decode",
            "value_ns": 5,
            "all_runtimes_ns": [6, 5],
        }
    )
    size = compare.ingest_record_row(common | {"kind": "compression_size", "value_bytes": 10, "uncompressed_bytes": 20})

    assert encode is not None and decode is not None and size is not None
    assert encode["name"] == "compress time/taxi"
    assert decode["name"] == "decompress time/taxi/gpu"
    assert (size["name"], size["unit"], size["value"]) == ("parquet size/taxi", "bytes", 10)
    assert compare.ingest_record_row({"kind": "vector_search_run"}) is None


def test_summary_reports_hot_alone_when_no_row_recorded_its_runs(tmp_path: Path) -> None:
    # Random Access, String Encoding and Compression report one value per row, so
    # there is no cold geomean to sit beside the hot one.
    base_rows = [
        stored_custom_row("base-sha", "random-access/vortex-tokio-local-disk", "ns", 100.0),
        stored_custom_row("base-sha", "random-access/parquet-tokio-local-disk", "ns", 200.0, file_format="parquet"),
    ]
    pr_rows = [
        stored_custom_row("pr-sha", "random-access/vortex-tokio-local-disk", "ns", 50.0),
        stored_custom_row("pr-sha", "random-access/parquet-tokio-local-disk", "ns", 200.0, file_format="parquet"),
    ]

    report = render_report(tmp_path, base_rows, pr_rows, "Random Access")

    summary = report.splitlines()[2]
    assert "**Vortex (geomean)**: hot 0.500x ✅" in summary
    assert "**Parquet (geomean)**: hot 1.000x ➖" in summary
    assert "cold" not in summary


def test_capture_file_sizes_reads_only_the_named_directory(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    for name, size in (("clickbench_partitioned", 42), ("clickbench_partitioned_10", 7)):
        format_dir = data_dir / name / "vortex-file-compressed"
        format_dir.mkdir(parents=True)
        (format_dir / "hits_0.vortex").write_bytes(b"x" * size)
    output_path = tmp_path / "sizes.jsonl"

    subprocess.run(
        [
            sys.executable,
            str(CAPTURE_SCRIPT),
            str(data_dir),
            "--benchmark",
            "clickbench",
            "--dataset-dir",
            "clickbench_partitioned",
            "--commit",
            "deadbeef",
            "-o",
            str(output_path),
        ],
        check=True,
    )

    records = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert [record["value"] for record in records] == [42]
