# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
COMPARE_SCRIPT = REPO_ROOT / "scripts" / "compare-benchmark-jsons.py"
CAPTURE_SCRIPT = REPO_ROOT / "scripts" / "capture-file-sizes.py"


def load_compare_module():
    spec = importlib.util.spec_from_file_location("compare_benchmark_jsons", COMPARE_SCRIPT)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


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


def render_report(
    tmp_path: Path,
    base_rows: list[dict[str, object]],
    pr_rows: list[dict[str, object]],
    benchmark_name: str,
) -> str:
    head_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    base_rows = [row | {"commit_id": head_commit} for row in base_rows]
    base_path = tmp_path / "base.jsonl"
    pr_path = tmp_path / "pr.jsonl"
    base_path.write_text("".join(f"{json.dumps(row)}\n" for row in base_rows), encoding="utf-8")
    pr_path.write_text("".join(f"{json.dumps(row)}\n" for row in pr_rows), encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(COMPARE_SCRIPT), str(base_path), str(pr_path), benchmark_name],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    return result.stdout


def markdown_row(report: str, name: str) -> list[str]:
    line = next(line for line in report.splitlines() if line.startswith(f"| {name} "))
    return [cell.strip() for cell in line.strip("|").split("|")]


def test_select_latest_baseline_rows_uses_latest_matching_benchmark_commit() -> None:
    compare = load_compare_module()
    history = pd.DataFrame(
        [
            stored_timing_row(
                "base-old",
                "tpch_q01/datafusion:parquet",
                100,
                "nvme",
                {"scale_factor": "1.0"},
            ),
            file_size_record_for("base-old", 100, "tpch", "1.0", "vortex-file-compressed", "part-0.vortex"),
            stored_timing_row(
                "base-current",
                "tpch_q01/datafusion:parquet",
                110,
                "nvme",
                {"scale_factor": "1.0"},
            ),
            file_size_record_for("base-current", 120, "tpch", "1.0", "vortex-file-compressed", "part-0.vortex"),
            stored_timing_row("base-other", "clickbench_q01/datafusion:parquet", 200, "nvme"),
        ]
    )
    pr = pd.DataFrame(
        [
            stored_timing_row(
                "pr-sha",
                "tpch_q01/datafusion:parquet",
                115,
                "nvme",
                {"scale_factor": "1.0"},
            ),
        ]
    )

    selected = compare.select_latest_baseline_rows(history, pr, {"base-old", "base-current"})

    assert set(selected["commit_id"]) == {"base-current"}
    assert len(selected) == 2


def test_read_latest_baseline_rows_streams_latest_matching_benchmark_commit(tmp_path: Path) -> None:
    compare = load_compare_module()
    history_path = tmp_path / "history.jsonl"
    history_rows = [
        stored_timing_row(
            "base-old",
            "tpch_q01/datafusion:parquet",
            100,
            "nvme",
            {"scale_factor": "1.0"},
        ),
        file_size_record_for("base-old", 100, "tpch", "1.0", "vortex-file-compressed", "part-0.vortex"),
        stored_timing_row(
            "base-current",
            "tpch_q01/datafusion:parquet",
            110,
            "nvme",
            {"scale_factor": "1.0"},
        ),
        file_size_record_for("base-current", 120, "tpch", "1.0", "vortex-file-compressed", "part-0.vortex"),
        stored_timing_row(
            "base-wrong-unit",
            "tpch_q01/datafusion:parquet",
            111,
            "nvme",
            {"scale_factor": "1.0"},
        )
        | {"unit": "ms"},
        stored_timing_row(
            "base-wrong-target",
            "tpch_q01/datafusion:parquet",
            112,
            "nvme",
            {"scale_factor": "1.0"},
            file_format="vortex",
        ),
        stored_timing_row("base-other", "clickbench_q01/datafusion:parquet", 200, "nvme"),
    ]
    history_path.write_text(
        "".join(f"{json.dumps(row)}\n" for row in history_rows),
        encoding="utf-8",
    )
    pr = pd.DataFrame(
        [
            stored_timing_row(
                "pr-sha",
                "tpch_q01/datafusion:parquet",
                115,
                "nvme",
                {"scale_factor": "1.0"},
            ),
        ]
    )

    selected = compare.read_latest_baseline_rows(history_path, pr, {"base-old", "base-current"})

    assert set(selected["commit_id"]) == {"base-current"}
    assert len(selected) == 2


def test_read_latest_baseline_rows_skips_commits_outside_git_tree(tmp_path: Path) -> None:
    compare = load_compare_module()
    history_path = tmp_path / "history.jsonl"
    history_rows = [
        stored_timing_row("base-reachable", "tpch_q01/datafusion:parquet", 100),
        file_size_record_for(
            "base-reachable",
            100,
            "tpch",
            "1.0",
            "vortex-file-compressed",
            "part-0.vortex",
        ),
        stored_timing_row("base-off-tree", "tpch_q01/datafusion:parquet", 110),
        file_size_record_for(
            "base-off-tree",
            120,
            "tpch",
            "1.0",
            "vortex-file-compressed",
            "part-0.vortex",
        ),
    ]
    history_path.write_text(
        "".join(f"{json.dumps(row)}\n" for row in history_rows),
        encoding="utf-8",
    )
    pr = pd.DataFrame([stored_timing_row("pr", "tpch_q01/datafusion:parquet", 105)])

    selected = compare.read_latest_baseline_rows(history_path, pr, {"base-reachable"})

    assert set(selected["commit_id"]) == {"base-reachable"}
    assert len(selected) == 2


def test_read_latest_baseline_rows_returns_empty_without_reachable_match(tmp_path: Path) -> None:
    compare = load_compare_module()
    history_path = tmp_path / "history.jsonl"
    history_path.write_text(
        f"{json.dumps(stored_timing_row('base-off-tree', 'tpch_q01/datafusion:parquet', 100))}\n",
        encoding="utf-8",
    )
    pr = pd.DataFrame([stored_timing_row("pr", "tpch_q01/datafusion:parquet", 105)])

    selected = compare.read_latest_baseline_rows(history_path, pr, set())

    assert selected.empty


def test_read_latest_baseline_rows_uses_last_result_from_rerun(tmp_path: Path) -> None:
    compare = load_compare_module()
    history_path = tmp_path / "history.jsonl"
    history_path.write_text(
        "".join(
            f"{json.dumps(row)}\n"
            for row in [
                stored_timing_row("base", "tpch_q01/datafusion:parquet", 100),
                stored_timing_row("base", "tpch_q01/datafusion:parquet", 110),
            ]
        ),
        encoding="utf-8",
    )
    pr = pd.DataFrame([stored_timing_row("pr", "tpch_q01/datafusion:parquet", 105)])

    selected = compare.read_latest_baseline_rows(history_path, pr, {"base"})

    assert len(selected) == 1
    assert selected.iloc[0]["value"] == 110


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


def test_random_access_attribution_excludes_lance_rows() -> None:
    compare = load_compare_module()
    rows = [
        timing_row("random-access/taxi/correlated/parquet-tokio-local-disk", 100, 110),
        timing_row("random-access/taxi/correlated/vortex-tokio-local-disk", 100, 99),
        timing_row("random-access/taxi/correlated/lance-tokio-local-disk", 100, 200),
        timing_row("random-access/taxi/uniform/parquet-tokio-local-disk", 100, 110),
        timing_row("random-access/taxi/uniform/vortex-tokio-local-disk", 100, 99),
        timing_row("random-access/taxi/uniform/lance-tokio-local-disk", 100, 200),
    ]
    df = pd.DataFrame(rows)
    df[["engine", "file_format", "query"]] = df["name"].apply(compare.extract_target_fields)

    analyses = compare.build_within_engine_statistical_analyses(df, threshold_pct=5)

    assert df.loc[df["file_format"] == "lance", "query"].isna().all()
    assert set(analyses) == {"random-access"}
    assert set(analyses["random-access"]["detail_df"]["file_format"]) == {
        "parquet",
        "vortex-file-compressed",
    }
    assert compare.build_verdict(analyses["random-access"])["status"] == "Likely improvement"


def test_random_access_report_keeps_lance_details_without_attribution(tmp_path: Path) -> None:
    names_and_values = [
        ("random-access/taxi/correlated/parquet-tokio-local-disk", 110),
        ("random-access/taxi/correlated/vortex-tokio-local-disk", 99),
        ("random-access/taxi/correlated/lance-tokio-local-disk", 200),
        ("random-access/taxi/uniform/parquet-tokio-local-disk", 110),
        ("random-access/taxi/uniform/vortex-tokio-local-disk", 99),
        ("random-access/taxi/uniform/lance-tokio-local-disk", 200),
    ]
    base_rows = [stored_timing_row("base-sha", name, 100) for name, _pr_value in names_and_values]
    pr_rows = [stored_timing_row("pr-sha", name, pr_value) for name, pr_value in names_and_values]

    report = render_report(tmp_path, base_rows, pr_rows, "Random Access")

    assert "**Attributed Vortex impact**: -10.0%" in report
    assert "<summary>random-access / lance / ns " in report


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
    assert "<summary>vortex / vortex-file-compressed / ms " in report
    assert "<summary>vortex / vortex-file-compressed / % " in report
    assert "<summary>vortex / vortex-file-compressed / ratio " in report
    assert "<summary>vortex / parquet / ms " in report
    assert markdown_row(report, "timing/fixture") == ["timing/fixture", "11.75", "12.5", "0.94"]
    assert markdown_row(report, "size/fixture") == ["size/fixture", "44.5", "45.25", "0.98"]
    assert markdown_row(report, "ratio/fixture") == ["ratio/fixture", "0.8", "0.75", "1.07"]
    assert markdown_row(report, "new timing/fixture") == [
        "new timing/fixture",
        "3.125",
        "—",
        "no baseline",
    ]


def test_comparison_report_handles_missing_benchmark_baseline(tmp_path: Path) -> None:
    base_rows = [stored_custom_row("base-sha", "other timing/fixture", "ms", 12.5)]
    pr_rows = [stored_custom_row("pr-sha", "new timing/fixture", "ms", 3.125)]

    report = render_report(tmp_path, base_rows, pr_rows, "New benchmark")

    assert "No baseline is available for this benchmark yet" in report
    assert "base none (ms)" in report
    assert markdown_row(report, "new timing/fixture") == [
        "new timing/fixture",
        "3.125",
        "—",
        "no baseline",
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
        "13",
        "12.5",
        "1.04",
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
    assert "**Vortex (hot geomean)**:" in report
    assert "**Parquet (hot geomean)**:" in report
    assert "How to read Verdict and Engines" in report
    assert "<summary>datafusion / vortex-file-compressed / ns " in report
    assert "<summary>datafusion / parquet / ns " in report
    assert "unknown / unknown" not in report


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

    assert markdown_row(report, "tpch_q01/datafusion:vortex-file-compressed") == [
        "tpch_q01/datafusion:vortex-file-compressed 🚀",
        "50",
        "100",
        "0.50",
        "2000",
        "1000",
        "2.00",
    ]
    assert "**Attributed Vortex impact**: -50.0%" in report
    assert "**Cold run (geomean)**: Vortex 2.000x ❌ · Parquet 1.000x ➖" in report


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
    assert "**Cold run (geomean)**: Vortex 4.000x ❌ · Parquet 1.000x ➖" in report
    assert markdown_row(report, "tpch_q01/datafusion:vortex-file-compressed") == [
        "tpch_q01/datafusion:vortex-file-compressed",
        "100",
        "100",
        "1.00",
        "400",
        "100",
        "4.00",
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
