# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

import stat
import textwrap
from pathlib import Path

import pytest
from bench_orchestrator import cli as cli_module
from bench_orchestrator.runner import builder as builder_module
from typer.testing import CliRunner

runner = CliRunner()

# Writes the files the real data-gen would, so prepare-slt can install them.
FAKE_DATA_GEN = textwrap.dedent(
    """\
    #!/usr/bin/env bash
    set -e
    echo "data-gen $*"
    case "$1" in
      tpch) dir=vortex-bench/data/tpch/0.1; stems="lineitem orders" ;;
      tpcds) dir=vortex-bench/data/tpcds/0.1; stems="store_sales_0001 item_0001" ;;
      clickbench) dir=vortex-bench/data/clickbench_partitioned_1; stems="hits_0" ;;
    esac
    if [ -n "${FAIL_DATASET:-}" ] && [ "$1" = "${FAIL_DATASET}" ]; then exit 3; fi
    mkdir -p "$dir/parquet" "$dir/vortex-file-compressed"
    for stem in $stems; do
      echo parquet > "$dir/parquet/$stem.parquet"
      [ -f "$dir/vortex-file-compressed/$stem.vortex" ] || echo fresh > "$dir/vortex-file-compressed/$stem.vortex"
    done
    """
)


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    data_gen = tmp_path / "target" / "release" / "data-gen"
    data_gen.parent.mkdir(parents=True)
    data_gen.write_text(FAKE_DATA_GEN, encoding="utf-8")
    data_gen.chmod(data_gen.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setattr(builder_module, "get_workspace_root", lambda: tmp_path)
    return tmp_path


def slt_files(workspace, dataset: str) -> list[str]:
    return sorted(p.name for p in (workspace / "vortex-sqllogictest" / "slt" / dataset / "data").iterdir())


@pytest.mark.parametrize("parallel", [False, True])
def test_prepare_slt_installs_renamed_fixtures(workspace, parallel) -> None:
    args = ["prepare-slt", "--no-build"] + (["--parallel"] if parallel else [])
    result = runner.invoke(cli_module.app, args)

    assert result.exit_code == 0, result.output
    assert slt_files(workspace, "tpch") == ["lineitem.parquet", "lineitem.vortex", "orders.parquet", "orders.vortex"]
    assert slt_files(workspace, "tpcds") == ["item.parquet", "item.vortex", "store_sales.parquet", "store_sales.vortex"]
    assert slt_files(workspace, "clickbench") == ["hits.parquet", "hits.vortex"]
    if parallel:
        assert "[tpcds] data-gen tpcds --formats parquet,vortex --opt scale-factor=0.1" in result.output


def test_prepare_slt_reconverts_stale_vortex(workspace) -> None:
    vortex_dir = workspace / "vortex-bench" / "data" / "tpch" / "0.1" / "vortex-file-compressed"
    vortex_dir.mkdir(parents=True)
    (vortex_dir / "lineitem.vortex").write_text("stale", encoding="utf-8")
    (vortex_dir / "dropped.vortex").write_text("stale", encoding="utf-8")

    result = runner.invoke(cli_module.app, ["prepare-slt", "--no-build", "tpch"])

    assert result.exit_code == 0, result.output
    data_dir = workspace / "vortex-sqllogictest" / "slt" / "tpch" / "data"
    assert (data_dir / "lineitem.vortex").read_text(encoding="utf-8") == "fresh\n"
    assert not (data_dir / "dropped.vortex").exists()


def test_prepare_slt_parallel_reports_every_failure(workspace, monkeypatch) -> None:
    monkeypatch.setenv("FAIL_DATASET", "tpcds")

    result = runner.invoke(cli_module.app, ["prepare-slt", "--no-build", "--parallel"])

    assert result.exit_code == 1
    assert "SLT data generation failed for: tpcds" in result.output
    # The other datasets still finish.
    assert slt_files(workspace, "tpch")
    assert slt_files(workspace, "clickbench")


def test_prepare_data_runs_several_benchmarks_in_parallel(workspace, monkeypatch) -> None:
    monkeypatch.setattr(
        cli_module.BenchmarkBuilder,
        "get_data_generator_path",
        lambda self: workspace / "target" / "release" / "data-gen",
    )
    monkeypatch.chdir(workspace)

    result = runner.invoke(
        cli_module.app,
        ["prepare-data", "tpch", "clickbench", "--format", "parquet", "--parallel", "--opt", "scale-factor=0.1"],
    )

    assert result.exit_code == 0, result.output
    assert "[tpch] data-gen tpch --formats parquet --opt scale-factor=0.1" in result.output
    assert "[clickbench] data-gen clickbench --formats parquet --opt scale-factor=0.1" in result.output
