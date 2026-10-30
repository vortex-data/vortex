# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Fixtures for the generated-data suites of `vortex-sqllogictest`."""

import shutil
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from .config import Benchmark

# `data-gen` writes each format under vortex-bench/data/<dataset>/<format>/.
FORMAT_DIRS = {"parquet": "parquet", "vortex": "vortex-file-compressed"}


class SltDataset(Enum):
    """Datasets that `vortex-sqllogictest` reads from git-ignored fixtures."""

    TPCH = "tpch"
    TPCDS = "tpcds"
    CLICKBENCH = "clickbench"


@dataclass(frozen=True)
class SltDatasetSpec:
    """How to generate one SLT dataset with `data-gen` and lay it out for the suites."""

    benchmark: Benchmark
    options: dict[str, str]
    # The `data-gen` output directory, relative to vortex-bench/data/.
    bench_subdir: str
    # Stripped from every file stem so the files match the tables the suites read.
    stem_suffix: str = ""
    formats: tuple[str, ...] = ("parquet", "vortex")


SLT_DATASETS: dict[SltDataset, SltDatasetSpec] = {
    SltDataset.TPCH: SltDatasetSpec(
        benchmark=Benchmark.TPCH,
        options={"scale-factor": "0.1"},
        bench_subdir="tpch/0.1",
    ),
    # Every TPC-DS table fits in one `<table>_0001` file at this scale factor.
    SltDataset.TPCDS: SltDatasetSpec(
        benchmark=Benchmark.TPCDS,
        options={"scale-factor": "0.1"},
        bench_subdir="tpcds/0.1",
        stem_suffix="_0001",
    ),
    # One shard (~1M rows) of the partitioned ClickBench `hits` table.
    SltDataset.CLICKBENCH: SltDatasetSpec(
        benchmark=Benchmark.CLICKBENCH,
        options={"files": "1"},
        bench_subdir="clickbench_partitioned_1",
        stem_suffix="_0",
    ),
}


def bench_data_dir(workspace_root: Path, spec: SltDatasetSpec) -> Path:
    return workspace_root / "vortex-bench" / "data" / spec.bench_subdir


def slt_data_dir(workspace_root: Path, dataset: SltDataset) -> Path:
    return workspace_root / "vortex-sqllogictest" / "slt" / dataset.value / "data"


def remove_stale_vortex(workspace_root: Path, spec: SltDatasetSpec) -> None:
    """Drop existing Vortex output, which `data-gen` would otherwise reuse.

    The fixtures must come from the current writer. The Parquet inputs are deterministic,
    so they are kept and reused.
    """
    shutil.rmtree(bench_data_dir(workspace_root, spec) / FORMAT_DIRS["vortex"], ignore_errors=True)


def install_fixtures(workspace_root: Path, dataset: SltDataset) -> None:
    """Copy the generated files into slt/<dataset>/data/ under the names the suites read."""
    spec = SLT_DATASETS[dataset]
    source_dir = bench_data_dir(workspace_root, spec)
    target_dir = slt_data_dir(workspace_root, dataset)

    shutil.rmtree(target_dir, ignore_errors=True)
    target_dir.mkdir(parents=True)
    for fmt in spec.formats:
        files = sorted((source_dir / FORMAT_DIRS[fmt]).glob(f"*.{fmt}"))
        if not files:
            raise RuntimeError(f"data-gen wrote no {fmt} files to {source_dir / FORMAT_DIRS[fmt]}")
        for path in files:
            if not path.stem.endswith(spec.stem_suffix):
                # A table split into several files would silently lose rows here.
                raise RuntimeError(f"expected {path.name} to end in {spec.stem_suffix}.{fmt}")
            stem = path.stem.removesuffix(spec.stem_suffix)
            shutil.copy2(path, target_dir / f"{stem}{path.suffix}")
