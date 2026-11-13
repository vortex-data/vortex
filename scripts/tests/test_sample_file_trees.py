# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

import importlib.util
import random
import sys
from collections import Counter
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "sample-file-trees.py"


def load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("sample_file_trees", SCRIPT)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    # `dataclass` resolves annotations through `sys.modules`.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


sft = load_module()


def files(*specs: tuple[str, int, int]) -> list[Any]:
    """`(name, bytes, lines)` triples as `VortexFile`s."""
    return [sft.VortexFile(Path(name), size, lines) for name, size, lines in specs]


# File names as the vortex-bench data generators write them; Vortex files keep the Parquet stem.
@pytest.mark.parametrize(
    ("file_name", "table"),
    [
        # TPC-H, TPC-DS and nested TPC-H: `split::part_file_name`, `{table}_{part:04}`.
        ("lineitem_0001.vortex", "lineitem"),
        ("lineitem_0012.vortex", "lineitem"),
        ("customer_0001.vortex", "customer"),
        ("customer_address_0001.vortex", "customer_address"),
        ("store_sales_0042.vortex", "store_sales"),
        # ClickBench: one file, partitioned `hits_{idx}`, or sorted shards `hits_{idx:03}`.
        ("hits.vortex", "hits"),
        ("hits_17.vortex", "hits"),
        ("hits_017.vortex", "hits"),
        # Single-file datasets and unsplit tables: the whole stem.
        ("sample.vortex", "sample"),
        ("stacktraces.vortex", "stacktraces"),
        ("gnomad.genomes.v3.1.2.hgdp_tgp.chr21.vortex", "gnomad.genomes.v3.1.2.hgdp_tgp.chr21"),
        ("movie_info_idx.vortex", "movie_info_idx"),
        ("orderitemnovelty_update.vortex", "orderitemnovelty_update"),
        ("test.vortex", "test"),
    ],
)
def test_table_name_matches_generated_files(file_name: str, table: str):
    assert sft.table_name(Path(file_name)) == table


def test_sample_keeps_one_file_per_table_then_fills_the_line_budget():
    dataset = files(
        ("nation.vortex", 10, 50),
        ("region.vortex", 10, 40),
        *((f"lineitem_{i:04}.vortex", 1000, 1700) for i in range(1, 11)),
        *((f"orders_{i:04}.vortex", 500, 1500) for i in range(1, 5)),
    )
    # The four per-table files take 3,290 lines; 3,500 more fit two further partitions.
    sample = sft.sample_files(dataset, 3290 + 3500, random.Random(0))

    tables = Counter(f.table for f in sample)
    assert set(tables) == {"nation", "region", "lineitem", "orders"}
    assert sum(tables.values()) == 4 + 2
    assert sum(f.lines for f in sample) <= 3290 + 3500
    assert len({f.path for f in sample}) == len(sample)


def test_sample_keeps_every_table_past_the_budget():
    dataset = files(("a_1.vortex", 10, 500), ("a_2.vortex", 10, 500), ("b.vortex", 10, 500))
    assert {f.table for f in sft.sample_files(dataset, 100, random.Random(0))} == {"a", "b"}


def test_sample_skips_a_file_too_large_for_the_budget_and_takes_smaller_ones():
    # `t_1` dwarfs the others, so it is drawn as the table's file; `t_2` then exceeds the budget.
    dataset = files(("t_1.vortex", 10**12, 1), ("t_2.vortex", 10**9, 1000), ("t_3.vortex", 1, 10))
    sample = sft.sample_files(dataset, 20, random.Random(0))
    assert [f.path.name for f in sample] == ["t_1.vortex", "t_3.vortex"]


def test_sample_takes_everything_within_budget():
    dataset = files(("a_1.vortex", 1, 5), ("a_2.vortex", 2, 5), ("b.vortex", 0, 5))
    assert len(sft.sample_files(dataset, 100, random.Random(0))) == 3


def test_sample_is_deterministic_for_a_seed():
    dataset = files(*((f"hits_{i}.vortex", i + 1, 100) for i in range(100)))
    assert sft.sample_files(dataset, 900, random.Random("abc")) == sft.sample_files(dataset, 900, random.Random("abc"))


def test_weighted_order_puts_large_files_first():
    dataset = files(("big.vortex", 1_000_000_000, 1), ("small.vortex", 1_000, 1))
    rng = random.Random(0)
    picks = Counter(sft.weighted_order(dataset, rng)[0].path.name for _ in range(1000))
    assert picks["big.vortex"] > 990
