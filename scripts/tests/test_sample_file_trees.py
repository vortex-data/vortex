# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

import importlib.util
import random
import sys
from collections import Counter
from pathlib import Path
from types import ModuleType

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


def files(*specs: tuple[str, int]) -> list:
    return [sft.VortexFile(Path(name), size) for name, size in specs]


def test_table_name_strips_partition_index():
    assert sft.table_name(Path("lineitem_0001.vortex")) == "lineitem"
    assert sft.table_name(Path("hits_17.vortex")) == "hits"
    assert sft.table_name(Path("part-3.vortex")) == "part"
    assert sft.table_name(Path("customer_address.vortex")) == "customer_address"
    assert sft.table_name(Path("nation.vortex")) == "nation"


def test_sample_has_one_file_per_table_plus_extra():
    dataset = files(
        ("nation.vortex", 10),
        ("region.vortex", 10),
        *((f"lineitem_{i:04}.vortex", 1000) for i in range(1, 11)),
        *((f"orders_{i:04}.vortex", 500) for i in range(1, 5)),
    )
    sample = sft.sample_files(dataset, 3, random.Random(0))

    tables = Counter(f.table for f in sample)
    assert set(tables) == {"nation", "region", "lineitem", "orders"}
    assert sum(tables.values()) == 4 + 3
    assert len({f.path for f in sample}) == len(sample)


def test_sample_takes_everything_when_extra_exceeds_remaining():
    dataset = files(("a_1.vortex", 1), ("a_2.vortex", 2), ("b.vortex", 0))
    assert len(sft.sample_files(dataset, 100, random.Random(0))) == 3


def test_sample_is_deterministic_for_a_seed():
    dataset = files(*((f"hits_{i}.vortex", i + 1) for i in range(100)))
    assert sft.sample_files(dataset, 5, random.Random("abc")) == sft.sample_files(dataset, 5, random.Random("abc"))


def test_weighted_sample_prefers_large_files():
    dataset = files(("big.vortex", 1_000_000_000), ("small.vortex", 1_000))
    rng = random.Random(0)
    picks = Counter(sft.weighted_sample(dataset, 1, rng)[0].path.name for _ in range(1000))
    assert picks["big.vortex"] > 990
