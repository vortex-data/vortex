# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

import json

import numpy as np
import pytest

FEATURES = [
    "ptype_bits",
    "signed",
    "len",
    "null_frac",
    "bits_bp",
    "bits_for",
    "bits_zz",
    "for_gain",
    "avg_run",
    "distinct",
    "distinct_frac",
    "top1_frac",
    "top10_cover",
    "sorted_frac",
    "bits_delta",
]
HEADER = (
    "tag,source,column,chunk,ptype,len,variant,ok,canonical_bytes,bytes,nbytes,root,tree,compress_ns,"
    "decode_ns_median,decode_ns_min,error"
)


def write_dataset(path, seed=0, sources=("tpch-sf1", "synthetic-runs", "clickbench"), chunks=12):
    """A small dataset in `vx-lab materialize` layout where the cheapest candidate depends on avg_run."""
    rng = np.random.default_rng(seed)
    rows, feats = [HEADER], ["source,column,chunk," + ",".join(FEATURES)]
    for s in sources:
        for c in range(4):
            for k in range(chunks):
                chunk = f"{s[:3]}{c}{k:04d}"
                n = 65536
                canon = n * 8
                avg_run = float(rng.choice([1, 2, 8, 64]))
                bits = int(rng.integers(4, 30))
                f = [64, 1, n, 0.0, bits, bits - 1, bits + 1, 1.0, avg_run, 1000, 0.01, 0.2, 0.5, 0.1, bits]
                feats.append(f"{s},c{c},{chunk}," + ",".join(str(v) for v in f))
                cands = {
                    "default": (
                        canon / (4 + avg_run / 8),
                        60_000,
                        "vortex.runend" if avg_run >= 8 else "fastlanes.for",
                    ),
                    "forced/for": (n * bits / 8, 25_000, "fastlanes.for"),
                    "forced/runend": (
                        n / avg_run * (bits + 16) / 8,
                        20_000 + 800_000 / avg_run,
                        "vortex.runend",
                    ),
                    "forced/bitpacking": (n * (bits + 1) / 8, 22_000, "fastlanes.bitpacked"),
                }
                for v, (b, dec, root) in cands.items():
                    b = int(b * rng.uniform(0.95, 1.05))
                    dec = int(dec * rng.uniform(0.9, 1.1))
                    rows.append(
                        f"stock,{s},c{c},{chunk},i64,{n},{v},1,{canon},{b},{b},{root},{root}(...),"
                        f"{dec * 5},{dec},{int(dec * 0.95)},"
                    )
                rows.append(
                    f"stock,{s},c{c},{chunk},i64,{n},forced/sparse,0,{canon},,,,,,,,no dominant value"
                )
    (path / "rows-stock.csv").write_text("\n".join(rows) + "\n")
    (path / "features-stock.csv").write_text("\n".join(feats) + "\n")
    (path / "dataset.json").write_text(
        json.dumps(
            {
                "plan_id": "p",
                "git_commit": "abc123",
                "machine_id": "m1",
                "chunks": len(sources) * 4 * chunks,
                "incomplete": 0,
                "timed_chunks": len(sources) * 4 * chunks,
                "max_runs": 1,
            }
        )
    )
    return path


@pytest.fixture
def dataset(tmp_path):
    return write_dataset(tmp_path)
