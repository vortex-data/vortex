# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

import json
import tempfile
import unittest
from pathlib import Path

import compare


class ComparisonTests(unittest.TestCase):
    def test_tied_ranks_and_constant_samples(self):
        self.assertEqual(compare.ranks([20, 10, 20, 30]), [1.5, 0, 1.5, 3])
        self.assertEqual(compare.correlation([2, 1, 2], [20, 10, 20]), 1)
        self.assertIsNone(compare.correlation([1, 1], [1, 2]))

    def test_recomputes_median_p90_and_preserves_real_repeat(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for backend, multiplier in (("real", 1), ("real-repeat", 1.1), ("simulated", 1.2)):
                rows = []
                for query in (1, 2):
                    rows.append(
                        {
                            "kind": "query_measurement",
                            "commit_sha": "abc",
                            "dataset": "tpch",
                            "query_idx": query,
                            "format": "vortex",
                            "engine": "datafusion",
                            "env_triple": "arm64",
                            "all_runtimes_ns": [round(value * multiplier * query * 1e6) for value in (10, 20, 30)],
                            "value_ns": 1,
                        }
                    )
                (root / f"tpch-{backend}.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
            rows, summary = compare.compare(root)
            self.assertEqual(rows[0]["real_median_ms"], 20)
            self.assertEqual(rows[0]["real_p90_ms"], 28)
            self.assertEqual(rows[0]["real_repeat_ratio"], 1.1)
            self.assertEqual(rows[0]["model_ratio"], 24 / 21)
            self.assertTrue(rows[0]["outside_10pct"])
            self.assertEqual(summary["spearman"], 1)
            (root / "tpch-real-repeat.jsonl").unlink()
            with self.assertRaisesRegex(ValueError, "Missing backend"):
                compare.compare(root)


if __name__ == "__main__":
    unittest.main()
