# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

import copy
import unittest

import compare


class ComparisonTests(unittest.TestCase):
    def test_rank_correlation_with_ties(self):
        self.assertEqual(compare.spearman([1, 1, 3], [10, 10, 30]), 1.0)
        self.assertEqual(compare.spearman([1, 2, 3], [30, 20, 10]), -1.0)
        self.assertIsNone(compare.spearman([1, 1], [2, 3]))
        self.assertAlmostEqual(compare.quantile([1, 2, 3, 4, 5], 0.9), 4.6)

    def records(self) -> list[compare.Record]:
        records = []
        for suite, count in (("tpch", 22), ("tpcds", 99)):
            records.append(
                {
                    "kind": "environment",
                    "scale_factor": "10",
                    "suite": suite,
                    "local_smoke": False,
                    "ci_run_id": "123",
                    "sha": "abc",
                    "region": "us-east-1",
                    "instance_id": "i-123",
                    "adaptive_io": False,
                    "tail_cache_bytes": 4194304,
                    "format": "vortex",
                    "cpus": 64,
                    "total_memory": 128000000000,
                }
            )
            for query in range(1, count + 1):
                for cache in ("cold", "warm"):
                    for iteration in range(3):
                        for backend in ("real", "model"):
                            counts = {
                                "get": 1,
                                "put": 0,
                                "head": 0,
                                "list": 0,
                                "bytes": 100,
                                "errors": 0,
                                "peak_inflight": 1,
                            }
                            records.append(
                                {
                                    "kind": "sample",
                                    "suite": suite,
                                    "query": query,
                                    "cache": cache,
                                    "iteration": iteration,
                                    "backend": backend,
                                    "error": None,
                                    "rows": 1,
                                    "seconds": query,
                                    "counts": counts,
                                    "setup_counts": counts,
                                    "requests": [
                                        {"operation": "GET", "path": "x", "requested_range": "0..100", "bytes": 100}
                                    ],
                                }
                            )
        return records

    def test_complete_matching_run_passes(self):
        result = compare.compare(self.records(), 3, ["tpch", "tpcds"])
        self.assertTrue(result["adequate"])
        self.assertFalse(result["default_ci_recommended"])
        self.assertEqual(len(result["rows"]), 242)

    def test_missing_samples_and_request_mismatches_fail(self):
        records = self.records()
        records.pop()
        self.assertFalse(compare.compare(records, 3, ["tpch", "tpcds"])["adequate"])
        records = self.records()
        model = next(r for r in records if r.get("backend") == "model")
        model["counts"] = copy.copy(model["counts"])
        model["counts"]["get"] += 1
        self.assertFalse(compare.compare(records, 3, ["tpch", "tpcds"])["adequate"])
        records = self.records()
        model = next(r for r in records if r.get("backend") == "model")
        model["requests"][0]["requested_range"] = "100..200"
        self.assertFalse(compare.compare(records, 3, ["tpch", "tpcds"])["adequate"])

    def test_local_smoke_cannot_validate_model(self):
        records = self.records()
        records[0]["local_smoke"] = True
        self.assertFalse(compare.compare(records, 3, ["tpch", "tpcds"])["adequate"])

    def test_mixed_ci_provenance_cannot_validate_model(self):
        records = self.records()
        environments = [r for r in records if r["kind"] == "environment"]
        environments[1]["ci_run_id"] = "456"
        self.assertFalse(compare.compare(records, 3, ["tpch", "tpcds"])["adequate"])

    def test_calibration_uses_request_durations(self):
        requests = [
            {"operation": "GET", "bytes": 1024, "first_byte_ms": 10, "duration_ms": 11, "inflight_at_start": 1},
            {"operation": "GET", "bytes": 1048576, "first_byte_ms": 10, "duration_ms": 20, "inflight_at_start": 1},
        ]
        parameters = compare.calibrate([{"kind": "sample", "backend": "real", "error": None, "requests": requests}])
        self.assertEqual(parameters["ttfb_ms"], 10)
        self.assertEqual(parameters["request_mib_per_sec"], 100)


if __name__ == "__main__":
    unittest.main()
