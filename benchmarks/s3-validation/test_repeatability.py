# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

import copy
import unittest

import compare
import repeatability


class RepeatabilityTests(unittest.TestCase):
    def records(self, model_seconds: float = 1) -> list[compare.Record]:
        records = [{"kind": "environment", "suite": "tpch", "iterations": 24}]
        for iteration in range(24):
            for backend in ("real", "real_repeat", "model"):
                counts = dict(get=1, put=0, list=0, head=0, bytes=1024, errors=0)
                records.append(
                    dict(
                        kind="sample",
                        suite="tpch",
                        query=1,
                        cache="cold",
                        iteration=iteration,
                        backend=backend,
                        seconds=model_seconds if backend == "model" else 1,
                        error=None,
                        rows=1,
                        counts=counts,
                        setup_counts=counts,
                        requests=[dict(operation="GET", path="x", requested_range="0..1024", bytes=1024)],
                    )
                )
        return records

    def test_identical_controls_and_model_have_no_excess_error(self):
        row = repeatability.analyze(self.records(), draws=100)["rows"][0]
        self.assertTrue(row["requests_match"])
        self.assertFalse(row["bias_ci_excludes_one"])
        self.assertFalse(row["error_exceeds_s3_repeatability"])
        self.assertEqual(row["excess_log_mae"], 0)

    def test_persistent_model_bias_is_not_dismissed_as_reference_noise(self):
        row = repeatability.analyze(self.records(2), draws=100)["rows"][0]
        self.assertTrue(row["bias_ci_excludes_one"])
        self.assertTrue(row["error_exceeds_s3_repeatability"])
        self.assertEqual(row["model_ratio_ci_low"], 2)
        self.assertEqual(row["model_ratio_ci_high"], 2)

    def test_missing_control_and_request_mismatch_are_invalid(self):
        records = self.records()
        records.pop()
        self.assertFalse(repeatability.analyze(records, draws=100)["rows"][0]["requests_match"])
        records = self.records()
        records[-1]["requests"][0]["bytes"] = 2048
        self.assertFalse(repeatability.analyze(records, draws=100)["rows"][0]["requests_match"])

    def test_block_bootstrap_is_repeatable(self):
        records = self.records()
        for record in records[1:]:
            record["seconds"] *= 1 + record["iteration"] * 0.01
        self.assertEqual(repeatability.analyze(records, draws=100), repeatability.analyze(records, draws=100))

    def test_paired_old_model_comparison_and_requests_are_checked(self):
        records = self.records()
        for record in records[1:].copy():
            if record["backend"] == "model":
                baseline = copy.deepcopy(record)
                baseline.update(backend="model_baseline", seconds=2)
                records.append(baseline)
        row = repeatability.analyze(records, draws=100)["rows"][0]
        self.assertTrue(row["requests_match"])
        self.assertGreater(row["improvement_ci_low"], 0)
        self.assertEqual(row["baseline_model_median_ratio"], 2)
        records[-1]["requests"][0]["bytes"] = 2048
        self.assertFalse(repeatability.analyze(records, draws=100)["rows"][0]["requests_match"])


if __name__ == "__main__":
    unittest.main()
