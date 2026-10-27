# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

import copy
import json
import tempfile
import unittest
from pathlib import Path

import learn


class LearningTests(unittest.TestCase):
    def records(self, run_id: str) -> list[dict]:
        return [
            {
                "kind": "environment",
                "ci_run_id": run_id,
                "sha": "abc",
                "region": "us-east-1",
                "instance_id": f"i-{run_id}",
                "cpus": 64,
                "total_memory": 128000000000,
                "format": "vortex-file-compressed",
                "source": "s3://test/data/",
                "suite": "tpch",
                "scale_factor": "10",
            },
            {
                "kind": "sample",
                "backend": "real",
                "query": 1,
                "error": None,
                "counts": {"errors": 0},
                "requests": [
                    {"operation": "GET", "bytes": 1024, "first_byte_ms": 10, "duration_ms": 11, "inflight_at_start": 1},
                    {
                        "operation": "GET",
                        "bytes": 1048576,
                        "first_byte_ms": 10,
                        "duration_ms": 20,
                        "inflight_at_start": 1,
                    },
                ],
            },
        ]

    def write(self, root: Path, name: str, records: list[dict]) -> Path:
        path = root / name
        path.write_text("\n".join(json.dumps(r) for r in records))
        return path

    def test_learns_from_distinct_runs_and_records_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            a = self.write(root, "a.jsonl", self.records("123"))
            second = self.records("456")
            second[0]["total_memory"] -= 4096
            b = self.write(root, "b.jsonl", second)
            profile = learn.learn([a, b])
            self.assertEqual(profile["training"]["run_ids"], ["123", "456"])
            self.assertEqual(profile["calibration_get_requests"], 4)
            self.assertEqual(len(profile["latency_cdf_ms"]), 1000)
            self.assertEqual(len(profile["training"]["sources"][0]["sha256"]), 64)
            learn.validate_profile(profile, "789", "us-east-1", "vortex")
            with self.assertRaisesRegex(ValueError, "excluded from training"):
                learn.validate_profile(profile, "123", "us-east-1", "vortex")

    def test_rejects_single_run_duplicates_and_local_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            a = self.write(root, "a.jsonl", self.records("123"))
            with self.assertRaisesRegex(ValueError, "distinct CI runs"):
                learn.learn([a])
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                learn.learn([a, a])
            duplicate = self.records("123")
            duplicate[1]["seconds"] = 99
            b = self.write(root, "duplicate.jsonl", duplicate)
            with self.assertRaisesRegex(ValueError, "Duplicate training sample"):
                learn.learn([a, b])

            local = self.records("456")
            local[0]["local_smoke"] = True
            b = self.write(root, "b.jsonl", local)
            with self.assertRaisesRegex(ValueError, "real S3 in CI"):
                learn.learn([a, b])

    def test_steady_profile_excludes_protocol_startup_without_query_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            for run_id in ("123", "456"):
                records = self.records(run_id)
                steady = copy.deepcopy(records[1])
                steady["query"] = 2
                records[1].update(query=99, backend="real_repeat")
                for request in records[1]["requests"]:
                    request.update(duration_ms=1000, first_byte_ms=990)
                records.append(steady)
                paths.append(self.write(root, run_id + ".jsonl", records))
            profile = learn.learn(paths, preserve_tail=True, exclude_startup=True)
            self.assertEqual(profile["profile_version"], 3)
            self.assertEqual(profile["calibration_get_requests"], 4)
            self.assertEqual(profile["cdf_probabilities"][-1], 1)
            self.assertEqual([s["excluded_startup_samples"] for s in profile["training"]["sources"]], [1, 1])
            learn.validate_profile(profile, "789", "us-east-1", "vortex")

    def test_primed_inputs_retain_their_first_real_measurements(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            for run_id in ("123", "456"):
                records = self.records(run_id)
                records[0]["source_warmup_seconds"] = 20
                paths.append(self.write(root, run_id + ".jsonl", records))
            profile = learn.learn(paths, preserve_tail=True, exclude_startup=True)
            self.assertEqual(profile["calibration_get_requests"], 4)
            self.assertEqual([s["excluded_startup_samples"] for s in profile["training"]["sources"]], [0, 0])

    def test_rejects_mixed_regions_and_changed_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            a = self.write(root, "a.jsonl", self.records("123"))
            b = self.write(root, "b.jsonl", self.records("456"))
            profile = learn.learn([a, b])
            environment = self.records("789")[0]
            environment["model"] = {
                key: profile[key]
                for key in ("ttfb_ms", "request_mib_per_sec", "max_inflight", "latency_cdf_ms", "seed")
            }
            result = {"environments": [environment]}
            learn.attach_profile(result, profile, "digest")
            changed = copy.deepcopy(result)
            changed["environments"][0]["model"]["ttfb_ms"] += 1
            with self.assertRaisesRegex(ValueError, "differs from the frozen"):
                learn.attach_profile(changed, profile, "digest")
            other = self.records("456")
            other[0]["region"] = "us-west-2"
            b = self.write(root, "b.jsonl", other)
            with self.assertRaisesRegex(ValueError, "separate profiles"):
                learn.learn([a, b])


if __name__ == "__main__":
    unittest.main()
