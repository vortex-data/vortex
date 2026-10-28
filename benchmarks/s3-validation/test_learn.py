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

    def test_windows_keep_sample_identity_without_inventing_independent_ci_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            windows = []
            for index in range(1, 5):
                records = self.records("123")
                records[0].update(measurement_window_id=str(index), measurement_window_started_unix=index * 10800)
                windows.append(self.write(root, f"window-{index}.jsonl", records))
            with self.assertRaisesRegex(ValueError, "distinct CI runs"):
                learn.learn(windows)
            earlier = self.write(root, "earlier.jsonl", self.records("456"))
            profile = learn.learn([earlier, *windows])
            self.assertEqual(profile["training"]["run_ids"], ["123", "456"])
            self.assertEqual(profile["calibration_get_requests"], 10)
            self.assertEqual(
                [s.get("measurement_window_id") for s in profile["training"]["sources"]], [None, "1", "2", "3", "4"]
            )
            changed = self.records("123")
            changed[0].update(measurement_window_id="1", measurement_window_started_unix=10800)
            changed[1]["seconds"] = 99
            duplicate = self.write(root, "duplicate-window.jsonl", changed)
            with self.assertRaisesRegex(ValueError, "Duplicate training sample"):
                learn.learn([earlier, *windows, duplicate])

    def test_phase_profile_uses_measured_headers_and_records_held_out_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for run_id in ("123", "456"):
                records = self.records(run_id)
                for request in records[1]["requests"]:
                    request["response_ms"] = 9.0
                model = copy.deepcopy(records[1])
                model.update(backend="model", model=dict(ttfb_ms=8.0, request_mib_per_sec=100.0))
                for request in model["requests"]:
                    request.update(
                        model_delay_ms=8.0,
                        response_ms=9.0,
                        duration_ms=8.0 + request["bytes"] / 1048576 / 100 * 1000 + 2.0,
                    )
                records.append(model)
                paths.append(self.write(Path(directory), run_id + ".jsonl", records))
            profile = learn.learn(paths, request_phases=True)
            self.assertEqual(profile["profile_version"], 4)
            self.assertEqual(profile["latency_cdf_ms"], [])
            self.assertEqual(profile["latency_samples"][-1]["cdf_probability"], 1)
            self.assertIn("body_timer_ms", profile["phase_calibration"])
            learn.validate_profile(profile, "789", "us-east-1", "vortex")
            with self.assertRaisesRegex(ValueError, "excluded from training"):
                learn.validate_profile(profile, "456", "us-east-1", "vortex")

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
