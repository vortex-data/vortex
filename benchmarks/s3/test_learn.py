# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

import json
import tempfile
import unittest
from pathlib import Path

import learn


class LearningTests(unittest.TestCase):
    def write(self, root: Path, run: str, **changes: object) -> Path:
        environment = {
            "kind": "environment",
            "schema_version": 1,
            "measurement_id": f"{run}-1",
            "ci_run_id": run,
            "ci_run_attempt": "1",
            "sha": "abc",
            "region": "us-east-1",
            "cpus": 64,
            "total_memory": 128000000000,
            "instance_id": f"i-{run}",
            "source": "s3://bucket/prefix/",
            "rounds": 1,
            "sizes": [4096],
            "concurrency": [1],
            "requests_per_cell": 2,
            "local_smoke": False,
            "endpoint_override": False,
            **changes,
        }
        records = [environment]
        for ordinal in range(2):
            records.append(
                {
                    "kind": "request",
                    "phase": "measure",
                    "operation": "GET",
                    "round": 0,
                    "ordinal": ordinal,
                    "bytes": 4096,
                    "range": {"start": 0, "end": 4096},
                    "requested_concurrency": 1,
                    "inflight_at_start": 1,
                    "status": "complete",
                    "response_ms": 10,
                    "first_byte_ms": 11,
                    "duration_ms": 12,
                }
            )
        records.extend(
            [
                {
                    "kind": "cell_summary",
                    "round": 0,
                    "range_bytes": 4096,
                    "requested_concurrency": 1,
                    "requests": 2,
                    "bytes": 8192,
                    "errors": 0,
                    "seconds": 0.02,
                    "peak_inflight": 1,
                    "avg_inflight": 1,
                },
                {"kind": "complete", "errors": 0},
            ]
        )
        path = root / f"{run}.jsonl"
        path.write_text("\n".join(json.dumps(record) for record in records) + "\n")
        return path

    def test_real_windows_generate_integer_weighted_joint_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = learn.learn([self.write(root, "123"), self.write(root, "456")])
            self.assertEqual(model["training"]["real_gets"], 4)
            self.assertEqual(model["metadata_header_us"], 10000)
            samples = model["bands"][0]["samples"]
            self.assertTrue(all(sample[:4] == [10000, 1000, 1000, 4096] for sample in samples))
            self.assertEqual(sum(sample[4] for sample in samples), 4)
            self.assertEqual(len(model["training"]["measurement_windows"]), 2)

    def test_one_window_duplicates_and_simulated_sources_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = self.write(root, "123")
            with self.assertRaisesRegex(ValueError, "two distinct"):
                learn.learn([first])
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                learn.learn([first, first])
            for changes in ({"local_smoke": True}, {"endpoint_override": True}, {"source": "file:///tmp/"}):
                with self.subTest(changes=changes):
                    second = self.write(root, "456", **changes)
                    with self.assertRaisesRegex(ValueError, "real S3"):
                        learn.learn([first, second])

    def test_mixed_hardware_and_regions_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = self.write(root, "123")
            for changes in ({"cpus": 32}, {"region": "eu-west-1"}, {"total_memory": 64000000000}):
                with self.subTest(changes=changes):
                    with self.assertRaisesRegex(ValueError, "separate profiles"):
                        learn.learn([first, self.write(root, "456", **changes)])

    def test_sparse_tail_retains_its_observed_mass(self):
        values = [(10000, 100, 1000, 4096)] * 9999 + [(23000000, 100, 1000, 4096)]
        samples = learn.compress(values, 100)
        self.assertEqual(sum(sample[4] for sample in samples), 10000)
        self.assertEqual(samples[-1], [23000000, 100, 1000, 4096, 1])


if __name__ == "__main__":
    unittest.main()
