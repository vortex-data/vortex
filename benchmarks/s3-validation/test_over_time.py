# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import over_time


class Clock:
    def __init__(self):
        self.now = 1000.0
        self.sleeps = []

    def time(self) -> float:
        return self.now

    def sleep(self, seconds: float):
        self.sleeps.append(seconds)
        self.now += seconds


class OverTimeTests(unittest.TestCase):
    def initialize(self, root: Path) -> Path:
        profile, binary = root / "profile.json", root / "binary"
        profile.write_text("{}")
        binary.write_bytes(b"frozen validator")
        output = root / "results"
        over_time.initialize(
            argparse.Namespace(
                output=output, profile=profile, binary=binary, source_root="s3://test/fixed", iterations=6
            )
        )
        return output

    def write_suite(self, command: list[str], **_kwargs: object):
        destination = Path(command[command.index("--output") + 1])
        suite = command[command.index("--suites") + 1]
        records = [
            {
                "kind": "environment",
                "suite": suite,
                "iterations": 6,
                "ci_run_id": "123",
                "sha": "abc",
                "region": "us-east-1",
                "instance_id": "i-1",
                "cpus": 64,
                "total_memory": 128000000000,
                "source": f"s3://test/fixed/{suite}/",
                "format": "vortex-file-compressed",
                "scale_factor": "10",
                "objects": [{"e_tag": "same"}],
                "adaptive_io": False,
                "tail_cache_bytes": 4194304,
                "model": {"seed": 20261027},
            }
        ]
        for iteration in range(6):
            for cache in ("cold", "warm"):
                for backend in ("real", "real_repeat", "model"):
                    counts = dict(get=1, put=0, list=0, head=0, bytes=1024, errors=0)
                    records.append(
                        dict(
                            kind="sample",
                            suite=suite,
                            query=1,
                            cache=cache,
                            iteration=iteration,
                            backend=backend,
                            seconds=1.02 if backend == "real_repeat" else 1,
                            error=None,
                            rows=1,
                            counts=counts,
                            setup_counts=counts,
                            requests=[
                                dict(
                                    operation="GET",
                                    path="x",
                                    requested_range="0..1024",
                                    bytes=1024,
                                    response_ms=10,
                                    duration_ms=12,
                                )
                            ],
                        )
                    )
        (destination / f"tuned-{suite}.jsonl").write_text("\n".join(json.dumps(r) for r in records) + "\n")

    def test_four_windows_use_fixed_starts_preserve_identity_and_stop(self):
        clock = Clock()

        def run(command: list[str], **kwargs: object):
            self.write_suite(command, **kwargs)
            clock.now += 123

        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(over_time.time, "time", clock.time),
            patch.object(over_time.time, "sleep", clock.sleep),
            patch.object(over_time, "QUERIES", {"tpch": [1], "tpcds": [1]}),
            patch.object(over_time.subprocess, "run", run),
            patch("builtins.print"),
        ):
            output = self.initialize(Path(directory))
            for index in range(1, 5):
                over_time.measure(output, index)
            over_time.report(output)
            result = json.loads((output / "over-time.json").read_text())
            self.assertTrue(result["complete"])
            self.assertEqual([w["started_unix"] for w in result["schedule"]["windows"]], [1000, 11800, 22600, 33400])
            self.assertLessEqual(max(clock.sleeps), 55)
            self.assertEqual(len(result["rows"]), 16)
            self.assertTrue(all(r["requests_match"] and r["real_drift_vs_first"] == 1 for r in result["rows"]))
            self.assertEqual(
                json.loads((output / "window-04/tuned-tpch.jsonl").read_text().splitlines()[0])[
                    "measurement_window_id"
                ],
                "4",
            )
            with self.assertRaisesRegex(ValueError, "once, sequentially"):
                over_time.measure(output, 4)

    def test_changed_objects_invalidate_window_and_partial_report_is_preserved(self):
        clock = Clock()
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(over_time.time, "time", clock.time),
            patch.object(over_time.time, "sleep", clock.sleep),
            patch.object(over_time, "QUERIES", {"tpch": [1], "tpcds": [1]}),
            patch.object(over_time.subprocess, "run", self.write_suite),
            patch("builtins.print"),
        ):
            output = self.initialize(Path(directory))
            over_time.measure(output, 1)
            schedule = json.loads((output / "schedule.json").read_text())
            schedule["data"]["tpch"]["objects"][0]["e_tag"] = "changed"
            over_time.save(output / "schedule.json", schedule)
            with self.assertRaisesRegex(ValueError, "physical I/O changed"):
                over_time.measure(output, 2)
            with self.assertRaisesRegex(ValueError, "Four successful"):
                over_time.report(output)
            self.assertEqual(json.loads((output / "schedule.json").read_text())["windows"][1]["status"], "failed")
            self.assertFalse(json.loads((output / "over-time.json").read_text())["complete"])

    def test_missed_cadence_does_not_silently_catch_up(self):
        clock = Clock()
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(over_time.time, "time", clock.time),
            patch.object(over_time.time, "sleep", clock.sleep),
            patch.object(over_time.subprocess, "run") as run,
            patch("builtins.print"),
        ):
            output = self.initialize(Path(directory))
            clock.now += 301
            with self.assertRaisesRegex(ValueError, "five minutes"):
                over_time.measure(output, 1)
            run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
