# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import measure


class MeasurementTests(unittest.TestCase):
    def records(self, measurement_id: str = "test") -> list[dict]:
        records = [
            {
                "kind": "environment",
                "schema_version": 1,
                "measurement_id": measurement_id,
                "rounds": 1,
                "sizes": [4, 8],
                "concurrency": [1, 4],
                "requests_per_cell": 2,
                "objects": [{"path": "object", "size": 128, "e_tag": "same"}],
            }
        ]
        for size in (4, 8):
            for concurrency in (1, 4):
                for ordinal in range(2):
                    records.append(
                        {
                            "kind": "request",
                            "phase": "measure",
                            "status": "complete",
                            "bytes": size,
                            "range": {"start": ordinal, "end": ordinal + size},
                            "round": 0,
                            "ordinal": ordinal,
                            "requested_concurrency": concurrency,
                            "response_ms": 1.0,
                            "first_byte_ms": 1.5,
                            "duration_ms": 3.0,
                        }
                    )
                records.append(
                    {
                        "kind": "cell_summary",
                        "round": 0,
                        "range_bytes": size,
                        "requested_concurrency": concurrency,
                        "requests": 2,
                        "bytes": size * 2,
                        "errors": 0,
                        "seconds": 0.01,
                        "peak_inflight": min(concurrency, 2),
                        "avg_inflight": 0.6,
                    }
                )
        records.append({"kind": "complete", "errors": 0})
        return records

    def write(self, path: Path, records: list[dict]):
        path.write_text("\n".join(json.dumps(r) for r in records) + "\n")

    def test_separates_header_body_and_aggregate_throughput(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            self.write(path, self.records())
            _, rows, samples = measure.read_window(path)
            self.assertEqual(len(rows), 4)
            self.assertEqual(len(samples), 8)
            self.assertTrue(all(r["headers_median_ms"] == 1 and r["body_p90_ms"] == 2 for r in rows))
            self.assertAlmostEqual(rows[0]["aggregate_mib_per_second"], 8 / 1048576 / 0.01)

    def test_rejects_partial_duplicate_and_inconsistent_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            cases = [
                ("Incomplete", lambda records: records.pop()),
                ("ordinal", lambda records: records[2].update(ordinal=0)),
                ("bytes", lambda records: records[1].update(bytes=3)),
                ("timing", lambda records: records[1].update(response_ms=None)),
                ("counters", lambda records: records[3].update(bytes=10)),
            ]
            for message, change in cases:
                with self.subTest(message=message):
                    records = self.records()
                    change(records)
                    self.write(path, records)
                    with self.assertRaisesRegex(ValueError, message):
                        measure.read_window(path)

    def cold(self, records: list[dict]) -> list[dict]:
        records[0].update(
            cold_source="s3://test/cold/",
            cold_concurrency=2,
            cold_requests={"4": 2, "8": 2},
            cold_objects=[],
        )
        complete = records.pop()
        for size in (4, 8):
            for ordinal in range(2):
                records.append(
                    {
                        "kind": "request",
                        "phase": "cold",
                        "path": f"cold/{size}/{ordinal}.bin",
                        "status": "complete",
                        "bytes": size,
                        "range": {"start": 0, "end": size},
                        "round": 0,
                        "ordinal": ordinal,
                        "requested_concurrency": 2,
                        "started_unix_ms": 7200000,
                        "last_modified_unix_ms": 0,
                        "response_ms": 10.0,
                        "first_byte_ms": 12.0,
                        "duration_ms": 20.0,
                    }
                )
            records.append(
                {
                    "kind": "cell_summary",
                    "phase": "cold",
                    "round": 0,
                    "range_bytes": size,
                    "requested_concurrency": 2,
                    "requests": 2,
                    "bytes": size * 2,
                    "errors": 0,
                    "seconds": 0.02,
                    "peak_inflight": 2,
                    "avg_inflight": 2.0,
                }
            )
        records.append(complete)
        return records

    def test_cold_reads_are_reported_separately_and_once_per_object(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            self.write(path, self.cold(self.records()))
            _, rows, samples = measure.read_window(path)
            cold = [r for r in rows if r["phase"] == "cold"]
            self.assertEqual([r["bytes_per_request"] for r in cold], [4, 8])
            self.assertTrue(all(r["total_p999_ms"] == 20 and r["min_hours_since_write"] == 2 for r in cold))
            tails = measure.tails(samples)
            pooled = [r for r in tails if r["concurrency"] == "all"]
            self.assertEqual([(r["phase"], r["bytes_per_request"]) for r in pooled][:2], [("cold", 4), ("cold", 8)])
            self.assertIn("| cold | 4 | 2 |", measure.tail_markdown(tails))
            for message, change in (
                ("more than once", lambda records: records[-3].update(path="cold/8/0.bin")),
                ("Cold coverage", lambda records: records.pop(-3)),
            ):
                with self.subTest(message=message):
                    records = self.cold(self.records())
                    change(records)
                    self.write(path, records)
                    with self.assertRaisesRegex(ValueError, message):
                        measure.read_window(path)

    def test_four_starts_are_anchored_and_no_fifth_window_runs(self):
        now, starts, sleeps = [1000.0], [], []

        def sleep(seconds: float):
            sleeps.append(seconds)
            now[0] += seconds

        def run(command: list[str], **_kwargs: object):
            starts.append(now[0])
            path = Path(command[command.index("--output") + 1])
            name = command[command.index("--measurement-id") + 1]
            self.write(path, self.records(name))
            now[0] += 120

        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(measure.time, "time", lambda: now[0]),
            patch.object(measure.time, "sleep", sleep),
            patch.object(measure.subprocess, "run", run),
            patch("builtins.print"),
        ):
            root = Path(directory)
            binary = root / "binary"
            binary.write_bytes(b"frozen collector")
            args = argparse.Namespace(
                output=root / "results",
                binary=binary,
                source="s3://test/prefix/",
                count=4,
                interval_seconds=10800,
                window_timeout_seconds=1800,
                seed=42,
                allow_local=False,
                probe_args=[],
            )
            measure.run(args, {})
            self.assertEqual(starts, [1000, 11800, 22600, 33400])
            self.assertLessEqual(max(sleeps), 55)
            plan = json.loads((args.output / "schedule.json").read_text())
            self.assertTrue(all(w["status"] == "success" for w in plan["windows"]))
            self.assertTrue((args.output / "measurements.csv").exists())
            with self.assertRaisesRegex(ValueError, "already contains evidence"):
                measure.run(args, {})

    def test_ci_resumes_fixed_windows_with_credentials_supplied_by_each_action(self):
        now, starts, inherited = [1000.0], [], []

        def sleep(seconds: float):
            now[0] += seconds

        def run(command: list[str], **kwargs: object):
            starts.append(now[0])
            inherited.append(kwargs["env"])
            path = Path(command[command.index("--output") + 1])
            self.write(path, self.records(command[command.index("--measurement-id") + 1]))
            now[0] += 120

        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(measure.time, "time", lambda: now[0]),
            patch.object(measure.time, "sleep", sleep),
            patch.object(measure.subprocess, "run", run),
            patch("builtins.print"),
        ):
            root = Path(directory)
            binary = root / "binary"
            binary.write_bytes(b"frozen collector")
            args = argparse.Namespace(
                output=root / "results",
                binary=binary,
                source="s3://test/prefix/",
                count=4,
                interval_seconds=10800,
                window_timeout_seconds=1800,
                seed=42,
                allow_local=False,
                probe_args=[],
            )
            measure.initialize(args)
            credentials = []
            for index in range(1, 5):
                plan = measure.load_plan(args)
                measure.wait_until(plan["windows"][index - 1]["planned_unix"])
                now[0] += 1  # The workflow action runs after the wait, before collection.
                env = {"AWS_ACCESS_KEY_ID": f"action-placeholder-{index}", "GITHUB_RUN_ID": "123"}
                credentials.append(env)
                measure.collect(args, env, measure.load_plan(args), index)
            self.assertEqual(starts, [1001, 11801, 22601, 33401])
            self.assertEqual(inherited, credentials)
            plan = measure.load_plan(args)
            measure.summarize(args, plan)
            self.assertTrue(all(w["status"] == "success" for w in plan["windows"]))
            with self.assertRaisesRegex(ValueError, "already attempted"):
                measure.collect(args, {}, plan, 1)
            args.seed += 1
            with self.assertRaisesRegex(ValueError, "configuration changed"):
                measure.load_plan(args)
            args.seed -= 1
            binary.write_bytes(b"changed collector")
            with self.assertRaisesRegex(ValueError, "binary changed"):
                measure.load_plan(args)
            binary.write_bytes(b"frozen collector")
            (args.output / "window-01.jsonl").write_text("changed trace")
            with self.assertRaisesRegex(ValueError, "trace changed"):
                measure.summarize(args, measure.load_plan(args))


if __name__ == "__main__":
    unittest.main()
