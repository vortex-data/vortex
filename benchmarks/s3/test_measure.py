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
            _, rows = measure.read_window(path)
            self.assertEqual(len(rows), 4)
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


if __name__ == "__main__":
    unittest.main()
