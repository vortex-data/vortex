# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

import contextlib
import importlib.util
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

SPEC = importlib.util.spec_from_file_location("bench_io", Path(__file__).parents[1] / "bench-io.py")
assert SPEC is not None and SPEC.loader is not None
BENCH_IO = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCH_IO)


class BenchIoTests(unittest.TestCase):
    def test_pressure_window_uses_totals_and_handles_unavailable_or_reset_counters(self):
        before = {"memory": "some avg10=20.0 total=100\nfull avg10=15.0 total=90"}
        after = {"memory": "some avg10=15.0 total=112\nfull avg10=10.0 total=99"}
        self.assertEqual(
            BENCH_IO.counter_delta(BENCH_IO.pressure_totals(before), BENCH_IO.pressure_totals(after)),
            {"memory/some": 12, "memory/full": 9},
        )
        self.assertIsNone(BENCH_IO.counter_delta({}, {}))
        self.assertIsNone(BENCH_IO.counter_delta({"compact_stall": 5}, {"compact_stall": 2}))
        self.assertIsNone(BENCH_IO.counter_delta({"compact_stall": 5}, {}))

    def test_cold_gate_waits_for_current_device_io_and_ignores_historical_load(self):
        devices = [{"counters": [0] * 8 + [1] + [0] * 2}] + [{"counters": [0] * 11}] * 6
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (
                mock.patch.object(BENCH_IO.time, "sleep") as sleep,
                mock.patch.object(BENCH_IO.os, "getloadavg", return_value=(80, 40, 20)),
                mock.patch.object(BENCH_IO, "cpu_usage", return_value={"busy_fraction": 0, "iowait_fraction": 0}),
                mock.patch.object(BENCH_IO, "pressure_snapshot", return_value={}),
                mock.patch.object(BENCH_IO, "competing_processes", return_value=[]),
                mock.patch.object(BENCH_IO, "storage_snapshot", side_effect=devices),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                BENCH_IO.wait_for_quiet(root, root, quiet_seconds=12)
                self.assertEqual(sleep.call_count, 7)
                sleep.assert_called_with(2)
            observations = [json.loads(line) for line in (root / "quiet-samples.jsonl").read_text().splitlines()]
        self.assertEqual([item["steady"] for item in observations], [0, 1, 2, 3, 4, 5, 6])
        self.assertTrue(all(not item["historical_load_gate"] for item in observations))
        self.assertTrue(all(item["quiet_seconds"] == 12 for item in observations))
        self.assertFalse(observations[0]["storage_device_idle"])

    def test_noise_detection_catches_native_benchmarks_without_bench_in_their_name(self):
        self.assertTrue(
            BENCH_IO.is_competing_process(
                "to_arrow", "/home/ec2-user/vortex-2/target/release/deps/to_arrow-123\0--bench\0"
            )
        )
        self.assertTrue(BENCH_IO.is_competing_process("rustc", "/opt/rustc\0"))
        self.assertFalse(BENCH_IO.is_competing_process("sshd", "/usr/sbin/sshd\0"))

    def test_storage_metrics_distinguish_cumulative_counters_from_current_queue_gauge(self):
        before = {"device": "disk", "ts_ns": 0, "counters": [10, 0, 100, 200, 5, 0, 20, 20, 99, 600, 1000]}
        after = {
            "device": "disk",
            "ts_ns": 1_000_000_000,
            "counters": [14, 0, 108, 240, 6, 0, 24, 24, 0, 750, 2000],
        }
        result = BENCH_IO.storage_delta(before, after)
        self.assertEqual(result["read_operations"], 4)
        self.assertEqual(result["read_bytes"], 4096)
        self.assertEqual(result["write_bytes"], 2048)
        self.assertEqual(result["read_latency_mean_ms"], 10)
        self.assertEqual(result["queue_depth_mean"], 1)
        self.assertAlmostEqual(result["read_MB_s"], 0.004096)

    def test_storage_counter_reset_or_unavailable_device_is_not_reported_as_zero(self):
        before = {"device": "disk", "ts_ns": 0, "counters": [10] * 11}
        after = {"device": "disk", "ts_ns": 1_000_000, "counters": [0] * 11}
        self.assertIsNone(BENCH_IO.storage_delta(before, after))
        self.assertIsNone(BENCH_IO.storage_delta(None, after))
        self.assertIsNone(BENCH_IO.storage_delta(before, None))

    def test_required_device_rejects_other_storage_and_memory_filesystems(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for snapshot in ({"device": "other-disk"}, None):
                with (
                    self.subTest(snapshot=snapshot),
                    mock.patch.object(BENCH_IO, "storage_snapshot", return_value=snapshot),
                    self.assertRaisesRegex(ValueError, "expected test-ssd"),
                ):
                    BENCH_IO.storage_locations({"temporary": root}, "test-ssd")

    @unittest.skipUnless(sys.platform.startswith("linux"), "Requires Linux page-cache accounting")
    def test_eviction_removes_resident_pages_and_inspection_does_not_reload_them(self):
        # Honor a selected test disk; the repository is the fallback when /tmp is tmpfs.
        with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR", Path(__file__).parents[2])) as directory:
            root = Path(directory)
            path = root / "data.vortex"
            with path.open("wb") as file:
                file.write(b"x" * (2 * os.sysconf("SC_PAGE_SIZE") + 1))
                file.flush()
                os.fsync(file.fileno())
            (root / "empty.vortex").touch()
            self.assertEqual(BENCH_IO.page_cache_residency([path])["resident_pages"], 3)
            result = BENCH_IO.evict_local_files(root)
            self.assertEqual(result["after"]["files"], 2)
            self.assertEqual(result["after"]["resident_pages"], 0)
            self.assertEqual(BENCH_IO.page_cache_residency([path])["resident_pages"], 0)

    def test_eviction_rejects_a_still_warm_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "data.vortex").write_bytes(b"data")
            with (
                mock.patch.object(BENCH_IO, "page_cache_residency", return_value={"resident_fraction": 0.5}),
                mock.patch.object(BENCH_IO.os, "posix_fadvise"),
                self.assertRaisesRegex(ValueError, "dataset pages resident"),
            ):
                BENCH_IO.evict_local_files(root)

    def test_saved_binary_control_uses_the_same_runtime_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = SimpleNamespace(
                binary=root / "candidate",
                baseline_binary=root / "previous",
                data_root=root,
                output=root,
                scale_factor="10.0",
                iterations=3,
                diagnostic_iterations=1,
                discard_iterations=2,
                local_read_concurrency=32,
                driver_trace=False,
                evict_local_files=False,
            )
            commands = []
            environments = []

            def execute(command: list[str], **kwargs: object):
                commands.append(command[0])
                environments.append(kwargs["env"])
                Path(command[command.index("-o") + 1]).write_text(json.dumps({"all_runtimes": [1, 2, 3]}))

            with (
                mock.patch.object(BENCH_IO.subprocess, "run", side_effect=execute),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                BENCH_IO.run(args, "tpch-q19", ("v2-previous", "direct"), "test")
                BENCH_IO.run(args, "tpch-q19", ("v2-unlimited", "direct"), "test")
                BENCH_IO.run(args, "tpch-q19", ("v2-previous-boxed", "direct"), "boxed")
        self.assertEqual(commands, [str(args.baseline_binary), str(args.binary), str(args.baseline_binary)])
        self.assertEqual(environments[0], environments[1])
        self.assertEqual(environments[2]["VORTEX_SCAN_IO_INLINE_FETCH"], "0")
        self.assertEqual(environments[2]["VORTEX_SCAN_GROUP_SPARSE_PROJECTION"], "0")
        self.assertEqual(environments[2]["VORTEX_SCAN_IO_FORGET_PRUNED_PROJECTION"], "0")

    def test_archived_cost_flags_cannot_change_saved_binary_controls(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = SimpleNamespace(
                binary=root / "candidate",
                baseline_binary=root / "previous",
                data_root=root,
                output=root,
                scale_factor="10.0",
                iterations=1,
                diagnostic_iterations=1,
                discard_iterations=0,
                local_read_concurrency=32,
                driver_trace=False,
                evict_local_files=False,
            )
            commands = []
            environments = []

            def execute(command: list[str], **kwargs: object):
                commands.append(command[0])
                environments.append(kwargs["env"])
                Path(command[command.index("-o") + 1]).write_text(json.dumps({"all_runtimes": [1]}))

            ambient = {
                "VORTEX_SCAN_IO_COALESCE_COST_BYTES": "291100",
                "VORTEX_SCAN_IO_COALESCE_USE_PERCENT": "10",
                "VORTEX_SCAN_IO_COALESCE_SPECULATIVE_BYTES": "65536",
            }
            with (
                mock.patch.dict(os.environ, ambient),
                mock.patch.object(BENCH_IO.subprocess, "run", side_effect=execute),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                BENCH_IO.run(args, "tpch-q19", ("v2-previous-demanded", "direct"), "previous")
                BENCH_IO.run(args, "tpch-q19", ("v2-io-file-handle-demanded-native-32", "direct"), "current")
            self.assertEqual(commands, [str(args.baseline_binary), str(args.binary)])
            self.assertEqual(environments[0], environments[1])
            self.assertTrue(all(key not in environments[0] for key in ambient))

    def test_two_variants_alternate_first_position(self):
        configs = [("baseline", "direct"), ("candidate", "direct")]
        rounds = [BENCH_IO.configuration_order(configs, idx) for idx in range(4)]
        self.assertEqual([order[0][0] for order in rounds], ["baseline", "candidate", "candidate", "baseline"])
        self.assertEqual(configs[0][0], "baseline")

    def test_each_variant_visits_every_position(self):
        configs = [(name, "direct") for name in ("a", "b", "c")]
        rounds = [BENCH_IO.configuration_order(configs, idx) for idx in range(6)]
        for position in range(3):
            self.assertEqual(sorted(order[position][0] for order in rounds), ["a", "a", "b", "b", "c", "c"])

    def test_cpu_usage_keeps_iowait_out_of_busy_time_and_records_steal(self):
        usage = BENCH_IO.cpu_usage([0] * 8, [10, 0, 10, 50, 20, 0, 0, 10])
        self.assertAlmostEqual(usage["busy_fraction"], 0.3)
        self.assertAlmostEqual(usage["iowait_fraction"], 0.2)
        self.assertAlmostEqual(usage["steal_fraction"], 0.1)
        self.assertIsNone(BENCH_IO.cpu_usage(None, None))

    def test_contended_round_is_retained_but_excluded_and_diagnostics_follow_timings(self):
        calls = []
        variants = ["v2-project-early", "v2-project-late"]
        observed = iter([[(123, "cargo")], [], []])
        times = iter([100.0, 10.0, 12.0])

        def round_result(*_: object) -> tuple[dict[str, dict[str, Any]], list[tuple[int, str]]]:
            calls.append("timing")
            value = next(times)
            return {f"{v}-direct": {"measured_ms": [value], "median_ms": value} for v in variants}, next(observed)

        def diagnostic(*_: object, **kwargs: object) -> dict[str, int]:
            self.assertTrue(kwargs["diagnostic"])
            calls.append("diagnostic")
            return {"local_reads": 1, "requested_bytes": 4, "repeated_bytes": 0}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root / "binary"
            binary.write_bytes(b"test binary")
            output = root / "results"
            argv = [
                "bench-io.py",
                "--binary",
                str(binary),
                "--data-root",
                str(root),
                "--output",
                str(output),
                "--workloads",
                "tpch-q19",
                "--variants",
                *variants,
                "--modes",
                "direct",
                "--rounds",
                "2",
                "--iterations",
                "3",
                "--wait-for-quiet",
                "--repeat-contended-rounds",
            ]
            with (
                mock.patch.object(sys, "argv", argv),
                mock.patch.object(BENCH_IO, "measured_round", side_effect=round_result),
                mock.patch.object(BENCH_IO, "run", side_effect=diagnostic),
                mock.patch.object(BENCH_IO, "competing_processes", return_value=[]),
                mock.patch.object(BENCH_IO, "wait_for_quiet") as wait,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                BENCH_IO.main()
            summary = json.loads((output / "summary.json").read_text())
        self.assertEqual([a["accepted"] for a in summary["attempts"]], [False, True, True])
        for entry in summary["results"]["tpch-q19"].values():
            self.assertEqual(entry["median_ms"], 11)
            self.assertEqual(len(entry["rounds"]), 2)
        self.assertEqual(calls, ["timing"] * 3 + ["diagnostic"] * 2)
        self.assertEqual(wait.call_count, 2)


if __name__ == "__main__":
    unittest.main()
