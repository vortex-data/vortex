# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Regressions for IO-empty intervals and filter backlog attribution."""

import importlib.util
import sys
import unittest
from pathlib import Path

SPEC = importlib.util.spec_from_file_location("scan_overlap", Path(__file__).parents[1] / "summarize-scan-overlap.py")
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class TestScanOverlap(unittest.TestCase):
    def test_explicit_read_timestamps_exclude_earlier_allocation_and_later_resume(self):
        read = MODULE.parse_read(
            {
                "queued_unix_ns": 100,
                "started_unix_ns": 120,
                "reading_unix_ns": 120,
                "completed_unix_ns": 150,
                "allocation_ns": 60,
                "queue_ns": 20,
                "read_ns": 30,
                "resume_ns": 80,
                "length": 4,
            }
        )
        self.assertEqual(read, MODULE.Read(100, 120, 120, 150, 4))

    def test_legacy_read_timestamps_reconstruct_allocation_before_read(self):
        read = MODULE.parse_read(
            {"completed_unix_ns": 150, "allocation_ns": 10, "queue_ns": 20, "read_ns": 30, "length": 4}
        )
        self.assertEqual(read, MODULE.Read(90, 110, 120, 150, 4))

    def test_empty_interval_is_split_by_backlog_without_splitting_the_gap(self):
        reads = [MODULE.Read(0, 0, 0, 20, 4), MODULE.Read(80, 80, 80, 100, 4)]
        query = {"completed_unix_ns": 100, "callback_ns": 100, "query_idx": 6, "iteration": 2}
        result = MODULE.summarize(reads, [(20, 40), (40, 80)], query, [(0, 1), (50, 0)])
        idle = result["io_idle"]
        self.assertEqual(idle["interior_empty_percent"], 60)
        self.assertEqual(idle["longest_interior_gap_ms"], 60 / 1e6)
        self.assertEqual(idle["interior_empty_with_compute_ms"], 60 / 1e6)
        self.assertEqual(idle["empty_with_unstarted_splits_percent"], 30)

    def test_queued_read_counts_as_outstanding(self):
        reads = [MODULE.Read(0, 50, 50, 100, 4)]
        query = {"completed_unix_ns": 100, "callback_ns": 100, "query_idx": 6, "iteration": 2}
        result = MODULE.summarize(reads, [(0, 100)], query)
        self.assertEqual(result["io_idle"]["interior_empty_percent"], 0)
        self.assertIsNone(result["io_idle"]["empty_with_unstarted_splits_ms"])

    def test_excludes_startup_and_completed_read_tail(self):
        reads = [MODULE.Read(20, 20, 20, 80, 4)]
        query = {"completed_unix_ns": 100, "callback_ns": 100, "query_idx": 6, "iteration": 2}
        result = MODULE.summarize(reads, [], query)
        self.assertEqual(result["io_idle"]["read_phase_ms"], 60 / 1e6)
        self.assertEqual(result["io_idle"]["interior_empty_percent"], 0)
