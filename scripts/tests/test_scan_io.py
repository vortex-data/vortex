# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path
from typing import Any

SPEC = importlib.util.spec_from_file_location("scan_io", Path(__file__).parents[1] / "scan-io.py")
assert SPEC is not None and SPEC.loader is not None
SCAN_IO = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SCAN_IO)


def event(name: str, ts: int, **fields: str | int) -> dict[str, Any]:
    return {"event": name, "ts_ns": ts, "run": 1, **fields}


def admit() -> dict[str, Any]:
    return event("admit", 0, root=0, session=42, file=0, row_start=0, row_end=100)


class ScanIoTests(unittest.TestCase):
    def test_lifetime_counters_distinguish_missing_fields_from_measured_zero(self):
        measured = SCAN_IO.lifetime_profile(
            [event("scope_clear", 10, future_queues=0, ready_fetches=2)], [], {}, {}, 20
        )
        self.assertEqual(measured["scope_totals"]["ready_fetches"], 2)
        self.assertIn("future_queues", measured["scope_counters_available"])
        self.assertNotIn("inline_fetches", measured["scope_counters_available"])
        legacy = SCAN_IO.lifetime_profile([event("scope_clear", 10, interests=3)], [], {}, {}, 20)
        self.assertNotIn("future_queues", legacy["scope_counters_available"])
        self.assertFalse(legacy["optional_withdrawals"]["available"])
        no_withdrawals = SCAN_IO.lifetime_profile([event("scope_clear", 10, forgets=0)], [], {}, {}, 20)
        self.assertTrue(no_withdrawals["optional_withdrawals"]["available"])
        self.assertEqual(no_withdrawals["optional_withdrawals"]["interests"], 0)

    def test_withdrawal_closes_only_its_scope_and_reregistration_starts_an_episode(self):
        bindings = [
            {"source": 1, "read": 10, "session": 1, "intent": "Announce", "ts_ns": 1, "length": 4},
            {"source": 1, "read": 10, "session": 2, "intent": "Fetch", "ts_ns": 2, "length": 4},
            {"source": 1, "read": 10, "session": 1, "intent": "Announce", "ts_ns": 20, "length": 4},
        ]
        withdrawal = {"source": 1, "read": 10, "ts_ns": 10}
        profile = SCAN_IO.registration_profile(
            bindings,
            {
                1: {"end_ns": 30, "status": "completed", "forget_events": [withdrawal]},
                2: {"end_ns": 40, "status": "completed"},
            },
            0,
            100,
        )
        self.assertEqual(profile["scope_interests"], 3)
        self.assertEqual(profile["scope_end_sources"], {"scope_forget": 1, "root_lifecycle": 2})
        self.assertEqual(profile["scope_references_per_registration"]["max"], 2)
        self.assertEqual(profile["scope_interest_bytes"]["mean"], 2.28)
        self.assertEqual(profile["unused_optional_by_scope_status"]["forgotten"]["interests"], 1)
        self.assertEqual(profile["unused_optional_by_scope_status"]["completed"]["interests"], 1)
        self.assertEqual(profile["unused_registrations_with_only_completed_scopes"], 0)

    def test_withdrawal_phase_and_reference_counts_do_not_claim_saved_bytes(self):
        events = [
            event("scope_forget", 5, session=1, source=1, read=0, length=4, references=2),
            event("scope_forget", 15, session=1, source=1, read=1, length=8, references=1),
            event("scope_forget", 25, session=1, source=1, read=2, length=16, references=1),
            event("scope_forget", 30, session=1, source=1, read=3, length=32, references=1),
        ]
        reads = {(1, 10): {"ts_ns": 10, "end_ns": 20}}
        members = {(1, read): (1, 10) for read in (0, 1, 2)}
        profile = SCAN_IO.lifetime_profile(events, [], reads, members, 100)
        withdrawals = profile["optional_withdrawals"]
        self.assertTrue(profile["available"])
        self.assertEqual(withdrawals["logical_bytes"], 60)
        self.assertEqual(withdrawals["last_reference_observed"], 3)
        self.assertEqual(withdrawals["other_references_observed"], 1)
        self.assertEqual(withdrawals["phase_reference_counts"]["before_physical_start"]["shared"], 1)
        self.assertNotIn("before_physical_start", withdrawals["last_reference_phase_logical_bytes"])
        self.assertEqual(
            withdrawals["phase_logical_bytes"],
            {
                "before_physical_start": 4,
                "inflight_before_physical_completion": 8,
                "after_physical_completion": 16,
                "no_started_read_in_window": 32,
            },
        )

    def test_analyze_uses_exact_withdrawal_even_without_admission(self):
        summary, _ = SCAN_IO.analyze(
            [
                {"event": "query_begin", "ts_ns": 0},
                event("binding", 1, session=1, source=1, read=10, owner=0, request=0, intent="Announce", length=4),
                event("scope_forget", 10, session=1, source=1, read=10, length=4, references=1),
                {"event": "query_end", "ts_ns": 100},
            ],
            build_trace=False,
        )
        self.assertEqual(summary["totals"]["registration_lifetimes"]["scope_end_sources"], {"scope_forget": 1})
        self.assertEqual(summary["totals"]["registration_lifetimes"]["scope_interest_bytes"]["mean"], 0.36)

    def test_wait_ends_when_work_is_runnable_after_multiple_deliveries(self):
        events = [
            admit(),
            event("advance_begin", 0),
            event("request", 10, root=0, owner=0, request=0, intent="Fetch", offset=0, length=4),
            event("request", 12, root=0, owner=0, request=1, intent="Fetch", offset=4, length=4),
            event("park", 15, owner=0),
            event("waiting", 20, ready=0, parked=1),
            event("completion", 30, owner=0, request=0),
            event("completion", 50, owner=0, request=1),
            event("unpark", 55, owner=0),
            event("advance_begin", 70),
            event("compute", 80, owner=0, start_ns=72),
            event("root_done", 90),
            event("end", 100),
        ]
        summary, trace = SCAN_IO.analyze(events)
        run = summary["runs"][0]
        self.assertEqual(run["blocked_ns"], 35)
        self.assertEqual(run["wake_delay_ns"], 15)
        self.assertEqual(run["owner_wait_ns"], 40)
        self.assertEqual(run["compute_ns"], 8)
        self.assertEqual(run["active_ns"], 40)
        self.assertEqual(run["fetch_latency_ns"]["max"], 38)
        self.assertFalse(run["partial"])
        self.assertTrue(trace["traceEvents"])

    def test_coalesced_read_binds_even_when_announced_before_admission(self):
        events = [
            {
                "event": "binding",
                "ts_ns": 0,
                "session": 42,
                "owner": 0,
                "request": 0,
                "intent": "Announce",
                "source": 10,
                "read": 7,
                "uri": "file with spaces",
            },
            admit(),
            event("advance_begin", 1),
            event("request", 2, root=0, owner=0, request=1, intent="Fetch", offset=4, length=4),
            {
                "event": "binding",
                "ts_ns": 3,
                "session": 42,
                "owner": 0,
                "request": 1,
                "intent": "Fetch",
                "source": 10,
                "read": 7,
                "uri": "file with spaces",
            },
            {"event": "read_start", "ts_ns": 5, "source": 10, "physical": 6, "offset": 0, "length": 12},
            {"event": "read_member", "ts_ns": 6, "source": 10, "physical": 6, "read": 7},
            {"event": "read_end", "ts_ns": 10, "source": 10, "physical": 6, "success": "true"},
            event("completion", 12, owner=0, request=1),
            event("root_done", 13),
            event("end", 14),
        ]
        summary, _ = SCAN_IO.analyze(events)
        run = summary["runs"][0]
        self.assertEqual(run["unbound_fetches"], 0)
        self.assertEqual(run["requests"][0]["physical"], (10, 6))
        self.assertEqual(summary["physical_reads"][0]["length"], 12)
        self.assertEqual(run["roots"][0]["uri"], "file with spaces")
        self.assertEqual(summary["totals"]["physical_bytes"], 12)

    def test_cancelled_wait_is_reported_as_partial_without_inventing_completion(self):
        summary, trace = SCAN_IO.analyze(
            [
                admit(),
                event("advance_begin", 0),
                event("request", 1, root=0, owner=0, request=0, intent="Fetch", offset=0, length=4),
                event("park", 2, owner=0, root=0),
                event("waiting", 3),
                event("cancel", 10, root=0),
                event("end", 11),
            ]
        )
        run = summary["runs"][0]
        self.assertTrue(run["cancelled"])
        self.assertTrue(run["partial"])
        self.assertEqual(run["undelivered_fetches"], 1)
        self.assertEqual(run["blocked_ns"], 0)
        self.assertEqual(run["unfinished_owner_wait_ns"], 8)
        fetch = next(e for e in trace["traceEvents"] if e["name"] == "fetch (unfinished)")
        self.assertTrue(fetch["args"]["censored"])
        self.assertNotIn("delivered_ns", run["requests"][0])

    def test_parser_sorts_events_and_chooses_last_complete_iteration(self):
        log = "\n".join(
            [
                "IO_ITERATION_BEGIN query=19",
                'DEBUG scan IO ts_ns=9 event="binding" session=42 uri="a b"',
                'DEBUG scan driver run=1 ts_ns=2 event="start"',
                "IO_ITERATION_END query=19",
                "IO_ITERATION_BEGIN query=19",
                'DEBUG scan driver run=2 ts_ns=10 event="start"',
                "IO_ITERATION_END query=19",
                'DEBUG scan driver run=3 ts_ns=20 event="end"',
            ]
        )
        self.assertEqual(SCAN_IO.parse_events(log)[0]["run"], 2)
        first = SCAN_IO.parse_events(log, 0)
        self.assertEqual(first[0]["ts_ns"], 2)
        self.assertEqual(first[1]["uri"], "a b")

    def test_union_does_not_double_count_concurrent_reads(self):
        self.assertEqual(SCAN_IO.union_ns([(0, 10), (5, 15), (20, 22)]), 17)

    def test_queue_depth_is_time_weighted_and_includes_zero(self):
        profile, series = SCAN_IO.depth_profile([(10, 20), (15, 25)], 0, 100)
        self.assertEqual(profile["histogram_ns"], {0: 85, 1: 10, 2: 5})
        self.assertEqual(profile["mean"], 0.2)
        self.assertEqual((profile["p50"], profile["p95"], profile["p99"]), (0, 1, 2))
        self.assertEqual(profile["peak"], 2)
        self.assertEqual(series, [(0, 0), (10, 1), (15, 2), (20, 1), (25, 0)])
        byte_depth, _ = SCAN_IO.depth_profile([(10, 20), (15, 25)], 0, 100, weights=[4, 6])
        self.assertEqual(byte_depth["histogram_ns"], {0: 85, 4: 5, 6: 5, 10: 5})
        self.assertEqual(byte_depth["mean"], 1)
        self.assertEqual(byte_depth["peak"], 10)

    def test_dfs_and_startup_follow_child_emission_and_parent_continuation(self):
        summary, _ = SCAN_IO.analyze(
            [
                {"event": "query_begin", "ts_ns": 0},
                admit(),
                event("spawn", 1, root=0, owner=0, kind="planner", stage="announce", state="NeedsCompute"),
                event("compute", 3, start_ns=2, owner=0, outcome="planner", state="NeedsCompute"),
                event("spawn", 4, root=0, owner=1, kind="planner", stage="filter", state="NeedsCompute"),
                event("compute", 7, start_ns=6, owner=1, outcome="continue", state="NeedsCompute"),
                event("compute", 9, start_ns=8, owner=1, outcome="continue", state="NeedsCompute"),
                event("compute", 11, start_ns=10, owner=1, outcome="io", state="Waiting"),
                event("park", 12, root=0, owner=1),
                event("compute", 14, start_ns=13, owner=0, outcome="done", state="Done"),
                event("retire", 15, owner=0),
                event("unpark", 20, owner=1, state="NeedsCompute"),
                event("compute", 22, start_ns=21, owner=1, outcome="morsel", state="Done"),
                event("spawn", 23, root=0, owner=2, kind="morsel", stage="projection", state="NeedsCompute"),
                event("retire", 24, owner=1),
                event("compute", 31, start_ns=30, owner=2, outcome="batch", rows=2, state="NeedsCompute"),
                event("compute", 36, start_ns=35, owner=2, outcome="done", state="Done"),
                event("retire", 37, owner=2),
                event("end", 40),
                {"event": "query_end", "ts_ns": 50},
            ],
            build_trace=False,
        )
        run = summary["runs"][0]
        self.assertEqual([w["dfs_path"] for w in run["owners"].values()], [[0], [0, 0], [0, 0, 0]])
        self.assertEqual([w["parent_owner"] for w in run["owners"].values()], [None, 0, 1])
        self.assertEqual(run["max_consecutive_visits_with_other_work_ready"], 3)
        self.assertEqual(run["max_consecutive_span_with_other_work_ready_ns"], 5)
        self.assertEqual(summary["query"]["dfs"]["max_depth"], 2)
        self.assertEqual(summary["query"]["dfs"]["owners_unknown"], 0)
        self.assertEqual(summary["query"]["startup_ns"]["first_compute"], 2)
        self.assertEqual(summary["query"]["startup_ns"]["first_morsel_compute"], 30)
        self.assertEqual(summary["query"]["startup_ns"]["first_batch_emitted"], 31)
        self.assertEqual(summary["stages"]["filter"]["root_first_compute_delay_ns"]["max"], 6)
        self.assertEqual(summary["stages"]["projection"]["owner_first_compute_delay_ns"]["max"], 7)
        self.assertEqual(summary["query"]["queues"]["ready_morsels"]["busy_ns"], 11)
        self.assertEqual(summary["query"]["queues"]["parked_planners"]["busy_ns"], 8)
        self.assertEqual(summary["stages"]["projection"]["batch_rows"]["max"], 2)
        interference = summary["stages"]["announce"]["scheduling_interference"]
        self.assertEqual(interference["ancestor_continuation_visits"], 3)
        self.assertEqual(interference["ancestor_continuation_compute_ns"], 3)
        self.assertNotIn("peer_branch_visits", interference)
        self.assertNotIn("earlier_ready_priority_bypassed", summary["totals"]["dfs_choices"])

    def test_repeated_visits_to_the_only_ready_owner_do_not_count_as_delaying_other_work(self):
        summary, _ = SCAN_IO.analyze(
            [
                admit(),
                event("spawn", 1, root=0, owner=0, kind="morsel", state="NeedsCompute"),
                event("compute", 3, start_ns=2, owner=0, outcome="continue", state="NeedsCompute"),
                event("compute", 5, start_ns=4, owner=0, outcome="continue", state="NeedsCompute"),
                event("compute", 7, start_ns=6, owner=0, outcome="done", state="Done"),
                event("end", 8),
            ],
            build_trace=False,
        )
        self.assertEqual(summary["runs"][0]["max_consecutive_owner_visits"], 3)
        self.assertEqual(summary["runs"][0]["max_consecutive_visits_with_other_work_ready"], 0)

    def test_cancelled_ready_work_ends_at_cancellation_and_has_no_fake_startup(self):
        summary, _ = SCAN_IO.analyze(
            [
                {"event": "query_begin", "ts_ns": 0},
                admit(),
                event("spawn", 10, root=0, owner=0, kind="morsel", stage="projection", state="NeedsCompute"),
                event("cancel", 30, root=0),
                event("end", 40),
                {"event": "query_end", "ts_ns": 100},
            ],
            build_trace=False,
        )
        queues = summary["query"]["queues"]
        self.assertEqual(queues["ready_morsels"]["busy_ns"], 20)
        self.assertEqual(queues["live_morsels"]["mean"], 0.2)
        self.assertEqual(summary["query"]["gaps"]["ready_morsels_without_compute_ns"], 20)
        self.assertIsNone(summary["query"]["startup_ns"]["first_compute"])
        self.assertIsNone(summary["stages"]["projection"]["first_compute_query_ns"])
        self.assertEqual(summary["stages"]["projection"]["owners_without_compute"], 1)

    def test_byte_coverage_and_completed_delivery_queue_include_censored_fetches(self):
        summary, _ = SCAN_IO.analyze(
            [
                {"event": "query_begin", "ts_ns": 0},
                admit(),
                event("request", 2, root=0, owner=0, request=0, intent="Fetch", offset=1, length=2),
                event("request", 3, root=0, owner=0, request=1, intent="Fetch", offset=7, length=3),
                {
                    "event": "binding",
                    "ts_ns": 4,
                    "session": 42,
                    "owner": 0,
                    "request": 0,
                    "intent": "Fetch",
                    "source": 1,
                    "read": 10,
                },
                {
                    "event": "binding",
                    "ts_ns": 4,
                    "session": 42,
                    "owner": 0,
                    "request": 1,
                    "intent": "Fetch",
                    "source": 1,
                    "read": 11,
                },
                {"event": "read_queued", "ts_ns": 5, "source": 1, "physical": 99},
                {"event": "read_start", "ts_ns": 10, "source": 1, "physical": 99, "offset": 0, "length": 10},
                {
                    "event": "read_member",
                    "ts_ns": 11,
                    "source": 1,
                    "physical": 99,
                    "read": 10,
                    "offset": 0,
                    "length": 3,
                },
                {
                    "event": "read_member",
                    "ts_ns": 11,
                    "source": 1,
                    "physical": 99,
                    "read": 11,
                    "offset": 7,
                    "length": 3,
                },
                {"event": "read_end", "ts_ns": 20, "source": 1, "physical": 99, "success": "true"},
                event("completion", 25, owner=0, request=0),
                event("cancel", 30, root=0),
                event("end", 35),
                {"event": "query_end", "ts_ns": 40},
            ],
            build_trace=False,
        )
        coverage = summary["totals"]["physical_coverage"]
        self.assertEqual(coverage["coalesced_member_union_bytes"], 6)
        self.assertEqual(coverage["coalescing_gap_bytes"], 4)
        self.assertEqual(coverage["fetch_requested_union_bytes"], 5)
        self.assertEqual(coverage["fetch_delivered_union_bytes"], 2)
        self.assertEqual(coverage["started_bytes_not_fetched_in_window"], 5)
        self.assertEqual(coverage["started_bytes_not_delivered_to_fetch_in_window"], 8)
        queues = summary["query"]["queues"]
        self.assertEqual(queues["physical_queued"]["busy_ns"], 5)
        self.assertEqual(queues["completed_reads_awaiting_fetch_delivery"]["busy_ns"], 10)
        self.assertEqual(queues["completed_reads_awaiting_fetch_delivery"]["mean"], 0.375)
        self.assertEqual(summary["query"]["request_byte_depth"]["physical_in_flight"]["peak"], 10)
        self.assertEqual(summary["query"]["request_byte_depth"]["completed_fetches_awaiting_delivery"]["mean"], 1)

    def test_streaming_parser_and_queue_only_trace(self):
        log = [
            "IO_ITERATION_BEGIN query=19 ts_ns=0\n",
            'DEBUG scan driver run=1 ts_ns=1 event="admit" root=0 session=42\n',
            'DEBUG scan driver run=1 ts_ns=2 event="spawn" root=0 owner=0 kind="morsel" '
            'stage="projection" state="NeedsCompute"\n',
            'DEBUG scan driver run=1 ts_ns=4 event="compute" start_ns=3 owner=0 outcome="done" state="Done"\n',
            'DEBUG scan driver run=1 ts_ns=5 event="end"\n',
            "IO_ITERATION_END query=19 ts_ns=6 rows=1\n",
        ]
        events = SCAN_IO.parse_events(iter(log))
        self.assertEqual(events, SCAN_IO.parse_events("".join(log)))
        summary, trace = SCAN_IO.analyze(events, build_trace=False, build_queue_trace=True)
        self.assertEqual(summary["query"]["startup_ns"]["first_morsel_compute"], 3)
        self.assertEqual({e["ph"] for e in trace["traceEvents"]}, {"M", "C", "i"})
        _, no_trace = SCAN_IO.analyze(events, build_trace=False)
        self.assertEqual(no_trace["traceEvents"], [])

    def test_missing_spawn_history_leaves_dfs_unknown(self):
        summary, _ = SCAN_IO.analyze(
            [
                event("spawn", 1, root=0, owner=0, kind="morsel", state="NeedsCompute"),
                event("compute", 3, start_ns=2, owner=0, outcome="done", state="Done"),
                event("end", 4),
            ],
            build_trace=False,
        )
        self.assertEqual(summary["query"]["dfs"]["owners_unknown"], 1)
        self.assertEqual(summary["query"]["dfs"]["steps_unknown"], 1)
        self.assertIsNone(summary["query"]["dfs"]["max_depth"])

    def test_compact_counters_retain_peaks_troughs_and_endpoints_at_their_timestamps(self):
        points = [(0, 0), (1, 2), (2, 9), (3, 6), (4, 3), (5, 4), (10, 0)]
        self.assertEqual(SCAN_IO.compact_counter_series(points, 10), [(0, 0), (2, 9), (5, 4), (10, 0)])
        self.assertEqual(SCAN_IO.compact_counter_series(points, 0), points)

    def test_joint_states_partition_overlapping_work_and_waits(self):
        profile = SCAN_IO.joint_state_time(
            {
                "compute": [(10, 20)],
                "runnable": [(20, 40)],
                "delivery_ready": [(35, 50)],
                "physical_IO": [(30, 80)],
                "pending_fetch": [(-10, 110)],
            },
            0,
            100,
        )
        self.assertEqual(profile["accounted_ns"], 100)
        self.assertEqual(
            profile["categories_ns"],
            {
                "compute_running": 10,
                "runnable_without_compute": 20,
                "delivery_ready_without_runnable_or_compute": 10,
                "IO_without_ready_delivery_or_compute": 30,
                "fetch_pending_without_IO_or_ready_work": 30,
            },
        )
        self.assertEqual(sum(s["elapsed_ns"] for s in profile["states"]), 100)

    def test_progress_milestones_use_work_weight_and_observed_denominator(self):
        progress = SCAN_IO.progress_profile([(40, 100), (10, 1), (20, 1)], 5)
        self.assertEqual(progress["observed_total"], 102)
        self.assertEqual(progress["time_to_fraction_ns"], {10: 35, 50: 35, 90: 35, 99: 35, 100: 35})
        self.assertEqual(SCAN_IO.progress_profile([], 0)["time_to_fraction_ns"], {})

    def test_dfs_peer_delay_is_distinct_from_ancestor_continuation_and_checks_priority(self):
        summary, _ = SCAN_IO.analyze(
            [
                admit(),
                event("spawn", 1, root=0, owner=0, kind="planner", stage="earlier", state="NeedsCompute"),
                event("admit", 2, root=1, session=43),
                event("spawn", 3, root=1, owner=1, kind="planner", stage="later", state="NeedsCompute"),
                event("compute", 5, start_ns=4, owner=1, outcome="done", state="Done"),
                event("compute", 7, start_ns=6, owner=0, outcome="done", state="Done"),
                event("end", 10),
            ],
            build_trace=False,
        )
        interference = summary["stages"]["earlier"]["scheduling_interference"]
        self.assertEqual(interference["peer_branch_visits"], 1)
        self.assertEqual(interference["peer_branch_compute_ns"], 1)
        self.assertEqual(interference["peer_branch_max_ready_age_ns"], 3)
        self.assertNotIn("ancestor_continuation_visits", interference)
        self.assertEqual(summary["totals"]["dfs_choices"]["earlier_ready_priority_bypassed"], 1)

    def test_release_metrics_use_the_completion_that_makes_work_runnable(self):
        summary, _ = SCAN_IO.analyze(
            [
                {"event": "query_begin", "ts_ns": 0},
                admit(),
                event("spawn", 1, root=0, owner=0, kind="morsel", stage="projection", state="NeedsCompute"),
                event("compute", 3, start_ns=2, owner=0, outcome="io", state="Waiting"),
                event("request", 4, root=0, owner=0, request=0, intent="Fetch", offset=0, length=2),
                event("request", 5, root=0, owner=0, request=1, intent="Fetch", offset=2, length=2),
                {
                    "event": "binding",
                    "ts_ns": 5,
                    "session": 42,
                    "owner": 0,
                    "request": 0,
                    "intent": "Fetch",
                    "source": 1,
                    "read": 10,
                },
                {
                    "event": "binding",
                    "ts_ns": 5,
                    "session": 42,
                    "owner": 0,
                    "request": 1,
                    "intent": "Fetch",
                    "source": 1,
                    "read": 11,
                },
                event("park", 6, root=0, owner=0),
                {"event": "read_start", "ts_ns": 7, "source": 1, "physical": 99, "offset": 0, "length": 4, "uri": "/f"},
                {"event": "read_member", "ts_ns": 8, "source": 1, "physical": 99, "read": 10, "offset": 0, "length": 2},
                {"event": "read_member", "ts_ns": 8, "source": 1, "physical": 99, "read": 11, "offset": 2, "length": 2},
                {
                    "event": "local_read",
                    "ts_ns": 9,
                    "start_ns": 7,
                    "path": "/f",
                    "offset": 0,
                    "length": 4,
                    "get_ns": 0,
                    "prepare_ns": 0,
                    "allocation_ns": 0,
                    "queue_ns": 0,
                    "read_ns": 1,
                    "resume_ns": 0,
                },
                {"event": "read_end", "ts_ns": 9, "source": 1, "physical": 99, "success": "true"},
                event("advance_begin", 9),
                event("completion", 10, owner=0, request=0),
                event("completion", 15, owner=0, request=1),
                event("unpark", 16, owner=0, state="NeedsCompute"),
                event("compute", 22, start_ns=20, owner=0, outcome="batch", rows=4, state="NeedsCompute"),
                event("batch", 23),
                event("cancel", 30, root=0),
                event("end", 31),
                {"event": "query_end", "ts_ns": 40},
            ],
            build_trace=False,
        )
        release = summary["runs"][0]["releases"][0]
        self.assertEqual(release["release_request"], 1)
        self.assertEqual(release["deliveries_during_park"], 2)
        self.assertEqual(release["file_completion_to_delivery_ns"], 6)
        self.assertEqual(release["pread_to_delivery_ns"], 7)
        self.assertEqual(release["file_ready_while_driver_advancing_ns"], 6)
        self.assertEqual(release["file_ready_outside_driver_advance_ns"], 0)
        self.assertEqual(release["file_ready_while_run_computing_ns"], 0)
        self.assertEqual(release["unpark_to_compute_ns"], 4)
        self.assertEqual(release["next_compute_outcome"], "batch")
        self.assertEqual(summary["totals"]["releases"]["computed_after_release"], 1)
        self.assertEqual(summary["totals"]["completion_fanout"]["fetch_requests_per_read"]["max"], 2)
        self.assertEqual(summary["totals"]["completion_fanout"]["runnable_releases_per_read"]["max"], 1)
        self.assertEqual(summary["totals"]["async_resume_to_file_completion_ns"]["max"], 1)
        self.assertEqual(summary["stages"]["projection"]["ready_wait_ended_without_compute_ns"]["max"], 8)
        self.assertEqual(summary["query"]["progress"]["emitted_rows"]["observed_total"], 4)

    def test_release_followed_by_cancellation_keeps_compute_missing(self):
        summary, _ = SCAN_IO.analyze(
            [
                admit(),
                event("spawn", 1, root=0, owner=0, kind="morsel", stage="projection", state="Waiting"),
                event("request", 2, root=0, owner=0, request=0, intent="Fetch", offset=0, length=2),
                event("park", 3, root=0, owner=0),
                event("completion", 5, owner=0, request=0),
                event("unpark", 6, owner=0, state="NeedsCompute"),
                event("cancel", 8, root=0),
                event("end", 10),
            ],
            build_trace=False,
        )
        release = summary["runs"][0]["releases"][0]
        self.assertNotIn("first_compute_ns", release)
        self.assertEqual(release["compute_not_observed_until_ns"], 8)
        self.assertEqual(summary["totals"]["releases"]["runnable_without_observed_compute"], 1)
        self.assertEqual(summary["stages"]["projection"]["ready_wait_ended_without_compute_ns"]["max"], 2)

    def test_registration_interests_deduplicate_owners_and_keep_other_scopes_alive(self):
        bindings = [
            {"source": 1, "read": 10, "session": 1, "owner": 0, "intent": "Announce", "ts_ns": 1, "length": 4},
            {"source": 1, "read": 10, "session": 1, "owner": 1, "intent": "Announce", "ts_ns": 2, "length": 4},
            {"source": 1, "read": 11, "session": 1, "owner": 0, "intent": "Announce", "ts_ns": 3, "length": 8},
            {"source": 1, "read": 10, "session": 2, "owner": 0, "intent": "Announce", "ts_ns": 5, "length": 4},
            {"source": 1, "read": 10, "session": 2, "owner": 0, "intent": "Fetch", "ts_ns": 6, "length": 4},
            {"source": 1, "read": 12, "session": 3, "owner": 0, "intent": "Announce", "ts_ns": 7, "length": 2},
        ]
        profile = SCAN_IO.registration_profile(
            bindings, {1: {"end_ns": 20, "status": "completed"}, 2: {"end_ns": 30, "status": "cancelled"}}, 0, 100
        )
        self.assertEqual(profile["scope_interests"], 4)
        self.assertEqual(profile["registrations"], 3)
        self.assertEqual(profile["registrations_shared_by_overlapping_scopes"], 1)
        self.assertEqual(profile["scope_references_per_registration"]["max"], 2)
        self.assertEqual(profile["scope_interest_bytes"]["peak"], 18)
        self.assertEqual(profile["with_unknown_scope_end"], 1)
        self.assertEqual(
            profile["unused_optional_by_scope_status"]["completed"], {"interests": 2, "interest_bytes": 12}
        )
        self.assertEqual(profile["unused_registrations_with_only_completed_scopes"], 1)
        self.assertFalse(profile["unused_completed_scope_physical"]["binding_inventory_available"])

    def test_unused_completed_scope_bytes_exclude_other_members_and_unknown_scopes(self):
        bindings = [
            {"source": 1, "read": 10, "session": 1, "intent": "Announce", "ts_ns": 1, "length": 4},
            {"source": 1, "read": 10, "session": 2, "intent": "Fetch", "ts_ns": 5, "length": 4},
            {"source": 1, "read": 11, "session": 1, "intent": "Announce", "ts_ns": 2, "length": 8},
            {"source": 1, "read": 12, "session": 3, "intent": "Announce", "ts_ns": 3, "length": 2},
            {"source": 1, "read": 13, "session": 1, "intent": "Announce", "ts_ns": 4, "length": 1},
        ]
        physical = [
            {
                "source": 1,
                "ts_ns": 6,
                "end_ns": 15,
                "offset": 0,
                "length": 16,
                "members": [
                    {"read": 10, "offset": 0, "length": 4},
                    {"read": 11, "offset": 2, "length": 8},
                    {"read": 12, "offset": 10, "length": 2},
                ],
            }
        ]
        profile = SCAN_IO.registration_profile(
            bindings,
            {1: {"end_ns": 20, "status": "completed"}, 2: {"end_ns": 30, "status": "completed"}},
            0,
            100,
            physical,
        )
        coverage = profile["unused_completed_scope_physical"]
        self.assertEqual(profile["unused_registrations_with_only_completed_scopes"], 2)
        self.assertEqual(coverage["registrations_in_started_reads"], 1)
        self.assertEqual(coverage["registrations_absent_from_started_reads"], 1)
        self.assertEqual(coverage["selected_member_union_bytes"], 8)
        self.assertEqual(coverage["exclusive_of_other_member_ranges_bytes"], 6)
        self.assertEqual(coverage["completed_exclusive_bytes"], 6)
        self.assertEqual(coverage["unfinished_exclusive_bytes"], 0)
        self.assertEqual(
            coverage["member_lifecycle_counts"],
            {
                "started_before_last_scope_end": 1,
                "completed_before_last_scope_end": 1,
            },
        )

    def test_compact_counter_resolution_does_not_change_measured_queue_depth_or_startup(self):
        events = [
            {"event": "query_begin", "ts_ns": 0},
            admit(),
            event("spawn", 1, root=0, owner=0, kind="morsel", stage="projection", state="NeedsCompute"),
            event("compute", 3, start_ns=2, owner=0, outcome="continue", state="NeedsCompute"),
            event("compute", 5, start_ns=4, owner=0, outcome="continue", state="NeedsCompute"),
            event("compute", 7, start_ns=6, owner=0, outcome="done", state="Done"),
            event("end", 8),
            {"event": "query_end", "ts_ns": 100},
        ]
        exact, exact_trace = SCAN_IO.analyze(events, build_trace=False, build_queue_trace=True, queue_resolution_ns=0)
        compact, compact_trace = SCAN_IO.analyze(
            events, build_trace=False, build_queue_trace=True, queue_resolution_ns=100
        )
        self.assertEqual(exact["query"], compact["query"])
        self.assertEqual(exact["stages"], compact["stages"])
        self.assertLess(len(compact_trace["traceEvents"]), len(exact_trace["traceEvents"]))
        end = next(e for e in compact_trace["traceEvents"] if e["name"] == "query end")
        self.assertEqual(end["ts"], 0.1)

    def test_exact_scope_clear_splits_reused_session_interests_into_episodes(self):
        bindings = [
            {"source": 1, "read": 10, "session": 1, "intent": "Announce", "ts_ns": 1, "length": 4},
            {"source": 1, "read": 10, "session": 1, "intent": "Fetch", "ts_ns": 20, "length": 4},
        ]
        profile = SCAN_IO.registration_profile(
            bindings,
            {
                1: {
                    "end_ns": 100,
                    "status": "unknown",
                    "clear_events": [
                        {"ts_ns": 10, "reason": "drop"},
                        {"ts_ns": 30, "reason": "clear"},
                    ],
                },
            },
            0,
            100,
        )
        self.assertEqual(profile["scope_interests"], 2)
        self.assertEqual(profile["scope_end_sources"], {"scope_clear": 2})
        self.assertEqual(profile["with_unknown_scope_end"], 0)
        self.assertEqual(profile["unused_optional_by_scope_status"]["dropped"]["interests"], 1)
        self.assertEqual(profile["scope_interest_bytes"]["mean"], 0.76)
        self.assertEqual(profile["unused_registrations_with_only_completed_scopes"], 0)

    def test_lifetime_metrics_preserve_wake_bursts_and_censored_notifications(self):
        events = [
            {"event": "session_wake", "ts_ns": 5, "session": 1},
            {"event": "session_wake", "ts_ns": 7, "session": 1},
            {"event": "session_poll_begin", "ts_ns": 10, "session": 1, "method": "completion"},
            {"event": "session_wake", "ts_ns": 11, "session": 1},
            {"event": "session_poll", "ts_ns": 12, "session": 1, "outcome": "pending"},
            {"event": "session_poll_begin", "ts_ns": 20, "session": 1, "method": "sweep"},
            {"event": "session_poll", "ts_ns": 23, "session": 1, "outcome": "ready"},
            {"event": "session_wake", "ts_ns": 25, "session": 1},
        ]
        profile = SCAN_IO.lifetime_profile(events, [], {}, {}, 100)
        self.assertEqual(profile["wake_to_next_session_poll_ns"]["count"], 3)
        self.assertEqual(profile["wake_to_next_session_poll_ns"]["max"], 9)
        self.assertEqual(profile["first_wake_in_burst_to_next_poll_ns"]["count"], 2)
        self.assertEqual(profile["wake_without_subsequent_poll_age_ns"]["count"], 1)
        self.assertEqual(profile["wake_without_subsequent_poll_age_ns"]["max"], 75)
        self.assertEqual(profile["wakes_during_session_poll"], 1)
        self.assertEqual(profile["session_poll_duration_ns"]["max"], 3)
        self.assertEqual(profile["session_poll_methods"], {"completion": 1, "sweep": 1})
        self.assertEqual(profile["session_poll_ends_without_begin"], 0)

    def test_last_reference_drop_distinguishes_inflight_from_completed_retention(self):
        bindings = [{"source": 1, "read": n, "length": 4} for n in range(4)]
        reads = {
            (1, 10): {"ts_ns": 5, "end_ns": 20},
            (1, 11): {"ts_ns": 5, "end_ns": 40},
            (1, 12): {"ts_ns": 5, "end_ns": 10},
        }
        members = {(1, 0): (1, 10), (1, 1): (1, 11), (1, 2): (1, 12)}
        events = [
            {"event": "registration_drop", "source": 1, "read": 0, "ts_ns": 30},
            {"event": "registration_drop", "source": 1, "read": 1, "ts_ns": 30},
            {"event": "registration_drop", "source": 1, "read": 3, "ts_ns": 30},
        ]
        profile = SCAN_IO.lifetime_profile(events, bindings, reads, members, 100)
        self.assertEqual(
            profile["drop_phase_counts"],
            {
                "after_physical_completion": 1,
                "inflight_before_physical_completion": 1,
                "no_started_read_in_window": 1,
            },
        )
        self.assertEqual(profile["completed_registration_retention_ns"]["max"], 10)
        self.assertEqual(profile["last_reference_to_physical_completion_ns"]["max"], 10)
        self.assertEqual(profile["completed_member_retention_censored_ns"]["max"], 90)
        self.assertEqual(profile["completed_member_interest_byte_ns"], 400)

    def test_selection_decisions_count_unchanged_and_fully_rejected_predicates(self):
        summary, _ = SCAN_IO.analyze(
            [
                admit(),
                event("spawn", 1, root=0, owner=0, kind="planner", stage="pruning", state="NeedsCompute"),
                event("selection", 10, root=0, owner=0, revision=1, before_rows=100, selected_rows=100),
                event("selection", 20, root=0, owner=0, revision=2, before_rows=100, selected_rows=0),
                event("retire", 30, owner=0),
                event("end", 40),
            ],
            build_trace=False,
        )
        selection = summary["totals"]["selection_decisions"]["pruning"]
        self.assertEqual(selection["decisions"], 2)
        self.assertEqual(selection["unchanged"], 1)
        self.assertEqual(selection["fully_rejected"], 1)
        self.assertEqual(selection["survivor_fraction"]["median"], 0.5)

    def test_rejection_to_clear_separates_full_root_partial_branch_and_missing_clear(self):
        root = {"session": 42, "file": 0, "row_start": 0, "row_end": 100}
        run = {
            "roots": {0: root, 1: {**root, "session": 43}},
            "owners": {0: root, 1: {**root, "row_end": 50}, 2: root},
            "selections": [
                event("selection", 10, root=0, owner=0, before_rows=100, selected_rows=0, stage="pruning"),
                event("selection", 20, root=0, owner=1, before_rows=50, selected_rows=0, stage="filter"),
                event("selection", 50, root=1, owner=2, before_rows=100, selected_rows=0, stage="pruning"),
                event("selection", 60, root=1, owner=2, before_rows=0, selected_rows=0, stage="pruning"),
            ],
        }
        profile = SCAN_IO.selection_release_profile([run], {42: [{"ts_ns": 40}]}, 100)
        self.assertEqual(profile["rejected_decisions_by_stage"], {"pruning": 2, "filter": 1})
        self.assertEqual(profile["rejection_to_root_clear_ns"]["whole_root"]["max"], 30)
        self.assertEqual(profile["rejection_to_root_clear_ns"]["partial_branch"]["max"], 20)
        self.assertEqual(profile["without_observed_clear_age_ns"]["whole_root"]["max"], 50)

    def test_query_clock_excludes_reads_finishing_during_metric_printing(self):
        events = SCAN_IO.parse_events(
            "\n".join(
                [
                    "IO_ITERATION_BEGIN query=19 ts_ns=0",
                    'DEBUG scan driver run=1 ts_ns=2 event="start"',
                    'DEBUG scan IO ts_ns=12 event="read_end" source=1 physical=2 success=true',
                    "IO_ITERATION_END query=19 ts_ns=10 query_ns=10 rows=1",
                ]
            )
        )
        self.assertEqual([e["event"] for e in events], ["query_begin", "start", "query_end"])

    def test_query_window_stage_wait_and_concurrency(self):
        summary, trace = SCAN_IO.analyze(
            [
                {"event": "query_begin", "ts_ns": 0},
                admit(),
                event("spawn", 10, owner=0, stage="filter", state="NeedsCompute"),
                event("compute", 30, start_ns=20, owner=0, outcome="io", state="Waiting"),
                event("park", 31, owner=0),
                {"event": "read_start", "ts_ns": 25, "source": 1, "physical": 1, "offset": 0, "length": 4},
                {"event": "read_start", "ts_ns": 40, "source": 1, "physical": 2, "offset": 4, "length": 4},
                {"event": "read_end", "ts_ns": 60, "source": 1, "physical": 1, "success": "true"},
                event("unpark", 65, owner=0),
                event("compute", 85, start_ns=75, owner=0, outcome="batch", rows=2, state="Done"),
                event("end", 90),
                {"event": "query_end", "ts_ns": 100, "rows": 2},
            ]
        )
        self.assertEqual(summary["query"]["wall_ns"], 100)
        self.assertEqual(summary["query"]["reads"]["peak"], 2)
        self.assertEqual(summary["query"]["reads"]["busy_ns"], 75)
        self.assertEqual(summary["query"]["source_io"][1]["in_flight"]["peak"], 2)
        self.assertEqual(summary["query"]["source_io"][1]["in_flight_request_bytes"]["peak"], 8)
        self.assertEqual(summary["query"]["io_compute_overlap_ns"], 15)
        self.assertEqual(len({e["tid"] for e in trace["traceEvents"] if e["name"].startswith("physical read")}), 2)
        stage = summary["stages"]["filter"]
        self.assertEqual(stage["schedule_total_ns"], 20)
        self.assertEqual(stage["park_total_ns"], 34)
        self.assertEqual(stage["rows"], 2)
        self.assertEqual(stage["outputs"], {"io": 1, "batch": 1})

    def test_local_admission_and_pread_phases_join_physical_read(self):
        summary, trace = SCAN_IO.analyze(
            [
                {
                    "event": "read_start",
                    "ts_ns": 0,
                    "source": 1,
                    "physical": 2,
                    "offset": 0,
                    "length": 4,
                    "uri": "/file",
                },
                {
                    "event": "local_read",
                    "ts_ns": 90,
                    "start_ns": 10,
                    "path": "/file",
                    "offset": 0,
                    "length": 4,
                    "get_ns": 10,
                    "admission_ns": 20,
                    "allocation_ns": 0,
                    "prepare_ns": 30,
                    "queue_ns": 10,
                    "read_ns": 40,
                    "resume_ns": 0,
                },
                {"event": "read_end", "ts_ns": 100, "source": 1, "physical": 2, "success": "true"},
            ]
        )
        self.assertEqual(summary["totals"]["unbound_local_reads"], 0)
        self.assertEqual(summary["physical_reads"][0]["local_phases_ns"]["admission_ns"], 20)
        admission = next(e for e in trace["traceEvents"] if e["name"] == "local admission")
        self.assertAlmostEqual(admission["dur"], 0.02)
        self.assertTrue(all(isinstance(e["pid"], int) for e in trace["traceEvents"]))
        self.assertIsNone(summary["totals"]["physical_queue_ns"])

    def test_legacy_read_phases_do_not_invent_get_admission_or_allocation(self):
        summary, trace = SCAN_IO.analyze(
            [
                {
                    "event": "read_start",
                    "ts_ns": 0,
                    "source": 1,
                    "physical": 2,
                    "offset": 0,
                    "length": 4,
                    "uri": "/file",
                },
                {
                    "event": "local_read",
                    "ts_ns": 90,
                    "start_ns": 10,
                    "path": "/file",
                    "offset": 0,
                    "length": 4,
                    "prepare_ns": 30,
                    "queue_ns": 0,
                    "read_ns": 40,
                    "resume_ns": 0,
                },
                {"event": "read_end", "ts_ns": 100, "source": 1, "physical": 2, "success": "true"},
            ]
        )
        phases = summary["totals"]["local_phases_ns"]
        for phase in ("get_ns", "admission_ns", "allocation_ns"):
            self.assertEqual(phases[phase]["count"], 0)
            self.assertNotIn(phase, summary["physical_reads"][0]["local_phases_ns"])
        self.assertEqual(phases["queue_ns"]["count"], 1)
        self.assertEqual(phases["queue_ns"]["median"], 0)
        self.assertEqual(summary["physical_reads"][0]["local_timestamps_ns"]["pread_started"], 40)
        self.assertFalse(any(e["name"] == "local admission" for e in trace["traceEvents"]))


if __name__ == "__main__":
    unittest.main()
