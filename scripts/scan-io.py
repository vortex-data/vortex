#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors
"""Explain V2 scan-driver work, fetches, physical reads, and compute waits.

Capture one diagnostic iteration separately from untraced benchmark timings:
  VORTEX_SCAN_V2=1 RUST_LOG=warn,vortex_scan::driver=debug \
    target/release_debug/datafusion-bench tpch --formats vortex --queries 19 \
    --iterations 1 --io-diagnostics --hide-progress-bar > /tmp/scan.log 2>&1
  python3 scripts/scan-io.py /tmp/scan.log --output /tmp/scan.json --trace /tmp/scan.trace.json

Open the trace in https://ui.perfetto.dev. Every run has a driver lane, owner wait
lanes, and fetch lanes; physical reads have separate lanes and flow links to fetches.
Use --iteration N (zero-based) when capturing multiple diagnostic iterations.
Use --queue-trace instead of --trace for a smaller timeline of queue counters and
stage startup. Both forms consume the same diagnostic log and incur the same
capture overhead. JSON-only analysis skips timeline construction.

Compute is wall time inside compute(), not CPU time. Driver-active time includes
compute and scheduling. Blocked time ends when an owner becomes runnable; the
remaining time until advance() resumes is wake/scheduler delay. Per-run sums can
overlap across workers and must not be interpreted as query wall time. Background
reads and partial/cancelled runs are retained and reported, never silently closed.
"""

from __future__ import annotations

import argparse
import bisect
import collections
import io
import json
import math
import re
import statistics
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

ANSI = re.compile(r"\x1b\[[0-9;]*m")
FIELD = re.compile(r'(\w+)=("(?:\\.|[^"\\])*"|[^\s]+)')
TERMINAL = {"batch", "root_done", "waiting", "idle", "error"}


def parse_events(log: str | Iterable[str], iteration: int = -1) -> list[dict[str, Any]]:
    """Parse structured fields; sort by the shared clock instead of log write order."""
    groups: list[list[dict[str, Any]]] = []
    events: list[dict[str, Any]] = []
    inside = False
    marked = False
    for raw_line in io.StringIO(log) if isinstance(log, str) else log:
        line = ANSI.sub("", raw_line).rstrip("\n")
        if line.startswith("IO_ITERATION_BEGIN"):
            marked = inside = True
            events = []
            fields = dict(FIELD.findall(line))
            if "ts_ns" in fields:
                events.append({"event": "query_begin", "ts_ns": int(fields["ts_ns"]), "query": int(fields["query"])})
        elif line.startswith("IO_ITERATION_END"):
            fields = dict(FIELD.findall(line))
            if "ts_ns" in fields:
                events = [e for e in events if e["ts_ns"] <= int(fields["ts_ns"])]
                events.append({"event": "query_end", **{key: int(value) for key, value in fields.items()}})
            groups.append(events)
            inside = False
        elif "scan driver" in line or "scan IO" in line or "local object-store read" in line:
            if marked and not inside:
                continue
            fields = {}
            for key, value in FIELD.findall(line):
                if value.startswith('"'):
                    fields[key] = json.loads(value)
                elif value.isdecimal():
                    fields[key] = int(value)
                else:
                    fields[key] = value
            if "local object-store read" in line and "start_ns" in fields:
                fields["event"] = "local_read"
            if "event" in fields and "ts_ns" in fields:
                events.append(fields)
    if marked:
        if not groups:
            raise ValueError("No complete IO_ITERATION; capture a finished diagnostic run")
        events = groups[iteration]
    if not events:
        raise ValueError("No driver events; rebuild and enable RUST_LOG=vortex_scan::driver=debug")
    return sorted(events, key=lambda event: event["ts_ns"])


def union_ns(intervals: list[tuple[int, int]]) -> int:
    total = 0
    stop = 0
    for start, end in sorted(intervals):
        total += max(0, end - max(start, stop))
        stop = max(stop, end)
    return total


def distribution(values: Sequence[int | float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "count": len(values),
        "min": min(values, default=0),
        "median": statistics.median(values or [0]),
        "p95": ordered[math.ceil(len(ordered) * 0.95) - 1] if ordered else 0,
        "p99": ordered[math.ceil(len(ordered) * 0.99) - 1] if ordered else 0,
        "max": max(values, default=0),
    }


def occupancy(intervals: list[tuple[int, int]], start: int, end: int) -> dict[str, int | float]:
    """Measure overlap within a window, including unfinished intervals clipped at its end."""
    changes = collections.Counter()
    for begin, stop in intervals:
        begin, stop = max(start, begin), min(end, stop)
        if stop > begin:
            changes[begin] += 1
            changes[stop] -= 1
    count = peak = busy = area = 0
    previous = start
    for ts, delta in sorted(changes.items()):
        area += (ts - previous) * count
        busy += (ts - previous) if count else 0
        count += delta
        peak = max(peak, count)
        previous = ts
    return {"peak": peak, "busy_ns": busy, "mean": area / (end - start) if end > start else 0}


def depth_profile(
    intervals: list[tuple[int, int]], start: int, end: int, *, weights: list[int] | None = None
) -> tuple[dict[str, Any], list[tuple[int, int]]]:
    """Time-weighted queue depth, including time at zero and clipping censored work."""
    changes: collections.Counter[int] = collections.Counter()
    for index, (begin, stop) in enumerate(intervals):
        begin, stop = max(start, begin), min(end, stop)
        if stop > begin:
            weight = weights[index] if weights is not None else 1
            changes[begin] += weight
            changes[stop] -= weight
    histogram: collections.Counter[int] = collections.Counter()
    previous, count, peak = start, 0, 0
    series = [(start, 0)]
    for ts, delta in sorted(changes.items()):
        if ts > previous:
            histogram[count] += ts - previous
        previous = ts
        if delta:
            count += delta
            peak = max(peak, count)
            series.append((ts, count))
    if end > previous:
        histogram[count] += end - previous
    duration = max(0, end - start)

    def quantile(fraction: float) -> int:
        cumulative = 0
        for depth, elapsed in sorted(histogram.items()):
            cumulative += elapsed
            if cumulative >= duration * fraction:
                return depth
        return 0

    return {
        "peak": peak,
        "mean": sum(depth * elapsed for depth, elapsed in histogram.items()) / duration if duration else 0,
        "busy_ns": duration - histogram[0],
        "p50": quantile(0.5),
        "p95": quantile(0.95),
        "p99": quantile(0.99),
        "histogram_ns": dict(sorted(histogram.items())),
    }, series


def compact_counter_series(series: list[tuple[int, int]], resolution_ns: int) -> list[tuple[int, int]]:
    """Retain each time bucket's extrema and endpoints without shifting their timestamps."""
    if resolution_ns <= 0 or not series:
        return series
    selected = []
    bucket = []
    previous = None

    def flush():
        positions = {
            0,
            len(bucket) - 1,
            min(range(len(bucket)), key=lambda i: bucket[i][1]),
            max(range(len(bucket)), key=lambda i: bucket[i][1]),
        }
        selected.extend(bucket[i] for i in sorted(positions))

    for point in series:
        index = (point[0] - series[0][0]) // resolution_ns
        if previous is not None and index != previous:
            flush()
            bucket = []
        bucket.append(point)
        previous = index
    flush()
    return selected


def joint_state_time(intervals: dict[str, list[tuple[int, int]]], start: int, end: int) -> dict[str, Any]:
    """Partition elapsed time by simultaneous observed states; overlapping waits are not added."""
    changes: dict[int, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    for name, spans in intervals.items():
        for begin, stop in spans:
            begin, stop = max(start, begin), min(end, stop)
            if stop > begin:
                changes[begin][name] += 1
                changes[stop][name] -= 1
    active: collections.Counter[str] = collections.Counter()
    elapsed: collections.Counter[tuple[str, ...]] = collections.Counter()
    previous = start
    for ts, deltas in sorted(changes.items()):
        if ts > previous:
            elapsed[tuple(sorted(name for name, count in active.items() if count > 0))] += ts - previous
        active.update(deltas)
        previous = ts
    if end > previous:
        elapsed[tuple(sorted(name for name, count in active.items() if count > 0))] += end - previous
    categories: collections.Counter[str] = collections.Counter()
    order = [
        ("compute", "compute_running"),
        ("runnable", "runnable_without_compute"),
        ("delivery_ready", "delivery_ready_without_runnable_or_compute"),
        ("physical_IO", "IO_without_ready_delivery_or_compute"),
        ("pending_fetch", "fetch_pending_without_IO_or_ready_work"),
        ("driver_advancing", "driver_advancing_without_observed_compute_or_fetch"),
    ]
    for states, duration in elapsed.items():
        category = next((category for state, category in order if state in states), "no_observed_scan_demand")
        categories[category] += duration
    return {
        "accounted_ns": sum(elapsed.values()),
        "categories_ns": dict(categories),
        "states": [{"present": list(states), "elapsed_ns": duration} for states, duration in sorted(elapsed.items())],
        "interpretation": "Categories use presence priority and are mutually exclusive. "
        "They are not causal bottleneck labels or CPU-idle time.",
    }


def progress_profile(points: list[tuple[int, int]], start: int) -> dict[str, Any]:
    """Milestones for observed completed work, without assuming the total future scan workload."""
    points = sorted((ts, weight) for ts, weight in points if weight > 0)
    total = sum(weight for _, weight in points)
    cumulative = 0
    milestones = {}
    for ts, weight in points:
        cumulative += weight
        for fraction in (10, 50, 90, 99, 100):
            if fraction not in milestones and cumulative * 100 >= total * fraction:
                milestones[fraction] = ts - start
    return {"events": len(points), "observed_total": total, "time_to_fraction_ns": milestones}


def registration_profile(
    bindings: list[dict[str, Any]],
    scopes: dict[int, dict[str, Any]],
    start: int,
    end: int,
    physical_reads: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Reconstruct split-held registrations until exact withdrawal, clear, or scope retirement."""
    interests = {}
    forgets = collections.defaultdict(list)
    for session, scope in scopes.items():
        for forgotten in scope.get("forget_events", []):
            forgets[(forgotten["source"], forgotten["read"], session)].append(forgotten)
    forget_times = {key: [e["ts_ns"] for e in values] for key, values in forgets.items()}
    for binding in bindings:
        scope = scopes.get(binding["session"], {"end_ns": end, "status": "unknown"})
        clears = scope.get("clear_events", [])
        episode = bisect.bisect_right([c["ts_ns"] for c in clears], binding["ts_ns"])
        exact_end = clears[episode] if episode < len(clears) else None
        end_source = "scope_clear" if exact_end else None
        registration = (binding["source"], binding["read"], binding["session"])
        forgotten_index = bisect.bisect_right(forget_times.get(registration, []), binding["ts_ns"])
        withdrawals = forgets.get(registration, [])
        if forgotten_index < len(withdrawals):
            forgotten = withdrawals[forgotten_index]
            if exact_end is None or forgotten["ts_ns"] <= exact_end["ts_ns"]:
                exact_end = forgotten
                end_source = "scope_forget"
        status = scope["status"]
        if end_source == "scope_forget":
            status = "forgotten"
        elif exact_end and status == "unknown" and exact_end.get("reason") == "drop":
            status = "dropped"
        key = (*registration, episode, exact_end["ts_ns"] if exact_end else None)
        interest = interests.setdefault(
            key,
            {
                "start_ns": binding["ts_ns"],
                "end_ns": exact_end["ts_ns"] if exact_end else scope["end_ns"],
                "length": binding.get("length"),
                "intents": set(),
                "status": status,
                "end_source": end_source or ("query_boundary" if status == "unknown" else "root_lifecycle"),
            },
        )
        interest["start_ns"] = min(interest["start_ns"], binding["ts_ns"])
        interest["intents"].add(binding["intent"])
    by_registration = collections.defaultdict(list)
    for key, interest in interests.items():
        by_registration[key[:2]].append(interest)
    profiles = [
        depth_profile([(i["start_ns"], i["end_ns"]) for i in refs], start, end)[0] for refs in by_registration.values()
    ]
    unused = [i for i in interests.values() if i["intents"] & {"Announce", "Prefetch"} and "Fetch" not in i["intents"]]
    known_lengths = [i for i in interests.values() if i["length"] is not None]
    unused_known = [i for i in unused if i["length"] is not None]
    unused_completed = {
        key
        for key, refs in by_registration.items()
        if all(i["status"] == "completed" for i in refs) and not any("Fetch" in i["intents"] for i in refs)
    }
    selected_unused = set()
    selected_phases = collections.Counter()
    unused_member_bytes = 0
    exclusive_bytes = 0
    completed_exclusive_bytes = 0
    missing_ranges = 0
    for read in physical_reads or []:
        candidate_ranges = []
        other_ranges = []
        for member in read.get("members", []):
            key = (read["source"], member["read"])
            candidate = key in unused_completed
            if candidate:
                selected_unused.add(key)
                scope_end = max(i["end_ns"] for i in by_registration[key])
                selected_phases[
                    "started_before_last_scope_end" if read["ts_ns"] < scope_end else "started_after_last_scope_end"
                ] += 1
                if "end_ns" in read:
                    selected_phases[
                        "completed_before_last_scope_end"
                        if read["end_ns"] < scope_end
                        else "completed_after_last_scope_end"
                    ] += 1
                else:
                    selected_phases["unfinished_at_query_boundary"] += 1
            if "offset" not in member or "length" not in member:
                missing_ranges += 1
                continue
            span = (
                max(read["offset"], member["offset"]),
                min(read["offset"] + read["length"], member["offset"] + member["length"]),
            )
            (candidate_ranges if candidate else other_ranges).append(span)
        unused_member_bytes += union_ns(candidate_ranges)
        exclusive = union_ns(candidate_ranges + other_ranges) - union_ns(other_ranges)
        exclusive_bytes += exclusive
        if "end_ns" in read:
            completed_exclusive_bytes += exclusive
    return {
        "registrations": len(by_registration),
        "scope_interests": len(interests),
        "with_unknown_scope_end": sum(i["end_source"] == "query_boundary" for i in interests.values()),
        "scope_end_sources": dict(collections.Counter(i["end_source"] for i in interests.values())),
        "optional_interest_to_withdrawal_ns": distribution(
            [i["end_ns"] - i["start_ns"] for i in interests.values() if i["end_source"] == "scope_forget"]
        ),
        "without_length": len(interests) - len(known_lengths),
        "registrations_shared_by_overlapping_scopes": sum(p["peak"] > 1 for p in profiles),
        "scope_references_per_registration": distribution([p["peak"] for p in profiles]),
        "live_scope_interests": depth_profile([(i["start_ns"], i["end_ns"]) for i in interests.values()], start, end)[
            0
        ],
        "scope_interest_bytes": depth_profile(
            [(i["start_ns"], i["end_ns"]) for i in known_lengths],
            start,
            end,
            weights=[i["length"] for i in known_lengths],
        )[0],
        "unused_optional_interests": len(unused),
        "unused_optional_interest_bytes": depth_profile(
            [(i["start_ns"], i["end_ns"]) for i in unused_known],
            start,
            end,
            weights=[i["length"] for i in unused_known],
        )[0],
        "unused_optional_by_scope_status": {
            status: {
                "interests": sum(i["status"] == status for i in unused),
                "interest_bytes": sum(i["length"] or 0 for i in unused if i["status"] == status),
            }
            for status in ("completed", "forgotten", "cancelled", "boundary", "dropped", "unknown")
        },
        "unused_registrations_with_only_completed_scopes": len(unused_completed),
        "unused_completed_scope_physical": {
            "binding_inventory_available": physical_reads is not None,
            "registrations_in_started_reads": len(selected_unused),
            "registrations_absent_from_started_reads": len(unused_completed - selected_unused),
            "selected_member_union_bytes": unused_member_bytes,
            "exclusive_of_other_member_ranges_bytes": exclusive_bytes,
            "completed_exclusive_bytes": completed_exclusive_bytes,
            "unfinished_exclusive_bytes": exclusive_bytes - completed_exclusive_bytes,
            "member_ranges_missing": missing_ranges,
            "member_lifecycle_counts": dict(selected_phases),
            "interpretation": "No Fetch in any observed scope, all scopes completed. Bytes are unions within "
            "started physical reads, excluding overlap with other members in the exclusive count. "
            "Scope retirement approximates clear, not the earlier pruning decision. These bytes are "
            "not proven avoidable: a physical range may start before pruning, or still cover the bytes "
            "as a coalescing gap after hint withdrawal. Registrations with unknown/cancelled scopes "
            "remain excluded.",
        },
        "interpretation": "Scope interests are inferred from unique (source, registration, session, episode) bindings "
        "until exact per-registration withdrawal or scope clear, or root retirement/cancellation/boundary "
        "when neither is available. Re-registration after withdrawal starts another episode. "
        "They are not Arc strong_count, resident memory, "
        "or proof of when partial pruning made a segment unnecessary. Unknown admission stays unknown.",
    }


def lifetime_profile(
    events: list[dict[str, Any]],
    bindings: list[dict[str, Any]],
    reads: dict[tuple[int, int], dict[str, Any]],
    members: dict[tuple[int, int], tuple[int, int]],
    end: int,
) -> dict[str, Any]:
    """Measure exact scope clearing, last Read release, and forwarded-wake-to-poll timing."""
    scopes = [e for e in events if e["event"] == "scope_clear"]
    forgets = [e for e in events if e["event"] == "scope_forget"]
    drops = [e for e in events if e["event"] == "registration_drop"]
    lengths = {(b["source"], b["read"]): b.get("length", 0) for b in bindings}
    drop_times = {(e["source"], e["read"]): e["ts_ns"] for e in drops}
    scope_totals = {
        field: sum(e.get(field, 0) for e in scopes)
        for field in (
            "interests",
            "unfetched",
            "unfetched_bytes",
            "announcements",
            "prefetches",
            "fetches",
            "inline_fetches",
            "ready_fetches",
            "future_queues",
            "new_registrations",
            "reused_registrations",
            "polls",
            "completion_polls",
            "completions",
            "forgets",
            "forgotten",
            "forgotten_bytes",
            "forget_fetch_pins",
        )
    }
    phases = collections.Counter()
    retained = []
    cancellation_tail = []
    for e in drops:
        key = (e["source"], e["read"])
        read = reads.get(members.get(key))
        if read is None:
            phases["no_started_read_in_window"] += 1
        elif e["ts_ns"] < read["ts_ns"]:
            phases["before_physical_start"] += 1
        elif "end_ns" not in read:
            phases["inflight_unfinished_at_boundary"] += 1
        elif e["ts_ns"] < read["end_ns"]:
            phases["inflight_before_physical_completion"] += 1
            cancellation_tail.append(read["end_ns"] - e["ts_ns"])
        else:
            phases["after_physical_completion"] += 1
            retained.append(e["ts_ns"] - read["end_ns"])
    residence_byte_ns = 0
    censored_retention = []
    for key, physical in members.items():
        read = reads.get(physical)
        if read is None or "end_ns" not in read:
            continue
        held_until = drop_times.get(key, end)
        residence_byte_ns += lengths.get(key, 0) * max(0, held_until - read["end_ns"])
        if key not in drop_times:
            censored_retention.append(max(0, end - read["end_ns"]))
    pending_wakes = collections.defaultdict(list)
    poll_begin = {}
    wake_delays = []
    wake_burst_delays = []
    poll_durations = []
    methods = collections.Counter()
    outcomes = collections.Counter()
    wakes_during_poll = 0
    polls_without_begin = 0
    for e in events:
        session = e.get("session")
        if e["event"] == "session_wake":
            pending_wakes[session].append(e["ts_ns"])
            wakes_during_poll += session in poll_begin
        elif e["event"] == "session_poll_begin":
            pending = pending_wakes.pop(session, [])
            wake_delays.extend(e["ts_ns"] - ts for ts in pending)
            if pending:
                wake_burst_delays.append(e["ts_ns"] - pending[0])
            poll_begin[session] = e["ts_ns"]
            methods[e.get("method", "unknown")] += 1
        elif e["event"] == "session_poll":
            begin = poll_begin.pop(session, None)
            if begin is None:
                polls_without_begin += 1
            else:
                poll_durations.append(e["ts_ns"] - begin)
            outcomes[e.get("outcome", "unknown")] += 1
    censored_wakes = [end - ts for times in pending_wakes.values() for ts in times]
    forget_phases = collections.Counter()
    forget_phase_bytes = collections.Counter()
    forget_phase_references = collections.Counter()
    forget_last_reference_bytes = collections.Counter()
    for forgotten in forgets:
        key = (forgotten["source"], forgotten["read"])
        read = reads.get(members.get(key))
        if read is None:
            phase = "no_started_read_in_window"
        elif forgotten["ts_ns"] < read["ts_ns"]:
            phase = "before_physical_start"
        elif "end_ns" not in read:
            phase = "inflight_unfinished_at_boundary"
        elif forgotten["ts_ns"] < read["end_ns"]:
            phase = "inflight_before_physical_completion"
        else:
            phase = "after_physical_completion"
        forget_phases[phase] += 1
        forget_phase_bytes[phase] += forgotten.get("length", 0)
        references = forgotten.get("references")
        reference_state = "unknown" if references is None else "last" if references == 1 else "shared"
        forget_phase_references[(phase, reference_state)] += 1
        if reference_state == "last":
            forget_last_reference_bytes[phase] += forgotten.get("length", 0)
    return {
        "available": bool(scopes or drops or methods or forgets),
        "cleared_scopes": len(scopes),
        "scope_totals": scope_totals,
        "scope_counters_available": sorted(field for field in scope_totals if any(field in e for e in scopes)),
        "optional_withdrawals": {
            "available": bool(forgets or any("forgets" in e for e in scopes)),
            "interests": len(forgets),
            "logical_bytes": sum(e.get("length", 0) for e in forgets),
            "without_length": sum("length" not in e for e in forgets),
            "without_reference_snapshot": sum("references" not in e for e in forgets),
            "references_at_withdrawal": distribution([e["references"] for e in forgets if "references" in e]),
            "last_reference_observed": sum(e.get("references") == 1 for e in forgets),
            "other_references_observed": sum(e.get("references", 0) > 1 for e in forgets),
            "wanted_at_withdrawal": dict(collections.Counter(str(e.get("wanted", "unknown")) for e in forgets)),
            "phase_counts": dict(forget_phases),
            "phase_logical_bytes": dict(forget_phase_bytes),
            "phase_reference_counts": {
                phase: {state: forget_phase_references[(phase, state)] for state in ("last", "shared", "unknown")}
                for phase in forget_phases
            },
            "last_reference_phase_logical_bytes": dict(forget_last_reference_bytes),
            "interpretation": "Withdrawn scope interests, not saved physical bytes. Reference counts are "
            "snapshots and other scopes may change concurrently. Unstarted withdrawn ranges can still "
            "be covered by another scope or by a coalescing gap. Clear-based totals omit unclosed scopes.",
        },
        "peak_interests_per_cleared_scope": distribution([e.get("peak_interests", 0) for e in scopes]),
        "dropped_registrations": len(drops),
        "drop_phase_counts": dict(phases),
        "completed_registration_retention_ns": distribution(retained),
        "last_reference_to_physical_completion_ns": distribution(cancellation_tail),
        "completed_member_interest_byte_ns": residence_byte_ns,
        "completed_member_retention_censored_ns": distribution(censored_retention),
        "wake_to_next_session_poll_ns": distribution(wake_delays),
        "first_wake_in_burst_to_next_poll_ns": distribution(wake_burst_delays),
        "wake_without_subsequent_poll_age_ns": distribution(censored_wakes),
        "wakes_during_session_poll": wakes_during_poll,
        "session_poll_duration_ns": distribution(poll_durations),
        "session_poll_methods": dict(methods),
        "session_poll_outcomes": dict(outcomes),
        "session_poll_ends_without_begin": polls_without_begin,
        "session_poll_begins_without_end": len(poll_begin),
        "interpretation": "Exact scope/drop events require vortex_file::scan_lifetime tracing. "
        "Scope sums include shared and overlapping ranges, not physical traffic. Wake metrics use "
        "forwarded IO wake notifications and the next session poll (sweep or completion); notifications "
        "can be redundant or occur during a poll. No subsequent poll is censored, not zero latency. "
        "Member residence counts logical byte-time, not RSS or complete buffer lifetimes.",
    }


def selection_release_profile(
    runs: list[dict[str, Any]], scope_clears: dict[int, list[dict[str, Any]]], end: int
) -> dict[str, Any]:
    """Measure rejection-to-root-clear lag without assigning shared hints to rejected row branches."""
    delays = collections.defaultdict(list)
    censored = collections.defaultdict(list)
    stages = collections.Counter()
    for run in runs:
        for decision in run["selections"]:
            if decision["selected_rows"] != 0 or decision["before_rows"] == 0:
                continue
            root = run["roots"].get(decision["root"], {})
            work = run["owners"].get(decision["owner"], {})
            stages[decision.get("stage", work.get("stage", "unknown"))] += 1
            fields = ("file", "row_start", "row_end")
            kind = (
                "unknown_scope"
                if any(field not in root or field not in work for field in fields)
                else "whole_root"
                if all(root[field] == work[field] for field in fields)
                else "partial_branch"
            )
            clears = scope_clears.get(root.get("session"), [])
            index = bisect.bisect_left([e["ts_ns"] for e in clears], decision["ts_ns"])
            if index < len(clears):
                delays[kind].append(clears[index]["ts_ns"] - decision["ts_ns"])
            else:
                censored[kind].append(end - decision["ts_ns"])
    return {
        "rejected_decisions_by_stage": dict(stages),
        "rejection_to_root_clear_ns": {kind: distribution(values) for kind, values in delays.items()},
        "without_observed_clear_age_ns": {kind: distribution(values) for kind, values in censored.items()},
        "interpretation": "Whole-root rejection covers the admitted row scope; partial branches can "
        "leave sibling work alive. This measures time to the next exact root IO clear, not per-segment "
        "pruning or avoidable IO. Missing clear events are censored at the query boundary.",
    }


def analyze(
    events: list[dict[str, Any]],
    min_compute_ns: int = 10_000,
    *,
    build_trace: bool = True,
    build_queue_trace: bool = False,
    queue_resolution_ns: int = 1_000_000,
) -> tuple[dict[str, Any], dict[str, Any]]:
    runs: dict[int, dict[str, Any]] = {}
    sessions = {}
    bindings = {}
    scope_clears = collections.defaultdict(list)
    scope_forgets = collections.defaultdict(list)
    reads: dict[tuple[int, int], dict[str, Any]] = {}
    queued = {}
    members: dict[tuple[int, int], tuple[int, int]] = {}
    timeline = []
    warnings = []
    local_reads = []
    local_intervals = collections.defaultdict(list)
    compute_intervals = []
    batch_points = []
    driver_intervals = []
    caller_intervals = []
    ready_intervals = []
    queued_by_kind = collections.defaultdict(list)
    queued_by_stage = collections.defaultdict(list)
    queued_by_depth = collections.defaultdict(list)
    parked_by_kind = collections.defaultdict(list)
    parked_by_stage = collections.defaultdict(list)
    depth_steps: collections.Counter[int] = collections.Counter()
    depth_compute: collections.Counter[int] = collections.Counter()
    unknown_depth_steps = 0
    first_steps: dict[str, dict[str, Any]] = {}
    stages: dict[str, dict[str, Any]] = collections.defaultdict(
        lambda: {
            "compute": [],
            "park": [],
            "schedule": [],
            "fetch": [],
            "batch_rows": [],
            "ready_censored": [],
            "fetch_censored": [],
            "interference": collections.Counter(),
            "outputs": collections.Counter(),
            "rows": 0,
        }
    )
    query = {e["event"]: e for e in events if e["event"] in {"query_begin", "query_end"}}
    window_end = query.get("query_end", {}).get("ts_ns", max(e["ts_ns"] for e in events))

    def stage_for(run: dict[str, Any], owner: int) -> str:
        return str(run["owners"].get(owner, {}).get("stage", "<unknown>"))

    def read_lane(read: dict[str, Any]) -> str:
        return f"source {read['source']} read {read['physical']}"

    def interval(name: str, start: int, end: int, pid: str, tid: str | int, args: dict[str, Any] | None = None):
        if end < start:
            raise ValueError(f"Negative interval for {name}: {start}..{end}")
        if not build_trace:
            return
        timeline.append(
            {
                "name": name,
                "ph": "X",
                "ts": start / 1000,
                "dur": (end - start) / 1000,
                "pid": pid,
                "tid": tid,
                "args": args or {},
            }
        )

    def record_ready(run: dict[str, Any], owner: int, stop: int, *, computed: bool = False):
        begin = run["queued_compute"].pop(owner, None)
        if begin is None:
            return
        work = run["owners"].get(owner, {})
        if not computed:
            stages[stage_for(run, owner)]["ready_censored"].append(stop - begin)
        queued_by_kind[work.get("kind", "unknown")].append((begin, stop))
        queued_by_stage[stage_for(run, owner)].append((begin, stop))
        if work.get("dfs_depth") is not None:
            queued_by_depth[work["dfs_depth"]].append((begin, stop))

    def first_step(name: str, run: dict[str, Any], event: dict[str, Any], ts: int):
        previous = run.setdefault("milestone_ts_ns", {}).get(name)
        if previous is None or ts < previous:
            run["milestone_ts_ns"][name] = ts
        if name not in first_steps or ts < first_steps[name]["ts_ns"]:
            work = run["owners"].get(event.get("owner"), {})
            first_steps[name] = {
                "ts_ns": ts,
                "run": run["run"],
                "root": work.get("root", event.get("root")),
                "owner": event.get("owner"),
                "stage": work.get("stage"),
                "dfs_depth": work.get("dfs_depth"),
            }

    # Resolve sessions and physical membership before analyzing the state machine: announcements
    # precede admission, and reads can complete before a later fetch binds to their registration.
    for e in events:
        key = (e.get("source", -1), e.get("physical", -1))
        if e["event"] == "admit":
            sessions[e["session"]] = (e["run"], e["root"])
        elif e["event"] == "binding":
            bindings[(e["session"], e["owner"], e["request"], e["intent"])] = e
        elif e["event"] == "scope_clear":
            scope_clears[e["session"]].append(e)
        elif e["event"] == "scope_forget":
            scope_forgets[e["session"]].append(e)
        elif e["event"] == "read_queued":
            queued[key] = e
        elif e["event"] == "read_start":
            reads[key] = e.copy()
            if key in queued:
                reads[key]["queued_ns"] = queued[key]["ts_ns"]
        elif e["event"] == "read_member":
            members[(e["source"], e["read"])] = key
            if key in reads:
                reads[key].setdefault("members", []).append({k: e[k] for k in ("read", "offset", "length") if k in e})
        elif e["event"] == "read_end" and key in reads:
            reads[key]["end_ns"] = e["ts_ns"]
            reads[key]["success"] = e["success"]
        elif e["event"] == "local_read":
            local_reads.append(e)
    by_range = collections.defaultdict(list)
    for read in reads.values():
        by_range[(read.get("uri", ""), read["offset"], read["length"])].append(read)
    unbound_local_reads = 0
    for local in local_reads:
        candidates = by_range[(local["path"], local["offset"], local["length"])]
        physical = next(
            (r for r in candidates if r["ts_ns"] <= local["start_ns"] and local["ts_ns"] <= r.get("end_ns", 0)), None
        )
        if physical is None:
            unbound_local_reads += 1
            continue
        physical["local_phases_ns"] = {
            phase: local[phase]
            for phase in (
                "get_ns",
                "admission_ns",
                "allocation_ns",
                "queue_ns",
                "read_ns",
                "resume_ns",
            )
            if phase in local
        }
        submitted = local["start_ns"] + local["prepare_ns"]
        read_start = submitted + local["queue_ns"]
        read_end = read_start + local["read_ns"]
        physical["local_timestamps_ns"] = {
            "pread_started": read_start,
            "pread_finished": read_end,
            "async_resumed": read_end + local["resume_ns"],
            "phase_logged": local["ts_ns"],
        }
        if (get_ns := local.get("get_ns")) is not None:
            get_end = local["start_ns"] + get_ns
            local_intervals["get"].append((local["start_ns"], get_end))
            if (admission_ns := local.get("admission_ns")) is not None:
                admission_end = get_end + admission_ns
                local_intervals["admission"].append((get_end, admission_end))
                interval("local admission", get_end, admission_end, "IO", read_lane(physical))
                if (allocation_ns := local.get("allocation_ns")) is not None:
                    local_intervals["allocation"].append((admission_end, admission_end + allocation_ns))
        local_intervals["blocking_queue"].append((submitted, read_start))
        local_intervals["pread"].append((read_start, read_end))
        local_intervals["resume"].append((read_end, read_end + local["resume_ns"]))
        interval("blocking pool queue", submitted, read_start, "IO", read_lane(physical))
        interval("pread", read_start, read_end, "IO", read_lane(physical))
    for read in reads.values():
        if "end_ns" in read:
            interval("physical read", read["ts_ns"], read["end_ns"], "IO", read_lane(read), read)
        else:
            interval(
                "physical read (unfinished)",
                read["ts_ns"],
                window_end,
                "IO",
                read_lane(read),
                read | {"censored": True},
            )
    session_uris = {session: binding["uri"] for (session, _, _, _), binding in bindings.items() if binding.get("uri")}
    for e in events:
        if "run" not in e:
            continue
        rid = e["run"]
        run = runs.setdefault(
            rid,
            {
                "run": rid,
                "roots": {},
                "requests": [],
                "compute_ns": 0,
                "active_ns": 0,
                "blocked_ns": 0,
                "wake_delay_ns": 0,
                "consumer_gap_ns": 0,
                "owner_wait_ns": 0,
                "max_ready": 0,
                "max_parked": 0,
                "steps": 0,
                "start_ns": e["ts_ns"],
                "end_ns": None,
                "cancelled": False,
                "cancelled_roots": {},
                "owners": {},
                "pending": {},
                "parks": {},
                "advance": None,
                "gap": None,
                "runnable": None,
                "ready_owners": {},
                "queued_compute": {},
                "_admitted_pending": set(),
                "max_consecutive_owner_visits": 0,
                "max_consecutive_visits_with_other_work_ready": 0,
                "max_consecutive_span_with_other_work_ready_ns": 0,
                "_consecutive_owner_visits": 0,
                "_consecutive_bypasses": 0,
                "_last_compute_owner": None,
                "batch_emission_ns": [],
                "releases": [],
                "selections": [],
                "_released": {},
                "dfs_choices": collections.Counter(),
                "driver_intervals": [],
                "compute_intervals": [],
            },
        )
        ts = e["ts_ns"]
        event = e["event"]
        run["max_ready"] = max(run["max_ready"], e.get("ready", 0))
        run["max_parked"] = max(run["max_parked"], e.get("parked", 0))
        pid = f"run {rid}"
        if event == "admit":
            run["roots"][e["root"]] = e | {"uri": session_uris.get(e["session"], "")}
            run["_admitted_pending"].add(e["root"])
        elif event == "spawn":
            work = e.copy()
            parent_id = run.pop("_emitting_owner", None)
            parent = run["owners"].get(parent_id)
            root_id = e.get("root")
            if root_id is None and parent is not None:
                root_id = parent.get("root")
                work["root"] = root_id
            elif root_id is None and len(run["_admitted_pending"]) == 1:
                root_id = next(iter(run["_admitted_pending"]))
                work["root"] = root_id
            path = None
            if parent is not None and parent.get("root") == root_id:
                parent_path = parent.get("dfs_path")
                if parent_path is not None:
                    path = parent_path + [parent["next_child"]]
                parent["next_child"] += 1
            elif root_id in run["_admitted_pending"]:
                path = [root_id]
                parent_id = None
                run["_admitted_pending"].remove(root_id)
            else:
                parent_id = None
            work.update(
                parent_owner=parent_id,
                dfs_path=path,
                dfs_depth=len(path) - 1 if path is not None else None,
                dfs_depth_source="reconstructed" if path is not None else "unavailable",
                next_child=0,
            )
            run["owners"][e["owner"]] = work
            if e.get("state") == "NeedsCompute":
                run["ready_owners"][e["owner"]] = ts
                run["queued_compute"][e["owner"]] = ts
        elif event == "advance_begin":
            if run["gap"] is not None:
                begin, outcome = run["gap"]
                if outcome == "waiting":
                    ready = run["runnable"]
                    if ready is None:
                        warnings.append(f"run {rid}: resumed without a recorded unpark")
                        ready = ts
                    ready = max(begin, min(ready, ts))
                    run["blocked_ns"] += ready - begin
                    run["wake_delay_ns"] += ts - ready
                    interval("IO blocked", begin, ready, pid, "driver")
                    interval("wake delay", ready, ts, pid, "driver")
                else:
                    run["consumer_gap_ns"] += ts - begin
                    caller_intervals.append((begin, ts))
                    interval("caller gap", begin, ts, pid, "driver")
            run["advance"] = ts
            run["gap"] = run["runnable"] = None
        elif event in TERMINAL:
            if event == "batch":
                first_step("batch_returned", run, e, ts)
            if run["advance"] is not None:
                run["active_ns"] += ts - run["advance"]
                driver_intervals.append((run["advance"], ts))
                run["driver_intervals"].append((run["advance"], ts))
                interval("driver active", run["advance"], ts, pid, "driver", {"outcome": event})
            run["advance"] = None
            run["gap"] = (ts, event)
        elif event == "compute":
            begin = e["start_ns"]
            elapsed = ts - begin
            work = run["owners"].get(e["owner"], {})
            release = run["_released"].pop(e["owner"], None)
            if release is not None:
                release.update(
                    first_compute_ns=begin,
                    unpark_to_compute_ns=begin - release["unpark_ns"],
                    next_compute_outcome=e.get("outcome", "unknown"),
                )
            work.setdefault("first_compute_ns", begin)
            first_step("compute", run, e, begin)
            kind = work.get("kind")
            if kind in {"planner", "morsel"}:
                first_step(f"{kind}_compute", run, e, begin)
            if e.get("outcome") == "batch":
                batch_points.append((ts, e.get("rows", 0)))
                work.setdefault("first_batch_ns", ts)
                first_step("batch_emitted", run, e, ts)
                run["batch_emission_ns"].append(ts)
            if e.get("outcome") in {"planner", "morsel"}:
                run["_emitting_owner"] = e["owner"]
            depth = work.get("dfs_depth")
            if depth is None:
                unknown_depth_steps += 1
            else:
                depth_steps[depth] += 1
                depth_compute[depth] += elapsed
            path = work.get("dfs_path")
            priority = path + [work["next_child"]] if path is not None else None
            for other_id, ready_ns in run["queued_compute"].items():
                if other_id == e["owner"]:
                    continue
                other = run["owners"].get(other_id, {})
                other_path = other.get("dfs_path")
                relationship = "unknown_ancestry"
                if path is not None and other_path is not None:
                    if path[: len(other_path)] == other_path:
                        relationship = "ancestor_continuation"
                    elif other_path[: len(path)] == path:
                        relationship = "descendant"
                    else:
                        relationship = "peer_branch"
                    other_priority = other_path + [other["next_child"]]
                    run["dfs_choices"]["comparisons"] += 1
                    if other_priority < priority:
                        run["dfs_choices"]["earlier_ready_priority_bypassed"] += 1
                else:
                    run["dfs_choices"]["unknown_comparisons"] += 1
                interference = stages[stage_for(run, other_id)]["interference"]
                interference[f"{relationship}_compute_ns"] += elapsed
                interference[f"{relationship}_visits"] += 1
                interference[f"{relationship}_max_ready_age_ns"] = max(
                    interference[f"{relationship}_max_ready_age_ns"], begin - ready_ns
                )
            same_owner = run["_last_compute_owner"] == e["owner"]
            others_ready = len(run["queued_compute"]) - (e["owner"] in run["queued_compute"])
            run["_consecutive_owner_visits"] = run["_consecutive_owner_visits"] + 1 if same_owner else 1
            run["_consecutive_bypasses"] = (
                (run["_consecutive_bypasses"] + 1 if same_owner else 1) if others_ready else 0
            )
            if run["_consecutive_bypasses"] == 1:
                run["_consecutive_bypass_start_ns"] = begin
            if run["_consecutive_bypasses"]:
                run["max_consecutive_span_with_other_work_ready_ns"] = max(
                    run["max_consecutive_span_with_other_work_ready_ns"], ts - run["_consecutive_bypass_start_ns"]
                )
            run["max_consecutive_owner_visits"] = max(
                run["max_consecutive_owner_visits"], run["_consecutive_owner_visits"]
            )
            run["max_consecutive_visits_with_other_work_ready"] = max(
                run["max_consecutive_visits_with_other_work_ready"], run["_consecutive_bypasses"]
            )
            run["_last_compute_owner"] = e["owner"]
            record_ready(run, e["owner"], begin, computed=True)
            run["compute_ns"] += elapsed
            run["steps"] += 1
            stage = stages[stage_for(run, e["owner"])]
            stage["compute"].append(elapsed)
            stage["outputs"][e.get("outcome", "unknown")] += 1
            stage["rows"] += e.get("rows", 0)
            if e.get("outcome") == "batch" and "rows" in e:
                stage["batch_rows"].append(e["rows"])
            compute_intervals.append((begin, ts))
            run["compute_intervals"].append((begin, ts))
            ready = run["ready_owners"].pop(e["owner"], None)
            if ready is not None:
                stage["schedule"].append(max(0, begin - ready))
                ready_intervals.append((ready, ts))
            if e.get("state") == "NeedsCompute":
                run["ready_owners"][e["owner"]] = ts
                run["queued_compute"][e["owner"]] = ts
            if elapsed >= min_compute_ns:
                interval(
                    stage_for(run, e["owner"]),
                    begin,
                    ts,
                    pid,
                    "driver",
                    e
                    | {
                        "scope": work,
                        "dfs_priority": work["dfs_path"] + [work["next_child"]]
                        if work.get("dfs_path") is not None
                        else None,
                    },
                )
        elif event == "selection":
            run["selections"].append(e)
        elif event == "request":
            request = e.copy()
            root = run["roots"][e["root"]]
            binding = bindings.get((root["session"], e["owner"], e["request"], e["intent"]))
            if binding:
                request["binding"] = {key: binding[key] for key in ("source", "read", "ts_ns")}
                physical = members.get((binding["source"], binding["read"]))
                if physical in reads:
                    request["physical"] = physical
            run["requests"].append(request)
            if e["intent"] == "Fetch":
                run["pending"][(e["owner"], e["request"])] = request
        elif event == "completion":
            request = run["pending"].pop((e["owner"], e["request"]), None)
            if request is None:
                warnings.append(f"run {rid}: unmatched completion {e['owner']}/{e['request']}")
            else:
                request["delivered_ns"] = ts
                parked = run["parks"].get(e["owner"])
                if parked is not None:
                    parked["deliveries"] = parked.get("deliveries", 0) + 1
                    parked["last_delivery"] = request
                stages[stage_for(run, e["owner"])]["fetch"].append(ts - request["ts_ns"])
                interval("fetch", request["ts_ns"], ts, pid, f"fetches owner {e['owner']}", request)
                physical = reads.get(request.get("physical"))
                if build_trace and physical and "end_ns" in physical:
                    flow = f"{rid}/{e['owner']}/{e['request']}"
                    timeline.extend(
                        [
                            {
                                "name": "delivery",
                                "ph": "s",
                                "id": flow,
                                "ts": physical["end_ns"] / 1000,
                                "pid": "IO",
                                "tid": read_lane(physical),
                            },
                            {
                                "name": "delivery",
                                "ph": "f",
                                "bp": "e",
                                "id": flow,
                                "ts": ts / 1000,
                                "pid": pid,
                                "tid": f"fetches owner {e['owner']}",
                            },
                        ]
                    )
        elif event == "park":
            record_ready(run, e["owner"], ts)
            run["parks"][e["owner"]] = e.copy()
            ready = run["ready_owners"].pop(e["owner"], None)
            if ready is not None:
                ready_intervals.append((ready, ts))
        elif event == "unpark":
            run["ready_owners"][e["owner"]] = ts
            if e.get("state", "NeedsCompute") == "NeedsCompute":
                run["queued_compute"][e["owner"]] = ts
            if run["gap"] is not None and run["gap"][1] == "waiting" and run["runnable"] is None:
                run["runnable"] = ts
            parked = run["parks"].pop(e["owner"], None)
            if parked:
                request = parked.get("last_delivery")
                release = {
                    "owner": e["owner"],
                    "stage": stage_for(run, e["owner"]),
                    "root": e.get("root", run["owners"].get(e["owner"], {}).get("root")),
                    "park_ns": parked["ts_ns"],
                    "unpark_ns": ts,
                    "state": e.get("state", "unknown"),
                    "deliveries_during_park": parked.get("deliveries", 0),
                    "release_request": request.get("request") if request else None,
                }
                if request is not None:
                    release["completion_to_unpark_ns"] = ts - request["delivered_ns"]
                    release["fetch_latency_ns"] = request["delivered_ns"] - request["ts_ns"]
                    physical = reads.get(request.get("physical"))
                    if physical and "end_ns" in physical:
                        release["physical"] = request["physical"]
                        release["file_completion_to_delivery_ns"] = request["delivered_ns"] - max(
                            request["ts_ns"], physical["end_ns"]
                        )
                        release["file_ready_ns"] = max(request["ts_ns"], physical["end_ns"])
                        release["delivery_ns"] = request["delivered_ns"]
                        if "local_timestamps_ns" in physical:
                            release["pread_to_delivery_ns"] = request["delivered_ns"] - max(
                                request["ts_ns"], physical["local_timestamps_ns"]["pread_finished"]
                            )
                run["releases"].append(release)
                if e.get("state", "NeedsCompute") == "NeedsCompute":
                    run["_released"][e["owner"]] = release
                run["owner_wait_ns"] += ts - parked["ts_ns"]
                stages[stage_for(run, e["owner"])]["park"].append(ts - parked["ts_ns"])
                parked_by_kind[run["owners"].get(e["owner"], {}).get("kind", "unknown")].append((parked["ts_ns"], ts))
                parked_by_stage[stage_for(run, e["owner"])].append((parked["ts_ns"], ts))
                interval("owner parked", parked["ts_ns"], ts, pid, f"owner {e['owner']}", parked)
        elif event == "cancel":
            run["cancelled"] = True
            run["cancelled_roots"][e.get("root")] = ts
            for owner, work in run["owners"].items():
                if work.get("root") == e.get("root"):
                    record_ready(run, owner, ts)
                    if "observed_end_ns" not in work:
                        work.update(observed_end_ns=ts, end_reason="cancel")
                    ready = run["ready_owners"].pop(owner, None)
                    if ready is not None:
                        ready_intervals.append((ready, ts))
        elif event == "end":
            for owner, work in run["owners"].items():
                record_ready(run, owner, ts)
                if "observed_end_ns" not in work:
                    work.update(observed_end_ns=ts, end_reason="driver_end")
            run["end_ns"] = ts
            run["cancelled"] |= e.get("ready", 0) > 0 or e.get("parked", 0) > 0
            ready_intervals.extend((begin, ts) for begin in run["ready_owners"].values())
            run["ready_owners"].clear()
        elif event == "retire":
            record_ready(run, e["owner"], ts)
            if e["owner"] in run["owners"]:
                run["owners"][e["owner"]].update(observed_end_ns=ts, end_reason="retire")
            ready = run["ready_owners"].pop(e["owner"], None)
            if ready is not None:
                ready_intervals.append((ready, ts))

    # The eager announcement happens before Run is created. Preserve it in the request inventory.
    published = {(run["run"], r["owner"], r["request"], r["intent"]) for run in runs.values() for r in run["requests"]}
    for (session, owner, request, intent), binding in bindings.items():
        if intent == "Fetch" or session not in sessions:
            continue
        rid, root = sessions[session]
        run = runs.get(rid)
        if run is not None and (rid, owner, request, intent) not in published:
            run["requests"].append(
                {
                    "run": rid,
                    "root": root,
                    "owner": owner,
                    "request": request,
                    "intent": intent,
                    "ts_ns": binding["ts_ns"],
                    "offset": binding.get("offset", 0),
                    "length": binding.get("length", 0),
                    "binding": {key: binding[key] for key in ("source", "read", "ts_ns")},
                }
            )
    for run in runs.values():
        boundary = run["end_ns"] if run["end_ns"] is not None else window_end
        if run["advance"] is not None:
            driver_intervals.append((run["advance"], boundary))
            run["driver_intervals"].append((run["advance"], boundary))
        if run["gap"] is not None and run["gap"][1] != "waiting":
            caller_intervals.append((run["gap"][0], boundary))
        for owner, work in run["owners"].items():
            stop = run["cancelled_roots"].get(work.get("root"), boundary)
            record_ready(run, owner, stop)
            if "observed_end_ns" not in work:
                work.update(observed_end_ns=stop, end_reason="observed_boundary")
        for release in run["_released"].values():
            work = run["owners"].get(release["owner"], {})
            release["compute_not_observed_until_ns"] = work.get("observed_end_ns", boundary)
        for release in run["releases"]:
            if "file_ready_ns" in release:
                ready, delivery = release["file_ready_ns"], release["delivery_ns"]
                for key, spans in (
                    ("driver_advancing", run["driver_intervals"]),
                    ("run_computing", run["compute_intervals"]),
                ):
                    release[f"file_ready_while_{key}_ns"] = union_ns(
                        [(max(a, ready), min(b, delivery)) for a, b in spans if b > ready and a < delivery]
                    )
                release["file_ready_outside_driver_advance_ns"] = (
                    delivery - ready - release["file_ready_while_driver_advancing_ns"]
                )
        run["startup_ns"] = {f"first_{key}": ts - run["start_ns"] for key, ts in run.get("milestone_ts_ns", {}).items()}
        run["batch_gap_ns"] = distribution(
            [b - a for a, b in zip(run["batch_emission_ns"], run["batch_emission_ns"][1:])]
        )
        run["unfinished_owner_wait_ns"] = 0
        for parked in run["parks"].values():
            work = run["owners"].get(parked["owner"], {})
            stop = run["cancelled_roots"].get(parked.get("root", work.get("root")), boundary)
            parked_by_kind[work.get("kind", "unknown")].append((parked["ts_ns"], stop))
            parked_by_stage[stage_for(run, parked["owner"])].append((parked["ts_ns"], stop))
            run["unfinished_owner_wait_ns"] += stop - parked["ts_ns"]
            interval(
                "owner parked (unfinished)",
                parked["ts_ns"],
                stop,
                f"run {run['run']}",
                f"owner {parked['owner']}",
                parked | {"censored": True},
            )
        run["unfinished_blocked_ns"] = 0
        if run["gap"] is not None and run["gap"][1] == "waiting":
            stop = run["runnable"] if run["runnable"] is not None else boundary
            run["unfinished_blocked_ns"] = stop - run["gap"][0]
            interval("IO blocked (unfinished)", run["gap"][0], stop, f"run {run['run']}", "driver", {"censored": True})
        for request in run["pending"].values():
            stop = run["cancelled_roots"].get(request["root"], boundary)
            request["observed_until_ns"] = stop
            stages[stage_for(run, request["owner"])]["fetch_censored"].append(stop - request["ts_ns"])
            interval(
                "fetch (unfinished)",
                request["ts_ns"],
                stop,
                f"run {run['run']}",
                f"fetches owner {request['owner']}",
                request | {"censored": True},
            )
        ready_intervals.extend((begin, window_end) for begin in run["ready_owners"].values())
        requests = run["requests"]
        fetches = [r for r in requests if r["intent"] == "Fetch"]
        latencies = [r["delivered_ns"] - r["ts_ns"] for r in fetches if "delivered_ns" in r]
        run["fetch_latency_ns"] = distribution(latencies)
        dispatched = [r for r in fetches if "physical" in r]
        delivered = [r for r in dispatched if "delivered_ns" in r and "end_ns" in reads[r["physical"]]]
        run["dispatch_wait_ns"] = distribution([max(0, reads[r["physical"]]["ts_ns"] - r["ts_ns"]) for r in dispatched])
        run["delivery_delay_ns"] = distribution(
            [r["delivered_ns"] - max(r["ts_ns"], reads[r["physical"]]["end_ns"]) for r in delivered]
        )
        run["fetches_already_read"] = sum(reads[r["physical"]]["end_ns"] <= r["ts_ns"] for r in delivered)
        run["intents"] = dict(collections.Counter(r["intent"] for r in requests))
        run["fetch_bytes"] = sum(r["length"] for r in fetches)
        run["unbound_fetches"] = sum("binding" not in r for r in fetches)
        run["undelivered_fetches"] = len(run["pending"])
        run["owners_still_parked"] = len(run["parks"])
        run["partial"] = run["end_ns"] is None or bool(run["pending"]) or bool(run["parks"])
        if run["end_ns"] is not None:
            run["wall_ns"] = run["end_ns"] - run["start_ns"]
        run["driver_overhead_ns"] = max(0, run["active_ns"] - run["compute_ns"])
        for key in (
            "pending",
            "parks",
            "advance",
            "gap",
            "runnable",
            "ready_owners",
            "queued_compute",
            "_admitted_pending",
            "_emitting_owner",
            "_consecutive_owner_visits",
            "_consecutive_bypasses",
            "_consecutive_bypass_start_ns",
            "_last_compute_owner",
            "_released",
        ):
            run.pop(key, None)

    completed_reads = [read for read in reads.values() if "end_ns" in read]
    begin = query.get("query_begin", {}).get("ts_ns", min(e["ts_ns"] for e in events))
    end = query.get("query_end", {}).get("ts_ns", max(e["ts_ns"] for e in events))
    physical_intervals = [(r["ts_ns"], r.get("end_ns", end)) for r in reads.values()]
    source_reads = collections.defaultdict(list)
    for read in reads.values():
        source_reads[read["source"]].append(read)
    source_io = {}
    for source, requests in source_reads.items():
        spans = [(r["ts_ns"], r.get("end_ns", end)) for r in requests]
        source_io[source] = {
            "uri": requests[0].get("uri", ""),
            "physical_reads": len(requests),
            "request_bytes": sum(r["length"] for r in requests),
            "unfinished_reads": sum("end_ns" not in r for r in requests),
            "in_flight": depth_profile(spans, begin, end)[0],
            "in_flight_request_bytes": depth_profile(spans, begin, end, weights=[r["length"] for r in requests])[0],
        }
    read_occupancy = occupancy(physical_intervals, begin, end)
    compute_occupancy = occupancy(compute_intervals, begin, end)
    both_union = occupancy(physical_intervals + compute_intervals, begin, end)["busy_ns"]
    all_fetches = [r for run in runs.values() for r in run["requests"] if r["intent"] == "Fetch"]
    fetched_members = {(r["binding"]["source"], r["binding"]["read"]) for r in all_fetches if "binding" in r}
    dispatched_fetches = [r for r in all_fetches if "physical" in r]
    delivered_fetches = [r for r in dispatched_fetches if "delivered_ns" in r and "end_ns" in reads[r["physical"]]]

    def profile(name: str, intervals: list[tuple[int, int]], weights: list[int] | None = None) -> dict[str, Any]:
        values, series = depth_profile(intervals, begin, end, weights=weights)
        if build_trace or build_queue_trace:
            if build_queue_trace:
                series = compact_counter_series(series, queue_resolution_ns)
            timeline.extend(
                {
                    "name": name,
                    "ph": "C",
                    "ts": ts / 1000,
                    "pid": "Queue depths",
                    "tid": name,
                    "args": {"bytes" if weights is not None else "depth": count},
                }
                for ts, count in series
            )
        return values

    owners_by_stage = collections.defaultdict(list)
    owners_by_kind = collections.defaultdict(list)
    owners_by_depth = collections.defaultdict(list)
    for run in runs.values():
        for work in run["owners"].values():
            owners_by_stage[work.get("stage", "<unknown>")].append((run, work))
            lifetime = (work["ts_ns"], work["observed_end_ns"])
            owners_by_kind[work.get("kind", "unknown")].append(lifetime)
            if work["dfs_depth"] is not None:
                owners_by_depth[work["dfs_depth"]].append(work)

    startup = {
        f"first_{name}": first_steps[name]["ts_ns"] - begin if name in first_steps else None
        for name in ("compute", "planner_compute", "morsel_compute", "batch_emitted", "batch_returned")
    }
    startup.update(
        first_physical_queued=min((r["ts_ns"] - begin for r in queued.values()), default=None),
        first_physical_started=min((r["ts_ns"] - begin for r in reads.values()), default=None),
        first_physical_completed=min((r["end_ns"] - begin for r in completed_reads), default=None),
        first_fetch_requested=min((r["ts_ns"] - begin for r in all_fetches), default=None),
        first_fetch_delivered=min(
            (r["delivered_ns"] - begin for r in all_fetches if "delivered_ns" in r), default=None
        ),
        first_local_pread_started=min((start - begin for start, _ in local_intervals["pread"]), default=None),
        first_local_pread_completed=min((stop - begin for _, stop in local_intervals["pread"]), default=None),
    )
    fetch_intervals = [(r["ts_ns"], r.get("delivered_ns", r.get("observed_until_ns", end))) for r in all_fetches]
    awaiting_delivery = [
        (max(r["ts_ns"], reads[r["physical"]]["end_ns"]), r.get("delivered_ns", r.get("observed_until_ns", end)))
        for r in dispatched_fetches
        if "end_ns" in reads[r["physical"]]
    ]
    queues = {
        "outstanding_fetches": profile("outstanding Fetch requests", fetch_intervals),
        "completed_reads_awaiting_fetch_delivery": profile(
            "physically complete Fetch requests awaiting delivery", awaiting_delivery
        ),
        "physical_queued": profile(
            "queued physical reads",
            [(r["ts_ns"], reads[key]["ts_ns"] if key in reads else end) for key, r in queued.items()],
        ),
        "physical_in_flight": profile("physical IO in flight", physical_intervals),
        "ready_planners": profile("ready planners", queued_by_kind["planner"]),
        "ready_morsels": profile("ready morsels", queued_by_kind["morsel"]),
        "ready_unknown": profile("ready work without kind", queued_by_kind["unknown"]),
        "parked_planners": profile("parked planners", parked_by_kind["planner"]),
        "parked_morsels": profile("parked morsels", parked_by_kind["morsel"]),
        "live_morsels": profile("live morsels", owners_by_kind["morsel"]),
    }
    complete_fetches = [r for r in dispatched_fetches if "end_ns" in reads[r["physical"]]]
    byte_depth = {
        "physical_in_flight": profile(
            "in-flight physical request bytes", physical_intervals, [r["length"] for r in reads.values()]
        ),
        "outstanding_fetches": profile(
            "outstanding logical Fetch bytes", fetch_intervals, [r["length"] for r in all_fetches]
        ),
        "completed_fetches_awaiting_delivery": profile(
            "complete logical Fetch bytes awaiting delivery", awaiting_delivery, [r["length"] for r in complete_fetches]
        ),
    }
    local_phase_depth = {
        phase: profile(f"bound completed local reads: {phase}", intervals)
        for phase, intervals in local_intervals.items()
    }

    def without_compute(intervals: list[tuple[int, int]]) -> int:
        return int(occupancy(intervals + compute_intervals, begin, end)["busy_ns"] - compute_occupancy["busy_ns"])

    gaps = {
        "ready_morsels_without_compute_ns": without_compute(queued_by_kind["morsel"]),
        "ready_planners_without_compute_ns": without_compute(queued_by_kind["planner"]),
        "completed_fetches_without_compute_ns": without_compute(awaiting_delivery),
        "pending_fetches_without_physical_reads_ns": occupancy(fetch_intervals + physical_intervals, begin, end)[
            "busy_ns"
        ]
        - read_occupancy["busy_ns"],
        "pending_fetches_without_reads_or_compute_ns": occupancy(
            fetch_intervals + physical_intervals + compute_intervals, begin, end
        )["busy_ns"]
        - both_union,
    }
    state_time = joint_state_time(
        {
            "compute": compute_intervals,
            "runnable": [span for spans in queued_by_kind.values() for span in spans],
            "delivery_ready": awaiting_delivery,
            "physical_IO": physical_intervals,
            "pending_fetch": fetch_intervals,
            "driver_advancing": driver_intervals,
            "caller_gap": caller_intervals,
        },
        begin,
        end,
    )
    releases = [release for run in runs.values() for release in run["releases"]]
    registration_scopes = {}
    for run in runs.values():
        for root_id, root in run["roots"].items():
            work = [w for w in run["owners"].values() if w.get("root") == root_id]
            completed = bool(work) and all(w["end_reason"] == "retire" for w in work)
            cancelled = root_id in run["cancelled_roots"]
            registration_scopes[root["session"]] = {
                "end_ns": run["cancelled_roots"].get(root_id, max((w["observed_end_ns"] for w in work), default=end)),
                "status": "completed" if completed else "cancelled" if cancelled else "boundary",
            }
    for session, clears in scope_clears.items():
        scope = registration_scopes.setdefault(session, {"end_ns": end, "status": "unknown"})
        scope["clear_events"] = clears
    for session, forgets in scope_forgets.items():
        scope = registration_scopes.setdefault(session, {"end_ns": end, "status": "unknown"})
        scope["forget_events"] = forgets
    registrations = registration_profile(list(bindings.values()), registration_scopes, begin, end, list(reads.values()))
    lifetimes = lifetime_profile(events, list(bindings.values()), reads, members, end)
    selections = collections.defaultdict(list)
    for run in runs.values():
        for decision in run["selections"]:
            selections[stage_for(run, decision["owner"])].append(decision)
    selection_totals = {
        stage: {
            "decisions": len(items),
            "before_rows": sum(e["before_rows"] for e in items),
            "selected_rows": sum(e["selected_rows"] for e in items),
            "fully_rejected": sum(e["selected_rows"] == 0 for e in items),
            "unchanged": sum(e["selected_rows"] == e["before_rows"] for e in items),
            "survivor_fraction": distribution([e["selected_rows"] / max(1, e["before_rows"]) for e in items]),
            "first_decision_query_ns": min(e["ts_ns"] for e in items) - begin,
        }
        for stage, items in selections.items()
    }

    def release_profile(items: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "episodes": len(items),
            "made_runnable": sum(r["state"] == "NeedsCompute" for r in items),
            "made_done": sum(r["state"] == "Done" for r in items),
            "without_matched_release_request": sum(r["release_request"] is None for r in items),
            "computed_after_release": sum("first_compute_ns" in r for r in items),
            "runnable_without_observed_compute": sum(
                r["state"] == "NeedsCompute" and "first_compute_ns" not in r for r in items
            ),
            "next_compute_outcomes": dict(
                collections.Counter(r["next_compute_outcome"] for r in items if "next_compute_outcome" in r)
            ),
            "deliveries_per_park": distribution([r["deliveries_during_park"] for r in items]),
            **{
                field: distribution([r[field] for r in items if field in r])
                for field in (
                    "fetch_latency_ns",
                    "completion_to_unpark_ns",
                    "file_completion_to_delivery_ns",
                    "pread_to_delivery_ns",
                    "unpark_to_compute_ns",
                    "file_ready_while_driver_advancing_ns",
                    "file_ready_while_run_computing_ns",
                    "file_ready_outside_driver_advance_ns",
                )
            },
        }

    releases_by_stage = collections.defaultdict(list)
    for release in releases:
        releases_by_stage[release["stage"]].append(release)
    physical_fetches = collections.defaultdict(list)
    physical_releases = collections.Counter()
    first_fetch = {}
    for request in dispatched_fetches:
        physical_fetches[request["physical"]].append(request)
        member = (request["binding"]["source"], request["binding"]["read"])
        first_fetch[member] = min(first_fetch.get(member, request["ts_ns"]), request["ts_ns"])
    for release in releases:
        if "physical" in release and release["state"] == "NeedsCompute":
            physical_releases[release["physical"]] += 1
    announcements = {}
    for binding in bindings.values():
        if binding["intent"] == "Announce":
            member = (binding["source"], binding["read"])
            announcements[member] = min(announcements.get(member, binding["ts_ns"]), binding["ts_ns"])
    announced_fetches = [m for m in first_fetch if m in announcements]
    early_members = [
        m
        for m in first_fetch
        if m in members and "end_ns" in reads[members[m]] and reads[members[m]]["end_ns"] <= first_fetch[m]
    ]
    late_members = [
        m
        for m in first_fetch
        if m in members and "end_ns" in reads[members[m]] and reads[members[m]]["end_ns"] > first_fetch[m]
    ]
    anticipation = {
        "fetched_registrations": len(first_fetch),
        "with_announcement": len(announced_fetches),
        "announced_then_fetched_ns": distribution([first_fetch[m] - announcements[m] for m in announced_fetches]),
        "file_read_complete_before_first_fetch": len(early_members),
        "read_ready_lead_ns": distribution([first_fetch[m] - reads[members[m]]["end_ns"] for m in early_members]),
        "first_fetch_before_file_completion": len(late_members),
        "fetch_wait_for_file_completion_ns": distribution(
            [reads[members[m]]["end_ns"] - first_fetch[m] for m in late_members]
        ),
        "first_fetch_with_unfinished_file_read": sum(
            "end_ns" not in reads[members[m]] for m in first_fetch if m in members
        ),
        "interpretation": "Registration-weighted anticipation, not a causal prefetch gain. "
        "Announce is a hint; one physical range can contain demanded and speculative members.",
    }
    fanout = {
        "fetch_requests_per_read": distribution([len(physical_fetches[key]) for key in reads]),
        "owners_per_read": distribution(
            [len({(r["run"], r["owner"]) for r in physical_fetches[key]}) for key in reads]
        ),
        "runnable_releases_per_read": distribution([physical_releases[key] for key in reads]),
        "completed_reads_without_runnable_release": sum(
            not physical_releases[key] for key in reads if "end_ns" in reads[key]
        ),
    }
    local_bound = [r for r in completed_reads if "local_timestamps_ns" in r]
    handoff_delays = [r["end_ns"] - r["local_timestamps_ns"]["async_resumed"] for r in local_bound]
    progress = {
        "completed_physical_request_bytes": progress_profile(
            [(r["end_ns"], r["length"]) for r in completed_reads], begin
        ),
        "delivered_logical_fetch_bytes": progress_profile(
            [(r["delivered_ns"], r["length"]) for r in delivered_fetches], begin
        ),
        "emitted_batches": progress_profile([(ts, 1) for ts, _ in batch_points], begin),
        "emitted_rows": progress_profile(batch_points, begin),
        "interpretation": "Fractions use completed/emitted work observed in this window. "
        "They do not measure total required work or externally visible result rows.",
    }
    fetch_ranges = collections.defaultdict(list)
    delivered_ranges = collections.defaultdict(list)
    for request in dispatched_fetches:
        read = reads[request["physical"]]
        a = max(read["offset"], request["offset"])
        b = min(read["offset"] + read["length"], request["offset"] + request["length"])
        if b > a:
            fetch_ranges[request["physical"]].append((a, b))
            if "delivered_ns" in request:
                delivered_ranges[request["physical"]].append((a, b))
    member_coverage = {}
    for key, read in reads.items():
        member_coverage[key] = union_ns(
            [
                (max(read["offset"], m["offset"]), min(read["offset"] + read["length"], m["offset"] + m["length"]))
                for m in read.get("members", [])
                if "offset" in m and "length" in m
            ]
        )
    requested_bytes = sum(r["length"] for r in reads.values())
    coverage = {
        "started_request_bytes": requested_bytes,
        "coalesced_member_union_bytes": sum(member_coverage.values()),
        "coalescing_gap_bytes": sum(
            max(0, read["length"] - member_coverage[key])
            for key, read in reads.items()
            if any("offset" in m and "length" in m for m in read.get("members", []))
        ),
        "reads_without_member_ranges": sum(
            not any("offset" in m and "length" in m for m in read.get("members", [])) for read in reads.values()
        ),
        "fetch_requested_union_bytes": sum(union_ns(ranges) for ranges in fetch_ranges.values()),
        "fetch_delivered_union_bytes": sum(union_ns(ranges) for ranges in delivered_ranges.values()),
        "started_bytes_not_fetched_in_window": requested_bytes
        - sum(union_ns(ranges) for ranges in fetch_ranges.values()),
        "started_bytes_not_delivered_to_fetch_in_window": requested_bytes
        - sum(union_ns(ranges) for ranges in delivered_ranges.values()),
        "unfinished_request_bytes": sum(read["length"] for read in reads.values() if "end_ns" not in read),
    }
    stage_startup = {}
    for name in stages.keys() | owners_by_stage.keys():
        work_items = owners_by_stage[name]
        computed = [(run, w) for run, w in work_items if "first_compute_ns" in w]
        root_first = {}
        root_admit = {}
        for run, work in computed:
            key = (run["run"], work.get("root"))
            ts = work["first_compute_ns"]
            root_first[key] = min(root_first.get(key, ts), ts)
            root = run["roots"].get(work.get("root"))
            if root is not None:
                root_admit[key] = root["ts_ns"]
        stage_startup[name] = {
            "owners_spawned": len(work_items),
            "owners_computed": len(computed),
            "owners_without_compute": len(work_items) - len(computed),
            "first_spawn_query_ns": min((w["ts_ns"] - begin for _, w in work_items), default=None),
            "first_compute_query_ns": min((w["first_compute_ns"] - begin for _, w in computed), default=None),
            "first_batch_query_ns": min(
                (w["first_batch_ns"] - begin for _, w in work_items if "first_batch_ns" in w), default=None
            ),
            "owner_first_compute_delay_ns": distribution([w["first_compute_ns"] - w["ts_ns"] for _, w in computed]),
            "root_first_compute_delay_ns": distribution(
                [ts - root_admit[key] for key, ts in root_first.items() if key in root_admit]
            ),
            "root_first_compute_query_ns": distribution([ts - begin for ts in root_first.values()]),
            "ready_queue_depth": profile(f"ready {name}", queued_by_stage[name]),
            "parked_queue_depth": profile(f"parked {name}", parked_by_stage[name]),
            "dfs_depth": distribution([w["dfs_depth"] for _, w in work_items if w["dfs_depth"] is not None]),
        }
    dfs = {
        "method": "Reconstructed from single-threaded child emission/spawn order within each run; root depth is zero.",
        "max_depth": max(owners_by_depth, default=None),
        "owners_known": sum(len(owners) for owners in owners_by_depth.values()),
        "owners_unknown": sum(w["dfs_depth"] is None for run in runs.values() for w in run["owners"].values()),
        "steps_unknown": unknown_depth_steps,
        "by_depth": {
            depth: {
                "owners": len(work_items),
                "morsels": sum(w.get("kind") == "morsel" for w in work_items),
                "compute_steps": depth_steps[depth],
                "compute_total_ns": depth_compute[depth],
                "first_compute_query_ns": min(
                    (w["first_compute_ns"] - begin for w in work_items if "first_compute_ns" in w), default=None
                ),
                "ready_queue_depth": profile(f"ready work at DFS depth {depth}", queued_by_depth[depth]),
            }
            for depth, work_items in sorted(owners_by_depth.items())
        },
    }
    batch_times = sorted(ts for run in runs.values() for ts in run["batch_emission_ns"])
    if build_trace or build_queue_trace:
        timeline.extend(
            {"name": name, "ph": "i", "s": "t", "ts": ts / 1000, "pid": "Startup", "tid": "query window"}
            for name, ts in (("query begin", begin), ("query end", end))
        )
        timeline.extend(
            {
                "name": f"first compute: {name}",
                "ph": "i",
                "s": "t",
                "ts": (stage["first_compute_query_ns"] + begin) / 1000,
                "pid": "Startup",
                "tid": "first compute by stage",
                "args": {"dfs_depth": stage["dfs_depth"], "owners_computed": stage["owners_computed"]},
            }
            for name, stage in stage_startup.items()
            if stage["first_compute_query_ns"] is not None
        )
    summary = {
        "query": {
            "start_ns": begin,
            "end_ns": end,
            "wall_ns": end - begin,
            "exact_window": len(query) == 2,
            "rows": query.get("query_end", {}).get("rows"),
            "reads": read_occupancy,
            "source_io": source_io,
            "compute": compute_occupancy,
            "compute_ready": occupancy(ready_intervals, begin, end),
            "io_compute_overlap_ns": read_occupancy["busy_ns"] + compute_occupancy["busy_ns"] - both_union,
            "observed_local": {phase: occupancy(intervals, begin, end) for phase, intervals in local_intervals.items()},
            "local_coverage": "Completed, bound local reads only; unfinished reads have no completion-phase log.",
            "startup_ns": startup,
            "first_compute_events": first_steps,
            "queues": queues,
            "request_byte_depth": byte_depth,
            "local_phase_depth": local_phase_depth,
            "dfs": dfs,
            "gaps": gaps,
            "state_time": state_time,
            "progress": progress,
            "batch_gap_ns": distribution([b - a for a, b in zip(batch_times, batch_times[1:])]),
        },
        "stages": {
            name: {
                **{f"{key}_ns": distribution(stage[key]) for key in ("compute", "park", "schedule", "fetch")},
                "compute_total_ns": sum(stage["compute"]),
                "park_total_ns": sum(stage["park"]),
                "schedule_total_ns": sum(stage["schedule"]),
                "outputs": dict(stage["outputs"]),
                "rows": stage["rows"],
                "batch_rows": distribution(stage["batch_rows"]),
                "ready_wait_ended_without_compute_ns": distribution(stage["ready_censored"]),
                "fetch_unfinished_age_ns": distribution(stage["fetch_censored"]),
                "scheduling_interference": dict(stage["interference"]),
                "releases": release_profile(releases_by_stage[name]),
                **stage_startup[name],
            }
            for name in sorted(stages.keys() | owners_by_stage.keys())
            for stage in [stages[name]]
        },
        "diagnostics": {
            "events": len(events),
            "compute_steps": sum(r["steps"] for r in runs.values()),
            "compute_trace_threshold_ns": min_compute_ns,
            "queue_trace_resolution_ns": queue_resolution_ns if build_queue_trace else 0,
        },
        "runs": list(runs.values()),
        "warnings": warnings,
        "physical_reads": list(reads.values()),
        "totals": {
            "runs": len(runs),
            "partial_runs": sum(r["partial"] for r in runs.values()),
            "physical_reads": len(reads),
            "unfinished_reads": len(reads) - len(completed_reads),
            "physical_bytes": sum(r["length"] for r in reads.values()),
            "physical_members": len(members),
            "physical_coverage": coverage,
            "releases": release_profile(releases),
            "dfs_choices": dict(sum((run["dfs_choices"] for run in runs.values()), collections.Counter())),
            "anticipation": anticipation,
            "registration_lifetimes": registrations,
            "io_lifetimes": lifetimes,
            "selection_decisions": selection_totals,
            "selection_release": selection_release_profile(list(runs.values()), scope_clears, end),
            "completion_fanout": fanout,
            "async_resume_to_file_completion_ns": distribution([delay for delay in handoff_delays if delay >= 0]),
            "local_phase_order_conflicts": sum(delay < 0 for delay in handoff_delays),
            "bound_local_phase_reads": len(local_bound),
            "fetch_unfinished_age_ns": distribution(
                [r["observed_until_ns"] - r["ts_ns"] for r in all_fetches if "observed_until_ns" in r]
            ),
            "members_not_fetched_in_window": len(set(members) - fetched_members),
            "fetches_already_read": sum(reads[r["physical"]]["end_ns"] <= r["ts_ns"] for r in delivered_fetches),
            "dispatch_wait_ns": distribution(
                [max(0, reads[r["physical"]]["ts_ns"] - r["ts_ns"]) for r in dispatched_fetches]
            ),
            "delivery_delay_ns": distribution(
                [r["delivered_ns"] - max(r["ts_ns"], reads[r["physical"]]["end_ns"]) for r in delivered_fetches]
            ),
            "physical_latency_ns": distribution([r["end_ns"] - r["ts_ns"] for r in completed_reads]),
            "fetch_latency_ns": distribution(
                [r["delivered_ns"] - r["ts_ns"] for r in all_fetches if "delivered_ns" in r]
            ),
            "local_phases_ns": {
                phase: distribution([r[phase] for r in local_reads if phase in r])
                for phase in ("get_ns", "admission_ns", "allocation_ns", "queue_ns", "read_ns", "resume_ns")
            },
            "unbound_local_reads": unbound_local_reads,
            "physical_busy_union_ns": union_ns([(r["ts_ns"], r["end_ns"]) for r in completed_reads]),
            "physical_queue_ns": distribution([r["ts_ns"] - r["queued_ns"] for r in reads.values() if "queued_ns" in r])
            if queued
            else None,
            "max_read_queue": max((e.get("queued", 0) for e in queued.values()), default=0) if queued else None,
            **{
                key: sum(r[key] for r in runs.values())
                for key in (
                    "compute_ns",
                    "active_ns",
                    "blocked_ns",
                    "unfinished_blocked_ns",
                    "unfinished_owner_wait_ns",
                    "wake_delay_ns",
                    "consumer_gap_ns",
                    "fetch_bytes",
                    "unbound_fetches",
                    "undelivered_fetches",
                )
            },
        },
    }
    processes = {}
    threads = {}
    metadata = []
    for entry in timeline:
        process = entry["pid"]
        thread = (process, entry["tid"])
        if process not in processes:
            processes[process] = len(processes)
            metadata.append(
                {"name": "process_name", "ph": "M", "pid": processes[process], "tid": 0, "args": {"name": str(process)}}
            )
        if thread not in threads:
            threads[thread] = len(threads) + 1
            metadata.append(
                {
                    "name": "thread_name",
                    "ph": "M",
                    "pid": processes[process],
                    "tid": threads[thread],
                    "args": {"name": str(thread[1])},
                }
            )
        entry["pid"] = processes[process]
        entry["tid"] = threads[thread]
    return summary, {"traceEvents": metadata + timeline, "displayTimeUnit": "ms"}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("log", type=Path)
    parser.add_argument("--iteration", type=int, default=-1, help="Diagnostic iteration, default: last complete")
    parser.add_argument("--output", type=Path, help="Save the full per-run/request JSON report")
    trace_output = parser.add_mutually_exclusive_group()
    trace_output.add_argument("--trace", type=Path, help="Save a full Perfetto/Chrome trace")
    trace_output.add_argument(
        "--queue-trace", type=Path, help="Save a compact queue-depth and stage-startup Perfetto trace"
    )
    parser.add_argument(
        "--queue-resolution-us",
        type=float,
        default=1000,
        help="Counter extrema/endpoints interval in microseconds for --queue-trace (default: 1000; 0: every change)",
    )
    parser.add_argument(
        "--min-compute-us",
        type=float,
        default=10,
        help="Timeline step threshold; all steps still contribute to compute totals",
    )
    args = parser.parse_args()
    if args.min_compute_us < 0:
        parser.error("min-compute-us must be nonnegative")
    if args.queue_resolution_us < 0:
        parser.error("queue-resolution-us must be nonnegative")
    with args.log.open() as log:
        events = parse_events(log, args.iteration)
    summary, trace = analyze(
        events,
        int(args.min_compute_us * 1000),
        build_trace=args.trace is not None,
        build_queue_trace=args.queue_trace is not None,
        queue_resolution_ns=int(args.queue_resolution_us * 1000),
    )
    totals = summary["totals"]
    query = summary["query"]
    print(
        f"{totals['runs']} runs; {totals['physical_reads']} physical reads; "
        f"{totals['physical_bytes'] / 1e6:.2f} MB; {totals['partial_runs']} partial runs"
    )
    print(
        f"Query window: {query['wall_ns'] / 1e6:.3f} ms; exact={query['exact_window']}; "
        f"result rows={query['rows']}; peak physical reads={query['reads']['peak']}; "
        f"mean reads in flight={query['reads']['mean']:.2f}"
    )
    print(
        f"Compute union: {query['compute']['busy_ns'] / 1e6:.3f} ms; "
        f"IO/compute overlap: {query['io_compute_overlap_ns'] / 1e6:.3f} ms; "
        f"ready/active scan work union: {query['compute_ready']['busy_ns'] / 1e6:.3f} ms"
    )
    print("Startup offsets from query begin (ms; missing milestones are not inferred):")
    for name, delay in query["startup_ns"].items():
        print(f"{name}: {delay / 1e6:.3f}" if delay is not None else f"{name}: absent")
    print("queue | time-weighted mean | p95 depth | peak depth")
    for name, queue in query["queues"].items():
        print(f"{name} | {queue['mean']:.2f} | {queue['p95']} | {queue['peak']}")
    print("request bytes | time-weighted mean MB | p95 MB | peak MB (not resident memory)")
    for name, queue in query["request_byte_depth"].items():
        print(f"{name} | {queue['mean'] / 1e6:.2f} | {queue['p95'] / 1e6:.2f} | {queue['peak'] / 1e6:.2f}")
    dfs = query["dfs"]
    print(
        f"DFS depth (reconstructed): max={dfs['max_depth']}; known owners={dfs['owners_known']}; "
        f"unknown owners={dfs['owners_unknown']}; unknown compute steps={dfs['steps_unknown']}"
    )
    print(
        "stage | first compute from query ms | per-root startup p95 ms | "
        "owner startup p95 ms | DFS depth | ready queue peak"
    )
    for name, stage in sorted(summary["stages"].items()):
        first = stage["first_compute_query_ns"]
        first_text = f"{first / 1e6:.3f}" if first is not None else "absent"
        root_delay = stage["root_first_compute_delay_ns"]
        root_text = f"{root_delay['p95'] / 1e6:.3f}" if root_delay["count"] else "absent"
        owner_delay = stage["owner_first_compute_delay_ns"]
        owner_text = f"{owner_delay['p95'] / 1e6:.3f}" if owner_delay["count"] else "absent"
        depth = stage["dfs_depth"]
        depth_text = f"{depth['min']}..{depth['max']}" if depth["count"] else "unknown"
        print(
            f"{name} | {first_text} | {root_text} | {owner_text} | {depth_text} | {stage['ready_queue_depth']['peak']}"
        )
    for name, elapsed in query["gaps"].items():
        print(f"{name}: {elapsed / 1e6:.3f} ms (no observed scan compute, not CPU-idle time)")
    runs = summary["runs"]
    max_visits = max((r["max_consecutive_visits_with_other_work_ready"] for r in runs), default=0)
    max_span = max((r["max_consecutive_span_with_other_work_ready_ns"] for r in runs), default=0)
    print(f"DFS same-owner visits while other work was ready: max={max_visits}; longest span={max_span / 1e6:.3f} ms")
    coverage = totals["physical_coverage"]
    print(
        f"Started request MB: {coverage['started_request_bytes'] / 1e6:.2f}; "
        f"coalescing gaps MB: {coverage['coalescing_gap_bytes'] / 1e6:.2f}; "
        f"not fetched in window MB: {coverage['started_bytes_not_fetched_in_window'] / 1e6:.2f}; "
        f"not delivered in window MB: {coverage['started_bytes_not_delivered_to_fetch_in_window'] / 1e6:.2f}"
    )
    print("Mutually exclusive observed query states (not CPU-idle or causal bottleneck attribution):")
    for name, elapsed in query["state_time"]["categories_ns"].items():
        print(f"{name}: {elapsed / 1e6:.3f} ms")
    releases = totals["releases"]
    print(
        f"Unpark episodes: {releases['episodes']}; made runnable: {releases['made_runnable']}; "
        f"computed afterward: {releases['computed_after_release']}; "
        f"missing release request: {releases['without_matched_release_request']}"
    )
    for field in (
        "file_completion_to_delivery_ns",
        "unpark_to_compute_ns",
        "file_ready_while_driver_advancing_ns",
        "file_ready_outside_driver_advance_ns",
    ):
        values = releases[field]
        print(f"Release {field}: p95={values['p95'] / 1e6:.3f} ms; count={values['count']}")
    print(
        f"DFS priority comparisons: {totals['dfs_choices'].get('comparisons', 0)}; "
        f"earlier runnable priorities bypassed: {totals['dfs_choices'].get('earlier_ready_priority_bypassed', 0)}"
    )
    print("stage | ancestor-continuation compute exposure ms | peer-branch compute exposure ms (overlapping owners)")
    for name, stage in summary["stages"].items():
        interference = stage["scheduling_interference"]
        print(
            f"{name} | {interference.get('ancestor_continuation_compute_ns', 0) / 1e6:.3f} | "
            f"{interference.get('peer_branch_compute_ns', 0) / 1e6:.3f}"
        )
    print("progress | observed work | time to 50% ms | time to 90% ms (observed denominator)")
    for name, values in query["progress"].items():
        if isinstance(values, dict):
            milestones = values["time_to_fraction_ns"]
            middle = f"{milestones[50] / 1e6:.3f}" if 50 in milestones else "absent"
            late = f"{milestones[90] / 1e6:.3f}" if 90 in milestones else "absent"
            print(f"{name} | {values['observed_total']} | {middle} | {late}")
    registrations = totals["registration_lifetimes"]
    print(
        f"Registrations: {registrations['registrations']}; scope interests: {registrations['scope_interests']}; "
        f"shared across overlapping scopes: {registrations['registrations_shared_by_overlapping_scopes']}; "
        f"unknown scope endpoint: {registrations['with_unknown_scope_end']}"
    )
    for status, values in registrations["unused_optional_by_scope_status"].items():
        print(
            f"Never fetched optional interests in {status} scopes: {values['interests']}; "
            f"interest MB={values['interest_bytes'] / 1e6:.2f} (ranges may be shared)"
        )
    unused_physical = registrations["unused_completed_scope_physical"]
    print(
        f"Never fetched registrations with all scopes completed: "
        f"{registrations['unused_registrations_with_only_completed_scopes']}; "
        f"selected in physical reads: {unused_physical['registrations_in_started_reads']}; "
        f"exclusive member MB={unused_physical['exclusive_of_other_member_ranges_bytes'] / 1e6:.3f} "
        "(not proven avoidable IO)"
    )
    life = totals["io_lifetimes"]
    print(
        f"Exact IO lifetime tracing available: {life['available']}; "
        f"cleared scopes: {life['cleared_scopes']}; dropped registrations: {life['dropped_registrations']}; "
        f"drop phases: {life['drop_phase_counts']}"
    )
    withdrawals = life["optional_withdrawals"]
    if withdrawals["interests"]:
        print(
            f"Optional withdrawals: {withdrawals['interests']}; logical MB={withdrawals['logical_bytes'] / 1e6:.3f}; "
            f"last reference observed={withdrawals['last_reference_observed']}; phases={withdrawals['phase_counts']} "
            "(not saved physical bytes)"
        )
    for field in (
        "wake_to_next_session_poll_ns",
        "completed_registration_retention_ns",
        "last_reference_to_physical_completion_ns",
    ):
        values = life[field]
        print(f"Lifetime {field}: p95={values['p95'] / 1e6:.3f} ms; count={values['count']}")
    for stage, selection in totals["selection_decisions"].items():
        print(
            f"Selection {stage}: decisions={selection['decisions']}; "
            f"fully rejected={selection['fully_rejected']}; unchanged={selection['unchanged']}; "
            f"survivor fraction p50={selection['survivor_fraction']['median']:.4f}"
        )
    for kind, values in totals["selection_release"]["rejection_to_root_clear_ns"].items():
        print(f"Rejection to root IO clear ({kind}): p95={values['p95'] / 1e6:.3f} ms; count={values['count']}")
    print("stage | compute ms | parked ms (overlapping) | ready-to-compute p95 ms | fetch p95 ms | steps")
    for name, stage in sorted(summary["stages"].items(), key=lambda item: -item[1]["compute_total_ns"]):
        print(
            f"{name} | {stage['compute_total_ns'] / 1e6:.3f} | {stage['park_total_ns'] / 1e6:.3f} | "
            f"{stage['schedule_ns']['p95'] / 1e6:.3f} | {stage['fetch_ns']['p95'] / 1e6:.3f} | "
            f"{stage['compute_ns']['count']}"
        )
    for phase, values in totals["local_phases_ns"].items():
        if not values["count"]:
            print(f"Local {phase}: unavailable")
            continue
        print(
            f"Local {phase}: p50={values['median'] / 1e6:.3f} ms "
            f"p95={values['p95'] / 1e6:.3f} ms max={values['max'] / 1e6:.3f} ms count={values['count']}"
        )
    print("Per-file sums across concurrent runs (milliseconds, not query wall time):")
    files = collections.defaultdict(lambda: collections.Counter())
    for run in summary["runs"]:
        uris = sorted({root["uri"] for root in run["roots"].values()})
        name = ", ".join(uris) or "<source without file binding>"
        files[name].update(
            {
                key: run[key]
                for key in (
                    "compute_ns",
                    "blocked_ns",
                    "wake_delay_ns",
                    "driver_overhead_ns",
                    "fetch_bytes",
                )
            }
        )
        files[name]["runs"] += 1
    print("file | runs | compute ms | IO blocked ms | wake delay ms | driver ms | fetch MB")
    for name, values in sorted(files.items()):
        print(
            f"{name} | {values['runs']} | "
            + " | ".join(
                f"{values[key] / 1e6:.3f}"
                for key in (
                    "compute_ns",
                    "blocked_ns",
                    "wake_delay_ns",
                    "driver_overhead_ns",
                    "fetch_bytes",
                )
            )
        )
    print(
        f"Unbound fetches: {totals['unbound_fetches']}; "
        f"undelivered: {totals['undelivered_fetches']}; unfinished reads: {totals['unfinished_reads']}"
    )
    for warning in summary["warnings"][:10]:
        print(f"WARNING: {warning}")
    if args.output:
        args.output.write_text(json.dumps(summary, separators=(",", ":")) + "\n")
    if args.trace:
        args.trace.write_text(json.dumps(trace) + "\n")
    if args.queue_trace:
        args.queue_trace.write_text(json.dumps(trace) + "\n")


if __name__ == "__main__":
    main()
