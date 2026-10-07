#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors
"""Attribute a Samply profile of duckdb-bench / datafusion-bench to scan vs engine categories.

Estimator: every sample is one sampling interval of one thread's wall time. A sample whose leaf
frame is a blocking wait is off-CPU; everything else is on-CPU. (threadCPUDelta is only used as a
cross-check: on macOS it is charged to whatever stack the thread has when sampled, which for
threads that block many times per millisecond is the wait, not the work.)

usage: prof_attr.py <profile.json.gz> [--json out.json] [--top N] [--quiet]
"""

import bisect
import collections
import json
import re
import sys
from typing import Any

import proflib

INTERVAL = 1.0  # ms; overwritten from the profile's meta.interval

WAIT_LEAF = {
    "__psynch_cvwait",
    "semaphore_wait_trap",
    "semaphore_timedwait_trap",
    "__ulock_wait",
    "__ulock_wait2",
    "kevent",
    "kevent64",
    "kevent_id",
    "mach_msg2_trap",
    "mach_msg_trap",
    "__psynch_mutexwait",
    "__select",
    "poll",
    "__semwait_signal",
    "__workq_kernreturn",
    "__sigsuspend",
    "__wait4",
    "__psynch_rw_rdlock",
    "__psynch_rw_wrlock",
}
# Frames that never decide a category on their own: allocator, libc copies, generic wrappers.
TRANSPARENT_LIBS = {
    "libsystem_malloc.dylib",
    "libsystem_platform.dylib",
    "libsystem_pthread.dylib",
    "libsystem_c.dylib",
    "libc++.1.dylib",
    "libc++abi.dylib",
    "libdispatch.dylib",
    "dyld",
    "libsystem_m.dylib",
    "libdyld.dylib",
    "",
}
ALLOC = re.compile(
    r"^(_?mi_|__rust_(alloc|dealloc|realloc)|__rdl_|_?malloc|_?free|_platform_mem|madvise|mmap|munmap|"
    r"alloc::raw_vec|<alloc::raw_vec|__bzero|_?realloc|_?calloc)"
)

# (category, regex) evaluated on each frame from leaf to root; first frame that matches wins.
KIND_RULES = [
    ("io: pread (kernel copy + in-kernel wait)", r"^(pread|read|readv|preadv)$"),
    (
        "io: open/close/stat syscalls",
        r"^(__open|open|close|fstat|fstat64|lseek|stat|stat64|access|fcntl|__fcntl|getattrlist|__open_nocancel|__close_nocancel|close_nocancel|open_nocancel)$",
    ),
    ("io: pool spin/yield", r"^(swtch_pri|cthread_yield)$"),
    (
        "io: dispatch + segment source",
        r"vortex_io::|vortex_file::segments|vortex_file::read|vortex_layout::segments|object_store::|tokio::fs|std::fs::|vortex_file::file|vortex_file::open|vortex_file::footer|blocking_pool",
    ),
    (
        "convert: to engine vectors",
        r"vortex_duckdb::exporter|vortex_duckdb::convert|vortex_duckdb::duckdb::vector|vortex_duckdb::duckdb::data_chunk|ArrayExporter|vortex_arrow|execute_arrow|to_arrow|vortex_datafusion::convert|vortex_datafusion::persistent::stream|SchemaAdapter|schema_adapter|arrow_cast|vortex_duckdb::file_reader::convert_result",
    ),
    ("decode: bitpacking", r"fastlanes::bitpacking|vortex_fastlanes::bitpacking|BitPacked"),
    ("decode: FoR", r"vortex_fastlanes::for|fastlanes::ffor|::FoR "),
    ("decode: delta", r"vortex_fastlanes::delta|fastlanes::delta"),
    ("decode: RLE", r"vortex_fastlanes::rle|fastlanes::rle"),
    ("decode: ALP", r"vortex_alp|alp::"),
    ("decode: FSST", r"vortex_fsst|fsst::"),
    ("decode: dict", r"vortex_array::arrays::dict|vortex_dict|arrays::dict"),
    ("decode: run-end", r"vortex_runend"),
    ("decode: decimal/datetime parts", r"vortex_decimal_byte_parts|vortex_datetime_parts"),
    ("decode: zstd/pco", r"vortex_zstd|zstd|vortex_pco|pco::"),
    (
        "decode: other encodings",
        r"vortex_zigzag|vortex_sequence|vortex_sparse|vortex_bytebool|vortex_onpair|onpair|arrays::constant|arrays::chunked|arrays::varbin|arrays::primitive|arrays::decimal|arrays::bool|arrays::struct_|arrays::extension|arrays::list|arrays::null|arrays::masked|vortex_array::patches|arrays::patched",
    ),
    ("decode: array deserialise", r"SerializedArray|vortex_array::serde|flatbuffers|array_future|vortex_flatbuffers"),
    (
        "filter: selection/mask kernels",
        r"arrays::filter|fixed_width::filter|vortex_mask|intersect_by_rank|vortex_buffer::bit|arrays::slice|compute::filter|compute::take|::take::|TakeKernel|FilterKernel|vortex_compute",
    ),
    (
        "filter: expression kernels",
        r"scalar_fn::fns|vortex_array::expr|vortex_array::scalar_fn|arrow_ord|arrow_string|arrow_arith|regex|memchr|vortex_array::compute|vortex_array::aggregate|aggregate_fn",
    ),
    (
        "pruning: zone maps/stats",
        r"layouts::zoned|pruning|vortex_array::stats|stats_set|StatsSet|zone_map|layouts::file_stats",
    ),
    (
        "array executor dispatch",
        r"vortex_array::executor|vortex_array::canonical|vortex_array::array::|vortex_array::columnar|vortex_array::mask::|vortex_array::dtype|vortex_array::scalar::|vortex_array::validity|vortex_array::|vortex_buffer|vortex_dtype|vortex_scalar|vortex_session",
    ),
    (
        "scan: layout readers/plan exec",
        r"vortex_layout::layouts|vortex_layout::plan|vortex_layout::reader|vortex_layout::scan|vortex_scan::|vortex_layout::|MaskFuture|mask_future",
    ),
    (
        "scan: open/bind/init (table fn)",
        r"vortex_duckdb|VortexMultiFileReader|VortexBaseReader|VortexReaderInterface|duckdb_reader_|vortex_datafusion|vortex::",
    ),
    (
        "scan: async/scheduling overhead",
        r"futures_util|futures_core|futures_channel|async_executor|async_io|async_task|parking|crossbeam|tokio::|event_listener|concurrent_queue|async_lock|std::sync|std::thread|parking_lot|core::ptr::drop|alloc::sync|alloc::task|core::task|oneshot",
    ),
]
KIND_RULES = [(c, re.compile(r)) for c, r in KIND_RULES]

SCAN_MARK = re.compile(
    r"vortex_layout|vortex_file|vortex_io::|vortex_scan|vortex_array|vortex_duckdb::(file_reader|exporter|table_function|convert|projection|column_statistics)|duckdb_reader_|VortexBaseReader|VortexMultiFileReader|"
    r"VortexReaderInterface|vortex_datafusion|vortex_fastlanes|vortex_fsst|vortex_alp|vortex_mask|blocking_pool|vortex_runend|vortex_buffer"
)
HARNESS_MARK = re.compile(r"vortex_bench::|duckdb_bench::|datafusion_bench::")
QUERY_MARK_DUCK = re.compile(r"execute_query_result")

PHASE_RULES = [
    (
        "export to engine",
        re.compile(
            r"ArrayExporter|vortex_duckdb::exporter|convert_result|vortex_datafusion::persistent::stream|vortex_arrow"
        ),
    ),
    (
        "pruning",
        re.compile(r"pruning_evaluation|layouts::zoned|plan::exec::zoned|ZonedExec|scan::v2::conjuncts.*prun|Pruning"),
    ),
    (
        "filter",
        re.compile(
            r"split_exec::<[^>]*>::\{closure#1\}|filter_evaluation|FilterPlanner|scan::planning::filter|FilterMorsel"
        ),
    ),
    (
        "projection",
        re.compile(
            r"split_exec::<[^>]*>::\{closure#2\}|projection_evaluation|ProjectionPlanner|ProjectionMorsel|scan::planning::projection"
        ),
    ),
    (
        "open/init/plan",
        re.compile(
            r"reader_initialize|reader_open|reader_bind|MultiFileInitGlobal|InitializeReader|scan::v2::repeated_scan|scan_builder|VortexOpener|vortex_datafusion::persistent::(opener|format|source)|get_statistics|footer"
        ),
    ),
]


def kind_of(stack: tuple[tuple[str, str], ...]) -> tuple[str, bool]:
    alloc = False
    for lib, name in reversed(stack):
        if ALLOC.search(name):
            alloc = True
            continue
        if lib == "libsystem_kernel.dylib":
            for cat, rx in KIND_RULES[:3]:
                if rx.search(name):
                    return cat, alloc
            continue
        if lib in TRANSPARENT_LIBS:
            continue
        if lib == "libduckdb.dylib":
            return "duckdb C API (called from scan)", alloc
        for cat, rx in KIND_RULES:
            if rx.search(name):
                return cat, alloc
    return "scan: other", alloc


def phase_of(stack: tuple[tuple[str, str], ...]) -> str:
    names = [n for _, n in stack]
    for ph, rx in PHASE_RULES:
        if ph in ("filter", "pruning", "export to engine"):
            if any(rx.search(n) for n in names):
                return ph
    for ph, rx in PHASE_RULES[3:]:
        if any(rx.search(n) for n in names):
            return ph
    return "other"


def analyse(path: str) -> dict[str, Any]:
    global INTERVAL
    samples = proflib.load(path)
    INTERVAL = proflib.INTERVAL
    is_duck = any("duckdb_query" == n for s in samples[:20000] for _, n in s["stack"]) or "duck" in path
    # query windows from the main thread (DuckDB only)
    main = sorted((s["time"], any(QUERY_MARK_DUCK.search(n) for _, n in s["stack"])) for s in samples if s["main"])
    main_times = [t for t, _ in main]

    def in_query(t: float) -> bool:
        if not is_duck or not main:
            return True
        i = bisect.bisect_left(main_times, t)
        i = min(max(i, 0), len(main) - 1)
        return main[i][1]

    res = collections.Counter()  # top-level buckets (thread-time, in samples)
    kinds = collections.Counter()  # scan on-CPU by kind
    phases = collections.Counter()  # scan on-CPU by phase
    kind_phase = collections.Counter()
    alloc_in = collections.Counter()
    selfc = collections.Counter()
    cpu_us = collections.Counter()
    n_in = 0
    io_tids, all_tids = set(), set()
    bins = collections.defaultdict(lambda: [0, 0, 0, 0])  # scan on-cpu, engine on-cpu, harness, scan blocked
    for s in samples:
        st = s["stack"]
        if not st or not in_query(s["time"]):
            res["(outside query window)"] += 1
            continue
        n_in += 1
        leaf_lib, leaf = st[-1]
        names = [n for _, n in st]
        blocked = leaf in WAIT_LEAF
        yielding = leaf in ("swtch_pri", "cthread_yield", "sched_yield")
        rnd = int(round(s["time"] / INTERVAL))
        is_io_thread = s["thread"].startswith("vortex-blocking-io") or (
            any("blocking::pool::Spawner>::spawn_thread" in n or "BlockingPool>::grow" in n for n in names[:8])
            and not any("multi_thread::worker" in n for n in names[:16])
        )
        if is_io_thread:
            io_tids.add(s["tid"])
        all_tids.add(s["tid"])
        scan = is_io_thread or any(SCAN_MARK.search(n) for n in names)
        harness_only = s["main"] and not any(n == "duckdb_query" or "datafusion" in n for n in names) and not scan
        if blocked:
            if is_io_thread:
                res["idle: io-pool thread parked"] += 1
            elif scan:
                res["scan: thread blocked inside scan (waiting for IO/another thread)"] += 1
                bins[rnd][3] += 1
                if leaf == "__psynch_mutexwait":
                    res["  of which: mutex wait"] += 1
            else:
                res["idle: engine thread waiting"] += 1
            continue
        if yielding:
            # spin-yield loops: the thread is runnable but burns little CPU per sample; account the
            # CPU from threadCPUDelta instead of counting the sample as a full interval of work.
            key = "scan: spin/yield (io pool + executor)" if scan else "engine: spin/yield"
            res[key] += 1
            cpu_us[key] += s["cpu"]
            continue
        cpu_us["on-cpu"] += s["cpu"]
        bins[rnd][0 if scan else (2 if harness_only else 1)] += 1
        if scan:
            k, alloc = kind_of(st)
            p = "io thread" if is_io_thread else phase_of(st)
            res["scan: on-CPU"] += 1
            kinds[k] += 1
            phases[p] += 1
            kind_phase[(p, k)] += 1
            if alloc:
                alloc_in["scan"] += 1
            selfc[(leaf_lib, leaf)] += 1
        elif harness_only:
            res["bench harness (main thread, outside engine)"] += 1
        else:
            res["engine: on-CPU (operators, planning, runtime)"] += 1
            if any(ALLOC.search(n) for n in names[-3:]):
                alloc_in["engine"] += 1
    # Wall share: every sampling round with any engine/scan activity is one unit of wall time, split
    # between scan and engine in proportion to the threads in each state during that round.
    wall_scan = wall_scan_cpu_only = wall_rounds = 0.0
    for sc, en, ha, bl in bins.values():
        if sc + en + bl == 0:
            continue
        wall_rounds += 1
        wall_scan += (sc + bl) / (sc + en + bl)
        if sc + en:
            wall_scan_cpu_only += sc / (sc + en)
    total_cpu_us = sum(s["cpu"] for s in samples if s["stack"] and in_query(s["time"]))
    return dict(
        path=path,
        engine="duckdb" if is_duck else "datafusion",
        res=res,
        kinds=kinds,
        phases=phases,
        kind_phase=kind_phase,
        alloc_in=alloc_in,
        selfc=selfc,
        n_in=n_in,
        total_cpu_ms=total_cpu_us / 1000.0,
        wall_rounds=wall_rounds,
        wall_scan=wall_scan,
        wall_scan_cpu_only=wall_scan_cpu_only,
        cpu_us=cpu_us,
        io_threads=len(io_tids),
        threads=len(all_tids),
    )


def summarise(a: dict[str, Any]) -> dict[str, Any]:
    res = a["res"]
    scan_cpu = res["scan: on-CPU"]
    scan_wait = res["scan: thread blocked inside scan (waiting for IO/another thread)"]
    eng = res["engine: on-CPU (operators, planning, runtime)"]
    har = res["bench harness (main thread, outside engine)"]
    busy = scan_cpu + scan_wait + eng
    oncpu = scan_cpu + eng
    return dict(
        scan_cpu=scan_cpu,
        scan_wait=scan_wait,
        engine_cpu=eng,
        harness=har,
        scan_share_cpu=scan_cpu / oncpu if oncpu else 0.0,
        scan_share_busy=(scan_cpu + scan_wait) / busy if busy else 0.0,
        oncpu_samples=oncpu + har,
        cpu_delta_ms=a["total_cpu_ms"],
        scan_share_wall=a["wall_scan"] / a["wall_rounds"] if a["wall_rounds"] else 0.0,
        scan_share_wall_cpu_only=a["wall_scan_cpu_only"] / a["wall_rounds"] if a["wall_rounds"] else 0.0,
        spin_cpu_ms=a["cpu_us"]["scan: spin/yield (io pool + executor)"] / 1000.0,
        spin_samples=res["scan: spin/yield (io pool + executor)"],
        io_threads=a["io_threads"],
        threads=a["threads"],
        res=dict(res),
        kinds=dict(a["kinds"]),
        phases=dict(a["phases"]),
    )


def main():
    args = sys.argv[1:]
    path = args[0]
    top = int(args[args.index("--top") + 1]) if "--top" in args else 12
    a = analyse(path)
    s = summarise(a)
    if "--json" in args:
        with open(args[args.index("--json") + 1], "w") as f:
            json.dump(s, f)
    if "--quiet" in args:
        return
    res = a["res"]
    busy = s["scan_cpu"] + s["scan_wait"] + s["engine_cpu"]
    oncpu = s["scan_cpu"] + s["engine_cpu"]
    print(f"# {path} ({a['engine']})")
    print(
        f"samples in query windows: {a['n_in']}; on-CPU samples {s['oncpu_samples']} (x interval ~= ms) vs "
        f"threadCPUDelta total {a['total_cpu_ms']:.0f} ms"
    )
    print(
        f"scan share of on-CPU time: {100 * s['scan_share_cpu']:.1f}%   "
        f"scan share of busy thread-time (on-CPU + blocked-in-scan): {100 * s['scan_share_busy']:.1f}%   "
        f"scan share of wall (per-round): {100 * s['scan_share_wall']:.1f}%"
    )
    print(f"threads seen in query windows: {a['threads']} (io-pool threads: {a['io_threads']})")
    print(
        f"spin/yield under scan: {s['spin_samples']} samples, {s['spin_cpu_ms']:.0f} ms CPU by threadCPUDelta "
        f"({100 * s['spin_cpu_ms'] / max(a['total_cpu_ms'], 1):.1f}% of all CPU)"
    )
    print("\n| bucket | samples | % of busy thread-time | % of on-CPU |")
    print("|---|---:|---:|---:|")
    for k, v in sorted(res.items(), key=lambda kv: -kv[1]):
        on = f"{100 * v / oncpu:.1f}" if ("on-CPU" in k) else ""
        b = f"{100 * v / busy:.1f}" if (k.startswith("scan") or k.startswith("engine")) else ""
        print(f"| {k} | {v} | {b} | {on} |")
    print("\n| scan on-CPU by kind | % of scan on-CPU | % of all on-CPU |")
    print("|---|---:|---:|")
    for k, v in a["kinds"].most_common():
        print(f"| {k} | {100 * v / max(s['scan_cpu'], 1):.1f} | {100 * v / max(oncpu, 1):.1f} |")
    print(
        f"| (alloc/free/memcpy anywhere under scan) | {100 * a['alloc_in']['scan'] / max(s['scan_cpu'], 1):.1f} | "
        f"{100 * a['alloc_in']['scan'] / max(oncpu, 1):.1f} |"
    )
    print("\n| scan on-CPU by phase | % of scan on-CPU | % of all on-CPU |")
    print("|---|---:|---:|")
    for k, v in a["phases"].most_common():
        print(f"| {k} | {100 * v / max(s['scan_cpu'], 1):.1f} | {100 * v / max(oncpu, 1):.1f} |")
    print("\nTop scan self frames (% of scan on-CPU):")
    for (lib, name), v in a["selfc"].most_common(top):
        print(f"  {100 * v / max(s['scan_cpu'], 1):5.1f}%  [{lib}] {name[:140]}")


if __name__ == "__main__":
    main()
