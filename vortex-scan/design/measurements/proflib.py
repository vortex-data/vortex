# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors
"""Load a Samply profile plus its --unstable-presymbolicate sidecar into flat per-sample stacks."""

import bisect
import gzip
import json
import os
import re
import sys
from typing import Any

INTERVAL = 1.0


class Syms:
    def __init__(self, sidecar_path: str):
        s = json.load(open(sidecar_path))
        self.strings = s["string_table"]
        self.libs = {}
        for d in s["data"]:
            table = d.get("symbol_table", [])
            known = {a: i for a, i in d.get("known_addresses", [])}
            rvas = [e["rva"] for e in table]
            self.libs[d["debug_name"]] = (table, known, rvas)

    def lookup(self, lib: str, addr: int) -> str | None:
        ent = self.libs.get(lib)
        if ent is None:
            return None
        table, known, rvas = ent
        i = known.get(addr)
        if i is None:
            j = bisect.bisect_right(rvas, addr) - 1
            if j < 0:
                return None
            e = table[j]
            if addr >= e["rva"] + e.get("size", 1 << 30):
                return None
            i = j
        return self.strings[table[i]["symbol"]]


_HASH = re.compile(r"::h[0-9a-f]{16}$")


def clean(sym: str) -> str:
    return _HASH.sub("", sym)


def load(profile_path: str) -> list[dict[str, Any]]:
    """Yield dicts: {thread, tid, time, cpu_us, stack: [(lib, sym), ... root->leaf]} per sample."""
    opener = gzip.open if profile_path.endswith(".gz") else open
    with opener(profile_path, "rt") as f:
        prof = json.load(f)
    sidecar = re.sub(r"\.json(\.gz)?$", ".json.syms.json", profile_path)
    syms = Syms(sidecar) if os.path.exists(sidecar) else None
    libs = prof["libs"]
    global INTERVAL
    INTERVAL = prof["meta"].get("interval", 1.0)
    main_pid = None
    for t in prof["threads"]:
        if t.get("isMainThread") and "bench" in (t.get("processName") or t.get("name") or ""):
            main_pid = t.get("pid")
            break
    out = []
    for t in prof["threads"]:
        if main_pid is not None and t.get("pid") != main_pid:
            continue
        strings = t["stringArray"]
        ft, fn, rt, st, sm = t["frameTable"], t["funcTable"], t["resourceTable"], t["stackTable"], t["samples"]
        frame_label = {}

        def label(fi: int) -> tuple[str, str]:
            v = frame_label.get(fi)
            if v is None:
                func = ft["func"][fi]
                res = fn["resource"][func]
                lib = ""
                if res is not None and res >= 0:
                    li = rt["lib"][res]
                    lib = libs[li]["debugName"] if li is not None else strings[rt["name"][res]]
                name = strings[fn["name"][func]]
                if syms is not None and name.startswith("0x"):
                    s = syms.lookup(lib, ft["address"][fi])
                    if s:
                        name = clean(s)
                v = (lib, name)
                frame_label[fi] = v
            return v

        stack_cache = {}

        def expand(si: int | None) -> tuple[tuple[str, str], ...]:
            if si is None:
                return ()
            v = stack_cache.get(si)
            if v is None:
                v = expand(st["prefix"][si]) + (label(st["frame"][si]),)
                stack_cache[si] = v
            return v

        cpu = sm.get("threadCPUDelta") or [0] * len(sm["stack"])
        times = sm.get("time") or [0] * len(sm["stack"])
        sys.setrecursionlimit(100000)
        for i, si in enumerate(sm["stack"]):
            out.append(
                {
                    "thread": t.get("name", ""),
                    "tid": t.get("tid"),
                    "main": bool(t.get("isMainThread")),
                    "time": times[i],
                    "cpu": max(cpu[i] or 0, 0),
                    "stack": expand(si),
                }
            )
    return out
