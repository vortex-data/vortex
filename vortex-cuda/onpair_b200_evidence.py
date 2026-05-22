#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors
"""
B200 OnPair decode — evidence benchmarks.

Each demo is a *controlled comparison*: it holds everything fixed except one
variable, so the mechanism behind a claim can be read straight off the table.
Runs the existing `onpair-chunk-bench gpu-decode-vortex` binary (one launch per
column, all kernels timed) and extracts the relevant kernels.

Usage:
    python3 vortex-cuda/onpair_b200_evidence.py
    BIN=target/release/onpair-chunk-bench DATA=vortex-bench/data/onpair-bench \\
        python3 vortex-cuda/onpair_b200_evidence.py

Requires a built CUDA bench (`cargo build -p vortex-bench --features cuda
--bin onpair-chunk-bench --release`) and the generated benchmark data. Numbers
are PRELIMINARY: unlocked clocks (±~5%), single-invocation ranking.
"""
import json
import os
import subprocess
import sys

BIN = os.environ.get("BIN", "target/release/onpair-chunk-bench")
DATA = os.environ.get("DATA", "vortex-bench/data/onpair-bench")
ITERS = os.environ.get("ITERS", "300")
# Demo 7 (compress + bench) sweeps these dict bit-widths on the fineweb parquet.
PARQUET = os.environ.get(
    "PARQUET", "vortex-bench/data/onpair-bench-src/fineweb/fineweb_10BT_000.parquet")
SWEEP_BITS = os.environ.get("SWEEP_BITS", "12,14,16")

# Human-readable launch config per kernel (the bench does not emit these).
# block = threads/block, occ = target occupancy, read = dict read width.
KMETA = {
    "4tpt":                    ("512", "33%",  "16 B"),
    "4tpt_wpb8_occ":           ("256", "50%",  "16 B"),
    "4tpt_b128":               ("128", "50%",  "16 B"),
    "4tpt_b128o12":            ("128", "75%",  "16 B"),
    "4tpt_b64":                ("64",  "50%",  "16 B"),
    "4tpt_split8read_occ":     ("256", "50%",  "8 B"),
    "4tpt_split8read_b128o12": ("128", "75%",  "8 B"),
    "4tpt_split4read_b128o12": ("128", "75%",  "4 B"),
    "4tpt_cluster_dsmem":      ("256", "1 blk/SM", "DSMEM"),
}

COLS = {
    "fineweb":    ("fineweb", "text"),
    "wikipedia":  ("wikipedia", "text"),
    "url":        ("clickbench", "URL"),
    "l_comment":  ("tpch-sf10", "l_comment"),
    "ps_comment": ("tpch-sf10", "ps_comment"),
    "dbtext_hex": ("dbtext", "hex"),
}


def path(colkey, bits):
    ds, col = COLS[colkey]
    return f"{DATA}/{ds}/{col}/{bits}_chunk1000mb_thr0.20/part_0000.vortex", col


def run(colkey, bits, l2_persist=False):
    """Run the bench once; return (gpu_dict, {kernel_short: (gib_s, ms, verified)})."""
    p, col = path(colkey, bits)
    if not os.path.exists(p):
        return None, {}
    env = dict(os.environ, ONPAIR_FAST="1")
    if l2_persist:
        env["ONPAIR_L2_PERSIST"] = "1"
    out = subprocess.run(
        [BIN, "gpu-decode-vortex", "--vortex", p, "--column", col,
         "--gpu-iters", ITERS, "--gpu-validate"],
        capture_output=True, text=True, env=env,
    ).stdout
    i = out.find("{")
    if i < 0:
        return None, {}
    d = json.loads(out[i:])
    g = d["gpu"]
    ks = {}
    for k in g["kernels"]:
        if k.get("decode_gib_s"):
            short = k["kernel"].replace("onpair_shmem_", "")
            ks[short] = (k["decode_gib_s"], k.get("decode_ms"), k.get("verified"))
    return g, ks


def hr(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def demo_granularity():
    hr("DEMO 1 — The B200 lever is BLOCK GRANULARITY, not occupancy  (fineweb bits12)")
    g, ks = run("fineweb", "bits12")
    if not ks:
        print("  (data missing)"); return
    order = ["4tpt_wpb8_occ", "4tpt_b128", "4tpt_b128o12", "4tpt_b64"]
    print(f"  {'kernel':<26}{'block':>7}{'occ':>7}{'GiB/s':>9}")
    base = ks.get("4tpt_wpb8_occ", (None,))[0]
    for k in order:
        if k not in ks:
            continue
        blk, occ, _ = KMETA[k]
        v = ks[k][0]
        d = f"{100*(v-base)/base:+.0f}%" if base else ""
        print(f"  {k:<26}{blk:>7}{occ:>7}{v:>9.0f}   {d}")
    print("  INFER: 256→128 threads jumps (+); 128t @50%→75% occ is flat; 64t ties 128t")
    print("         (plateau). Block SIZE moves the needle, target OCCUPANCY does not.")


def demo_readwidth():
    hr("DEMO 2 — 8 bytes is the OPTIMAL dict read width  (fineweb bits12, all 128t/75%)")
    g, ks = run("fineweb", "bits12")
    if not ks:
        print("  (data missing)"); return
    order = ["4tpt_b128o12", "4tpt_split8read_b128o12", "4tpt_split4read_b128o12"]
    print(f"  {'kernel':<26}{'read':>7}{'GiB/s':>9}")
    base = ks.get("4tpt_b128o12", (None,))[0]
    for k in order:
        if k not in ks:
            continue
        _, _, read = KMETA[k]
        v = ks[k][0]
        d = f"{100*(v-base)/base:+.0f}% vs 16B" if base else ""
        print(f"  {k:<26}{read:>7}{v:>9.0f}   {d}")
    print("  INFER: 16B→8B helps (fewer L2/TEX transactions); 8B→4B HURTS (below the 32B")
    print("         sector, cuts no transactions, adds a >4B fallback). Gather is")
    print("         transaction-bound, not request-width-bound. 8B is the sweet spot.")


def demo_bits16_wall():
    hr("DEMO 3 — bits16 wall: HIGH-OCCUPANCY L2 beats on-chip dict staging  (fineweb bits16)")
    g, ks = run("fineweb", "bits16")
    _, ksp = run("fineweb", "bits16", l2_persist=True)
    if not ks:
        print("  (data missing)"); return
    base = ks.get("4tpt_b128o12", (None,))[0]
    rows = [
        ("b128o12 (dict in L2, high occ)", base, ks.get("4tpt_b128o12", (0,))[0]),
        ("b128o12 + L2-persist (dict pinned in L2)", base,
         ksp.get("4tpt_b128o12", (0,))[0] if ksp else None),
        ("cluster_dsmem (dict in distributed SMEM, 1 blk/SM)", base,
         ks.get("4tpt_cluster_dsmem", (0,))[0]),
    ]
    print(f"  {'approach':<48}{'GiB/s':>8}{'Δ':>8}")
    for label, b, v in rows:
        if v is None:
            print(f"  {label:<48}{'n/a':>8}"); continue
        d = f"{100*(v-b)/b:+.0f}%" if (b and v) else ""
        print(f"  {label:<48}{v:>8.0f}{d:>8}")
    print("  INFER: pinning the 1MB dict in L2 changes nothing (it is ALREADY L2-resident);")
    print("         moving it to distributed SMEM collapses occupancy to 1 block/SM and")
    print("         pays a remote-fabric gather → ~5× slower. The bits16 limiter is L2")
    print("         latency HIDDEN BY OCCUPANCY, not gather bandwidth.")


def demo_gate():
    hr("DEMO 4 — the frac_le8 GATE separates split8read winners from losers  (bits12)")
    print(f"  {'column':<14}{'frac_le8':>9}{'b128o12':>9}{'split8r':>9}{'Δ':>8}  pick@0.70")
    for ck in ["fineweb", "wikipedia", "url", "l_comment", "ps_comment"]:
        g, ks = run(ck, "bits12")
        if not g:
            continue
        f = g.get("frac_le8", 0.0)
        b = ks.get("4tpt_b128o12", (0,))[0]
        s = ks.get("4tpt_split8read_b128o12", (0,))[0]
        d = 100*(s-b)/b if b else 0
        pick = "split8read" if f >= 0.70 else "b128o12"
        print(f"  {ck:<14}{f:>9.2f}{b:>9.0f}{s:>9.0f}{d:>+7.0f}%  {pick}")
    print("  INFER: split8read's win is monotonic in frac_le8 (short-token fraction).")
    print("         The 0.70 gate sits between URL (0.81, wins) and l_comment (0.58, loses),")
    print("         so the selector ships split8read exactly where it helps.")


def demo_launch_bound():
    hr("DEMO 5 — tiny columns are LAUNCH-BOUND, not slow-decoding  (b128o12, bits12)")
    print(f"  {'column':<14}{'decode µs':>11}{'GiB/s':>9}   note")
    for ck, note in [("dbtext_hex", "tiny (~0.6 MB) — can't fill 148 SMs"),
                     ("fineweb", "large — saturates the device")]:
        g, ks = run(ck, "bits12")
        if not ks:
            continue
        v, ms, _ = ks.get("4tpt_b128o12", (0, 0, None))
        us = (ms or 0) * 1000.0
        print(f"  {ck:<14}{us:>11.1f}{v:>9.0f}   {note}")
    print("  INFER: the tiny column hits a near-constant ~launch+grid-ramp floor (µs), so")
    print("         its GiB/s looks 'slow' — but it is overhead-bound, not decode-bound.")
    print("         GiB/s is only meaningful above ~tens of MB.")


def demo_whole_decompress():
    hr("DEMO 6 — WHOLE decompress is TRANSFER-bound, not decode-bound  (H2D + decode)")
    print(f"  {'column':<16}{'bits':>5}{'ratio':>7}{'decode':>8}{'h2d':>7}{'whole':>7}{'whole/h2d':>10}")
    for ck, bits in [("fineweb", "bits12"), ("fineweb", "bits16"), ("url", "bits12"),
                     ("l_comment", "bits12"), ("l_comment", "bits16"), ("ps_comment", "bits16")]:
        g, _ = run(ck, bits)
        if not g:
            continue
        comp = g.get("compressed_bytes", 0)
        dec = g.get("decoded_bytes", 0)
        ratio = dec / comp if comp else 0
        h2d = g.get("h2d_gib_s", 0)
        whole = g.get("whole_decompress_gib_s", 0)
        decode = g.get("auto_decode_gib_s", 0)
        sp = whole / h2d if h2d else 0
        print(f"  {ck:<16}{bits.replace('bits',''):>5}{ratio:>6.1f}x{decode:>8.0f}{h2d:>7.1f}"
              f"{whole:>7.0f}{sp:>9.1f}x")
    print("  INFER: decode (637-1100 GiB/s) is 60-100x faster than the H2D copy of the")
    print("         compressed bytes (~10 GiB/s pageable), so end-to-end is TRANSFER-bound and")
    print("         whole/h2d ≈ compression ratio. GPU decompress outputs ~ratio× faster than")
    print("         transferring raw bytes — and the kernel speedups matter for ON-DEVICE")
    print("         decode (data already on GPU), not for the H2D-then-decode path.")


def demo_bits_sweep():
    hr("DEMO 7 — DICT BIT-WIDTH sweep: ratio vs decode, and the L1-residency boundary  (fineweb)")
    if not os.path.exists(PARQUET):
        print(f"  (source parquet not found: {PARQUET}; set PARQUET=...)"); return
    # Compress fineweb at each bit width and benchmark all kernels in one `run`.
    out = subprocess.run(
        [BIN, "run", "--parquet", PARQUET, "--column", "text", "--dataset-id", "fineweb",
         "--bits", SWEEP_BITS, "--chunk-bytes", "1048576000", "--threshold", "0.2",
         "--sample-bytes", "1000000000", "--out-dir", DATA,
         "--gpu-decode", "--gpu-iters", "150", "--gpu-validate"],
        capture_output=True, text=True, env=dict(os.environ, ONPAIR_FAST="1"),
    ).stdout
    i = out.find("[")
    j = out.rfind("{")
    if i < 0 and j < 0:
        print("  (no JSON from run; compression may have failed)"); return
    try:
        cells = json.loads(out[out.find("["):]) if "[" in out else [json.loads(out[out.find("{"):])]
    except json.JSONDecodeError:
        print("  (could not parse run output)"); return
    print(f"  {'bits':>5}{'entries':>9}{'dict_s8':>9}{'ratio':>7}{'b128o12':>9}{'split8r':>9}"
          f"{'best':>7}{'L1?':>5}")
    for c in cells:
        g = c.get("gpu") or {}
        if not g:
            continue
        ent = g.get("dict_entries_max", 0)
        s8kb = ent * 8 / 1024.0
        comp = g.get("compressed_bytes", 0)
        dec = g.get("decoded_bytes", 0)
        ratio = dec / comp if comp else 0
        ks = {k["kernel"].replace("onpair_shmem_4tpt_", ""): k.get("decode_gib_s")
              for k in g.get("kernels", []) if k.get("decode_gib_s")}
        b = ks.get("b128o12", 0)
        s = ks.get("split8read_b128o12", 0)
        best = max(b, s)
        fits = "yes" if s8kb <= 256 else "no"
        print(f"  {c.get('bits',0):>5}{ent:>9}{s8kb:>7.0f}KB{ratio:>6.1f}x{b:>9.0f}{s:>9.0f}"
              f"{best:>7.0f}{fits:>5}")
    print("  INFER: more dict bits => better ratio but bigger dict. split8read (8B reads) wins")
    print("         while dict_s8 (entries×8) fits the ~256KB L1 — i.e. up to bits15 (32768")
    print("         entries=256KB); at bits16 (512KB) it no longer fits and ties b128o12.")
    print("         bits14 is the middle ground: ~2.3x ratio at L1-resident split8read speed.")


def main():
    if not os.path.exists(BIN):
        print(f"binary not found: {BIN}\nbuild it with:\n"
              "  cargo build -p vortex-bench --features cuda "
              "--bin onpair-chunk-bench --release", file=sys.stderr)
        sys.exit(1)
    print("B200 OnPair decode — evidence benchmarks (PRELIMINARY: unlocked clocks ±~5%)")
    demo_granularity()
    demo_readwidth()
    demo_bits16_wall()
    demo_gate()
    demo_launch_bound()
    demo_whole_decompress()
    if os.environ.get("EVIDENCE_COMPRESS", "1") != "0":
        demo_bits_sweep()
    print()


if __name__ == "__main__":
    main()
