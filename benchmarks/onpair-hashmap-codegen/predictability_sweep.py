#!/usr/bin/env python3
"""Sweep long-map hit rate over an OPHASH01 trace and time each binary.

The GCC-versus-LLVM gap on this workload only appears when the fingerprint and
key-compare branches are unpredictable, so timing a single trace hides the
mechanism. This rewrites the long-probe section of an existing trace to a chosen
hit rate, leaving dictionary entries, short entries, and short probes untouched.

usage: predictability_sweep.py TRACE BINARY [BINARY ...]
"""

import os
import random
import re
import struct
import subprocess
import sys
import tempfile

HIT_PERCENTS = [0, 10, 25, 50, 75, 90, 100]
SEED = 7


def rewrite(trace, hit_pct, output):
    blob = open(trace, "rb").read()
    if blob[:8] != b"OPHASH01":
        raise SystemExit("not an OPHASH01 trace")
    short_entries, long_entries, short_probes, long_probes = struct.unpack_from("<QQQQ", blob, 8)

    long_entry_offset = 40 + short_entries * 11
    keys = [
        struct.unpack_from("<Q", blob, long_entry_offset + index * 10)[0]
        for index in range(long_entries)
    ]
    long_probe_offset = long_entry_offset + long_entries * 10 + short_probes * 9

    rng = random.Random(SEED)
    out = bytearray(blob[:long_probe_offset])
    for _ in range(long_probes):
        if rng.randrange(100) < hit_pct:
            key = keys[rng.randrange(len(keys))]
        else:
            # Set the top bit: OnPair long keys are 8-byte text prefixes, so this
            # is guaranteed absent rather than merely unlikely.
            key = rng.getrandbits(64) | (1 << 63)
        out += struct.pack("<Q", key)
    open(output, "wb").write(bytes(out))
    return long_probes


def measure(binary, trace):
    environment = dict(os.environ, HASH_WARMUPS="2", HASH_ITERATIONS="11")
    output = subprocess.run(
        [binary, trace], capture_output=True, text=True, check=True, env=environment
    ).stdout
    found = re.search(r"long_ms=([0-9.]+)", output)
    if not found:
        raise SystemExit(f"no long_ms in output of {binary}")
    return float(found.group(1))


def main():
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    trace, binaries = sys.argv[1], sys.argv[2:]

    names = [os.path.basename(binary) for binary in binaries]
    print("hit_pct," + ",".join(f"{name}_ns_per_probe" for name in names))
    with tempfile.TemporaryDirectory() as directory:
        for hit_pct in HIT_PERCENTS:
            path = os.path.join(directory, f"p{hit_pct}.oph")
            probes = rewrite(trace, hit_pct, path)
            timings = [measure(binary, path) * 1e6 / probes for binary in binaries]
            print(f"{hit_pct}," + ",".join(f"{value:.3f}" for value in timings))


if __name__ == "__main__":
    main()
