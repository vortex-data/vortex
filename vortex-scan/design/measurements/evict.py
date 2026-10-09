#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors
"""Evict files from the macOS unified buffer cache without root, and report page residency.

For each file: mmap it read-only/shared, msync(MS_INVALIDATE) (the same trick vmtouch -e uses on
non-Linux), and report page residency via mincore() before and after. Use --check to only report,
--per-file to print one line per file that has resident pages.
"""

import ctypes
import ctypes.util
import os
import sys

libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
libc.mmap.restype = ctypes.c_void_p
libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_longlong]
libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
libc.msync.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_char_p]

PROT_READ = 0x1
MAP_SHARED = 0x1
MS_INVALIDATE = 0x2
MAP_FAILED = ctypes.c_void_p(-1).value
PAGE = os.sysconf("SC_PAGE_SIZE")
_INCORE = bytes(b & 1 for b in range(256))


def resident(addr: int, size: int) -> tuple[int, int]:
    npages = (size + PAGE - 1) // PAGE
    vec = ctypes.create_string_buffer(npages)
    if libc.mincore(addr, size, vec) != 0:
        raise OSError(ctypes.get_errno(), "mincore")
    return vec.raw.translate(_INCORE).count(b"\x01"), npages


def process(path: str, evict: bool) -> tuple[int, int, int]:
    """Return (resident_pages_before, resident_pages_after, total_pages)."""
    size = os.path.getsize(path)
    if size == 0:
        return 0, 0, 0
    fd = os.open(path, os.O_RDONLY)
    try:
        addr = libc.mmap(None, size, PROT_READ, MAP_SHARED, fd, 0)
        if addr == MAP_FAILED or addr is None:
            raise OSError(ctypes.get_errno(), f"mmap {path}")
        try:
            before, npages = resident(addr, size)
            after = before
            if evict and before:
                if libc.msync(addr, size, MS_INVALIDATE) != 0:
                    raise OSError(ctypes.get_errno(), f"msync {path}")
                after, _ = resident(addr, size)
            return before, after, npages
        finally:
            libc.munmap(addr, size)
    finally:
        os.close(fd)


def list_files(paths: list[str]) -> list[str]:
    files = []
    for a in paths:
        if os.path.isdir(a):
            for root, _, names in os.walk(a):
                files.extend(os.path.join(root, n) for n in names)
        else:
            files.append(a)
    return sorted(files)


def sweep(paths: list[str], evict: bool) -> tuple[int, int, int, dict[str, tuple[int, int, int]]]:
    """Return (total_before, total_after, total_pages, {file: (before, after, pages)})."""
    per = {}
    tb = ta = tp = 0
    for f in list_files(paths):
        b, a, n = process(f, evict)
        per[f] = (b, a, n)
        tb += b
        ta += a
        tp += n
    return tb, ta, tp, per


def main():
    args = sys.argv[1:]
    evict = True
    per_file = False
    while args and args[0].startswith("--"):
        if args[0] == "--check":
            evict = False
        elif args[0] == "--per-file":
            per_file = True
        args = args[1:]
    tb, ta, tp, per = sweep(args, evict)
    if per_file:
        for f, (b, a, n) in per.items():
            if b:
                print(f"  {os.path.basename(f)}: {b * PAGE / 1e6:.1f} MB resident of {n * PAGE / 1e6:.1f} MB")

    def pct(x: int) -> float:
        return 100.0 * x / tp if tp else 0.0

    print(
        f"files={len(per)} pages={tp} resident_before={tb} ({pct(tb):.1f}%) "
        f"resident_after={ta} ({pct(ta):.1f}%) mode={'evict' if evict else 'check'}"
    )


if __name__ == "__main__":
    main()
