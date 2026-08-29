#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PATHS = (
    "vortex-array/src/arrays/filter/execute",
    "vortex-array/src/arrays/fixed_width/filter.rs",
    "vortex-array/src/arrays/fixed_width/take",
    "vortex-array/src/arrays/interleave/execute",
    "vortex-array/src/patches.rs",
    "vortex-array/src/scalar_fn/fns",
)
TYPES = r"(?:ByteBuffer|Buffer|BitBuffer|ByteBufferMut|BufferMut|BitBufferMut)"
METHODS = "|".join(
    (
        "with_capacity",
        "with_capacity_aligned",
        "with_capacity_preferred_aligned",
        "zeroed",
        "zeroed_aligned",
        "empty",
        "empty_aligned",
        "copy_from",
        "copy_from_aligned",
        "full",
        "new_set",
        "new_unset",
        "collect_bool",
        "collect_bool_multiversioned",
        "from_trusted_len_iter",
        "try_from_trusted_len_iter",
    )
)
STATIC_ALLOCATION = re.compile(rf"\b{TYPES}::(?:{METHODS})\s*\(")


def rust_files(path: Path):
    files = path.rglob("*.rs") if path.is_dir() else (path,)
    return (file for file in files if file.name != "tests.rs" and "tests" not in file.relative_to(ROOT).parts)


failures = []
for relative in PATHS:
    for file in rust_files(ROOT / relative):
        for number, line in enumerate(file.read_text().splitlines(), 1):
            if STATIC_ALLOCATION.search(line):
                failures.append(f"{file.relative_to(ROOT)}:{number}: {line.strip()}")

if failures:
    print("Engine buffers must use an allocator-aware constructor ending in `_in`.")
    print("\n".join(failures))
    raise SystemExit(1)
