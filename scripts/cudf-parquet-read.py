#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Times a full GPU Parquet read with cuDF.

`cudf.read_parquet` performs the entire read on the device: page header decode,
codec decompression, dictionary/RLE/plain decoding and column assembly. That makes it
the like-for-like opponent for the Vortex GPU backend, which also decodes all the way
to canonical arrays on device.

Timing excludes interpreter start, `import cudf`, CUDA context creation and any JIT
warm-up, all of which are paid once per process and are not part of a read. A warm-up
read runs first for exactly that reason.

Emits one JSON object on stdout so the benchmark can parse it.
"""

import argparse
import json
import sys
import time
from datetime import date


def synchronize() -> None:
    """Block until queued device work finishes.

    `cudf.read_parquet` returns a materialized DataFrame, but synchronizing explicitly
    keeps the measurement honest if that ever stops being true.
    """
    try:
        import cupy

        cupy.cuda.runtime.deviceSynchronize()
    except ImportError:
        pass


def normalize(frame):
    """Collapses representation differences that are not value differences.

    A Parquet DATE column comes back from pyarrow as a column of `datetime.date`
    objects but from cuDF as `datetime64[s]`. Those hold the same instants, yet
    `check_dtype=False` does not bridge them because one side is `object`, so the
    comparison reports every row as different. Coercing both sides to datetime64
    compares the dates themselves.
    """
    import pandas as pd

    for name in frame.columns:
        column = frame[name]
        if column.dtype == object and len(column) and isinstance(column.iloc[0], date):
            frame[name] = pd.to_datetime(column)
    return frame


def verify(path: str, frame) -> None:
    """Fails unless the GPU read matches a CPU Parquet read of the same file."""
    import pandas as pd
    from pandas.testing import assert_frame_equal

    expected = normalize(pd.read_parquet(path))
    actual = normalize(frame.to_pandas())

    # cuDF and pyarrow can land on different-but-equivalent dtypes (nullable vs numpy
    # backed, for instance), so compare values and leave dtype policy out of it.
    assert_frame_equal(actual, expected, check_dtype=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="Parquet file to read")
    parser.add_argument(
        "--iterations", type=int, default=1, help="timed reads to perform"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="cross-check the GPU read against a CPU Parquet read",
    )
    args = parser.parse_args()

    import cudf

    # Warm-up: pays CUDA context creation and any first-call JIT so they stay out of
    # the timed reads below.
    warmup = cudf.read_parquet(args.path)
    synchronize()

    if args.verify:
        verify(args.path, warmup)

    rows, columns = warmup.shape
    del warmup

    runs_ns = []
    for _ in range(max(args.iterations, 1)):
        start = time.perf_counter_ns()
        frame = cudf.read_parquet(args.path)
        synchronize()
        runs_ns.append(time.perf_counter_ns() - start)
        del frame

    json.dump(
        {
            "min_ns": min(runs_ns),
            "runs_ns": runs_ns,
            "rows": int(rows),
            "columns": int(columns),
            "verified": bool(args.verify),
        },
        sys.stdout,
    )
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
