# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Mojo SIMD run-end decode kernels.

Provides 8 exports: 4 with u32 ends and 4 with u64 ends, for {1,2,4,8}-byte
value widths.  Each uses a 4x-unrolled SIMD broadcast fill loop.
"""

from std.memory import UnsafePointer

# SIMD lane counts per value width (256-bit register)
alias W1 = 32  # 1-byte values
alias W2 = 16  # 2-byte values
alias W4 = 8   # 4-byte values
alias W8 = 4   # 8-byte values


fn _runend_decode[VT: DType, ET: DType, W: Int](
    ends_addr: Int,
    vals_addr: Int,
    dst_addr: Int,
    n_runs: Int,
    out_len: Int,
):
    """Decode run-end encoded data by broadcast-filling each run.

    `ends` contains `n_runs` monotonically increasing end positions (exclusive).
    `vals` contains `n_runs` values, one per run.
    Fills `dst` with `out_len` decoded elements.
    """
    var _anchor_v: Scalar[VT] = 0
    comptime VP = type_of(UnsafePointer(to=_anchor_v))
    var vals = VP(unsafe_from_address=vals_addr)
    var dst = VP(unsafe_from_address=dst_addr)

    var _anchor_e: Scalar[ET] = 0
    comptime EP = type_of(UnsafePointer(to=_anchor_e))
    var ends = EP(unsafe_from_address=ends_addr)

    var pos = 0
    for run in range(n_runs):
        var end = Int((ends + run).load())
        if end > out_len:
            end = out_len
        var val = (vals + run).load()
        var splat = SIMD[VT, W](val)

        # Number of elements to fill for this run
        var run_len = end - pos
        var filled = 0
        var run4 = (run_len // (4 * W)) * (4 * W)

        # 4x unrolled SIMD broadcast fill
        while filled < run4:
            (dst + pos + filled).store(splat)
            (dst + pos + filled + W).store(splat)
            (dst + pos + filled + 2 * W).store(splat)
            (dst + pos + filled + 3 * W).store(splat)
            filled += 4 * W

        # Scalar remainder
        while filled < run_len:
            (dst + pos + filled).store(val)
            filled += 1

        pos = end


# ===========================================================================
# Exports with u32 ends
# ===========================================================================

@export("vortex_runend_decode_1byte")
fn runend_decode_1byte(ends: Int, vals: Int, dst: Int, n_runs: Int, out_len: Int):
    _runend_decode[DType.uint8, DType.uint32, W1](ends, vals, dst, n_runs, out_len)

@export("vortex_runend_decode_2byte")
fn runend_decode_2byte(ends: Int, vals: Int, dst: Int, n_runs: Int, out_len: Int):
    _runend_decode[DType.uint16, DType.uint32, W2](ends, vals, dst, n_runs, out_len)

@export("vortex_runend_decode_4byte")
fn runend_decode_4byte(ends: Int, vals: Int, dst: Int, n_runs: Int, out_len: Int):
    _runend_decode[DType.uint32, DType.uint32, W4](ends, vals, dst, n_runs, out_len)

@export("vortex_runend_decode_8byte")
fn runend_decode_8byte(ends: Int, vals: Int, dst: Int, n_runs: Int, out_len: Int):
    _runend_decode[DType.uint64, DType.uint32, W8](ends, vals, dst, n_runs, out_len)

# ===========================================================================
# Exports with u64 ends
# ===========================================================================

@export("vortex_runend_decode_1byte_u64ends")
fn runend_decode_1byte_u64ends(ends: Int, vals: Int, dst: Int, n_runs: Int, out_len: Int):
    _runend_decode[DType.uint8, DType.uint64, W1](ends, vals, dst, n_runs, out_len)

@export("vortex_runend_decode_2byte_u64ends")
fn runend_decode_2byte_u64ends(ends: Int, vals: Int, dst: Int, n_runs: Int, out_len: Int):
    _runend_decode[DType.uint16, DType.uint64, W2](ends, vals, dst, n_runs, out_len)

@export("vortex_runend_decode_4byte_u64ends")
fn runend_decode_4byte_u64ends(ends: Int, vals: Int, dst: Int, n_runs: Int, out_len: Int):
    _runend_decode[DType.uint32, DType.uint64, W4](ends, vals, dst, n_runs, out_len)

@export("vortex_runend_decode_8byte_u64ends")
fn runend_decode_8byte_u64ends(ends: Int, vals: Int, dst: Int, n_runs: Int, out_len: Int):
    _runend_decode[DType.uint64, DType.uint64, W8](ends, vals, dst, n_runs, out_len)
