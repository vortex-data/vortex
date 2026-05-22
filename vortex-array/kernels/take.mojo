# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

"""Mojo SIMD take/filter gather kernels for Vortex primitive arrays.

Provides 16 take exports (`vortex_take_{1,2,4,8}byte_{u8,u16,u32,u64}idx`) and
4 filter exports (`vortex_filter_{1,2,4,8}byte`).  Each uses a 4x-unrolled
SIMD gather loop sized to the value width.
"""

from std.memory import UnsafePointer

# SIMD lane counts per value width (256-bit register)
alias W1 = 32  # 1-byte values: 256 / 8
alias W2 = 16  # 2-byte values: 256 / 16
alias W4 = 8   # 4-byte values: 256 / 32
alias W8 = 4   # 8-byte values: 256 / 64


# ---------------------------------------------------------------------------
# Generic take (gather by typed index)
# ---------------------------------------------------------------------------

fn _take[VT: DType, IT: DType, W: Int](src_addr: Int, idx_addr: Int, dst_addr: Int, n: Int):
    """Gather `n` values from `src` at positions given by `idx` into `dst`.

    Uses a 4x-unrolled inner loop; each unroll gathers W lanes.
    """
    # Anchor variable so we can use type_of to get the right UnsafePointer type.
    var _anchor_v: Scalar[VT] = 0
    comptime VP = type_of(UnsafePointer(to=_anchor_v))
    var src = VP(unsafe_from_address=src_addr)
    var dst = VP(unsafe_from_address=dst_addr)

    var _anchor_i: Scalar[IT] = 0
    comptime IP = type_of(UnsafePointer(to=_anchor_i))
    var idx = IP(unsafe_from_address=idx_addr)

    var i = 0
    var n4 = (n // (4 * W)) * (4 * W)

    # 4x unrolled SIMD gather
    while i < n4:
        var ix0 = (idx + i).load[width=W]().cast[DType.uint64]()
        var ix1 = (idx + i + W).load[width=W]().cast[DType.uint64]()
        var ix2 = (idx + i + 2 * W).load[width=W]().cast[DType.uint64]()
        var ix3 = (idx + i + 3 * W).load[width=W]().cast[DType.uint64]()
        (dst + i).store(src.gather(ix0))
        (dst + i + W).store(src.gather(ix1))
        (dst + i + 2 * W).store(src.gather(ix2))
        (dst + i + 3 * W).store(src.gather(ix3))
        i += 4 * W

    # Scalar remainder
    while i < n:
        var offset = (idx + i).load().cast[DType.uint64]()
        (dst + i).store((src + Int(offset)).load())
        i += 1


# ---------------------------------------------------------------------------
# Generic filter (gather with u64/usize indices)
# ---------------------------------------------------------------------------

fn _filter[VT: DType, W: Int](src_addr: Int, idx_addr: Int, dst_addr: Int, n: Int):
    """Filter-gather `n` values from `src` at u64 positions `idx` into `dst`."""
    var _anchor_v: Scalar[VT] = 0
    comptime VP = type_of(UnsafePointer(to=_anchor_v))
    var src = VP(unsafe_from_address=src_addr)
    var dst = VP(unsafe_from_address=dst_addr)

    var _anchor_u: UInt64 = 0
    comptime UP = type_of(UnsafePointer(to=_anchor_u))
    var idx = UP(unsafe_from_address=idx_addr)

    var i = 0
    var n4 = (n // (4 * W)) * (4 * W)

    # 4x unrolled SIMD gather
    while i < n4:
        var ix0 = (idx + i).load[width=W]()
        var ix1 = (idx + i + W).load[width=W]()
        var ix2 = (idx + i + 2 * W).load[width=W]()
        var ix3 = (idx + i + 3 * W).load[width=W]()
        (dst + i).store(src.gather(ix0))
        (dst + i + W).store(src.gather(ix1))
        (dst + i + 2 * W).store(src.gather(ix2))
        (dst + i + 3 * W).store(src.gather(ix3))
        i += 4 * W

    # Scalar remainder
    while i < n:
        var offset = (idx + i).load()
        (dst + i).store((src + Int(offset)).load())
        i += 1


# ===========================================================================
# Take exports: 16 combinations of {1,2,4,8}-byte values x {u8,u16,u32,u64} indices
# ===========================================================================

# --- 1-byte values ---
@export("vortex_take_1byte_u8idx")
fn take_1byte_u8idx(src: Int, idx: Int, dst: Int, n: Int):
    _take[DType.uint8, DType.uint8, W1](src, idx, dst, n)

@export("vortex_take_1byte_u16idx")
fn take_1byte_u16idx(src: Int, idx: Int, dst: Int, n: Int):
    _take[DType.uint8, DType.uint16, W1](src, idx, dst, n)

@export("vortex_take_1byte_u32idx")
fn take_1byte_u32idx(src: Int, idx: Int, dst: Int, n: Int):
    _take[DType.uint8, DType.uint32, W1](src, idx, dst, n)

@export("vortex_take_1byte_u64idx")
fn take_1byte_u64idx(src: Int, idx: Int, dst: Int, n: Int):
    _take[DType.uint8, DType.uint64, W1](src, idx, dst, n)

# --- 2-byte values ---
@export("vortex_take_2byte_u8idx")
fn take_2byte_u8idx(src: Int, idx: Int, dst: Int, n: Int):
    _take[DType.uint16, DType.uint8, W2](src, idx, dst, n)

@export("vortex_take_2byte_u16idx")
fn take_2byte_u16idx(src: Int, idx: Int, dst: Int, n: Int):
    _take[DType.uint16, DType.uint16, W2](src, idx, dst, n)

@export("vortex_take_2byte_u32idx")
fn take_2byte_u32idx(src: Int, idx: Int, dst: Int, n: Int):
    _take[DType.uint16, DType.uint32, W2](src, idx, dst, n)

@export("vortex_take_2byte_u64idx")
fn take_2byte_u64idx(src: Int, idx: Int, dst: Int, n: Int):
    _take[DType.uint16, DType.uint64, W2](src, idx, dst, n)

# --- 4-byte values ---
@export("vortex_take_4byte_u8idx")
fn take_4byte_u8idx(src: Int, idx: Int, dst: Int, n: Int):
    _take[DType.uint32, DType.uint8, W4](src, idx, dst, n)

@export("vortex_take_4byte_u16idx")
fn take_4byte_u16idx(src: Int, idx: Int, dst: Int, n: Int):
    _take[DType.uint32, DType.uint16, W4](src, idx, dst, n)

@export("vortex_take_4byte_u32idx")
fn take_4byte_u32idx(src: Int, idx: Int, dst: Int, n: Int):
    _take[DType.uint32, DType.uint32, W4](src, idx, dst, n)

@export("vortex_take_4byte_u64idx")
fn take_4byte_u64idx(src: Int, idx: Int, dst: Int, n: Int):
    _take[DType.uint32, DType.uint64, W4](src, idx, dst, n)

# --- 8-byte values ---
@export("vortex_take_8byte_u8idx")
fn take_8byte_u8idx(src: Int, idx: Int, dst: Int, n: Int):
    _take[DType.uint64, DType.uint8, W8](src, idx, dst, n)

@export("vortex_take_8byte_u16idx")
fn take_8byte_u16idx(src: Int, idx: Int, dst: Int, n: Int):
    _take[DType.uint64, DType.uint16, W8](src, idx, dst, n)

@export("vortex_take_8byte_u32idx")
fn take_8byte_u32idx(src: Int, idx: Int, dst: Int, n: Int):
    _take[DType.uint64, DType.uint32, W8](src, idx, dst, n)

@export("vortex_take_8byte_u64idx")
fn take_8byte_u64idx(src: Int, idx: Int, dst: Int, n: Int):
    _take[DType.uint64, DType.uint64, W8](src, idx, dst, n)

# ===========================================================================
# Filter exports: 4 combinations of {1,2,4,8}-byte values with u64 indices
# ===========================================================================

@export("vortex_filter_1byte")
fn filter_1byte(src: Int, idx: Int, dst: Int, n: Int):
    _filter[DType.uint8, W1](src, idx, dst, n)

@export("vortex_filter_2byte")
fn filter_2byte(src: Int, idx: Int, dst: Int, n: Int):
    _filter[DType.uint16, W2](src, idx, dst, n)

@export("vortex_filter_4byte")
fn filter_4byte(src: Int, idx: Int, dst: Int, n: Int):
    _filter[DType.uint32, W4](src, idx, dst, n)

@export("vortex_filter_8byte")
fn filter_8byte(src: Int, idx: Int, dst: Int, n: Int):
    _filter[DType.uint64, W8](src, idx, dst, n)
