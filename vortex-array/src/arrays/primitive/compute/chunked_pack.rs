// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Shared SIMD-friendly bit-packing helper used by the primitive compare and between
//! kernels.
//!
//! `BitBuffer::collect_bool` already chunks 64 bits per `u64` write, but its closure-
//! based interface defeats auto-vectorization for non-trivial predicates: at 100k
//! u16 elements a `cmp + collect_bool` runs ~2× slower than this byte-chunked variant
//! because the compiler can't see through the closure to unroll + vectorize the 64-wide
//! predicate evaluation. With 8 cmps per output byte the inner loop is small enough to
//! be unrolled fully and the predicate vectorizes cleanly.

use vortex_buffer::BitBuffer;
use vortex_buffer::ByteBufferMut;

/// Build a `BitBuffer` of `slice.len()` bits where bit `i` is `pred(slice[i])`.
///
/// Packs 8 elements into one output byte per iteration. The inner 8-step loop is
/// short enough to inline and vectorize, giving ~2× over `collect_bool` for typical
/// per-element predicates.
#[inline]
pub(super) fn chunked_pack<T: Copy, F: Fn(T) -> bool>(slice: &[T], pred: F) -> BitBuffer {
    let len = slice.len();
    let bytes_len = len.div_ceil(8);
    let mut bytes = ByteBufferMut::zeroed(bytes_len);
    let dst = bytes.as_mut_slice();

    let full = len / 8;
    for chunk_idx in 0..full {
        let base = chunk_idx * 8;
        let mut b = 0u8;
        // The 8-step inner loop unrolls fully and lets the compiler vectorize `pred`.
        for j in 0..8 {
            // SAFETY: base + j < full*8 <= len.
            let v = unsafe { *slice.get_unchecked(base + j) };
            b |= u8::from(pred(v)) << j;
        }
        // SAFETY: chunk_idx < full <= bytes_len.
        unsafe { *dst.get_unchecked_mut(chunk_idx) = b };
    }

    let tail = full * 8;
    if tail < len {
        let mut b = 0u8;
        for j in 0..(len - tail) {
            // SAFETY: tail + j < len.
            let v = unsafe { *slice.get_unchecked(tail + j) };
            b |= u8::from(pred(v)) << j;
        }
        // SAFETY: full < bytes_len when len % 8 != 0.
        unsafe { *dst.get_unchecked_mut(full) = b };
    }

    BitBuffer::new(bytes.freeze(), len)
}
