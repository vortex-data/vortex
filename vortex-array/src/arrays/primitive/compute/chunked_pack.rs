// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! SIMD-friendly bit-packing helper shared by the primitive compare and between kernels.

use vortex_buffer::BitBuffer;
use vortex_buffer::ByteBufferMut;

/// Build a `BitBuffer` where bit `i` is `pred(slice[i])`. Packs 8 elements per output byte;
/// the 8-step inner loop unrolls and vectorises, ~2× faster than `BitBuffer::collect_bool`
/// for typical per-element predicates because the closure body stays visible to the
/// optimiser.
#[inline]
pub(super) fn chunked_pack<T: Copy, F: Fn(T) -> bool>(slice: &[T], pred: F) -> BitBuffer {
    let len = slice.len();
    let bytes_len = len.div_ceil(8);
    let mut bytes = ByteBufferMut::zeroed(bytes_len);
    let dst = bytes.as_mut_slice();

    let full = len / 8;
    for chunk_idx in 0..full {
        let base = chunk_idx * 8;
        let mut byte = 0u8;
        for j in 0..8 {
            // SAFETY: base + j < full*8 <= len.
            let v = unsafe { *slice.get_unchecked(base + j) };
            byte |= u8::from(pred(v)) << j;
        }
        // SAFETY: chunk_idx < full <= bytes_len.
        unsafe { *dst.get_unchecked_mut(chunk_idx) = byte };
    }

    let tail = full * 8;
    if tail < len {
        let mut byte = 0u8;
        for j in 0..(len - tail) {
            // SAFETY: tail + j < len.
            let v = unsafe { *slice.get_unchecked(tail + j) };
            byte |= u8::from(pred(v)) << j;
        }
        // SAFETY: full < bytes_len when len % 8 != 0.
        unsafe { *dst.get_unchecked_mut(full) = byte };
    }

    BitBuffer::new(bytes.freeze(), len)
}
