// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! NEON byte-table take for `u8` codes and at most 16 one-byte values.

use std::arch::aarch64::uint8x16_t;
use std::arch::aarch64::vdupq_n_u8;
use std::arch::aarch64::vld1q_u8;
use std::arch::aarch64::vmaxq_u8;
use std::arch::aarch64::vmaxvq_u8;
use std::arch::aarch64::vqtbl1q_u8;
use std::arch::aarch64::vst1q_u8;

use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;

use super::FixedWidthTakeValue;

pub(crate) fn take<T: FixedWidthTakeValue>(values: &[T], indices: &[u8]) -> Option<Buffer<T>> {
    if values.is_empty() || values.len() > 16 || size_of::<T>() != 1 || indices.len() < 64 {
        return None;
    }

    let mut table = [0u8; 16];
    // SAFETY: one-byte values have the same representation as bytes, and the length is <= 16.
    unsafe {
        std::ptr::copy_nonoverlapping(
            values.as_ptr().cast::<u8>(),
            table.as_mut_ptr(),
            values.len(),
        );
    }

    let mut output = BufferMut::<T>::with_capacity(indices.len());
    let output_ptr = output.spare_capacity_mut().as_mut_ptr().cast::<u8>();
    // SAFETY: AArch64 always provides NEON. Both pointers advance only by complete vectors.
    let (offset, max_code) = unsafe { take_vectors(&table, indices, output_ptr) };

    for offset in offset..indices.len() {
        let code = usize::from(indices[offset]);
        assert!(
            code < values.len(),
            "take index {code} out of bounds for length {}",
            values.len()
        );
        // SAFETY: the code was checked and this reserved output position is uninitialized.
        unsafe {
            output_ptr
                .add(offset)
                .write(*values.as_ptr().add(code).cast::<u8>())
        };
    }
    assert!(
        usize::from(max_code) < values.len(),
        "take index {max_code} out of bounds for length {}",
        values.len()
    );
    // SAFETY: the vector loop and scalar remainder initialized every output value.
    unsafe { output.set_len(indices.len()) };
    Some(output.freeze())
}

unsafe fn take_vectors(table: &[u8; 16], indices: &[u8], output: *mut u8) -> (usize, u8) {
    let table = unsafe { vld1q_u8(table.as_ptr()) };
    let mut max_codes: uint8x16_t = unsafe { vdupq_n_u8(0) };
    let mut offset = 0;
    while offset + 16 <= indices.len() {
        let codes = unsafe { vld1q_u8(indices.as_ptr().add(offset)) };
        max_codes = unsafe { vmaxq_u8(max_codes, codes) };
        unsafe { vst1q_u8(output.add(offset), vqtbl1q_u8(table, codes)) };
        offset += 16;
    }
    (offset, unsafe { vmaxvq_u8(max_codes) })
}
