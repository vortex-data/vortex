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
use vortex_buffer::BufferAllocatorRef;
use vortex_buffer::BufferMut;

use super::super::FixedWidthTakeValue;
use super::super::take_values_fallback;
use crate::dtype::PType;
use crate::dtype::UnsignedPType;

// SAFETY: u8 has no padding or uninitialized bytes.
unsafe impl FixedWidthTakeValue for u8 {
    fn take<I: UnsignedPType>(
        values: &[Self],
        indices: &[I],
        allocator: &BufferAllocatorRef,
    ) -> Buffer<Self> {
        take(values, indices, allocator)
    }
}

// SAFETY: i8 has no padding or uninitialized bytes.
unsafe impl FixedWidthTakeValue for i8 {
    fn take<I: UnsignedPType>(
        values: &[Self],
        indices: &[I],
        allocator: &BufferAllocatorRef,
    ) -> Buffer<Self> {
        take(values, indices, allocator)
    }
}

// SAFETY: Byte arrays have no padding and every byte is initialized.
unsafe impl<const N: usize> FixedWidthTakeValue for [u8; N] {
    fn take<I: UnsignedPType>(
        values: &[Self],
        indices: &[I],
        allocator: &BufferAllocatorRef,
    ) -> Buffer<Self> {
        take(values, indices, allocator)
    }
}

fn take<T: FixedWidthTakeValue, I: UnsignedPType>(
    values: &[T],
    indices: &[I],
    allocator: &BufferAllocatorRef,
) -> Buffer<T> {
    if I::PTYPE != PType::U8
        || values.is_empty()
        || values.len() > 16
        || size_of::<T>() != 1
        || indices.len() < 64
    {
        return take_values_fallback(values, indices, allocator);
    }

    // SAFETY: the sealed index type is u8, as checked above.
    let indices: &[u8] =
        unsafe { std::slice::from_raw_parts(indices.as_ptr().cast(), indices.len()) };

    let mut table = [values[0]; 16];
    table[..values.len()].copy_from_slice(values);

    let mut output = BufferMut::<T>::with_capacity_in(indices.len(), allocator.clone());
    let spare = output.spare_capacity_mut();
    let output_ptr = spare.as_mut_ptr().cast::<u8>();
    // SAFETY: AArch64 always provides NEON. T is one byte with no uninitialized bytes, the table
    // contains 16 values, and the output has capacity for every index.
    let (offset, max_code) = unsafe { take_vectors(&table, indices, output_ptr) };

    for offset in offset..indices.len() {
        let code = usize::from(indices[offset]);
        assert!(
            code < values.len(),
            "take index {code} out of bounds for length {}",
            values.len()
        );
        spare[offset].write(values[code]);
    }
    assert!(
        usize::from(max_code) < values.len(),
        "take index {max_code} out of bounds for length {}",
        values.len()
    );
    // SAFETY: the vector loop and scalar remainder initialized every output value.
    unsafe { output.set_len(indices.len()) };
    output.freeze()
}

unsafe fn take_vectors<T: FixedWidthTakeValue>(
    table: &[T; 16],
    indices: &[u8],
    output: *mut u8,
) -> (usize, u8) {
    let table = unsafe { vld1q_u8(table.as_ptr().cast::<u8>()) };
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
