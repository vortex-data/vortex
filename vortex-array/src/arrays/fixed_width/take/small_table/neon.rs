// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! NEON byte-table take for `u8` codes and at most 64 one-byte values.

use std::arch::aarch64::uint8x16_t;
use std::arch::aarch64::uint8x16x2_t;
use std::arch::aarch64::uint8x16x4_t;
use std::arch::aarch64::vdupq_n_u8;
use std::arch::aarch64::vld1q_u8;
use std::arch::aarch64::vmaxq_u8;
use std::arch::aarch64::vmaxvq_u8;
use std::arch::aarch64::vqtbl1q_u8;
use std::arch::aarch64::vqtbl2q_u8;
use std::arch::aarch64::vqtbl4q_u8;
use std::arch::aarch64::vst1q_u8;

use vortex_buffer::Buffer;
use vortex_buffer::BufferAllocatorRef;
use vortex_buffer::BufferMut;

use super::super::FixedWidthTakeValue;
use crate::dtype::PType;
use crate::dtype::UnsignedPType;

pub(crate) fn try_take<T: FixedWidthTakeValue, I: UnsignedPType>(
    values: &[T],
    indices: &[I],
    allocator: &BufferAllocatorRef,
) -> Option<Buffer<T>> {
    if I::PTYPE != PType::U8
        || values.is_empty()
        || values.len() > 64
        || size_of::<T>() != 1
        || indices.len() < 64
    {
        return None;
    }

    // SAFETY: the sealed index type is u8, as checked above.
    let indices: &[u8] =
        unsafe { std::slice::from_raw_parts(indices.as_ptr().cast(), indices.len()) };

    let mut table = [values[0]; 64];
    table[..values.len()].copy_from_slice(values);

    let mut output = BufferMut::<T>::with_capacity_in(indices.len(), allocator.clone());
    let spare = output.spare_capacity_mut();
    let output_ptr = spare.as_mut_ptr().cast::<u8>();
    // SAFETY: AArch64 always provides NEON. T is one byte with no uninitialized bytes, the table
    // contains 64 initialized values, and the output has capacity for every index.
    let (offset, max_code) = unsafe { take_vectors(&table, values.len(), indices, output_ptr) };

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
    Some(output.freeze())
}

unsafe fn take_vectors<T: FixedWidthTakeValue>(
    table: &[T; 64],
    table_len: usize,
    indices: &[u8],
    output: *mut u8,
) -> (usize, u8) {
    let table0 = unsafe { vld1q_u8(table.as_ptr().cast::<u8>()) };
    let table1 = unsafe { vld1q_u8(table.as_ptr().add(16).cast::<u8>()) };
    let table2 = unsafe { vld1q_u8(table.as_ptr().add(32).cast::<u8>()) };
    let table3 = unsafe { vld1q_u8(table.as_ptr().add(48).cast::<u8>()) };
    let mut max_codes: uint8x16_t = unsafe { vdupq_n_u8(0) };
    let mut offset = 0;
    while offset + 16 <= indices.len() {
        let codes = unsafe { vld1q_u8(indices.as_ptr().add(offset)) };
        max_codes = unsafe { vmaxq_u8(max_codes, codes) };
        // SAFETY: AArch64 guarantees NEON support.
        let decoded = unsafe {
            if table_len <= 16 {
                vqtbl1q_u8(table0, codes)
            } else if table_len <= 32 {
                vqtbl2q_u8(uint8x16x2_t(table0, table1), codes)
            } else {
                vqtbl4q_u8(uint8x16x4_t(table0, table1, table2, table3), codes)
            }
        };
        unsafe { vst1q_u8(output.add(offset), decoded) };
        offset += 16;
    }
    (offset, unsafe { vmaxvq_u8(max_codes) })
}
