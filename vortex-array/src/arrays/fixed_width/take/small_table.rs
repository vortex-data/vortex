// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Byte take from a table of at most 32 one-byte values, with SIMD byte lookups.

use std::mem::MaybeUninit;
use std::slice;

use vortex_buffer::Buffer;
use vortex_buffer::BufferAllocatorRef;
use vortex_buffer::BufferMut;
use vortex_buffer::take_bytes;

use super::FixedWidthTakeValue;
use super::take_values_fallback;
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
        || values.len() > 32
        || size_of::<T>() != 1
        || indices.len() < 64
    {
        return take_values_fallback(values, indices, allocator);
    }

    // SAFETY: the sealed index type is u8, as checked above.
    let indices: &[u8] = unsafe { slice::from_raw_parts(indices.as_ptr().cast(), indices.len()) };

    // SAFETY: T is one byte with no uninitialized bytes, as checked above and as
    // FixedWidthTakeValue guarantees.
    let table: &[u8] = unsafe { slice::from_raw_parts(values.as_ptr().cast(), values.len()) };

    let mut output = BufferMut::<T>::with_capacity_in(indices.len(), allocator.clone());

    // SAFETY: the spare capacity holds at least indices.len() values of one byte each.
    let spare = unsafe {
        slice::from_raw_parts_mut(
            output
                .spare_capacity_mut()
                .as_mut_ptr()
                .cast::<MaybeUninit<u8>>(),
            indices.len(),
        )
    };
    take_bytes(table, indices, spare);

    // SAFETY: take_bytes initialized a byte, which is a valid T, for every index.
    unsafe { output.set_len(indices.len()) };
    output.freeze()
}
