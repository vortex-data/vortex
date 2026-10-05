// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! NEON register-table take for small dictionaries with byte codes.

mod planes;
mod table;

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
    // The byte-table footprint and interleaving cost determine which widths beat scalar take.
    let max_values = match size_of::<T>() {
        1 => 256,
        2 => 128,
        4 => 64,
        8 if indices.len() >= 1_024 => 32,
        _ => 0,
    };
    if I::PTYPE != PType::U8
        || values.is_empty()
        || values.len() > max_values
        || indices.len() < 64
    {
        return take_values_fallback(values, indices, allocator);
    }

    // SAFETY: the sealed index type is u8, as checked above.
    let indices: &[u8] =
        unsafe { std::slice::from_raw_parts(indices.as_ptr().cast(), indices.len()) };

    // SAFETY: FixedWidthTakeValue guarantees initialized bytes. The dispatch above bounds the
    // complete dictionary to 256 bytes and excludes zero-width values.
    let value_bytes = unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), size_of_val(values))
    };

    let mut output = BufferMut::<T>::with_capacity_in(indices.len(), allocator.clone());
    let spare = output.spare_capacity_mut();
    let output_ptr = spare.as_mut_ptr().cast::<u8>();
    // SAFETY: AArch64 provides NEON, the table is fully initialized, and output has capacity for
    // every input code. Each specialization covers the dictionary's entire byte representation.
    let (offset, max_code) = unsafe {
        match size_of::<T>() {
            2 => planes::take_vectors::<2>(value_bytes, indices, output_ptr),
            4 => planes::take_vectors::<4>(value_bytes, indices, output_ptr),
            8 => planes::take_vectors::<8>(value_bytes, indices, output_ptr),
            _ => {
                let mut table = [0u8; 256];
                table[..value_bytes.len()].copy_from_slice(value_bytes);
                match value_bytes.len() {
                    1..=16 => table::take_vectors::<T, 16>(&table, indices, output_ptr),
                    17..=32 => table::take_vectors::<T, 32>(&table, indices, output_ptr),
                    33..=48 => table::take_vectors::<T, 48>(&table, indices, output_ptr),
                    49..=64 => table::take_vectors::<T, 64>(&table, indices, output_ptr),
                    65..=128 => table::take_vectors::<T, 128>(&table, indices, output_ptr),
                    129..=192 => table::take_vectors::<T, 192>(&table, indices, output_ptr),
                    _ => table::take_vectors::<T, 256>(&table, indices, output_ptr),
                }
            }
        }
    };

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
