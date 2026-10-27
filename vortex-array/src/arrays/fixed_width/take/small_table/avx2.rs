// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! AVX2 byte-table take for `u8` codes and at most 16 one-byte values.

cfg_if::cfg_if! {
    if #[cfg(target_arch = "x86")] {
        use std::arch::x86 as arch;
    } else {
        use std::arch::x86_64 as arch;
    }
}

use arch::__m256i;
use arch::_mm_loadu_si128;
use arch::_mm256_broadcastsi128_si256;
use arch::_mm256_loadu_si256;
use arch::_mm256_or_si256;
use arch::_mm256_set1_epi8;
use arch::_mm256_setzero_si256;
use arch::_mm256_shuffle_epi8;
use arch::_mm256_storeu_si256;
use arch::_mm256_subs_epu8;
use arch::_mm256_testz_si256;
use vortex_buffer::Buffer;
use vortex_buffer::BufferAllocatorRef;
use vortex_buffer::BufferMut;
use vortex_error::VortexExpect;

use super::super::FixedWidthTakeValue;
use super::super::HAS_AVX2;
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
        || !*HAS_AVX2
    {
        return take_values_fallback(values, indices, allocator);
    }

    // SAFETY: the sealed index type is u8, as checked above.
    let indices: &[u8] =
        unsafe { std::slice::from_raw_parts(indices.as_ptr().cast(), indices.len()) };
    // SAFETY: AVX2 was detected above. Values are one byte with no uninitialized bytes,
    // and the table contains between 1 and 16 values.
    unsafe { take_avx2(values, indices, allocator) }
}

#[target_feature(enable = "avx2")]
unsafe fn take_avx2<T: FixedWidthTakeValue>(
    values: &[T],
    indices: &[u8],
    allocator: &BufferAllocatorRef,
) -> Buffer<T> {
    let mut table = [values[0]; 16];
    table[..values.len()].copy_from_slice(values);
    // SAFETY: the table contains 16 initialized one-byte values.
    let table = unsafe { _mm_loadu_si128(table.as_ptr().cast()) };
    let table = _mm256_broadcastsi128_si256(table);
    let limit = _mm256_set1_epi8(
        i8::try_from(values.len() - 1).vortex_expect("table contains at most 16 values"),
    );
    let mut invalid = _mm256_setzero_si256();
    let mut output = BufferMut::<T>::with_capacity_in(indices.len(), allocator.clone());
    let spare = output.spare_capacity_mut();
    let output_ptr = spare.as_mut_ptr().cast::<u8>();

    let mut offset = 0;
    while offset + 32 <= indices.len() {
        // SAFETY: the loop condition guarantees a complete 32-byte vector is in bounds.
        let codes = unsafe { _mm256_loadu_si256(indices.as_ptr().add(offset).cast()) };
        invalid = _mm256_or_si256(invalid, _mm256_subs_epu8(codes, limit));
        // SAFETY: the output has capacity for every index and T is one byte wide.
        unsafe {
            _mm256_storeu_si256(
                output_ptr.add(offset).cast::<__m256i>(),
                _mm256_shuffle_epi8(table, codes),
            )
        };
        offset += 32;
    }
    assert_eq!(
        _mm256_testz_si256(invalid, invalid),
        1,
        "take index out of bounds"
    );

    for offset in offset..indices.len() {
        let code = usize::from(indices[offset]);
        assert!(
            code < values.len(),
            "take index {code} out of bounds for length {}",
            values.len()
        );
        spare[offset].write(values[code]);
    }
    // SAFETY: the vector loop and scalar remainder initialized every output value.
    unsafe { output.set_len(indices.len()) };
    output.freeze()
}
