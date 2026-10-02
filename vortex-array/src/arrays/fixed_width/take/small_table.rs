// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Byte-table take for `u8` codes and at most 16 one-byte values.

#[cfg(all(target_arch = "aarch64", target_endian = "little"))]
mod arch {
    use std::arch::aarch64::uint8x16_t;
    use std::arch::aarch64::vdupq_n_u8;
    use std::arch::aarch64::vld1q_u8;
    use std::arch::aarch64::vmaxq_u8;
    use std::arch::aarch64::vmaxvq_u8;
    use std::arch::aarch64::vqtbl1q_u8;
    use std::arch::aarch64::vst1q_u8;

    use vortex_buffer::Buffer;
    use vortex_buffer::BufferMut;

    use super::super::FixedWidthTakeValue;

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
}

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
mod arch {
    use std::arch::x86_64::__m256i;
    use std::arch::x86_64::_mm_loadu_si128;
    use std::arch::x86_64::_mm256_broadcastsi128_si256;
    use std::arch::x86_64::_mm256_loadu_si256;
    use std::arch::x86_64::_mm256_or_si256;
    use std::arch::x86_64::_mm256_set1_epi8;
    use std::arch::x86_64::_mm256_setzero_si256;
    use std::arch::x86_64::_mm256_shuffle_epi8;
    use std::arch::x86_64::_mm256_storeu_si256;
    use std::arch::x86_64::_mm256_subs_epu8;
    use std::arch::x86_64::_mm256_testz_si256;

    use vortex_buffer::Buffer;
    use vortex_buffer::BufferMut;

    use super::super::FixedWidthTakeValue;

    pub(crate) fn take<T: FixedWidthTakeValue>(values: &[T], indices: &[u8]) -> Option<Buffer<T>> {
        if values.is_empty() || values.len() > 16 || size_of::<T>() != 1 || indices.len() < 64 {
            return None;
        }
        // SAFETY: the caller detects AVX2 before entering this architecture-specific module.
        Some(unsafe { take_avx2(values, indices) })
    }

    #[target_feature(enable = "avx2")]
    unsafe fn take_avx2<T: FixedWidthTakeValue>(values: &[T], indices: &[u8]) -> Buffer<T> {
        let mut table = [0u8; 16];
        unsafe {
            std::ptr::copy_nonoverlapping(
                values.as_ptr().cast::<u8>(),
                table.as_mut_ptr(),
                values.len(),
            );
        }
        let table = unsafe { _mm_loadu_si128(table.as_ptr().cast()) };
        let table = _mm256_broadcastsi128_si256(table);
        let limit = _mm256_set1_epi8((values.len() - 1) as i8);
        let mut invalid = _mm256_setzero_si256();
        let mut output = BufferMut::<T>::with_capacity(indices.len());
        let output_ptr = output.spare_capacity_mut().as_mut_ptr().cast::<u8>();

        let mut offset = 0;
        while offset + 32 <= indices.len() {
            let codes = unsafe { _mm256_loadu_si256(indices.as_ptr().add(offset).cast()) };
            invalid = _mm256_or_si256(invalid, _mm256_subs_epu8(codes, limit));
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
            unsafe {
                output_ptr
                    .add(offset)
                    .write(*values.as_ptr().add(code).cast::<u8>())
            };
        }
        unsafe { output.set_len(indices.len()) };
        output.freeze()
    }
}

pub(super) use arch::take;
