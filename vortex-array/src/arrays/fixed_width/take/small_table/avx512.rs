// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! AVX-512 VBMI byte-table take for `u8` codes and at most 32 one-byte values.

use std::sync::LazyLock;

cfg_if::cfg_if! {
    if #[cfg(target_arch = "x86")] {
        use std::arch::x86 as arch;
    } else {
        use std::arch::x86_64 as arch;
    }
}

use arch::_mm512_cmpgt_epu8_mask;
use arch::_mm512_loadu_si512;
use arch::_mm512_max_epu8;
use arch::_mm512_permutexvar_epi8;
use arch::_mm512_set1_epi8;
use arch::_mm512_setzero_si512;
use arch::_mm512_storeu_si512;
use vortex_buffer::Buffer;
use vortex_buffer::BufferAllocatorRef;
use vortex_buffer::BufferMut;
use vortex_error::VortexExpect;

use super::super::FixedWidthTakeValue;

pub(super) static HAS_AVX512_VBMI: LazyLock<bool> = LazyLock::new(|| {
    is_x86_feature_detected!("avx512f")
        && is_x86_feature_detected!("avx512bw")
        && is_x86_feature_detected!("avx512vbmi")
});

/// Look up byte codes in a register table, checking every code against the dictionary length.
///
/// # Safety
///
/// Requires AVX-512F, BW, and VBMI, one-byte T, and between 1 and 32 values.
#[target_feature(enable = "avx512f,avx512bw,avx512vbmi")]
pub(super) unsafe fn take_avx512<T: FixedWidthTakeValue>(
    values: &[T],
    indices: &[u8],
    allocator: &BufferAllocatorRef,
) -> Buffer<T> {
    let mut table = [values[0]; 64];
    table[..values.len()].copy_from_slice(values);
    // SAFETY: The padded table contains 64 initialized one-byte values.
    let table = unsafe { _mm512_loadu_si512(table.as_ptr().cast()) };
    let mut max_codes = _mm512_setzero_si512();
    let mut output = BufferMut::<T>::with_capacity_in(indices.len(), allocator.clone());
    let spare = output.spare_capacity_mut();
    let output_ptr = spare.as_mut_ptr().cast::<u8>();

    let mut offset = 0;
    while offset + 64 <= indices.len() {
        // SAFETY: The loop condition guarantees a complete 64-byte input and output vector.
        let codes = unsafe { _mm512_loadu_si512(indices.as_ptr().add(offset).cast()) };
        max_codes = _mm512_max_epu8(max_codes, codes);
        let taken = _mm512_permutexvar_epi8(codes, table);
        // SAFETY: Output has capacity for every index and T is one byte wide.
        unsafe { _mm512_storeu_si512(output_ptr.add(offset).cast(), taken) };
        offset += 64;
    }
    let limit = _mm512_set1_epi8(
        i8::try_from(values.len() - 1).vortex_expect("table contains at most 32 values"),
    );
    // VPERMB masks indices, so explicitly reject invalid codes rather than silently wrapping.
    assert_eq!(
        _mm512_cmpgt_epu8_mask(max_codes, limit),
        0,
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
    // SAFETY: The vector loop and checked scalar tail initialized every output element.
    unsafe { output.set_len(indices.len()) };
    output.freeze()
}
