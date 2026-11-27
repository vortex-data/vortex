// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Software prefetch hints.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::_MM_HINT_ET0;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::_mm_prefetch;

/// Hint that the cache lines holding `values` are about to be written.
///
/// A no-op on targets other than x86-64.
#[inline]
pub fn prefetch_write<T>(values: &[T]) {
    #[cfg(target_arch = "x86_64")]
    {
        let start = values.as_ptr().cast::<i8>();
        for offset in (0..size_of_val(values)).step_by(64) {
            // SAFETY: a prefetch has no observable effect, and `offset` stays inside `values`.
            unsafe { _mm_prefetch::<_MM_HINT_ET0>(start.wrapping_add(offset)) };
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    let _ = values;
}
