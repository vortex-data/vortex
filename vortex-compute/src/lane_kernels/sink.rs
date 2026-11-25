// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Writable lane sinks for in-place kernels.

use crate::lane_kernels::source::IndexedSource;

/// An [`IndexedSource`] that also supports unchecked indexed writes — the binding
/// for in-place kernels.
///
/// For `&mut [T]`, both the read and write types are `T`.
pub trait IndexedSink: IndexedSource {
    /// The per-lane write type.
    type Write: Copy;

    /// Write `value` into lane `i` without bounds checking.
    ///
    /// # Safety
    ///
    /// `i` must be strictly less than `self.len()`.
    unsafe fn set_unchecked(&mut self, i: usize, value: Self::Write);
}

impl<T: Copy> IndexedSink for &mut [T] {
    type Write = T;
    #[inline]
    unsafe fn set_unchecked(&mut self, i: usize, value: T) {
        // SAFETY: caller guarantees i < self.len().
        unsafe { *<[T]>::get_unchecked_mut(self, i) = value };
    }
}
