// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Writable lane sinks for in-place kernels.

use crate::lane_kernels::source::IndexedSource;

/// An [`IndexedSource`] that also supports unchecked indexed writes — the binding
/// for in-place kernels.
///
/// A sink writes the type that it reads. Writing another type into the same storage could leave a
/// bit pattern that is invalid for the element type of that storage, such as `2` in a `bool`. Use
/// an out-of-place kernel from [`IndexedSourceExt`] to convert between types.
///
/// [`IndexedSourceExt`]: crate::lane_kernels::map_into::IndexedSourceExt
pub trait IndexedSink: IndexedSource {
    /// Write `value` into lane `i` without bounds checking.
    ///
    /// # Safety
    ///
    /// `i` must be strictly less than `self.len()`.
    unsafe fn set_unchecked(&mut self, i: usize, value: Self::Item);
}

impl<T: Copy> IndexedSink for &mut [T] {
    #[inline]
    unsafe fn set_unchecked(&mut self, i: usize, value: T) {
        // SAFETY: caller guarantees i < self.len().
        unsafe { *<[T]>::get_unchecked_mut(self, i) = value };
    }
}
