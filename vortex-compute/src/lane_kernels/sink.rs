// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Writable lane sink — the [`IndexedSink`] trait and the [`ReinterpretSink`] adapter
//! for in-place type-punning kernels.

use std::marker::PhantomData;
use std::mem::align_of;
use std::mem::size_of;

use crate::lane_kernels::source::IndexedSource;

/// An [`IndexedSource`] that also supports unchecked indexed writes — the binding
/// for in-place kernels.
///
/// `Write` is the type written by `set_unchecked` and may differ from
/// `IndexedSource::Item` (the read type). For the canonical `&mut [T]` impl
/// both are `T`. The decoupling is what makes [`ReinterpretSink`] possible —
/// a wrapper that reads `F` and writes `T` over the same backing memory when
/// the two have identical size and alignment.
///
/// Implemented for `&mut [T]`; not implemented for `LaneZip` (you can't write a
/// `(A, B)` pair back to two separate sources via a single index).
pub trait IndexedSink: IndexedSource {
    /// The per-lane write type. Equal to `<Self as IndexedSource>::Item` for
    /// `&mut [T]`; different for [`ReinterpretSink`].
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

/// A sink that reads `F`-values and writes `T`-values over the same backing
/// slice of `F`, reinterpreting each `T` as `F`-bits on write.
///
/// Requires `size_of::<F>() == size_of::<T>()` and `align_of::<F>() == align_of::<T>()`.
/// Both hold for any pair of `NativePType` primitives with equal byte width
/// (e.g. `u32` ↔ `f32`, `u64` ↔ `i64`, `f64` ↔ `u64`).
///
/// Use this when an in-place kernel needs to convert lanes between two
/// types of identical width without allocating a second buffer. After the
/// kernel completes every slot holds a valid `T`-bit pattern; the caller
/// can recover a typed view via `BufferMut::transmute::<T>()`.
pub struct ReinterpretSink<'a, F, T> {
    slice: &'a mut [F],
    _phantom: PhantomData<T>,
}

impl<'a, F, T> ReinterpretSink<'a, F, T> {
    /// Construct a `ReinterpretSink` from `&mut [F]`.
    ///
    /// # Panics
    ///
    /// Panics if `size_of::<F>() != size_of::<T>()` or
    /// `align_of::<F>() != align_of::<T>()`.
    pub fn new(slice: &'a mut [F]) -> Self {
        assert_eq!(
            size_of::<F>(),
            size_of::<T>(),
            "ReinterpretSink requires F and T to have the same size",
        );
        assert_eq!(
            align_of::<F>(),
            align_of::<T>(),
            "ReinterpretSink requires F and T to have the same alignment",
        );
        Self {
            slice,
            _phantom: PhantomData,
        }
    }
}

impl<F: Copy, T: Copy> IndexedSource for ReinterpretSink<'_, F, T> {
    type Item = F;
    #[inline]
    fn len(&self) -> usize {
        self.slice.len()
    }
    #[inline]
    unsafe fn get_unchecked(&self, i: usize) -> F {
        // SAFETY: caller guarantees i < self.slice.len(). Pointer arithmetic
        // avoids method-resolution ambiguity between `<[F]>::get_unchecked` and
        // `IndexedSource::get_unchecked`.
        unsafe { *self.slice.as_ptr().add(i) }
    }
}

impl<F: Copy, T: Copy> IndexedSink for ReinterpretSink<'_, F, T> {
    type Write = T;
    #[inline]
    unsafe fn set_unchecked(&mut self, i: usize, value: T) {
        // SAFETY: caller guarantees i < self.slice.len(); `new` enforces
        // size_of::<F>() == size_of::<T>() and align_of::<F>() == align_of::<T>(),
        // so the F-slot can hold a `T` without overflow or misalignment.
        unsafe {
            let ptr = self.slice.as_mut_ptr().add(i) as *mut T;
            ptr.write(value);
        }
    }
}
