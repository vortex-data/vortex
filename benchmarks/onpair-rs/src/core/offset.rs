// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Row-offset width abstraction.
//!
//! A column's per-row code offsets mirror the width of the Arrow array it was
//! built from: `u32` (binary) or `u64` (large-binary).

mod sealed {
    pub trait Sealed {}
    impl Sealed for u32 {}
    impl Sealed for u64 {}
}

/// Width of a row offset. Sealed to `u32` and `u64` to match the Arrow binary
/// and large-binary layouts.
pub trait Offset: sealed::Sealed + Copy + Clone + Default + std::fmt::Debug + 'static {
    /// Widen to `usize`. Infallible on the supported 64-bit host.
    fn to_usize(self) -> usize;
    /// Narrow from `usize`, truncating if out of range. The caller guarantees
    /// the value fits.
    fn from_usize(n: usize) -> Self;
}

impl Offset for u32 {
    #[inline]
    fn to_usize(self) -> usize {
        self as usize
    }
    #[inline]
    fn from_usize(n: usize) -> Self {
        n as u32
    }
}

impl Offset for u64 {
    #[inline]
    fn to_usize(self) -> usize {
        self as usize
    }
    #[inline]
    fn from_usize(n: usize) -> Self {
        n as u64
    }
}
