// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::mem::MaybeUninit;

use super::CHUNK_LEN;

/// A reusable L1-resident scratch buffer of [`CHUNK_LEN`] elements.
///
/// The scratch is heap-allocated so it does not bloat caller stack frames. It is uninit by
/// default and is written to by each producer chunk; readers only ever see the initialized
/// prefix returned alongside.
pub struct Scratch<T> {
    buf: Box<[MaybeUninit<T>; CHUNK_LEN]>,
}

impl<T> Scratch<T> {
    /// Construct a new uninitialized scratch buffer.
    pub fn new() -> Self {
        let buf: Box<[MaybeUninit<T>; CHUNK_LEN]> =
            Box::new([const { MaybeUninit::<T>::uninit() }; CHUNK_LEN]);
        Self { buf }
    }

    /// Capacity in elements.
    #[inline]
    pub fn capacity(&self) -> usize {
        CHUNK_LEN
    }

    /// Borrow the underlying storage as a slice of uninitialized cells.
    ///
    /// Producers write into this slice and return the initialized prefix to the driver.
    #[inline]
    pub fn as_uninit_mut(&mut self) -> &mut [MaybeUninit<T>; CHUNK_LEN] {
        &mut self.buf
    }
}

impl<T> Default for Scratch<T> {
    fn default() -> Self {
        Self::new()
    }
}
