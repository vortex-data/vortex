// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Defines a [`Buffer`] wrapper whose values are known to be strictly sorted.

use std::ops::Deref;

use vortex_buffer::Buffer;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

/// A buffer whose values are known to be strictly sorted in ascending order.
///
/// Dereferences to the inner [`Buffer`], which in turn dereferences to a slice.
#[derive(Clone, Debug)]
pub struct StrictSortedBuffer<T> {
    buffer: Buffer<T>,
}

impl<T> StrictSortedBuffer<T> {
    /// Create a new buffer without checking that the values are strictly increasing.
    ///
    /// # Safety
    ///
    /// The values must be strictly increasing. Callers of [`StrictSortedBuffer`] rely on this
    /// invariant, for example to binary search the buffer.
    pub unsafe fn new_unchecked(buffer: Buffer<T>) -> Self {
        Self { buffer }
    }

    /// Return the sorted buffer.
    pub fn into_inner(self) -> Buffer<T> {
        self.buffer
    }
}

impl<T: Ord> StrictSortedBuffer<T> {
    /// Create a new buffer, failing if the values are not strictly increasing.
    pub fn try_new(buffer: Buffer<T>) -> VortexResult<Self> {
        for (idx, window) in buffer.windows(2).enumerate() {
            if window[0] >= window[1] {
                vortex_bail!(
                    "buffer values must be strictly increasing at positions {} and {}",
                    idx,
                    idx + 1
                );
            }
        }
        Ok(Self { buffer })
    }
}

impl<T> Default for StrictSortedBuffer<T> {
    fn default() -> Self {
        Self {
            buffer: Buffer::default(),
        }
    }
}

impl<T> Deref for StrictSortedBuffer<T> {
    type Target = Buffer<T>;

    fn deref(&self) -> &Self::Target {
        &self.buffer
    }
}

#[cfg(test)]
mod tests {
    use vortex_buffer::Buffer;

    use super::StrictSortedBuffer;

    #[test]
    fn rejects_unsorted_values() {
        let err = StrictSortedBuffer::try_new(Buffer::from_iter([3, 1])).unwrap_err();
        assert!(err.to_string().contains("strictly increasing"));
    }

    #[test]
    fn rejects_duplicate_values() {
        let err = StrictSortedBuffer::try_new(Buffer::from_iter([1, 1])).unwrap_err();
        assert!(err.to_string().contains("strictly increasing"));
    }
}
