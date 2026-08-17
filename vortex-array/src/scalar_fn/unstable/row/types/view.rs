// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Length reporting for row-loop views.
//!
//! [`ViewLen`] lets input and output abstractions expose the number of addressable rows through
//! their borrowed view types.

use vortex_buffer::BitBuffer;

/// The number of rows addressable through a row-loop view.
pub trait ViewLen {
    /// Return the number of addressable rows.
    fn len(&self) -> usize;

    /// Return whether the view contains no rows.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl ViewLen for () {
    fn len(&self) -> usize {
        0
    }
}

impl ViewLen for BitBuffer {
    fn len(&self) -> usize {
        BitBuffer::len(self)
    }
}

impl<T> ViewLen for [T] {
    fn len(&self) -> usize {
        <[T]>::len(self)
    }
}

impl<T> ViewLen for Vec<T> {
    fn len(&self) -> usize {
        Vec::len(self)
    }
}

impl<T: ViewLen + ?Sized> ViewLen for &T {
    fn len(&self) -> usize {
        T::len(self)
    }
}

impl<T: ViewLen + ?Sized> ViewLen for &mut T {
    fn len(&self) -> usize {
        T::len(self)
    }
}
