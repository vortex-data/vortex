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

macro_rules! impl_tuple_view_len {
    ($first:ident; $($rest:ident : $idx:tt),*) => {
        impl<$first: ViewLen, $($rest: ViewLen),*> ViewLen for ($first, $($rest,)*) {
            fn len(&self) -> usize {
                let len = self.0.len();
                $(assert_eq!(self.$idx.len(), len, "tuple views must have equal lengths");)*

                len
            }
        }
    };
}

impl_tuple_view_len!(A;);
impl_tuple_view_len!(A; B: 1);
impl_tuple_view_len!(A; B: 1, C: 2);
impl_tuple_view_len!(A; B: 1, C: 2, D: 3);
impl_tuple_view_len!(A; B: 1, C: 2, D: 3, E: 4);
impl_tuple_view_len!(A; B: 1, C: 2, D: 3, E: 4, F: 5);
impl_tuple_view_len!(A; B: 1, C: 2, D: 3, E: 4, F: 5, G: 6);
impl_tuple_view_len!(A; B: 1, C: 2, D: 3, E: 4, F: 5, G: 6, H: 7);
impl_tuple_view_len!(A; B: 1, C: 2, D: 3, E: 4, F: 5, G: 6, H: 7, I: 8);
impl_tuple_view_len!(A; B: 1, C: 2, D: 3, E: 4, F: 5, G: 6, H: 7, I: 8, J: 9);
impl_tuple_view_len!(A; B: 1, C: 2, D: 3, E: 4, F: 5, G: 6, H: 7, I: 8, J: 9, K: 10);
impl_tuple_view_len!(A; B: 1, C: 2, D: 3, E: 4, F: 5, G: 6, H: 7, I: 8, J: 9, K: 10, L: 11);

#[cfg(test)]
mod tests {
    use super::ViewLen;

    #[test]
    fn tuple_len_returns_common_len() {
        let first: &[i64] = &[1, 2];
        let second: &[i64] = &[3, 4];

        assert_eq!((first, second).len(), 2);
    }

    #[test]
    #[should_panic(expected = "tuple views must have equal lengths")]
    fn tuple_len_rejects_mismatch() {
        let first: &[i64] = &[1];
        let second: &[i64] = &[2, 3];

        let _ = (first, second).len();
    }
}
