// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use itertools::ProcessResults;

/// Trait for all types which have a known upper-bound.
///
/// Functions that receive a `TrustedLen` iterator can assume that it's `size_hint` is exact,
/// and can pre-allocate memory, unroll loops, or otherwise optimize their implementations
/// accordingly.
///
/// # Safety
///
/// The type which implements this trait must provide an exact `Some` upper-bound for its
/// `size_hint` method. Failure to do so can trigger undefined behavior in users of the trait.
pub unsafe trait TrustedLen: Iterator {}

/// An adapter that turns any iterator into a `TrustedLen` iterator.
///
/// # Safety
///
/// The caller must guarantee that the wrapped iterator does indeed have an exact length.
pub struct TrustedLenAdapter<I> {
    inner: I,
    len: usize,
    #[cfg(debug_assertions)]
    count: usize,
}

impl<I: Iterator> Iterator for TrustedLenAdapter<I> {
    type Item = I::Item;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        match self.inner.next() {
            None => {
                #[cfg(debug_assertions)]
                {
                    assert_eq!(
                        self.len, self.count,
                        "TrustedLenAdapter: iterator ended early"
                    );
                }
                None
            }
            Some(item) => {
                #[cfg(debug_assertions)]
                {
                    self.count += 1;
                    assert!(
                        self.count <= self.len,
                        "TrustedLenAdapter: iterator yielded more items than promised"
                    );
                }
                Some(item)
            }
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.len, Some(self.len))
    }
}

unsafe impl<I: Iterator> TrustedLen for TrustedLenAdapter<I> {}

pub trait TrustedLenExt: Iterator + Sized {
    /// Wraps this iterator in a `TrustedLenAdapter`.
    ///
    /// # Safety
    ///
    /// The caller must guarantee that the iterator does indeed have an exact length.
    unsafe fn trusted_len(self) -> TrustedLenAdapter<Self> {
        let (lower, maybe_upper) = self.size_hint();
        if let Some(upper) = maybe_upper {
            assert_eq!(
                lower, upper,
                "TrustedLenExt: iterator size hints must match if upper bound is given"
            );
        }

        TrustedLenAdapter {
            inner: self,
            len: lower,
            #[cfg(debug_assertions)]
            count: 0,
        }
    }
}

impl<I: Iterator> TrustedLenExt for I {}

macro_rules! impl_for_range {
    ($($typ:ty),*) => {
        $(
            unsafe impl TrustedLen for std::ops::Range<$typ> {}
            unsafe impl TrustedLen for std::ops::RangeInclusive<$typ> {}
            // StepBy
            // This is only fine for iterators that are TrustedRandomAccess but instead of adding another trait we just declare step by of ranges as supported
            unsafe impl TrustedLen for std::iter::StepBy<std::ops::Range<$typ>> {}
            unsafe impl TrustedLen for std::iter::StepBy<std::ops::RangeInclusive<$typ>> {}
        )*
    };
}

impl_for_range!(u8, u16, u32, u64, i8, i16, i32, i64, usize);

// std::slice related types
unsafe impl<T> TrustedLen for std::slice::Iter<'_, T> {}

unsafe impl<T> TrustedLen for std::slice::IterMut<'_, T> {}

// Iterator types
unsafe impl<B, I, F> TrustedLen for std::iter::Map<I, F>
where
    I: TrustedLen,
    F: FnMut(I::Item) -> B,
{
}

unsafe impl<I> TrustedLen for std::iter::Skip<I> where I: TrustedLen {}

unsafe impl<'a, I, T: 'a> TrustedLen for std::iter::Copied<I>
where
    I: TrustedLen<Item = &'a T>,
    T: Copy,
{
}

unsafe impl<'a, I, T: 'a> TrustedLen for std::iter::Cloned<I>
where
    I: TrustedLen<Item = &'a T>,
    T: Clone,
{
}

unsafe impl<T> TrustedLen for std::vec::IntoIter<T> {}

// Arrays
unsafe impl<T, const N: usize> TrustedLen for std::array::IntoIter<T, N> {}

// Buffer
unsafe impl<T> TrustedLen for crate::Iter<'_, T> {}
unsafe impl<T: Copy> TrustedLen for crate::BufferIterator<T> {}

// ProcessResults
unsafe impl<'a, I, T: 'a, E: 'a> TrustedLen for ProcessResults<'a, I, E> where
    I: TrustedLen<Item = Result<T, E>>
{
}

// Enumerate
unsafe impl<I, T> TrustedLen for std::iter::Enumerate<I> where I: TrustedLen<Item = T> {}

// Zip
unsafe impl<T, U> TrustedLen for std::iter::Zip<T, U>
where
    T: TrustedLen,
    U: TrustedLen,
{
}

// Arrow bit iterators
unsafe impl<'a> TrustedLen for crate::bit::BitChunkIterator<'a> {}
unsafe impl<'a> TrustedLen for crate::bit::UnalignedBitChunkIterator<'a> {}
