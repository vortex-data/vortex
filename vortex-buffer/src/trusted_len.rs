// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

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
}

impl<I: Iterator> Iterator for TrustedLenAdapter<I> {
    type Item = I::Item;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let item = self.inner.next();
        if item.is_some() {
            debug_assert!(
                self.len > 0,
                "TrustedLenAdapter: iterator yielded more items than promised"
            );
            self.len = self.len.saturating_sub(1);
        } else {
            debug_assert_eq!(self.len, 0, "TrustedLenAdapter: iterator ended early");
            self.len = 0;
        }
        item
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.len, Some(self.len))
    }
}

unsafe impl<I: Iterator> TrustedLen for TrustedLenAdapter<I> {}

/// Extension trait for wrapping any iterator in a [`TrustedLenAdapter`].
pub trait TrustedLenExt: Iterator + Sized {
    /// Wraps this iterator in a `TrustedLenAdapter`.
    ///
    /// # Safety
    ///
    /// The caller must guarantee that the iterator does indeed have an exact length.
    unsafe fn trusted_len(self) -> TrustedLenAdapter<Self> {
        let (lower_bound, upper_bound_opt) = self.size_hint();
        if let Some(upper_bound) = upper_bound_opt {
            assert_eq!(
                lower_bound, upper_bound,
                "TrustedLenExt: iterator size hints must match if upper bound is given"
            );
        }

        TrustedLenAdapter {
            inner: self,
            len: lower_bound,
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
unsafe impl<'a, I, T: 'a, E: 'a> TrustedLen for itertools::ProcessResults<'a, I, E> where
    I: TrustedLen<Item = Result<T, E>>
{
}

// Enumerate
unsafe impl<I, T> TrustedLen for std::iter::Enumerate<I> where I: TrustedLen<Item = T> {}

// Zip
//
// Both halves must be `TrustedLen`. `Zip::size_hint` reports `min(a_upper, b_upper)`, so an
// untrusted `U` that under-reports its upper bound while yielding more items makes the `Zip`
// under-report too. `BufferMut::extend_trusted` reserves exactly that many slots and then
// writes without bounds checks, so an unconstrained `U` turns a safe `Iterator` impl into an
// out-of-bounds heap write.
/// A `Zip` is only `TrustedLen` when both halves are, so it cannot be built from an iterator
/// without an exact upper bound, such as [`std::iter::repeat`]:
///
/// ```compile_fail
/// use vortex_buffer::trusted_len::TrustedLen;
///
/// fn assert_trusted_len<I: TrustedLen>(_: I) {}
/// assert_trusted_len([1u8, 2, 3].into_iter().zip(std::iter::repeat(0u8)));
/// ```
///
/// Bound the other half instead, e.g. with [`std::iter::repeat_n`]:
///
/// ```
/// use vortex_buffer::trusted_len::TrustedLen;
///
/// fn assert_trusted_len<I: TrustedLen>(_: I) {}
/// assert_trusted_len([1u8, 2, 3].into_iter().zip(std::iter::repeat_n(0u8, 3)));
/// ```
unsafe impl<T, U> TrustedLen for std::iter::Zip<T, U>
where
    T: TrustedLen,
    U: TrustedLen,
{
}

unsafe impl<T: Clone> TrustedLen for std::iter::RepeatN<T> {}

// Arrow bit iterators
unsafe impl<'a> TrustedLen for crate::bit::BitIterator<'a> {}
unsafe impl<'a> TrustedLen for crate::bit::BitChunkIterator<'a> {}
unsafe impl<'a> TrustedLen for crate::bit::UnalignedBitChunkIterator<'a> {}

#[cfg(test)]
mod tests {
    use super::TrustedLenExt;
    use crate::BufferMut;

    #[test]
    fn test_trusted_len_adapter_tracks_remaining_len() {
        // SAFETY: `[10u32, 20, 30, 40].into_iter()` has an exact size hint.
        let mut iter = unsafe { [10u32, 20, 30, 40].into_iter().trusted_len() };
        assert_eq!(iter.size_hint(), (4, Some(4)));

        assert_eq!(iter.next(), Some(10));
        assert_eq!(iter.size_hint(), (3, Some(3)));

        let buf = BufferMut::from_trusted_len_iter(iter);
        assert_eq!(buf.as_slice(), &[20, 30, 40]);

        // Exhausting the adapter leaves its size hint at 0.
        // SAFETY: `[1u32].into_iter()` has an exact size hint.
        let mut single = unsafe { [1u32].into_iter().trusted_len() };
        assert_eq!(single.next(), Some(1));
        assert_eq!(single.size_hint(), (0, Some(0)));
        assert_eq!(single.next(), None);
        assert_eq!(single.size_hint(), (0, Some(0)));
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "TrustedLenAdapter: iterator ended early")]
    fn test_trusted_len_adapter_detects_early_end() {
        struct ShortIter(usize);
        impl Iterator for ShortIter {
            type Item = u32;
            fn next(&mut self) -> Option<Self::Item> {
                None
            }
            fn size_hint(&self) -> (usize, Option<usize>) {
                (self.0, Some(self.0))
            }
        }

        // SAFETY: intentionally invalid iterator to test the debug assertion.
        let mut iter = unsafe { ShortIter(2).trusted_len() };
        let _ = iter.next();
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "TrustedLenAdapter: iterator yielded more items than promised")]
    fn test_trusted_len_adapter_detects_overrun() {
        struct LongIter(usize);
        impl Iterator for LongIter {
            type Item = u32;
            fn next(&mut self) -> Option<Self::Item> {
                (self.0 > 0).then(|| {
                    self.0 -= 1;
                    1
                })
            }
            fn size_hint(&self) -> (usize, Option<usize>) {
                (1, Some(1))
            }
        }

        // SAFETY: intentionally invalid iterator to test the debug assertion.
        let mut iter = unsafe { LongIter(2).trusted_len() };
        assert_eq!(iter.next(), Some(1));
        let _ = iter.next();
    }
}
