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
    #[cfg(debug_assertions)]
    count: usize,
}

impl<I: Iterator> Iterator for TrustedLenAdapter<I> {
    type Item = I::Item;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let item = self.inner.next();
        #[cfg(debug_assertions)]
        {
            if item.is_some() {
                self.count += 1;
                assert!(
                    self.count <= self.len,
                    "TrustedLenAdapter: iterator yielded more items than promised"
                );
            } else {
                assert_eq!(
                    self.len, self.count,
                    "TrustedLenAdapter: iterator ended early"
                );
            }
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

    /// Reports `hint` items but yields `actual`.
    #[cfg(debug_assertions)]
    struct MisreportedLen {
        hint: usize,
        actual: usize,
    }

    #[cfg(debug_assertions)]
    impl Iterator for MisreportedLen {
        type Item = usize;

        fn next(&mut self) -> Option<usize> {
            self.actual = self.actual.checked_sub(1)?;
            Some(self.actual)
        }

        fn size_hint(&self) -> (usize, Option<usize>) {
            (self.hint, Some(self.hint))
        }
    }

    #[test]
    fn adapter_yields_inner_items() {
        // SAFETY: a `Vec` iterator reports its exact length.
        let mut iter = unsafe { vec![1, 2, 3].into_iter().trusted_len() };
        assert_eq!(iter.size_hint(), (3, Some(3)));
        assert_eq!(iter.by_ref().collect::<Vec<_>>(), [1, 2, 3]);
        assert_eq!(iter.next(), None);
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "TrustedLenAdapter: iterator yielded more items than promised")]
    fn adapter_rejects_extra_items() {
        // SAFETY: the length is misreported on purpose, and nothing relies on it beyond the
        // adapter's own debug check.
        let iter = unsafe { MisreportedLen { hint: 2, actual: 3 }.trusted_len() };
        drop(iter.collect::<Vec<_>>());
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "TrustedLenAdapter: iterator ended early")]
    fn adapter_rejects_missing_items() {
        // SAFETY: the length is misreported on purpose, and nothing relies on it beyond the
        // adapter's own debug check.
        let iter = unsafe { MisreportedLen { hint: 3, actual: 2 }.trusted_len() };
        drop(iter.collect::<Vec<_>>());
    }
}
