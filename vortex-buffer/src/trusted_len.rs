//  SPDX-License-Identifier: Apache-2.0
//  SPDX-FileCopyrightText: Copyright the Vortex contributors

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

macro_rules! impl_for_range {
    ($($typ:ty),*) => {
        $(
            unsafe impl TrustedLen for std::ops::Range<$typ> {}
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
