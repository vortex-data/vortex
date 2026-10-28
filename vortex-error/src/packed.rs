// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A one-word encoding of `VortexResult<Option<T>>` for small `T`.

use std::fmt;
use std::fmt::Debug;
use std::fmt::Formatter;
use std::marker::PhantomData;
use std::mem::ManuallyDrop;
use std::sync::Arc;

use crate::ErrorInner;
use crate::VortexError;
use crate::VortexResult;

// An error is stored as its `Arc` data pointer, which must leave the low bit free for the tag.
const _: () = assert!(align_of::<ErrorInner>() >= 2);
const _: () = assert!(size_of::<PackedResultOption<usize>>() == size_of::<usize>());

mod sealed {
    pub trait Sealed {}
}

/// A value that fits in one bit less than a `usize`, so it can share a word with a tag.
///
/// This trait is sealed: it is implemented for `u8`, `u16`, `usize` and, on 64-bit targets,
/// `u32`.
pub trait PackedValue: Copy + sealed::Sealed {
    #[doc(hidden)]
    fn to_bits(self) -> usize;

    #[doc(hidden)]
    fn from_bits(bits: usize) -> Self;
}

macro_rules! impl_packed_value {
    ($($T:ty),*) => {$(
        impl sealed::Sealed for $T {}

        impl PackedValue for $T {
            #[inline]
            fn to_bits(self) -> usize {
                usize::from(self)
            }

            #[inline]
            #[expect(
                clippy::cast_possible_truncation,
                reason = "the bits were produced by `to_bits` from this type"
            )]
            fn from_bits(bits: usize) -> Self {
                bits as $T
            }
        }
    )*};
}

impl_packed_value!(u8, u16);

#[cfg(target_pointer_width = "64")]
impl sealed::Sealed for u32 {}

#[cfg(target_pointer_width = "64")]
impl PackedValue for u32 {
    #[inline]
    fn to_bits(self) -> usize {
        self as usize
    }

    #[inline]
    #[expect(
        clippy::cast_possible_truncation,
        reason = "the bits were produced by `to_bits` from a `u32`"
    )]
    fn from_bits(bits: usize) -> Self {
        bits as u32
    }
}

impl sealed::Sealed for usize {}

impl PackedValue for usize {
    /// # Panics
    ///
    /// Panics if the top bit of `self` is set.
    #[inline]
    fn to_bits(self) -> usize {
        assert!(
            self >> (usize::BITS - 1) == 0,
            "{self} is too large to pack into a PackedResultOption"
        );
        self
    }

    #[inline]
    fn from_bits(bits: usize) -> Self {
        bits
    }
}

/// A [`VortexResult<Option<T>>`] packed into a single word.
///
/// `VortexResult<Option<T>>` normally takes two words, since the error's pointer has only one
/// invalid value (null) for the compiler to use as a tag, and the three states need two. This
/// type also uses the error pointer's alignment:
///
/// - `0` is `Ok(None)`.
/// - An odd value is `Ok(Some(v))`, with `v` in the upper bits.
/// - An even, nonzero value is `Err`, holding the error's [`Arc`] data pointer.
///
/// This only pays off when many results are stored, for example one per chunk. A plain
/// `VortexResult<Option<T>>` return value already comes back in two registers.
///
/// ```
/// use vortex_error::PackedResultOption;
/// use vortex_error::VortexResult;
/// use vortex_error::vortex_err;
///
/// let results: Vec<PackedResultOption<u32>> = vec![
///     Ok(Some(7)).into(),
///     Ok(None).into(),
///     Err(vortex_err!("failed")).into(),
/// ];
/// assert_eq!(size_of_val(&results[0]), size_of::<usize>());
///
/// let mut it = results.into_iter().map(PackedResultOption::into_result);
/// assert_eq!(it.next().unwrap().unwrap(), Some(7));
/// assert_eq!(it.next().unwrap().unwrap(), None);
/// assert!(it.next().unwrap().is_err());
/// ```
#[repr(transparent)]
pub struct PackedResultOption<T: PackedValue> {
    // A pointer rather than a `usize`, so an error keeps its provenance. Values are stored as
    // pointers without provenance, which are never dereferenced.
    ptr: *const ErrorInner,
    _value: PhantomData<T>,
}

// SAFETY: the only shared state is the error, and `VortexError` is `Send` and `Sync`.
unsafe impl<T: PackedValue + Send> Send for PackedResultOption<T> {}
// SAFETY: as above.
unsafe impl<T: PackedValue + Sync> Sync for PackedResultOption<T> {}

const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<VortexError>();
};

const NONE: usize = 0;
const SOME_TAG: usize = 1;

impl<T: PackedValue> PackedResultOption<T> {
    const fn from_ptr(ptr: *const ErrorInner) -> Self {
        Self {
            ptr,
            _value: PhantomData,
        }
    }

    const fn from_bits(bits: usize) -> Self {
        Self::from_ptr(std::ptr::without_provenance(bits))
    }

    fn bits(&self) -> usize {
        self.ptr.addr()
    }

    /// Packs `Ok(None)`.
    pub const fn none() -> Self {
        Self::from_bits(NONE)
    }

    /// Packs `Ok(Some(value))`.
    ///
    /// # Panics
    ///
    /// Panics if `value` is a `usize` with its top bit set.
    pub fn some(value: T) -> Self {
        Self::from_bits((value.to_bits() << 1) | SOME_TAG)
    }

    /// Packs `Err(err)`.
    pub fn err(err: VortexError) -> Self {
        Self::from_ptr(Arc::into_raw(err.0))
    }

    /// Returns `true` if this holds an error.
    pub fn is_err(&self) -> bool {
        let bits = self.bits();
        bits != NONE && bits & SOME_TAG == 0
    }

    /// Unpacks into a [`VortexResult<Option<T>>`].
    pub fn into_result(self) -> VortexResult<Option<T>> {
        let this = ManuallyDrop::new(self);
        // SAFETY: `this` is not dropped, so ownership of the error moves to the result.
        unsafe { this.read_result() }
    }

    /// Returns the unpacked result, cloning the error if there is one.
    ///
    /// Cloning an error is cheap: it increments a reference count.
    pub fn to_result(&self) -> VortexResult<Option<T>> {
        if self.is_err() {
            // SAFETY: the pointer came from `Arc::into_raw` in `err` and `self` still owns that
            // reference, so the allocation is live.
            unsafe { Arc::increment_strong_count(self.ptr) };
        }
        // SAFETY: for an error, the increment above gave this call its own reference.
        unsafe { self.read_result() }
    }

    /// # Safety
    ///
    /// For an error, the caller must own a strong reference that this call can take over.
    unsafe fn read_result(&self) -> VortexResult<Option<T>> {
        match self.bits() {
            NONE => Ok(None),
            bits if bits & SOME_TAG != 0 => Ok(Some(T::from_bits(bits >> 1))),
            // SAFETY: an even, nonzero value is a pointer from `Arc::into_raw` in `err`, and the
            // caller passes us a strong reference to it.
            _ => Err(VortexError(unsafe { Arc::from_raw(self.ptr) })),
        }
    }
}

impl<T: PackedValue> From<VortexResult<Option<T>>> for PackedResultOption<T> {
    fn from(result: VortexResult<Option<T>>) -> Self {
        match result {
            Ok(None) => Self::none(),
            Ok(Some(value)) => Self::some(value),
            Err(err) => Self::err(err),
        }
    }
}

impl<T: PackedValue> From<PackedResultOption<T>> for VortexResult<Option<T>> {
    fn from(packed: PackedResultOption<T>) -> Self {
        packed.into_result()
    }
}

impl<T: PackedValue> Drop for PackedResultOption<T> {
    fn drop(&mut self) {
        if self.is_err() {
            // SAFETY: the pointer came from `Arc::into_raw` in `err` and `self` owns it.
            drop(unsafe { Arc::from_raw(self.ptr) });
        }
    }
}

impl<T: PackedValue> Clone for PackedResultOption<T> {
    fn clone(&self) -> Self {
        self.to_result().into()
    }
}

impl<T: PackedValue + Debug> Debug for PackedResultOption<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        Debug::fmt(&self.to_result(), f)
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;
    use crate::vortex_err;

    #[rstest]
    #[case::none(None)]
    #[case::zero(Some(0))]
    #[case::max(Some(usize::MAX >> 1))]
    fn round_trips_values(#[case] value: Option<usize>) -> VortexResult<()> {
        let packed = PackedResultOption::from(Ok(value));
        assert!(!packed.is_err());
        assert_eq!(packed.to_result()?, value);
        assert_eq!(packed.into_result()?, value);
        Ok(())
    }

    #[test]
    fn round_trips_errors() {
        let packed = PackedResultOption::<u16>::from(Err(vortex_err!("boom")));
        assert!(packed.is_err());

        let cloned = packed.clone();
        let borrowed = packed.to_result();
        drop(packed);

        for result in [borrowed, cloned.into_result()] {
            assert!(result.is_err_and(|err| err.to_string().contains("boom")));
        }
    }

    #[test]
    #[should_panic(expected = "too large to pack")]
    fn rejects_values_using_the_tag_bit() {
        drop(PackedResultOption::some(usize::MAX));
    }
}
