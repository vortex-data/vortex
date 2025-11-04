// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Variable-length binary types and related traits.

use std::fmt::Debug;

use crate::binaryview::{BinaryViewVector, BinaryViewVectorMut};
use crate::{Vector, VectorMut};

impl<T: BinaryViewType> From<BinaryViewVector<T>> for Vector {
    fn from(value: BinaryViewVector<T>) -> Self {
        T::upcast(value)
    }
}

impl<T: BinaryViewType> From<BinaryViewVectorMut<T>> for VectorMut {
    fn from(value: BinaryViewVectorMut<T>) -> Self {
        T::upcast(value)
    }
}

/// Trait to mark supported binary view types.
pub trait BinaryViewType: Debug + Sized + private::Sealed {
    /// The slice type for this variable binary type.
    type Slice: ?Sized + AsRef<[u8]>;

    /// Validate if a set of bytes conforms to the logical type constraints of the native `Slice`.
    fn validate(bytes: &[u8]) -> bool;

    /// Returns the bytes as the native `Slice` type
    /// for this binary view vector.
    ///
    /// # Safety
    ///
    /// The caller must check beforehand that bytes return from the vector conform to the type
    /// requirements of this binary type.
    ///
    /// Failure to do so can result in undefined behavior or incorrect results in downstream
    /// vector operations.
    unsafe fn from_bytes_unchecked(bytes: &[u8]) -> &Self::Slice;

    /// Downcast the provided object to a type-specific instance.
    fn downcast<V: BinaryViewDowncast>(visitor: V) -> V::Output<Self>;

    /// Upcast a type-specific instance to a generic instance.
    fn upcast<V: BinaryViewTypeUpcast>(input: V::Input<Self>) -> V;
}

/// [`BinaryType`] for UTF-8 strings.
#[derive(Clone, Debug)]
pub struct StringType;
impl BinaryViewType for StringType {
    type Slice = str;

    #[inline(always)]
    fn validate(bytes: &[u8]) -> bool {
        std::str::from_utf8(bytes).is_ok()
    }

    unsafe fn from_bytes_unchecked(bytes: &[u8]) -> &Self::Slice {
        // SAFETY: vectors should be checked at the boundary for upholding the UTF8 variant,
        //  or only be built from vectors that are known to satisfy the variant.
        unsafe { std::str::from_utf8_unchecked(bytes) }
    }

    fn downcast<V: BinaryViewDowncast>(visitor: V) -> V::Output<Self> {
        visitor.into_string()
    }

    fn upcast<V: BinaryViewTypeUpcast>(input: V::Input<Self>) -> V {
        V::from_string(input)
    }
}

/// [`BinaryType`] for raw binary data.
#[derive(Clone, Debug)]
pub struct BinaryType;
impl BinaryViewType for BinaryType {
    type Slice = [u8];

    #[inline(always)]
    fn validate(_bytes: &[u8]) -> bool {
        true
    }

    unsafe fn from_bytes_unchecked(bytes: &[u8]) -> &Self::Slice {
        bytes
    }

    fn downcast<V: BinaryViewDowncast>(visitor: V) -> V::Output<Self> {
        visitor.into_binary()
    }

    fn upcast<V: BinaryViewTypeUpcast>(input: V::Input<Self>) -> V {
        V::from_binary(input)
    }
}

/// Trait for downcasting generic variable binary types to specific types.
pub trait BinaryViewDowncast {
    /// The output type after downcasting.
    type Output<T: BinaryViewType>;

    /// Downcast to a binary type.
    fn into_binary(self) -> Self::Output<BinaryType>;
    /// Downcast to a string type.
    fn into_string(self) -> Self::Output<StringType>;
}

/// Trait for upcasting specific variable binary types to generic types.
pub trait BinaryViewTypeUpcast {
    /// The input type for upcasting.
    type Input<T: BinaryViewType>;

    /// Upcast from a binary type.
    fn from_binary(input: Self::Input<BinaryType>) -> Self;
    /// Upcast from a string type.
    fn from_string(input: Self::Input<StringType>) -> Self;
}

/// Private module to seal the `BinaryViewType` trait.
mod private {
    /// Sealed trait to prevent external implementations of
    /// [`BinaryViewType`](super::BinaryViewType).
    pub trait Sealed {}

    impl Sealed for super::StringType {}
    impl Sealed for super::BinaryType {}
}
