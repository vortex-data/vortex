// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Variable-length binary types and related traits.

use std::fmt::Debug;

use crate::{VarBinVector, VarBinVectorMut, Vector, VectorMut};

impl<T: VarBinType> From<VarBinVector<T>> for Vector {
    fn from(value: VarBinVector<T>) -> Self {
        T::upcast(value)
    }
}

impl<T: VarBinType> From<VarBinVectorMut<T>> for VectorMut {
    fn from(value: VarBinVectorMut<T>) -> Self {
        T::upcast(value)
    }
}

/// Trait to mark supported binary view types.
pub trait VarBinType: Debug + Sized + private::Sealed {
    /// The slice type for this variable binary type.
    type Slice: ?Sized + AsRef<[u8]>;

    /// Downcast the provided object to a type-specific instance.
    fn downcast<V: VarBinTypeDowncast>(visitor: V) -> V::Output<Self>;

    /// Upcast a type-specific instance to a generic instance.
    fn upcast<V: VarBinTypeUpcast>(input: V::Input<Self>) -> V;
}

/// [`BinaryType`] for UTF-8 strings.
#[derive(Clone, Debug)]
pub struct StringType;
impl VarBinType for StringType {
    type Slice = str;

    fn downcast<V: VarBinTypeDowncast>(visitor: V) -> V::Output<Self> {
        visitor.into_string()
    }

    fn upcast<V: VarBinTypeUpcast>(input: V::Input<Self>) -> V {
        V::from_string(input)
    }
}

/// [`BinaryType`] for raw binary data.
#[derive(Clone, Debug)]
pub struct BinaryType;
impl VarBinType for BinaryType {
    type Slice = [u8];

    fn downcast<V: VarBinTypeDowncast>(visitor: V) -> V::Output<Self> {
        visitor.into_binary()
    }

    fn upcast<V: VarBinTypeUpcast>(input: V::Input<Self>) -> V {
        V::from_binary(input)
    }
}

/// Trait for downcasting generic variable binary types to specific types.
pub trait VarBinTypeDowncast {
    /// The output type after downcasting.
    type Output<T: VarBinType>;

    /// Downcast to a binary type.
    fn into_binary(self) -> Self::Output<BinaryType>;
    /// Downcast to a string type.
    fn into_string(self) -> Self::Output<StringType>;
}

/// Trait for upcasting specific variable binary types to generic types.
pub trait VarBinTypeUpcast {
    /// The input type for upcasting.
    type Input<T: VarBinType>;

    /// Upcast from a binary type.
    fn from_binary(input: Self::Input<BinaryType>) -> Self;
    /// Upcast from a string type.
    fn from_string(input: Self::Input<StringType>) -> Self;
}

/// Private module to seal the [`VarBinType`] trait.
mod private {
    /// Sealed trait to prevent external implementations of [`VarBinType`].
    pub trait Sealed {}

    impl Sealed for super::StringType {}
    impl Sealed for super::BinaryType {}
}
