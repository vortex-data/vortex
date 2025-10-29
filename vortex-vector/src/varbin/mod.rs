// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::{Vector, VectorMut};
use std::fmt::Debug;

mod vector;
pub use vector::VarBinVector;

mod vector_mut;
pub use vector_mut::VarBinVectorMut;
use vortex_error::vortex_panic;

mod view;

/// Type alias for non-utf8 variable-length binary vectors.
pub type BinaryVector = VarBinVector<BinaryType>;
/// Type alias for mutable non-utf8 variable-length binary vectors.
pub type BinaryVectorMut = VarBinVectorMut<BinaryType>;
/// Type alias for UTF-8 variable-length string vectors.
pub type StringVector = VarBinVector<StringType>;
/// Type alias for mutable UTF-8 variable-length string vectors.
pub type StringVectorMut = VarBinVectorMut<StringType>;

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

pub trait VarBinTypeDowncast {
    type Output<T: VarBinType>;

    fn into_binary(self) -> Self::Output<BinaryType>;
    fn into_string(self) -> Self::Output<StringType>;
}

pub trait VarBinTypeUpcast {
    type Input<T: VarBinType>;

    fn from_binary(input: Self::Input<BinaryType>) -> Self;
    fn from_string(input: Self::Input<StringType>) -> Self;
}

impl VarBinTypeDowncast for Vector {
    type Output<T: VarBinType> = VarBinVector<T>;

    fn into_binary(self) -> Self::Output<BinaryType> {
        if let Vector::Binary(v) = self {
            return v;
        }
        vortex_panic!("Expected BinaryVector, got {self:?}");
    }

    fn into_string(self) -> Self::Output<StringType> {
        if let Vector::String(v) = self {
            return v;
        }
        vortex_panic!("Expected StringVector, got {self:?}");
    }
}

impl VarBinTypeUpcast for Vector {
    type Input<T: VarBinType> = VarBinVector<T>;

    fn from_binary(input: Self::Input<BinaryType>) -> Self {
        Vector::Binary(input)
    }

    fn from_string(input: Self::Input<StringType>) -> Self {
        Vector::String(input)
    }
}

impl VarBinTypeDowncast for VectorMut {
    type Output<T: VarBinType> = VarBinVectorMut<T>;

    fn into_binary(self) -> Self::Output<BinaryType> {
        if let VectorMut::Binary(v) = self {
            return v;
        }
        vortex_panic!("Expected BinaryVector, got {self:?}");
    }

    fn into_string(self) -> Self::Output<StringType> {
        if let VectorMut::String(v) = self {
            return v;
        }
        vortex_panic!("Expected StringVector, got {self:?}");
    }
}

impl VarBinTypeUpcast for VectorMut {
    type Input<T: VarBinType> = VarBinVectorMut<T>;

    fn from_binary(input: Self::Input<BinaryType>) -> Self {
        VectorMut::Binary(input)
    }

    fn from_string(input: Self::Input<StringType>) -> Self {
        VectorMut::String(input)
    }
}

mod private {
    pub trait Sealed {}
    impl Sealed for super::StringType {}
    impl Sealed for super::BinaryType {}
}
