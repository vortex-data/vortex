// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Definition of the [`Vector`] type, which represents immutable and fully decompressed (canonical)
//! array data.
//!
//! [`Vector`] can be transformed into the [`VectorMut`] type if it is owned.

use vortex_error::vortex_panic;

use crate::{
    BinaryVector, BoolVector, DecimalVector, NullVector, PrimitiveVector, StringVector,
    StructVector, VectorMut, VectorOps, match_each_vector,
};

/// An enum over all kinds of immutable vectors, which represent fully decompressed (canonical)
/// array data.
///
/// Most of the behavior of `Vector` is described by the [`VectorOps`] trait. Note that vectors are
/// **always** considered as nullable, and it is the responsibility of the user to not add any
/// nullable data to a vector they want to keep as non-nullable.
///
/// The mutable equivalent of this type is [`VectorMut`], which implements the
/// [`VectorMutOps`](crate::VectorMutOps) trait.
#[derive(Debug, Clone)]
pub enum Vector {
    /// Null vectors.
    Null(NullVector),
    /// Boolean vectors.
    Bool(BoolVector),
    /// Decimal
    Decimal(DecimalVector),
    /// Primitive vectors.
    ///
    /// Note that [`PrimitiveVector`] is an enum over the different possible (generic)
    /// [`PVector<T>`](crate::PVector)s. See the documentation for more information.
    Primitive(PrimitiveVector),
    /// String vectors
    String(StringVector),
    /// Binary vectors
    Binary(BinaryVector),
    // List
    // List(ListVector),
    // FixedList
    // FixedList(FixedListVector),
    /// Vectors of Struct elements.
    Struct(StructVector),
}

impl VectorOps for Vector {
    type Mutable = VectorMut;

    fn len(&self) -> usize {
        match_each_vector!(self, |v| { v.len() })
    }

    fn validity(&self) -> &vortex_mask::Mask {
        match_each_vector!(self, |v| { v.validity() })
    }

    fn try_into_mut(self) -> Result<VectorMut, Self>
    where
        Self: Sized,
    {
        match_each_vector!(self, |v| {
            v.try_into_mut().map(VectorMut::from).map_err(Vector::from)
        })
    }
}

impl Vector {
    /// Returns a reference to the inner [`NullVector`] if `self` is of that variant.
    pub fn as_null(&self) -> &NullVector {
        if let Vector::Null(v) = self {
            return v;
        }
        vortex_panic!("Expected NullVector, got {self:?}");
    }

    /// Returns a reference to the inner [`BoolVector`] if `self` is of that variant.
    pub fn as_bool(&self) -> &BoolVector {
        if let Vector::Bool(v) = self {
            return v;
        }
        vortex_panic!("Expected BoolVector, got {self:?}");
    }

    /// Returns a reference to the inner [`PrimitiveVector`] if `self` is of that variant.
    pub fn as_primitive(&self) -> &PrimitiveVector {
        if let Vector::Primitive(v) = self {
            return v;
        }
        vortex_panic!("Expected PrimitiveVector, got {self:?}");
    }

    /// Returns a reference to the inner [`StringVector`] if `self` is of that variant.
    pub fn as_string(&self) -> &StringVector {
        if let Vector::String(v) = self {
            return v;
        }
        vortex_panic!("Expected StringVector, got {self:?}");
    }

    /// Returns a reference to the inner [`BinaryVector`] if `self` is of that variant.
    pub fn as_binary(&self) -> &BinaryVector {
        if let Vector::Binary(v) = self {
            return v;
        }
        vortex_panic!("Expected BinaryVector, got {self:?}");
    }

    /// Returns a reference to the inner [`StructVector`] if `self` is of that variant.
    pub fn as_struct(&self) -> &StructVector {
        if let Vector::Struct(v) = self {
            return v;
        }
        vortex_panic!("Expected StructVector, got {self:?}");
    }

    /// Consumes `self` and returns the inner [`NullVector`] if `self` is of that variant.
    pub fn into_null(self) -> NullVector {
        if let Vector::Null(v) = self {
            return v;
        }
        vortex_panic!("Expected NullVector, got {self:?}");
    }

    /// Consumes `self` and returns the inner [`BoolVector`] if `self` is of that variant.
    pub fn into_bool(self) -> BoolVector {
        if let Vector::Bool(v) = self {
            return v;
        }
        vortex_panic!("Expected BoolVector, got {self:?}");
    }

    /// Consumes `self` and returns the inner [`PrimitiveVector`] if `self` is of that variant.
    pub fn into_primitive(self) -> PrimitiveVector {
        if let Vector::Primitive(v) = self {
            return v;
        }
        vortex_panic!("Expected PrimitiveVector, got {self:?}");
    }

    /// Consumes `self` and returns the inner [`StringVector`] if `self` is of that variant.
    #[allow(clippy::same_name_method)] // Same as VarBinTypeDowncast
    pub fn into_string(self) -> StringVector {
        if let Vector::String(v) = self {
            return v;
        }
        vortex_panic!("Expected StringVector, got {self:?}");
    }

    /// Consumes `self` and returns the inner [`BinaryVector`] if `self` is of that variant.
    #[allow(clippy::same_name_method)] // Same as VarBinTypeDowncast
    pub fn into_binary(self) -> BinaryVector {
        if let Vector::Binary(v) = self {
            return v;
        }
        vortex_panic!("Expected BinaryVector, got {self:?}");
    }

    /// Consumes `self` and returns the inner [`StructVector`] if `self` is of that variant.
    pub fn into_struct(self) -> StructVector {
        if let Vector::Struct(v) = self {
            return v;
        }
        vortex_panic!("Expected StructVector, got {self:?}");
    }
}
