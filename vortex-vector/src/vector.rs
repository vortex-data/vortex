// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Definition of the [`Vector`] type, which represents immutable and fully decompressed (canonical)
//! array data.
//!
//! [`Vector`] can be transformed into the [`VectorMut`] type if it is owned.

use vortex_error::vortex_panic;

use crate::macros::match_each_vector;
use crate::{BoolVector, NullVector, PrimitiveVector, VectorMut, VectorOps};

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
    /// Primitive vectors.
    ///
    /// Note that [`PrimitiveVector`] is an enum over the different possible (generic)
    /// [`PVector<T>`](crate::PVector)s. See the documentation for more information.
    Primitive(PrimitiveVector),
    // Decimal
    // Decimal(DecimalVector),
    // String
    // String(StringVector),
    // Binary
    // Binary(BinaryVector),
    // List
    // List(ListVector),
    // FixedList
    // FixedList(FixedListVector),
    // Struct
    // Struct(StructVector),
    // Extension
    // Extension(ExtensionVector),
}

impl VectorOps for Vector {
    type Mutable = VectorMut;

    fn len(&self) -> usize {
        match_each_vector!(self, |v| { v.len() })
    }

    fn validity(&self) -> &vortex_mask::Mask {
        match_each_vector!(self, |v| { v.validity() })
    }

    fn try_into_mut(self) -> Result<Self::Mutable, Self>
    where
        Self: Sized,
    {
        match_each_vector!(self, |v| {
            v.try_into_mut().map(VectorMut::from).map_err(Vector::from)
        })
    }
}

impl Vector {
    /// Consumes `self` and returns the inner `NullVector` if `self` is of that variant.
    pub fn into_null(self) -> NullVector {
        if let Vector::Null(v) = self {
            return v;
        }
        vortex_panic!("Expected NullVector, got {self:?}");
    }

    /// Consumes `self` and returns the inner `BoolVector` if `self` is of that variant.
    pub fn into_bool(self) -> BoolVector {
        if let Vector::Bool(v) = self {
            return v;
        }
        vortex_panic!("Expected BoolVector, got {self:?}");
    }

    /// Consumes `self` and returns the inner `PrimitiveVector` if `self` is of that variant.
    pub fn into_primitive(self) -> PrimitiveVector {
        if let Vector::Primitive(v) = self {
            return v;
        }
        vortex_panic!("Expected PrimitiveVector, got {self:?}");
    }
}
