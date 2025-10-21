// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Definition of the [`Vector`] type, which represent immutable and fully decompressed (canonical)
//! array data.

use vortex_dtype::Nullability;

use crate::{BoolVector, NullVector, PrimitiveVector, VectorMut, VectorOps, match_each_vector};

/// An enum over all kinds of immutable vectors, which represent fully decompressed (canonical)
/// array data.
///
/// Most of the behavior of `Vector` is described by the [`VectorOps`] trait.
///
/// The mutable equivalent of this type is [`VectorMut`], which implements.
#[derive(Debug, Clone)]
pub enum Vector {
    /// Null
    Null(NullVector),
    /// Bool
    Bool(BoolVector),
    // TODO(connor): Document that this is an enum, not a struct (to represent all possible
    // primitive native generics).
    /// Primitive
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

    fn nullability(&self) -> Nullability {
        match_each_vector!(self, |v| { v.nullability() })
    }

    fn len(&self) -> usize {
        match_each_vector!(self, |v| { v.len() })
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
