// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Definition of the [`Vector`] and [`VectorMut`] types, which represent fully decompressed
//! (canonical) array data.

use vortex_dtype::DType;
use vortex_error::vortex_panic;

use crate::{
    BoolVector, BoolVectorMut, NullVector, NullVectorMut, PrimitiveVector, PrimitiveVectorMut,
};

/// Helper macros for working with the different variants of [`Vector`] and [`VectorMut`].
///
/// All macros are exported at the crate level with `#[macro_use]`.
mod macros;

/// Definition and implementation of [`VectorOps`](ops::VectorOps) and
/// [`VectorMutOps`](ops::VectorMutOps) for [`Vector`] and [`VectorMut`], respecitively.
pub(super) mod ops;

/// An enum over all kinds of immutable vectors, which represent fully decompressed (canonical)
/// array data.
///
/// Most of the behavior of `Vector` is described by the [`VectorOps`] trait.
///
/// The mutable equivalent of this type is [`VectorMut`], which implements.
///
/// [`VectorOps`]: crate::VectorOps
#[derive(Debug, Clone)]
pub enum Vector {
    /// Null
    Null(NullVector),
    /// Bool
    Bool(BoolVector),
    /// Primitive
    ///
    /// TODO(connor): Document that this is an enum, not a struct (to represent all possible
    /// primitive native generics).
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

/// An enum over all kinds of mutable vectors, which represent fully decompressed (canonical) array
/// data.
///
/// Most of the behavior of `VectorMut` is described by the [`VectorMutOps`] trait.
///
/// The immutable equivalent of this type is [`Vector`].
///
/// [`VectorMutOps`]: crate::VectorMutOps
#[derive(Debug, Clone)]
pub enum VectorMut {
    /// Null
    Null(NullVectorMut),
    /// Bool
    Bool(BoolVectorMut),
    /// Primitive
    Primitive(PrimitiveVectorMut),
}

impl VectorMut {
    /// Create a new mutable vector with the given capacity and dtype.
    pub fn with_capacity(capacity: usize, dtype: &DType) -> Self {
        match dtype {
            DType::Null => NullVectorMut::new(0).into(), // `NullVector` has `usize::MAX` capacity.
            DType::Bool(n) => BoolVectorMut::with_capacity(capacity, *n).into(),
            DType::Primitive(ptype, nullability) => {
                PrimitiveVectorMut::with_capacity(capacity, *ptype, *nullability).into()
            }
            _ => vortex_panic!("Unsupported dtype for VectorMut"),
        }
    }
}
