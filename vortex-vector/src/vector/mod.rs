// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_dtype::DType;
use vortex_error::vortex_panic;

use crate::{
    BoolVector, BoolVectorMut, NullVector, NullVectorMut, PrimitiveVector, PrimitiveVectorMut,
};

mod macros;
pub(super) mod ops;

/// An enum over all vector types.
pub enum Vector {
    /// Null
    Null(NullVector),
    /// Bool
    Bool(BoolVector),
    /// Primitive
    ///
    /// TODO(connor): Document that this is an enum, not a struct.
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

/// An enum over all mutable vector types.
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
            DType::Null => NullVectorMut::new(0).into(),
            DType::Bool(n) => BoolVectorMut::with_capacity(capacity, *n).into(),
            DType::Primitive(ptype, nullability) => {
                PrimitiveVectorMut::with_capacity(capacity, *ptype, *nullability).into()
            }
            _ => vortex_panic!("Unsupported dtype for VectorMut"),
        }
    }
}
