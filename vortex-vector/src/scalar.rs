// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::binaryview::{BinaryScalar, StringScalar};
use crate::bool::BoolScalar;
use crate::decimal::DecimalScalar;
use crate::fixed_size_list::FixedSizeListScalar;
use crate::listview::ListViewScalar;
use crate::null::NullScalar;
use crate::primitive::PrimitiveScalar;
use crate::struct_::StructScalar;
use crate::{ScalarOps, VectorMut, match_each_scalar};

/// Represents a scalar value of any supported type.
pub enum Scalar {
    /// Null scalars are always null.
    Null(NullScalar),
    /// Boolean scalars hold the boolean value in an Option, where None represents null.
    Bool(BoolScalar),
    /// Decimal scalars hold the decimal value in a DScalar, or else None for null.
    Decimal(DecimalScalar),
    /// Primitive scalars hold the primitive value in a PScalar, or else None for null.
    Primitive(PrimitiveScalar),
    /// String scalars hold the string data in a BufferString, or else None for null.
    String(StringScalar),
    /// Binary scalars hold the binary data in a ByteBuffer, or else None for null.
    Binary(BinaryScalar),
    /// Variable-size list scalars hold the list elements in a vector, or else None for null.
    List(ListViewScalar),
    /// Fixed-size list scalars hold the list elements in a vector, or else None for null.
    FixedSizeList(FixedSizeListScalar),
    /// Struct scalars are represented as a length-1 struct vector.
    Struct(StructScalar),
}

impl ScalarOps for Scalar {
    fn is_valid(&self) -> bool {
        match_each_scalar!(self, |v| { v.is_valid() })
    }

    fn repeat(&self, n: usize) -> VectorMut {
        match_each_scalar!(self, |v| { v.repeat(n) })
    }
}
