// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::decimal::DScalar;
use crate::primitive::PScalar;
use crate::struct_::StructScalar;
use crate::{ScalarOps, Vector, VectorMut};
use vortex_buffer::{BufferString, ByteBuffer};
use crate::bool::BoolVectorMut;
use crate::null::NullVectorMut;

pub enum Scalar {
    /// Null scalars are always null.
    Null,
    /// Boolean scalars hold the boolean value in an Option, where None represents null.
    Bool(Option<bool>),
    /// Decimal scalars hold the decimal value in a DScalar, or else None for null.
    Decimal(DScalar),
    /// Primitive scalars hold the primitive value in a PScalar, or else None for null.
    Primitive(PScalar),
    /// String scalars hold the string data in a BufferString, or else None for null.
    String(Option<BufferString>),
    /// Binary scalars hold the binary data in a ByteBuffer, or else None for null.
    Binary(Option<ByteBuffer>),
    /// Variable-size list scalars hold the list elements in a vector, or else None for null.
    List(Option<Vector>),
    /// Fixed-size list scalars hold the list elements in a vector, or else None for null.
    FixedSizeList(Option<Vector>),
    /// Struct scalars are represented as a length-1 struct vector.
    Struct(StructScalar),
}

impl ScalarOps for Scalar {
    fn is_valid(&self) -> bool {
        match self {
            Scalar::Null => false
            Scalar::Bool(v) => v.is_some(),
            Scalar::Decimal(v) => v.is_valid(),
            Scalar::Primitive(v) => v.is_valid(),
            Scalar::String(v) => v.is_some(),
            Scalar::Binary(v) => v.is_some(),
            Scalar::List(v) => v.is_some(),
            Scalar::FixedSizeList(v) => v.is_some(),
            Scalar::Struct(v) => v.is_valid(),
        }
    }

    fn repeat(&self, n: usize) -> VectorMut {
        match self {
            Scalar::Null => NullVectorMut::new(n).into(),
            Scalar::Bool(v) => {
                let mut vec = BoolVectorMut::with_capacity(n);
                vec.append_
                vec.into()
            },
            Scalar::Decimal(v) => v.repeat(n),
            Scalar::Primitive(v) => v.repeat(n),
            Scalar::String(v) => VectorMut::string_from_option(v.clone()).repeat(n),
            Scalar::Binary(v) => VectorMut::binary_from_option(v.clone()).repeat(n),
            Scalar::List(v) => match v {
                Some(vec) => vec.repeat(n),
                None => VectorMut::list_null(n),
            },
            Scalar::FixedSizeList(v) => match v {
                Some(vec) => vec.repeat(n),
                None => VectorMut::fixed_size_list_null(n),
            },
            Scalar::Struct(v) => v.repeat(n),
        }
    }
}
