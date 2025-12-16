// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_dtype::DType;
use vortex_error::vortex_panic;

use crate::ScalarOps;
use crate::VectorMut;
use crate::binaryview::BinaryScalar;
use crate::binaryview::BinaryType;
use crate::binaryview::BinaryViewScalar;
use crate::binaryview::StringScalar;
use crate::binaryview::StringType;
use crate::bool::BoolScalar;
use crate::decimal::DecimalScalar;
use crate::fixed_size_list::FixedSizeListScalar;
use crate::listview::ListViewScalar;
use crate::match_each_scalar;
use crate::null::NullScalar;
use crate::primitive::PrimitiveScalar;
use crate::struct_::StructScalar;

/// Represents a scalar value of any supported type.
#[derive(Clone, Debug, PartialEq)]
pub enum Scalar {
    /// Null scalars are always null.
    Null(NullScalar),
    /// Boolean scalars hold the boolean value in an [`Option`], where [`None`] represents null.
    Bool(BoolScalar),
    /// Decimal scalars hold the decimal value in a [`DScalar`](crate::decimal::DScalar), or else
    /// [`None`] for null.
    Decimal(DecimalScalar),
    /// Primitive scalars hold the primitive value in a [`PScalar`](crate::primitive::PScalar), or
    /// else [`None`] for null.
    Primitive(PrimitiveScalar),
    /// String scalars hold the string data in a [`BufferString`](vortex_buffer::BufferString), or
    /// else [`None`] for null.
    String(StringScalar),
    /// Binary scalars hold the binary data in a [`ByteBuffer`](vortex_buffer::ByteBuffer), or else
    /// [`None`] for null.
    Binary(BinaryScalar),
    /// Variable-size list scalars hold the list elements in a vector, or else [`None`] for null.
    List(ListViewScalar),
    /// Fixed-size list scalars hold the list elements in a vector, or else [`None`] for null.
    FixedSizeList(FixedSizeListScalar),
    /// Struct scalars are represented as a length-1 struct vector.
    Struct(StructScalar),
}

impl ScalarOps for Scalar {
    fn is_valid(&self) -> bool {
        match_each_scalar!(self, |v| { v.is_valid() })
    }

    fn mask_validity(&mut self, mask: bool) {
        match_each_scalar!(self, |v| { v.mask_validity(mask) })
    }

    fn repeat(&self, n: usize) -> VectorMut {
        match_each_scalar!(self, |v| { v.repeat(n) })
    }
}

impl Scalar {
    /// Creates a zero/default scalar of the given [`DType`].
    ///
    /// For numeric types, this returns 0. For booleans, this returns `false`. For strings/binary,
    /// this returns an empty value. For complex types, this creates a scalar with zero/default
    /// elements.
    ///
    /// # Panics
    ///
    /// Panics if the dtype is [`DType::Null`], since null scalars cannot be zero.
    pub fn zero(dtype: &DType) -> Self {
        match dtype {
            DType::Null => vortex_panic!("Cannot create zero scalar for Null dtype"),
            DType::Bool(_) => BoolScalar::zero().into(),
            DType::Primitive(ptype, _) => PrimitiveScalar::zero(*ptype).into(),
            DType::Decimal(decimal_dtype, _) => DecimalScalar::zero(decimal_dtype).into(),
            DType::Utf8(_) => BinaryViewScalar::<StringType>::zero().into(),
            DType::Binary(_) => BinaryViewScalar::<BinaryType>::zero().into(),
            DType::List(..) => ListViewScalar::zero(dtype).into(),
            DType::FixedSizeList(..) => FixedSizeListScalar::zero(dtype).into(),
            DType::Struct(..) => StructScalar::zero(dtype).into(),
            DType::Extension(ext) => Self::zero(ext.storage_dtype()),
        }
    }

    /// Creates a null scalar of the given [`DType`].
    ///
    /// # Panics
    ///
    /// Panics if the dtype is not nullable.
    pub fn null(dtype: &DType) -> Self {
        if !dtype.is_nullable() {
            vortex_panic!(
                "Cannot create null scalar for non-nullable dtype: {}",
                dtype
            );
        }
        match dtype {
            DType::Null => NullScalar.into(),
            DType::Bool(_) => BoolScalar::null().into(),
            DType::Primitive(ptype, _) => PrimitiveScalar::null(*ptype).into(),
            DType::Decimal(decimal_dtype, _) => DecimalScalar::null(decimal_dtype).into(),
            DType::Utf8(_) => BinaryViewScalar::<StringType>::null().into(),
            DType::Binary(_) => BinaryViewScalar::<BinaryType>::null().into(),
            DType::List(..) => ListViewScalar::null(dtype).into(),
            DType::FixedSizeList(..) => FixedSizeListScalar::null(dtype).into(),
            DType::Struct(..) => StructScalar::null(dtype).into(),
            DType::Extension(ext) => Self::null(ext.storage_dtype()),
        }
    }

    /// Converts the [`Scalar`] into a [`NullScalar`].
    pub fn into_null(self) -> NullScalar {
        if let Scalar::Null(scalar) = self {
            return scalar;
        }
        vortex_panic!("Cannot convert non-null Scalar into NullScalar");
    }

    /// Converts the [`Scalar`] into a [`BoolScalar`].
    pub fn into_bool(self) -> BoolScalar {
        if let Scalar::Bool(scalar) = self {
            return scalar;
        }
        vortex_panic!("Cannot convert non-bool Scalar into BoolScalar");
    }

    /// Converts the [`Scalar`] into a [`BoolScalar`].
    pub fn to_bool(&self) -> &BoolScalar {
        if let Scalar::Bool(scalar) = self {
            return scalar;
        }
        vortex_panic!("Cannot convert non-bool Scalar into BoolScalar");
    }

    /// Converts the [`Scalar`] into a [`DecimalScalar`].
    pub fn into_decimal(self) -> DecimalScalar {
        if let Scalar::Decimal(scalar) = self {
            return scalar;
        }
        vortex_panic!("Cannot convert non-decimal Scalar into DecimalScalar");
    }

    /// Converts the [`Scalar`] into a [`PrimitiveScalar`].
    pub fn into_primitive(self) -> PrimitiveScalar {
        if let Scalar::Primitive(scalar) = self {
            return scalar;
        }
        vortex_panic!("Cannot convert non-primitive Scalar into PrimitiveScalar");
    }

    /// Converts the [`Scalar`] into a [`StringScalar`].
    pub fn into_string(self) -> StringScalar {
        if let Scalar::String(scalar) = self {
            return scalar;
        }
        vortex_panic!("Cannot convert non-string Scalar into StringScalar");
    }

    /// Converts the [`Scalar`] into a [`BinaryScalar`].
    pub fn into_binary(self) -> BinaryScalar {
        if let Scalar::Binary(scalar) = self {
            return scalar;
        }
        vortex_panic!("Cannot convert non-binary Scalar into BinaryScalar");
    }

    /// Converts the [`Scalar`] into a [`ListViewScalar`].
    pub fn into_list(self) -> ListViewScalar {
        if let Scalar::List(scalar) = self {
            return scalar;
        }
        vortex_panic!("Cannot convert non-list Scalar into ListViewScalar");
    }

    /// Converts the [`Scalar`] into a [`FixedSizeListScalar`].
    pub fn into_fixed_size_list(self) -> FixedSizeListScalar {
        if let Scalar::FixedSizeList(scalar) = self {
            return scalar;
        }
        vortex_panic!("Cannot convert non-fixed-size-list Scalar into FixedSizeListScalar");
    }

    /// Converts the [`Scalar`] into a [`StructScalar`].
    pub fn into_struct(self) -> StructScalar {
        if let Scalar::Struct(scalar) = self {
            return scalar;
        }
        vortex_panic!("Cannot convert non-struct Scalar into StructScalar");
    }
}

impl Scalar {
    /// Returns a reference to the inner [`NullScalar`].
    pub fn as_null(&self) -> &NullScalar {
        if let Scalar::Null(scalar) = self {
            return scalar;
        }
        vortex_panic!("Cannot convert non-null Scalar into NullScalar");
    }

    /// Returns a reference to the inner [`BoolScalar`].
    pub fn as_bool(&self) -> &BoolScalar {
        if let Scalar::Bool(scalar) = self {
            return scalar;
        }
        vortex_panic!("Cannot convert non-bool Scalar into BoolScalar");
    }

    /// Returns a reference to the inner [`DecimalScalar`].
    pub fn as_decimal(&self) -> &DecimalScalar {
        if let Scalar::Decimal(scalar) = self {
            return scalar;
        }
        vortex_panic!("Cannot convert non-decimal Scalar into DecimalScalar");
    }

    /// Returns a reference to the inner [`PrimitiveScalar`].
    pub fn as_primitive(&self) -> &PrimitiveScalar {
        if let Scalar::Primitive(scalar) = self {
            return scalar;
        }
        vortex_panic!("Cannot convert non-primitive Scalar into PrimitiveScalar");
    }

    /// Returns a reference to the inner [`StringScalar`].
    pub fn as_string(&self) -> &StringScalar {
        if let Scalar::String(scalar) = self {
            return scalar;
        }
        vortex_panic!("Cannot convert non-string Scalar into StringScalar");
    }

    /// Returns a reference to the inner [`BinaryScalar`].
    pub fn as_binary(&self) -> &BinaryScalar {
        if let Scalar::Binary(scalar) = self {
            return scalar;
        }
        vortex_panic!("Cannot convert non-binary Scalar into BinaryScalar");
    }

    /// Returns a reference to the inner [`ListViewScalar`].
    pub fn as_list(&self) -> &ListViewScalar {
        if let Scalar::List(scalar) = self {
            return scalar;
        }
        vortex_panic!("Cannot convert non-list Scalar into ListViewScalar");
    }

    /// Returns a reference to the inner [`FixedSizeListScalar`].
    pub fn as_fixed_size_list(&self) -> &FixedSizeListScalar {
        if let Scalar::FixedSizeList(scalar) = self {
            return scalar;
        }
        vortex_panic!("Cannot convert non-fixed-size-list Scalar into FixedSizeListScalar");
    }

    /// Returns a reference to the inner [`StructScalar`].
    pub fn as_struct(&self) -> &StructScalar {
        if let Scalar::Struct(scalar) = self {
            return scalar;
        }
        vortex_panic!("Cannot convert non-struct Scalar into StructScalar");
    }
}
