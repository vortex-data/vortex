// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::vortex_panic;

use crate::ScalarOps;
use crate::VectorMut;
use crate::binaryview::BinaryScalar;
use crate::binaryview::StringScalar;
use crate::bool::BoolScalar;
use crate::decimal::DecimalScalar;
use crate::fixed_size_list::FixedSizeListScalar;
use crate::listview::ListViewScalar;
use crate::match_each_scalar;
use crate::null::NullScalar;
use crate::primitive::PrimitiveScalar;
use crate::struct_::StructScalar;

/// Represents a scalar value of any supported type.
#[derive(Clone, Debug)]
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

    fn mask_validity(&mut self, mask: bool) {
        match_each_scalar!(self, |v| { v.mask_validity(mask) })
    }

    fn repeat(&self, n: usize) -> VectorMut {
        match_each_scalar!(self, |v| { v.repeat(n) })
    }
}

impl Scalar {
    /// Converts the `Scalar` into a `NullScalar`.
    pub fn into_null(self) -> NullScalar {
        if let Scalar::Null(scalar) = self {
            return scalar;
        }
        vortex_panic!("Cannot convert non-null Scalar into NullScalar");
    }

    /// Converts the `Scalar` into a `BoolScalar`.
    pub fn into_bool(self) -> BoolScalar {
        if let Scalar::Bool(scalar) = self {
            return scalar;
        }
        vortex_panic!("Cannot convert non-bool Scalar into BoolScalar");
    }

    /// Converts the `Scalar` into a `BoolScalar`.
    pub fn to_bool(&self) -> &BoolScalar {
        if let Scalar::Bool(scalar) = self {
            return scalar;
        }
        vortex_panic!("Cannot convert non-bool Scalar into BoolScalar");
    }

    /// Converts the `Scalar` into a `DecimalScalar`.
    pub fn into_decimal(self) -> DecimalScalar {
        if let Scalar::Decimal(scalar) = self {
            return scalar;
        }
        vortex_panic!("Cannot convert non-decimal Scalar into DecimalScalar");
    }

    /// Converts the `Scalar` into a `PrimitiveScalar`.
    pub fn into_primitive(self) -> PrimitiveScalar {
        if let Scalar::Primitive(scalar) = self {
            return scalar;
        }
        vortex_panic!("Cannot convert non-primitive Scalar into PrimitiveScalar");
    }

    /// Converts the `Scalar` into a `StringScalar`.
    pub fn into_string(self) -> StringScalar {
        if let Scalar::String(scalar) = self {
            return scalar;
        }
        vortex_panic!("Cannot convert non-string Scalar into StringScalar");
    }

    /// Converts the `Scalar` into a `BinaryScalar`.
    pub fn into_binary(self) -> BinaryScalar {
        if let Scalar::Binary(scalar) = self {
            return scalar;
        }
        vortex_panic!("Cannot convert non-binary Scalar into BinaryScalar");
    }

    /// Converts the `Scalar` into a `ListViewScalar`.
    pub fn into_list(self) -> ListViewScalar {
        if let Scalar::List(scalar) = self {
            return scalar;
        }
        vortex_panic!("Cannot convert non-list Scalar into ListViewScalar");
    }

    /// Converts the `Scalar` into a `FixedSizeListScalar`.
    pub fn into_fixed_size_list(self) -> FixedSizeListScalar {
        if let Scalar::FixedSizeList(scalar) = self {
            return scalar;
        }
        vortex_panic!("Cannot convert non-fixed-size-list Scalar into FixedSizeListScalar");
    }

    /// Converts the `Scalar` into a `StructScalar`.
    pub fn into_struct(self) -> StructScalar {
        if let Scalar::Struct(scalar) = self {
            return scalar;
        }
        vortex_panic!("Cannot convert non-struct Scalar into StructScalar");
    }
}

impl Scalar {
    /// Converts the `Scalar` into a `NullScalar`.
    pub fn as_null(&self) -> &NullScalar {
        if let Scalar::Null(scalar) = self {
            return scalar;
        }
        vortex_panic!("Cannot convert non-null Scalar into NullScalar");
    }

    /// Converts the `Scalar` into a `BoolScalar`.
    pub fn as_bool(&self) -> &BoolScalar {
        if let Scalar::Bool(scalar) = self {
            return scalar;
        }
        vortex_panic!("Cannot convert non-bool Scalar into BoolScalar");
    }

    /// Converts the `Scalar` into a `DecimalScalar`.
    pub fn as_decimal(&self) -> &DecimalScalar {
        if let Scalar::Decimal(scalar) = self {
            return scalar;
        }
        vortex_panic!("Cannot convert non-decimal Scalar into DecimalScalar");
    }

    /// Converts the `Scalar` into a `PrimitiveScalar`.
    pub fn as_primitive(&self) -> &PrimitiveScalar {
        if let Scalar::Primitive(scalar) = self {
            return scalar;
        }
        vortex_panic!("Cannot convert non-primitive Scalar into PrimitiveScalar");
    }

    /// Converts the `Scalar` into a `StringScalar`.
    pub fn as_string(&self) -> &StringScalar {
        if let Scalar::String(scalar) = self {
            return scalar;
        }
        vortex_panic!("Cannot convert non-string Scalar into StringScalar");
    }

    /// Converts the `Scalar` into a `BinaryScalar`.
    pub fn as_binary(&self) -> &BinaryScalar {
        if let Scalar::Binary(scalar) = self {
            return scalar;
        }
        vortex_panic!("Cannot convert non-binary Scalar into BinaryScalar");
    }

    /// Converts the `Scalar` into a `ListViewScalar`.
    pub fn as_list(&self) -> &ListViewScalar {
        if let Scalar::List(scalar) = self {
            return scalar;
        }
        vortex_panic!("Cannot convert non-list Scalar into ListViewScalar");
    }

    /// Converts the `Scalar` into a `FixedSizeListScalar`.
    pub fn as_fixed_size_list(&self) -> &FixedSizeListScalar {
        if let Scalar::FixedSizeList(scalar) = self {
            return scalar;
        }
        vortex_panic!("Cannot convert non-fixed-size-list Scalar into FixedSizeListScalar");
    }

    /// Converts the `Scalar` into a `StructScalar`.
    pub fn as_struct(&self) -> &StructScalar {
        if let Scalar::Struct(scalar) = self {
            return scalar;
        }
        vortex_panic!("Cannot convert non-struct Scalar into StructScalar");
    }
}
