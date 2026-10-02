// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Integer dtype helpers shared by decimal storage and the Narrow encoding.
//!
//! The existing [`DecimalType`] enum names the six native signed storage widths. These helpers map
//! those widths to integer dtypes, including the built-in wide integer extensions.

use super::DType;
use super::DecimalType;
use super::Nullability;
use super::PType;
use crate::extension::integer::IntegerWidth;
use crate::extension::integer::WideInteger;

/// Returns the signed integer dtype with the requested native width.
pub fn integer_dtype(values_type: DecimalType, nullability: Nullability) -> DType {
    let ptype = match values_type {
        DecimalType::I8 => PType::I8,
        DecimalType::I16 => PType::I16,
        DecimalType::I32 => PType::I32,
        DecimalType::I64 => PType::I64,
        DecimalType::I128 => {
            return DType::Extension(WideInteger::new(IntegerWidth::I128, nullability).erased());
        }
        DecimalType::I256 => {
            return DType::Extension(WideInteger::new(IntegerWidth::I256, nullability).erased());
        }
    };

    DType::Primitive(ptype, nullability)
}

/// Returns the native signed width for a primitive or built-in wide integer dtype.
pub fn signed_integer_type(dtype: &DType) -> Option<DecimalType> {
    match dtype {
        DType::Primitive(PType::I8, _) => Some(DecimalType::I8),
        DType::Primitive(PType::I16, _) => Some(DecimalType::I16),
        DType::Primitive(PType::I32, _) => Some(DecimalType::I32),
        DType::Primitive(PType::I64, _) => Some(DecimalType::I64),
        _ => WideInteger::width(dtype).map(IntegerWidth::values_type),
    }
}

/// Returns the byte width of any supported integer dtype.
pub fn integer_byte_width(dtype: &DType) -> Option<usize> {
    if let DType::Primitive(ptype, _) = dtype
        && ptype.is_int()
    {
        return Some(ptype.byte_width());
    }

    signed_integer_type(dtype).map(|values_type| values_type.byte_width())
}
