// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Integer dtype helpers shared by decimal storage and the Narrow encoding.
//!
//! The existing [`DecimalType`] enum names the six native signed storage widths. These helpers map
//! those widths to integer dtypes, including the built-in wide integer extensions.

use std::sync::LazyLock;

use super::DType;
use super::DecimalType;
use super::Nullability;
use super::PType;
use crate::extension::integer::IntegerWidth;
use crate::extension::integer::WideInteger;

// Wide integer dtypes have only two widths and two nullabilities. Sharing them avoids allocating
// extension metadata and its byte-storage dtype each time a decimal result is constructed.
static WIDE_INTEGER_DTYPES: LazyLock<[[DType; 2]; 2]> = LazyLock::new(|| {
    [IntegerWidth::I128, IntegerWidth::I256].map(|width| {
        [Nullability::NonNullable, Nullability::Nullable]
            .map(|nullability| DType::Extension(WideInteger::new(width, nullability).erased()))
    })
});

/// Returns the signed integer dtype with the requested native width.
pub fn integer_dtype(values_type: DecimalType, nullability: Nullability) -> DType {
    let width_index = match values_type {
        DecimalType::I8 => return DType::Primitive(PType::I8, nullability),
        DecimalType::I16 => return DType::Primitive(PType::I16, nullability),
        DecimalType::I32 => return DType::Primitive(PType::I32, nullability),
        DecimalType::I64 => return DType::Primitive(PType::I64, nullability),
        DecimalType::I128 => 0,
        DecimalType::I256 => 1,
    };
    let nullability_index = match nullability {
        Nullability::NonNullable => 0,
        Nullability::Nullable => 1,
    };

    WIDE_INTEGER_DTYPES[width_index][nullability_index].clone()
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

/// Returns whether a checked signed cast crosses a wide integer dtype.
pub(crate) fn is_wide_integer_cast(source: &DType, target: &DType) -> bool {
    (WideInteger::width(source).is_some() || WideInteger::width(target).is_some())
        && signed_integer_type(source).is_some()
        && signed_integer_type(target).is_some()
}
