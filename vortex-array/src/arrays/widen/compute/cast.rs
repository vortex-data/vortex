// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Cast reductions for decimal and primitive [`Widen`] arrays.
//!
//! Same-scale decimal casts retarget the `Widen` when the child already fits the target. Primitive
//! casts can retain a `Widen` for wider integer targets, after matching the child's signedness to the
//! target. Both paths leave required conversions and nullability checks to the child's cast.

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::array::ArrayView;
use crate::arrays::Widen;
use crate::arrays::widen::WidenArraySlotsExt;
use crate::arrays::widen::compute::wrap_values;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::dtype::DecimalDType;
use crate::dtype::PType;
use crate::scalar_fn::fns::cast::CastReduce;

impl CastReduce for Widen {
    fn cast(array: ArrayView<'_, Self>, dtype: &DType) -> VortexResult<Option<ArrayRef>> {
        let values = array.values();

        match values.dtype() {
            DType::Decimal(values_dtype, _) => cast_decimal(values, *values_dtype, dtype),
            DType::Primitive(values_ptype, _) => cast_primitive(values, *values_ptype, dtype),
            _ => Ok(None),
        }
    }
}

fn cast_decimal(
    values: &ArrayRef,
    values_dtype: DecimalDType,
    dtype: &DType,
) -> VortexResult<Option<ArrayRef>> {
    if let DType::Decimal(target_dtype, _) = dtype
        && values_dtype.scale() == target_dtype.scale()
        && values_dtype.canonical_values_type() < target_dtype.canonical_values_type()
    {
        // The child's declared precision proves that every value fits the target. Only nullability
        // may still need a check, so leave the width conversion to Widen execution.
        let values = values.cast(values.dtype().with_nullability(dtype.nullability()))?;

        return wrap_values(values, dtype.clone()).map(Some);
    }

    // Rescaling, narrower targets, and primitive conversions still need the child's cast. The child
    // has the parent's values, scale, and nulls, so these conversions preserve the logical result.
    match dtype {
        DType::Decimal(..) | DType::Primitive(..) => values.cast(dtype.clone()).map(Some),
        _ => Ok(None),
    }
}

fn cast_primitive(
    values: &ArrayRef,
    values_ptype: PType,
    dtype: &DType,
) -> VortexResult<Option<ArrayRef>> {
    let DType::Primitive(target_ptype, nullability) = dtype else {
        return Ok(None);
    };

    let cast_ptype = match values_ptype_for_cast(values_ptype, *target_ptype) {
        Some(cast_ptype) if target_ptype.byte_width() > cast_ptype.byte_width() => cast_ptype,
        // A float needs value conversion, and an integer target no wider than the converted child
        // cannot retain a Widen. Casting directly also checks overflow and nullability.
        _ => return values.cast(dtype.clone()).map(Some),
    };

    // Match the child's signedness and nullability to the target. If both already match, the cast
    // returns the original child. Otherwise, it converts the child and checks negative values and
    // nulls as needed. Widen handles the remaining width increase because every child value fits.
    let values = values.cast(DType::Primitive(cast_ptype, *nullability))?;

    wrap_values(values, dtype.clone()).map(Some)
}

/// Returns the type that the values take when the `Widen` is cast to the integer type `target`, or
/// `None` for a float target.
///
/// The result has the signedness of `target`. An unsigned type fits a signed type only one width
/// up, so unsigned values with a signed target widen one step. Signed values with an unsigned
/// target keep their width, and the cast of the values rejects a negative value.
fn values_ptype_for_cast(values: PType, target: PType) -> Option<PType> {
    if target.is_float() {
        return None;
    }

    let cast_ptype = match (values.is_signed_int(), target.is_signed_int()) {
        // `validate_dtypes` keeps the values narrower than 64 bits.
        (false, true) => match values {
            PType::U8 => PType::I16,
            PType::U16 => PType::I32,
            _ => PType::I64,
        },
        (true, false) => values.to_unsigned(),
        _ => values,
    };

    Some(cast_ptype)
}
