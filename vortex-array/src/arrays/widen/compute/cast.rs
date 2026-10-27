// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::array::ArrayView;
use crate::arrays::Widen;
use crate::arrays::widen::WidenArraySlotsExt;
use crate::arrays::widen::compute::wrap_values;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::dtype::PType;
use crate::scalar_fn::fns::cast::CastReduce;

impl CastReduce for Widen {
    fn cast(array: ArrayView<'_, Self>, dtype: &DType) -> VortexResult<Option<ArrayRef>> {
        let values = array.values();
        let (DType::Primitive(target_ptype, nullability), DType::Primitive(values_ptype, _)) =
            (dtype, values.dtype())
        else {
            return Ok(None);
        };

        let cast_ptype = match cast_values_ptype(*values_ptype, *target_ptype) {
            Some(cast_ptype) if target_ptype.byte_width() > cast_ptype.byte_width() => cast_ptype,
            // A float target, or an integer target no wider than `cast_ptype`, goes to the values,
            // whose cast checks every value and every null.
            _ => return values.cast(dtype.clone()).map(Some),
        };

        // A wider integer target holds every value of `cast_ptype`, so the values stay narrow.
        // They take the target nullability, so that a cast to a non-nullable target still fails on
        // a null.
        let values = values.cast(DType::Primitive(cast_ptype, *nullability))?;
        wrap_values(values, dtype.clone()).map(Some)
    }
}

/// Returns the type that the values take when the `Widen` is cast to the integer type `target`, or
/// `None` for a float target.
///
/// The result has the signedness of `target`. An unsigned type fits a signed type only one width
/// up, so unsigned values with a signed target widen one step. Signed values with an unsigned
/// target keep their width, and the cast of the values rejects a negative value.
fn cast_values_ptype(values: PType, target: PType) -> Option<PType> {
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
