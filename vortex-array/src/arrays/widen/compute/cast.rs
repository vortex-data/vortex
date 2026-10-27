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

        let storage_ptype = match storage_ptype(*values_ptype, *target_ptype) {
            Some(storage_ptype) if target_ptype.byte_width() > storage_ptype.byte_width() => {
                storage_ptype
            }
            // A float target, or an integer target no wider than the storage type, goes to the
            // values, whose cast checks every value and every null.
            _ => return values.cast(dtype.clone()).map(Some),
        };

        // A wider integer target holds every value of the storage type, so the values stay narrow.
        // They take the target nullability, so that a cast to a non-nullable target still fails on
        // a null.
        let values = values.cast(DType::Primitive(storage_ptype, *nullability))?;
        wrap_values(values, dtype.clone()).map(Some)
    }
}

/// Returns the integer type with the signedness of `target` that the values are cast to before a
/// wider integer cast, or `None` when `target` is a float.
///
/// A signed type holds an unsigned type of the same width only up to its maximum, so an unsigned
/// `values` type and a signed target give the next wider signed type. A signed `values` type and an
/// unsigned target give the unsigned type of the same width, because a negative value fits no
/// unsigned type, and the cast of the values rejects it.
fn storage_ptype(values: PType, target: PType) -> Option<PType> {
    if target.is_float() {
        return None;
    }

    let storage = match (values.is_signed_int(), target.is_signed_int()) {
        // `validate_dtypes` keeps the values narrower than 64 bits.
        (false, true) => match values {
            PType::U8 => PType::I16,
            PType::U16 => PType::I32,
            _ => PType::I64,
        },
        (true, false) => values.to_unsigned(),
        _ => values,
    };

    Some(storage)
}
