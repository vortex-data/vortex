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
use crate::scalar_fn::fns::cast::CastReduce;

impl CastReduce for Widen {
    fn cast(array: ArrayView<'_, Self>, dtype: &DType) -> VortexResult<Option<ArrayRef>> {
        let values = array.values();
        let (DType::Primitive(target_ptype, _), DType::Primitive(values_ptype, _)) =
            (dtype, values.dtype())
        else {
            return Ok(None);
        };
        if !target_ptype.is_int() || target_ptype.is_signed_int() != values_ptype.is_signed_int() {
            return Ok(None);
        }

        // The cast of the values checks every value and every null against the target.
        if target_ptype.byte_width() <= values_ptype.byte_width() {
            return values.cast(dtype.clone()).map(Some);
        }

        // A wider target holds every value, so the values keep their type. They take the target
        // nullability, so that a cast to a non-nullable target still fails on a null.
        let values = values.cast(values.dtype().with_nullability(dtype.nullability()))?;
        wrap_values(values, dtype.clone()).map(Some)
    }
}
