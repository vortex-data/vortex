// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::array::ArrayView;
use crate::arrays::Widen;
use crate::arrays::widen::WidenArraySlotsExt;
use crate::arrays::widen::compute::integer_storage_type;
use crate::arrays::widen::compute::wrap_values;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::scalar::Scalar;
use crate::scalar_fn::fns::fill_null::FillNullReduce;

impl FillNullReduce for Widen {
    fn fill_null(
        array: ArrayView<'_, Self>,
        fill_value: &Scalar,
    ) -> VortexResult<Option<ArrayRef>> {
        let Some(ptype) = integer_storage_type(array, fill_value)? else {
            return Ok(None);
        };

        // The values widen to a type that holds the fill value. When only the logical type holds
        // it, the result is a plain array of the logical type.
        let values = array.values();
        let widened = values.cast(DType::Primitive(ptype, values.dtype().nullability()))?;
        let fill_value =
            fill_value.cast(&DType::Primitive(ptype, fill_value.dtype().nullability()))?;
        let filled = widened.fill_null(fill_value)?;
        let dtype = array.dtype().with_nullability(filled.dtype().nullability());

        wrap_values(filled, dtype).map(Some)
    }
}
