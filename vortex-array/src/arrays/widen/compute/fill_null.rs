// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::array::ArrayView;
use crate::arrays::Widen;
use crate::arrays::widen::WidenArraySlotsExt;
use crate::arrays::widen::compute::cast_keeping_nullability;
use crate::arrays::widen::compute::constant_fits_values;
use crate::arrays::widen::compute::ptype_holding_values_and_scalar;
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
        if matches!(array.values().dtype(), DType::Decimal(..)) {
            // A fill value within the child precision fills the child, so the child holds exactly
            // the values of the filled `Widen` and its label stays true. A fill value outside the
            // child precision needs a wider child, which only the values can choose, so the rule
            // declines and execution fills the canonical array.
            if !fill_value.dtype().eq_ignore_nullability(array.dtype())
                || constant_fits_values(fill_value, array.values().dtype()) != Some(true)
            {
                return Ok(None);
            }

            let fill_dtype = array
                .values()
                .dtype()
                .with_nullability(fill_value.dtype().nullability());
            let filled = array.values().fill_null(fill_value.cast(&fill_dtype)?)?;
            let dtype = array.dtype().with_nullability(filled.dtype().nullability());

            return wrap_values(filled, dtype).map(Some);
        }

        let Some(ptype) = ptype_holding_values_and_scalar(array, fill_value)? else {
            return Ok(None);
        };

        // The values widen to a type that holds the fill value. When only the logical type holds
        // it, the result is a plain array of the logical type.
        let widened = cast_keeping_nullability(array.values(), ptype)?;
        let fill_value =
            fill_value.cast(&DType::Primitive(ptype, fill_value.dtype().nullability()))?;
        let filled = widened.fill_null(fill_value)?;
        let dtype = array.dtype().with_nullability(filled.dtype().nullability());

        wrap_values(filled, dtype).map(Some)
    }
}
