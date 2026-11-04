// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::array::ArrayView;
use crate::arrays::Widen;
use crate::arrays::widen::WidenArraySlotsExt;
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
        if !fill_value.dtype().eq_ignore_nullability(array.dtype()) {
            return Ok(None);
        }

        let values = array.values();
        let dtype = match values.dtype() {
            DType::Primitive(..) => {
                let Some(ptype) = ptype_holding_values_and_scalar(array, fill_value) else {
                    return Ok(None);
                };

                DType::Primitive(ptype, values.dtype().nullability())
            }
            DType::Decimal(..)
                if constant_fits_values(fill_value, values.dtype()) == Some(true) =>
            {
                values.dtype().clone()
            }
            // Decimal fills outside the child precision use the canonical fill kernel.
            _ => return Ok(None),
        };
        let widened = values.cast(dtype.clone())?;
        let fill_value =
            fill_value.cast(&dtype.with_nullability(fill_value.dtype().nullability()))?;
        let filled = widened.fill_null(fill_value)?;
        let dtype = array.dtype().with_nullability(filled.dtype().nullability());

        wrap_values(filled, dtype).map(Some)
    }
}
