// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::array::ArrayView;
use crate::arrays::Widen;
use crate::arrays::widen::WidenArraySlotsExt;
use crate::arrays::widen::compute::compare::constant_fits_child;
use crate::arrays::widen::compute::wrap_values;
use crate::builtins::ArrayBuiltins;
use crate::scalar::Scalar;
use crate::scalar_fn::fns::fill_null::FillNullReduce;

impl FillNullReduce for Widen {
    fn fill_null(
        array: ArrayView<'_, Self>,
        fill_value: &Scalar,
    ) -> VortexResult<Option<ArrayRef>> {
        let values = array.values();
        if constant_fits_child(fill_value, values.dtype()) != Some(true) {
            return Ok(None);
        }

        let fill_dtype = values
            .dtype()
            .with_nullability(fill_value.dtype().nullability());
        let filled = values.fill_null(fill_value.cast(&fill_dtype)?)?;
        let dtype = array.dtype().with_nullability(filled.dtype().nullability());

        wrap_values(filled, dtype).map(Some)
    }
}
