// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::IntoArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::scalar::Scalar;
use vortex_array::scalar_fn::fns::fill_null::FillNullReduce;
use vortex_error::VortexResult;

use crate::RunEnd;
use crate::RunEndData;

impl FillNullReduce for RunEnd {
    fn fill_null(
        array: ArrayView<'_, Self>,
        fill_value: &Scalar,
    ) -> VortexResult<Option<ArrayRef>> {
        let new_values = array.values().fill_null(fill_value.clone())?;
        // SAFETY: modifying values only, does not affect ends
        Ok(Some(
            unsafe {
                RunEndData::new_unchecked(
                    array.ends().clone(),
                    new_values,
                    array.offset(),
                    array.len(),
                )
            }
            .into_array(),
        ))
    }
}
