// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::ArrayView;
use crate::IntoArray;
use crate::arrays::runend::RunEnd;
use crate::arrays::runend::array::RunEndArrayExt;
use crate::arrays::runend::array::RunEndArraySlotsExt;
use crate::builtins::ArrayBuiltins;
use crate::scalar::Scalar;
use crate::scalar_fn::fns::fill_null::FillNullReduce;

impl FillNullReduce for RunEnd {
    fn fill_null(
        array: ArrayView<'_, Self>,
        fill_value: &Scalar,
    ) -> VortexResult<Option<ArrayRef>> {
        let new_values = array.values().fill_null(fill_value.clone())?;
        // SAFETY: modifying values only, does not affect ends
        Ok(Some(
            unsafe {
                RunEnd::new_unchecked(
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
