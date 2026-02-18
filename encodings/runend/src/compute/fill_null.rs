// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::expr::FillNullReduce;
use vortex_array::scalar::Scalar;
use vortex_error::VortexResult;

use crate::RunEndArray;
use crate::RunEndArrayParts;
use crate::RunEndVTable;

impl FillNullReduce for RunEndVTable {
    fn fill_null(array: &RunEndArray, fill_value: &Scalar) -> VortexResult<Option<ArrayRef>> {
        let RunEndArrayParts { values, ends } = array.clone().into_parts();
        let new_values = values.fill_null(fill_value.clone())?;
        // SAFETY: modifying values only, does not affect ends
        Ok(Some(
            unsafe { RunEndArray::new_unchecked(ends, new_values, array.offset(), array.len()) }
                .into_array(),
        ))
    }
}
