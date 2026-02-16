// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use super::DictArray;
use super::DictVTable;
use crate::Array;
use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::ConstantArray;
use crate::arrays::ScalarFnArrayExt;
use crate::expr::Like;
use crate::expr::LikeOptions;
use crate::expr::LikeReduce;
use crate::optimizer::ArrayOptimizer;

impl LikeReduce for DictVTable {
    fn like(
        array: &DictArray,
        pattern: &dyn Array,
        options: LikeOptions,
    ) -> VortexResult<Option<ArrayRef>> {
        // If we have more values than codes, it is faster to canonicalize first.
        if array.values().len() > array.codes().len() {
            return Ok(None);
        }
        if let Some(pattern) = pattern.as_constant() {
            let pattern = ConstantArray::new(pattern, array.values().len()).into_array();

            let values = Like
                .try_new_array(array.len(), options, [array.values().clone(), pattern])?
                .optimize()?;

            // SAFETY: LIKE preserves the len of the values, so codes are still pointing at
            //  valid positions.
            // Preserve all_values_referenced since codes are unchanged.
            unsafe {
                Ok(Some(
                    DictArray::new_unchecked(array.codes().clone(), values)
                        .set_all_values_referenced(array.has_all_values_referenced())
                        .into_array(),
                ))
            }
        } else {
            Ok(None)
        }
    }
}
