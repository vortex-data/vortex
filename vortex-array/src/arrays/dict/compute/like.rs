// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use super::DictArray;
use super::DictVTable;
use crate::Array;
use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::ConstantArray;
use crate::compute::LikeKernel;
use crate::compute::LikeKernelAdapter;
use crate::compute::LikeOptions;
use crate::compute::like;
use crate::register_kernel;

impl LikeKernel for DictVTable {
    fn like(
        &self,
        array: &DictArray,
        pattern: &dyn Array,
        options: LikeOptions,
    ) -> VortexResult<Option<ArrayRef>> {
        // if we have more values than codes, it is faster to canonicalise first.
        if array.values().len() > array.codes().len() {
            return Ok(None);
        }
        if let Some(pattern) = pattern.as_constant() {
            let pattern = ConstantArray::new(pattern, array.values().len()).into_array();
            let values = like(array.values(), &pattern, options)?;

            // SAFETY: LIKE preserves the len of the values, so codes are still pointing at
            //  valid positions.
            // Preserve all_values_referenced since codes are unchanged
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

register_kernel!(LikeKernelAdapter(DictVTable).lift());
