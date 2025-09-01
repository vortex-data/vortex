// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::arrays::ConstantArray;
use vortex_array::compute::{LikeKernel, LikeKernelAdapter, LikeOptions, like};
use vortex_array::{Array, ArrayRef, IntoArray, register_kernel};
use vortex_error::VortexResult;

use crate::{DictArray, DictVTable};

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
            unsafe {
                Ok(Some(
                    DictArray::new_unchecked(array.codes().clone(), values).into_array(),
                ))
            }
        } else {
            Ok(None)
        }
    }
}

register_kernel!(LikeKernelAdapter(DictVTable).lift());
