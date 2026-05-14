// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::slice::SliceKernel;
use vortex_error::VortexResult;

use crate::ConstantArray;
use crate::Sparse;
use crate::SparseExt as _;

impl SliceKernel for Sparse {
    fn slice(
        array: ArrayView<'_, Self>,
        range: Range<usize>,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let Some(new_patches) = array.patches().slice(range.clone())? else {
            return Ok(Some(
                ConstantArray::new(array.fill_scalar().clone(), range.len()).into_array(),
            ));
        };

        // If the number of values in the sparse array matches the array length, then all
        // values are in fact patches, since patches are sorted this is the correct values.
        if new_patches.array_len() == new_patches.values().len() {
            return Ok(Some(new_patches.into_values()));
        }

        // SAFETY:
        // patches slice will ensure that dtype of patches is unchanged and the indices and
        // values match
        Ok(Some(
            unsafe { Sparse::new_unchecked(new_patches, array.fill_scalar().clone()) }.into_array(),
        ))
    }
}
