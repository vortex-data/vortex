// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::slice::SliceKernel;
use vortex_error::VortexResult;

use crate::alp_rd::ALPRD;

impl SliceKernel for ALPRD {
    fn slice(
        array: ArrayView<'_, Self>,
        range: Range<usize>,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let left_parts_exceptions = array
            .left_parts_patches()
            .map(|patches| patches.slice(range.clone()))
            .transpose()?
            .flatten();

        // SAFETY: slicing components does not change the encoded values
        Ok(Some(unsafe {
            ALPRD::new_unchecked(
                array.dtype().clone(),
                array.left_parts().slice(range.clone())?,
                array.left_parts_dictionary().clone(),
                array.right_parts().slice(range)?,
                array.right_bit_width(),
                left_parts_exceptions,
            )
            .into_array()
        }))
    }
}
