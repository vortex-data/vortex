// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::slice::SliceKernel;
use vortex_error::VortexResult;

use crate::EliasFano;
use crate::array::EliasFanoArraySlotsExt;
use crate::ef;
use crate::malformed;

impl SliceKernel for EliasFano {
    fn slice(
        array: ArrayView<'_, Self>,
        range: Range<usize>,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let data = array.data();
        let bounds = ef::slice_bounds(
            &data.window(array.len()),
            data.upper_len(),
            usize::try_from(data.num_elements())?,
            range.clone(),
        )
        .map_err(malformed)?;
        let sliced = data.sliced(&bounds, data.first_rank() + range.start as u64);
        // Executing a lazy child slice can decode the whole child.
        let lower = array.lower().slice(range.clone())?;
        // SAFETY: the geometry is unchanged, and `slice_bounds` checked the window and its samples.
        let sliced = unsafe { EliasFano::new_unchecked(sliced, lower, range.len()) };
        Ok(Some(sliced.into_array()))
    }
}
