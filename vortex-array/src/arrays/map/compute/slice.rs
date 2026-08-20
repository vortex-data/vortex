// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::array::ArrayView;
use crate::arrays::ListView;
use crate::arrays::map::Map;
use crate::arrays::map::MapArrayExt;
use crate::arrays::map::MapArraySlotsExt;
use crate::arrays::map::compute::rebuild_map_from_array;
use crate::arrays::slice::SliceKernel;
use crate::arrays::slice::SliceReduce;
use crate::executor::ExecutionCtx;

impl SliceReduce for Map {
    fn slice(array: ArrayView<'_, Self>, range: Range<usize>) -> VortexResult<Option<ArrayRef>> {
        let Some(sliced_entries) =
            <ListView as SliceReduce>::slice(array.entries().as_::<ListView>(), range)?
        else {
            return Ok(None);
        };

        rebuild_map_from_array(array.map_dtype().clone(), sliced_entries).map(Some)
    }
}

impl SliceKernel for Map {
    fn slice(
        array: ArrayView<'_, Self>,
        range: Range<usize>,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        <Self as SliceReduce>::slice(array, range)
    }
}
