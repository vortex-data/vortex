// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::array::ArrayView;
use crate::arrays::ListView;
use crate::arrays::dict::TakeExecute;
use crate::arrays::dict::TakeReduce;
use crate::arrays::map::Map;
use crate::arrays::map::MapArrayExt;
use crate::arrays::map::MapArraySlotsExt;
use crate::arrays::map::compute::rebuild_map_from_array;
use crate::executor::ExecutionCtx;

impl TakeReduce for Map {
    fn take(array: ArrayView<'_, Self>, indices: &ArrayRef) -> VortexResult<Option<ArrayRef>> {
        let Some(entries) =
            <ListView as TakeReduce>::take(array.entries().as_::<ListView>(), indices)?
        else {
            return Ok(None);
        };

        rebuild_map_from_array(array.map_dtype().clone(), entries).map(Some)
    }
}

impl TakeExecute for Map {
    fn take(
        array: ArrayView<'_, Self>,
        indices: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let Some(entries) =
            <ListView as TakeExecute>::take(array.entries().as_::<ListView>(), indices, ctx)?
        else {
            return Ok(None);
        };

        rebuild_map_from_array(array.map_dtype().clone(), entries).map(Some)
    }
}
