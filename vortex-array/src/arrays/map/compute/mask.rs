// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::array::ArrayView;
use crate::arrays::ListView;
use crate::arrays::map::Map;
use crate::arrays::map::MapArrayExt;
use crate::arrays::map::compute::rebuild_map_from_array;
use crate::executor::ExecutionCtx;
use crate::scalar_fn::fns::mask::MaskKernel;
use crate::scalar_fn::fns::mask::MaskReduce;

impl MaskReduce for Map {
    fn mask(array: ArrayView<'_, Self>, mask: &ArrayRef) -> VortexResult<Option<ArrayRef>> {
        let Some(entries) = <ListView as MaskReduce>::mask(array.entries(), mask)? else {
            return Ok(None);
        };

        rebuild_map_from_array(array.map_dtype().clone(), entries).map(Some)
    }
}

impl MaskKernel for Map {
    fn mask(
        array: ArrayView<'_, Self>,
        mask: &ArrayRef,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        <Self as MaskReduce>::mask(array, mask)
    }
}
