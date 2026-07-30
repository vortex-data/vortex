// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::ArrayRef;
use crate::array::ArrayView;
use crate::arrays::ListViewArray;
use crate::arrays::filter::FilterKernel;
use crate::arrays::filter::FilterReduce;
use crate::arrays::listview::ListViewArraySlotsExt;
use crate::arrays::map::Map;
use crate::arrays::map::MapArrayExt;
use crate::arrays::map::compute::rebuild_map;
use crate::executor::ExecutionCtx;

impl FilterReduce for Map {
    fn filter(array: ArrayView<'_, Self>, mask: &Mask) -> VortexResult<Option<ArrayRef>> {
        let entries = array.entries();

        // SAFETY: filtering row metadata keeps offsets and sizes paired, preserves the original
        // elements, and filters validity to the same output length. The zero-copy-to-list flag is
        // not carried over: dropping a non-empty row leaves a gap in the referenced elements.
        let filtered_entries = unsafe {
            ListViewArray::new_unchecked(
                entries.elements().clone(),
                entries.offsets().filter(mask.clone())?,
                entries.sizes().filter(mask.clone())?,
                entries.validity()?.filter(mask)?,
            )
        };

        rebuild_map(array.map_dtype().clone(), filtered_entries).map(Some)
    }
}

impl FilterKernel for Map {
    fn filter(
        array: ArrayView<'_, Self>,
        mask: &Mask,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        <Self as FilterReduce>::filter(array, mask)
    }
}
