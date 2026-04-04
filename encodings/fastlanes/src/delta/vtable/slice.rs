// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::cmp::min;
use std::ops::Range;

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::IntoArray;
use vortex_array::arrays::slice::SliceReduce;
use vortex_error::VortexResult;

use crate::delta::array::DeltaArrayExt;
use crate::delta::array::lane_count;
use crate::delta::vtable::Delta;

impl SliceReduce for Delta {
    fn slice(array: ArrayView<'_, Self>, range: Range<usize>) -> VortexResult<Option<ArrayRef>> {
        let physical_start = range.start + array.offset();
        let physical_stop = range.end + array.offset();

        let start_chunk = physical_start / 1024;
        let stop_chunk = physical_stop.div_ceil(1024);

        let bases = array.bases();
        let deltas = array.deltas();
        let lanes = lane_count(array.dtype().as_ptype());

        let new_bases =
            bases.slice(min(start_chunk * lanes, bases.len())..min(stop_chunk * lanes, bases.len()))?;

        let new_deltas = deltas.slice(
            min(start_chunk * 1024, deltas.len())..min(stop_chunk * 1024, deltas.len()),
        )?;

        Ok(Some(
            Delta::try_new(new_bases, new_deltas, physical_start % 1024, range.len())?.into_array(),
        ))
    }
}
