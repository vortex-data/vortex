// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use num_traits::AsPrimitive;
use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::ArrayView;
use crate::IntoArray;
use crate::arrays::Primitive;
use crate::arrays::runend::RunEnd;
use crate::arrays::runend::array::RunEndArrayExt;
use crate::arrays::runend::array::RunEndArraySlotsExt;
use crate::arrays::slice::SliceReduce;
use crate::match_each_unsigned_integer_ptype;

/// Slices without reading buffers beyond host-resident primitive ends, so that scalar functions
/// above the slice can still be pushed into the run values.
impl SliceReduce for RunEnd {
    fn slice(array: ArrayView<'_, Self>, range: Range<usize>) -> VortexResult<Option<ArrayRef>> {
        let Some(ends) = array.ends().as_opt::<Primitive>() else {
            return Ok(None);
        };
        if !ends.ptype().is_unsigned_int() || !ends.buffer_handle().is_on_host() {
            return Ok(None);
        }
        let start = range.start + array.offset();
        let end = range.end + array.offset();
        let (first_run, end_run) = match_each_unsigned_integer_ptype!(ends.ptype(), |T| {
            let ends = ends.as_slice::<T>();
            (
                ends.partition_point(|&e| AsPrimitive::<usize>::as_(e) <= start),
                (ends.partition_point(|&e| AsPrimitive::<usize>::as_(e) < end) + 1).min(ends.len()),
            )
        });

        // SAFETY: the kept runs cover `start..end`, and the offset re-bases the ends onto it.
        Ok(Some(
            unsafe {
                RunEnd::new_unchecked(
                    array.ends().slice(first_run..end_run)?,
                    array.values().slice(first_run..end_run)?,
                    start,
                    range.len(),
                )
            }
            .into_array(),
        ))
    }
}
