// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::cmp::max;
use std::ops::Range;

use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::SliceKernel;
use vortex_array::vtable::ValidityHelper;
use vortex_error::VortexResult;

use crate::BitPackedArray;
use crate::BitPackedVTable;

impl SliceKernel for BitPackedVTable {
    fn slice(
        array: &BitPackedArray,
        range: Range<usize>,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let offset_start = range.start + array.offset() as usize;
        let offset_stop = range.end + array.offset() as usize;
        let offset = offset_start % 1024;
        let block_start = max(0, offset_start - offset);
        let block_stop = offset_stop.div_ceil(1024) * 1024;

        let encoded_start = (block_start / 8) * array.bit_width() as usize;
        let encoded_stop = (block_stop / 8) * array.bit_width() as usize;

        // slice the buffer using the encoded start/stop values
        // SAFETY: slicing packed values without decoding preserves invariants
        Ok(Some(unsafe {
            BitPackedArray::new_unchecked(
                array.packed().slice(encoded_start..encoded_stop),
                array.dtype().clone(),
                array.validity().clone().slice(range.clone())?,
                array
                    .patches()
                    .map(|p| p.slice(range.clone()))
                    .transpose()?
                    .flatten(),
                array.bit_width(),
                range.len(),
                offset as u16,
            )
            .into_array()
        }))
    }
}
