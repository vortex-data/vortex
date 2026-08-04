// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::ChunkedArray;
use vortex_error::VortexResult;
use vortex_mask::AllOr;
use vortex_mask::Mask;

use super::Zstd;
use super::unsliced_validity;

impl Zstd {
    /// Returns each visible frame's row range relative to the slice. Leading, intervening and
    /// trailing nulls belong to an adjacent frame so every logical row is covered.
    pub(crate) fn frame_row_ranges(
        array: ArrayView<'_, Self>,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<Vec<Range<usize>>>> {
        let data = array.data();
        if data.frames.len() < 2 {
            return Ok(None);
        }
        // Legacy variable-width frames without counts cannot be mapped to rows.
        if !array.dtype().is_primitive() && data.metadata.frames.iter().any(|f| f.n_values == 0) {
            return Ok(None);
        }

        let validity = unsliced_validity(array).execute_mask(data.unsliced_n_rows(), ctx)?;
        let count = validity.true_count();
        let starts = data.frame_value_starts(array.dtype(), count)?;
        if count == 0 || starts.last() != Some(&count) {
            return Ok(None);
        }

        let mut ranges = Vec::with_capacity(data.frames.len());
        let mut add_range = |start: usize, end: usize| {
            let start = start.max(data.slice_start());
            let end = end.min(data.slice_stop());
            if start < end {
                ranges.push(start - data.slice_start()..end - data.slice_start());
            }
        };
        let bits = validity.bit_buffer();
        let mut previous_row = 0;
        let mut previous_value = 0;
        for &value in &starts[1..starts.len() - 1] {
            if value == previous_value || value >= count {
                return Ok(None);
            }
            // Resume at the previous frame boundary so nullable arrays scan their validity once.
            let row = match bits {
                AllOr::All => value,
                AllOr::None => return Ok(None),
                AllOr::Some(bits) => {
                    let remaining = bits.as_view().slice(previous_row..);
                    let Some(offset) = remaining.select(value - previous_value) else {
                        return Ok(None);
                    };
                    previous_row + offset
                }
            };
            add_range(previous_row, row);
            previous_row = row;
            previous_value = value;
        }
        add_range(previous_row, data.unsliced_n_rows());
        Ok(Some(ranges))
    }

    /// Filters contiguous groups of selected frames. Returns `None` when the full slice is needed.
    pub(crate) fn filter_selected_frames(
        array: ArrayView<'_, Self>,
        frames: &[Range<usize>],
        mask: &Mask,
    ) -> VortexResult<Option<ArrayRef>> {
        let Some(selection) = mask.values() else {
            return Ok(None);
        };
        let mut spans: Vec<Range<usize>> = Vec::new();
        for frame in frames {
            if selection.bit_buffer().count_range(frame.start, frame.end) == 0 {
                continue;
            }
            match spans.last_mut() {
                Some(last) if last.end == frame.start => last.end = frame.end,
                _ => spans.push(frame.clone()),
            }
        }
        if spans.len() == 1 && spans[0] == (0..array.len()) {
            return Ok(None);
        }

        let mut chunks = spans
            .into_iter()
            .map(|span| array.as_ref().slice(span.clone())?.filter(mask.slice(span)))
            .collect::<VortexResult<Vec<_>>>()?;
        if chunks.len() == 1 {
            return Ok(chunks.pop());
        }
        Ok(Some(
            ChunkedArray::try_new(chunks, array.dtype().clone())?.into_array(),
        ))
    }
}
