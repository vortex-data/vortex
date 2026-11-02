// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Sparse primitive demands, materialized only by an owner that can serve them.

use std::iter;
use std::ops::Range;

use vortex_mask::Mask;
use vortex_session::registry::ReadContext;

/// Byte demand for a non-null primitive segment, subject to verification of its encoding tree.
#[derive(Clone, Debug)]
pub struct PrimitiveRead {
    /// Encoding registry used to identify the serialized array.
    pub array_ctx: ReadContext,
    /// Expected length of the primitive value buffer.
    pub buffer_len: usize,
    width: usize,
    offset: usize,
    selection: Mask,
}

impl PrimitiveRead {
    /// Keeps the logical selection without enumerating its byte ranges.
    pub(crate) fn new(
        array_ctx: ReadContext,
        buffer_len: usize,
        width: usize,
        offset: usize,
        selection: Mask,
    ) -> Self {
        Self {
            array_ctx,
            buffer_len,
            width,
            offset,
            selection,
        }
    }

    /// Builds sorted byte ranges covering the selected rows, or declines an expensive read.
    ///
    /// Owners should first reject incompatible segment lengths and unsupported sparse I/O.
    /// This walks only as far as needed to exceed 32 ranges or one eighth of the value buffer.
    pub fn ranges(&self) -> Option<Vec<Range<usize>>> {
        if self.width == 0
            || self
                .offset
                .checked_add(self.selection.len())?
                .checked_mul(self.width)?
                > self.buffer_len
        {
            return None;
        }
        match &self.selection {
            Mask::AllFalse(_) => None,
            Mask::AllTrue(len) => self.collect_ranges(iter::once((0, *len))),
            Mask::Values(values) => match values.cached_slices() {
                Some(slices) => self.collect_ranges(slices.iter().copied()),
                None => self.collect_ranges(values.bit_buffer().set_slices()),
            },
        }
    }

    fn collect_ranges(
        &self,
        rows: impl Iterator<Item = (usize, usize)>,
    ) -> Option<Vec<Range<usize>>> {
        let mut ranges: Vec<Range<usize>> = Vec::new();
        let mut bytes = 0;
        for (start, end) in rows {
            let range = (self.offset + start) * self.width..(self.offset + end) * self.width;
            if let Some(previous) = ranges.last_mut()
                && range.start - previous.end <= 4096
            {
                bytes += range.end - previous.end;
                previous.end = range.end;
            } else {
                bytes += range.len();
                ranges.push(range);
            }
            if ranges.len() > 32 || bytes > self.buffer_len / 8 {
                return None;
            }
        }
        Some(ranges)
    }
}

#[cfg(test)]
mod tests;
