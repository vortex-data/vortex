// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::BitBuffer;
use vortex_buffer::BitBufferMut;
use vortex_buffer::BufferMut;
use vortex_error::VortexResult;
use vortex_mask::AllOr;
use vortex_mask::Mask;

use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::ListArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::list::ListArrayExt;
use crate::arrays::list::ListArraySlotsExt;
use crate::dtype::IntegerPType;
use crate::match_each_integer_ptype;

impl ListArray {
    /// Rebuilds the [`ListArray`] over only the elements that valid lists reference: nothing before
    /// the first list, after the last or under a null list.
    ///
    /// A list with no null lists and nothing before the first list or after the last is returned
    /// as it is, after reading two offsets.
    pub fn rebuild_with_referenced_valid_elements(
        &self,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ListArray> {
        let validity = self.list_validity();
        if validity.definitely_no_nulls()
            && self.offset_at(0, ctx)? == 0
            && self.offset_at(self.len(), ctx)? == self.elements().len()
        {
            return Ok(self.clone());
        }

        let offsets = self.offsets().clone().execute::<PrimitiveArray>(ctx)?;
        let valid = validity.execute_mask(self.len(), ctx)?;
        match_each_integer_ptype!(offsets.ptype(), |O| {
            self.rebuild_with_valid(offsets.as_slice::<O>(), &valid)
        })
    }

    fn rebuild_with_valid<O: IntegerPType>(
        &self,
        offsets: &[O],
        valid: &Mask,
    ) -> VortexResult<ListArray> {
        let all = [(0, self.len())];
        let (first, last): (usize, usize) = (offsets[0].as_(), offsets[self.len()].as_());
        // When no null list holds elements, all the lists are kept as one run: the null lists in it
        // stay empty.
        let runs = if null_lists_hold_elements(valid, |index| offsets[index] != offsets[index + 1])
        {
            valid.slices().unwrap_or_else(|| &all, || &[])
        } else if first == 0 && last == self.elements().len() {
            return Ok(self.clone());
        } else {
            &all
        };

        // Keep the elements of each run, merging runs whose elements touch, and move each offset
        // down by the elements dropped before it, so that null lists are empty.
        let mut kept: Vec<(usize, usize)> = Vec::with_capacity(runs.len());
        let mut ends = BufferMut::copy_from(offsets);
        let mut dropped = offsets[0];
        let mut null_from = 0;
        for &(run_start, run_end) in runs {
            let (start, stop): (usize, usize) = (offsets[run_start].as_(), offsets[run_end].as_());
            match kept.last_mut() {
                Some(range) if range.1 == start => range.1 = stop,
                _ if start < stop => kept.push((start, stop)),
                _ => {}
            }
            ends[null_from..run_start].fill(offsets[null_from] - dropped);
            dropped += offsets[run_start] - offsets[null_from];
            for end in &mut ends[run_start..run_end] {
                *end = *end - dropped;
            }
            null_from = run_end;
        }
        ends[null_from..].fill(offsets[null_from] - dropped);

        let mut bits = BitBufferMut::new_unset(self.elements().len());
        for &(start, stop) in &kept {
            bits.fill_range(start, stop, true);
        }
        // With the ranges cached on the mask, the filter slices one range and copies several range
        // by range.
        let elements = self
            .elements()
            .filter(Mask::from_buffer_with_slices(bits.freeze(), kept))?;
        // SAFETY: the elements are the runs' elements in order, and every offset moved down by the
        // elements dropped before it, so each valid list keeps its elements and each null list is
        // empty.
        Ok(unsafe {
            ListArray::new_unchecked(elements, ends.freeze().into_array(), self.list_validity())
        })
    }
}

/// Whether a null list holds elements, given which lists hold elements.
pub(crate) fn null_lists_hold_elements(
    valid: &Mask,
    mut holds_elements: impl FnMut(usize) -> bool,
) -> bool {
    match valid.bit_buffer() {
        AllOr::All => false,
        AllOr::None => (0..valid.len()).any(holds_elements),
        AllOr::Some(valid) => {
            BitBuffer::collect_bool(valid.len(), &mut holds_elements)
                .bitand_not(valid)
                .true_count()
                > 0
        }
    }
}
