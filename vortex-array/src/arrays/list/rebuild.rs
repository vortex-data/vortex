// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::BitBufferMut;
use vortex_buffer::BufferMut;
use vortex_error::VortexResult;
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
    /// The elements are sliced when only the ends go, and copied a run of valid lists at a time
    /// when a null list holds elements. An array with nothing to drop is returned as it is.
    pub fn rebuild_with_referenced_valid_elements(
        &self,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ListArray> {
        let offsets = self.offsets().clone().execute::<PrimitiveArray>(ctx)?;
        let validity = self.list_validity().execute_mask(self.len(), ctx)?;
        let all = [(0, self.len())];
        let valid = validity.slices().unwrap_or_else(|| &all, || &[]);
        match_each_integer_ptype!(offsets.ptype(), |O| {
            self.rebuild_with_valid_runs(offsets.as_slice::<O>(), valid)
        })
    }

    fn rebuild_with_valid_runs<O: IntegerPType>(
        &self,
        offsets: &[O],
        valid: &[(usize, usize)],
    ) -> VortexResult<ListArray> {
        let kept = referenced_valid_ranges(offsets, valid);
        if kept.iter().map(|(start, end)| end - start).sum::<usize>() == self.elements().len() {
            return Ok(self.clone());
        }
        let elements = match kept.as_slice() {
            [] => self.elements().slice(0..0)?,
            [(start, end)] => self.elements().slice(*start..*end)?,
            _ => {
                let mut bits = BitBufferMut::new_unset(self.elements().len());
                for &(start, end) in &kept {
                    bits.fill_range(start, end, true);
                }
                // With the ranges cached on the mask, the filter copies one range at a time.
                self.elements()
                    .filter(Mask::from_buffer_with_slices(bits.freeze(), kept))?
            }
        };
        let offsets = shift_past_null_lists(offsets, valid);
        // SAFETY: the elements are the ranges of the runs of valid lists, in order, and every offset
        // moved down by the elements dropped before it, so each valid list keeps its elements and
        // each null list is empty.
        Ok(unsafe {
            ListArray::new_unchecked(elements, offsets.into_array(), self.list_validity())
        })
    }
}

/// The element ranges of the runs of valid lists, with touching ranges merged.
fn referenced_valid_ranges<O: IntegerPType>(
    offsets: &[O],
    valid: &[(usize, usize)],
) -> Vec<(usize, usize)> {
    let mut ranges: Vec<(usize, usize)> = Vec::with_capacity(valid.len());
    for &(first, end) in valid {
        let (start, end): (usize, usize) = (offsets[first].as_(), offsets[end].as_());
        match ranges.last_mut() {
            _ if start == end => {}
            Some(last) if last.1 == start => last.1 = end,
            _ => ranges.push((start, end)),
        }
    }
    ranges
}

/// The offsets with every offset moved down by the elements dropped before it: those before the
/// first list and those under null lists. A null list then starts and ends where the valid lists
/// before it end.
fn shift_past_null_lists<O: IntegerPType>(offsets: &[O], valid: &[(usize, usize)]) -> BufferMut<O> {
    let mut shifted = BufferMut::copy_from(offsets);
    let mut dropped = offsets[0];
    let mut null_from = 0;
    for &(valid_start, valid_end) in valid {
        shifted[null_from..valid_start].fill(offsets[null_from] - dropped);
        dropped += offsets[valid_start] - offsets[null_from];
        for offset in &mut shifted[valid_start..valid_end] {
            *offset = *offset - dropped;
        }
        null_from = valid_end;
    }
    // The null lists after the last valid one, and the end of the last list.
    shifted[null_from..].fill(offsets[null_from] - dropped);
    shifted
}
