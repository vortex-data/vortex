// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::BufferAllocatorRef;
use vortex_buffer::BufferMut;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_mask::AllOr;
use vortex_mask::Mask;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::ListArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::list::ListArrayExt;
use crate::arrays::list::ListArraySlotsExt;
use crate::arrays::primitive::PrimitiveArrayExt;
use crate::dtype::IntegerPType;
use crate::dtype::PType;
use crate::match_each_unsigned_integer_ptype;
use crate::validity::Validity;

impl ListArray {
    /// Rebuilds the [`ListArray`] over only the elements that valid lists reference: nothing before
    /// the first list, after the last or under a null list.
    ///
    /// Null lists come out empty, the offsets start at 0, end at the length of the elements and
    /// keep their type, and the validity is kept. An array with nothing to drop is returned as it
    /// is.
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
        let offsets_ptype = offsets.ptype();
        // Offsets are never negative, so they are rebuilt unsigned and reinterpreted back.
        let offsets = offsets.reinterpret_cast(offsets_ptype.to_unsigned());
        let valid = validity.execute_mask(self.len(), ctx)?;
        match_each_unsigned_integer_ptype!(offsets.ptype(), |O| {
            self.rebuild_with_valid(offsets.as_slice::<O>(), offsets_ptype, &valid, ctx)
        })
    }

    fn rebuild_with_valid<O: IntegerPType>(
        &self,
        offsets: &[O],
        offsets_ptype: PType,
        valid: &Mask,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ListArray> {
        let (starts, end) = (&offsets[..self.len()], offsets[self.len()].as_());
        let null_runs = null_runs_holding_elements(starts, end, valid);
        if null_runs.is_empty() && offsets[0].as_() == 0 && end == self.elements().len() {
            return Ok(self.clone());
        }

        let (elements, mut offsets) =
            drop_unreferenced_elements(self.elements(), starts, end, &null_runs, ctx.allocator())?;
        offsets.push(
            num_traits::cast(elements.len())
                .vortex_expect("the elements kept fit the offsets' type"),
        );
        let offsets = PrimitiveArray::new(offsets.freeze(), Validity::NonNullable)
            .reinterpret_cast(offsets_ptype)
            .into_array();
        // SAFETY: the offsets keep their type, start at 0, never decrease and end at the length of
        // the elements, which are the valid lists' elements in order.
        Ok(unsafe { ListArray::new_unchecked(elements, offsets, self.list_validity()) })
    }
}

/// The runs of null lists that hold elements, among lists laid out back to back: list `i` starts
/// at `starts[i]` and ends where the next list starts, or at `end` for the last list.
pub(crate) fn null_runs_holding_elements<O: IntegerPType>(
    starts: &[O],
    end: usize,
    valid: &Mask,
) -> Vec<(usize, usize)> {
    let start_of = |index: usize| starts.get(index).map_or(end, |start| start.as_());
    let mut runs = Vec::new();
    // Lists never start before the previous one, so a run of null lists holds elements if and
    // only if the list after it starts later than the run does.
    let mut push_if_holding = |run_start: usize, run_end: usize| {
        if run_start < run_end && start_of(run_start) != start_of(run_end) {
            runs.push((run_start, run_end));
        }
    };
    match valid.bit_buffer() {
        AllOr::All => {}
        AllOr::None => push_if_holding(0, valid.len()),
        AllOr::Some(bits) => {
            let mut run_start = 0;
            // The empty slice at the end closes a run of null lists at the end.
            for (valid_start, valid_end) in bits.set_slices().chain([(bits.len(), bits.len())]) {
                push_if_holding(run_start, valid_start);
                run_start = valid_end;
            }
        }
    }
    runs
}

/// Drops the elements before the first of the lists laid out back to back, after the last, and
/// under the runs of null lists from [`null_runs_holding_elements`], which come out empty.
///
/// Returns the elements kept and where each list now starts, with room to push one more offset.
pub(crate) fn drop_unreferenced_elements<O: IntegerPType>(
    elements: &ArrayRef,
    starts: &[O],
    end: usize,
    null_runs: &[(usize, usize)],
    allocator: &BufferAllocatorRef,
) -> VortexResult<(ArrayRef, BufferMut<O>)> {
    let start_of = |index: usize| starts.get(index).map_or(end, |start| start.as_());
    let first = start_of(0);
    // The ranges of elements kept, counted from the first list's start.
    let mut kept = Vec::with_capacity(null_runs.len() + 1);
    let mut kept_start = 0;
    for &(run_start, run_end) in null_runs {
        if kept_start < start_of(run_start) - first {
            kept.push((kept_start, start_of(run_start) - first));
        }
        kept_start = start_of(run_end) - first;
    }
    if kept_start < end - first {
        kept.push((kept_start, end - first));
    }
    let referenced = elements.slice(first..end)?;
    let elements = if let [(start, stop)] = kept[..] {
        referenced.slice(start..stop)?
    } else {
        // With the ranges cached on the mask, the filter copies the elements range by range.
        referenced.filter(Mask::from_slices(end - first, kept))?
    };

    let mut offsets = BufferMut::with_capacity_in(starts.len() + 1, allocator.clone());
    let mut dropped = starts.first().copied().unwrap_or_else(O::zero);
    let mut from = 0;
    for &(run_start, run_end) in null_runs {
        offsets.extend(starts[from..run_start].iter().map(|&start| start - dropped));
        offsets.push_n(starts[run_start] - dropped, run_end - run_start);
        // A run at the end leaves no list after it to move down.
        if let Some(&next) = starts.get(run_end) {
            dropped += next - starts[run_start];
        }
        from = run_end;
    }
    offsets.extend(starts[from..].iter().map(|&start| start - dropped));
    Ok((elements, offsets))
}
