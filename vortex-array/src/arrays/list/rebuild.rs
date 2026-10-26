// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::BitBuffer;
use vortex_buffer::BitBufferMut;
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
    /// Compacts the [`ListArray`] to only the elements that valid lists reference: nothing before
    /// the first list, after the last or under a null list.
    ///
    /// Null lists come out empty, the offsets start at 0, end at the length of the elements and
    /// keep their type, and the validity is kept. An array with nothing to drop is returned as it
    /// is.
    pub fn compact(&self, ctx: &mut ExecutionCtx) -> VortexResult<ListArray> {
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
        let null_runs =
            null_runs_holding_elements(valid, |index| offsets[index] != offsets[index + 1]);
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

/// The runs of null lists that hold elements, given which lists hold elements.
pub(crate) fn null_runs_holding_elements(
    valid: &Mask,
    mut holds_elements: impl FnMut(usize) -> bool,
) -> Vec<(usize, usize)> {
    match valid.bit_buffer() {
        AllOr::All => Vec::new(),
        AllOr::None if (0..valid.len()).any(&mut holds_elements) => vec![(0, valid.len())],
        AllOr::None => Vec::new(),
        AllOr::Some(valid) => BitBuffer::collect_bool(valid.len(), &mut holds_elements)
            .bitand_not(valid)
            .set_slices()
            .collect(),
    }
}

/// Drops the elements before the first of the lists laid out back to back, after the last, and
/// under the runs of null lists from [`null_runs_holding_elements`], which come out empty.
///
/// `end` is where the last list ends. It is a `usize` rather than an `O` because a list view's last
/// list can end past what its offsets' type holds: with `u8` offsets `[1, 5, 105]` and sizes
/// `[4, 100, 200]`, it ends at 305.
///
/// Returns the elements kept and where each list now starts. A [`ListArray`] needs one more offset,
/// where its last list ends, so the buffer has room to push it without reallocating.
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
    let mut offsets = BufferMut::with_capacity_in(starts.len() + 1, allocator.clone());
    offsets.extend_from_slice(starts);
    let mut dropped = starts.first().copied().unwrap_or_else(O::zero);
    let mut from = 0;
    for &(run_start, run_end) in null_runs {
        if kept_start < start_of(run_start) - first {
            kept.push((kept_start, start_of(run_start) - first));
        }
        kept_start = start_of(run_end) - first;

        for offset in &mut offsets[from..run_start] {
            *offset = *offset - dropped;
        }
        offsets[run_start..run_end].fill(starts[run_start] - dropped);
        // A run at the end leaves no list after it to move down.
        if let Some(&next) = starts.get(run_end) {
            dropped += next - starts[run_start];
        }
        from = run_end;
    }
    if kept_start < end - first {
        kept.push((kept_start, end - first));
    }
    for offset in &mut offsets[from..] {
        *offset = *offset - dropped;
    }

    let referenced = elements.slice(first..end)?;
    let elements = match kept[..] {
        [] => referenced.slice(0..0)?,
        [(start, stop)] => referenced.slice(start..stop)?,
        _ => {
            let mut bits = BitBufferMut::new_unset(end - first);
            for &(start, stop) in &kept {
                bits.fill_range(start, stop, true);
            }
            // With the ranges cached on the mask, the filter copies the elements range by range.
            referenced.filter(Mask::from_buffer_with_slices(bits.freeze(), kept))?
        }
    };
    Ok((elements, offsets))
}
