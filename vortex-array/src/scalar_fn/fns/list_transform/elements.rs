// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use num_traits::AsPrimitive;
use vortex_buffer::BitBufferMut;
use vortex_buffer::BufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_mask::Mask;

use crate::ArrayRef;
use crate::Canonical;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::ConstantArray;
use crate::arrays::FixedSizeListArray;
use crate::arrays::List;
use crate::arrays::ListArray;
use crate::arrays::ListViewArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::fixed_size_list::FixedSizeListArrayExt;
use crate::arrays::fixed_size_list::FixedSizeListArraySlotsExt;
use crate::arrays::list::ListArrayExt;
use crate::arrays::list::ListArraySlotsExt;
use crate::arrays::listview::ListViewArrayExt;
use crate::arrays::listview::ListViewArraySlotsExt;
use crate::dtype::DType;
use crate::match_each_integer_ptype;
use crate::validity::Validity;

/// The elements a per-element lambda runs over, list after list, and how to rebuild the lists
/// around one result per element.
///
/// Every element of a non-null list is included. A null list's elements are dropped when they must
/// not reach the lambda, for a body that can fail; otherwise they stay, and their results stay
/// hidden behind the list's validity. Nothing is copied unless the included ranges are not already
/// laid out back to back.
pub(super) struct ListElements {
    elements: ArrayRef,
    /// Where each list's elements start, then where the last one's end: `n + 1` offsets from 0.
    offsets: ArrayRef,
    validity: Validity,
}

impl ListElements {
    pub(super) fn new(
        list: ArrayRef,
        keep_null_lists: bool,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Self> {
        // A list array whose elements all reach the lambda is used as it is.
        if let Some(list) = list.as_opt::<List>() {
            let validity = list.list_validity();
            let keeps_every_element =
                keep_null_lists || matches!(validity, Validity::NonNullable | Validity::AllValid);
            if keeps_every_element && list.offset_at(0)? == 0 {
                return Ok(Self {
                    elements: list.sliced_elements()?,
                    offsets: list.offsets().clone(),
                    validity,
                });
            }
        }

        let (elements, starts, mut sizes, validity) = list_parts(list, ctx)?;
        if !keep_null_lists && !matches!(validity, Validity::NonNullable | Validity::AllValid) {
            let valid = validity.execute_mask(sizes.len(), ctx)?;
            for (list, size) in sizes.iter_mut().enumerate() {
                if !valid.value(list) {
                    *size = 0;
                }
            }
        }
        let elements = gather(elements, &starts, &sizes)?;
        let mut offsets = BufferMut::<u64>::with_capacity(sizes.len() + 1);
        let mut end = 0;
        offsets.push(end);
        for size in sizes {
            end += size;
            offsets.push(end);
        }
        Ok(Self {
            elements,
            offsets: offsets.into_array(),
            validity,
        })
    }

    /// The number of elements each list contributes, in list order.
    fn sizes(&self, ctx: &mut ExecutionCtx) -> VortexResult<Vec<u64>> {
        let offsets = to_u64s(&self.offsets, ctx)?;
        Ok(offsets
            .windows(2)
            .map(|bound| bound[1] - bound[0])
            .collect())
    }

    /// The number of elements.
    pub(super) fn len(&self) -> usize {
        self.elements.len()
    }

    /// The elements, list after list.
    pub(super) fn elements(&self) -> &ArrayRef {
        &self.elements
    }

    /// Each element's position within its list: sizes `[2, 0, 3]` give `[0, 1, 0, 1, 2]`.
    pub(super) fn indices(&self, ctx: &mut ExecutionCtx) -> VortexResult<ArrayRef> {
        // A list's positions are written in whole blocks, `0..BLOCK` then `BLOCK..2 * BLOCK` and so
        // on, and the last block may spill past the list's end: the next list overwrites the spill,
        // and the buffer has room for the final one. A list no longer than a block then costs one
        // store and no branch on its length, which a position-by-position loop mispredicts at the
        // end of most lists.
        const BLOCK: usize = 16;
        let len = self.len();
        let mut indices = BufferMut::<u64>::with_capacity(len + BLOCK);
        let out = indices.as_mut_ptr();
        let first: [u64; BLOCK] = std::array::from_fn(|i| i as u64);
        let mut start = 0;
        for size in self.sizes(ctx)? {
            let end = start + usize::try_from(size)?;
            assert!(end <= len, "list sizes add up to more than {len} elements");
            let (mut at, mut block) = (start, first);
            // An empty list writes a block too, which the next list overwrites.
            loop {
                // SAFETY: `at` is `start` or below `end`, so at most `len`, and the buffer has
                // capacity for `len + BLOCK`.
                unsafe { out.add(at).cast::<[u64; BLOCK]>().write_unaligned(block) };
                at += BLOCK;
                if at >= end {
                    break;
                }
                block = block.map(|index| index + BLOCK as u64);
            }
            start = end;
        }
        // SAFETY: every position below `start` was written by the list that holds it.
        unsafe { indices.set_len(start) };
        Ok(indices.freeze().into_array())
    }

    /// One value per list, repeated for each of that list's elements: sizes `[2, 0, 3]` turn
    /// `[a, b, c]` into `[a, a, c, c, c]`.
    pub(super) fn repeat(
        &self,
        values: ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        if let Some(value) = values.as_constant() {
            return Ok(ConstantArray::new(value, self.len()).into_array());
        }
        // TODO(mk): repeat each value as one run of a run-end array instead of copying it once per
        // element, once run-end lives in vortex-array.
        let mut lists = BufferMut::<u64>::with_capacity(self.len());
        for (list, size) in self.sizes(ctx)?.into_iter().enumerate() {
            lists.push_n(list as u64, usize::try_from(size)?);
        }
        // The body may read the repeated values once for every slice it takes, so compute them once.
        Ok(values
            .take(lists.freeze().into_array())?
            .execute::<Canonical>(ctx)?
            .into_array())
    }

    /// The lists again, with one value per element as their new elements.
    pub(super) fn rebuild(self, values: ArrayRef) -> VortexResult<ArrayRef> {
        vortex_ensure!(
            values.len() == self.len(),
            "expected one value per element: {} values for {} elements",
            values.len(),
            self.len()
        );
        // SAFETY: the offsets start at 0, never decrease, and end at the number of elements,
        // which is `values.len()`; the validity is the input lists'.
        Ok(unsafe { ListArray::new_unchecked(values, self.offsets, self.validity) }.into_array())
    }
}

/// The elements, the start and size of each list's range of them, and the lists' validity.
type ListParts = (ArrayRef, Vec<u64>, Vec<u64>, Validity);

fn list_parts(list: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<ListParts> {
    if let Some(list) = list.as_opt::<List>() {
        let bounds = to_u64s(list.offsets(), ctx)?;
        let starts = bounds[..list.len()].to_vec();
        let sizes = bounds.windows(2).map(|bound| bound[1] - bound[0]).collect();
        return Ok((list.elements().clone(), starts, sizes, list.list_validity()));
    }
    if let DType::FixedSizeList(_, size, _) = list.dtype() {
        let size = u64::from(*size);
        let list = list.execute::<FixedSizeListArray>(ctx)?;
        let starts = (0..list.len() as u64).map(|i| i * size).collect();
        let sizes = vec![size; list.len()];
        let validity = list.fixed_size_list_validity();
        return Ok((list.elements().clone(), starts, sizes, validity));
    }
    let list = list.execute::<ListViewArray>(ctx)?;
    let starts = to_u64s(list.offsets(), ctx)?;
    let sizes = to_u64s(list.sizes(), ctx)?;
    let validity = list.listview_validity();
    Ok((list.elements().clone(), starts, sizes, validity))
}

/// The elements at `starts[i]..starts[i] + sizes[i]` of each list, in list order.
fn gather(elements: ArrayRef, starts: &[u64], sizes: &[u64]) -> VortexResult<ArrayRef> {
    let mut range: Option<(u64, u64)> = None;
    let (mut back_to_back, mut in_order) = (true, true);
    for (&start, &size) in starts.iter().zip(sizes).filter(|(_, size)| **size > 0) {
        if let Some((_, end)) = range {
            back_to_back &= start == end;
            in_order &= start >= end;
        }
        range = Some((range.map_or(start, |(first, _)| first), start + size));
    }

    let Some((first, end)) = range else {
        return elements.slice(0..0);
    };
    if back_to_back {
        return elements.slice(usize::try_from(first)?..usize::try_from(end)?);
    }
    if in_order {
        // Copy the ranges, keeping their order: a filter by a mask of them.
        let mut included = BitBufferMut::new_unset(elements.len());
        for (&start, &size) in starts.iter().zip(sizes) {
            included.fill_range(
                usize::try_from(start)?,
                usize::try_from(start + size)?,
                true,
            );
        }
        return elements.filter(Mask::from(included.freeze()));
    }
    // Ranges that overlap or run out of order, as in a list view.
    let total = usize::try_from(sizes.iter().sum::<u64>())?;
    let mut indices = BufferMut::<u64>::with_capacity(total);
    for (&start, &size) in starts.iter().zip(sizes) {
        indices.extend(start..start + size);
    }
    elements.take(indices.freeze().into_array())
}

fn to_u64s(array: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Vec<u64>> {
    let array = array.clone().execute::<PrimitiveArray>(ctx)?;
    Ok(match_each_integer_ptype!(array.ptype(), |T| {
        array
            .as_slice::<T>()
            .iter()
            .map(|&value| value.as_())
            .collect()
    }))
}
