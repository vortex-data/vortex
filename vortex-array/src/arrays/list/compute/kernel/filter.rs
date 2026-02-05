// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use num_traits::Zero;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_dtype::match_each_integer_ptype;
use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::Array;
use crate::ArrayRef;
use crate::Canonical;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::ConstantArray;
use crate::arrays::FilterArray;
use crate::arrays::FilterVTable;
use crate::arrays::ListArray;
use crate::arrays::ListVTable;
use crate::arrays::list::compute::element_mask_from_offsets;
use crate::kernel::ExecuteParentKernel;
use crate::validity::Validity;
use crate::vtable::ValidityHelper;

#[derive(Debug)]
pub(super) struct ListFilterKernel;

impl ExecuteParentKernel<ListVTable> for ListFilterKernel {
    type Parent = FilterVTable;

    fn execute_parent(
        &self,
        array: &ListArray,
        parent: &FilterArray,
        _child_idx: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let selection = match parent.filter_mask() {
            Mask::AllTrue(_) | Mask::AllFalse(_) => return Ok(None),
            Mask::Values(v) => v,
        };

        let new_validity = match array.validity() {
            Validity::NonNullable => Validity::NonNullable,
            Validity::AllValid => Validity::AllValid,
            Validity::AllInvalid => {
                let elements = Canonical::empty(array.element_dtype()).into_array();
                let offsets = ConstantArray::new(0u64, selection.true_count() + 1).into_array();
                return Ok(Some(unsafe {
                    ListArray::new_unchecked(elements, offsets, Validity::AllInvalid).into_array()
                }));
            }
            Validity::Array(a) => Validity::Array(a.filter(parent.filter_mask().clone())?),
        };

        // TODO(ngates): for ultra-sparse masks, we don't need to optimize the entire offsets.
        let offsets = array.offsets().clone();

        let (new_offsets, element_mask) =
            match_each_integer_ptype!(offsets.dtype().as_ptype(), |O| {
                let offsets_buffer = offsets.execute::<Buffer<O>>(ctx)?;
                let offsets = offsets_buffer.as_slice();
                let mut new_offsets = BufferMut::<O>::with_capacity(selection.true_count() + 1);

                let mut offset = O::zero();
                unsafe { new_offsets.push_unchecked(offset) };
                for idx in selection.indices() {
                    let size = offsets[idx + 1] - offsets[*idx];
                    offset += size;
                    unsafe { new_offsets.push_unchecked(offset) };
                }

                // TODO(ngates): for very dense masks, there may be no point in filtering the elements,
                //  and instead we should construct a view against the unfiltered elements.
                let element_mask = element_mask_from_offsets::<O>(offsets, selection);

                (new_offsets.freeze().into_array(), element_mask)
            });

        let new_elements = array.sliced_elements()?.filter(element_mask)?;

        // SAFETY: new_offsets are monotonically increasing starting from 0 with length
        // true_count + 1, and the elements have been filtered to match.
        Ok(Some(unsafe {
            ListArray::new_unchecked(new_elements, new_offsets, new_validity).into_array()
        }))
    }
}
