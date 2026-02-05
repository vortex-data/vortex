// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_dtype::Nullability;
use vortex_dtype::match_each_integer_ptype;
use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::Array;
use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::FilterArray;
use crate::arrays::FilterVTable;
use crate::arrays::ListArray;
use crate::arrays::ListVTable;
use crate::arrays::ListViewArray;
use crate::arrays::list::compute::element_mask_from_offsets;
use crate::builders::ArrayBuilder;
use crate::builders::ListViewBuilder;
use crate::kernel::ExecuteParentKernel;
use crate::validity::Validity;
use crate::vtable::ValidityHelper;

#[derive(Debug)]
pub(super) struct ListFilterKernel;

impl ExecuteParentKernel<ListVTable> for ListFilterKernel {
    type Parent = FilterVTable;

    // TODO(joe): Remove the vector usage?
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
                let mut builder = ListViewBuilder::<u32, u32>::with_capacity(
                    array.element_dtype().clone(),
                    Nullability::Nullable,
                    selection.true_count(),
                    0,
                );
                builder.append_nulls(selection.true_count());
                return Ok(Some(builder.finish()));
            }
            Validity::Array(a) => Validity::Array(a.filter(parent.filter_mask().clone())?),
        };

        // TODO(ngates): for ultra-sparse masks, we don't need to optimize the entire offsets.
        let offsets = array.offsets().clone();

        let (new_offsets, new_sizes, element_mask) =
            match_each_integer_ptype!(offsets.dtype().as_ptype(), |O| {
                let offsets_buffer = offsets.execute::<Buffer<O>>(ctx)?;
                let offsets = offsets_buffer.as_slice();
                let mut new_offsets = BufferMut::<O>::with_capacity(selection.true_count());
                let mut new_sizes = BufferMut::<O>::with_capacity(selection.true_count());

                let mut offset = 0;
                for idx in selection.indices() {
                    let start = offsets[*idx];
                    let end = offsets[idx + 1];
                    let size = end - start;
                    unsafe { new_offsets.push_unchecked(offset) };
                    unsafe { new_sizes.push_unchecked(size) };
                    offset += size;
                }

                // TODO(ngates): for very dense masks, there may be no point in filtering the elements,
                //  and instead we should construct a view against the unfiltered elements.
                let element_mask = element_mask_from_offsets::<O>(offsets, selection);

                (
                    new_offsets.freeze().into_array(),
                    new_sizes.freeze().into_array(),
                    element_mask,
                )
            });

        let new_elements = array.sliced_elements()?.filter(element_mask)?;

        Ok(Some(
            unsafe {
                ListViewArray::new_unchecked(new_elements, new_offsets, new_sizes, new_validity)
            }
            .into_array(),
        ))
    }
}
