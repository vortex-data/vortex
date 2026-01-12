// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use vortex_buffer::BufferMut;
use vortex_dtype::PTypeDowncastExt;
use vortex_dtype::match_each_integer_ptype;
use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_vector::VectorMutOps;
use vortex_vector::listview::ListViewVector;
use vortex_vector::listview::ListViewVectorMut;
use vortex_vector::primitive::PVector;
use vortex_vector::primitive::PrimitiveVector;

use crate::Array;
use crate::Canonical;
use crate::ExecutionCtx;
use crate::VectorExecutor;
use crate::arrays::FilterArray;
use crate::arrays::FilterVTable;
use crate::arrays::ListArray;
use crate::arrays::ListVTable;
use crate::arrays::list::compute::element_mask_from_offsets;
use crate::kernel::ExecuteParentKernel;
use crate::mask::MaskExecutor;
use crate::matchers::Exact;
use crate::validity::Validity;
use crate::vectors::VectorIntoArray;
use crate::vtable::ValidityHelper;

#[derive(Debug)]
pub(super) struct ListFilterKernel;

impl ExecuteParentKernel<ListVTable> for ListFilterKernel {
    type Parent = Exact<FilterVTable>;

    fn parent(&self) -> Self::Parent {
        Exact::from(&FilterVTable)
    }

    // TODO(joe): should this use Vector?
    fn execute_parent(
        &self,
        array: &ListArray,
        parent: &FilterArray,
        _child_idx: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<Canonical>> {
        let selection = match parent.filter_mask() {
            Mask::AllTrue(_) | Mask::AllFalse(_) => return Ok(None),
            Mask::Values(v) => v,
        };

        // TODO(ngates): for ultra-sparse masks, we don't need to optimize the entire offsets.
        let offsets = array
            .offsets()
            .execute(ctx)?
            .to_vector(ctx)?
            .into_primitive();

        let new_validity = match array.validity() {
            Validity::NonNullable | Validity::AllValid => Mask::new_true(selection.true_count()),
            Validity::AllInvalid => {
                let mut vec = ListViewVectorMut::with_capacity(
                    array.elements().dtype(),
                    selection.true_count(),
                    0,
                );
                vec.append_nulls(selection.true_count());
                return Ok(Some(vec.freeze().into_array(array.dtype()).to_canonical()));
            }
            Validity::Array(a) => a.filter(parent.filter_mask().clone())?.execute_mask(ctx)?,
        };

        let (new_offsets, new_sizes) = match_each_integer_ptype!(offsets.ptype(), |O| {
            let offsets = (&offsets).downcast::<O>().elements().as_slice();
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

            let new_offsets = PrimitiveVector::from(PVector::<O>::new(
                new_offsets.freeze(),
                Mask::new_true(selection.true_count()),
            ));
            let new_sizes = PrimitiveVector::from(PVector::<O>::new(
                new_sizes.freeze(),
                Mask::new_true(selection.true_count()),
            ));

            (new_offsets, new_sizes)
        });

        // TODO(ngates): for very dense masks, there may be no point in filtering the elements,
        //  and instead we should construct a view against the unfiltered elements.
        let element_mask = match_each_integer_ptype!(offsets.ptype(), |O| {
            element_mask_from_offsets::<O>((&offsets).downcast::<O>().elements(), selection)
        });

        let new_elements = array
            .sliced_elements()
            .filter(element_mask)?
            .execute(ctx)?
            .to_vector(ctx)?;

        Ok(Some(
            unsafe {
                ListViewVector::new_unchecked(
                    Arc::new(new_elements),
                    new_offsets,
                    new_sizes,
                    new_validity,
                )
            }
            .into_array(array.dtype())
            .to_canonical(),
        ))
    }
}
