// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use vortex_error::{VortexExpect, VortexResult};
use vortex_vector::Vector;
use vortex_vector::fixed_size_list::FixedSizeListVector;

use crate::ArrayRef;
use crate::arrays::{FixedSizeListArray, FixedSizeListVTable};
use crate::execution::{BatchKernelRef, BindCtx, kernel};
use crate::vtable::{OperatorVTable, ValidityHelper};

impl OperatorVTable<FixedSizeListVTable> for FixedSizeListVTable {
    fn bind(
        array: &FixedSizeListArray,
        selection: Option<&ArrayRef>,
        ctx: &mut dyn BindCtx,
    ) -> VortexResult<BatchKernelRef> {
        let validity = ctx.bind_validity(array.validity(), array.len(), selection)?;

        let list_size = array.list_size();
        let elem_dtype = array
            .dtype()
            .as_fixed_size_list_element_opt()
            .vortex_expect("`FixedSizeListArray` `DType` was somehow not `FixedSizeList`")
            .clone();

        // TODO(connor): Should we raise an error if a child kernel returns a data-full `elements`?
        // Technically nothing bad will happen if we don't because of this edge case handling below.

        // If the size of each list is 0, then we know that the child elements must empty. Even if
        // the child kernel incorrectly gives us some data, we can (correctly) just throw it away.
        if list_size == 0 {
            return Ok(kernel(move || {
                let elements = Vector::empty(&elem_dtype);
                let validity_mask = validity.execute()?;

                Ok(
                    FixedSizeListVector::try_new(Arc::new(elements), list_size, validity_mask)?
                        .into(),
                )
            }));
        }

        // Bind the child elements by "expanding" the selection mask out by `list_size` so that we
        // correctly select all of the child elements we need.
        let expanded_selection = expand_selection(selection, list_size);
        let elements_kernel = ctx.bind(array.elements(), expanded_selection.as_ref())?;

        Ok(kernel(move || {
            if list_size != 0 {
                let elements = elements_kernel.execute()?;
                let validity_mask = validity.execute()?;

                Ok(
                    FixedSizeListVector::try_new(Arc::new(elements), list_size, validity_mask)?
                        .into(),
                )
            } else {
                todo!()
            }
        }))
    }
}

/// Takes a selection mask and "expands" it out by duplicating each bit `list_size` times.
///
/// If `selection` is not `None`, the output array is guaranteed to have
/// `selection.len() * list_size` total bits.
fn expand_selection(_selection: Option<&ArrayRef>, _list_size: u32) -> Option<ArrayRef> {
    todo!(
        "TODO(connor)[FixedSizeList]: We need some sort of `ExpandArray` that takes the bits and
        duplicates them, this would be similar to a:
        `RunEndArray(selection_mask, ends=Constant(list_size)`
        (but without depending on the `vortex-runend` encoding crate"
    )
}
