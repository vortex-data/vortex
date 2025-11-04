// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_compute::filter::Filter;
use vortex_dtype::match_each_native_ptype;
use vortex_error::VortexResult;
use vortex_vector::primitive::PVector;

use crate::arrays::{MaskedVTable, PrimitiveArray, PrimitiveVTable};
use crate::execution::{BatchKernelRef, BindCtx, kernel};
use crate::vtable::{OperatorVTable, ValidityHelper};
use crate::{ArrayRef, IntoArray};

impl OperatorVTable<PrimitiveVTable> for PrimitiveVTable {
    fn bind(
        array: &PrimitiveArray,
        selection: Option<&ArrayRef>,
        ctx: &mut dyn BindCtx,
    ) -> VortexResult<BatchKernelRef> {
        let mask = ctx.bind_selection(array.len(), selection)?;
        let validity = ctx.bind_validity(array.validity(), array.len(), selection)?;

        match_each_native_ptype!(array.ptype(), |T| {
            let elements = array.buffer::<T>();
            Ok(kernel(move || {
                let mask = mask.execute()?;
                let validity = validity.execute()?;

                // Note that validity already has the mask applied so we only need to apply it to
                // the elements.
                let elements = elements.filter(&mask);

                Ok(PVector::try_new(elements, validity)?.into())
            }))
        })
    }

    fn reduce_parent(
        array: &PrimitiveArray,
        parent: &ArrayRef,
        _child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        // Push-down masking of validity from parent MaskedVTable.
        if let Some(masked) = parent.as_opt::<MaskedVTable>() {
            return Ok(Some(
                PrimitiveArray::from_byte_buffer(
                    array.byte_buffer().clone(),
                    array.ptype(),
                    array.validity().clone().and(masked.validity().clone()),
                )
                .into_array(),
            ));
        }

        Ok(None)
    }
}
