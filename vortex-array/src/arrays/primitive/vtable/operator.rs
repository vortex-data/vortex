// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::Buffer;
use vortex_compute::filter::Filter;
use vortex_dtype::match_each_native_ptype;
use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_vector::Vector;
use vortex_vector::primitive::PVector;

use crate::arrays::{MaskedVTable, PrimitiveArray, PrimitiveVTable};
use crate::execution::ExecutionCtx;
use crate::vtable::{OperatorVTable, ValidityHelper};
use crate::{ArrayRef, IntoArray};

impl OperatorVTable<PrimitiveVTable> for PrimitiveVTable {
    fn execute_batch(
        array: &PrimitiveArray,
        selection: &Mask,
        _ctx: &mut dyn ExecutionCtx,
    ) -> VortexResult<Vector> {
        let validity = array.validity_mask().filter(selection);
        match_each_native_ptype!(array.ptype(), |P| {
            let elements = array.buffer::<P>().filter(selection);
            Ok(PVector::<P>::try_new(elements, validity)?.into())
        })
    }

    fn reduce_parent(
        array: &PrimitiveArray,
        parent: &ArrayRef,
        _child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        // Push-down masking of `validity` from the parent `MaskedArray`.
        if let Some(masked) = parent.as_opt::<MaskedVTable>() {
            let masked_array = match_each_native_ptype!(array.ptype(), |T| {
                // SAFETY: Since we are only flipping some bits in the validity, all invariants that
                // were upheld are still upheld.
                unsafe {
                    PrimitiveArray::new_unchecked(
                        Buffer::<T>::from_byte_buffer(array.byte_buffer().clone()),
                        array.validity().clone().and(masked.validity().clone()),
                    )
                }
                .into_array()
            });

            return Ok(Some(masked_array));
        }

        Ok(None)
    }
}
