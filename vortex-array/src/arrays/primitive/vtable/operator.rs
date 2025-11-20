// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::Buffer;
use vortex_compute::filter::Filter;
use vortex_dtype::match_each_native_ptype;
use vortex_error::VortexResult;
use vortex_vector::primitive::PVector;

use crate::array::transform::{ArrayParentReduceRule, ArrayRuleContext};
use crate::arrays::{MaskedArray, MaskedVTable, PrimitiveArray, PrimitiveVTable};
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

        match_each_native_ptype!(array.ptype(), |P| {
            let elements = array.buffer::<P>();
            Ok(kernel(move || {
                let mask = mask.execute()?;
                let validity = validity.execute()?;

                // Note that validity already has the mask applied so we only need to apply it to
                // the elements.
                let elements = elements.filter(&mask);

                Ok(PVector::<P>::try_new(elements, validity)?.into())
            }))
        })
    }
}

/// Rule to push down validity masking from MaskedArray parent into PrimitiveArray child.
///
/// When a PrimitiveArray is wrapped by a MaskedArray, this rule merges the mask's validity
/// with the PrimitiveArray's existing validity, eliminating the need for the MaskedArray wrapper.
pub struct PrimitiveMaskedValidityRule;

impl ArrayParentReduceRule<PrimitiveVTable, MaskedVTable> for PrimitiveMaskedValidityRule {
    fn reduce_parent(
        &self,
        array: &PrimitiveArray,
        parent: &MaskedArray,
        _child_idx: usize,
        _ctx: &ArrayRuleContext,
    ) -> VortexResult<Option<ArrayRef>> {
        // Merge the parent's validity mask into the child's validity
        // TODO(joe): make this lazy
        let masked_array = match_each_native_ptype!(array.ptype(), |T| {
            // SAFETY: Since we are only flipping some bits in the validity, all invariants that
            // were upheld are still upheld.
            unsafe {
                PrimitiveArray::new_unchecked(
                    Buffer::<T>::from_byte_buffer(array.byte_buffer().clone()),
                    array.validity().clone().and(parent.validity().clone()),
                )
            }
            .into_array()
        });

        Ok(Some(masked_array))
    }
}
