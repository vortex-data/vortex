// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::Buffer;
use vortex_dtype::match_each_native_ptype;
use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::MaskedArray;
use crate::arrays::MaskedVTable;
use crate::arrays::PrimitiveArray;
use crate::arrays::PrimitiveVTable;
use crate::optimizer::rules::ArrayParentReduceRule;
use crate::optimizer::rules::Exact;
use crate::vtable::ValidityHelper;

/// Rule to push down validity masking from MaskedArray parent into PrimitiveArray child.
///
/// When a PrimitiveArray is wrapped by a MaskedArray, this rule merges the mask's validity
/// with the PrimitiveArray's existing validity, eliminating the need for the MaskedArray wrapper.
#[derive(Default, Debug)]
pub struct PrimitiveMaskedValidityRule;

impl ArrayParentReduceRule<Exact<PrimitiveVTable>, Exact<MaskedVTable>>
    for PrimitiveMaskedValidityRule
{
    fn child(&self) -> Exact<PrimitiveVTable> {
        Exact::from(&PrimitiveVTable)
    }

    fn parent(&self) -> Exact<MaskedVTable> {
        Exact::from(&MaskedVTable)
    }

    fn reduce_parent(
        &self,
        array: &PrimitiveArray,
        parent: &MaskedArray,
        _child_idx: usize,
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
