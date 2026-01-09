// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::Buffer;
use vortex_dtype::match_each_native_ptype;
use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::LEGACY_SESSION;
use crate::VortexSessionExecute;
use crate::arrays::MaskedArray;
use crate::arrays::MaskedVTable;
use crate::arrays::PrimitiveArray;
use crate::arrays::PrimitiveVTable;
use crate::matchers::Exact;
use crate::optimizer::rules::ArrayParentReduceRule;
use crate::optimizer::rules::ParentRuleSet;
use crate::vtable::ValidityHelper;

pub(super) const RULES: ParentRuleSet<PrimitiveVTable> =
    ParentRuleSet::new(&[ParentRuleSet::lift(&PrimitiveMaskedValidityRule)]);

/// Rule to push down validity masking from MaskedArray parent into PrimitiveArray child.
///
/// When a PrimitiveArray is wrapped by a MaskedArray, this rule merges the mask's validity
/// with the PrimitiveArray's existing validity, eliminating the need for the MaskedArray wrapper.
#[derive(Default, Debug)]
pub struct PrimitiveMaskedValidityRule;

impl ArrayParentReduceRule<PrimitiveVTable> for PrimitiveMaskedValidityRule {
    type Parent = Exact<MaskedVTable>;

    fn parent(&self) -> Exact<MaskedVTable> {
        Exact::from(&MaskedVTable)
    }

    fn reduce_parent(
        &self,
        array: &PrimitiveArray,
        parent: &MaskedArray,
        _child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        let ctx = LEGACY_SESSION.create_execution_ctx();
        // Merge the parent's validity mask into the child's validity
        // TODO(joe): make this lazy
        let masked_array = match_each_native_ptype!(array.ptype(), |T| {
            // SAFETY: Since we are only flipping some bits in the validity, all invariants that
            // were upheld are still upheld.
            unsafe {
                PrimitiveArray::new_unchecked(
                    Buffer::<T>::from_byte_buffer(array.buffer_handle(&ctx).bytes().clone()),
                    array.validity().clone().and(parent.validity().clone()),
                )
            }
            .into_array()
        });

        Ok(Some(masked_array))
    }
}
