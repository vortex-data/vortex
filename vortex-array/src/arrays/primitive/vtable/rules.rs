// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_dtype::match_each_native_ptype;
use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::MaskedArray;
use crate::arrays::MaskedVTable;
use crate::arrays::PrimitiveArray;
use crate::arrays::PrimitiveVTable;
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
    type Parent = MaskedVTable;

    fn reduce_parent(
        &self,
        array: &PrimitiveArray,
        parent: &MaskedArray,
        _child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        // Merge the parent's validity mask into the child's validity
        // TODO(joe): make this lazy
        let masked_array = match_each_native_ptype!(array.ptype(), |T| {
            // SAFETY: masking validity does not change PrimitiveArray invariants
            unsafe {
                PrimitiveArray::new_unchecked_from_handle(
                    array.buffer_handle().clone(),
                    array.ptype(),
                    array.validity().clone().and(parent.validity().clone()),
                )
            }
            .into_array()
        });

        Ok(Some(masked_array))
    }
}
