// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::MaskedArray;
use crate::arrays::MaskedVTable;
use crate::arrays::PrimitiveArray;
use crate::arrays::PrimitiveVTable;
use crate::arrays::SliceReduceAdaptor;
use crate::expr::MaskReduceAdaptor;
use crate::optimizer::rules::ArrayParentReduceRule;
use crate::optimizer::rules::ParentRuleSet;
use crate::vtable::ValidityHelper;

pub(crate) const RULES: ParentRuleSet<PrimitiveVTable> = ParentRuleSet::new(&[
    ParentRuleSet::lift(&PrimitiveMaskedValidityRule),
    ParentRuleSet::lift(&MaskReduceAdaptor(PrimitiveVTable)),
    ParentRuleSet::lift(&SliceReduceAdaptor(PrimitiveVTable)),
]);

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
        // TODO(joe): make this lazy
        // Merge the parent's validity mask into the child's validity
        let new_validity = array.validity().clone().and(parent.validity().clone())?;

        // SAFETY: masking validity does not change PrimitiveArray invariants
        let masked_array = unsafe {
            PrimitiveArray::new_unchecked_from_handle(
                array.buffer_handle().clone(),
                array.ptype(),
                new_validity,
            )
        };

        Ok(Some(masked_array.into_array()))
    }
}
