// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::BoolArray;
use crate::arrays::BoolVTable;
use crate::arrays::MaskedArray;
use crate::arrays::MaskedVTable;
use crate::arrays::SliceReduceAdaptor;
use crate::optimizer::rules::ArrayParentReduceRule;
use crate::optimizer::rules::ParentRuleSet;
use crate::vtable::ValidityHelper;

pub(crate) const RULES: ParentRuleSet<BoolVTable> = ParentRuleSet::new(&[
    ParentRuleSet::lift(&BoolMaskedValidityRule),
    ParentRuleSet::lift(&SliceReduceAdaptor(BoolVTable)),
]);

/// Rule to push down validity masking from MaskedArray parent into BoolArray child.
///
/// When a BoolArray is wrapped by a MaskedArray, this rule merges the mask's validity
/// with the BoolArray's existing validity, eliminating the need for the MaskedArray wrapper.
#[derive(Default, Debug)]
pub struct BoolMaskedValidityRule;

impl ArrayParentReduceRule<BoolVTable> for BoolMaskedValidityRule {
    type Parent = MaskedVTable;

    fn reduce_parent(
        &self,
        array: &BoolArray,
        parent: &MaskedArray,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        if child_idx > 0 {
            return Ok(None);
        }

        // Merge the parent's validity mask into the child's validity
        // TODO(joe): make this lazy
        Ok(Some(
            BoolArray::new(
                array.to_bit_buffer(),
                array.validity().clone().and(parent.validity().clone()),
            )
            .into_array(),
        ))
    }
}
