// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::Bool;
use crate::arrays::BoolArray;
use crate::arrays::Masked;
use crate::arrays::filter::FilterReduceAdaptor;
use crate::arrays::slice::SliceReduceAdaptor;
use crate::optimizer::rules::ArrayParentReduceRule;
use crate::optimizer::rules::ParentRuleSet;
use crate::scalar_fn::fns::cast::CastReduceAdaptor;
use crate::scalar_fn::fns::mask::MaskReduceAdaptor;

pub(crate) const RULES: ParentRuleSet<Bool> = ParentRuleSet::new(&[
    ParentRuleSet::lift(&BoolMaskedValidityRule),
    ParentRuleSet::lift(&CastReduceAdaptor(Bool)),
    ParentRuleSet::lift(&MaskReduceAdaptor(Bool)),
    ParentRuleSet::lift(&SliceReduceAdaptor(Bool)),
    ParentRuleSet::lift(&FilterReduceAdaptor(Bool)),
]);

/// Rule to push down validity masking from MaskedArray parent into BoolArray child.
///
/// When a BoolArray is wrapped by a MaskedArray, this rule merges the mask's validity
/// with the BoolArray's existing validity, eliminating the need for the MaskedArray wrapper.
#[derive(Default, Debug)]
pub struct BoolMaskedValidityRule;

impl ArrayParentReduceRule<Bool> for BoolMaskedValidityRule {
    type Parent = Masked;

    fn reduce_parent(
        &self,
        array: ArrayView<'_, Bool>,
        parent: ArrayView<'_, Masked>,
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
                array.validity().and(parent.validity())?,
            )
            .into_array(),
        ))
    }
}
