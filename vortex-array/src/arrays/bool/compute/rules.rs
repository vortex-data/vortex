// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_session::registry::CachedId;

use crate::ArrayRef;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::Bool;
use crate::arrays::BoolArray;
use crate::arrays::Masked;
use crate::arrays::bool::BoolArrayExt;
use crate::arrays::filter::FilterReduceAdaptor;
use crate::arrays::slice::SliceReduceAdaptor;
use crate::optimizer::rules::ArrayParentReduceRule;
use crate::optimizer::rules::ParentRuleDense;
use crate::optimizer::rules::ParentRuleEntry;
use crate::optimizer::rules::ParentRuleSet;
use crate::scalar_fn::fns::cast::CastReduceAdaptor;
use crate::scalar_fn::fns::mask::MaskReduceAdaptor;

static KEYED_PARENT_RULES: [ParentRuleEntry<Bool>; 5] = [
    ParentRuleSet::lift_id(CachedId::new("vortex.masked"), &BoolMaskedValidityRule),
    ParentRuleSet::lift_id(CachedId::new("vortex.cast"), &CastReduceAdaptor(Bool)),
    ParentRuleSet::lift_id(CachedId::new("vortex.mask"), &MaskReduceAdaptor(Bool)),
    ParentRuleSet::lift_id(CachedId::new("vortex.slice"), &SliceReduceAdaptor(Bool)),
    ParentRuleSet::lift_id(CachedId::new("vortex.filter"), &FilterReduceAdaptor(Bool)),
];

static KEYED_PARENT_RULES_DENSE: ParentRuleDense<Bool> = ParentRuleDense::new();

pub(crate) static RULES: ParentRuleSet<Bool> =
    ParentRuleSet::new_indexed(&KEYED_PARENT_RULES, &KEYED_PARENT_RULES_DENSE, &[]);

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
                array.validity()?.and(parent.validity()?)?,
            )
            .into_array(),
        ))
    }
}
