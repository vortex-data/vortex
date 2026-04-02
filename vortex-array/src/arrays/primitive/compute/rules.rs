// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::Masked;
use crate::arrays::Primitive;
use crate::arrays::PrimitiveArray;
use crate::arrays::slice::SliceReduceAdaptor;
use crate::optimizer::rules::ArrayParentReduceRule;
use crate::optimizer::rules::ParentRuleSet;
use crate::scalar_fn::fns::mask::MaskReduceAdaptor;

pub(crate) const RULES: ParentRuleSet<Primitive> = ParentRuleSet::new(&[
    ParentRuleSet::lift(&PrimitiveMaskedValidityRule),
    ParentRuleSet::lift(&MaskReduceAdaptor(Primitive)),
    ParentRuleSet::lift(&SliceReduceAdaptor(Primitive)),
]);

/// Rule to push down validity masking from MaskedArray parent into PrimitiveArray child.
///
/// When a PrimitiveArray is wrapped by a MaskedArray, this rule merges the mask's validity
/// with the PrimitiveArray's existing validity, eliminating the need for the MaskedArray wrapper.
#[derive(Default, Debug)]
pub struct PrimitiveMaskedValidityRule;

impl ArrayParentReduceRule<Primitive> for PrimitiveMaskedValidityRule {
    type Parent = Masked;

    fn reduce_parent(
        &self,
        array: ArrayView<'_, Primitive>,
        parent: ArrayView<'_, Masked>,
        _child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        // TODO(joe): make this lazy
        // Merge the parent's validity mask into the child's validity
        let new_validity = array.validity().and(parent.validity())?;

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
