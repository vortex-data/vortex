// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_dtype::match_each_decimal_value_type;
use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::DecimalArray;
use crate::arrays::DecimalVTable;
use crate::arrays::MaskedArray;
use crate::arrays::MaskedVTable;
use crate::arrays::SliceReduce;
use crate::arrays::SliceReduceAdaptor;
use crate::optimizer::rules::ArrayParentReduceRule;
use crate::optimizer::rules::ParentRuleSet;
use crate::vtable::ValidityHelper;

pub(crate) static RULES: ParentRuleSet<DecimalVTable> = ParentRuleSet::new(&[
    ParentRuleSet::lift(&DecimalMaskedValidityRule),
    ParentRuleSet::lift(&SliceReduceAdaptor(DecimalVTable)),
]);

/// Rule to push down validity masking from MaskedArray parent into DecimalArray child.
///
/// When a DecimalArray is wrapped by a MaskedArray, this rule merges the mask's validity
/// with the DecimalArray's existing validity, eliminating the need for the MaskedArray wrapper.
#[derive(Default, Debug)]
pub struct DecimalMaskedValidityRule;

impl ArrayParentReduceRule<DecimalVTable> for DecimalMaskedValidityRule {
    type Parent = MaskedVTable;

    fn reduce_parent(
        &self,
        array: &DecimalArray,
        parent: &MaskedArray,
        _child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        // Merge the parent's validity mask into the child's validity
        // TODO(joe): make this lazy
        let masked_array = match_each_decimal_value_type!(array.values_type(), |D| {
            // SAFETY: Since we are only flipping some bits in the validity, all invariants that
            // were upheld are still upheld.
            unsafe {
                DecimalArray::new_unchecked(
                    array.buffer::<D>(),
                    array.decimal_dtype(),
                    array.validity().clone().and(parent.validity().clone()),
                )
            }
            .into_array()
        });

        Ok(Some(masked_array))
    }
}

impl SliceReduce for DecimalVTable {
    fn slice(array: &Self::Array, range: Range<usize>) -> VortexResult<Option<ArrayRef>> {
        let result = match_each_decimal_value_type!(array.values_type(), |D| {
            let sliced = array.buffer::<D>().slice(range.clone());
            let validity = array.validity().clone().slice(range)?;
            // SAFETY: Slicing preserves all DecimalArray invariants
            unsafe { DecimalArray::new_unchecked(sliced, array.decimal_dtype(), validity) }
                .into_array()
        });
        Ok(Some(result))
    }
}
