// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::Decimal;
use crate::arrays::DecimalArray;
use crate::arrays::Masked;
use crate::arrays::slice::SliceReduce;
use crate::arrays::slice::SliceReduceAdaptor;
use crate::match_each_decimal_value_type;
use crate::optimizer::rules::ArrayParentReduceRule;
use crate::optimizer::rules::ParentRuleSet;
use crate::scalar_fn::fns::mask::MaskReduceAdaptor;

pub(crate) static RULES: ParentRuleSet<Decimal> = ParentRuleSet::new(&[
    ParentRuleSet::lift(&DecimalMaskedValidityRule),
    ParentRuleSet::lift(&MaskReduceAdaptor(Decimal)),
    ParentRuleSet::lift(&SliceReduceAdaptor(Decimal)),
]);

/// Rule to push down validity masking from MaskedArray parent into DecimalArray child.
///
/// When a DecimalArray is wrapped by a MaskedArray, this rule merges the mask's validity
/// with the DecimalArray's existing validity, eliminating the need for the MaskedArray wrapper.
#[derive(Default, Debug)]
pub struct DecimalMaskedValidityRule;

impl ArrayParentReduceRule<Decimal> for DecimalMaskedValidityRule {
    type Parent = Masked;

    fn reduce_parent(
        &self,
        array: ArrayView<'_, Decimal>,
        parent: ArrayView<'_, Masked>,
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
                    array.validity().and(parent.validity())?,
                )
            }
            .into_array()
        });

        Ok(Some(masked_array))
    }
}

impl SliceReduce for Decimal {
    fn slice(array: ArrayView<'_, Self>, range: Range<usize>) -> VortexResult<Option<ArrayRef>> {
        let result = match_each_decimal_value_type!(array.values_type(), |D| {
            let sliced = array.buffer::<D>().slice(range.clone());
            let validity = array.validity().slice(range)?;
            // SAFETY: Slicing preserves all DecimalArray invariants
            unsafe { DecimalArray::new_unchecked(sliced, array.decimal_dtype(), validity) }
                .into_array()
        });
        Ok(Some(result))
    }
}
