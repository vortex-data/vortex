// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::Decimal;
use crate::arrays::DecimalArray;
use crate::arrays::Masked;
use crate::arrays::MaskedArray;
use crate::arrays::filter::FilterReduce;
use crate::arrays::filter::FilterReduceAdaptor;
use crate::arrays::slice::SliceReduce;
use crate::arrays::slice::SliceReduceAdaptor;
use crate::match_each_decimal_value_type;
use crate::optimizer::rules::ArrayParentReduceRule;
use crate::optimizer::rules::ParentRuleSet;
use crate::scalar_fn::fns::mask::MaskReduceAdaptor;
use crate::vtable::ValidityHelper;

pub(crate) static RULES: ParentRuleSet<Decimal> = ParentRuleSet::new(&[
    ParentRuleSet::lift(&DecimalMaskedValidityRule),
    ParentRuleSet::lift(&MaskReduceAdaptor(Decimal)),
    ParentRuleSet::lift(&SliceReduceAdaptor(Decimal)),
    ParentRuleSet::lift(&FilterReduceAdaptor(Decimal)),
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
                    array.validity().clone().and(parent.validity().clone())?,
                )
            }
            .into_array()
        });

        Ok(Some(masked_array))
    }
}

impl SliceReduce for Decimal {
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

impl FilterReduce for Decimal {
    fn filter(array: &DecimalArray, mask: &Mask) -> VortexResult<Option<ArrayRef>> {
        let ranges: Vec<Range<usize>> = mask
            .slices()
            .unwrap_or_else(|| unreachable!(), || unreachable!())
            .iter()
            .map(|&(s, e)| s..e)
            .collect();
        let result = match_each_decimal_value_type!(array.values_type(), |D| {
            // SAFETY: Filtering preserves all DecimalArray invariants — values within
            // precision bounds remain valid, and we correctly filter the validity.
            unsafe {
                DecimalArray::new_unchecked_handle(
                    array.buffer_handle().filter_typed::<D>(&ranges)?,
                    array.values_type(),
                    array.decimal_dtype(),
                    array.validity().filter(mask)?,
                )
            }
            .into_array()
        });
        Ok(Some(result))
    }
}
