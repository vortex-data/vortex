// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_compute::filter::Filter;
use vortex_dtype::{PrecisionScale, match_each_decimal_value_type};
use vortex_error::VortexResult;
use vortex_vector::decimal::DVector;

use crate::array::transform::{ArrayParentReduceRule, ArrayRuleContext};
use crate::arrays::{DecimalArray, DecimalVTable, MaskedArray, MaskedVTable};
use crate::execution::{BatchKernelRef, BindCtx, kernel};
use crate::vtable::{OperatorVTable, ValidityHelper};
use crate::{ArrayRef, IntoArray};

impl OperatorVTable<DecimalVTable> for DecimalVTable {
    fn bind(
        array: &DecimalArray,
        selection: Option<&ArrayRef>,
        ctx: &mut dyn BindCtx,
    ) -> VortexResult<BatchKernelRef> {
        let mask = ctx.bind_selection(array.len(), selection)?;
        let validity = ctx.bind_validity(array.validity(), array.len(), selection)?;

        match_each_decimal_value_type!(array.values_type(), |D| {
            let elements = array.buffer::<D>();
            let ps = PrecisionScale::<D>::try_from(&array.decimal_dtype())?;

            Ok(kernel(move || {
                let mask = mask.execute()?;
                let validity = validity.execute()?;

                // Note that validity already has the mask applied so we only need to apply it to
                // the elements.
                let elements = elements.filter(&mask);

                Ok(DVector::<D>::try_new(ps, elements, validity)?.into())
            }))
        })
    }
}

/// Rule to push down validity masking from MaskedArray parent into DecimalArray child.
///
/// When a DecimalArray is wrapped by a MaskedArray, this rule merges the mask's validity
/// with the DecimalArray's existing validity, eliminating the need for the MaskedArray wrapper.
pub struct DecimalMaskedValidityRule;

impl ArrayParentReduceRule<DecimalVTable, MaskedVTable> for DecimalMaskedValidityRule {
    fn reduce_parent(
        &self,
        array: &DecimalArray,
        parent: &MaskedArray,
        _child_idx: usize,
        _ctx: &ArrayRuleContext,
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
