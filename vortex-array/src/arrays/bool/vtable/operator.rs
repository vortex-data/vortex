// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_compute::filter::Filter;
use vortex_error::VortexResult;
use vortex_vector::bool::BoolVector;

use crate::array::transform::{ArrayParentReduceRule, ArrayRuleContext};
use crate::arrays::{BoolArray, BoolVTable, MaskedArray, MaskedVTable};
use crate::execution::{BatchKernelRef, BindCtx, kernel};
use crate::vtable::{OperatorVTable, ValidityHelper};
use crate::{ArrayRef, IntoArray};

impl OperatorVTable<BoolVTable> for BoolVTable {
    fn bind(
        array: &BoolArray,
        selection: Option<&ArrayRef>,
        ctx: &mut dyn BindCtx,
    ) -> VortexResult<BatchKernelRef> {
        let bits = array.buffer.clone();
        let mask = ctx.bind_selection(array.len(), selection)?;
        let validity = ctx.bind_validity(array.validity(), array.len(), selection)?;

        Ok(kernel(move || {
            let mask = mask.execute()?;
            let validity = validity.execute()?;

            // Note that validity already has the mask applied so we only need to apply it to bits.
            let bits = bits.filter(&mask);

            Ok(BoolVector::try_new(bits, validity)?.into())
        }))
    }
}

/// Rule to push down validity masking from MaskedArray parent into BoolArray child.
///
/// When a BoolArray is wrapped by a MaskedArray, this rule merges the mask's validity
/// with the BoolArray's existing validity, eliminating the need for the MaskedArray wrapper.
pub struct BoolMaskedValidityRule;

impl ArrayParentReduceRule<BoolVTable, MaskedVTable> for BoolMaskedValidityRule {
    fn reduce_parent(
        &self,
        array: &BoolArray,
        parent: &MaskedArray,
        _child_idx: usize,
        _ctx: &ArrayRuleContext,
    ) -> VortexResult<Option<ArrayRef>> {
        // Merge the parent's validity mask into the child's validity
        // TODO(joe): make this lazy
        Ok(Some(
            BoolArray::from_bit_buffer(
                array.bit_buffer().clone(),
                array.validity().clone().and(parent.validity().clone()),
            )
            .into_array(),
        ))
    }
}
