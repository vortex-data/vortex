// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_compute::filter::Filter;
use vortex_error::VortexResult;
use vortex_vector::bool::BoolVector;

use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::BoolArray;
use crate::arrays::BoolVTable;
use crate::arrays::MaskedArray;
use crate::arrays::MaskedVTable;
use crate::execution::BatchKernelRef;
use crate::execution::BindCtx;
use crate::execution::kernel;
use crate::transform::ArrayParentReduceRule;
use crate::transform::ArrayRuleContext;
use crate::vtable::OperatorVTable;
use crate::vtable::ValidityHelper;

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
#[derive(Default, Debug)]
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
