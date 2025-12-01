// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::BoolArray;
use crate::arrays::BoolVTable;
use crate::arrays::MaskedArray;
use crate::arrays::MaskedVTable;
use crate::optimizer::rules::ArrayParentReduceRule;
use crate::optimizer::rules::Exact;
use crate::vtable::ValidityHelper;

/// Rule to push down validity masking from MaskedArray parent into BoolArray child.
///
/// When a BoolArray is wrapped by a MaskedArray, this rule merges the mask's validity
/// with the BoolArray's existing validity, eliminating the need for the MaskedArray wrapper.
#[derive(Default, Debug)]
pub struct BoolMaskedValidityRule;

impl ArrayParentReduceRule<Exact<BoolVTable>, Exact<MaskedVTable>> for BoolMaskedValidityRule {
    fn child(&self) -> Exact<BoolVTable> {
        Exact::from(&BoolVTable)
    }

    fn parent(&self) -> Exact<MaskedVTable> {
        Exact::from(&MaskedVTable)
    }

    fn reduce_parent(
        &self,
        array: &BoolArray,
        parent: &MaskedArray,
        _child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        // Merge the parent's validity mask into the child's validity
        // TODO(joe): make this lazy
        Ok(Some(
            BoolArray::from_bit_buffer(
                array.bit_buffer().clone(),
                array.validity().clone().and(parent.validity().clone())?,
            )
            .into_array(),
        ))
    }
}
