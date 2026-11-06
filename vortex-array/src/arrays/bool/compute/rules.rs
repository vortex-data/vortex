// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_session::VortexSession;

use crate::ArrayRef;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::Bool;
use crate::arrays::BoolArray;
use crate::arrays::Filter;
use crate::arrays::Masked;
use crate::arrays::Slice;
use crate::arrays::bool::BoolArrayExt;
use crate::arrays::filter::FilterReduceAdaptor;
use crate::arrays::slice::SliceReduceAdaptor;
use crate::optimizer::kernels::ArrayKernelsExt;
use crate::optimizer::rules::ArrayParentReduceRule;
use crate::scalar_fn::fns::cast::Cast;
use crate::scalar_fn::fns::cast::CastReduceAdaptor;
use crate::scalar_fn::fns::mask::Mask as MaskFn;
use crate::scalar_fn::fns::mask::MaskReduceAdaptor;

pub(crate) fn register_parent_rules(session: &VortexSession) {
    let kernels = session.kernels();
    kernels.register_reduce_parent_rule(Masked, Bool, const { &BoolMaskedValidityRule });
    kernels.register_reduce_parent_rule(Cast, Bool, const { &CastReduceAdaptor(Bool) });
    kernels.register_reduce_parent_rule(MaskFn, Bool, const { &MaskReduceAdaptor(Bool) });
    kernels.register_reduce_parent_rule(Slice, Bool, const { &SliceReduceAdaptor(Bool) });
    kernels.register_reduce_parent_rule(Filter, Bool, const { &FilterReduceAdaptor(Bool) });
}

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

        let bit_buffer = array.to_bit_buffer();
        // Merge the parent's validity mask into the child's validity
        // TODO(joe): make this lazy
        let validity = array.validity()?.and(parent.validity()?)?;

        // Safety:
        // we know all elements are valid, the AND operation will fail if mismatched.
        let array = unsafe { BoolArray::new_unchecked(bit_buffer, validity).into_array() };

        Ok(Some(array))
    }
}
