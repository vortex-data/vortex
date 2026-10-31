// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_session::VortexSession;

use crate::ArrayRef;
use crate::ArrayVTable;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::Masked;
use crate::arrays::Primitive;
use crate::arrays::PrimitiveArray;
use crate::arrays::Slice;
use crate::arrays::slice::SliceReduceAdaptor;
use crate::optimizer::kernels::ArrayKernelsExt;
use crate::optimizer::rules::ArrayParentReduceRule;
use crate::scalar_fn::ScalarFnVTable;
use crate::scalar_fn::fns::cast::Cast;
use crate::scalar_fn::fns::cast::CastReduceAdaptor;
use crate::scalar_fn::fns::mask::Mask as MaskFn;
use crate::scalar_fn::fns::mask::MaskReduceAdaptor;

/// Registers the parent-reduce rules for [`Primitive`] children in the session kernels.
pub(crate) fn register_parent_rules(session: &VortexSession) {
    let kernels = session.kernels();
    kernels.register_reduce_parent_rule(Masked.id(), Primitive, PrimitiveMaskedValidityRule);
    kernels.register_reduce_parent_rule(Cast.id(), Primitive, CastReduceAdaptor(Primitive));
    kernels.register_reduce_parent_rule(MaskFn.id(), Primitive, MaskReduceAdaptor(Primitive));
    kernels.register_reduce_parent_rule(Slice.id(), Primitive, SliceReduceAdaptor(Primitive));
}

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
        let new_validity = array.validity()?.and(parent.validity()?)?;

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
