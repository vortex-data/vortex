// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::MaskedArray;
use crate::arrays::Primitive;
use crate::arrays::PrimitiveArray;
use crate::arrays::slice::SliceReduceAdaptor;
use crate::optimizer::rules::ArrayReduceRule;
use crate::optimizer::rules::ParentRuleSet;
use crate::optimizer::rules::ReduceRuleSet;
use crate::scalar_fn::fns::cast::CastReduceAdaptor;
use crate::scalar_fn::fns::mask::MaskReduceAdaptor;
use crate::validity::Validity;

pub(crate) const RULES: ParentRuleSet<Primitive> = ParentRuleSet::new(&[
    ParentRuleSet::lift(&CastReduceAdaptor(Primitive)),
    ParentRuleSet::lift(&MaskReduceAdaptor(Primitive)),
    ParentRuleSet::lift(&SliceReduceAdaptor(Primitive)),
]);

/// Layer-1 (self-rewrite) reduce rules for [`Primitive`].
pub(crate) const REDUCE_RULES: ReduceRuleSet<Primitive> =
    ReduceRuleSet::new(&[&PrimitiveLiftValidityRule]);

/// Lift a primitive array's per-element validity into a [`MaskedArray`] wrapper.
///
/// This is the inverse of [`super`]'s old pushdown rule: instead of merging a wrapper's validity
/// down into the primitive, definedness is hoisted *out* of the primitive so the child is left as
/// pure (`NonNullable`) data. Only fires for [`Validity::Array`] — the case that actually stores a
/// validity buffer — so all-valid / all-invalid metadata stays embedded and the rule terminates
/// (the lifted child is `NonNullable`, which never re-fires).
#[derive(Default, Debug)]
pub struct PrimitiveLiftValidityRule;

impl ArrayReduceRule<Primitive> for PrimitiveLiftValidityRule {
    fn reduce(&self, array: ArrayView<'_, Primitive>) -> VortexResult<Option<ArrayRef>> {
        let validity = array.validity()?;
        if !matches!(validity, Validity::Array(_)) {
            return Ok(None);
        }

        // SAFETY: dropping validity does not change the values buffer or ptype.
        let pure = unsafe {
            PrimitiveArray::new_unchecked_from_handle(
                array.buffer_handle().clone(),
                array.ptype(),
                Validity::NonNullable,
            )
        };

        Ok(Some(
            MaskedArray::try_new(pure.into_array(), validity)?.into_array(),
        ))
    }
}
