// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::array::ArrayView;
use crate::arrays::Reversed;
use crate::arrays::reversed::ReversedArrayExt as _;
use crate::optimizer::rules::ArrayParentReduceRule;
use crate::optimizer::rules::ParentRuleSet;

/// Parent rules for [`ReversedArray`](crate::arrays::ReversedArray).
///
/// Registers the double-reversal cancellation rule: `Reversed(Reversed(x)) → x`.
/// When an inner `ReversedArray` sees another `ReversedArray` as its parent,
/// it returns its own child, eliminating both wrappers with zero data movement.
pub(super) const PARENT_RULES: ParentRuleSet<Reversed> =
    ParentRuleSet::new(&[ParentRuleSet::lift(&DoubleReversalCancelRule)]);

/// Cancels two nested reversals: `Reversed(Reversed(x)) → x`.
#[derive(Debug)]
struct DoubleReversalCancelRule;

impl ArrayParentReduceRule<Reversed> for DoubleReversalCancelRule {
    type Parent = Reversed;

    fn reduce_parent(
        &self,
        array: ArrayView<'_, Reversed>,
        _parent: ArrayView<'_, Reversed>,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        debug_assert_eq!(child_idx, 0, "ReversedArray has exactly one child");
        Ok(Some(array.child().clone()))
    }
}
