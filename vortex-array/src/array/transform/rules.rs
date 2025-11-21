// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::array::ArrayRef;
use crate::array::transform::context::ArrayRuleContext;
use crate::vtable::VTable;

/// Trait for matching parent array types in parent reduce rules
pub trait ArrayParentMatcher: Send + Sync + 'static {
    type View<'a>;

    /// Try to match the given parent array to this matcher type
    fn try_match(parent: &ArrayRef) -> Option<Self::View<'_>>;
}

/// Matches any parent type (wildcard matcher)
pub struct AnyParent;

impl ArrayParentMatcher for AnyParent {
    type View<'a> = &'a ArrayRef;

    fn try_match(parent: &ArrayRef) -> Option<Self::View<'_>> {
        Some(parent)
    }
}

/// All VTable types can be specific parent matchers
impl<V: VTable> ArrayParentMatcher for V {
    type View<'a> = &'a V::Array;

    fn try_match(parent: &ArrayRef) -> Option<Self::View<'_>> {
        parent.as_opt::<V>()
    }
}

/// A rewrite rule that transforms arrays based on the array itself and its children
pub trait ArrayReduceRule<V: VTable>: Send + Sync {
    /// Attempt to rewrite this array.
    ///
    /// Returns:
    /// - `Ok(Some(new_array))` if the rule applied successfully
    /// - `Ok(None)` if the rule doesn't apply
    /// - `Err(e)` if an error occurred
    fn reduce(&self, array: &V::Array, ctx: &ArrayRuleContext) -> VortexResult<Option<ArrayRef>>;
}

/// A rewrite rule that transforms arrays based on parent context
pub trait ArrayParentReduceRule<Child: VTable, Parent: ArrayParentMatcher>: Send + Sync {
    /// Attempt to rewrite this child array given information about its parent.
    ///
    /// Returns:
    /// - `Ok(Some(new_array))` if the rule applied successfully
    /// - `Ok(None)` if the rule doesn't apply
    /// - `Err(e)` if an error occurred
    fn reduce_parent(
        &self,
        array: &Child::Array,
        parent: Parent::View<'_>,
        child_idx: usize,
        ctx: &ArrayRuleContext,
    ) -> VortexResult<Option<ArrayRef>>;
}
