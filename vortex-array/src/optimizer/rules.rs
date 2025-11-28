// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;
use std::marker::PhantomData;

use vortex_error::VortexResult;

use crate::array::ArrayRef;
use crate::vtable::VTable;

/// Trait for matching parent array types in parent reduce rules
pub trait ArrayParentMatcher: Send + Sync + 'static {
    type View<'a>;

    /// Try to match the given parent array to this matcher type
    fn try_match(parent: &ArrayRef) -> Option<Self::View<'_>>;
}

/// Matches any parent type (wildcard matcher)
#[derive(Debug)]
pub struct AnyArrayParent;

impl ArrayParentMatcher for AnyArrayParent {
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
pub trait ArrayReduceRule<V: VTable>: Debug + Send + Sync + 'static {
    /// Attempt to rewrite this array.
    ///
    /// Returns:
    /// - `Ok(Some(new_array))` if the rule applied successfully
    /// - `Ok(None)` if the rule doesn't apply
    /// - `Err(e)` if an error occurred
    fn reduce(&self, array: &V::Array) -> VortexResult<Option<ArrayRef>>;
}

/// A rewrite rule that transforms arrays based on parent context
pub trait ArrayParentReduceRule<Child: VTable, Parent: ArrayParentMatcher>:
    Debug + Send + Sync + 'static
{
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
    ) -> VortexResult<Option<ArrayRef>>;
}

/// Dynamic trait for array reduce rules
pub trait DynArrayReduceRule: Debug + Send + Sync {
    fn reduce(&self, array: &ArrayRef) -> VortexResult<Option<ArrayRef>>;
}

/// Dynamic trait for array parent reduce rules
pub trait DynArrayParentReduceRule: Debug + Send + Sync {
    fn reduce_parent(
        &self,
        array: &ArrayRef,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>>;
}

/// Adapter for ArrayReduceRule
pub(crate) struct ArrayReduceRuleAdapter<V: VTable, R> {
    rule: R,
    _phantom: PhantomData<V>,
}

impl<V: VTable, R> ArrayReduceRuleAdapter<V, R> {
    pub(crate) fn new(rule: R) -> Self {
        Self {
            rule,
            _phantom: PhantomData,
        }
    }
}

impl<V: VTable, R: Debug> Debug for ArrayReduceRuleAdapter<V, R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ArrayReduceRuleAdapter")
            .field("rule", &self.rule)
            .finish()
    }
}

/// Adapter for ArrayParentReduceRule
pub(crate) struct ArrayParentReduceRuleAdapter<Child: VTable, Parent: ArrayParentMatcher, R> {
    rule: R,
    _phantom: PhantomData<(Child, Parent)>,
}

impl<Child: VTable, Parent: ArrayParentMatcher, R> ArrayParentReduceRuleAdapter<Child, Parent, R> {
    pub(crate) fn new(rule: R) -> Self {
        Self {
            rule,
            _phantom: PhantomData,
        }
    }
}

impl<Child: VTable, Parent: ArrayParentMatcher, R: Debug> Debug
    for ArrayParentReduceRuleAdapter<Child, Parent, R>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ArrayParentReduceRuleAdapter")
            .field("rule", &self.rule)
            .finish()
    }
}

impl<V, R> DynArrayReduceRule for ArrayReduceRuleAdapter<V, R>
where
    V: VTable,
    R: ArrayReduceRule<V>,
{
    fn reduce(&self, array: &ArrayRef) -> VortexResult<Option<ArrayRef>> {
        let Some(view) = array.as_opt::<V>() else {
            return Ok(None);
        };
        self.rule.reduce(view)
    }
}

impl<Child, Parent, R> DynArrayParentReduceRule for ArrayParentReduceRuleAdapter<Child, Parent, R>
where
    Child: VTable,
    Parent: ArrayParentMatcher,
    R: ArrayParentReduceRule<Child, Parent>,
{
    fn reduce_parent(
        &self,
        array: &ArrayRef,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        let Some(view) = array.as_opt::<Child>() else {
            return Ok(None);
        };
        let Some(parent_view) = Parent::try_match(parent) else {
            return Ok(None);
        };
        self.rule.reduce_parent(view, parent_view, child_idx)
    }
}
