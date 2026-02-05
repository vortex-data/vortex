// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::type_name;
use std::fmt::Debug;
use std::marker::PhantomData;

use vortex_error::VortexResult;

use crate::array::ArrayRef;
use crate::matcher::Matcher;
use crate::vtable::VTable;

/// A rewrite rule that transforms arrays based on their own content
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
pub trait ArrayParentReduceRule<V: VTable>: Debug + Send + Sync + 'static {
    type Parent: Matcher;

    /// Attempt to rewrite this child array given information about its parent.
    ///
    /// Returns:
    /// - `Ok(Some(new_array))` if the rule applied successfully
    /// - `Ok(None)` if the rule doesn't apply
    /// - `Err(e)` if an error occurred
    fn reduce_parent(
        &self,
        array: &V::Array,
        parent: <Self::Parent as Matcher>::Match<'_>,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>>;
}

/// Dynamic trait for array parent reduce rules
pub trait DynArrayParentReduceRule<V: VTable>: Debug + Send + Sync {
    fn matches(&self, parent: &ArrayRef) -> bool;

    fn reduce_parent(
        &self,
        array: &V::Array,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>>;
}

/// Adapter for ArrayParentReduceRule
pub struct ParentReduceRuleAdapter<V, R> {
    rule: R,
    _phantom: PhantomData<V>,
}

impl<V: VTable, R: ArrayParentReduceRule<V>> Debug for ParentReduceRuleAdapter<V, R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ArrayParentReduceRuleAdapter")
            .field("parent", &type_name::<R::Parent>())
            .field("rule", &self.rule)
            .finish()
    }
}

impl<V: VTable, K: ArrayParentReduceRule<V>> DynArrayParentReduceRule<V>
    for ParentReduceRuleAdapter<V, K>
{
    fn matches(&self, parent: &ArrayRef) -> bool {
        K::Parent::matches(parent)
    }

    fn reduce_parent(
        &self,
        child: &V::Array,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        let Some(parent_view) = K::Parent::try_match(parent) else {
            return Ok(None);
        };
        self.rule.reduce_parent(child, parent_view, child_idx)
    }
}

pub struct ReduceRuleSet<V: VTable> {
    rules: &'static [&'static dyn ArrayReduceRule<V>],
}

impl<V: VTable> ReduceRuleSet<V> {
    /// Create a new reduction rule set with the given rules.
    pub const fn new(rules: &'static [&'static dyn ArrayReduceRule<V>]) -> Self {
        Self { rules }
    }

    /// Evaluate the reduction rules on the given array.
    pub fn evaluate(&self, array: &V::Array) -> VortexResult<Option<ArrayRef>> {
        for rule in self.rules.iter() {
            if let Some(reduced) = rule.reduce(array)? {
                return Ok(Some(reduced));
            }
        }
        Ok(None)
    }
}

/// A set of parent reduction rules for a specific child array encoding.
pub struct ParentRuleSet<V: VTable> {
    rules: &'static [&'static dyn DynArrayParentReduceRule<V>],
}

impl<V: VTable> ParentRuleSet<V> {
    /// Create a new parent rule set with the given rules.
    ///
    /// Use [`ParentRuleSet::lift`] to lift static rules into dynamic trait objects.
    pub const fn new(rules: &'static [&'static dyn DynArrayParentReduceRule<V>]) -> Self {
        Self { rules }
    }

    /// Lift the given rule into a dynamic trait object.
    pub const fn lift<R: ArrayParentReduceRule<V>>(
        rule: &'static R,
    ) -> &'static dyn DynArrayParentReduceRule<V> {
        // Assert that self is zero-sized
        const {
            assert!(
                !(size_of::<R>() != 0),
                "Rule must be zero-sized to be lifted"
            );
        }
        unsafe { &*(rule as *const R as *const ParentReduceRuleAdapter<V, R>) }
    }

    /// Evaluate the parent reduction rules on the given child and parent arrays.
    pub fn evaluate(
        &self,
        child: &V::Array,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        for rule in self.rules.iter() {
            if !rule.matches(parent) {
                continue;
            }
            if let Some(reduced) = rule.reduce_parent(child, parent, child_idx)? {
                // Debug assertions because these checks are already run elsewhere.
                #[cfg(debug_assertions)]
                {
                    vortex_error::vortex_ensure!(
                        reduced.len() == parent.len(),
                        "Reduced array length mismatch from {:?}\nFrom:\n{}\nTo:\n{}",
                        rule,
                        parent.encoding_id(),
                        reduced.encoding_id()
                    );
                    vortex_error::vortex_ensure!(
                        reduced.dtype() == parent.dtype(),
                        "Reduced array dtype mismatch from {:?}\nFrom:\n{}\nTo:\n{}",
                        rule,
                        parent.encoding_id(),
                        reduced.encoding_id()
                    );
                }

                return Ok(Some(reduced));
            }
        }
        Ok(None)
    }
}
