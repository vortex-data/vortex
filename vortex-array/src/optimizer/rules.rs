// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::type_name;
use std::fmt::Debug;
use std::marker::PhantomData;

use vortex_error::VortexResult;

use crate::array::ArrayRef;
use crate::vtable::ArrayId;
use crate::vtable::VTable;

/// Trait for matching array types in optimizer rules
pub trait Matcher: Send + Sync + 'static {
    type View<'a>;

    /// Return the key for this matcher
    fn key(&self) -> MatchKey;

    /// Try to match the given array to this matcher type
    fn try_match<'a>(&self, array: &'a ArrayRef) -> Option<Self::View<'a>>;
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum MatchKey {
    Any,
    Array(ArrayId),
}

/// Matches any array type (wildcard matcher)
#[derive(Debug)]
pub struct AnyArray;
impl Matcher for AnyArray {
    type View<'a> = &'a ArrayRef;

    fn key(&self) -> MatchKey {
        MatchKey::Any
    }

    fn try_match<'a>(&self, array: &'a ArrayRef) -> Option<Self::View<'a>> {
        Some(array)
    }
}

/// Matches a specific Array by its encoding ID.
#[derive(Debug)]
pub struct Exact<V: VTable> {
    id: ArrayId,
    _phantom: PhantomData<V>,
}
impl<V: VTable> Matcher for Exact<V> {
    type View<'a> = &'a V::Array;

    fn key(&self) -> MatchKey {
        MatchKey::Array(self.id.clone())
    }

    fn try_match<'a>(&self, parent: &'a ArrayRef) -> Option<Self::View<'a>> {
        parent.as_opt::<V>()
    }
}
impl<V: VTable> Exact<V> {
    /// Create a new Exact matcher for the given ArrayId.
    ///
    /// # Safety
    ///
    /// The optimizer will attempt to downcast the array to the type V when matching.
    /// If an array with the given ID does not match type V, the rule will silently not be applied.
    pub unsafe fn new_unchecked(id: ArrayId) -> Self {
        Self {
            id,
            _phantom: PhantomData,
        }
    }
}
impl<V: VTable> From<&'static V> for Exact<V> {
    fn from(vtable: &'static V) -> Self {
        Self {
            id: vtable.id(),
            _phantom: PhantomData,
        }
    }
}

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

    /// Returns the matcher for the parent array
    fn parent(&self) -> Self::Parent;

    /// Attempt to rewrite this child array given information about its parent.
    ///
    /// Returns:
    /// - `Ok(Some(new_array))` if the rule applied successfully
    /// - `Ok(None)` if the rule doesn't apply
    /// - `Err(e)` if an error occurred
    fn reduce_parent(
        &self,
        child: &V::Array,
        parent: <Self::Parent as Matcher>::View<'_>,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>>;
}

/// Dynamic trait for array parent reduce rules
pub trait DynArrayParentReduceRule<V: VTable>: Debug + Send + Sync {
    fn parent_key(&self) -> MatchKey;

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

impl<V: VTable, R: ArrayParentReduceRule<V>> DynArrayParentReduceRule<V>
    for ParentReduceRuleAdapter<V, R>
{
    fn parent_key(&self) -> MatchKey {
        self.rule.parent().key()
    }

    fn reduce_parent(
        &self,
        child: &V::Array,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        let Some(parent_view) = self.rule.parent().try_match(parent) else {
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
            if let MatchKey::Array(id) = rule.parent_key()
                && parent.encoding_id() != id
            {
                continue;
            }
            if let Some(reduced) = rule.reduce_parent(child, parent, child_idx)? {
                return Ok(Some(reduced));
            }
        }
        Ok(None)
    }
}
