// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::TypeId;
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
    fn key() -> MatchKey;

    /// Try to match the given array to this matcher type
    fn try_match(array: &ArrayRef) -> Option<Self::View<'_>>;
}

/// A key used to look up a subset of rules in a rule registry
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum RuleKey {
    /// Reduce an array.
    Reduce(MatchKey),
    /// Reduce an array with its parent.
    ReduceParent { parent: MatchKey, child: MatchKey },
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum MatchKey {
    Any,
    Type(TypeId),
    ArrayId(ArrayId),
}

/// Matches any array type (wildcard matcher)
#[derive(Debug)]
pub struct AnyArray;
impl Matcher for AnyArray {
    type View<'a> = &'a ArrayRef;

    fn key() -> MatchKey {
        MatchKey::Any
    }

    fn try_match(parent: &ArrayRef) -> Option<Self::View<'_>> {
        Some(parent)
    }
}

/// Matches a specific Array by its VTable type.
#[derive(Debug)]
pub struct Exact<V: VTable>(PhantomData<V>);
impl<V: VTable> Matcher for Exact<V> {
    type View<'a> = &'a V::Array;

    fn key() -> MatchKey {
        MatchKey::Type(TypeId::of::<V>())
    }

    fn try_match(parent: &ArrayRef) -> Option<Self::View<'_>> {
        parent.as_opt::<V>()
    }
}

/// A rewrite rule that transforms arrays based on the array itself and its children
pub trait ArrayReduceRule<M: Matcher>: Debug + Send + Sync + 'static {
    /// Attempt to rewrite this array.
    ///
    /// Returns:
    /// - `Ok(Some(new_array))` if the rule applied successfully
    /// - `Ok(None)` if the rule doesn't apply
    /// - `Err(e)` if an error occurred
    fn reduce(&self, array: M::View<'_>) -> VortexResult<Option<ArrayRef>>;
}

/// A rewrite rule that transforms arrays based on parent context
pub trait ArrayParentReduceRule<Child: Matcher, Parent: Matcher>:
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
        child: Child::View<'_>,
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
pub(crate) struct ArrayReduceRuleAdapter<M: Matcher, R> {
    rule: R,
    _phantom: PhantomData<M>,
}

impl<M: Matcher, R> ArrayReduceRuleAdapter<M, R> {
    pub(crate) fn new(rule: R) -> Self {
        Self {
            rule,
            _phantom: PhantomData,
        }
    }
}

impl<M: Matcher, R: Debug> Debug for ArrayReduceRuleAdapter<M, R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ArrayReduceRuleAdapter")
            .field("matcher", &type_name::<M>())
            .field("rule", &self.rule)
            .finish()
    }
}

/// Adapter for ArrayParentReduceRule
pub(crate) struct ArrayParentReduceRuleAdapter<Child: Matcher, Parent: Matcher, R> {
    rule: R,
    _phantom: PhantomData<(Child, Parent)>,
}

impl<Child: Matcher, Parent: Matcher, R> ArrayParentReduceRuleAdapter<Child, Parent, R> {
    pub(crate) fn new(rule: R) -> Self {
        Self {
            rule,
            _phantom: PhantomData,
        }
    }
}

impl<Child: Matcher, Parent: Matcher, R: Debug> Debug
    for ArrayParentReduceRuleAdapter<Child, Parent, R>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ArrayParentReduceRuleAdapter")
            .field("rule", &self.rule)
            .finish()
    }
}

impl<M: Matcher, R: ArrayReduceRule<M>> DynArrayReduceRule for ArrayReduceRuleAdapter<M, R> {
    fn reduce(&self, array: &ArrayRef) -> VortexResult<Option<ArrayRef>> {
        let Some(view) = M::try_match(array) else {
            return Ok(None);
        };
        self.rule.reduce(view)
    }
}

impl<Child, Parent, R> DynArrayParentReduceRule for ArrayParentReduceRuleAdapter<Child, Parent, R>
where
    Child: Matcher,
    Parent: Matcher,
    R: ArrayParentReduceRule<Child, Parent>,
{
    fn reduce_parent(
        &self,
        child: &ArrayRef,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        let Some(child_view) = Child::try_match(child) else {
            return Ok(None);
        };
        let Some(parent_view) = Parent::try_match(parent) else {
            return Ok(None);
        };
        self.rule.reduce_parent(child_view, parent_view, child_idx)
    }
}
