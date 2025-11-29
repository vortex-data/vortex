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

/// A rewrite rule that transforms arrays based on the array itself and its children
pub trait ArrayReduceRule<M: Matcher>: Debug + Send + Sync + 'static {
    /// Returns the matcher for this rule
    fn matcher(&self) -> M;

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
    /// Returns the matcher for the child array
    fn child(&self) -> Child;

    /// Returns the matcher for the parent array
    fn parent(&self) -> Parent;

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
    fn key(&self) -> MatchKey;

    fn reduce(&self, array: &ArrayRef) -> VortexResult<Option<ArrayRef>>;
}

/// Dynamic trait for array parent reduce rules
pub trait DynArrayParentReduceRule: Debug + Send + Sync {
    fn child_key(&self) -> MatchKey;

    fn parent_key(&self) -> MatchKey;

    fn reduce_parent(
        &self,
        array: &ArrayRef,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>>;
}

/// Adapter for ArrayReduceRule
pub(crate) struct ReduceRuleAdapter<M, R> {
    rule: R,
    _phantom: PhantomData<M>,
}

impl<M, R> ReduceRuleAdapter<M, R> {
    pub(crate) fn new(rule: R) -> Self {
        Self {
            rule,
            _phantom: PhantomData,
        }
    }
}

impl<M: Matcher, R: ArrayReduceRule<M>> Debug for ReduceRuleAdapter<M, R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ArrayReduceRuleAdapter")
            .field("matcher", &type_name::<M>())
            .field("rule", &self.rule)
            .finish()
    }
}

/// Adapter for ArrayParentReduceRule
pub(crate) struct ParentReduceRuleAdapter<Child: Matcher, Parent: Matcher, R> {
    rule: R,
    _phantom: PhantomData<(Child, Parent)>,
}

impl<Child: Matcher, Parent: Matcher, R> ParentReduceRuleAdapter<Child, Parent, R> {
    pub(crate) fn new(rule: R) -> Self {
        Self {
            rule,
            _phantom: PhantomData,
        }
    }
}

impl<Child: Matcher, Parent: Matcher, R: Debug> Debug
    for ParentReduceRuleAdapter<Child, Parent, R>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ArrayParentReduceRuleAdapter")
            .field("child", &type_name::<Child>())
            .field("parent", &type_name::<Parent>())
            .field("rule", &self.rule)
            .finish()
    }
}

impl<M: Matcher, R: ArrayReduceRule<M>> DynArrayReduceRule for ReduceRuleAdapter<M, R> {
    fn key(&self) -> MatchKey {
        self.rule.matcher().key()
    }

    fn reduce(&self, array: &ArrayRef) -> VortexResult<Option<ArrayRef>> {
        let Some(view) = self.rule.matcher().try_match(array) else {
            return Ok(None);
        };
        self.rule.reduce(view)
    }
}

impl<Child, Parent, R> DynArrayParentReduceRule for ParentReduceRuleAdapter<Child, Parent, R>
where
    Child: Matcher,
    Parent: Matcher,
    R: ArrayParentReduceRule<Child, Parent>,
{
    fn child_key(&self) -> MatchKey {
        self.rule.child().key()
    }

    fn parent_key(&self) -> MatchKey {
        self.rule.parent().key()
    }

    fn reduce_parent(
        &self,
        child: &ArrayRef,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        let Some(child_view) = self.rule.child().try_match(child) else {
            return Ok(None);
        };
        let Some(parent_view) = self.rule.parent().try_match(parent) else {
            return Ok(None);
        };
        self.rule.reduce_parent(child_view, parent_view, child_idx)
    }
}
