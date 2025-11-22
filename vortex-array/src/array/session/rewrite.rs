// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;
use std::marker::PhantomData;
use std::sync::Arc;

use vortex_error::VortexResult;
use vortex_utils::aliases::dash_map::DashMap;

use crate::array::ArrayRef;
use crate::array::transform::context::ArrayRuleContext;
use crate::array::transform::rules::{
    AnyArrayParent, ArrayParentMatcher, ArrayParentReduceRule, ArrayReduceRule,
};
use crate::vtable::{ArrayId, VTable};

/// Dynamic trait for array reduce rules
pub trait DynArrayReduceRule: Debug + Send + Sync {
    fn reduce(&self, array: &ArrayRef, ctx: &ArrayRuleContext) -> VortexResult<Option<ArrayRef>>;
}

/// Dynamic trait for array parent reduce rules
pub trait DynArrayParentReduceRule: Debug + Send + Sync {
    fn reduce_parent(
        &self,
        array: &ArrayRef,
        parent: &ArrayRef,
        child_idx: usize,
        ctx: &ArrayRuleContext,
    ) -> VortexResult<Option<ArrayRef>>;
}

/// Adapter for ArrayReduceRule
struct ArrayReduceRuleAdapter<V: VTable, R> {
    rule: R,
    _phantom: PhantomData<V>,
}

impl<V: VTable, R: Debug> Debug for ArrayReduceRuleAdapter<V, R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ArrayReduceRuleAdapter")
            .field("rule", &self.rule)
            .finish()
    }
}

/// Adapter for ArrayParentReduceRule
struct ArrayParentReduceRuleAdapter<Child: VTable, Parent: ArrayParentMatcher, R> {
    rule: R,
    _phantom: PhantomData<(Child, Parent)>,
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
    fn reduce(&self, array: &ArrayRef, ctx: &ArrayRuleContext) -> VortexResult<Option<ArrayRef>> {
        let Some(view) = array.as_opt::<V>() else {
            return Ok(None);
        };
        self.rule.reduce(view, ctx)
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
        ctx: &ArrayRuleContext,
    ) -> VortexResult<Option<ArrayRef>> {
        let Some(view) = array.as_opt::<Child>() else {
            return Ok(None);
        };
        let Some(parent_view) = Parent::try_match(parent) else {
            return Ok(None);
        };
        self.rule.reduce_parent(view, parent_view, child_idx, ctx)
    }
}

/// Inner struct that holds all the rule registries.
/// Wrapped in a single Arc by ArrayRewriteRuleRegistry for efficient cloning.
#[derive(Default, Debug)]
struct ArrayRewriteRuleRegistryInner {
    /// Reduce rules indexed by encoding ID
    reduce_rules: DashMap<ArrayId, Vec<Arc<dyn DynArrayReduceRule>>>,
    /// Parent reduce rules for specific parent types, indexed by (child_id, parent_id)
    parent_rules: DashMap<(ArrayId, ArrayId), Vec<Arc<dyn DynArrayParentReduceRule>>>,
    /// Wildcard parent rules (match any parent), indexed by child_id only
    any_parent_rules: DashMap<ArrayId, Vec<Arc<dyn DynArrayParentReduceRule>>>,
}

/// Registry of array rewrite rules.
///
/// Stores rewrite rules indexed by the encoding ID they apply to.
#[derive(Clone, Debug)]
pub struct ArrayRewriteRuleRegistry {
    inner: Arc<ArrayRewriteRuleRegistryInner>,
}

impl Default for ArrayRewriteRuleRegistry {
    fn default() -> Self {
        Self {
            inner: Arc::new(ArrayRewriteRuleRegistryInner::default()),
        }
    }
}

impl ArrayRewriteRuleRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a reduce rule for a specific array encoding.
    pub fn register_reduce_rule<V, R>(&self, vtable: &V, rule: R)
    where
        V: VTable,
        R: ArrayReduceRule<V> + 'static,
    {
        let adapter = ArrayReduceRuleAdapter {
            rule,
            _phantom: PhantomData,
        };
        let encoding_id = V::id(vtable);
        self.inner
            .reduce_rules
            .entry(encoding_id)
            .or_default()
            .push(Arc::new(adapter));
    }

    /// Register a parent rule for a specific parent type.
    pub fn register_parent_rule<Child, Parent, R>(
        &self,
        child_encoding: &Child,
        parent_encoding: &Parent,
        rule: R,
    ) where
        Child: VTable,
        Parent: VTable,
        R: ArrayParentReduceRule<Child, Parent> + 'static,
    {
        let adapter = ArrayParentReduceRuleAdapter {
            rule,
            _phantom: PhantomData,
        };
        let child_id = Child::id(child_encoding);
        let parent_id = Parent::id(parent_encoding);
        self.inner
            .parent_rules
            .entry((child_id, parent_id))
            .or_default()
            .push(Arc::new(adapter));
    }

    /// Register a parent rule that matches ANY parent type (wildcard).
    pub fn register_any_parent_rule<Child, R>(&self, child_encoding: &Child, rule: R)
    where
        Child: VTable,
        R: ArrayParentReduceRule<Child, AnyArrayParent> + 'static,
    {
        let adapter = ArrayParentReduceRuleAdapter {
            rule,
            _phantom: PhantomData,
        };
        let child_id = Child::id(child_encoding);
        self.inner
            .any_parent_rules
            .entry(child_id)
            .or_default()
            .push(Arc::new(adapter));
    }

    /// Execute a callback with all reduce rules for a given encoding ID.
    pub(crate) fn with_reduce_rules<F, R>(&self, id: &ArrayId, f: F) -> R
    where
        F: FnOnce(&mut dyn Iterator<Item = &dyn DynArrayReduceRule>) -> R,
    {
        f(&mut self
            .inner
            .reduce_rules
            .get(id)
            .iter()
            .flat_map(|v| v.value())
            .map(|arc| arc.as_ref()))
    }

    /// Execute a callback with all parent reduce rules for a given child and parent encoding ID.
    ///
    /// Returns rules from both specific parent rules (if parent_id provided) and "any parent" wildcard rules.
    pub(crate) fn with_parent_rules<F, R>(
        &self,
        child_id: &ArrayId,
        parent_id: Option<&ArrayId>,
        f: F,
    ) -> R
    where
        F: FnOnce(&mut dyn Iterator<Item = &dyn DynArrayParentReduceRule>) -> R,
    {
        let specific_entry = parent_id.and_then(|pid| {
            self.inner
                .parent_rules
                .get(&(child_id.clone(), pid.clone()))
        });
        let wildcard_entry = self.inner.any_parent_rules.get(child_id);

        f(&mut specific_entry
            .iter()
            .flat_map(|v| v.value())
            .chain(wildcard_entry.iter().flat_map(|v| v.value()))
            .map(|arc| arc.as_ref()))
    }
}
