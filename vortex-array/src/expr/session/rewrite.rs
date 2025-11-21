// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;
use std::marker::PhantomData;
use std::sync::Arc;

use vortex_error::VortexResult;
use vortex_utils::aliases::dash_map::DashMap;

use crate::expr::transform::rules::{
    AnyParent, ParentMatcher, ParentReduceRule, ReduceRule, RuleContext, TypedRuleContext,
};
use crate::expr::transform::{
    DynParentReduceRule, DynReduceRule, DynTypedParentReduceRule, DynTypedReduceRule,
};
use crate::expr::{ExprId, Expression, VTable};

/// Adapter for ReduceRule
struct ReduceRuleAdapter<V: VTable, R> {
    rule: R,
    _phantom: PhantomData<V>,
}

impl<V: VTable, R: Debug> Debug for ReduceRuleAdapter<V, R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ReduceRuleAdapter")
            .field("rule", &self.rule)
            .finish()
    }
}

/// Adapter for ParentReduceRule
struct ReduceParentRuleAdapter<Child: VTable, Parent: ParentMatcher, R> {
    rule: R,
    _phantom: PhantomData<(Child, Parent)>,
}

impl<Child: VTable, Parent: ParentMatcher, R: Debug> Debug
    for ReduceParentRuleAdapter<Child, Parent, R>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ReduceParentRuleAdapter")
            .field("rule", &self.rule)
            .finish()
    }
}

impl<V, R> DynReduceRule for ReduceRuleAdapter<V, R>
where
    V: VTable,
    R: Debug + Send + Sync + 'static + ReduceRule<V, RuleContext>,
{
    fn reduce(&self, expr: &Expression, ctx: &RuleContext) -> VortexResult<Option<Expression>> {
        let Some(view) = expr.as_opt::<V>() else {
            return Ok(None);
        };
        self.rule.reduce(&view, ctx)
    }
}

impl<V, R> DynTypedReduceRule for ReduceRuleAdapter<V, R>
where
    V: VTable,
    R: Debug + Send + Sync + 'static + ReduceRule<V, TypedRuleContext>,
{
    fn reduce(
        &self,
        expr: &Expression,
        ctx: &TypedRuleContext,
    ) -> VortexResult<Option<Expression>> {
        let Some(view) = expr.as_opt::<V>() else {
            return Ok(None);
        };
        self.rule.reduce(&view, ctx)
    }
}

impl<Child, Parent, R> DynParentReduceRule for ReduceParentRuleAdapter<Child, Parent, R>
where
    Child: VTable,
    Parent: ParentMatcher,
    R: Debug + Send + Sync + 'static + ParentReduceRule<Child, Parent, RuleContext>,
{
    fn reduce_parent(
        &self,
        expr: &Expression,
        parent: &Expression,
        child_idx: usize,
        ctx: &RuleContext,
    ) -> VortexResult<Option<Expression>> {
        let Some(view) = expr.as_opt::<Child>() else {
            return Ok(None);
        };
        let Some(parent_view) = Parent::try_match(parent) else {
            return Ok(None);
        };
        self.rule.reduce_parent(&view, parent_view, child_idx, ctx)
    }
}

impl<Child, Parent, R> DynTypedParentReduceRule for ReduceParentRuleAdapter<Child, Parent, R>
where
    Child: VTable,
    Parent: ParentMatcher,
    R: Debug + Send + Sync + 'static + ParentReduceRule<Child, Parent, TypedRuleContext>,
{
    fn reduce_parent(
        &self,
        expr: &Expression,
        parent: &Expression,
        child_idx: usize,
        ctx: &TypedRuleContext,
    ) -> VortexResult<Option<Expression>> {
        let Some(view) = expr.as_opt::<Child>() else {
            return Ok(None);
        };
        let Some(parent_view) = Parent::try_match(parent) else {
            return Ok(None);
        };
        self.rule.reduce_parent(&view, parent_view, child_idx, ctx)
    }
}

type RuleRegistry<Rule> = DashMap<ExprId, Vec<Arc<Rule>>>;
type ParentRuleRegistry<Rule> = DashMap<(ExprId, ExprId), Vec<Arc<Rule>>>;

/// Inner struct that holds all the rule registries.
/// Wrapped in a single Arc by RewriteRuleRegistry for efficient cloning.
#[derive(Default, Debug)]
struct RewriteRuleRegistryInner {
    /// Typed reduce rules (require TypedRewriteContext), indexed by expression ID
    typed_reduce_rules: RuleRegistry<dyn DynTypedReduceRule>,
    /// Untyped reduce rules (require only RewriteContext), indexed by expression ID
    reduce_rules: RuleRegistry<dyn DynReduceRule>,
    /// Parent reduce rules for specific parent types, indexed by (child_id, parent_id)
    typed_parent_rules: ParentRuleRegistry<dyn DynTypedParentReduceRule>,
    /// Parent reduce rules for specific parent types, indexed by (child_id, parent_id)
    parent_rules: ParentRuleRegistry<dyn DynParentReduceRule>,
    /// Wildcard parent rules (match any parent), indexed by child_id only
    typed_any_parent_rules: RuleRegistry<dyn DynTypedParentReduceRule>,
    /// Wildcard parent rules (match any parent), indexed by child_id only
    any_parent_rules: RuleRegistry<dyn DynParentReduceRule>,
}

/// Registry of expression rewrite rules.
///
/// Stores rewrite rules indexed by the expression ID they apply to.
/// Typed and untyped rules are stored separately for better organization.
#[derive(Clone, Debug)]
pub struct RewriteRuleRegistry {
    inner: Arc<RewriteRuleRegistryInner>,
}

impl Default for RewriteRuleRegistry {
    fn default() -> Self {
        Self {
            inner: Arc::new(RewriteRuleRegistryInner::default()),
        }
    }
}

impl RewriteRuleRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a generic reduce rule that uses Typed context.
    /// Use this for rules that need access to dtype information.
    pub fn register_typed_reduce_rule<V, R>(&mut self, vtable: &'static V, rule: R)
    where
        V: VTable,
        R: 'static + ReduceRule<V, TypedRuleContext>,
    {
        let adapter = ReduceRuleAdapter {
            rule,
            _phantom: PhantomData,
        };
        self.inner
            .typed_reduce_rules
            .entry(vtable.id())
            .or_default()
            .push(Arc::new(adapter));
    }

    /// Register a generic reduce rule that only uses Untyped context.
    /// Use this for rules that don't need access to dtype information.
    pub fn register_reduce_rule<V, R>(&mut self, vtable: &'static V, rule: R)
    where
        V: VTable,
        R: 'static + ReduceRule<V, RuleContext>,
    {
        let adapter = ReduceRuleAdapter {
            rule,
            _phantom: PhantomData,
        };
        self.inner
            .reduce_rules
            .entry(vtable.id())
            .or_default()
            .push(Arc::new(adapter));
    }

    /// Register a parent rule for a specific parent type.
    pub fn register_parent_rule_specific<Child, Parent, R>(
        &mut self,
        child_vtable: &'static Child,
        parent_vtable: &'static Parent,
        rule: R,
    ) where
        Child: VTable,
        Parent: VTable,
        R: 'static + ParentReduceRule<Child, Parent, RuleContext>,
    {
        let adapter = ReduceParentRuleAdapter {
            rule,
            _phantom: PhantomData,
        };
        self.inner
            .parent_rules
            .entry((child_vtable.id(), parent_vtable.id()))
            .or_default()
            .push(Arc::new(adapter));
    }

    /// Register a parent rule that matches ANY parent type (wildcard).
    pub fn register_parent_rule_any<Child, R>(&mut self, child_vtable: &'static Child, rule: R)
    where
        Child: VTable,
        R: 'static + ParentReduceRule<Child, AnyParent, RuleContext>,
    {
        let adapter = ReduceParentRuleAdapter {
            rule,
            _phantom: PhantomData,
        };
        self.inner
            .any_parent_rules
            .entry(child_vtable.id())
            .or_default()
            .push(Arc::new(adapter));
    }

    /// Register a typed parent rule for a specific parent type.
    pub fn register_typed_parent_rule_specific<Child, Parent, R>(
        &mut self,
        child_vtable: &'static Child,
        parent_vtable: &'static Parent,
        rule: R,
    ) where
        Child: VTable,
        Parent: VTable,
        R: 'static + ParentReduceRule<Child, Parent, TypedRuleContext>,
    {
        let adapter = ReduceParentRuleAdapter {
            rule,
            _phantom: PhantomData,
        };
        self.inner
            .typed_parent_rules
            .entry((child_vtable.id(), parent_vtable.id()))
            .or_default()
            .push(Arc::new(adapter));
    }

    /// Register a typed parent rule that matches ANY parent type (wildcard).
    pub fn register_typed_parent_rule_any<Child, R>(
        &mut self,
        child_vtable: &'static Child,
        rule: R,
    ) where
        Child: VTable,
        R: 'static + ParentReduceRule<Child, AnyParent, TypedRuleContext>,
    {
        let adapter = ReduceParentRuleAdapter {
            rule,
            _phantom: PhantomData,
        };
        self.inner
            .typed_any_parent_rules
            .entry(child_vtable.id())
            .or_default()
            .push(Arc::new(adapter));
    }

    /// Execute a callback with all typed reduce rules for a given expression ID.
    pub(crate) fn with_typed_reduce_rules<F, R>(&self, id: &ExprId, f: F) -> R
    where
        F: FnOnce(&mut dyn Iterator<Item = &dyn DynTypedReduceRule>) -> R,
    {
        f(&mut self
            .inner
            .typed_reduce_rules
            .get(id)
            .iter()
            .flat_map(|v| v.value())
            .map(|arc| arc.as_ref()))
    }

    /// Execute a callback with all untyped reduce rules for a given expression ID.
    pub(crate) fn with_reduce_rules<F, R>(&self, id: &ExprId, f: F) -> R
    where
        F: FnOnce(&mut dyn Iterator<Item = &dyn DynReduceRule>) -> R,
    {
        f(&mut self
            .inner
            .reduce_rules
            .get(id)
            .iter()
            .flat_map(|v| v.value())
            .map(|arc| arc.as_ref()))
    }

    /// Execute a callback with all untyped parent reduce rules for a given child and parent expression ID.
    ///
    /// Returns rules from both specific parent rules (if parent_id provided) and "any parent" wildcard rules.
    pub(crate) fn with_parent_rules<F, R>(
        &self,
        child_id: &ExprId,
        parent_id: Option<&ExprId>,
        f: F,
    ) -> R
    where
        F: FnOnce(&mut dyn Iterator<Item = &dyn DynParentReduceRule>) -> R,
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

    /// Execute a callback with all typed parent reduce rules for a given child and parent expression ID.
    ///
    /// Returns rules from both specific parent rules (if parent_id provided) and "any parent" wildcard rules.
    pub(crate) fn with_typed_parent_rules<F, R>(
        &self,
        child_id: &ExprId,
        parent_id: Option<&ExprId>,
        f: F,
    ) -> R
    where
        F: FnOnce(&mut dyn Iterator<Item = &dyn DynTypedParentReduceRule>) -> R,
    {
        let specific_entry = parent_id.and_then(|pid| {
            self.inner
                .typed_parent_rules
                .get(&(child_id.clone(), pid.clone()))
        });
        let wildcard_entry = self.inner.typed_any_parent_rules.get(child_id);

        f(&mut specific_entry
            .iter()
            .flat_map(|v| v.value())
            .chain(wildcard_entry.iter().flat_map(|v| v.value()))
            .map(|arc| arc.as_ref()))
    }
}
