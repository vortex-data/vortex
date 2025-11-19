// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;
use std::marker::PhantomData;
use std::sync::Arc;

use vortex_error::VortexResult;
use vortex_utils::aliases::hash_map::HashMap;

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

/// Adapter for ParentReduceRule
struct ReduceParentRuleAdapter<Child: VTable, Parent: ParentMatcher, R> {
    rule: R,
    _phantom: PhantomData<(Child, Parent)>,
}

impl<V, R> DynReduceRule for ReduceRuleAdapter<V, R>
where
    V: VTable,
    R: ReduceRule<V, RuleContext>,
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
    R: ReduceRule<V, TypedRuleContext>,
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
    R: ParentReduceRule<Child, Parent, RuleContext>,
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
    R: ParentReduceRule<Child, Parent, TypedRuleContext>,
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

type RuleRegistry<Rule> = HashMap<ExprId, Vec<Arc<Rule>>>;
type ParentRuleRegistry<Rule> = HashMap<(ExprId, ExprId), Vec<Arc<Rule>>>;

/// Registry of expression rewrite rules.
///
/// Stores rewrite rules indexed by the expression ID they apply to.
/// Typed and untyped rules are stored separately for better organization.
#[derive(Default)]
pub struct RewriteRuleRegistry {
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

// TODO(joe): follow up with rule debug info.
impl Debug for RewriteRuleRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RewriteRuleRegistry")
            .field("typed_reduce_rules_count", &self.typed_reduce_rules.len())
            .field("reduce_rules_count", &self.reduce_rules.len())
            .field("typed_parent_rules", &self.typed_parent_rules.len())
            .field("parent_rules_count", &self.parent_rules.len())
            .field(
                "typed_any_parent_rules_count",
                &self.typed_any_parent_rules.len(),
            )
            .field("any_parent_rules_count", &self.any_parent_rules.len())
            .finish()
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
        R: 'static,
        R: ReduceRule<V, TypedRuleContext>,
    {
        let adapter = ReduceRuleAdapter {
            rule,
            _phantom: PhantomData,
        };
        self.typed_reduce_rules
            .entry(vtable.id())
            .or_default()
            .push(Arc::new(adapter));
    }

    /// Register a generic reduce rule that only uses Untyped context.
    /// Use this for rules that don't need access to dtype information.
    pub fn register_reduce_rule<V, R>(&mut self, vtable: &'static V, rule: R)
    where
        V: VTable,
        R: 'static,
        R: ReduceRule<V, RuleContext>,
    {
        let adapter = ReduceRuleAdapter {
            rule,
            _phantom: PhantomData,
        };
        self.reduce_rules
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
        R: 'static,
        R: ParentReduceRule<Child, Parent, RuleContext>,
    {
        let adapter = ReduceParentRuleAdapter {
            rule,
            _phantom: PhantomData,
        };
        self.parent_rules
            .entry((child_vtable.id(), parent_vtable.id()))
            .or_default()
            .push(Arc::new(adapter));
    }

    /// Register a parent rule that matches ANY parent type (wildcard).
    pub fn register_parent_rule_any<Child, R>(&mut self, child_vtable: &'static Child, rule: R)
    where
        Child: VTable,
        R: 'static,
        R: ParentReduceRule<Child, AnyParent, RuleContext>,
    {
        let adapter = ReduceParentRuleAdapter {
            rule,
            _phantom: PhantomData,
        };
        self.any_parent_rules
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
        R: 'static,
        R: ParentReduceRule<Child, Parent, TypedRuleContext>,
    {
        let adapter = ReduceParentRuleAdapter {
            rule,
            _phantom: PhantomData,
        };
        self.typed_parent_rules
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
        R: 'static,
        R: ParentReduceRule<Child, AnyParent, TypedRuleContext>,
    {
        let adapter = ReduceParentRuleAdapter {
            rule,
            _phantom: PhantomData,
        };
        self.typed_any_parent_rules
            .entry(child_vtable.id())
            .or_default()
            .push(Arc::new(adapter));
    }

    /// Get all typed reduce rules for a given expression ID.
    pub(crate) fn typed_reduce_rules_for(
        &self,
        id: &ExprId,
    ) -> impl Iterator<Item = &Arc<dyn DynTypedReduceRule>> {
        self.typed_reduce_rules
            .get(id)
            .into_iter()
            .flat_map(|v| v.iter())
    }

    /// Get all untyped reduce rules for a given expression ID.
    pub(crate) fn reduce_rules_for(
        &self,
        id: &ExprId,
    ) -> impl Iterator<Item = &Arc<dyn DynReduceRule>> {
        self.reduce_rules.get(id).into_iter().flat_map(|v| v.iter())
    }

    /// Get all untyped parent reduce rules for a given child and parent expression ID pair.
    ///
    /// Returns both specific parent rules and wildcard "any parent" rules.
    pub(crate) fn parent_rules_for(
        &self,
        child_id: &ExprId,
        parent_id: &ExprId,
    ) -> impl Iterator<Item = &Arc<dyn DynParentReduceRule>> {
        let specific = self
            .parent_rules
            .get(&(child_id.clone(), parent_id.clone()))
            .into_iter()
            .flat_map(|v| v.iter());

        let wildcard = self
            .any_parent_rules
            .get(child_id)
            .into_iter()
            .flat_map(|v| v.iter());

        specific.chain(wildcard)
    }

    /// Get all the typed parent reduce rules for a given child and parent expression ID pair.
    ///
    /// Returns both specific parent rules and wildcard "any parent" rules.
    pub(crate) fn typed_parent_rules_for(
        &self,
        child_id: &ExprId,
        parent_id: &ExprId,
    ) -> impl Iterator<Item = &Arc<dyn DynTypedParentReduceRule>> {
        let specific = self
            .typed_parent_rules
            .get(&(child_id.clone(), parent_id.clone()))
            .into_iter()
            .flat_map(|v| v.iter());

        let wildcard = self
            .typed_any_parent_rules
            .get(child_id)
            .into_iter()
            .flat_map(|v| v.iter());

        specific.chain(wildcard)
    }
}
