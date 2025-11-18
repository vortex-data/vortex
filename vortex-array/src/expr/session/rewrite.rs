// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;
use std::marker::PhantomData;
use std::sync::Arc;

use vortex_error::VortexResult;
use vortex_utils::aliases::hash_map::HashMap;

use crate::expr::transform::rules::{ParentReduceRule, ReduceRule, RuleContext, TypedRuleContext};
use crate::expr::transform::{
    DynParentReduceRule, DynReduceRule, DynTypedParentReduceRule, DynTypedReduceRule,
};
use crate::expr::{ExprId, Expression, VTable};

/// Universal adapter for both ReduceRule and ParentReduceRule with any context type.
struct RuleAdapter<V: VTable, R> {
    rule: R,
    _phantom: PhantomData<V>,
}

impl<V: VTable, R> RuleAdapter<V, R> {
    fn new(rule: R) -> Self {
        Self {
            rule,
            _phantom: PhantomData,
        }
    }
}

impl<V, R> DynReduceRule for RuleAdapter<V, R>
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

impl<V, R> DynTypedReduceRule for RuleAdapter<V, R>
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

impl<V, R> DynParentReduceRule for RuleAdapter<V, R>
where
    V: VTable,
    R: ParentReduceRule<V, RuleContext>,
{
    fn reduce_parent(
        &self,
        expr: &Expression,
        parent: &Expression,
        child_idx: usize,
        ctx: &RuleContext,
    ) -> VortexResult<Option<Expression>> {
        let Some(view) = expr.as_opt::<V>() else {
            return Ok(None);
        };
        self.rule.reduce_parent(&view, parent, child_idx, ctx)
    }
}

impl<V, R> DynTypedParentReduceRule for RuleAdapter<V, R>
where
    V: VTable,
    R: ParentReduceRule<V, TypedRuleContext>,
{
    fn reduce_parent(
        &self,
        expr: &Expression,
        parent: &Expression,
        child_idx: usize,
        ctx: &TypedRuleContext,
    ) -> VortexResult<Option<Expression>> {
        let Some(view) = expr.as_opt::<V>() else {
            return Ok(None);
        };
        self.rule.reduce_parent(&view, parent, child_idx, ctx)
    }
}

type RuleRegistry<Rule> = HashMap<ExprId, Vec<Arc<Rule>>>;

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
    /// Parent reduce rules, indexed by expression ID
    typed_parent_rules: RuleRegistry<dyn DynTypedParentReduceRule>,
    /// Parent reduce rules, indexed by expression ID
    parent_rules: RuleRegistry<dyn DynParentReduceRule>,
}

// TODO(joe): follow up with rule debug info.
impl Debug for RewriteRuleRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RewriteRuleRegistry")
            .field("typed_reduce_rules_count", &self.typed_reduce_rules.len())
            .field("reduce_rules_count", &self.reduce_rules.len())
            .field("typed_parent_rules", &self.typed_parent_rules.len())
            .field("parent_rules_count", &self.parent_rules.len())
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
        let adapter = RuleAdapter::new(rule);
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
        let adapter = RuleAdapter::new(rule);
        self.reduce_rules
            .entry(vtable.id())
            .or_default()
            .push(Arc::new(adapter));
    }

    pub fn register_parent_rule<V, R>(&mut self, vtable: &'static V, rule: R)
    where
        V: VTable,
        R: 'static,
        R: ParentReduceRule<V, RuleContext>,
    {
        let adapter = RuleAdapter::new(rule);
        self.parent_rules
            .entry(vtable.id())
            .or_default()
            .push(Arc::new(adapter));
    }

    /// Register a parent reduce rule.
    pub fn register_typed_parent_rule<V, R>(&mut self, vtable: &'static V, rule: R)
    where
        V: VTable,
        R: 'static,
        R: ParentReduceRule<V, TypedRuleContext>,
    {
        let adapter = RuleAdapter::new(rule);
        self.typed_parent_rules
            .entry(vtable.id())
            .or_default()
            .push(Arc::new(adapter));
    }

    /// Get all typed reduce rules for a given expression ID.
    pub(crate) fn typed_reduce_rules_for(&self, id: &ExprId) -> &[Arc<dyn DynTypedReduceRule>] {
        self.typed_reduce_rules
            .get(id)
            .map(|v| v.as_slice())
            .unwrap_or_default()
    }

    /// Get all untyped reduce rules for a given expression ID.
    pub(crate) fn reduce_rules_for(&self, id: &ExprId) -> &[Arc<dyn DynReduceRule>] {
        self.reduce_rules
            .get(id)
            .map(|v| v.as_slice())
            .unwrap_or_default()
    }

    /// Get all untyped parent reduce rules for a given expression ID.
    pub(crate) fn parent_rules_for(&self, id: &ExprId) -> &[Arc<dyn DynParentReduceRule>] {
        self.parent_rules
            .get(id)
            .map(|v| v.as_slice())
            .unwrap_or_default()
    }

    /// Get all the typed parent reduce rules for a given expression ID.
    pub(crate) fn typed_parent_rules_for(
        &self,
        id: &ExprId,
    ) -> &[Arc<dyn DynTypedParentReduceRule>] {
        self.typed_parent_rules
            .get(id)
            .map(|v| v.as_slice())
            .unwrap_or_default()
    }
}
