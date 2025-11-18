// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::marker::PhantomData;
use std::sync::Arc;

use vortex_error::VortexResult;
use vortex_utils::aliases::hash_map::HashMap;

use crate::expr::transform::TypedRewriteContext;
use crate::expr::transform::rules::{ParentReduceRule, ReduceRule, RewriteContext};
use crate::expr::{ExprId, Expression, VTable};

/// Type-erased wrapper for ReduceRule that allows dynamic dispatch.
pub(crate) trait DynReduceRule: Send + Sync {
    fn reduce_dyn(
        &self,
        expr: &Expression,
        ctx: &dyn RewriteContext,
    ) -> VortexResult<Option<Expression>>;
}

pub(crate) trait DynTypedReduceRule: Send + Sync {
    fn reduce_dyn_typed(
        &self,
        expr: &Expression,
        ctx: &dyn TypedRewriteContext,
    ) -> VortexResult<Option<Expression>>;
}

/// Type-erased wrapper for ParentReduceRule that allows dynamic dispatch.
pub(crate) trait DynParentReduceRule: Send + Sync {
    fn reduce_parent_dyn(
        &self,
        expr: &Expression,
        parent: &Expression,
        child_idx: usize,
        ctx: &dyn RewriteContext,
    ) -> VortexResult<Option<Expression>>;
}

pub(crate) trait DynTypedParentReduceRule: Send + Sync {
    fn reduce_parent_dyn_typed(
        &self,
        expr: &Expression,
        parent: &Expression,
        child_idx: usize,
        ctx: &dyn TypedRewriteContext,
    ) -> VortexResult<Option<Expression>>;
}

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

// Implement DynReduceRule for any ReduceRule with RewriteContext
impl<V, R> DynReduceRule for RuleAdapter<V, R>
where
    V: VTable,
    for<'a> R: ReduceRule<V, &'a dyn RewriteContext>,
{
    fn reduce_dyn(
        &self,
        expr: &Expression,
        ctx: &dyn RewriteContext,
    ) -> VortexResult<Option<Expression>> {
        let Some(view) = expr.as_opt::<V>() else {
            return Ok(None);
        };
        self.rule.reduce(&view, ctx)
    }
}

// Implement DynTypedReduceRule for any ReduceRule with TypedRewriteContext
impl<V, R> DynTypedReduceRule for RuleAdapter<V, R>
where
    V: VTable,
    for<'a> R: ReduceRule<V, &'a dyn TypedRewriteContext>,
{
    fn reduce_dyn_typed(
        &self,
        expr: &Expression,
        ctx: &dyn TypedRewriteContext,
    ) -> VortexResult<Option<Expression>> {
        let Some(view) = expr.as_opt::<V>() else {
            return Ok(None);
        };
        self.rule.reduce(&view, ctx)
    }
}

// Implement DynParentReduceRule for any ParentReduceRule with RewriteContext
impl<V, R> DynParentReduceRule for RuleAdapter<V, R>
where
    V: VTable,
    for<'a> R: ParentReduceRule<V, &'a dyn RewriteContext>,
{
    fn reduce_parent_dyn(
        &self,
        expr: &Expression,
        parent: &Expression,
        child_idx: usize,
        ctx: &dyn RewriteContext,
    ) -> VortexResult<Option<Expression>> {
        let Some(view) = expr.as_opt::<V>() else {
            return Ok(None);
        };
        self.rule.reduce_parent(&view, parent, child_idx, ctx)
    }
}

// Implement DynTypedParentReduceRule for any ParentReduceRule with TypedRewriteContext
impl<V, R> DynTypedParentReduceRule for RuleAdapter<V, R>
where
    V: VTable,
    for<'a> R: ParentReduceRule<V, &'a dyn TypedRewriteContext>,
{
    fn reduce_parent_dyn_typed(
        &self,
        expr: &Expression,
        parent: &Expression,
        child_idx: usize,
        ctx: &dyn TypedRewriteContext,
    ) -> VortexResult<Option<Expression>> {
        let Some(view) = expr.as_opt::<V>() else {
            return Ok(None);
        };
        self.rule.reduce_parent(&view, parent, child_idx, ctx)
    }
}

/// Registry of expression rewrite rules.
///
/// Stores rewrite rules indexed by the expression ID they apply to.
/// Typed and untyped rules are stored separately for better organization.
#[derive(Default)]
pub struct RewriteRuleRegistry {
    /// Typed reduce rules (require TypedRewriteContext), indexed by expression ID
    typed_reduce_rules: HashMap<ExprId, Vec<Arc<dyn DynTypedReduceRule>>>,
    /// Untyped reduce rules (require only RewriteContext), indexed by expression ID
    reduce_rules: HashMap<ExprId, Vec<Arc<dyn DynReduceRule>>>,
    /// Parent reduce rules, indexed by expression ID
    parent_rules: HashMap<ExprId, Vec<Arc<dyn DynParentReduceRule>>>,
    /// Parent reduce rules, indexed by expression ID
    typed_parent_rules: HashMap<ExprId, Vec<Arc<dyn DynTypedParentReduceRule>>>,
}

impl std::fmt::Debug for RewriteRuleRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RewriteRuleRegistry")
            .field("typed_reduce_rules_count", &self.typed_reduce_rules.len())
            .field("reduce_rules_count", &self.reduce_rules.len())
            .field("parent_rules_count", &self.parent_rules.len())
            .finish()
    }
}

impl RewriteRuleRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a generic reduce rule that uses TypedRewriteContext.
    /// Use this for rules that need access to dtype information.
    pub fn register_typed_reduce_rule<V, R>(&mut self, vtable: &'static V, rule: R)
    where
        V: VTable,
        R: 'static,
        for<'a> R: ReduceRule<V, &'a dyn TypedRewriteContext>,
    {
        let id = vtable.id();
        let adapter = RuleAdapter::new(rule);
        self.typed_reduce_rules
            .entry(id)
            .or_default()
            .push(Arc::new(adapter));
    }

    /// Register a generic reduce rule that only uses RewriteContext (non-typed).
    /// Use this for rules that don't need access to dtype information.
    pub fn register_reduce_rule<V, R>(&mut self, vtable: &'static V, rule: R)
    where
        V: VTable,
        R: 'static,
        for<'a> R: ReduceRule<V, &'a dyn RewriteContext>,
    {
        let id = vtable.id();
        let adapter = RuleAdapter::new(rule);
        self.reduce_rules
            .entry(id)
            .or_default()
            .push(Arc::new(adapter));
    }

    pub fn register_parent_rule<V, R>(&mut self, vtable: &'static V, rule: R)
    where
        V: VTable,
        R: 'static,
        for<'a> R: ParentReduceRule<V, &'a dyn RewriteContext>,
    {
        let id = vtable.id();
        let adapter = RuleAdapter::new(rule);
        self.parent_rules
            .entry(id)
            .or_default()
            .push(Arc::new(adapter));
    }

    /// Register a parent reduce rule.
    pub fn register_typed_parent_rule<V, R>(&mut self, vtable: &'static V, rule: R)
    where
        V: VTable,
        R: 'static,
        for<'a> R: ParentReduceRule<V, &'a dyn TypedRewriteContext>,
    {
        let id = vtable.id();
        let adapter = RuleAdapter::new(rule);
        self.typed_parent_rules
            .entry(id)
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

    /// Get all parent reduce rules for a given expression ID.
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
