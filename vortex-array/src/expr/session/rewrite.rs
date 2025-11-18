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

/// Concrete wrapper that implements DynReduceRule for rules with `&dyn RewriteContext` context.
struct ReduceRuleAdapter<V, R>
where
    V: VTable,
    for<'a> R: ReduceRule<V, &'a dyn RewriteContext>,
{
    rule: R,
    _phantom: PhantomData<V>,
}

impl<V, R> ReduceRuleAdapter<V, R>
where
    V: VTable,
    for<'a> R: ReduceRule<V, &'a dyn RewriteContext>,
{
    fn new(rule: R) -> Self {
        Self {
            rule,
            _phantom: PhantomData,
        }
    }
}

impl<V, R> DynReduceRule for ReduceRuleAdapter<V, R>
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

/// Concrete wrapper that implements DynReduceRule for rules with `&dyn TypedRewriteContext` context.
struct TypedReduceRuleAdapter<V, R>
where
    V: VTable,
    for<'a> R: ReduceRule<V, &'a dyn TypedRewriteContext>,
{
    rule: R,
    _phantom: PhantomData<V>,
}

impl<V, R> TypedReduceRuleAdapter<V, R>
where
    V: VTable,
    for<'a> R: ReduceRule<V, &'a dyn TypedRewriteContext>,
{
    fn new(rule: R) -> Self {
        Self {
            rule,
            _phantom: PhantomData,
        }
    }
}

impl<V, R> DynTypedReduceRule for TypedReduceRuleAdapter<V, R>
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

/// Concrete wrapper that implements DynParentReduceRule for a specific VTable type.
struct ParentReduceRuleAdapter<V: VTable, R: ParentReduceRule<V>> {
    rule: R,
    _phantom: PhantomData<V>,
}

impl<V: VTable, R: ParentReduceRule<V>> ParentReduceRuleAdapter<V, R> {
    fn new(rule: R) -> Self {
        Self {
            rule,
            _phantom: PhantomData,
        }
    }
}

impl<V: VTable, R: ParentReduceRule<V>> DynParentReduceRule for ParentReduceRuleAdapter<V, R> {
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

pub(crate) trait DynTypedParentReduceRule: Send + Sync {
    fn reduce_parent_dyn(
        &self,
        expr: &Expression,
        parent: &Expression,
        child_idx: usize,
        ctx: &dyn TypedRewriteContext,
    ) -> VortexResult<Option<Expression>>;
}

struct TypedParentReduceRuleAdapter<V: VTable, R: ParentReduceRule<V>> {
    rule: R,
    _phantom: PhantomData<V>,
}

impl<V: VTable, R: ParentReduceRule<V>> TypedParentReduceRuleAdapter<V, R> {
    fn new(rule: R) -> Self {
        Self {
            rule,
            _phantom: PhantomData,
        }
    }
}

impl<V: VTable, R: ParentReduceRule<V>> DynTypedParentReduceRule
    for TypedParentReduceRuleAdapter<V, R>
{
    fn reduce_parent_dyn(
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
    typed_parent_rules: HashMap<ExprId, Vec<Arc<dyn DynParentReduceRule>>>,
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
        let adapter = TypedReduceRuleAdapter::new(rule);
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
        let adapter = ReduceRuleAdapter::new(rule);
        self.reduce_rules
            .entry(id)
            .or_default()
            .push(Arc::new(adapter));
    }

    /// Register a parent reduce rule.
    pub fn register_parent_rule<V: VTable, R: ParentReduceRule<V> + 'static>(
        &mut self,
        vtable: &'static V,
        rule: R,
    ) {
        let id = vtable.id();
        let adapter = ParentReduceRuleAdapter::new(rule);
        self.parent_rules
            .entry(id)
            .or_default()
            .push(Arc::new(adapter));
    }

    /// Get all typed reduce rules for a given expression ID.
    pub(crate) fn typed_reduce_rules_for(
        &self,
        id: &ExprId,
    ) -> Option<&[Arc<dyn DynTypedReduceRule>]> {
        self.typed_reduce_rules.get(id).map(|v| v.as_slice())
    }

    /// Get all untyped reduce rules for a given expression ID.
    pub(crate) fn reduce_rules_for(&self, id: &ExprId) -> Option<&[Arc<dyn DynReduceRule>]> {
        self.reduce_rules.get(id).map(|v| v.as_slice())
    }

    /// Get all parent reduce rules for a given expression ID.
    pub(crate) fn parent_rules_for(&self, id: &ExprId) -> Option<&[Arc<dyn DynParentReduceRule>]> {
        self.parent_rules.get(id).map(|v| v.as_slice())
    }

    /// Get all the typed parent reduce rules for a given expression ID.
    pub(crate) fn typed_parent_rules_for(
        &self,
        id: &ExprId,
    ) -> Option<&[Arc<dyn DynParentReduceRule>]> {
        self.typed_parent_rules.get(id).map(|v| v.as_slice())
    }
}
