// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use vortex_error::VortexResult;
use vortex_session::registry::Registry;
use vortex_session::{Ref, SessionExt};
use vortex_utils::aliases::hash_map::HashMap;

use crate::expr::exprs::between::Between;
use crate::expr::exprs::binary::Binary;
use crate::expr::exprs::cast::Cast;
use crate::expr::exprs::get_item::GetItem;
use crate::expr::exprs::get_item::transform::PackGetItemRule;
use crate::expr::exprs::is_null::IsNull;
use crate::expr::exprs::like::Like;
use crate::expr::exprs::list_contains::ListContains;
use crate::expr::exprs::literal::Literal;
use crate::expr::exprs::merge::Merge;
use crate::expr::exprs::merge::transform::RemoveMergeRule;
use crate::expr::exprs::not::Not;
use crate::expr::exprs::pack::Pack;
use crate::expr::exprs::root::Root;
use crate::expr::exprs::select::Select;
use crate::expr::exprs::select::transform::RemoveSelectRule;
use crate::expr::transform::traits::{
    ChildReduceRule, ParentReduceRule, ReduceRule, RewriteContext,
};
use crate::expr::{ExprId, ExprVTable, Expression, VTable};

/// Registry of expression vtables.
pub type ExprRegistry = Registry<ExprVTable>;

/// Type-erased wrapper for ReduceRule that allows dynamic dispatch.
pub(crate) trait DynReduceRule: Send + Sync {
    fn reduce_dyn(
        &self,
        expr: &Expression,
        ctx: &dyn RewriteContext,
    ) -> VortexResult<Option<Expression>>;
}

/// Concrete wrapper that implements DynReduceRule for a specific VTable type.
struct ReduceRuleAdapter<V: VTable, R: ReduceRule<V>> {
    rule: R,
    _phantom: std::marker::PhantomData<V>,
}

impl<V: VTable, R: ReduceRule<V>> ReduceRuleAdapter<V, R> {
    fn new(rule: R) -> Self {
        Self {
            rule,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<V: VTable, R: ReduceRule<V>> DynReduceRule for ReduceRuleAdapter<V, R> {
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

/// Type-erased wrapper for ChildReduceRule that allows dynamic dispatch.
pub(crate) trait DynChildReduceRule: Send + Sync {
    fn reduce_child_dyn(
        &self,
        expr: &Expression,
        child: &Expression,
        child_idx: usize,
        ctx: &dyn RewriteContext,
    ) -> VortexResult<Option<Expression>>;
}

/// Concrete wrapper that implements DynChildReduceRule for a specific VTable type.
struct ChildReduceRuleAdapter<V: VTable, R: ChildReduceRule<V>> {
    rule: R,
    _phantom: std::marker::PhantomData<V>,
}

impl<V: VTable, R: ChildReduceRule<V>> ChildReduceRuleAdapter<V, R> {
    fn new(rule: R) -> Self {
        Self {
            rule,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<V: VTable, R: ChildReduceRule<V>> DynChildReduceRule for ChildReduceRuleAdapter<V, R> {
    fn reduce_child_dyn(
        &self,
        expr: &Expression,
        child: &Expression,
        child_idx: usize,
        ctx: &dyn RewriteContext,
    ) -> VortexResult<Option<Expression>> {
        let Some(view) = expr.as_opt::<V>() else {
            return Ok(None);
        };
        self.rule.reduce_child(&view, child, child_idx, ctx)
    }
}

/// Type-erased wrapper for ParentReduceRule that allows dynamic dispatch.
pub(crate) trait DynParentReduceRule: Send + Sync {
    fn reduce_parent_dyn(
        &self,
        expr: &Expression,
        parent: &Expression,
        ctx: &dyn RewriteContext,
    ) -> VortexResult<Option<Expression>>;
}

/// Concrete wrapper that implements DynParentReduceRule for a specific VTable type.
struct ParentReduceRuleAdapter<V: VTable, R: ParentReduceRule<V>> {
    rule: R,
    _phantom: std::marker::PhantomData<V>,
}

impl<V: VTable, R: ParentReduceRule<V>> ParentReduceRuleAdapter<V, R> {
    fn new(rule: R) -> Self {
        Self {
            rule,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<V: VTable, R: ParentReduceRule<V>> DynParentReduceRule for ParentReduceRuleAdapter<V, R> {
    fn reduce_parent_dyn(
        &self,
        expr: &Expression,
        parent: &Expression,
        ctx: &dyn RewriteContext,
    ) -> VortexResult<Option<Expression>> {
        let Some(view) = expr.as_opt::<V>() else {
            return Ok(None);
        };
        self.rule.reduce_parent(&view, parent, ctx)
    }
}

/// Registry of expression rewrite rules.
///
/// Stores rewrite rules indexed by the expression ID they apply to.
#[derive(Default)]
pub struct RewriteRuleRegistry {
    /// Generic reduce rules (no context needed), indexed by expression ID
    reduce_rules: HashMap<ExprId, Vec<Arc<dyn DynReduceRule>>>,
    /// Child reduce rules, indexed by expression ID
    child_rules: HashMap<ExprId, Vec<Arc<dyn DynChildReduceRule>>>,
    /// Parent reduce rules, indexed by expression ID
    parent_rules: HashMap<ExprId, Vec<Arc<dyn DynParentReduceRule>>>,
}

impl std::fmt::Debug for RewriteRuleRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RewriteRuleRegistry")
            .field("reduce_rules_count", &self.reduce_rules.len())
            .field("child_rules_count", &self.child_rules.len())
            .field("parent_rules_count", &self.parent_rules.len())
            .finish()
    }
}

impl RewriteRuleRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a generic reduce rule.
    pub fn register_reduce_rule<V: VTable, R: ReduceRule<V> + 'static>(
        &mut self,
        vtable: &'static V,
        rule: R,
    ) {
        let id = vtable.id();
        let adapter = ReduceRuleAdapter::new(rule);
        self.reduce_rules
            .entry(id)
            .or_default()
            .push(Arc::new(adapter));
    }

    /// Register a child reduce rule.
    pub fn register_child_rule<V: VTable, R: ChildReduceRule<V> + 'static>(
        &mut self,
        vtable: &'static V,
        rule: R,
    ) {
        let id = vtable.id();
        let adapter = ChildReduceRuleAdapter::new(rule);
        self.child_rules
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

    /// Get all generic reduce rules for a given expression ID.
    pub(crate) fn reduce_rules_for(&self, id: &ExprId) -> Option<&[Arc<dyn DynReduceRule>]> {
        self.reduce_rules.get(id).map(|v| v.as_slice())
    }

    /// Get all child reduce rules for a given expression ID.
    pub(crate) fn child_rules_for(&self, id: &ExprId) -> Option<&[Arc<dyn DynChildReduceRule>]> {
        self.child_rules.get(id).map(|v| v.as_slice())
    }

    /// Get all parent reduce rules for a given expression ID.
    pub(crate) fn parent_rules_for(&self, id: &ExprId) -> Option<&[Arc<dyn DynParentReduceRule>]> {
        self.parent_rules.get(id).map(|v| v.as_slice())
    }
}

/// Session state for expression vtables and rewrite rules.
#[derive(Debug)]
pub struct ExprSession {
    registry: ExprRegistry,
    rewrite_rules: RewriteRuleRegistry,
}

impl ExprSession {
    pub fn registry(&self) -> &ExprRegistry {
        &self.registry
    }

    /// Get the rewrite rule registry.
    pub fn rewrite_rules(&self) -> &RewriteRuleRegistry {
        &self.rewrite_rules
    }

    /// Register an expression vtable in the session, replacing any existing vtable with the same ID.
    pub fn register(&self, expr: ExprVTable) {
        self.registry.register(expr)
    }

    /// Register expression vtables in the session, replacing any existing vtables with the same IDs.
    pub fn register_many(&self, exprs: impl IntoIterator<Item = ExprVTable>) {
        self.registry.register_many(exprs);
    }

    /// Register a generic reduce rule in the session.
    pub fn register_reduce_rule<V: VTable>(
        &mut self,
        vtable: &'static V,
        rule: impl ReduceRule<V> + 'static,
    ) {
        self.rewrite_rules.register_reduce_rule(vtable, rule);
    }

    /// Register a child reduce rule in the session.
    pub fn register_child_rule<V: VTable>(
        &mut self,
        vtable: &'static V,
        rule: impl ChildReduceRule<V> + 'static,
    ) {
        self.rewrite_rules.register_child_rule(vtable, rule);
    }

    /// Register a parent reduce rule in the session.
    pub fn register_parent_rule<V: VTable>(
        &mut self,
        vtable: &'static V,
        rule: impl ParentReduceRule<V> + 'static,
    ) {
        self.rewrite_rules.register_parent_rule(vtable, rule);
    }
}

impl Default for ExprSession {
    fn default() -> Self {
        let expressions = ExprRegistry::default();

        // Register built-in expressions here if needed.
        expressions.register_many([
            ExprVTable::from_static(&Between),
            ExprVTable::from_static(&Binary),
            ExprVTable::from_static(&Cast),
            ExprVTable::from_static(&GetItem),
            ExprVTable::from_static(&IsNull),
            ExprVTable::from_static(&Like),
            ExprVTable::from_static(&ListContains),
            ExprVTable::from_static(&Literal),
            ExprVTable::from_static(&Merge),
            ExprVTable::from_static(&Not),
            ExprVTable::from_static(&Pack),
            ExprVTable::from_static(&Root),
            ExprVTable::from_static(&Select),
        ]);

        // Register built-in rewrite rules
        let mut rewrite_rules = RewriteRuleRegistry::new();
        rewrite_rules.register_reduce_rule(&Select, RemoveSelectRule);
        rewrite_rules.register_reduce_rule(&Merge, RemoveMergeRule);
        rewrite_rules.register_child_rule(&GetItem, PackGetItemRule);

        Self {
            registry: expressions,
            rewrite_rules,
        }
    }
}

/// Extension trait for accessing expression session data.
pub trait ExprSessionExt: SessionExt {
    /// Returns the expression vtable registry.
    fn expressions(&self) -> Ref<'_, ExprSession> {
        self.get::<ExprSession>()
    }
}
impl<S: SessionExt> ExprSessionExt for S {}
