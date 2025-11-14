// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use vortex_session::registry::Registry;
use vortex_session::{Ref, SessionExt};
use vortex_utils::aliases::hash_map::HashMap;

use crate::expr::exprs::between::Between;
use crate::expr::exprs::binary::Binary;
use crate::expr::exprs::cast::Cast;
use crate::expr::exprs::get_item::GetItem;
use crate::expr::exprs::is_null::IsNull;
use crate::expr::exprs::like::Like;
use crate::expr::exprs::list_contains::ListContains;
use crate::expr::exprs::literal::Literal;
use crate::expr::exprs::merge::Merge;
use crate::expr::exprs::not::Not;
use crate::expr::exprs::pack::Pack;
use crate::expr::exprs::root::Root;
use crate::expr::exprs::select::Select;
use crate::expr::transform::remove_select::RemoveSelectRule;
use crate::expr::transform::simplify::PackGetItemRule;
use crate::expr::transform::traits::{ChildReduceRule, ParentReduceRule, ReduceRule};
use crate::expr::{ExprId, ExprVTable};

/// Registry of expression vtables.
pub type ExprRegistry = Registry<ExprVTable>;

/// Registry of expression rewrite rules.
///
/// Stores rewrite rules indexed by the expression ID they apply to.
#[derive(Default)]
pub struct RewriteRuleRegistry {
    /// Generic reduce rules (no context needed), indexed by expression ID
    reduce_rules: HashMap<ExprId, Vec<Arc<dyn ReduceRule>>>,
    /// Child reduce rules, indexed by expression ID
    child_rules: HashMap<ExprId, Vec<Arc<dyn ChildReduceRule>>>,
    /// Parent reduce rules, indexed by expression ID
    parent_rules: HashMap<ExprId, Vec<Arc<dyn ParentReduceRule>>>,
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
    pub fn register_reduce_rule(&mut self, rule: Arc<dyn ReduceRule>) {
        let id = rule.id();
        self.reduce_rules.entry(id).or_default().push(rule);
    }

    /// Register a child reduce rule.
    pub fn register_child_rule(&mut self, rule: Arc<dyn ChildReduceRule>) {
        let id = rule.id();
        self.child_rules.entry(id).or_default().push(rule);
    }

    /// Register a parent reduce rule.
    pub fn register_parent_rule(&mut self, rule: Arc<dyn ParentReduceRule>) {
        let id = rule.id();
        self.parent_rules.entry(id).or_default().push(rule);
    }

    /// Get all generic reduce rules for a given expression ID.
    pub fn reduce_rules_for(&self, id: &ExprId) -> Option<&[Arc<dyn ReduceRule>]> {
        self.reduce_rules.get(id).map(|v| v.as_slice())
    }

    /// Get all child reduce rules for a given expression ID.
    pub fn child_rules_for(&self, id: &ExprId) -> Option<&[Arc<dyn ChildReduceRule>]> {
        self.child_rules.get(id).map(|v| v.as_slice())
    }

    /// Get all parent reduce rules for a given expression ID.
    pub fn parent_rules_for(&self, id: &ExprId) -> Option<&[Arc<dyn ParentReduceRule>]> {
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
    pub fn register_reduce_rule(&mut self, rule: impl ReduceRule + 'static) {
        self.rewrite_rules.register_reduce_rule(Arc::new(rule));
    }

    /// Register a child reduce rule in the session.
    pub fn register_child_rule(&mut self, rule: impl ChildReduceRule + 'static) {
        self.rewrite_rules.register_child_rule(Arc::new(rule));
    }

    /// Register a parent reduce rule in the session.
    pub fn register_parent_rule(&mut self, rule: impl ParentReduceRule + 'static) {
        self.rewrite_rules.register_parent_rule(Arc::new(rule));
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
        rewrite_rules.register_reduce_rule(Arc::new(RemoveSelectRule));
        rewrite_rules.register_child_rule(Arc::new(PackGetItemRule));

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
