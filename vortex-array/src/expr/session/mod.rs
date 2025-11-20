// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod rewrite;

pub use rewrite::RewriteRuleRegistry;
use vortex_session::registry::Registry;
use vortex_session::{Ref, SessionExt};

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
use crate::expr::transform::rules::{
    AnyParent, ParentReduceRule, ReduceRule, RuleContext, TypedRuleContext,
};
use crate::expr::{ExprVTable, VTable};

/// Registry of expression vtables.
pub type ExprRegistry = Registry<ExprVTable>;

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

    /// Register a generic reduce rule that uses Typed context.
    /// Use this for rules that need access to dtype information.
    pub fn register_typed_reduce_rule<V, R>(&mut self, vtable: &'static V, rule: R)
    where
        V: VTable,
        R: 'static,
        R: ReduceRule<V, TypedRuleContext>,
    {
        self.rewrite_rules.register_typed_reduce_rule(vtable, rule);
    }

    /// Register a reduce rule that uses Untyped context.
    /// Use this for rules that don't need access to dtype information.
    pub fn register_reduce_rule<V, R>(&mut self, vtable: &'static V, rule: R)
    where
        V: VTable,
        R: 'static,
        R: ReduceRule<V, RuleContext>,
    {
        self.rewrite_rules.register_reduce_rule(vtable, rule);
    }

    /// Register a parent reduce rule for a specific parent type.
    pub fn register_parent_rule<Child, Parent, R>(
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
        self.rewrite_rules
            .register_parent_rule_specific(child_vtable, parent_vtable, rule);
    }

    /// Register a parent rule that matches ANY parent type (wildcard).
    pub fn register_any_parent_rule<Child, R>(&mut self, child_vtable: &'static Child, rule: R)
    where
        Child: VTable,
        R: 'static,
        R: ParentReduceRule<Child, AnyParent, RuleContext>,
    {
        self.rewrite_rules
            .register_parent_rule_any(child_vtable, rule);
    }

    /// Register a typed parent reduce rule for a specific parent type.
    pub fn register_typed_parent_rule<Child, Parent, R>(
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
        self.rewrite_rules
            .register_typed_parent_rule_specific(child_vtable, parent_vtable, rule);
    }

    /// Register a typed parent rule that matches ANY parent type (wildcard).
    pub fn register_typed_any_parent_rule<Child, R>(
        &mut self,
        child_vtable: &'static Child,
        rule: R,
    ) where
        Child: VTable,
        R: 'static,
        R: ParentReduceRule<Child, AnyParent, TypedRuleContext>,
    {
        self.rewrite_rules
            .register_typed_parent_rule_any(child_vtable, rule);
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
        rewrite_rules.register_typed_reduce_rule(&Select, RemoveSelectRule);
        rewrite_rules.register_typed_reduce_rule(&Merge, RemoveMergeRule);
        rewrite_rules.register_reduce_rule(&GetItem, PackGetItemRule);

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
