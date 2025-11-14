// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Traits for extensible expression rewrite rules.
//!
//! These traits allow external crates to define custom expression optimization rules
//! that can be registered with the expression session.

use vortex_dtype::DType;
use vortex_error::VortexResult;

use crate::expr::{Expression, ExpressionView, VTable};

/// A rewrite rule that transforms expressions without needing context.
///
/// Called during bottom-up traversal after children have been processed.
/// This is useful for self-contained rewrites like: `select(...) -> pack(get_item(...))`
///
/// # Type Parameters
/// * `V` - The VTable type this rule applies to. The rule will only be invoked for expressions
///   with this vtable type, providing compile-time type safety.
pub trait ReduceRule<V: VTable>: Send + Sync {
    /// Try to rewrite an expression.
    ///
    /// # Arguments
    /// * `expr` - The expression to potentially rewrite (already downcast to type V)
    /// * `ctx` - Context for the rewrite (dtype, etc.)
    ///
    /// # Returns
    /// * `Some(new_expr)` if the rule applies and produces a rewritten expression
    /// * `None` if the rule does not apply
    fn reduce(
        &self,
        expr: &ExpressionView<V>,
        ctx: &dyn RewriteContext,
    ) -> VortexResult<Option<Expression>>;
}

/// A rewrite rule that can transform expressions based on child context.
///
/// Called during bottom-up traversal after children have been processed.
/// This is useful for rules like: `pack(...).get_item(field) -> field_expr`
///
/// # Type Parameters
/// * `V` - The VTable type this rule applies to. The rule will only be invoked for expressions
///   with this vtable type, providing compile-time type safety.
pub trait ChildReduceRule<V: VTable>: Send + Sync {
    /// Try to rewrite an expression based on one of its children.
    ///
    /// # Arguments
    /// * `expr` - The expression to potentially rewrite (already downcast to type V)
    /// * `child` - One of the expression's children
    /// * `child_idx` - The index of the child in the expression's children array
    /// * `ctx` - Context for the rewrite (dtype, etc.)
    ///
    /// # Returns
    /// * `Some(new_expr)` if the rule applies and produces a rewritten expression
    /// * `None` if the rule does not apply
    fn reduce_child(
        &self,
        expr: &ExpressionView<V>,
        child: &Expression,
        child_idx: usize,
        ctx: &dyn RewriteContext,
    ) -> VortexResult<Option<Expression>>;
}

/// A rewrite rule that can transform expressions based on parent context.
///
/// Called during top-down traversal from the root.
/// This is useful for rules that need to know about the parent expression.
///
/// Note: This rule is only called for non-root expressions (i.e., when there is a parent).
///
/// # Type Parameters
/// * `V` - The VTable type this rule applies to. The rule will only be invoked for expressions
///   with this vtable type, providing compile-time type safety.
pub trait ParentReduceRule<V: VTable>: Send + Sync {
    /// Try to rewrite an expression based on its parent.
    ///
    /// # Arguments
    /// * `expr` - The expression to potentially rewrite (already downcast to type V)
    /// * `parent` - The parent expression (always present - rule not called for root)
    /// * `ctx` - Context for the rewrite (dtype, etc.)
    ///
    /// # Returns
    /// * `Some(new_expr)` if the rule applies and produces a rewritten expression
    /// * `None` if the rule does not apply
    fn reduce_parent(
        &self,
        expr: &ExpressionView<V>,
        parent: &Expression,
        ctx: &dyn RewriteContext,
    ) -> VortexResult<Option<Expression>>;
}

/// Context available to rewrite rules during expression optimization.
pub trait RewriteContext {
    /// The dtype of the expression scope (root array).
    fn dtype(&self) -> &DType;
}

/// Simple implementation of RewriteContext.
#[derive(Debug)]
pub struct SimpleRewriteContext<'a> {
    pub dtype: &'a DType,
}

impl<'a> RewriteContext for SimpleRewriteContext<'a> {
    fn dtype(&self) -> &DType {
        self.dtype
    }
}
