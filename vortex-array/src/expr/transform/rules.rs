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
pub trait ReduceRule<V: VTable, C: Context>: Send + Sync {
    /// Try to rewrite an expression.
    ///
    /// # Arguments
    /// * `expr` - The expression to potentially rewrite (already downcast to type V)
    /// * `ctx` - Context for the rewrite (dtype, etc.)
    ///
    /// # Returns
    /// * `Some(new_expr)` if the rule applies and produces a rewritten expression
    /// * `None` if the rule does not apply
    fn reduce(&self, expr: &ExpressionView<V>, ctx: C) -> VortexResult<Option<Expression>>;
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
pub trait ParentReduceRule<V: VTable, C: Context>: Send + Sync {
    /// Try to rewrite an expression based on its parent.
    ///
    /// # Arguments
    /// * `expr` - The expression to potentially rewrite (already downcast to type V)
    /// * `parent` - The parent expression (always present - rule not called for root)
    /// * `child_idx` - The index of the child expression within the parent.
    /// * `ctx` - Context for the rewrite (dtype, etc.)
    ///
    /// # Returns
    /// * `Some(new_expr)` if the rule applies and produces a rewritten expression
    /// * `None` if the rule does not apply
    fn reduce_parent(
        &self,
        expr: &ExpressionView<V>,
        parent: &Expression,
        child_idx: usize,
        ctx: C,
    ) -> VortexResult<Option<Expression>>;
}

pub trait Context {}

// Blanket implementation: all references to Context implementors also implement Context
impl<T: Context + ?Sized> Context for &T {}

/// Base context for rewrite rules.
pub trait RewriteContext: Context {}

// Blanket implementation: all references to RewriteContext implementors also implement RewriteContext
impl<T: RewriteContext + ?Sized> RewriteContext for &T {}

/// Context available to rewrite rules during expression optimization.
/// Extends `RewriteContext` and provides access to dtype information.
///
/// Any `TypedRewriteContext` can be used as a `RewriteContext`, but not vice versa.
pub trait TypedRewriteContext: RewriteContext {
    fn dtype(&self) -> &DType;
}

/// Context for untyped rewrite rules.
#[derive(Debug, Default)]
pub struct EmptyRewriteContext;

impl Context for EmptyRewriteContext {}

impl RewriteContext for EmptyRewriteContext {}

/// Simple implementation that supports both RewriteContext and TypedRewriteContext.
#[derive(Debug)]
pub struct RootRewriteContext<'a> {
    pub dtype: &'a DType,
}

impl<'a> Context for RootRewriteContext<'a> {}

impl<'a> RewriteContext for RootRewriteContext<'a> {}

impl<'a> TypedRewriteContext for RootRewriteContext<'a> {
    fn dtype(&self) -> &DType {
        self.dtype
    }
}

/// Type-erased wrappers that allows dynamic dispatch.
pub(crate) trait DynReduceRule: Send + Sync {
    fn reduce(
        &self,
        expr: &Expression,
        ctx: &dyn RewriteContext,
    ) -> VortexResult<Option<Expression>>;
}

pub(crate) trait DynTypedReduceRule: Send + Sync {
    fn reduce(
        &self,
        expr: &Expression,
        ctx: &dyn TypedRewriteContext,
    ) -> VortexResult<Option<Expression>>;
}

pub(crate) trait DynParentReduceRule: Send + Sync {
    fn reduce_parent(
        &self,
        expr: &Expression,
        parent: &Expression,
        child_idx: usize,
        ctx: &dyn RewriteContext,
    ) -> VortexResult<Option<Expression>>;
}

pub(crate) trait DynTypedParentReduceRule: Send + Sync {
    fn reduce_parent(
        &self,
        expr: &Expression,
        parent: &Expression,
        child_idx: usize,
        ctx: &dyn TypedRewriteContext,
    ) -> VortexResult<Option<Expression>>;
}
