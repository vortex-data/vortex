// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Traits for extensible expression rewrite rules.
//!
//! These traits allow external crates to define custom expression optimization rules
//! that can be registered with the expression session.

use vortex_dtype::DType;
use vortex_error::VortexResult;

use crate::expr::{Expression, ExpressionView, VTable};

/// Trait that abstracts over parent matching - allows both specific and wildcard parent types.
pub trait ParentMatcher: Send + Sync + 'static {
    /// The view type returned when matching succeeds.
    type View<'a>;

    /// Try to match/downcast the parent expression.
    /// Returns Some if the parent matches this matcher's criteria, None otherwise.
    fn try_match(parent: &Expression) -> Option<Self::View<'_>>;
}

/// Marker type representing "any parent" - matches all parent expressions.
pub struct AnyParent;

impl ParentMatcher for AnyParent {
    type View<'a> = &'a Expression;

    fn try_match(parent: &Expression) -> Option<Self::View<'_>> {
        Some(parent)
    }
}

/// All VTable types can be used as specific parent matchers.
impl<V: VTable> ParentMatcher for V {
    type View<'a> = ExpressionView<'a, V>;

    fn try_match(parent: &Expression) -> Option<Self::View<'_>> {
        parent.as_opt::<V>()
    }
}

/// A rewrite rule that transforms expressions without needing context.
///
/// Called during bottom-up traversal after children have been processed.
/// This is useful for self-contained rewrites like: `select(...) -> pack(get_item(...))`
///
/// # Type Parameters
/// * `V` - The VTable type this rule applies to. The rule will only be invoked for expressions
///   with this vtable type, providing compile-time type safety.
pub trait ReduceRule<V: VTable, C: RewriteContext>: Send + Sync {
    /// Try to rewrite an expression.
    ///
    /// # Arguments
    /// * `expr` - The expression to potentially rewrite (already downcast to type V)
    /// * `ctx` - Context for the rewrite (dtype, etc.)
    ///
    /// # Returns
    /// * `Some(new_expr)` if the rule applies and produces a rewritten expression
    /// * `None` if the rule does not apply
    fn reduce(&self, expr: &ExpressionView<V>, ctx: &C) -> VortexResult<Option<Expression>>;
}

/// A rewrite rule that can transform expressions based on parent context.
///
/// Called during top-down traversal from the root.
/// This is useful for rules that need to know about the parent expression.
///
/// Note: This rule is only called for non-root expressions (i.e., when there is a parent).
///
/// # Type Parameters
/// * `Child` - The VTable type this rule applies to (the child expression type). The rule will only
///   be invoked for expressions with this vtable type, providing compile-time type safety.
/// * `Parent` - The parent matcher. Can be a specific VTable type (e.g., `Binary`) for typed parent
///   access, or `AnyParent` to match any parent type with untyped access.
/// * `C` - The rewrite context type (RuleContext or TypedRuleContext)
pub trait ParentReduceRule<Child: VTable, Parent: ParentMatcher, C: RewriteContext>:
    Send + Sync
{
    /// Try to rewrite an expression based on its parent.
    ///
    /// # Arguments
    /// * `expr` - The expression to potentially rewrite (already downcast to type Child)
    /// * `parent` - The parent view (type depends on Parent matcher - typed for specific VTables,
    ///   untyped `&Expression` for `AnyParent`)
    /// * `child_idx` - The index of the child expression within the parent.
    /// * `ctx` - Context for the rewrite (dtype, etc.)
    ///
    /// # Returns
    /// * `Some(new_expr)` if the rule applies and produces a rewritten expression
    /// * `None` if the rule does not apply
    fn reduce_parent(
        &self,
        expr: &ExpressionView<Child>,
        parent: Parent::View<'_>,
        child_idx: usize,
        ctx: &C,
    ) -> VortexResult<Option<Expression>>;
}

/// Sealed trait for rewrite rule contexts.
///
/// This trait cannot be implemented outside this module. Only `Typed` and `Untyped`
/// implement this trait.
pub trait RewriteContext: private::Sealed {}

mod private {
    /// Sealing trait to prevent external implementations of `RewriteContext`.
    pub trait Sealed {}
}

/// Typed context for rewrite rules that need access to dtype information.
#[derive(Debug, Clone)]
pub struct TypedRuleContext {
    /// This is the root dtype of the expression
    dtype: DType,
}

impl TypedRuleContext {
    pub fn new(dtype: DType) -> Self {
        Self { dtype }
    }

    pub fn dtype(&self) -> &DType {
        &self.dtype
    }
}

impl private::Sealed for TypedRuleContext {}
impl RewriteContext for TypedRuleContext {}

/// A context for rewrite rules that don't need dtype information.
#[derive(Debug, Clone, Copy, Default)]
pub struct RuleContext;

impl private::Sealed for RuleContext {}
impl RewriteContext for RuleContext {}

/// Type-erased wrappers that allows dynamic dispatch.
pub(crate) trait DynReduceRule: Send + Sync {
    fn reduce(&self, expr: &Expression, ctx: &RuleContext) -> VortexResult<Option<Expression>>;
}

pub(crate) trait DynTypedReduceRule: Send + Sync {
    fn reduce(&self, expr: &Expression, ctx: &TypedRuleContext)
    -> VortexResult<Option<Expression>>;
}

pub(crate) trait DynParentReduceRule: Send + Sync {
    fn reduce_parent(
        &self,
        expr: &Expression,
        parent: &Expression,
        child_idx: usize,
        ctx: &RuleContext,
    ) -> VortexResult<Option<Expression>>;
}

pub(crate) trait DynTypedParentReduceRule: Send + Sync {
    fn reduce_parent(
        &self,
        expr: &Expression,
        parent: &Expression,
        child_idx: usize,
        ctx: &TypedRuleContext,
    ) -> VortexResult<Option<Expression>>;
}
