// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Session-registered rewrite rules for aggregate-backed stats expressions.

use std::fmt::Debug;
use std::sync::Arc;

use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_session::VortexSession;
use vortex_utils::iter::ReduceBalancedIterExt;

use crate::dtype::DType;
use crate::expr::BoundExpressionRef;
use crate::scalar_fn::ScalarFnId;
use crate::scalar_fn::ScalarFnVTableExt;
use crate::scalar_fn::fns::binary::Binary;
use crate::scalar_fn::fns::operators::Operator;
use crate::stats::session::StatsSessionExt;

mod builtins;

pub(crate) use builtins::register_builtins;

/// Shared reference to a stats rewrite rule.
pub type StatsRewriteRuleRef = Arc<dyn StatsRewriteRule>;

/// A plugin-provided rule for predicates whose root scalar function matches this rule.
///
/// Rules do not produce expressions equivalent to `expr`. They produce optional sufficient
/// conditions over stats for the current scope:
///
/// - a falsifier evaluating to `true` proves that `expr` is false for every row in the scope;
/// - a satisfier evaluating to `true` proves that `expr` is true for every row in the scope.
///
/// Returning `None` means this rule cannot prove anything for the expression. A returned proof
/// expression that evaluates to `false` or `null` is also inconclusive.
///
/// Multiple rules may be registered for the same scalar function. Their proofs are combined with
/// `OR`, so every proof returned by an individual rule must be sound on its own.
///
/// `expr` is the full predicate expression whose root scalar function id is
/// [`Self::scalar_fn_id`]. Use [`StatsRewriteCtx`] to resolve dtypes and recursively rewrite child
/// predicates.
pub trait StatsRewriteRule: Debug + Send + Sync + 'static {
    /// Returns the scalar function id handled by this rule.
    fn scalar_fn_id(&self) -> ScalarFnId;

    /// Returns a stats-backed proof that `expr` is false for the current scope.
    ///
    /// If the returned expression evaluates to `true` against the scope's stats, then `expr` is
    /// guaranteed to be false for every row in that scope. A returned proof expression that
    /// evaluates to `false` or `null` is inconclusive.
    ///
    /// Returns `Ok(None)` when this rule cannot construct a sound falsity proof for `expr`.
    fn falsify(
        &self,
        expr: &BoundExpressionRef,
        ctx: &StatsRewriteCtx<'_>,
    ) -> VortexResult<Option<BoundExpressionRef>> {
        _ = expr;
        _ = ctx;
        Ok(None)
    }

    /// Returns a stats-backed proof that `expr` is true for the current scope.
    ///
    /// If the returned expression evaluates to `true` against the scope's stats, then `expr` is
    /// guaranteed to be true for every row in that scope. A returned proof expression that
    /// evaluates to `false` or `null` is inconclusive.
    ///
    /// This is not the complement of [`Self::falsify`]; both methods are one-way proofs and may be
    /// implemented independently.
    ///
    /// Returns `Ok(None)` when this rule cannot construct a sound truth proof for `expr`.
    fn satisfy(
        &self,
        expr: &BoundExpressionRef,
        ctx: &StatsRewriteCtx<'_>,
    ) -> VortexResult<Option<BoundExpressionRef>> {
        _ = expr;
        _ = ctx;
        Ok(None)
    }
}

/// Context passed to stats rewrite rules.
pub struct StatsRewriteCtx<'a> {
    session: &'a VortexSession,
}

impl<'a> StatsRewriteCtx<'a> {
    /// Create a rewrite context for `session`.
    pub fn new(session: &'a VortexSession) -> Self {
        Self { session }
    }

    /// Returns the session that owns the rewrite registry.
    pub fn session(&self) -> &'a VortexSession {
        self.session
    }

    /// Return the dtype of `expr` within this rewrite scope.
    pub fn return_dtype(&self, expr: &BoundExpressionRef) -> VortexResult<DType> {
        Ok(expr.dtype().clone())
    }

    /// Rewrite `expr` into a stats-backed falsifier.
    pub fn falsify(&self, expr: &BoundExpressionRef) -> VortexResult<Option<BoundExpressionRef>> {
        self.ensure_predicate(expr)?;
        rewrite(expr, self, StatsRewriteRule::falsify)
    }

    /// Rewrite `expr` into a stats-backed satisfier.
    pub fn satisfy(&self, expr: &BoundExpressionRef) -> VortexResult<Option<BoundExpressionRef>> {
        self.ensure_predicate(expr)?;
        rewrite(expr, self, StatsRewriteRule::satisfy)
    }

    fn ensure_predicate(&self, expr: &BoundExpressionRef) -> VortexResult<()> {
        let dtype = self.return_dtype(expr)?;
        vortex_ensure!(
            matches!(dtype, DType::Bool(_)),
            "Stats rewrites require a boolean predicate, got {dtype}",
        );
        Ok(())
    }
}

fn rewrite(
    expr: &BoundExpressionRef,
    ctx: &StatsRewriteCtx<'_>,
    apply: fn(
        &dyn StatsRewriteRule,
        &BoundExpressionRef,
        &StatsRewriteCtx<'_>,
    ) -> VortexResult<Option<BoundExpressionRef>>,
) -> VortexResult<Option<BoundExpressionRef>> {
    // The scope alone proves nothing about the rows it contains.
    let Some(scalar_fn) = expr.as_scalar() else {
        return Ok(None);
    };
    let rules = ctx.session().stats().rewrite_rules_for(scalar_fn.id());
    let Some(rules) = rules else {
        return Ok(None);
    };

    let mut rewrites = Vec::new();
    for rule in rules.iter() {
        if let Some(rewrite) = apply(rule.as_ref(), expr, ctx)? {
            rewrites.push(rewrite);
        }
    }

    rewrites
        .into_iter()
        .try_reduce_balanced(|lhs, rhs| Binary.try_new_bound_expr(Operator::Or, [lhs, rhs]))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use vortex_error::VortexResult;

    use super::StatsRewriteCtx;
    use super::StatsRewriteRule;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::expr::BoundExpressionRef;
    use crate::expr::lit;
    use crate::expr::or;
    use crate::scalar_fn::ScalarFnId;
    use crate::scalar_fn::ScalarFnVTable;
    use crate::scalar_fn::fns::literal::Literal;
    use crate::stats::session::StatsSessionExt;

    #[derive(Debug)]
    struct StaticLiteralRule {
        falsifier: Option<BoundExpressionRef>,
        satisfier: Option<BoundExpressionRef>,
    }

    impl StatsRewriteRule for StaticLiteralRule {
        fn scalar_fn_id(&self) -> ScalarFnId {
            Literal.id()
        }

        fn falsify(
            &self,
            _expr: &BoundExpressionRef,
            _ctx: &StatsRewriteCtx<'_>,
        ) -> VortexResult<Option<BoundExpressionRef>> {
            Ok(self.falsifier.clone())
        }

        fn satisfy(
            &self,
            _expr: &BoundExpressionRef,
            _ctx: &StatsRewriteCtx<'_>,
        ) -> VortexResult<Option<BoundExpressionRef>> {
            Ok(self.satisfier.clone())
        }
    }

    #[test]
    fn combines_multiple_falsifiers_with_or() -> VortexResult<()> {
        let session = crate::array_session();
        let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        session.stats().register_rewrite(StaticLiteralRule {
            falsifier: Some(lit(false).bind(&dtype)?),
            satisfier: None,
        });
        session.stats().register_rewrite(StaticLiteralRule {
            falsifier: Some(lit(true).bind(&dtype)?),
            satisfier: None,
        });

        assert_eq!(
            lit(true).bind(&dtype)?.falsify(&session)?,
            Some(or(lit(false), lit(true)).bind(&dtype)?)
        );
        Ok(())
    }

    #[test]
    fn combines_multiple_satisfiers_with_or() -> VortexResult<()> {
        let session = crate::array_session();
        let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        session.stats().register_rewrite(StaticLiteralRule {
            falsifier: None,
            satisfier: Some(lit(false).bind(&dtype)?),
        });
        session.stats().register_rewrite(StaticLiteralRule {
            falsifier: None,
            satisfier: Some(lit(true).bind(&dtype)?),
        });

        assert_eq!(
            lit(true).bind(&dtype)?.satisfy(&session)?,
            Some(or(lit(false), lit(true)).bind(&dtype)?)
        );
        Ok(())
    }

    #[test]
    fn unregistered_expression_has_no_rewrite() -> VortexResult<()> {
        let session = crate::array_session();
        let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);

        let expr = lit(true).bind(&dtype)?;
        assert_eq!(Arc::clone(&expr).falsify(&session)?, None);
        assert_eq!(expr.satisfy(&session)?, None);
        Ok(())
    }

    #[test]
    fn non_predicate_expression_errors() -> VortexResult<()> {
        let session = crate::array_session();
        let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);

        let expr = lit(7).bind(&dtype)?;
        assert!(Arc::clone(&expr).falsify(&session).is_err());
        assert!(expr.satisfy(&session).is_err());
        Ok(())
    }
}
