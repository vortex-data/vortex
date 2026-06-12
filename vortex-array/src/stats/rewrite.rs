// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Session-registered rewrite rules for aggregate-backed stats expressions.

use std::fmt::Debug;
use std::sync::Arc;

use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_session::VortexSession;

use crate::dtype::DType;
use crate::expr::Expression;
use crate::expr::or_collect;
use crate::scalar_fn::ScalarFnId;
use crate::stats::session::StatsSessionExt;

mod builtins;

pub(crate) use builtins::register_builtins;

/// Shared reference to a stats rewrite rule.
pub(crate) type StatsRewriteRuleRef = Arc<dyn StatsRewriteRule>;

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
pub(crate) trait StatsRewriteRule: Debug + Send + Sync + 'static {
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
        expr: &Expression,
        ctx: &StatsRewriteCtx<'_>,
    ) -> VortexResult<Option<Expression>> {
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
        expr: &Expression,
        ctx: &StatsRewriteCtx<'_>,
    ) -> VortexResult<Option<Expression>> {
        _ = expr;
        _ = ctx;
        Ok(None)
    }
}

/// Context passed to stats rewrite rules.
pub(crate) struct StatsRewriteCtx<'a> {
    session: &'a VortexSession,
    scope: &'a DType,
}

impl<'a> StatsRewriteCtx<'a> {
    /// Create a rewrite context for `session`.
    pub(crate) fn new(session: &'a VortexSession, scope: &'a DType) -> Self {
        Self { session, scope }
    }

    /// Returns the session that owns the rewrite registry.
    pub(crate) fn session(&self) -> &'a VortexSession {
        self.session
    }

    /// Return the dtype of `expr` within this rewrite scope.
    pub(crate) fn return_dtype(&self, expr: &Expression) -> VortexResult<DType> {
        expr.return_dtype(self.scope)
    }

    /// Rewrite `expr` into a stats-backed falsifier.
    pub(crate) fn falsify(&self, expr: &Expression) -> VortexResult<Option<Expression>> {
        self.ensure_predicate(expr)?;
        rewrite(expr, self, StatsRewriteRule::falsify)
    }

    /// Rewrite `expr` into a stats-backed satisfier.
    pub(crate) fn satisfy(&self, expr: &Expression) -> VortexResult<Option<Expression>> {
        self.ensure_predicate(expr)?;
        rewrite(expr, self, StatsRewriteRule::satisfy)
    }

    fn ensure_predicate(&self, expr: &Expression) -> VortexResult<()> {
        let dtype = self.return_dtype(expr)?;
        vortex_ensure!(
            matches!(dtype, DType::Bool(_)),
            "Stats rewrites require a boolean predicate, got {dtype}",
        );
        Ok(())
    }
}

fn rewrite(
    expr: &Expression,
    ctx: &StatsRewriteCtx<'_>,
    apply: fn(
        &dyn StatsRewriteRule,
        &Expression,
        &StatsRewriteCtx<'_>,
    ) -> VortexResult<Option<Expression>>,
) -> VortexResult<Option<Expression>> {
    let rules = ctx
        .session()
        .stats()
        .rewrite_rules_for(expr.scalar_fn().id());
    let Some(rules) = rules else {
        return Ok(None);
    };

    let mut rewrites = Vec::new();
    for rule in rules.iter() {
        if let Some(rewrite) = apply(rule.as_ref(), expr, ctx)? {
            rewrites.push(rewrite);
        }
    }

    Ok(or_collect(rewrites))
}

#[cfg(test)]
mod tests {
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use super::StatsRewriteCtx;
    use super::StatsRewriteRule;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::expr::Expression;
    use crate::expr::lit;
    use crate::expr::or;
    use crate::scalar_fn::ScalarFnId;
    use crate::scalar_fn::ScalarFnVTable;
    use crate::scalar_fn::fns::literal::Literal;
    use crate::stats::session::StatsSession;
    use crate::stats::session::StatsSessionExt;

    #[derive(Debug)]
    struct StaticLiteralRule {
        falsifier: Option<Expression>,
        satisfier: Option<Expression>,
    }

    impl StatsRewriteRule for StaticLiteralRule {
        fn scalar_fn_id(&self) -> ScalarFnId {
            Literal.id()
        }

        fn falsify(
            &self,
            _expr: &Expression,
            _ctx: &StatsRewriteCtx<'_>,
        ) -> VortexResult<Option<Expression>> {
            Ok(self.falsifier.clone())
        }

        fn satisfy(
            &self,
            _expr: &Expression,
            _ctx: &StatsRewriteCtx<'_>,
        ) -> VortexResult<Option<Expression>> {
            Ok(self.satisfier.clone())
        }
    }

    #[test]
    fn combines_multiple_falsifiers_with_or() -> VortexResult<()> {
        let session = VortexSession::empty().with::<StatsSession>();
        let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        session.stats().register_rewrite(StaticLiteralRule {
            falsifier: Some(lit(false)),
            satisfier: None,
        });
        session.stats().register_rewrite(StaticLiteralRule {
            falsifier: Some(lit(true)),
            satisfier: None,
        });

        assert_eq!(
            lit(true).falsify(&dtype, &session)?,
            Some(or(lit(false), lit(true)))
        );
        Ok(())
    }

    #[test]
    fn combines_multiple_satisfiers_with_or() -> VortexResult<()> {
        let session = VortexSession::empty().with::<StatsSession>();
        let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        session.stats().register_rewrite(StaticLiteralRule {
            falsifier: None,
            satisfier: Some(lit(false)),
        });
        session.stats().register_rewrite(StaticLiteralRule {
            falsifier: None,
            satisfier: Some(lit(true)),
        });

        assert_eq!(
            lit(true).satisfy(&dtype, &session)?,
            Some(or(lit(false), lit(true)))
        );
        Ok(())
    }

    #[test]
    fn unregistered_expression_has_no_rewrite() -> VortexResult<()> {
        let session = VortexSession::empty().with::<StatsSession>();
        let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);

        assert_eq!(lit(true).falsify(&dtype, &session)?, None);
        assert_eq!(lit(true).satisfy(&dtype, &session)?, None);
        Ok(())
    }

    #[test]
    fn non_predicate_expression_errors() {
        let session = VortexSession::empty().with::<StatsSession>();
        let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);

        assert!(lit(7).falsify(&dtype, &session).is_err());
        assert!(lit(7).satisfy(&dtype, &session).is_err());
    }
}
