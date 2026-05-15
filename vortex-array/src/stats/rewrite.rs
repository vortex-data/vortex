// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Session-registered rewrite rules for aggregate-backed stats expressions.

use std::fmt::Debug;
use std::sync::Arc;

use vortex_error::VortexResult;
use vortex_session::VortexSession;

use crate::expr::Expression;
use crate::expr::or_collect;
use crate::scalar_fn::ScalarFnId;
use crate::stats::session::StatsRewriteSessionExt;

/// Shared reference to a stats rewrite rule.
pub(crate) type StatsRewriteRuleRef = Arc<dyn StatsRewriteRule>;

/// A plugin-provided rule that rewrites predicates into stats-backed proof expressions.
///
/// A falsifier evaluates to `true` only when the original predicate is definitely false for the
/// current stats scope. A satisfier evaluates to `true` only when the original predicate is
/// definitely true for the current stats scope. Returning `None` means the rule cannot prove
/// anything for the expression.
#[allow(dead_code)]
pub(crate) trait StatsRewriteRule: Debug + Send + Sync + 'static {
    /// The scalar function ID this rule applies to.
    fn scalar_fn_id(&self) -> ScalarFnId;

    /// Rewrite an expression into a stats-backed falsifier.
    fn falsify(
        &self,
        expr: &Expression,
        ctx: &StatsRewriteCtx<'_>,
    ) -> VortexResult<Option<Expression>> {
        _ = expr;
        _ = ctx;
        Ok(None)
    }

    /// Rewrite an expression into a stats-backed satisfier.
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
}

impl<'a> StatsRewriteCtx<'a> {
    /// Create a rewrite context for `session`.
    pub(crate) fn new(session: &'a VortexSession) -> Self {
        Self { session }
    }

    /// Returns the session that owns the rewrite registry.
    pub(crate) fn session(&self) -> &'a VortexSession {
        self.session
    }

    /// Rewrite `expr` into a stats-backed falsifier.
    pub(crate) fn falsify(&self, expr: &Expression) -> VortexResult<Option<Expression>> {
        rewrite(expr, self, StatsRewriteRule::falsify)
    }

    /// Rewrite `expr` into a stats-backed satisfier.
    pub(crate) fn satisfy(&self, expr: &Expression) -> VortexResult<Option<Expression>> {
        rewrite(expr, self, StatsRewriteRule::satisfy)
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
        .stats_rewrites()
        .rules_for(expr.scalar_fn().id());
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
    use crate::expr::Expression;
    use crate::expr::lit;
    use crate::expr::or;
    use crate::scalar_fn::ScalarFnId;
    use crate::scalar_fn::ScalarFnVTable;
    use crate::scalar_fn::fns::literal::Literal;
    use crate::stats::session::StatsRewriteSession;
    use crate::stats::session::StatsRewriteSessionExt;

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
        let session = VortexSession::empty().with::<StatsRewriteSession>();
        session.stats_rewrites().register(StaticLiteralRule {
            falsifier: Some(lit(false)),
            satisfier: None,
        });
        session.stats_rewrites().register(StaticLiteralRule {
            falsifier: Some(lit(true)),
            satisfier: None,
        });

        assert_eq!(lit(7).falsify(&session)?, Some(or(lit(false), lit(true))));
        Ok(())
    }

    #[test]
    fn combines_multiple_satisfiers_with_or() -> VortexResult<()> {
        let session = VortexSession::empty().with::<StatsRewriteSession>();
        session.stats_rewrites().register(StaticLiteralRule {
            falsifier: None,
            satisfier: Some(lit(false)),
        });
        session.stats_rewrites().register(StaticLiteralRule {
            falsifier: None,
            satisfier: Some(lit(true)),
        });

        assert_eq!(lit(7).satisfy(&session)?, Some(or(lit(false), lit(true))));
        Ok(())
    }

    #[test]
    fn unregistered_expression_has_no_rewrite() -> VortexResult<()> {
        let session = VortexSession::empty().with::<StatsRewriteSession>();

        assert_eq!(lit(7).falsify(&session)?, None);
        assert_eq!(lit(7).satisfy(&session)?, None);
        Ok(())
    }
}
