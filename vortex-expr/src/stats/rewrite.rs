// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Session-registered rewrite rules for aggregate-backed stats expressions.

use std::fmt::Debug;
use std::sync::Arc;

use vortex_array::dtype::DType;
use vortex_array::scalar_fn::ScalarFnId;
use vortex_array::scalar_fn::builtin::binary::Binary;
use vortex_array::scalar_fn::builtin::operators::Operator;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_session::VortexSession;
use vortex_utils::iter::ReduceBalancedIterExt;

use crate::BoundExpression;
use crate::ScalarFnExprExt;
use crate::stats::session::StatsSessionExt;

mod builtins;
#[cfg(test)]
mod fn_tests;
pub mod helpers;

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
/// [`Self::scalar_fn_id`].
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
        expr: &BoundExpression,
        session: &VortexSession,
    ) -> VortexResult<Option<BoundExpression>> {
        _ = expr;
        _ = session;
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
        expr: &BoundExpression,
        session: &VortexSession,
    ) -> VortexResult<Option<BoundExpression>> {
        _ = expr;
        _ = session;
        Ok(None)
    }
}

fn ensure_predicate(expr: &BoundExpression) -> VortexResult<()> {
    let dtype = expr.dtype();
    vortex_ensure!(
        matches!(dtype, DType::Bool(_)),
        "Stats rewrites require a boolean predicate, got {dtype}",
    );
    Ok(())
}

/// Rewrite `expr` into a stats-backed falsifier.
pub fn falsify(
    expr: &BoundExpression,
    session: &VortexSession,
) -> VortexResult<Option<BoundExpression>> {
    ensure_predicate(expr)?;
    rewrite(expr, session, StatsRewriteRule::falsify)
}

/// Rewrite `expr` into a stats-backed satisfier.
pub fn satisfy(
    expr: &BoundExpression,
    session: &VortexSession,
) -> VortexResult<Option<BoundExpression>> {
    ensure_predicate(expr)?;
    rewrite(expr, session, StatsRewriteRule::satisfy)
}

/// Statistics-based pruning methods on [`BoundExpression`] predicates.
pub trait BoundExpressionPruneExt {
    /// Return an expression that proves this predicate is definitely false from statistics.
    fn falsify(&self, session: &VortexSession) -> VortexResult<Option<BoundExpression>>;

    /// Return an expression that proves this predicate is definitely true from statistics.
    fn satisfy(&self, session: &VortexSession) -> VortexResult<Option<BoundExpression>>;
}

impl BoundExpressionPruneExt for BoundExpression {
    fn falsify(&self, session: &VortexSession) -> VortexResult<Option<BoundExpression>> {
        falsify(self, session)
    }

    fn satisfy(&self, session: &VortexSession) -> VortexResult<Option<BoundExpression>> {
        satisfy(self, session)
    }
}

fn rewrite(
    expr: &BoundExpression,
    session: &VortexSession,
    apply: fn(
        &dyn StatsRewriteRule,
        &BoundExpression,
        &VortexSession,
    ) -> VortexResult<Option<BoundExpression>>,
) -> VortexResult<Option<BoundExpression>> {
    // The scope alone proves nothing about the rows it contains.
    let Some(scalar_fn) = expr.as_scalar() else {
        return Ok(None);
    };
    let rules = session.stats().rewrite_rules_for(scalar_fn.id());
    let Some(rules) = rules else {
        return Ok(None);
    };

    let mut rewrites = Vec::new();
    for rule in rules.iter() {
        if let Some(rewrite) = apply(rule.as_ref(), expr, session)? {
            rewrites.push(rewrite);
        }
    }

    rewrites
        .into_iter()
        .try_reduce_balanced(|lhs, rhs| Binary.try_new_bound_expr(Operator::Or, [lhs, rhs]))
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::scalar_fn::ScalarFnId;
    use vortex_array::scalar_fn::ScalarFnVTable;
    use vortex_array::scalar_fn::builtin::not::Not;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use super::StatsRewriteRule;
    use crate::BoundExpression;
    use crate::lit;
    use crate::not;
    use crate::or;
    use crate::stats::rewrite::BoundExpressionPruneExt;
    use crate::stats::session::StatsSessionExt;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

    #[derive(Debug)]
    struct StaticNotRule {
        falsifier: Option<BoundExpression>,
        satisfier: Option<BoundExpression>,
    }

    impl StatsRewriteRule for StaticNotRule {
        fn scalar_fn_id(&self) -> ScalarFnId {
            Not.id()
        }

        fn falsify(
            &self,
            _expr: &BoundExpression,
            _session: &VortexSession,
        ) -> VortexResult<Option<BoundExpression>> {
            Ok(self.falsifier.clone())
        }

        fn satisfy(
            &self,
            _expr: &BoundExpression,
            _session: &VortexSession,
        ) -> VortexResult<Option<BoundExpression>> {
            Ok(self.satisfier.clone())
        }
    }

    #[test]
    fn combines_multiple_falsifiers_with_or() -> VortexResult<()> {
        let session = VortexSession::empty();
        let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        session.stats().register_rewrite(StaticNotRule {
            falsifier: Some(lit(false).bind(&dtype)?),
            satisfier: None,
        });
        session.stats().register_rewrite(StaticNotRule {
            falsifier: Some(lit(true).bind(&dtype)?),
            satisfier: None,
        });

        assert_eq!(
            not(lit(true)).bind(&dtype)?.falsify(&session)?,
            Some(or(lit(false), lit(true)).bind(&dtype)?)
        );
        Ok(())
    }

    #[test]
    fn combines_multiple_satisfiers_with_or() -> VortexResult<()> {
        let session = VortexSession::empty();
        let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        session.stats().register_rewrite(StaticNotRule {
            falsifier: None,
            satisfier: Some(lit(false).bind(&dtype)?),
        });
        session.stats().register_rewrite(StaticNotRule {
            falsifier: None,
            satisfier: Some(lit(true).bind(&dtype)?),
        });

        assert_eq!(
            not(lit(true)).bind(&dtype)?.satisfy(&session)?,
            Some(or(lit(false), lit(true)).bind(&dtype)?)
        );
        Ok(())
    }

    #[test]
    fn unregistered_expression_has_no_rewrite() -> VortexResult<()> {
        let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);

        let expr = not(lit(true)).bind(&dtype)?;
        assert_eq!(expr.falsify(&SESSION)?, None);
        assert_eq!(expr.satisfy(&SESSION)?, None);
        Ok(())
    }

    #[test]
    fn non_predicate_expression_errors() -> VortexResult<()> {
        let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);

        let expr = lit(7).bind(&dtype)?;
        assert!(expr.falsify(&SESSION).is_err());
        assert!(expr.satisfy(&SESSION).is_err());
        Ok(())
    }
}
