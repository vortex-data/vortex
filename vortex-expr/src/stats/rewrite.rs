// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Session-registered rewrite rules for aggregate-backed stats expressions.

use std::fmt::Debug;
use std::sync::Arc;

use vortex_array::dtype::DType;
use vortex_array::expr::BoundExpression;
use vortex_array::scalar_fn::ScalarFnId;
use vortex_array::scalar_fn::ScalarFnVTableExt;
use vortex_array::scalar_fn::fns::binary::Binary;
use vortex_array::scalar_fn::fns::operators::Operator;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_session::VortexSession;
use vortex_utils::iter::ReduceBalancedIterExt;

use crate::stats::rewrite::cache::is_cacheable;
use crate::stats::session::RewriteLookup;
use crate::stats::session::StatsSessionExt;

mod builtins;
mod cache;
mod dynamic_expression;
#[cfg(test)]
mod fn_tests;
mod like_variant;

pub(crate) use builtins::register_builtins;
pub(crate) use cache::RewriteCache;
pub(crate) use dynamic_expression::StatsProof;

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
///
/// The session caches the proof of each predicate while the predicate is alive, until a new rule
/// is registered. Thus a proof must only depend on `expr` and on the registered rules. A proof
/// must not hold the session or `expr` itself, because the session cache would then keep them.
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
    rewrite(expr, session, StatsProof::Falsify)
}

/// Rewrite `expr` into a stats-backed satisfier.
pub fn satisfy(
    expr: &BoundExpression,
    session: &VortexSession,
) -> VortexResult<Option<BoundExpression>> {
    ensure_predicate(expr)?;
    rewrite(expr, session, StatsProof::Satisfy)
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

/// Rewrites `expr` with the rules of the session, and keeps the result in the session cache.
fn rewrite(
    expr: &BoundExpression,
    session: &VortexSession,
    proof: StatsProof,
) -> VortexResult<Option<BoundExpression>> {
    // The scope alone proves nothing about the rows it contains.
    let Some(scalar_fn) = expr.as_scalar() else {
        return Ok(None);
    };

    let stats = session.stats();
    if !is_cacheable(expr) {
        let rules = stats.rewrite_rules_for(scalar_fn.id());
        drop(stats);
        return apply_rules(expr, session, proof, rules.as_deref().unwrap_or_default());
    }

    let (revision, rules) = match stats.lookup_rewrite(proof, expr, scalar_fn.id()) {
        RewriteLookup::Cached(rewrite) => return Ok(rewrite),
        RewriteLookup::Uncached { revision, rules } => (revision, rules),
    };
    drop(stats);

    let rewrite = apply_rules(expr, session, proof, rules.as_deref().unwrap_or_default())?;
    session
        .stats()
        .cache_rewrite(proof, revision, expr, rewrite.clone());
    Ok(rewrite)
}

fn apply_rules(
    expr: &BoundExpression,
    session: &VortexSession,
    proof: StatsProof,
    rules: &[StatsRewriteRuleRef],
) -> VortexResult<Option<BoundExpression>> {
    let mut rewrites = Vec::new();
    for rule in rules {
        let rewrite = match proof {
            StatsProof::Falsify => rule.falsify(expr, session)?,
            StatsProof::Satisfy => rule.satisfy(expr, session)?,
        };
        rewrites.extend(rewrite);
    }

    rewrites
        .into_iter()
        .try_reduce_balanced(|lhs, rhs| Binary.try_new_bound_expr(Operator::Or, [lhs, rhs]))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::LazyLock;
    use std::sync::atomic::AtomicUsize;
    use std::sync::atomic::Ordering;

    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::expr::BoundExpression;
    use vortex_array::expr::lit;
    use vortex_array::expr::not;
    use vortex_array::expr::or;
    use vortex_array::expr::root;
    use vortex_array::scalar_fn::ScalarFnId;
    use vortex_array::scalar_fn::ScalarFnVTable;
    use vortex_array::scalar_fn::fns::not::Not;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use super::StatsRewriteRule;
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

    /// Counts the calls to the rule, and proves nothing.
    #[derive(Debug, Default)]
    struct CountingNotRule {
        calls: Arc<AtomicUsize>,
    }

    impl StatsRewriteRule for CountingNotRule {
        fn scalar_fn_id(&self) -> ScalarFnId {
            Not.id()
        }

        fn falsify(
            &self,
            _expr: &BoundExpression,
            _session: &VortexSession,
        ) -> VortexResult<Option<BoundExpression>> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(None)
        }

        fn satisfy(
            &self,
            _expr: &BoundExpression,
            _session: &VortexSession,
        ) -> VortexResult<Option<BoundExpression>> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(None)
        }
    }

    #[test]
    fn session_caches_rewrites_of_the_same_tree() -> VortexResult<()> {
        let session = VortexSession::empty();
        let calls = Arc::new(AtomicUsize::new(0));
        session.stats().register_rewrite(CountingNotRule {
            calls: Arc::clone(&calls),
        });

        // A scan binds its filter once, and proves the same tree for each file.
        let dtype = DType::Bool(Nullability::NonNullable);
        let expr = not(root()).bind(&dtype)?;
        for _ in 0..3 {
            let session = session.clone();
            assert_eq!(expr.falsify(&session)?, None);
            assert_eq!(expr.satisfy(&session)?, None);
        }
        assert_eq!(calls.load(Ordering::SeqCst), 2);

        // A new bind is a new tree, thus a new key.
        assert_eq!(not(root()).bind(&dtype)?.falsify(&session)?, None);
        assert_eq!(calls.load(Ordering::SeqCst), 3);
        Ok(())
    }

    #[test]
    fn registering_a_rule_clears_cached_rewrites() -> VortexResult<()> {
        let session = VortexSession::empty();
        let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        let expr = not(lit(true)).bind(&dtype)?;
        assert_eq!(expr.falsify(&session)?, None);

        session.stats().register_rewrite(StaticNotRule {
            falsifier: Some(lit(true).bind(&dtype)?),
            satisfier: None,
        });
        assert_eq!(expr.falsify(&session)?, Some(lit(true).bind(&dtype)?));
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
