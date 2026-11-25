// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Session-registered derivations of a scalar function's result statistics from the statistics
//! of its inputs.
//!
//! Stats rewrite rules bound sub-expressions of a predicate, such as the min and max of
//! `cast(col)` in `cast(col) > 5`. Statistics are stored only for columns, so a statistic of
//! `cast(col)` must be expressed in terms of the statistics of `col`. A [`StatDerivation`]
//! registered for a scalar function says how.

use std::fmt::Debug;
use std::sync::Arc;

use vortex_array::aggregate_fn::AggregateFnRef;
use vortex_array::expr::BoundExpression;
use vortex_array::scalar_fn::ScalarFnId;
use vortex_session::VortexSession;

use crate::stats::bound;
use crate::stats::session::StatsSessionExt;

/// Shared reference to a statistic derivation.
pub type StatDerivationRef = Arc<dyn StatDerivation>;

/// Expresses the statistics of a scalar function's result in terms of the statistics of its
/// inputs.
///
/// At most one derivation is registered per scalar function. When one is registered, it decides
/// every statistic of expressions rooted at that function: stats rewrites never read a statistic
/// stored for such an expression itself.
pub trait StatDerivation: Debug + Send + Sync + 'static {
    /// Returns the scalar function id whose statistics this derivation describes.
    fn scalar_fn_id(&self) -> ScalarFnId;

    /// Expresses the statistic `aggregate_fn` of `expr`, whose root scalar function id is
    /// [`Self::scalar_fn_id`], as an expression over statistics of other expressions.
    ///
    /// Returns `None` when the statistic cannot be known from statistics.
    fn derive_stat(
        &self,
        expr: &BoundExpression,
        aggregate_fn: &AggregateFnRef,
        session: &VortexSession,
    ) -> Option<BoundExpression>;
}

/// Expresses the statistic `aggregate_fn` of `expr` as an expression over the statistics that
/// can be read for the current scope.
///
/// If a [`StatDerivation`] is registered for the root scalar function of `expr`, it decides the
/// statistic. Otherwise this is a reference to `aggregate_fn` of `expr` itself. Returns `None`
/// when the statistic cannot be known, for example when the aggregate does not support the dtype
/// of `expr`.
pub fn stat_expr(
    expr: &BoundExpression,
    aggregate_fn: &AggregateFnRef,
    session: &VortexSession,
) -> Option<BoundExpression> {
    if let Some(scalar_fn) = expr.as_scalar()
        && let Some(derivation) = session.stats().stat_derivation_for(scalar_fn.id())
    {
        return derivation.derive_stat(expr, aggregate_fn, session);
    }

    // The aggregate may not support the expression's dtype, e.g. min/max over structs,
    // even when the predicate itself is well-typed. Such stats cannot be lowered later,
    // so do not reference them in the rewrite.
    aggregate_fn
        .return_dtype(expr.dtype())
        .is_some()
        .then(|| bound::stat(expr.clone(), aggregate_fn.clone()))
}
