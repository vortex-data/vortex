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

#[cfg(test)]
mod tests {
    use vortex_array::aggregate_fn::AggregateFnRef;
    use vortex_array::aggregate_fn::AggregateFnVTableExt;
    use vortex_array::aggregate_fn::NumericalAggregateOpts;
    use vortex_array::aggregate_fn::fns::max::Max;
    use vortex_array::aggregate_fn::fns::min::Min;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::StructFields;
    use vortex_array::expr::BoundExpression;
    use vortex_array::expr::bound::lit;
    use vortex_array::expr::col;
    use vortex_array::scalar_fn::ScalarFnId;
    use vortex_array::scalar_fn::ScalarFnVTable;
    use vortex_error::VortexResult;
    use vortex_functions::byte_length::ByteLength;
    use vortex_functions::expr::byte_length;
    use vortex_session::VortexSession;

    use super::StatDerivation;
    use super::stat_expr;
    use crate::stats::session::StatsSessionExt;

    /// Bounds the max byte length of every value by a fixed length, and knows no other statistic.
    #[derive(Debug)]
    struct MaxByteLength(u64);

    impl StatDerivation for MaxByteLength {
        fn scalar_fn_id(&self) -> ScalarFnId {
            ByteLength.id()
        }

        fn derive_stat(
            &self,
            _expr: &BoundExpression,
            aggregate_fn: &AggregateFnRef,
            _session: &VortexSession,
        ) -> Option<BoundExpression> {
            aggregate_fn.is::<Max>().then(|| lit(self.0))
        }
    }

    fn byte_length_of_column() -> VortexResult<BoundExpression> {
        byte_length(col("s")).bind(&DType::Struct(
            StructFields::from_iter([("s", DType::Utf8(Nullability::NonNullable))]),
            Nullability::NonNullable,
        ))
    }

    #[test]
    fn registered_derivation_decides_every_statistic() -> VortexResult<()> {
        let session = VortexSession::empty();
        session.stats().register_stat_derivation(MaxByteLength(8));
        let expr = byte_length_of_column()?;

        assert_eq!(
            stat_expr(
                &expr,
                &Max.bind(NumericalAggregateOpts::skip_nans()),
                &session
            ),
            Some(lit(8u64))
        );
        // The derivation knows no min, and no stored min of `byte_length(s)` is read instead.
        assert_eq!(
            stat_expr(
                &expr,
                &Min.bind(NumericalAggregateOpts::skip_nans()),
                &session
            ),
            None
        );
        Ok(())
    }

    #[test]
    fn later_registration_replaces_earlier() -> VortexResult<()> {
        let session = VortexSession::empty();
        session.stats().register_stat_derivation(MaxByteLength(8));
        session.stats().register_stat_derivation(MaxByteLength(16));

        assert_eq!(
            stat_expr(
                &byte_length_of_column()?,
                &Max.bind(NumericalAggregateOpts::skip_nans()),
                &session
            ),
            Some(lit(16u64))
        );
        Ok(())
    }
}
