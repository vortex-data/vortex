// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Registering statistic derivations for a scalar function from outside vortex-expr.

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
use vortex_expr::stats::derive::StatDerivation;
use vortex_expr::stats::derive::stat_expr;
use vortex_expr::stats::session::StatsSessionExt;
use vortex_session::VortexSession;

use crate::byte_length::ByteLength;
use crate::expr::byte_length;

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
