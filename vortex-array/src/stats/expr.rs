// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Expression constructors for statistics backed by aggregate functions.

use vortex_error::VortexExpect;

use crate::aggregate_fn::AggregateFnRef;
use crate::aggregate_fn::AggregateFnVTableExt;
use crate::aggregate_fn::EmptyOptions;
use crate::aggregate_fn::NumericalAggregateOpts;
use crate::aggregate_fn::fns::all_nan::AllNan;
use crate::aggregate_fn::fns::all_non_nan::AllNonNan;
use crate::aggregate_fn::fns::all_non_null::AllNonNull;
use crate::aggregate_fn::fns::all_null::AllNull;
use crate::aggregate_fn::fns::min_max::MinMax;
use crate::aggregate_fn::fns::nan_count::NanCount;
use crate::aggregate_fn::fns::null_count::NullCount;
use crate::aggregate_fn::fns::sum::Sum;
use crate::expr::BoundExpression;
use crate::scalar_fn::ScalarFnVTableExt;
pub use crate::scalar_fn::fns::stat::StatFn;
pub use crate::scalar_fn::fns::stat::StatOptions;

/// Constructors for statistics over typed expression inputs.
pub mod bound {
    use super::*;

    /// Read a stored aggregate statistic.
    pub fn stat(expr: BoundExpression, aggregate_fn: AggregateFnRef) -> BoundExpression {
        StatFn
            .try_new_bound_expr(StatOptions::new(aggregate_fn), [expr])
            .vortex_expect("stat expressions must use an aggregate supported by the child dtype")
    }

    /// Read a nullable `{ min, max }` statistic.
    pub fn min_max(expr: BoundExpression) -> BoundExpression {
        stat(expr, MinMax.bind(NumericalAggregateOpts::skip_nans()))
    }

    /// Read a nullable sum statistic.
    pub fn sum(expr: BoundExpression) -> BoundExpression {
        stat(expr, Sum.bind(NumericalAggregateOpts::skip_nans()))
    }

    /// Read a nullable null-count statistic.
    pub fn null_count(expr: BoundExpression) -> BoundExpression {
        stat(expr, NullCount.bind(EmptyOptions))
    }

    /// Read a nullable all-null statistic.
    pub fn all_null(expr: BoundExpression) -> BoundExpression {
        stat(expr, AllNull.bind(EmptyOptions))
    }

    /// Read a nullable all-NaN statistic.
    pub fn all_nan(expr: BoundExpression) -> BoundExpression {
        stat(expr, AllNan.bind(EmptyOptions))
    }

    /// Read a nullable all-non-null statistic.
    pub fn all_non_null(expr: BoundExpression) -> BoundExpression {
        stat(expr, AllNonNull.bind(EmptyOptions))
    }

    /// Read a nullable all-non-NaN statistic.
    pub fn all_non_nan(expr: BoundExpression) -> BoundExpression {
        stat(expr, AllNonNan.bind(EmptyOptions))
    }

    /// Read a nullable NaN-count statistic.
    pub fn nan_count(expr: BoundExpression) -> BoundExpression {
        stat(expr, NanCount.bind(EmptyOptions))
    }
}
