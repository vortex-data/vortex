// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Convert finalized source statistics into DataFusion planner statistics.
//!
//! Bounds retain their precision. Missing or unrepresentable values remain unknown, while a known
//! null sum remains distinct from a missing sum. Null extrema cannot provide a range bound.

use std::sync::LazyLock;

use datafusion_common::ColumnStatistics;
use datafusion_common::ScalarValue;
use datafusion_common::stats::Precision;
use vortex::array::aggregate_fn::AggregateFnRef;
use vortex::array::aggregate_fn::AggregateFnVTableExt;
use vortex::array::aggregate_fn::EmptyOptions;
use vortex::array::aggregate_fn::NumericalAggregateOpts;
use vortex::array::aggregate_fn::fns::is_constant::IsConstant;
use vortex::array::aggregate_fn::fns::max::Max;
use vortex::array::aggregate_fn::fns::min::Min;
use vortex::array::aggregate_fn::fns::null_count::NullCount;
use vortex::array::aggregate_fn::fns::sum::Sum;
use vortex::array::aggregate_fn::fns::uncompressed_size_in_bytes::UncompressedSizeInBytes;
use vortex::array::stats::AggregateResults;
use vortex::dtype::DType;
use vortex::error::VortexResult;
use vortex::expr::stats::Precision as VortexPrecision;
use vortex::scalar::Scalar;

use crate::PrecisionExt;
use crate::convert::TryToDataFusion;

static MIN: LazyLock<AggregateFnRef> =
    LazyLock::new(|| Min.bind(NumericalAggregateOpts::skip_nans()));
static MAX: LazyLock<AggregateFnRef> =
    LazyLock::new(|| Max.bind(NumericalAggregateOpts::skip_nans()));
static SUM: LazyLock<AggregateFnRef> =
    LazyLock::new(|| Sum.bind(NumericalAggregateOpts::skip_nans()));
static NULL_COUNT: LazyLock<AggregateFnRef> = LazyLock::new(|| NullCount.bind(EmptyOptions));
static UNCOMPRESSED_SIZE: LazyLock<AggregateFnRef> =
    LazyLock::new(|| UncompressedSizeInBytes.bind(EmptyOptions));
static IS_CONSTANT: LazyLock<AggregateFnRef> = LazyLock::new(|| IsConstant.bind(EmptyOptions));

/// Convert finalized results for a field with the given dtype.
pub(crate) fn aggregate_results_to_df(
    results: &AggregateResults,
    dtype: &DType,
) -> VortexResult<ColumnStatistics> {
    let null_count = results
        .get_result(&NULL_COUNT)
        .and_then(|value| usize::try_from(&value).ok());
    let byte_size = results
        .get_result(&UNCOMPRESSED_SIZE)
        .and_then(|value| usize::try_from(&value).ok());
    let is_constant = results
        .get_result(&IS_CONSTANT)
        .and_then(|value| bool::try_from(&value).ok());

    let extrema_dtype = (!matches!(dtype, DType::Null)).then(|| dtype.as_nullable());
    let min = results.get_result(&MIN).and_then(non_null_scalar);
    let max = results.get_result(&MAX).and_then(non_null_scalar);

    Ok(ColumnStatistics {
        null_count: null_count.to_df(),
        min_value: scalar_result_to_df(min, extrema_dtype.as_ref()).to_df(),
        max_value: scalar_result_to_df(max, extrema_dtype.as_ref()).to_df(),
        sum_value: scalar_result_to_df(results.get_result(&SUM), SUM.return_dtype(dtype).as_ref())
            .to_df(),
        distinct_count: is_constant_to_distinct_count(is_constant),
        byte_size: byte_size.to_df(),
    })
}

fn non_null_scalar(value: Scalar) -> Option<Scalar> {
    (!value.is_null()).then_some(value)
}

fn scalar_result_to_df(
    value: VortexPrecision<Scalar>,
    expected_dtype: Option<&DType>,
) -> VortexPrecision<ScalarValue> {
    value.and_then(|value| {
        // Historical scalars can retain a non-nullable field type. Only root nullability may
        // differ from the current aggregate return type.
        if value.dtype().as_nullable() != expected_dtype?.as_nullable() {
            return None;
        }

        value.try_to_df().ok()
    })
}

pub(crate) fn is_constant_to_distinct_count(
    is_constant: VortexPrecision<bool>,
) -> Precision<usize> {
    match is_constant.as_exact() {
        Some(true) => Precision::Exact(1),
        Some(false) | None => Precision::Absent,
    }
}

#[cfg(test)]
mod tests;
