// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

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

use crate::PrecisionExt;
use crate::convert::TryToDataFusion;

/// Convert finalized file summaries, preserving unknown values and bound precision.
pub(crate) fn aggregate_results_to_df(
    results: &AggregateResults,
    dtype: &DType,
) -> VortexResult<ColumnStatistics> {
    let null_count = results
        .get_result(&NullCount.bind(EmptyOptions))
        .and_then(|value| usize::try_from(&value).ok());
    let byte_size = results
        .get_result(&UncompressedSizeInBytes.bind(EmptyOptions))
        .and_then(|value| usize::try_from(&value).ok());
    let is_constant = results
        .get_result(&IsConstant.bind(EmptyOptions))
        .and_then(|value| bool::try_from(&value).ok());
    let options = NumericalAggregateOpts::skip_nans();
    Ok(ColumnStatistics {
        null_count: null_count.to_df(),
        min_value: scalar_result_to_df(results, &Min.bind(options), dtype).to_df(),
        max_value: scalar_result_to_df(results, &Max.bind(options), dtype).to_df(),
        sum_value: scalar_result_to_df(results, &Sum.bind(options), dtype).to_df(),
        distinct_count: is_constant_to_distinct_count(is_constant),
        byte_size: byte_size.to_df(),
    })
}

fn scalar_result_to_df(
    results: &AggregateResults,
    aggregate: &AggregateFnRef,
    dtype: &DType,
) -> VortexPrecision<ScalarValue> {
    results.get_result(aggregate).and_then(|value| {
        if value.dtype() != &aggregate.return_dtype(dtype)? {
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
mod tests {
    use rstest::rstest;
    use vortex::dtype::Nullability;
    use vortex::dtype::PType;
    use vortex::scalar::Scalar;

    use super::*;

    #[rstest]
    #[case(true, Precision::Exact(1))]
    #[case(false, Precision::Absent)]
    fn constant_distinct_count(
        #[case] constant: bool,
        #[case] expected: Precision<usize>,
    ) -> VortexResult<()> {
        let dtype = DType::Bool(Nullability::NonNullable);
        let results = AggregateResults::try_new(
            &dtype,
            [(
                IsConstant.bind(EmptyOptions),
                VortexPrecision::Exact(constant.into()),
            )],
        )?;
        assert_eq!(
            aggregate_results_to_df(&results, &dtype)?.distinct_count,
            expected
        );
        Ok(())
    }

    #[rstest]
    #[case(true)]
    #[case(false)]
    fn extrema_keep_precision(#[case] exact: bool) -> VortexResult<()> {
        let dtype = DType::from(PType::I32);
        let scalar = Scalar::primitive(7i32, Nullability::Nullable);
        let result = if exact {
            VortexPrecision::Exact(scalar)
        } else {
            VortexPrecision::Inexact(scalar)
        };
        let results = AggregateResults::try_new(
            &dtype,
            [(Min.bind(NumericalAggregateOpts::skip_nans()), result)],
        )?;
        let expected = ScalarValue::Int32(Some(7));
        let expected = if exact {
            Precision::Exact(expected)
        } else {
            Precision::Inexact(expected)
        };
        let stats = aggregate_results_to_df(&results, &dtype)?;
        assert_eq!(stats.min_value, expected);
        assert_eq!(stats.max_value, Precision::Absent);
        assert_eq!(stats.null_count, Precision::Absent);
        Ok(())
    }

    #[test]
    fn exact_null_sum_differs_from_missing_metadata() -> VortexResult<()> {
        let dtype = DType::from(PType::I64);
        let results = AggregateResults::try_new(
            &dtype,
            [(
                Sum.bind(NumericalAggregateOpts::skip_nans()),
                VortexPrecision::Exact(Scalar::null(dtype.as_nullable())),
            )],
        )?;
        let stats = aggregate_results_to_df(&results, &dtype)?;
        assert_eq!(stats.sum_value, Precision::Exact(ScalarValue::Int64(None)));
        assert_eq!(stats.min_value, Precision::Absent);
        Ok(())
    }

    #[rstest]
    #[case::unsupported(DType::Null)]
    #[case::incompatible(DType::Bool(Nullability::NonNullable))]
    fn incompatible_column_type_has_no_extrema(#[case] dtype: DType) -> VortexResult<()> {
        let results = AggregateResults::try_new(
            &DType::from(PType::I32),
            [(
                Min.bind(NumericalAggregateOpts::skip_nans()),
                VortexPrecision::Exact(Scalar::primitive(7i32, Nullability::Nullable)),
            )],
        )?;
        assert_eq!(
            aggregate_results_to_df(&results, &dtype)?.min_value,
            Precision::Absent
        );
        Ok(())
    }
}
