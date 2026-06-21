// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use datafusion_common::ColumnStatistics;
use datafusion_common::ScalarValue;
use datafusion_common::stats::Precision;
use vortex::array::aggregate_fn::AggregateFnRef;
use vortex::array::aggregate_fn::AggregateFnVTableExt;
use vortex::array::aggregate_fn::EmptyOptions;
use vortex::array::aggregate_fn::NumericalAggregateOpts;
use vortex::array::aggregate_fn::fns::max::Max;
use vortex::array::aggregate_fn::fns::min::Min;
use vortex::array::aggregate_fn::fns::null_count::NullCount;
use vortex::array::aggregate_fn::fns::sum::Sum;
use vortex::array::aggregate_fn::fns::uncompressed_size_in_bytes::UncompressedSizeInBytes;
use vortex::array::stats::StatsSet;
use vortex::dtype::DType;
use vortex::dtype::Nullability;
use vortex::dtype::PType;
use vortex::error::VortexExpect;
use vortex::error::VortexResult;
use vortex::error::vortex_err;
use vortex::expr::stats::Precision as VortexPrecision;
use vortex::expr::stats::Stat;
use vortex::scalar::Scalar;

use crate::PrecisionExt;
use crate::convert::TryToDataFusion;

const MIN_INDEX: usize = 0;
const MAX_INDEX: usize = 1;
const SUM_INDEX: usize = 2;
const NULL_COUNT_INDEX: usize = 3;
const BYTE_SIZE_INDEX: usize = 4;

pub(crate) fn column_statistics_aggregate_fns() -> Vec<AggregateFnRef> {
    vec![
        Min.bind(NumericalAggregateOpts::default()),
        Max.bind(NumericalAggregateOpts::default()),
        Sum.bind(NumericalAggregateOpts::default()),
        NullCount.bind(EmptyOptions),
        UncompressedSizeInBytes.bind(EmptyOptions),
    ]
}

pub(crate) fn aggregate_stats_to_df(
    stats: &[VortexPrecision<Scalar>],
) -> VortexResult<ColumnStatistics> {
    if stats.len() != BYTE_SIZE_INDEX + 1 {
        return Err(vortex_err!(
            "expected {} aggregate statistics, got {}",
            BYTE_SIZE_INDEX + 1,
            stats.len()
        ));
    }

    Ok(ColumnStatistics {
        null_count: scalar_u64_to_df_usize(&stats[NULL_COUNT_INDEX])?,
        min_value: scalar_to_df(&stats[MIN_INDEX])?,
        max_value: scalar_to_df(&stats[MAX_INDEX])?,
        sum_value: scalar_to_df(&stats[SUM_INDEX])?,
        distinct_count: Precision::Absent,
        byte_size: scalar_u64_to_df_usize(&stats[BYTE_SIZE_INDEX])?,
    })
}

/// Convert a stats set for an array with the given dtype.
#[allow(dead_code)]
pub(crate) fn stats_set_to_df(
    stats_set: &StatsSet,
    dtype: &DType,
) -> VortexResult<ColumnStatistics> {
    // Update the total size in bytes.
    let column_size = stats_set.get_as::<usize>(Stat::UncompressedSizeInBytes, &PType::U64.into());

    // TODO(connor): There's a lot that can go wrong here, should probably handle this
    // more gracefully...
    // Find the min statistic.
    let min = stats_set.get(Stat::Min).and_then(|stat_val| {
        Scalar::try_new(
            Stat::Min
                .dtype(dtype)
                .vortex_expect("must have a valid dtype"),
            Some(stat_val),
        )
        .vortex_expect("`Stat::Min` somehow had an incompatible `DType`")
        .try_to_df()
        .ok()
    });

    // Find the max statistic.
    let max = stats_set.get(Stat::Max).and_then(|stat_val| {
        Scalar::try_new(
            Stat::Max
                .dtype(dtype)
                .vortex_expect("must have a valid dtype"),
            Some(stat_val),
        )
        .vortex_expect("`Stat::Max` somehow had an incompatible `DType`")
        .try_to_df()
        .ok()
    });

    // Find the sum statistic
    let sum = stats_set.get(Stat::Sum).and_then(|stat_val| {
        Scalar::try_new(
            Stat::Sum
                .dtype(dtype)
                .vortex_expect("must have a valid dtype"),
            Some(stat_val),
        )
        .vortex_expect("`Stat::Sum` somehow had an incompatible `DType`")
        .try_to_df()
        .ok()
    });

    let null_count = stats_set.get_as::<usize>(Stat::NullCount, &PType::U64.into());

    Ok(ColumnStatistics {
        null_count: null_count.to_df(),
        min_value: min.to_df(),
        max_value: max.to_df(),
        sum_value: sum.to_df(),
        distinct_count: is_constant_to_distinct_count(
            stats_set.get_as::<bool>(Stat::IsConstant, &DType::Bool(Nullability::NonNullable)),
        ),
        byte_size: column_size.to_df(),
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

fn scalar_to_df(stat: &VortexPrecision<Scalar>) -> VortexResult<Precision<ScalarValue>> {
    match stat {
        VortexPrecision::Exact(scalar) => Ok(Precision::Exact(scalar.try_to_df()?)),
        VortexPrecision::Inexact(scalar) => Ok(Precision::Inexact(scalar.try_to_df()?)),
        VortexPrecision::Absent => Ok(Precision::Absent),
    }
}

fn scalar_u64_to_df_usize(stat: &VortexPrecision<Scalar>) -> VortexResult<Precision<usize>> {
    match stat {
        VortexPrecision::Exact(scalar) => Ok(Precision::Exact(scalar_u64_to_usize(scalar)?)),
        VortexPrecision::Inexact(scalar) => Ok(Precision::Inexact(scalar_u64_to_usize(scalar)?)),
        VortexPrecision::Absent => Ok(Precision::Absent),
    }
}

fn scalar_u64_to_usize(scalar: &Scalar) -> VortexResult<usize> {
    let Some(value) = scalar.as_primitive().typed_value::<u64>() else {
        return Err(vortex_err!("expected u64 statistic scalar, got {}", scalar));
    };
    Ok(usize::try_from(value).unwrap_or(usize::MAX))
}

#[cfg(test)]
mod tests {
    use vortex::expr::stats::Precision as VortexPrecision;

    use super::*;

    #[test]
    fn is_constant_false_does_not_imply_one_distinct_value() -> VortexResult<()> {
        let false_constant = StatsSet::of(Stat::IsConstant, VortexPrecision::exact(false));
        let false_stats = stats_set_to_df(&false_constant, &DType::Bool(Nullability::NonNullable))?;

        assert_eq!(false_stats.distinct_count, Precision::Absent);

        let true_constant = StatsSet::of(Stat::IsConstant, VortexPrecision::exact(true));
        let true_stats = stats_set_to_df(&true_constant, &DType::Bool(Nullability::NonNullable))?;

        assert_eq!(true_stats.distinct_count, Precision::Exact(1));

        Ok(())
    }
}
