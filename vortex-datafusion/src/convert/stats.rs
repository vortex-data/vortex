// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use datafusion_common::ColumnStatistics;
use datafusion_common::stats::Precision;
use vortex::array::stats::StatsSet;
use vortex::dtype::DType;
use vortex::dtype::Nullability;
use vortex::dtype::PType;
use vortex::error::VortexExpect;
use vortex::error::VortexResult;
use vortex::expr::stats::Precision as VortexPrecision;
use vortex::expr::stats::Stat;
use vortex::scalar::Scalar;

use crate::PrecisionExt;
use crate::convert::TryToDataFusion;

/// Convert a stats set for an array with the given dtype.
pub(crate) fn stats_set_to_df(
    stats_set: &StatsSet,
    dtype: &DType,
) -> VortexResult<ColumnStatistics> {
    // Update the total size in bytes.
    let column_size = stats_set.get_as::<usize>(Stat::UncompressedSizeInBytes, &PType::U64.into());

    // TODO(connor): There's a lot that can go wrong here, should probably handle this
    // more gracefully...
    // Find the min statistic.
    let min = stats_set.get(Stat::Min).and_then(|pstat_val| {
        pstat_val
            .map(|stat_val| {
                Scalar::try_new(
                    Stat::Min
                        .dtype(dtype)
                        .vortex_expect("must have a valid dtype"),
                    Some(stat_val),
                )
                .vortex_expect("`Stat::Min` somehow had an incompatible `DType`")
                .try_to_df()
                .ok()
            })
            .transpose()
    });

    // Find the max statistic.
    let max = stats_set.get(Stat::Max).and_then(|pstat_val| {
        pstat_val
            .map(|stat_val| {
                Scalar::try_new(
                    Stat::Max
                        .dtype(dtype)
                        .vortex_expect("must have a valid dtype"),
                    Some(stat_val),
                )
                .vortex_expect("`Stat::Max` somehow had an incompatible `DType`")
                .try_to_df()
                .ok()
            })
            .transpose()
    });

    // Find the sum statistic
    let sum = stats_set.get(Stat::Sum).and_then(|pstat_val| {
        pstat_val
            .map(|stat_val| {
                Scalar::try_new(
                    Stat::Sum
                        .dtype(dtype)
                        .vortex_expect("must have a valid dtype"),
                    Some(stat_val),
                )
                .vortex_expect("`Stat::Sum` somehow had an incompatible `DType`")
                .try_to_df()
                .ok()
            })
            .transpose()
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
    is_constant: Option<VortexPrecision<bool>>,
) -> Precision<usize> {
    match is_constant.and_then(VortexPrecision::as_exact) {
        Some(true) => Precision::Exact(1),
        Some(false) | None => Precision::Absent,
    }
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
