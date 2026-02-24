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
        distinct_count: stats_set
            .get_as::<bool>(Stat::IsConstant, &DType::Bool(Nullability::NonNullable))
            .and_then(|is_constant| is_constant.as_exact().map(|_| Precision::Exact(1)))
            .unwrap_or(Precision::Absent),
        byte_size: column_size.to_df(),
    })
}
