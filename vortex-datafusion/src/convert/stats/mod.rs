// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use datafusion_common::ColumnStatistics;
use datafusion_common::ScalarValue;
use datafusion_common::stats::Precision;
use vortex::array::stats::StatsSet;
use vortex::dtype::DType;
use vortex::dtype::Nullability;
use vortex::dtype::PType;
use vortex::error::VortexResult;
use vortex::expr::stats::Precision as VortexPrecision;
use vortex::expr::stats::Stat;
use vortex::extension::datetime::AnyTemporal;
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

    let null_count = stats_set
        .get_as::<usize>(Stat::NullCount, &PType::U64.into())
        .to_df();
    let min_value = scalar_stat_to_df(stats_set, Stat::Min, dtype).to_df();
    let max_value = scalar_stat_to_df(stats_set, Stat::Max, dtype).to_df();
    let distinct_count =
        constant_distinct_count(stats_set, dtype, &null_count, &min_value, &max_value);

    Ok(ColumnStatistics {
        null_count,
        min_value,
        max_value,
        sum_value: scalar_stat_to_df(stats_set, Stat::Sum, dtype).to_df(),
        distinct_count,
        byte_size: column_size.to_df(),
    })
}

/// Read one NaN-skipping statistic (min, max or sum) and convert it to DataFusion, or `Absent` if it
/// does not apply to `dtype`, cannot be represented, or the column may hold NaN.
fn scalar_stat_to_df(
    stats_set: &StatsSet,
    stat: Stat,
    dtype: &DType,
) -> VortexPrecision<ScalarValue> {
    if !is_nan_free(stats_set, dtype) {
        return VortexPrecision::Absent;
    }

    stats_set.get(stat).and_then(|stat_value| {
        Scalar::try_new(stat.dtype(dtype)?, Some(stat_value))
            .ok()?
            .try_to_df()
            .ok()
    })
}

/// Whether `stats_set` proves the column holds no NaN.
pub(crate) fn is_nan_free(stats_set: &StatsSet, dtype: &DType) -> bool {
    !dtype.is_float()
        || stats_set.get_as::<u64>(Stat::NaNCount, &PType::U64.into()) == VortexPrecision::Exact(0)
}

/// Infer a singleton from exact bounds, excluding nulls and, for floats, NaNs.
pub(crate) fn constant_distinct_count(
    stats_set: &StatsSet,
    dtype: &DType,
    null_count: &Precision<usize>,
    min: &Precision<ScalarValue>,
    max: &Precision<ScalarValue>,
) -> Precision<usize> {
    match stats_set
        .get_as::<bool>(Stat::IsConstant, &DType::Bool(Nullability::NonNullable))
        .as_exact()
    {
        Some(true) => return Precision::Exact(1),
        Some(false) => return Precision::Absent,
        None => {}
    }

    // TODO: Replace this inference with IsConstant once file statistics aggregate and persist it.
    let supports_bounds = match dtype {
        DType::Bool(_)
        | DType::Primitive(..)
        | DType::Utf8(_)
        | DType::Binary(_)
        | DType::Decimal(..) => true,
        DType::Extension(ext) => ext.metadata_opt::<AnyTemporal>().is_some(),
        _ => false,
    };
    if !supports_bounds || *null_count != Precision::Exact(0) {
        return Precision::Absent;
    }

    // Float extrema skip NaNs, so equal extrema alone cannot prove a constant.
    if !is_nan_free(stats_set, dtype) {
        return Precision::Absent;
    }

    // Schema casts can collapse distinct bounds. Require equal source bounds as well.
    match (stats_set.get(Stat::Min), stats_set.get(Stat::Max)) {
        (VortexPrecision::Exact(min), VortexPrecision::Exact(max)) if min == max => {}
        _ => return Precision::Absent,
    }
    match (min, max) {
        (Precision::Exact(min), Precision::Exact(max)) if !min.is_null() && min == max => {
            Precision::Exact(1)
        }
        _ => Precision::Absent,
    }
}

#[cfg(test)]
mod tests;
