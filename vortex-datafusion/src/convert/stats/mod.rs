// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use arrow_schema::DataType;
use datafusion_common::ColumnStatistics;
use datafusion_common::ScalarValue;
use datafusion_common::stats::Precision;
use vortex::aggregate_fn::fns::approx_distinct::ApproxDistinct;
use vortex::array::stats::StatsSet;
use vortex::dtype::DType;
use vortex::dtype::Nullability;
use vortex::dtype::PType;
use vortex::error::VortexResult;
use vortex::expr::stats::Precision as VortexPrecision;
use vortex::expr::stats::Stat;
use vortex::extension::datetime::AnyTemporal;
use vortex::layout::layouts::file_stats::AggregateStats;
use vortex::scalar::Scalar;

use crate::PrecisionExt;
use crate::convert::TryToDataFusion;

/// Convert a stats set for an array with the given dtype.
///
/// `byte_size` is the uncompressed size recorded by Vortex, which is an estimate of the size
/// DataFusion's join selection, `with_fetch` and filter estimates compare and scale.
pub(crate) fn stats_set_to_df(
    stats_set: &StatsSet,
    dtype: &DType,
) -> VortexResult<ColumnStatistics> {
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
        byte_size: uncompressed_size_to_df(stats_set),
    })
}

/// The size of a column of `num_rows` values of `data_type`, as DataFusion estimates it for
/// Parquet: the value width times the row count for fixed-width types, otherwise the size of the
/// encoded column before general-purpose compression.
///
/// For a variable-width column that is `encoded_size`, the bytes of the column's segments in the
/// file, falling back to the uncompressed size recorded in `stats_set`. Parquet reports the
/// uncompressed size of its column chunks, which are dictionary encoded, and Vortex's encoded
/// size is its counterpart; its uncompressed size instead counts the canonical layout (16-byte
/// string views), several times larger for a dictionary of short strings.
///
/// Both are reported inexact. DataFusion's join selection uses `total_byte_size` instead of the
/// row count whenever it is known: a build side is collected into one partition only below
/// `hash_join_single_partition_threshold` (1 MiB in DataFusion 55, 4 MiB later), and
/// `hash_join_single_partition_threshold_rows` (128 Ki rows) is no longer consulted. Reporting
/// the canonical size made a projected `date_dim` exceed the threshold that Parquet's stays
/// under, and partitioned its joins with `store_sales` (TPC-DS Q2, Q54 and Q59).
pub(crate) fn byte_size_to_df(
    stats_set: &StatsSet,
    data_type: &DataType,
    num_rows: usize,
    encoded_size: Option<u64>,
) -> Precision<usize> {
    match data_type.primitive_width() {
        Some(width) => num_rows
            .checked_mul(width)
            .map_or(Precision::Absent, Precision::Inexact),
        None => match encoded_size.and_then(|size| usize::try_from(size).ok()) {
            Some(size) => Precision::Inexact(size),
            None => uncompressed_size_to_df(stats_set),
        },
    }
}

/// The total of the column byte sizes, or `Absent` if any column's size is unknown.
pub(crate) fn sum_byte_sizes<'a>(
    columns: impl IntoIterator<Item = &'a ColumnStatistics>,
) -> Precision<usize> {
    columns
        .into_iter()
        .fold(Precision::Exact(0), |total, column| {
            total.add(&column.byte_size)
        })
}

fn uncompressed_size_to_df(stats_set: &StatsSet) -> Precision<usize> {
    stats_set
        .get_as::<usize>(Stat::UncompressedSizeInBytes, &PType::U64.into())
        .to_df()
        .to_inexact()
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

/// The distinct count of a column with `non_null_rows` non-null rows from its
/// [`ApproxDistinct`] sketch, if the file records one.
///
/// The sketch estimates, so the count is inexact, except that a sketch that saw nothing proves a
/// column holds no non-null value. The estimate is capped at the number of non-null rows, which
/// it can exceed by its error on columns of unique values.
pub(crate) fn approx_distinct_count(
    aggregates: &AggregateStats,
    non_null_rows: Precision<usize>,
) -> Precision<usize> {
    let Some(estimate) = aggregates
        .iter()
        .filter(|stat| stat.aggregate_fn().is::<ApproxDistinct>())
        .find_map(|stat| stat.value().clone().as_exact())
        .and_then(|estimate| u64::try_from(&estimate).ok())
        .and_then(|estimate| usize::try_from(estimate).ok())
    else {
        return Precision::Absent;
    };
    if estimate == 0 {
        return Precision::Exact(0);
    }
    match non_null_rows {
        Precision::Exact(rows) | Precision::Inexact(rows) => {
            Precision::Inexact(estimate.min(rows).max(1))
        }
        Precision::Absent => Precision::Inexact(estimate),
    }
}

#[cfg(test)]
mod tests;
