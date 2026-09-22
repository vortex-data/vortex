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

    let null_count = stats_set.get_as::<usize>(Stat::NullCount, &PType::U64.into());

    Ok(ColumnStatistics {
        null_count: null_count.to_df(),
        min_value: scalar_stat_to_df(stats_set, Stat::Min, dtype).to_df(),
        max_value: scalar_stat_to_df(stats_set, Stat::Max, dtype).to_df(),
        sum_value: scalar_stat_to_df(stats_set, Stat::Sum, dtype).to_df(),
        distinct_count: is_constant_to_distinct_count(
            stats_set.get_as::<bool>(Stat::IsConstant, &DType::Bool(Nullability::NonNullable)),
        ),
        byte_size: column_size.to_df(),
    })
}

/// Read one scalar-valued statistic and convert it to DataFusion, or `Absent` if it does not apply
/// to `dtype` or cannot be represented.
fn scalar_stat_to_df(
    stats_set: &StatsSet,
    stat: Stat,
    dtype: &DType,
) -> VortexPrecision<ScalarValue> {
    stats_set.get(stat).and_then(|stat_value| {
        Scalar::try_new(stat.dtype(dtype)?, Some(stat_value))
            .ok()?
            .try_to_df()
            .ok()
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
    use datafusion_common::ScalarValue as DFScalarValue;
    use rstest::rstest;
    use vortex::expr::stats::Precision as VortexPrecision;
    use vortex::scalar::ScalarValue;

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

    /// A statistic the column dtype cannot carry comes back `Absent`.
    #[rstest]
    #[case::min_of_null_column(Stat::Min, DType::Null, ScalarValue::from(1i32))]
    #[case::max_of_null_column(Stat::Max, DType::Null, ScalarValue::from(1i32))]
    #[case::sum_of_utf8_column(
        Stat::Sum,
        DType::Utf8(Nullability::NonNullable),
        ScalarValue::from(1i32)
    )]
    #[case::value_disagrees_with_dtype(
        Stat::Min,
        DType::Bool(Nullability::NonNullable),
        ScalarValue::from("not a bool")
    )]
    fn unconvertible_statistics_are_absent(
        #[case] stat: Stat,
        #[case] dtype: DType,
        #[case] value: ScalarValue,
    ) -> VortexResult<()> {
        let stats = stats_set_to_df(&StatsSet::of(stat, VortexPrecision::exact(value)), &dtype)?;

        for reported in [stats.min_value, stats.max_value, stats.sum_value] {
            assert_eq!(reported, Precision::Absent);
        }
        Ok(())
    }

    /// A statistic that does convert keeps its precision.
    #[rstest]
    #[case::exact(
        VortexPrecision::exact(ScalarValue::from(7i32)),
        Precision::Exact(DFScalarValue::Int32(Some(7)))
    )]
    #[case::inexact(
        VortexPrecision::inexact(ScalarValue::from(7i32)),
        Precision::Inexact(DFScalarValue::Int32(Some(7)))
    )]
    fn convertible_statistics_keep_their_precision(
        #[case] min: VortexPrecision<ScalarValue>,
        #[case] expected: Precision<DFScalarValue>,
    ) -> VortexResult<()> {
        let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        let stats = stats_set_to_df(&StatsSet::of(Stat::Min, min), &dtype)?;

        assert_eq!(stats.min_value, expected);
        Ok(())
    }
}
