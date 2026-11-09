// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use rstest::rstest;
use vortex::dtype::DType;
use vortex::dtype::FieldPath;
use vortex::dtype::Nullability;
use vortex::dtype::PType;
use vortex::error::VortexExpect;
use vortex::error::VortexResult;
use vortex::expr::Expression;
use vortex::expr::and;
use vortex::expr::cast;
use vortex::expr::col;
use vortex::expr::eq;
use vortex::expr::lit;
use vortex::expr::not_eq;
use vortex::expr::or;
use vortex::expr::stats::Precision;
use vortex::expr::stats::Stat;
use vortex::file::FileStatistics;
use vortex::layout::layouts::file_stats::AggregateStat;
use vortex::layout::layouts::file_stats::AggregateStats;
use vortex::scalar::Scalar;

use super::binary_flag_matches;
use super::group_parts;
use super::part_ranges;

#[rstest]
#[case::whole(0..30, &[0, 10, 20, 30], vec![0..10, 10..20, 20..30])]
#[case::clipped(5..25, &[0, 10, 20, 30], vec![5..10, 10..20, 20..25])]
#[case::inside_one_split(12..18, &[0, 10, 20, 30], vec![12..18])]
#[case::empty(5..5, &[0, 10], vec![])]
fn parts_follow_splits(
    #[case] rows: Range<u64>,
    #[case] splits: &[u64],
    #[case] expected: Vec<Range<u64>>,
) {
    assert_eq!(part_ranges(rows, splits), expected);
}

#[rstest]
#[case::empty(vec![], 8, vec![])]
#[case::short_adjacent(vec![0..3, 3..6, 6..9, 9..12], 8, vec![0..9, 9..12])]
#[case::exact_boundary(vec![0..4, 4..8, 8..12], 8, vec![0..8, 8..12])]
#[case::large_parts(vec![0..16, 16..32], 8, vec![0..16, 16..32])]
#[case::clipped(vec![5..10, 10..20, 20..25], 8, vec![5..20, 20..25])]
#[case::gaps(vec![0..3, 5..7, 7..12], 8, vec![0..3, 5..12])]
fn groups_small_adjacent_parts(
    #[case] parts: Vec<Range<u64>>,
    #[case] minimum: u64,
    #[case] expected: Vec<Range<u64>>,
) {
    assert_eq!(group_parts(parts, minimum), expected);
}

fn flag_stats(
    min: i16,
    max: i16,
    sum: Precision<Scalar>,
    null_count: Precision<Scalar>,
) -> FileStatistics {
    let aggregates = [
        (Stat::Min, Precision::exact(Scalar::from(min))),
        (Stat::Max, Precision::exact(Scalar::from(max))),
        (Stat::Sum, sum),
        (Stat::NullCount, null_count),
    ]
    .into_iter()
    .map(|(stat, value)| {
        AggregateStat::from_value(
            stat.aggregate_fn()
                .vortex_expect("numeric stat has an aggregate"),
            value,
        )
    })
    .collect();
    FileStatistics::new(
        [AggregateStats::new(aggregates)].into(),
        [DType::Primitive(PType::I16, Nullability::Nullable)].into(),
        [FieldPath::root().push("flag")].into(),
    )
}

#[rstest]
#[case::zero(eq(col("flag"), lit(0i16)), Some(70))]
#[case::one(eq(col("flag"), lit(1i16)), Some(20))]
#[case::not_zero(not_eq(col("flag"), lit(0i16)), Some(20))]
#[case::not_one(not_eq(col("flag"), lit(1i16)), Some(70))]
#[case::swapped(eq(lit(1i16), col("flag")), Some(20))]
#[case::widened(
    eq(
        cast(col("flag"), DType::Primitive(PType::I64, Nullability::Nullable)),
        lit(1i64),
    ),
    Some(20)
)]
#[case::null_literal(
    eq(
        col("flag"),
        lit(Scalar::null(DType::Primitive(PType::I16, Nullability::Nullable))),
    ),
    None
)]
#[case::unknown(eq(col("unknown"), lit(1i16)), None)]
#[case::outside_domain(eq(col("flag"), lit(2i16)), None)]
#[case::known_conjunct(
    and(eq(col("flag"), lit(1i16)), eq(col("unknown"), lit(1i16))),
    Some(20)
)]
#[case::conjunction_bound(and(eq(col("flag"), lit(0i16)), eq(col("flag"), lit(1i16))), Some(20))]
#[case::disjunction(or(eq(col("flag"), lit(1i16)), eq(col("unknown"), lit(1i16))), None)]
fn flag_counts_bound_static_filter_matches(
    #[case] filter: Expression,
    #[case] expected: Option<u64>,
) -> VortexResult<()> {
    let dtype = DType::struct_(
        [
            ("flag", DType::Primitive(PType::I16, Nullability::Nullable)),
            (
                "unknown",
                DType::Primitive(PType::I16, Nullability::Nullable),
            ),
        ],
        Nullability::NonNullable,
    );
    // Of 100 rows, 10 are null, 20 are one, and 70 are zero.
    let stats = flag_stats(0, 1, Precision::exact(20i64), Precision::exact(10u64));
    assert_eq!(
        binary_flag_matches(&filter.bind(&dtype)?, &stats, 100),
        expected
    );
    Ok(())
}

#[rstest]
#[case::missing_sum(0, 1, Precision::Absent, Precision::exact(0u64))]
#[case::approximate_sum(0, 1, Precision::inexact(20i64), Precision::exact(0u64))]
#[case::missing_nulls(0, 1, Precision::exact(20i64), Precision::Absent)]
#[case::negative(-1, 1, Precision::exact(20i64), Precision::exact(0u64))]
#[case::not_binary(0, 2, Precision::exact(20i64), Precision::exact(0u64))]
#[case::invalid_sum(0, 1, Precision::exact(101i64), Precision::exact(0u64))]
#[case::invalid_nulls(0, 1, Precision::exact(20i64), Precision::exact(101u64))]
fn unusable_flag_statistics_remain_unknown(
    #[case] min: i16,
    #[case] max: i16,
    #[case] sum: Precision<Scalar>,
    #[case] null_count: Precision<Scalar>,
) -> VortexResult<()> {
    let dtype = DType::struct_(
        [("flag", DType::Primitive(PType::I16, Nullability::Nullable))],
        Nullability::NonNullable,
    );
    let filter = eq(col("flag"), lit(1i16)).bind(&dtype)?;
    assert_eq!(
        binary_flag_matches(&filter, &flag_stats(min, max, sum, null_count), 100),
        None,
    );
    Ok(())
}
