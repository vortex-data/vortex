// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use itertools::Itertools;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::aggregate_fn::fns::sum::sum;
use vortex_array::arrays::StructArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::dtype::StructFields;
use vortex_array::expr::Expression;
use vortex_array::expr::stats::Precision;
use vortex_array::expr::stats::Stat;
use vortex_array::expr::stats::StatsProvider;
use vortex_array::stats::StatsSet;
use vortex_array::validity::Validity;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_mask::Mask;
use vortex_session::VortexSession;

use crate::layouts::zoned::builder::MAX_IS_TRUNCATED;
use crate::layouts::zoned::builder::MIN_IS_TRUNCATED;
use crate::layouts::zoned::builder::StatsArrayBuilder;
use crate::layouts::zoned::builder::stats_builder_with_capacity;

/// A zone map containing statistics for a column.
/// Each row of the zone map corresponds to a chunk of the column.
///
/// Note that it's possible for the zone map to have no statistics.
#[derive(Clone)]
pub struct ZoneMap {
    // The struct array backing the zone map
    array: StructArray,
    // The statistics that are included in the table.
    stats: Arc<[Stat]>,
}

impl ZoneMap {
    /// Create [`ZoneMap`] of given column_dtype from given array. Validates that the array matches expected
    /// structure for given list of stats.
    pub fn try_new(
        column_dtype: DType,
        array: StructArray,
        stats: Arc<[Stat]>,
    ) -> VortexResult<Self> {
        let expected_dtype = Self::dtype_for_stats_table(&column_dtype, &stats);
        if &expected_dtype != array.dtype() {
            vortex_bail!("Array dtype does not match expected zone map dtype: {expected_dtype}");
        }

        // SAFETY: We checked that the
        Ok(unsafe { Self::new_unchecked(array, stats) })
    }

    /// Creates [`ZoneMap`] without validating return array against expected stats.
    ///
    /// # Safety
    ///
    /// Assumes that the input struct array has the correct statistics as fields. Or in other words,
    /// the [`DType`] of the input array is equal to the result of [`Self::dtype_for_stats_table`].
    pub unsafe fn new_unchecked(array: StructArray, stats: Arc<[Stat]>) -> Self {
        Self { array, stats }
    }

    /// Returns the [`DType`] of the statistics table given a set of statistics and column [`DType`].
    pub fn dtype_for_stats_table(column_dtype: &DType, present_stats: &[Stat]) -> DType {
        assert!(present_stats.is_sorted(), "Stats must be sorted");
        DType::Struct(
            StructFields::from_iter(
                present_stats
                    .iter()
                    .filter_map(|stat| {
                        stat.dtype(column_dtype)
                            .or_else(|| {
                                // Backward compat: older files may have stored stats (e.g. Sum)
                                // for extension types by resolving through the storage dtype.
                                if let DType::Extension(ext) = column_dtype {
                                    stat.dtype(ext.storage_dtype())
                                } else {
                                    None
                                }
                            })
                            .map(|dtype| (stat, dtype.as_nullable()))
                    })
                    .flat_map(|(s, dt)| match s {
                        Stat::Max => vec![
                            (s.name(), dt),
                            (MAX_IS_TRUNCATED, DType::Bool(Nullability::NonNullable)),
                        ],
                        Stat::Min => vec![
                            (s.name(), dt),
                            (MIN_IS_TRUNCATED, DType::Bool(Nullability::NonNullable)),
                        ],
                        _ => vec![(s.name(), dt)],
                    }),
            ),
            Nullability::NonNullable,
        )
    }

    /// Returns the underlying [`StructArray`] backing the zone map
    pub fn array(&self) -> &StructArray {
        &self.array
    }

    /// Returns the list of stats included in the zone map.
    pub fn present_stats(&self) -> &Arc<[Stat]> {
        &self.stats
    }

    /// Returns an aggregated stats set for the table.
    pub fn to_stats_set(&self, stats: &[Stat], ctx: &mut ExecutionCtx) -> VortexResult<StatsSet> {
        let mut stats_set = StatsSet::default();
        for &stat in stats {
            let Some(array) = self.get_stat(stat)? else {
                continue;
            };

            // Different stats need different aggregations
            match stat {
                // For stats that are associative, we can just compute them over the stat column
                Stat::Min | Stat::Max | Stat::Sum => {
                    if let Some(s) = array.statistics().compute_stat(stat)?
                        && let Some(v) = s.into_value()
                    {
                        stats_set.set(stat, Precision::exact(v))
                    }
                }
                // These stats sum up
                Stat::NullCount | Stat::NaNCount | Stat::UncompressedSizeInBytes => {
                    if let Some(sum_value) = sum(&array, ctx)?
                        .cast(&DType::Primitive(PType::U64, Nullability::Nullable))?
                        .into_value()
                    {
                        stats_set.set(stat, Precision::exact(sum_value));
                    }
                }
                // We could implement these aggregations in the future, but for now they're unused
                Stat::IsConstant | Stat::IsSorted | Stat::IsStrictSorted => {}
            }
        }
        Ok(stats_set)
    }

    /// Returns the array for a given stat.
    pub fn get_stat(&self, stat: Stat) -> VortexResult<Option<ArrayRef>> {
        Ok(self.array.unmasked_field_by_name_opt(stat.name()).cloned())
    }

    /// Apply a pruning predicate against the ZoneMap, yielding a mask indicating which zones can
    /// be pruned.
    ///
    /// The expression provided should be the result of converting an existing `VortexExpr` via
    /// [`checked_pruning_expr`][vortex_array::expr::pruning::checked_pruning_expr] into a prunable
    /// expression that can be evaluated on a zone map.
    ///
    /// All zones where the predicate evaluates to `true` can be skipped entirely.
    pub fn prune(&self, predicate: &Expression, session: &VortexSession) -> VortexResult<Mask> {
        let mut ctx = session.create_execution_ctx();
        self.array
            .clone()
            .into_array()
            .apply(predicate)?
            .execute::<Mask>(&mut ctx)
    }
}

// TODO(ngates): we should make it such that the zone map stores a mirror of the DType
//  underneath each stats column. For example, `min: i32` for an `i32` array.
//  Or `min: {a: i32, b: i32}` for a struct array of type `{a: i32, b: i32}`.
//  See: <https://github.com/vortex-data/vortex/issues/1835>
/// Accumulates statistics for a column.
pub struct StatsAccumulator {
    builders: Vec<Box<dyn StatsArrayBuilder>>,
    length: usize,
}

impl StatsAccumulator {
    pub fn new(dtype: &DType, stats: &[Stat], max_variable_length_statistics_size: usize) -> Self {
        let builders = stats
            .iter()
            .filter_map(|&s| {
                s.dtype(dtype).map(|stat_dtype| {
                    stats_builder_with_capacity(
                        s,
                        &stat_dtype.as_nullable(),
                        1024,
                        max_variable_length_statistics_size,
                    )
                })
            })
            .collect::<Vec<_>>();

        Self {
            builders,
            length: 0,
        }
    }

    pub fn push_chunk_without_compute(&mut self, array: &ArrayRef) -> VortexResult<()> {
        for builder in self.builders.iter_mut() {
            if let Some(Precision::Exact(v)) = array.statistics().get(builder.stat()) {
                builder.append_scalar(v.cast(&v.dtype().as_nullable())?)?;
            } else {
                builder.append_null();
            }
        }
        self.length += 1;
        Ok(())
    }

    pub fn push_chunk(&mut self, array: &ArrayRef) -> VortexResult<()> {
        for builder in self.builders.iter_mut() {
            if let Some(v) = array.statistics().compute_stat(builder.stat())? {
                builder.append_scalar(v.cast(&v.dtype().as_nullable())?)?;
            } else {
                builder.append_null();
            }
        }
        self.length += 1;
        Ok(())
    }

    /// Finishes the accumulator into a [`ZoneMap`].
    ///
    /// Returns `None` if none of the requested statistics can be computed, for example they are
    /// not applicable to the column's data type.
    pub fn as_stats_table(&mut self) -> VortexResult<Option<ZoneMap>> {
        let mut names = Vec::new();
        let mut fields = Vec::new();
        let mut stats = Vec::new();

        for builder in self
            .builders
            .iter_mut()
            // We sort the stats so the DType is deterministic based on which stats are present.
            .sorted_unstable_by_key(|b| b.stat())
        {
            let values = builder.finish();

            // We drop any all-null stats columns
            if values.all_invalid()? {
                continue;
            }

            stats.push(builder.stat());
            names.extend(values.names);
            fields.extend(values.arrays);
        }

        if names.is_empty() {
            return Ok(None);
        }

        Ok(Some(ZoneMap {
            array: StructArray::try_new(names.into(), fields, self.length, Validity::NonNullable)
                .vortex_expect("Failed to create zone map"),
            stats: stats.into(),
        }))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::ToCanonical;
    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::StructArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::builders::ArrayBuilder;
    use vortex_array::builders::VarBinViewBuilder;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::FieldPath;
    use vortex_array::dtype::FieldPathSet;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::expr::gt;
    use vortex_array::expr::gt_eq;
    use vortex_array::expr::lit;
    use vortex_array::expr::lt;
    use vortex_array::expr::pruning::checked_pruning_expr;
    use vortex_array::expr::root;
    use vortex_array::expr::stats::Stat;
    use vortex_array::validity::Validity;
    use vortex_buffer::BitBuffer;
    use vortex_buffer::buffer;
    use vortex_error::VortexExpect;

    use crate::layouts::zoned::MAX_IS_TRUNCATED;
    use crate::layouts::zoned::MIN_IS_TRUNCATED;
    use crate::layouts::zoned::zone_map::StatsAccumulator;
    use crate::layouts::zoned::zone_map::ZoneMap;
    use crate::test::SESSION;

    #[rstest]
    #[case(DType::Utf8(Nullability::NonNullable))]
    #[case(DType::Binary(Nullability::NonNullable))]
    fn truncates_accumulated_stats(#[case] dtype: DType) {
        let mut builder = VarBinViewBuilder::with_capacity(dtype.clone(), 2);
        builder.append_value("Value to be truncated");
        builder.append_value("untruncated");
        let mut builder2 = VarBinViewBuilder::with_capacity(dtype, 2);
        builder2.append_value("Another");
        builder2.append_value("wait a minute");
        let mut acc =
            StatsAccumulator::new(builder.dtype(), &[Stat::Max, Stat::Min, Stat::Sum], 12);
        acc.push_chunk(&builder.finish())
            .vortex_expect("push_chunk should succeed for test data");
        acc.push_chunk(&builder2.finish())
            .vortex_expect("push_chunk should succeed for test data");
        let stats_table = acc
            .as_stats_table()
            .unwrap()
            .expect("Must have stats table");
        assert_eq!(
            stats_table.array.names().as_ref(),
            &[
                Stat::Max.name(),
                MAX_IS_TRUNCATED,
                Stat::Min.name(),
                MIN_IS_TRUNCATED,
            ]
        );
        assert_eq!(
            stats_table
                .array
                .unmasked_field(1)
                .to_bool()
                .to_bit_buffer(),
            BitBuffer::from(vec![false, true])
        );
        assert_eq!(
            stats_table
                .array
                .unmasked_field(3)
                .to_bool()
                .to_bit_buffer(),
            BitBuffer::from(vec![true, false])
        );
    }

    #[test]
    fn always_adds_is_truncated_column() {
        let array = buffer![0, 1, 2].into_array();
        let mut acc = StatsAccumulator::new(array.dtype(), &[Stat::Max, Stat::Min, Stat::Sum], 12);
        acc.push_chunk(&array)
            .vortex_expect("push_chunk should succeed for test array");
        let stats_table = acc
            .as_stats_table()
            .unwrap()
            .expect("Must have stats table");
        assert_eq!(
            stats_table.array.names().as_ref(),
            &[
                Stat::Max.name(),
                MAX_IS_TRUNCATED,
                Stat::Min.name(),
                MIN_IS_TRUNCATED,
                Stat::Sum.name(),
            ]
        );
        assert_eq!(
            stats_table
                .array
                .unmasked_field(1)
                .to_bool()
                .to_bit_buffer(),
            BitBuffer::from(vec![false])
        );
        assert_eq!(
            stats_table
                .array
                .unmasked_field(3)
                .to_bool()
                .to_bit_buffer(),
            BitBuffer::from(vec![false])
        );
    }

    #[rstest]
    fn test_zone_map_prunes() {
        // All stats that are known at pruning time.
        let stats = FieldPathSet::from_iter([
            FieldPath::from_iter([Stat::Min.name().into()]),
            FieldPath::from_iter([Stat::Max.name().into()]),
        ]);

        // Construct a zone map with 3 zones:
        //
        // +----------+----------+
        // |  a_min   |  a_max   |
        // +----------+----------+
        // |  1       |  5       |
        // +----------+----------+
        // |  2       |  6       |
        // +----------+----------+
        // |  3       |  7       |
        // +----------+----------+
        let zone_map = ZoneMap::try_new(
            PType::I32.into(),
            StructArray::from_fields(&[
                (
                    "max",
                    PrimitiveArray::new(buffer![5i32, 6i32, 7i32], Validity::AllValid).into_array(),
                ),
                (
                    "max_is_truncated",
                    BoolArray::from_iter([false, false, false]).into_array(),
                ),
                (
                    "min",
                    PrimitiveArray::new(buffer![1i32, 2i32, 3i32], Validity::AllValid).into_array(),
                ),
                (
                    "min_is_truncated",
                    BoolArray::from_iter([false, false, false]).into_array(),
                ),
            ])
            .unwrap(),
            Arc::new([Stat::Max, Stat::Min]),
        )
        .unwrap();

        // A >= 6
        // => A.max < 6
        let expr = gt_eq(root(), lit(6i32));
        let (pruning_expr, _) = checked_pruning_expr(&expr, &stats).unwrap();
        let mask = zone_map.prune(&pruning_expr, &SESSION).unwrap();
        assert_arrays_eq!(
            mask.into_array(),
            BoolArray::from_iter([true, false, false])
        );

        // A > 5
        // => A.max <= 5
        let expr = gt(root(), lit(5i32));
        let (pruning_expr, _) = checked_pruning_expr(&expr, &stats).unwrap();
        let mask = zone_map.prune(&pruning_expr, &SESSION).unwrap();
        assert_arrays_eq!(
            mask.into_array(),
            BoolArray::from_iter([true, false, false])
        );

        // A < 2
        // => A.min >= 2
        let expr = lt(root(), lit(2i32));
        let (pruning_expr, _) = checked_pruning_expr(&expr, &stats).unwrap();
        let mask = zone_map.prune(&pruning_expr, &SESSION).unwrap();
        assert_arrays_eq!(mask.into_array(), BoolArray::from_iter([false, true, true]));
    }
}
