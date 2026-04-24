//! Runtime view of a zoned layout's auxiliary per-zone statistics table.

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::aggregate_fn::fns::sum::sum;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::struct_::StructArrayExt;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::expr::Expression;
use vortex_array::expr::stats::Precision;
use vortex_array::expr::stats::Stat;
use vortex_array::stats::StatsSet;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_mask::Mask;
use vortex_session::VortexSession;

pub use crate::layouts::zoned::builder::StatsAccumulator;
use crate::layouts::zoned::schema::stats_table_dtype;

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
        let expected_dtype = stats_table_dtype(&column_dtype, &stats);
        if &expected_dtype != array.dtype() {
            vortex_bail!("Array dtype does not match expected zone map dtype: {expected_dtype}");
        }

        // SAFETY: We checked that the array matches the expected stats-table schema.
        Ok(unsafe { Self::new_unchecked(array, stats) })
    }

    /// Creates [`ZoneMap`] without validating return array against expected stats.
    ///
    /// # Safety
    ///
    /// Assumes that the input struct array has the correct statistics as fields. Or in other words,
    /// the [`DType`] of the input array is equal to the result of `stats_table_dtype`.
    pub unsafe fn new_unchecked(array: StructArray, stats: Arc<[Stat]>) -> Self {
        Self { array, stats }
    }

    /// Returns the [`DType`] of the statistics table given a set of statistics and column [`DType`].
    ///
    /// This remains as a compatibility wrapper around the zoned schema helper.
    #[deprecated(note = "use `stats_table_dtype` from `crate::layouts::zoned::schema` instead")]
    pub fn dtype_for_stats_table(column_dtype: &DType, present_stats: &[Stat]) -> DType {
        stats_table_dtype(column_dtype, present_stats)
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
                    if let Some(s) = array.statistics().compute_stat(stat, ctx)?
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

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use vortex_array::IntoArray;
    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::StructArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::dtype::FieldPath;
    use vortex_array::dtype::FieldPathSet;
    use vortex_array::dtype::PType;
    use vortex_array::expr::gt;
    use vortex_array::expr::gt_eq;
    use vortex_array::expr::lit;
    use vortex_array::expr::lt;
    use vortex_array::expr::pruning::checked_pruning_expr;
    use vortex_array::expr::root;
    use vortex_array::expr::stats::Stat;
    use vortex_array::validity::Validity;
    use vortex_buffer::buffer;

    use crate::layouts::zoned::zone_map::ZoneMap;
    use crate::test::SESSION;

    #[test]
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
