//! Runtime view of a zoned layout's auxiliary per-zone statistics table.

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::dtype::DType;
use vortex_array::expr::Expression;
use vortex_array::expr::stats::Stat;
use vortex_array::Executor;
use vortex_array::scalar_fn::internal::row_count::contains_row_count;
use vortex_array::scalar_fn::internal::row_count::substitute_row_count;
use vortex_array::validity::Validity;
use vortex_buffer::buffer;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_mask::Mask;
use vortex_runend::RunEnd;
use vortex_session::VortexSession;

use crate::layouts::zoned::schema::stats_table_dtype;

/// A zone map containing statistics for a column.
/// Each row of the zone map corresponds to a chunk of the column.
///
/// Note that it's possible for the zone map to have no statistics.
#[derive(Clone)]
pub struct ZoneMap {
    // The struct array backing the zone map
    array: StructArray,
    // The length of each zone in the zone map.
    zone_len: u64,
    // Number of rows that the zone map covers
    row_count: u64,
}

impl ZoneMap {
    /// Create [`ZoneMap`] of given column_dtype from given array. Validates that the array matches expected
    /// structure for given list of stats.
    pub fn try_new(
        column_dtype: DType,
        array: StructArray,
        stats: Arc<[Stat]>,
        zone_len: u64,
        row_count: u64,
    ) -> VortexResult<Self> {
        let expected_dtype = stats_table_dtype(&column_dtype, &stats);
        if &expected_dtype != array.dtype() {
            vortex_bail!("Array dtype does not match expected zone map dtype: {expected_dtype}");
        }

        // SAFETY: We checked that the array matches the expected stats-table schema.
        Ok(unsafe { Self::new_unchecked(array, zone_len, row_count) })
    }

    /// Creates [`ZoneMap`] without validating return array against expected stats.
    ///
    /// # Safety
    ///
    /// Assumes that the input struct array has the correct statistics as fields. Or in other words,
    pub unsafe fn new_unchecked(array: StructArray, zone_len: u64, row_count: u64) -> Self {
        Self {
            array,
            zone_len,
            row_count,
        }
    }

    /// Returns the [`DType`] of the statistics table given a set of statistics and column [`DType`].
    ///
    /// This remains as a compatibility wrapper around the zoned schema helper.
    #[deprecated(note = "use `stats_table_dtype` from `crate::layouts::zoned::schema` instead")]
    pub fn dtype_for_stats_table(column_dtype: &DType, present_stats: &[Stat]) -> DType {
        stats_table_dtype(column_dtype, present_stats)
    }

    /// Apply a pruning predicate to this zone map.
    ///
    /// `predicate` should be the result of converting a filter with
    /// [`checked_pruning_expr`]. The returned mask has one value per zone, where
    /// `true` means the zone cannot contain matching rows and can be skipped.
    ///
    /// If the predicate contains [`row_count`][vortex_array::scalar_fn::internal::row_count]
    /// placeholders, they are replaced after [`ArrayRef::apply`] with per-zone
    /// counts derived from `zone_len` and `row_count`. Uniform zones use a
    /// [`ConstantArray`]; a short final zone uses a run-end encoded array.
    ///
    /// [`checked_pruning_expr`]: vortex_array::expr::pruning::checked_pruning_expr
    pub fn prune(&self, predicate: &Expression, session: &VortexSession) -> VortexResult<Mask> {
        let mut ctx = session.create_execution_ctx();
        let num_zones = self.array.len();

        let applied = self.array.clone().into_array().apply(predicate)?;

        if num_zones == 0 || !contains_row_count(&applied) {
            return Ok(applied.null_as_false().execute::<Mask>(&mut ctx)?);
        }

        let row_count_array = row_count_array(self.zone_len, self.row_count, num_zones)?;
        let substituted = substitute_row_count(applied, &row_count_array)?;
        Ok(substituted
            .null_as_false().execute::<Mask>(&mut ctx)?)
    }
}

/// Build per-zone row counts for a zone map.
///
/// `zone_len` is the nominal zone size; only the final zone may be shorter. The
/// result is a [`ConstantArray`] for uniform zone sizes, otherwise a two-run
/// run-end encoded array whose trailing run carries the final zone length.
fn row_count_array(zone_len: u64, row_count: u64, num_zones: usize) -> VortexResult<ArrayRef> {
    let last_zone_len = row_count - zone_len.saturating_mul((num_zones as u64) - 1);
    if num_zones == 1 || last_zone_len == zone_len {
        return Ok(ConstantArray::new(last_zone_len, num_zones).into_array());
    }

    let ends = unsafe {
        PrimitiveArray::new_unchecked(
            buffer![num_zones as u64 - 1, num_zones as u64],
            Validity::NonNullable,
        )
    }
    .into_array();
    let values = unsafe {
        PrimitiveArray::new_unchecked(buffer![zone_len, last_zone_len], Validity::NonNullable)
    }
    .into_array();

    // SAFETY: `ends` are strictly increasing, terminate at `num_zones`, and align one-to-one
    // with the non-null run values.
    Ok(unsafe { RunEnd::new_unchecked(ends, values, 0, num_zones) }.into_array())
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
    use vortex_array::expr::is_not_null;
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
            3,
            10,
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

    #[test]
    fn row_count_prunes_short_trailing_zone() {
        let zone_map = ZoneMap::try_new(
            PType::U64.into(),
            StructArray::from_fields(&[(
                "null_count",
                PrimitiveArray::new(buffer![0u64, 0, 2], Validity::AllValid).into_array(),
            )])
            .unwrap(),
            Arc::new([Stat::NullCount]),
            4,
            10,
        )
        .unwrap();

        let available_stats =
            FieldPathSet::from_iter([FieldPath::from_iter([Stat::NullCount.name().into()])]);
        let expr = is_not_null(root());
        let (pruning_expr, _) = checked_pruning_expr(&expr, &available_stats).unwrap();

        let mask = zone_map.prune(&pruning_expr, &SESSION).unwrap();
        assert_arrays_eq!(
            mask.into_array(),
            BoolArray::from_iter([false, false, true])
        );
    }

    #[test]
    fn row_count_prunes_all_null_uniform_zones() {
        let zone_map = ZoneMap::try_new(
            PType::U64.into(),
            StructArray::from_fields(&[(
                "null_count",
                PrimitiveArray::new(buffer![0u64, 4, 0], Validity::AllValid).into_array(),
            )])
            .unwrap(),
            Arc::new([Stat::NullCount]),
            4,
            12,
        )
        .unwrap();

        let available_stats =
            FieldPathSet::from_iter([FieldPath::from_iter([Stat::NullCount.name().into()])]);
        let expr = is_not_null(root());
        let (pruning_expr, _) = checked_pruning_expr(&expr, &available_stats).unwrap();

        // All three zones have length 4 (total rows = 12).
        let mask = zone_map.prune(&pruning_expr, &SESSION).unwrap();
        assert_arrays_eq!(
            mask.into_array(),
            BoolArray::from_iter([false, true, false])
        );
    }
}
