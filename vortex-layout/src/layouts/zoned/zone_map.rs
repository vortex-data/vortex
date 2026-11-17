// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Native compatibility helpers for zone statistics tables.

use std::ops::Deref;
use std::sync::Arc;

use vortex_array::aggregate_fn::AggregateFnRef;
use vortex_array::arrays::StructArray;
use vortex_array::dtype::DType;
use vortex_array::expr::stats::Stat;
use vortex_error::VortexResult;
#[cfg(test)]
use vortex_error::vortex_bail;

use crate::layouts::zoned::schema::legacy_stats_table_dtype;

/// A zone map with native legacy schema compatibility.
#[derive(Clone)]
pub struct ZoneMap {
    #[cfg(test)]
    column_dtype: DType,
    inner: vortex_scan::execution::zone_map::ZoneMap,
}

impl Deref for ZoneMap {
    type Target = vortex_scan::execution::zone_map::ZoneMap;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl ZoneMap {
    /// Validates an aggregate statistics table before constructing its zone map.
    pub fn try_new(
        column_dtype: DType,
        array: StructArray,
        aggregate_fns: Arc<[AggregateFnRef]>,
        zone_len: u64,
        row_count: u64,
    ) -> VortexResult<Self> {
        Ok(Self {
            #[cfg(test)]
            column_dtype: column_dtype.clone(),
            inner: vortex_scan::execution::zone_map::ZoneMap::try_new(
                column_dtype,
                array,
                aggregate_fns,
                zone_len,
                row_count,
            )?,
        })
    }

    pub(super) unsafe fn new_unchecked(
        column_dtype: DType,
        array: StructArray,
        aggregate_fns: Arc<[AggregateFnRef]>,
        zone_len: u64,
        row_count: u64,
    ) -> Self {
        Self {
            #[cfg(test)]
            column_dtype: column_dtype.clone(),
            // SAFETY: The caller validates the native statistics schema.
            inner: unsafe {
                vortex_scan::execution::zone_map::ZoneMap::new_unchecked(
                    column_dtype,
                    array,
                    aggregate_fns,
                    zone_len,
                    row_count,
                )
            },
        }
    }

    /// Returns the [`DType`] of the statistics table given a set of statistics and column [`DType`].
    ///
    /// This remains as a compatibility wrapper around the zoned schema helper.
    #[deprecated(note = "use aggregate-function zoned stats instead")]
    pub fn dtype_for_stats_table(column_dtype: &DType, present_stats: &[Stat]) -> DType {
        legacy_stats_table_dtype(column_dtype, present_stats)
    }

    #[cfg(test)]
    fn try_new_legacy(
        column_dtype: DType,
        array: StructArray,
        stats: Arc<[Stat]>,
        zone_len: u64,
        row_count: u64,
    ) -> VortexResult<Self> {
        let expected_dtype = legacy_stats_table_dtype(&column_dtype, &stats);
        if &expected_dtype != array.dtype() {
            vortex_bail!("Array dtype does not match expected zone map dtype: {expected_dtype}");
        }

        // SAFETY: We checked that the array matches the expected legacy stats-table schema.
        Ok(unsafe { Self::new_unchecked(column_dtype, array, Arc::new([]), zone_len, row_count) })
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::StructArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::expr::BoundExpression;
    use vortex_array::expr::Expression;
    use vortex_array::expr::cast;
    use vortex_array::expr::is_not_null;
    use vortex_array::expr::is_null;
    use vortex_array::expr::lit;
    use vortex_array::expr::not_eq;
    use vortex_array::expr::root;
    use vortex_array::expr::stats::Stat;
    use vortex_array::validity::Validity;
    use vortex_buffer::Buffer;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_expr::stats::all_non_null;
    use vortex_expr::stats::all_null;
    use vortex_expr::stats::rewrite::BoundExpressionPruneExt;
    use vortex_mask::Mask;

    use crate::layouts::zoned::zone_map::ZoneMap;
    use crate::test::SESSION;

    fn falsify(expr: &Expression, dtype: DType) -> BoundExpression {
        expr.bind(&dtype)
            .unwrap()
            .falsify(&SESSION)
            .unwrap()
            .unwrap()
    }

    fn prune(zone_map: &ZoneMap, predicate: &Expression) -> VortexResult<Mask> {
        zone_map.prune(&predicate.bind(&zone_map.column_dtype)?, &SESSION)
    }

    #[test]
    fn row_count_prunes_short_trailing_zone() {
        let zone_map = ZoneMap::try_new_legacy(
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

        let expr = is_not_null(root());
        let pruning_expr = falsify(&expr, PType::U64.into());

        let mask = zone_map.prune(&pruning_expr, &SESSION).unwrap();
        assert_arrays_eq!(
            mask.into_array(),
            BoolArray::from_iter([false, false, true]),
            &mut SESSION.create_execution_ctx()
        );
    }

    /// `zone_len`, `row_count` and the zone-map length all come from the file and nothing ties them
    /// together, so a file can claim more zone coverage than the layout has rows. Pruning must
    /// report that rather than underflowing the per-zone row-count arithmetic.
    #[rstest]
    // 4 zones of 4 rows claims 12 rows of leading coverage for a 10-row layout. (3 zones of 4 is
    // legal — 4 + 4 + 2 — and is what `row_count_prunes_short_trailing_zone` covers.)
    #[case::zones_overrun_row_count(4, 4, 10)]
    // The leading zones alone already exceed the row count.
    #[case::leading_zones_overrun(8, 3, 10)]
    // A zero row count with a non-empty zone map.
    #[case::no_rows_but_zones_present(4, 2, 0)]
    fn row_count_rejects_zones_overrunning_the_layout(
        #[case] zone_len: u64,
        #[case] num_zones: usize,
        #[case] row_count: u64,
    ) {
        let null_counts = PrimitiveArray::new(
            Buffer::<u64>::from_iter(std::iter::repeat_n(0u64, num_zones)),
            Validity::AllValid,
        )
        .into_array();
        let zone_map = ZoneMap::try_new_legacy(
            PType::U64.into(),
            StructArray::from_fields(&[("null_count", null_counts)]).unwrap(),
            Arc::new([Stat::NullCount]),
            zone_len,
            row_count,
        )
        .unwrap();

        let pruning_expr = falsify(&is_not_null(root()), PType::U64.into());
        let message = zone_map
            .prune(&pruning_expr, &SESSION)
            .err()
            .map(|err| err.to_string())
            .unwrap_or_default();
        assert!(
            message.contains("more than the"),
            "zone_len {zone_len} x {num_zones} zones over {row_count} rows should have been \
             rejected, got: {message}"
        );
    }

    #[test]
    fn row_count_substitution_handles_empty_zone_map() {
        let zone_map = ZoneMap::try_new_legacy(
            PType::U64.into(),
            StructArray::from_fields(&[(
                "null_count",
                PrimitiveArray::new::<u64>(buffer![], Validity::AllValid).into_array(),
            )])
            .unwrap(),
            Arc::new([Stat::NullCount]),
            4,
            0,
        )
        .unwrap();

        let expr = is_not_null(root());
        let pruning_expr = falsify(&expr, PType::U64.into());

        let mask = zone_map.prune(&pruning_expr, &SESSION).unwrap();
        assert_eq!(mask.len(), 0);
    }

    #[test]
    fn is_null_falsification_uses_null_count() {
        let zone_map = ZoneMap::try_new_legacy(
            PType::U64.into(),
            StructArray::from_fields(&[(
                "null_count",
                PrimitiveArray::new(buffer![0u64, 4, 2], Validity::AllValid).into_array(),
            )])
            .unwrap(),
            Arc::new([Stat::NullCount]),
            4,
            10,
        )
        .unwrap();

        let expr = is_null(root());
        let pruning_expr = falsify(&expr, PType::U64.into());

        let mask = zone_map.prune(&pruning_expr, &SESSION).unwrap();
        assert_arrays_eq!(
            mask.into_array(),
            BoolArray::from_iter([true, false, false]),
            &mut SESSION.create_execution_ctx()
        );
    }

    #[test]
    fn all_null_stat_fn_lowers_to_null_count_and_row_count() {
        let zone_map = ZoneMap::try_new_legacy(
            PType::U64.into(),
            StructArray::from_fields(&[(
                "null_count",
                PrimitiveArray::new(buffer![0u64, 4, 2], Validity::AllValid).into_array(),
            )])
            .unwrap(),
            Arc::new([Stat::NullCount]),
            4,
            10,
        )
        .unwrap();

        let mask = prune(&zone_map, &all_null(root())).unwrap();
        assert_arrays_eq!(
            mask.into_array(),
            BoolArray::from_iter([false, true, true]),
            &mut SESSION.create_execution_ctx()
        );
    }

    #[test]
    fn all_non_null_stat_fn_lowers_to_null_count() {
        let zone_map = ZoneMap::try_new_legacy(
            PType::U64.into(),
            StructArray::from_fields(&[(
                "null_count",
                PrimitiveArray::new(buffer![0u64, 4, 2], Validity::AllValid).into_array(),
            )])
            .unwrap(),
            Arc::new([Stat::NullCount]),
            4,
            10,
        )
        .unwrap();

        let mask = prune(&zone_map, &all_non_null(root())).unwrap();
        assert_arrays_eq!(
            mask.into_array(),
            BoolArray::from_iter([true, false, false]),
            &mut SESSION.create_execution_ctx()
        );
    }

    #[test]
    fn float_cast_min_max_stat_fn_uses_source_nan_count() {
        let zone_map = ZoneMap::try_new_legacy(
            PType::F32.into(),
            StructArray::from_fields(&[
                (
                    "max",
                    PrimitiveArray::new(buffer![5.0f32, 5.0], Validity::AllValid).into_array(),
                ),
                (
                    "max_is_truncated",
                    BoolArray::from_iter([false, false]).into_array(),
                ),
                (
                    "min",
                    PrimitiveArray::new(buffer![5.0f32, 5.0], Validity::AllValid).into_array(),
                ),
                (
                    "min_is_truncated",
                    BoolArray::from_iter([false, false]).into_array(),
                ),
                (
                    "nan_count",
                    PrimitiveArray::new(buffer![1u64, 0], Validity::AllValid).into_array(),
                ),
            ])
            .unwrap(),
            Arc::new([Stat::Max, Stat::Min, Stat::NaNCount]),
            4,
            8,
        )
        .unwrap();

        let cast_dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        let expr = not_eq(cast(root(), cast_dtype), lit(5i32));
        let pruning_expr = falsify(&expr, PType::F32.into());

        let mask = zone_map.prune(&pruning_expr, &SESSION).unwrap();
        assert_arrays_eq!(
            mask.into_array(),
            BoolArray::from_iter([false, true]),
            &mut SESSION.create_execution_ctx()
        );
    }

    #[test]
    fn row_count_prunes_all_null_uniform_zones() {
        let zone_map = ZoneMap::try_new_legacy(
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

        let expr = is_not_null(root());
        let pruning_expr = falsify(&expr, PType::U64.into());

        // All three zones have length 4 (total rows = 12).
        let mask = zone_map.prune(&pruning_expr, &SESSION).unwrap();
        assert_arrays_eq!(
            mask.into_array(),
            BoolArray::from_iter([false, true, false]),
            &mut SESSION.create_execution_ctx()
        );
    }

    #[test]
    fn raw_stat_fixture_rejects_wrong_schema() -> VortexResult<()> {
        let array = StructArray::from_fields(&[(
            "null_count",
            PrimitiveArray::new(buffer![0u32], Validity::AllValid).into_array(),
        )])?;
        let error =
            ZoneMap::try_new_legacy(PType::U64.into(), array, Arc::new([Stat::NullCount]), 4, 4)
                .err()
                .expect("null counts must be nullable u64");
        assert!(
            error
                .to_string()
                .contains("Array dtype does not match expected zone map dtype")
        );
        Ok(())
    }
}
