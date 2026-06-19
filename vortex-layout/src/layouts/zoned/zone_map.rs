//! Runtime view of a zoned layout's auxiliary per-zone statistics table.

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::aggregate_fn::AggregateFnRef;
use vortex_array::aggregate_fn::AggregateFnSatisfaction;
use vortex_array::aggregate_fn::fns::all_nan::AllNan;
use vortex_array::aggregate_fn::fns::all_non_nan::AllNonNan;
use vortex_array::aggregate_fn::fns::all_non_null::AllNonNull;
use vortex_array::aggregate_fn::fns::all_null::AllNull;
use vortex_array::aggregate_fn::fns::bounded_max::BOUNDED_MAX_BOUND;
use vortex_array::aggregate_fn::fns::bounded_max::BoundedMax;
use vortex_array::aggregate_fn::fns::nan_count::NanCount;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::struct_::StructArrayExt;
use vortex_array::dtype::DType;
use vortex_array::expr::Expression;
use vortex_array::expr::eq;
use vortex_array::expr::get_item;
use vortex_array::expr::is_root;
use vortex_array::expr::lit;
use vortex_array::expr::root;
use vortex_array::expr::stats::Stat;
use vortex_array::expr::traversal::NodeExt;
use vortex_array::expr::traversal::Transformed;
use vortex_array::scalar_fn::EmptyOptions;
use vortex_array::scalar_fn::ScalarFnVTableExt;
use vortex_array::scalar_fn::fns::stat::StatFn;
use vortex_array::scalar_fn::internal::row_count::RowCount;
use vortex_array::scalar_fn::internal::row_count::contains_row_count;
use vortex_array::scalar_fn::internal::row_count::substitute_row_count;
use vortex_array::stats::bind::StatBinder;
use vortex_array::stats::bind::bind_stats;
use vortex_array::validity::Validity;
use vortex_buffer::buffer;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_mask::Mask;
use vortex_runend::RunEnd;
use vortex_session::VortexSession;

use crate::layouts::zoned::schema::aggregate_stats_table_dtype;
use crate::layouts::zoned::schema::legacy_stats_table_dtype;

/// A zone map containing statistics for a column.
/// Each row of the zone map corresponds to a chunk of the column.
///
/// Note that it's possible for the zone map to have no statistics.
#[derive(Clone)]
pub struct ZoneMap {
    // The dtype of the data column this zone map describes.
    column_dtype: DType,
    // The struct array backing the zone map
    array: StructArray,
    // Aggregate functions stored in the zone map, ordered by their stats-table fields.
    aggregate_fns: Arc<[AggregateFnRef]>,
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
        aggregate_fns: Arc<[AggregateFnRef]>,
        zone_len: u64,
        row_count: u64,
    ) -> VortexResult<Self> {
        let expected_dtype = aggregate_stats_table_dtype(&column_dtype, &aggregate_fns);
        if &expected_dtype != array.dtype() {
            vortex_bail!("Array dtype does not match expected zone map dtype: {expected_dtype}");
        }

        // SAFETY: We checked that the array matches the expected stats-table schema.
        Ok(unsafe { Self::new_unchecked(column_dtype, array, aggregate_fns, zone_len, row_count) })
    }

    pub(super) unsafe fn new_unchecked(
        column_dtype: DType,
        array: StructArray,
        aggregate_fns: Arc<[AggregateFnRef]>,
        zone_len: u64,
        row_count: u64,
    ) -> Self {
        Self {
            column_dtype,
            array,
            aggregate_fns,
            zone_len,
            row_count,
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

    /// Apply a pruning predicate to this zone map.
    ///
    /// `predicate` should be a stats rewrite expression such as the result of
    /// [`Expression::falsify`]. The returned mask has one value per zone, where
    /// `true` means the zone cannot contain matching rows and can be skipped.
    ///
    /// If the predicate contains [`row_count`][vortex_array::scalar_fn::internal::row_count]
    /// placeholders, they are replaced after [`ArrayRef::apply`] with per-zone
    /// counts derived from `zone_len` and `row_count`. Uniform zones use a
    /// [`ConstantArray`]; a short final zone uses a run-end encoded array.
    /// `row_count` is a layout property rather than a stored stats field, and the
    /// final zone may be shorter than the nominal zone length, so it is materialized
    /// only after the predicate has been lowered to the zone-map table.
    pub fn prune(&self, predicate: &Expression, session: &VortexSession) -> VortexResult<Mask> {
        let mut ctx = session.create_execution_ctx();
        let num_zones = self.array.len();
        let predicate = self.lower_stats(predicate.clone())?;

        let applied = self.array.clone().into_array().apply(&predicate)?;

        if !contains_row_count(&applied) {
            return applied.null_as_false().execute(&mut ctx);
        }

        let row_count_array = row_count_array(self.zone_len, self.row_count, num_zones)?;
        let substituted = substitute_row_count(applied, &row_count_array)?;
        substituted.null_as_false().execute(&mut ctx)
    }

    fn lower_stats(&self, predicate: Expression) -> VortexResult<Expression> {
        let predicate = self.lower_non_float_nan_stats(predicate)?;
        let binder = ZoneMapStatsBinder { zone_map: self };
        bind_stats(predicate, &binder)?.optimize_recursive(self.array.dtype())
    }

    fn lower_non_float_nan_stats(&self, predicate: Expression) -> VortexResult<Expression> {
        predicate
            .transform_down(|expr| {
                if !expr.is::<StatFn>() {
                    return Ok(Transformed::no(expr));
                }

                let options = expr.as_::<StatFn>();
                let aggregate_fn = options.aggregate_fn();
                let input_dtype = expr.child(0).return_dtype(&self.column_dtype)?;

                if has_nans(&input_dtype) {
                    return Ok(Transformed::no(expr));
                }

                if aggregate_fn.is::<NanCount>() {
                    return Ok(Transformed::yes(lit(0u64)));
                }

                if aggregate_fn.is::<AllNan>() {
                    return Ok(Transformed::yes(lit(false)));
                }

                if aggregate_fn.is::<AllNonNan>() {
                    return Ok(Transformed::yes(lit(true)));
                }

                Ok(Transformed::no(expr))
            })
            .map(Transformed::into_inner)
    }
}

struct ZoneMapStatsBinder<'a> {
    zone_map: &'a ZoneMap,
}

impl StatBinder for ZoneMapStatsBinder<'_> {
    fn scope(&self) -> &DType {
        &self.zone_map.column_dtype
    }

    fn bind_aggregate(
        &self,
        input: &Expression,
        aggregate_fn: &AggregateFnRef,
        _stat_dtype: &DType,
    ) -> VortexResult<Option<Expression>> {
        if !is_root(input) {
            return Ok(None);
        }

        if let Some(stat_expr) = self.zone_map.aggregate_field_expr(aggregate_fn) {
            return Ok(Some(stat_expr));
        }

        if aggregate_fn.is::<AllNull>() {
            return Ok(self
                .zone_map
                .stat_field_expr(Stat::NullCount)
                .map(|null_count| eq(null_count, row_count_expr())));
        }

        if aggregate_fn.is::<AllNonNull>() {
            return Ok(self
                .zone_map
                .stat_field_expr(Stat::NullCount)
                .map(|null_count| eq(null_count, lit(0u64))));
        }

        if aggregate_fn.is::<AllNan>() {
            return Ok(self
                .zone_map
                .stat_field_expr(Stat::NaNCount)
                .map(|nan_count| eq(nan_count, row_count_expr())));
        }

        if aggregate_fn.is::<AllNonNan>() {
            return Ok(self
                .zone_map
                .stat_field_expr(Stat::NaNCount)
                .map(|nan_count| eq(nan_count, lit(0u64))));
        }

        if let Some(stat) = Stat::from_aggregate_fn(aggregate_fn) {
            return Ok(self.zone_map.stat_field_expr(stat));
        }

        Ok(None)
    }
}

impl ZoneMap {
    fn aggregate_field_expr(&self, requested: &AggregateFnRef) -> Option<Expression> {
        let field_name = requested.to_string();
        if self.array.unmasked_field_by_name_opt(&field_name).is_some() {
            return Some(aggregate_result_expr(
                requested,
                get_item(field_name, root()),
            ));
        }

        let mut approximate = None;
        for stored in self.aggregate_fns.iter() {
            let field_name = stored.to_string();
            if self.array.unmasked_field_by_name_opt(&field_name).is_none() {
                continue;
            }

            match stored.can_satisfy(requested) {
                AggregateFnSatisfaction::Exact => {
                    return Some(aggregate_result_expr(stored, get_item(field_name, root())));
                }
                AggregateFnSatisfaction::Approximate => {
                    approximate = Some(aggregate_result_expr(stored, get_item(field_name, root())));
                }
                AggregateFnSatisfaction::No => {}
            }
        }

        approximate
    }

    fn stat_field_expr(&self, stat: Stat) -> Option<Expression> {
        if let Some(aggregate_fn) = stat.aggregate_fn()
            && let Some(expr) = self.aggregate_field_expr(&aggregate_fn)
        {
            return Some(expr);
        }

        self.legacy_stat_field_expr(stat)
    }

    fn legacy_stat_field_expr(&self, stat: Stat) -> Option<Expression> {
        if self.array.unmasked_field_by_name_opt(stat.name()).is_some() {
            return Some(get_item(stat.name(), root()));
        }

        None
    }
}

fn aggregate_result_expr(stored: &AggregateFnRef, state_expr: Expression) -> Expression {
    if stored.is::<BoundedMax>() {
        get_item(BOUNDED_MAX_BOUND, state_expr)
    } else {
        state_expr
    }
}

fn row_count_expr() -> Expression {
    RowCount.new_expr(EmptyOptions, [])
}

fn has_nans(dtype: &DType) -> bool {
    matches!(dtype, DType::Primitive(ptype, _) if ptype.is_float())
}

/// Build per-zone row counts for a zone map.
///
/// `zone_len` is the nominal zone size; only the final zone may be shorter. The
/// result is a [`ConstantArray`] for uniform zone sizes, otherwise a two-run
/// run-end encoded array whose trailing run carries the final zone length.
fn row_count_array(zone_len: u64, row_count: u64, num_zones: usize) -> VortexResult<ArrayRef> {
    if num_zones == 0 {
        return Ok(ConstantArray::new(0u64, 0).into_array());
    }

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
    use std::num::NonZeroUsize;
    use std::sync::Arc;

    use vortex_array::IntoArray;
    use vortex_array::aggregate_fn::AggregateFnVTableExt;
    use vortex_array::aggregate_fn::EmptyOptions;
    use vortex_array::aggregate_fn::fns::all_non_null::AllNonNull;
    use vortex_array::aggregate_fn::fns::all_null::AllNull;
    use vortex_array::aggregate_fn::fns::bounded_max::BOUNDED_MAX_BOUND;
    use vortex_array::aggregate_fn::fns::bounded_max::BOUNDED_MAX_UNKNOWN;
    use vortex_array::aggregate_fn::fns::bounded_max::BoundedMax;
    use vortex_array::aggregate_fn::fns::bounded_max::BoundedMaxOptions;
    use vortex_array::aggregate_fn::fns::bounded_min::BoundedMin;
    use vortex_array::aggregate_fn::fns::bounded_min::BoundedMinOptions;
    use vortex_array::aggregate_fn::fns::max::Max;
    use vortex_array::aggregate_fn::fns::min::Min;
    use vortex_array::aggregate_fn::fns::nan_count::NanCount;
    use vortex_array::aggregate_fn::fns::null_count::NullCount;
    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::StructArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::DecimalDType;
    use vortex_array::dtype::FieldNames;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::expr::Expression;
    use vortex_array::expr::cast;
    use vortex_array::expr::gt;
    use vortex_array::expr::gt_eq;
    use vortex_array::expr::is_not_null;
    use vortex_array::expr::is_null;
    use vortex_array::expr::lit;
    use vortex_array::expr::lt;
    use vortex_array::expr::not_eq;
    use vortex_array::expr::root;
    use vortex_array::expr::stats::Stat;
    use vortex_array::stats::all_nan;
    use vortex_array::stats::all_non_nan;
    use vortex_array::stats::all_non_null;
    use vortex_array::stats::all_null;
    use vortex_array::validity::Validity;
    use vortex_buffer::buffer;

    use crate::layouts::zoned::zone_map::ZoneMap;
    use crate::test::SESSION;

    fn falsify(expr: &Expression, dtype: DType) -> Expression {
        expr.falsify(&dtype, &SESSION).unwrap().unwrap()
    }

    fn default_bounded_stat_max_bytes() -> NonZeroUsize {
        // SAFETY: 64 is non-zero.
        unsafe { NonZeroUsize::new_unchecked(64) }
    }

    #[test]
    fn test_zone_map_prunes() {
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
        let max = Max.bind(EmptyOptions);
        let min = Min.bind(EmptyOptions);
        let zone_map = ZoneMap::try_new(
            PType::I32.into(),
            StructArray::from_fields(&[
                (
                    max.to_string(),
                    PrimitiveArray::new(buffer![5i32, 6i32, 7i32], Validity::AllValid).into_array(),
                ),
                (
                    min.to_string(),
                    PrimitiveArray::new(buffer![1i32, 2i32, 3i32], Validity::AllValid).into_array(),
                ),
            ])
            .unwrap(),
            Arc::new([max, min]),
            3,
            10,
        )
        .unwrap();

        // A >= 6
        // => A.max < 6
        let expr = gt_eq(root(), lit(6i32));
        let pruning_expr = falsify(&expr, PType::I32.into());
        let mask = zone_map.prune(&pruning_expr, &SESSION).unwrap();
        assert_arrays_eq!(
            mask.into_array(),
            BoolArray::from_iter([true, false, false])
        );

        // A > 5
        // => A.max <= 5
        let expr = gt(root(), lit(5i32));
        let pruning_expr = falsify(&expr, PType::I32.into());
        let mask = zone_map.prune(&pruning_expr, &SESSION).unwrap();
        assert_arrays_eq!(
            mask.into_array(),
            BoolArray::from_iter([true, false, false])
        );

        // A < 2
        // => A.min >= 2
        let expr = lt(root(), lit(2i32));
        let pruning_expr = falsify(&expr, PType::I32.into());
        let mask = zone_map.prune(&pruning_expr, &SESSION).unwrap();
        assert_arrays_eq!(mask.into_array(), BoolArray::from_iter([false, true, true]));
    }

    #[test]
    fn bounded_display_names_satisfy_min_max_rewrites() {
        let bounded_max = BoundedMax.bind(BoundedMaxOptions {
            max_bytes: default_bounded_stat_max_bytes(),
        });
        let bounded_min = BoundedMin.bind(BoundedMinOptions {
            max_bytes: default_bounded_stat_max_bytes(),
        });
        let zone_map = ZoneMap::try_new(
            PType::I32.into(),
            StructArray::from_fields(&[
                (
                    bounded_max.to_string(),
                    StructArray::try_new(
                        [BOUNDED_MAX_BOUND, BOUNDED_MAX_UNKNOWN].into(),
                        vec![
                            PrimitiveArray::new(buffer![5i32, 6i32, 7i32], Validity::AllValid)
                                .into_array(),
                            BoolArray::from_iter([false, false, false]).into_array(),
                        ],
                        3,
                        Validity::AllValid,
                    )
                    .unwrap()
                    .into_array(),
                ),
                (
                    bounded_min.to_string(),
                    PrimitiveArray::new(buffer![1i32, 2i32, 3i32], Validity::AllValid).into_array(),
                ),
            ])
            .unwrap(),
            Arc::new([bounded_max, bounded_min]),
            3,
            10,
        )
        .unwrap();

        let expr = gt(root(), lit(5i32));
        let pruning_expr = falsify(&expr, PType::I32.into());
        let mask = zone_map.prune(&pruning_expr, &SESSION).unwrap();
        assert_arrays_eq!(
            mask.into_array(),
            BoolArray::from_iter([true, false, false])
        );

        let expr = lt(root(), lit(2i32));
        let pruning_expr = falsify(&expr, PType::I32.into());
        let mask = zone_map.prune(&pruning_expr, &SESSION).unwrap();
        assert_arrays_eq!(mask.into_array(), BoolArray::from_iter([false, true, true]));
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
            BoolArray::from_iter([false, false, true])
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
            BoolArray::from_iter([true, false, false])
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

        let mask = zone_map.prune(&all_null(root()), &SESSION).unwrap();
        assert_arrays_eq!(mask.into_array(), BoolArray::from_iter([false, true, true]));
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

        let mask = zone_map.prune(&all_non_null(root()), &SESSION).unwrap();
        assert_arrays_eq!(
            mask.into_array(),
            BoolArray::from_iter([true, false, false])
        );
    }

    #[test]
    fn all_null_stat_fn_lowers_to_null_count_field() {
        let null_count = NullCount.bind(EmptyOptions);
        let zone_map = ZoneMap::try_new(
            PType::U64.into(),
            StructArray::from_fields(&[(
                null_count.to_string(),
                PrimitiveArray::new(buffer![4u64, 0, 2], Validity::AllValid).into_array(),
            )])
            .unwrap(),
            Arc::new([null_count]),
            4,
            10,
        )
        .unwrap();

        let mask = zone_map.prune(&all_null(root()), &SESSION).unwrap();
        assert_arrays_eq!(mask.into_array(), BoolArray::from_iter([true, false, true]));

        let mask = zone_map.prune(&all_non_null(root()), &SESSION).unwrap();
        assert_arrays_eq!(
            mask.into_array(),
            BoolArray::from_iter([false, true, false])
        );
    }

    #[test]
    fn all_nan_stat_fn_lowers_to_nan_count_field() {
        let nan_count = NanCount.bind(EmptyOptions);
        let zone_map = ZoneMap::try_new(
            PType::F32.into(),
            StructArray::from_fields(&[(
                nan_count.to_string(),
                PrimitiveArray::new(buffer![4u64, 0, 2], Validity::AllValid).into_array(),
            )])
            .unwrap(),
            Arc::new([nan_count]),
            4,
            10,
        )
        .unwrap();

        let mask = zone_map.prune(&all_nan(root()), &SESSION).unwrap();
        assert_arrays_eq!(mask.into_array(), BoolArray::from_iter([true, false, true]));

        let mask = zone_map.prune(&all_non_nan(root()), &SESSION).unwrap();
        assert_arrays_eq!(
            mask.into_array(),
            BoolArray::from_iter([false, true, false])
        );
    }

    #[test]
    fn non_float_nan_stat_fns_lower_to_constants() {
        let zone_map = ZoneMap::try_new(
            PType::I32.into(),
            StructArray::try_new(FieldNames::empty(), vec![], 2, Validity::NonNullable).unwrap(),
            Arc::new([]),
            4,
            8,
        )
        .unwrap();

        let mask = zone_map.prune(&all_nan(root()), &SESSION).unwrap();
        assert_arrays_eq!(mask.into_array(), BoolArray::from_iter([false, false]));

        let mask = zone_map.prune(&all_non_nan(root()), &SESSION).unwrap();
        assert_arrays_eq!(mask.into_array(), BoolArray::from_iter([true, true]));
    }

    #[test]
    fn unavailable_stat_fn_lowers_to_unknown_mask() {
        let zone_map = ZoneMap::try_new(
            PType::U64.into(),
            StructArray::try_new(FieldNames::empty(), vec![], 3, Validity::NonNullable).unwrap(),
            Arc::new([]),
            4,
            10,
        )
        .unwrap();

        let mask = zone_map.prune(&all_non_null(root()), &SESSION).unwrap();
        assert_arrays_eq!(
            mask.into_array(),
            BoolArray::from_iter([false, false, false])
        );

        let expr = gt(root(), lit(5u64));
        let pruning_expr = falsify(&expr, PType::U64.into());
        let mask = zone_map.prune(&pruning_expr, &SESSION).unwrap();
        assert_arrays_eq!(
            mask.into_array(),
            BoolArray::from_iter([false, false, false])
        );
    }

    #[test]
    fn float_min_max_stat_fn_requires_nan_count() {
        let max = Max.bind(EmptyOptions);
        let zone_map = ZoneMap::try_new(
            PType::F32.into(),
            StructArray::from_fields(&[(
                max.to_string(),
                PrimitiveArray::new(buffer![5.0f32, 6.0, 7.0], Validity::AllValid).into_array(),
            )])
            .unwrap(),
            Arc::new([max.clone()]),
            4,
            12,
        )
        .unwrap();

        let expr = gt(root(), lit(5.0f32));
        let pruning_expr = falsify(&expr, PType::F32.into());
        let mask = zone_map.prune(&pruning_expr, &SESSION).unwrap();
        assert_arrays_eq!(
            mask.into_array(),
            BoolArray::from_iter([false, false, false])
        );

        let nan_count = NanCount.bind(EmptyOptions);
        let zone_map = ZoneMap::try_new(
            PType::F32.into(),
            StructArray::from_fields(&[
                (
                    max.to_string(),
                    PrimitiveArray::new(buffer![5.0f32, 6.0, 7.0], Validity::AllValid).into_array(),
                ),
                (
                    nan_count.to_string(),
                    PrimitiveArray::new(buffer![0u64, 0, 0], Validity::AllValid).into_array(),
                ),
            ])
            .unwrap(),
            Arc::new([max, nan_count]),
            4,
            12,
        )
        .unwrap();

        let mask = zone_map.prune(&pruning_expr, &SESSION).unwrap();
        assert_arrays_eq!(
            mask.into_array(),
            BoolArray::from_iter([true, false, false])
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
        assert_arrays_eq!(mask.into_array(), BoolArray::from_iter([false, true]));
    }

    #[test]
    fn fixed_size_list_min_max_stat_fn_lowers_to_unknown_mask() {
        // Regression test for issue #8189: Min/Max is defined for FixedSizeList<T>
        // when T is orderable. If the zone map does not carry the requested stat,
        // lowering should produce an unknown typed null rather than rejecting the dtype.
        let elem_dtype = Arc::new(DType::Decimal(
            DecimalDType::new(10, 2),
            Nullability::Nullable,
        ));
        let column_dtype = DType::FixedSizeList(elem_dtype, 1, Nullability::Nullable);

        let zone_map = ZoneMap::try_new(
            column_dtype,
            StructArray::try_new(FieldNames::empty(), vec![], 3, Validity::NonNullable).unwrap(),
            Arc::new([]),
            4,
            10,
        )
        .unwrap();

        let max_fn = Stat::Max
            .aggregate_fn()
            .expect("max should have an aggregate function");
        let predicate = is_null(vortex_array::stats::stat(root(), max_fn));

        // Missing StatFn lowers to a nullable null literal, so `is_null(...)` is true for every zone.
        let mask = zone_map.prune(&predicate, &SESSION).unwrap();
        assert_arrays_eq!(mask.into_array(), BoolArray::from_iter([true, true, true]));
    }

    #[test]
    fn unsupported_aggregate_input_dtype_errors() {
        let zone_map = ZoneMap::try_new(
            DType::Null,
            StructArray::try_new(FieldNames::empty(), vec![], 3, Validity::NonNullable).unwrap(),
            Arc::new([]),
            4,
            10,
        )
        .unwrap();

        let max_fn = Stat::Max
            .aggregate_fn()
            .expect("max should have an aggregate function");
        let predicate = is_null(vortex_array::stats::stat(root(), max_fn));
        let error = zone_map.prune(&predicate, &SESSION).unwrap_err();

        assert!(
            error
                .to_string()
                .contains("Aggregate function vortex.max() does not support input dtype null"),
            "{error}"
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
            BoolArray::from_iter([false, true, false])
        );
    }

    #[test]
    fn all_null_stat_fn_lowers_to_aggregate_field() {
        let all_null_agg = AllNull.bind(EmptyOptions);
        let zone_map = ZoneMap::try_new(
            PType::U64.into(),
            StructArray::from_fields(&[(
                all_null_agg.to_string(),
                BoolArray::from_iter([Some(false), Some(true), Some(true)]).into_array(),
            )])
            .unwrap(),
            Arc::new([all_null_agg]),
            4,
            10,
        )
        .unwrap();

        let mask = zone_map.prune(&all_null(root()), &SESSION).unwrap();
        assert_arrays_eq!(mask.into_array(), BoolArray::from_iter([false, true, true]));
    }

    #[test]
    fn all_non_null_stat_fn_lowers_to_aggregate_field() {
        let all_non_null_agg = AllNonNull.bind(EmptyOptions);
        let zone_map = ZoneMap::try_new(
            PType::U64.into(),
            StructArray::from_fields(&[(
                all_non_null_agg.to_string(),
                BoolArray::from_iter([Some(true), Some(false), Some(false)]).into_array(),
            )])
            .unwrap(),
            Arc::new([all_non_null_agg]),
            4,
            10,
        )
        .unwrap();

        let mask = zone_map.prune(&all_non_null(root()), &SESSION).unwrap();
        assert_arrays_eq!(
            mask.into_array(),
            BoolArray::from_iter([true, false, false])
        );
    }
}
