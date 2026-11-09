//! Runtime view of a zoned layout's auxiliary per-zone statistics table.

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use vortex_array::ArrayRef;
use vortex_array::Columnar;
use vortex_array::ExecutionCtx;
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
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::RunEnd;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::struct_::StructArrayExt;
use vortex_array::dtype::DType;
use vortex_array::expr::stats::Stat;
use vortex_array::scalar_fn::EmptyOptions;
use vortex_array::validity::Validity;
use vortex_buffer::buffer;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure_eq;
use vortex_expr::ArrayExprExt;
use vortex_expr::BoundExpression;
use vortex_expr::Expression;
use vortex_expr::ScalarFnExprExt;
use vortex_expr::eq;
use vortex_expr::get_item;
use vortex_expr::lambda::row_count::RowCount;
use vortex_expr::lambda::row_count::contains_row_count;
use vortex_expr::lambda::row_count::substitute_row_count;
use vortex_expr::lit;
use vortex_expr::root;
use vortex_expr::stats::bind::StatBinder;
use vortex_expr::stats::bind::bind_stats;
use vortex_mask::Mask;
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
    /// [`BoundExpressionPruneExt::falsify`]. The returned mask has one value per zone, where
    /// `true` means the zone cannot contain matching rows and can be skipped.
    ///
    /// If the predicate contains [`row_count`][vortex_expr::lambda::row_count]
    /// placeholders, they are replaced after [`ArrayRef::apply_bound`] with per-zone
    /// counts derived from `zone_len` and `row_count`. Uniform zones use a
    /// [`ConstantArray`]; a short final zone uses a run-end encoded array.
    /// `row_count` is a layout property rather than a stored stats field, and the
    /// final zone may be shorter than the nominal zone length, so it is materialized
    /// only after the predicate has been lowered to the zone-map table.
    ///
    /// [`BoundExpressionPruneExt::falsify`]: vortex_expr::stats::rewrite::BoundExpressionPruneExt::falsify
    pub fn prune(
        &self,
        predicate: &BoundExpression,
        session: &VortexSession,
    ) -> VortexResult<Mask> {
        let predicate = self.lower_stats(predicate.clone())?;
        self.prune_lowered(&predicate, session)
    }

    /// Apply a predicate that [`Self::lower_stats`] already lowered to this zone map's
    /// stats table.
    ///
    /// Lowering depends only on the column and stats-table dtypes, so callers that evaluate
    /// the same predicate many times lower it once and call this function for each evaluation.
    pub(super) fn prune_lowered(
        &self,
        predicate: &BoundExpression,
        session: &VortexSession,
    ) -> VortexResult<Mask> {
        let mut ctx = session.create_execution_ctx();
        let num_zones = self.array.len();

        let array = self.array.clone().into_array();
        let applied = array.apply_bound(predicate)?;

        if !contains_row_count(&applied) {
            return null_as_false_mask(applied, &mut ctx);
        }

        let row_count_array = row_count_array(self.zone_len, self.row_count, num_zones)?;
        let substituted = substitute_row_count(applied, &row_count_array)?;
        null_as_false_mask(substituted, &mut ctx)
    }

    /// Rewrite the stats references in `predicate` into expressions over the stats table.
    pub(super) fn lower_stats(&self, predicate: BoundExpression) -> VortexResult<BoundExpression> {
        let binder = ZoneMapStatsBinder { zone_map: self };
        bind_stats(predicate, &binder)
    }
}

/// This is equivalent to `fill_null(false)?.execute::<Mask>(ctx)`.
/// However, in the case of very frequend and very short operations the cost of
/// filling nulls and then canonicalizing is greater than canonicalizing and
/// then intersecting the bitbuffer.
fn null_as_false_mask(array: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Mask> {
    if !array.dtype().is_nullable() {
        return array.execute::<Mask>(ctx);
    }
    let len = array.len();
    Ok(match array.execute::<Columnar>(ctx)? {
        Columnar::Constant(c) => Mask::new(len, c.scalar().as_bool().value().unwrap_or(false)),
        Columnar::Canonical(c) => {
            let bool = c.into_array().execute::<BoolArray>(ctx)?;
            match bool.as_ref().validity()? {
                Validity::NonNullable | Validity::AllValid => {
                    Mask::from_buffer(bool.into_bit_buffer())
                }
                Validity::AllInvalid => Mask::new_false(len),
                Validity::Array(v) => {
                    let validity_bits = v.execute::<BoolArray>(ctx)?.into_bit_buffer();
                    Mask::from_buffer(bool.into_bit_buffer() & &validity_bits)
                }
            }
        }
    })
}

struct ZoneMapStatsBinder<'a> {
    zone_map: &'a ZoneMap,
}

impl StatBinder for ZoneMapStatsBinder<'_> {
    fn bind_aggregate(
        &self,
        input: &BoundExpression,
        aggregate_fn: &AggregateFnRef,
        _stat_dtype: &DType,
    ) -> VortexResult<Option<BoundExpression>> {
        if !input.is_root() {
            return Ok(None);
        }
        vortex_ensure_eq!(
            input.dtype(),
            &self.zone_map.column_dtype,
            "Stats predicate root dtype does not match zone-map column dtype"
        );

        if let Some(stat_expr) = self.zone_map.aggregate_field_expr(aggregate_fn) {
            return Ok(Some(self.bind_target(stat_expr)?));
        }

        if aggregate_fn.is::<AllNull>() {
            return self
                .zone_map
                .stat_field_expr(Stat::NullCount)
                .map(|null_count| self.bind_target(eq(null_count, row_count_expr())))
                .transpose();
        }

        if aggregate_fn.is::<AllNonNull>() {
            return self
                .zone_map
                .stat_field_expr(Stat::NullCount)
                .map(|null_count| self.bind_target(eq(null_count, lit(0u64))))
                .transpose();
        }

        if aggregate_fn.is::<AllNan>() {
            return self
                .zone_map
                .stat_field_expr(Stat::NaNCount)
                .map(|nan_count| self.bind_target(eq(nan_count, row_count_expr())))
                .transpose();
        }

        if aggregate_fn.is::<AllNonNan>() {
            return self
                .zone_map
                .stat_field_expr(Stat::NaNCount)
                .map(|nan_count| self.bind_target(eq(nan_count, lit(0u64))))
                .transpose();
        }

        if let Some(stat) = Stat::from_aggregate_fn(aggregate_fn) {
            return self
                .zone_map
                .stat_field_expr(stat)
                .map(|expr| self.bind_target(expr))
                .transpose();
        }

        Ok(None)
    }
}

impl ZoneMapStatsBinder<'_> {
    fn bind_target(&self, expr: Expression) -> VortexResult<BoundExpression> {
        expr.bind(self.zone_map.array.dtype())
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

/// Build per-zone row counts for a zone map.
///
/// `zone_len` is the nominal zone size; only the final zone may be shorter. The
/// result is a [`ConstantArray`] for uniform zone sizes, otherwise a two-run
/// run-end encoded array whose trailing run carries the final zone length.
///
/// All three arguments come from the file — `zone_len` from the zoned layout's metadata,
/// `row_count` from the layout node, `num_zones` from the zone map's own length — and nothing ties
/// them together, so a file may declare more zone coverage than the layout has rows.
fn row_count_array(zone_len: u64, row_count: u64, num_zones: usize) -> VortexResult<ArrayRef> {
    if num_zones == 0 {
        return Ok(ConstantArray::new(0u64, 0).into_array());
    }

    let leading_rows = zone_len.saturating_mul((num_zones as u64) - 1);
    let Some(last_zone_len) = row_count.checked_sub(leading_rows) else {
        vortex_bail!(
            "Zone map declares {num_zones} zones of {zone_len} rows, which is more than the \
             {row_count} rows of the layout"
        );
    };
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
    use vortex_array::expr::stats::Stat;
    use vortex_array::validity::Validity;
    use vortex_buffer::Buffer;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_expr::BoundExpression;
    use vortex_expr::Expression;
    use vortex_expr::cast;
    use vortex_expr::is_not_null;
    use vortex_expr::is_null;
    use vortex_expr::lit;
    use vortex_expr::not_eq;
    use vortex_expr::root;
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
