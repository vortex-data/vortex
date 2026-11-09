//! Evaluation of per-zone statistics during scans.

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
use vortex_array::dtype::Nullability;
use vortex_array::dtype::StructFields;
use vortex_array::expr::BoundExpression;
use vortex_array::expr::Expression;
use vortex_array::expr::eq;
use vortex_array::expr::get_item;
use vortex_array::expr::lit;
use vortex_array::expr::root;
use vortex_array::stats::Stat;
use vortex_array::scalar_fn::EmptyOptions;
use vortex_array::scalar_fn::ScalarFnVTableExt;
use vortex_array::scalar_fn::internal::row_count::RowCount;
use vortex_array::scalar_fn::internal::row_count::contains_row_count;
use vortex_array::scalar_fn::internal::row_count::substitute_row_count;
use vortex_array::validity::Validity;
use vortex_buffer::buffer;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure_eq;
use vortex_expr::stats::bind::StatBinder;
use vortex_expr::stats::bind::bind_stats;
use vortex_mask::Mask;
use vortex_session::VortexSession;

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

    /// Constructs a zone map with an already validated statistics table.
    ///
    /// # Safety
    /// The table fields must hold valid statistics for `column_dtype` and `aggregate_fns`.
    unsafe fn new_unchecked(
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

    /// Apply a pruning predicate to this zone map.
    ///
    /// `predicate` should be a stats rewrite expression such as the result of
    /// [`BoundExpressionPruneExt::falsify`]. The returned mask has one value per zone, where
    /// `true` means the zone cannot contain matching rows and can be skipped.
    ///
    /// If the predicate contains [`row_count`][vortex_array::scalar_fn::internal::row_count]
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
    pub fn prune_lowered(
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
    pub fn lower_stats(&self, predicate: BoundExpression) -> VortexResult<BoundExpression> {
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
        stat.aggregate_fn()
            .and_then(|aggregate_fn| self.aggregate_field_expr(&aggregate_fn))
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

/// Return the auxiliary stats-table schema for a zoned layout.
fn aggregate_stats_table_dtype(column_dtype: &DType, aggregate_fns: &[AggregateFnRef]) -> DType {
    DType::Struct(
        StructFields::from_iter(aggregate_fns.iter().filter_map(|aggregate_fn| {
            aggregate_state_dtype(column_dtype, aggregate_fn)
                .map(|dtype| (aggregate_fn.to_string(), dtype.as_nullable()))
        })),
        Nullability::NonNullable,
    )
}

/// Returns an aggregate state dtype, resolving extension storage types when needed.
fn aggregate_state_dtype(column_dtype: &DType, aggregate_fn: &AggregateFnRef) -> Option<DType> {
    aggregate_fn.state_dtype(column_dtype).or_else(|| {
        if let DType::Extension(ext) = column_dtype {
            aggregate_fn.state_dtype(ext.storage_dtype())
        } else {
            None
        }
    })
}

#[cfg(test)]
mod tests;
