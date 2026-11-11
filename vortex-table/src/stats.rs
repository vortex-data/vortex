// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Column statistics for every file of a table, as one columnar table with a row per file.
//!
//! Each file's footer carries file-level aggregates (min, max, null count, ...) for its columns.
//! [`TableStats`] lays them out the way a zone map lays out per-zone aggregates:
//!
//! ```text
//! struct {
//!   <column>: struct { <aggregate>: <value>?, ... },
//!   ...
//! }
//! ```
//!
//! with one row per manifest entry and a null wherever a file lacks an aggregate. A filter's
//! falsification proof is then bound to this table and evaluated once, giving a mask over every
//! file of the table, instead of being evaluated against each footer in turn.

use vortex_array::ArrayRef;
use vortex_array::Columnar;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::aggregate_fn::AggregateFnRef;
use vortex_array::aggregate_fn::AggregateFnSatisfaction;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::builders::ArrayBuilder;
use vortex_array::builders::builder_with_capacity_in;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldName;
use vortex_array::dtype::FieldPath;
use vortex_array::expr::BoundExpression;
use vortex_array::expr::Expression;
use vortex_array::expr::cast;
use vortex_array::expr::get_item;
use vortex_array::expr::root;
use vortex_array::scalar::Scalar;
use vortex_array::scalar_fn::fns::cast::Cast;
use vortex_array::scalar_fn::fns::get_item::GetItem;
use vortex_array::scalar_fn::internal::row_count::contains_row_count;
use vortex_array::scalar_fn::internal::row_count::substitute_row_count;
use vortex_array::validity::Validity;
use vortex_buffer::BufferAllocatorRef;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure_eq;
use vortex_expr::stats::bind::StatBinder;
use vortex_expr::stats::bind::bind_stats;
use vortex_expr::stats::rewrite::BoundExpressionPruneExt;
use vortex_file::Footer;
use vortex_mask::Mask;
use vortex_session::VortexSession;

/// The aggregates recorded for one top-level column, across the files of a table.
struct ColumnStats {
    name: FieldName,
    /// The aggregates some file records for the column, in the order of the column's fields.
    aggregates: Vec<AggregateFnRef>,
}

/// File-level column statistics of every file of a table, one row per file.
pub struct TableStats {
    /// The dtype of the table's files.
    dtype: DType,
    columns: Vec<ColumnStats>,
    /// `struct { <column>: struct { <aggregate>: value? } }`, one row per file.
    table: ArrayRef,
    /// Rows in each file, for proofs that compare a count against the file's row count.
    row_counts: ArrayRef,
    session: VortexSession,
}

impl TableStats {
    /// The statistics of `footers`, the files of a table in table order, all of `dtype`.
    pub fn try_new<'a>(
        footers: impl IntoIterator<Item = &'a Footer>,
        dtype: &DType,
        session: &VortexSession,
    ) -> VortexResult<Self> {
        let footers: Vec<&Footer> = footers.into_iter().collect();
        for footer in &footers {
            vortex_ensure_eq!(
                footer.dtype(),
                dtype,
                "a file's dtype differs from the table's"
            );
        }
        let fields = dtype.as_struct_fields_opt();
        let allocator = BufferAllocatorRef::statically_allocated();
        let mut columns = Vec::new();
        let mut column_arrays = Vec::new();
        for (name, _) in fields
            .iter()
            .flat_map(|fields| fields.names().iter().zip(fields.fields()))
        {
            let path = FieldPath::from_name(name.clone());
            let per_file: Vec<_> = footers
                .iter()
                .map(|footer| {
                    footer
                        .statistics()
                        .and_then(|stats| stats.get_by_path(&path))
                        .map(|(aggregates, _)| aggregates)
                })
                .collect();
            // Every aggregate any file records, with the dtype of its first value.
            let mut aggregates: Vec<(AggregateFnRef, DType)> = Vec::new();
            for stats in per_file.iter().flatten() {
                for stat in stats.iter() {
                    let Some(value) = stat.value().clone().into_inner() else {
                        continue;
                    };
                    if !aggregates.iter().any(|(f, _)| f == stat.aggregate_fn()) {
                        aggregates.push((stat.aggregate_fn().clone(), value.dtype().as_nullable()));
                    }
                }
            }
            if aggregates.is_empty() {
                continue;
            }
            let mut names = Vec::with_capacity(aggregates.len());
            let mut values = Vec::with_capacity(aggregates.len());
            for (aggregate_fn, value_dtype) in &aggregates {
                let mut builder = builder_with_capacity_in(value_dtype, footers.len(), &allocator);
                for stats in &per_file {
                    let value = stats.and_then(|stats| {
                        stats
                            .iter()
                            .find(|stat| stat.aggregate_fn() == aggregate_fn)
                            .and_then(|stat| stat.value().clone().into_inner())
                    });
                    append(builder.as_mut(), value, value_dtype)?;
                }
                names.push(FieldName::from(aggregate_fn.to_string()));
                values.push(builder.finish());
            }
            let column =
                StructArray::try_new(names.into(), values, footers.len(), Validity::NonNullable)?;
            columns.push(ColumnStats {
                name: name.clone(),
                aggregates: aggregates.into_iter().map(|(f, _)| f).collect(),
            });
            column_arrays.push(column.into_array());
        }
        let table = StructArray::try_new(
            columns
                .iter()
                .map(|c| c.name.clone())
                .collect::<Vec<_>>()
                .into(),
            column_arrays,
            footers.len(),
            Validity::NonNullable,
        )?
        .into_array();
        let row_counts =
            PrimitiveArray::from_iter(footers.iter().map(|footer| footer.row_count())).into_array();
        Ok(Self {
            dtype: dtype.clone(),
            columns,
            table,
            row_counts,
            session: session.clone(),
        })
    }

    /// Files in the table.
    pub fn len(&self) -> usize {
        self.table.len()
    }

    /// Whether the table has no files.
    pub fn is_empty(&self) -> bool {
        self.table.is_empty()
    }

    /// The table of statistics, one row per file.
    pub fn table(&self) -> &ArrayRef {
        &self.table
    }

    /// One value per file: `true` where the statistics prove `filter` matches no row of the file.
    pub fn prune(&self, filter: &Expression) -> VortexResult<Mask> {
        let len = self.len();
        let Some(proof) = filter.bind(&self.dtype)?.falsify(&self.session)? else {
            return Ok(Mask::new_false(len));
        };
        let proof = bind_stats(proof, &TableStatsBinder { stats: self })?;
        let mut ctx = self.session.create_execution_ctx();
        let mut applied = self.table.clone().apply_bound(&proof)?;
        if contains_row_count(&applied) {
            applied = substitute_row_count(applied, &self.row_counts)?;
        }
        Ok(match applied.execute::<Columnar>(&mut ctx)? {
            Columnar::Constant(c) => Mask::new(len, c.scalar().as_bool().value().unwrap_or(false)),
            Columnar::Canonical(c) => {
                // A null proof proves nothing, so it keeps the file.
                let pruned = c.into_array().execute::<BoolArray>(&mut ctx)?;
                match pruned.as_ref().validity()? {
                    Validity::NonNullable | Validity::AllValid => {
                        Mask::from_buffer(pruned.into_bit_buffer())
                    }
                    Validity::AllInvalid => Mask::new_false(len),
                    Validity::Array(valid) => {
                        let valid = valid.execute::<BoolArray>(&mut ctx)?.into_bit_buffer();
                        Mask::from_buffer(pruned.into_bit_buffer() & &valid)
                    }
                }
            }
        })
    }
}

fn append(
    builder: &mut dyn ArrayBuilder,
    value: Option<Scalar>,
    dtype: &DType,
) -> VortexResult<()> {
    match value {
        Some(value) if value.dtype() == dtype => builder.append_scalar(&value),
        Some(value) => match value.cast(dtype) {
            Ok(value) => builder.append_scalar(&value),
            Err(_) => {
                builder.append_null();
                Ok(())
            }
        },
        None => {
            builder.append_null();
            Ok(())
        }
    }
}

/// Binds each aggregate a proof asks for to its column of the stats table.
struct TableStatsBinder<'a> {
    stats: &'a TableStats,
}

impl StatBinder for TableStatsBinder<'_> {
    fn bind_aggregate(
        &self,
        input: &BoundExpression,
        aggregate_fn: &AggregateFnRef,
        stat_dtype: &DType,
    ) -> VortexResult<Option<BoundExpression>> {
        let Some(column) = top_level_field(input) else {
            return Ok(None);
        };
        let Some(stats) = self.stats.columns.iter().find(|c| c.name == *column) else {
            return Ok(None);
        };
        // The stored aggregate that best satisfies the requested one, as a footer resolves it.
        let mut best = None;
        for stored in &stats.aggregates {
            match stored.can_satisfy(aggregate_fn) {
                AggregateFnSatisfaction::Exact => {
                    best = Some(stored);
                    break;
                }
                AggregateFnSatisfaction::Approximate => best = Some(stored),
                AggregateFnSatisfaction::No => {}
            }
        }
        let Some(stored) = best else {
            return Ok(None);
        };
        let value = get_item(stored.to_string(), get_item(column.clone(), root()));
        let bound = value.bind(self.stats.table.dtype())?;
        let value = if bound.dtype() == stat_dtype {
            value
        } else {
            cast(value, stat_dtype.clone())
        };
        Ok(Some(value.bind(self.stats.table.dtype())?))
    }
}

/// The top-level column `expr` reads, through casts, if it reads exactly one.
fn top_level_field(expr: &BoundExpression) -> Option<&FieldName> {
    if expr.is::<Cast>() {
        return top_level_field(expr.child(0));
    }
    let name = expr.as_opt::<GetItem>()?;
    expr.child(0).is_root().then_some(name)
}

#[cfg(test)]
mod tests;
