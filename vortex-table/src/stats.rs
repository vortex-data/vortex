// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Per-file column statistics, kept in the manifest so that a scan drops files without reading
//! their footers.
//!
//! The stats table has one row per file and one struct field per column, holding that column's
//! file-level `min`, `max`, `null_count` and `nan_count`. A filter's falsification proof is bound
//! to the table once and evaluated for every file in one pass.

use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::aggregate_fn::AggregateFnRef;
use vortex_array::aggregate_fn::fns::all_non_null::AllNonNull;
use vortex_array::aggregate_fn::fns::all_null::AllNull;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::struct_::StructArrayExt;
use vortex_array::builders::builder_with_capacity_in;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldName;
use vortex_array::dtype::FieldNames;
use vortex_array::dtype::FieldPath;
use vortex_array::scalar_fn::EmptyOptions;
use vortex_array::scalar_fn::builtin::cast::Cast;
use vortex_array::scalar_fn::builtin::get_item::GetItem;
use vortex_array::stats::Stat;
use vortex_array::validity::Validity;
use vortex_buffer::BufferAllocatorRef;
use vortex_error::VortexResult;
use vortex_expr::ArrayExprExt;
use vortex_expr::BoundExpression;
use vortex_expr::Expression;
use vortex_expr::ScalarFnExprExt;
use vortex_expr::cast;
use vortex_expr::eq;
use vortex_expr::get_item;
use vortex_expr::lambda::row_count::RowCount;
use vortex_expr::lambda::row_count::substitute_row_count;
use vortex_expr::lit;
use vortex_expr::root;
use vortex_expr::stats::bind::StatBinder;
use vortex_expr::stats::bind::bind_stats;
use vortex_expr::stats::rewrite::BoundExpressionPruneExt;
use vortex_file::FileStatistics;
use vortex_mask::Mask;
use vortex_session::VortexSession;

/// The statistics the stats table records for each column.
const STATS: [Stat; 4] = [Stat::Min, Stat::Max, Stat::NullCount, Stat::NaNCount];

/// The file-level statistics of a table's files, one row per file.
#[derive(Clone, Debug)]
pub struct TableStats {
    dtype: DType,
    array: StructArray,
}

impl TableStats {
    /// The stats table `array` of files whose rows have `dtype`.
    pub fn new(dtype: DType, array: StructArray) -> Self {
        Self { dtype, array }
    }

    /// The stats table of files whose rows have `dtype`, from each file's footer statistics.
    ///
    /// A column gets a statistic when at least one file records it.
    pub fn from_files(dtype: DType, files: &[Option<&FileStatistics>]) -> VortexResult<Self> {
        let mut names = Vec::new();
        let mut columns = Vec::new();
        if let Some(fields) = dtype.as_struct_fields_opt() {
            for (name, column_dtype) in fields.names().iter().zip(fields.fields()) {
                if let Some(column) = column_stats(name, &column_dtype, files)? {
                    names.push(name.clone());
                    columns.push(column);
                }
            }
        }
        let array = StructArray::try_new(
            FieldNames::from(names),
            columns,
            files.len(),
            Validity::NonNullable,
        )?;
        Ok(Self { dtype, array })
    }

    /// The dtype of the files' rows.
    pub fn dtype(&self) -> &DType {
        &self.dtype
    }

    /// The stats table, one row per file.
    pub fn array(&self) -> &StructArray {
        &self.array
    }

    /// Which files `filter` may match: a file is false when its statistics prove that no row
    /// matches. `row_counts` holds each file's row count.
    pub fn may_match(
        &self,
        filter: &Expression,
        row_counts: &[u64],
        session: &VortexSession,
    ) -> VortexResult<Mask> {
        let all = Mask::new_true(self.array.len());
        let Some(proof) = filter.bind(&self.dtype)?.falsify(session)? else {
            return Ok(all);
        };
        let proof = bind_stats(proof, &TableStatsBinder { stats: self })?;
        let applied = self.array.clone().into_array().apply_bound(&proof)?;
        let row_counts = PrimitiveArray::from_iter(row_counts.iter().copied()).into_array();
        let pruned = substitute_row_count(applied, &row_counts)?
            .fill_null(false)?
            .execute::<Mask>(&mut session.create_execution_ctx())?;
        Ok(all.bitand_not(&pruned))
    }
}

/// The stats of one column across `files`, or `None` when no file records any.
fn column_stats(
    name: &FieldName,
    dtype: &DType,
    files: &[Option<&FileStatistics>],
) -> VortexResult<Option<ArrayRef>> {
    let path = FieldPath::from_name(name.clone());
    let mut names = Vec::new();
    let mut values = Vec::new();
    for stat in STATS {
        let (Some(aggregate_fn), Some(stat_dtype)) = (stat.aggregate_fn(), stat.dtype(dtype))
        else {
            continue;
        };
        let stat_dtype = stat_dtype.as_nullable();
        let mut builder =
            builder_with_capacity_in(&stat_dtype, files.len(), BufferAllocatorRef::static_ref());
        let mut any = false;
        for file in files {
            let value = file
                .and_then(|stats| stats.get_by_path(&path))
                .and_then(|(aggregates, _)| aggregates.get(&aggregate_fn).into_inner())
                .and_then(|value| value.cast(&stat_dtype).ok());
            match value {
                Some(value) => {
                    builder.append_scalar(&value)?;
                    any = true;
                }
                None => builder.append_null(),
            }
        }
        if any {
            names.push(FieldName::from(stat.name()));
            values.push(builder.finish());
        }
    }
    if names.is_empty() {
        return Ok(None);
    }
    Ok(Some(
        StructArray::try_new(
            FieldNames::from(names),
            values,
            files.len(),
            Validity::NonNullable,
        )?
        .into_array(),
    ))
}

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
        let Some(column) = top_level_column(input) else {
            return Ok(None);
        };
        let stat_ref = |stat: Stat| self.stat_ref(column, stat);
        let expr = if aggregate_fn.is::<AllNull>() {
            stat_ref(Stat::NullCount).map(|count| eq(count, RowCount.new_expr(EmptyOptions, [])))
        } else if aggregate_fn.is::<AllNonNull>() {
            stat_ref(Stat::NullCount).map(|count| eq(count, lit(0u64)))
        } else {
            Stat::from_aggregate_fn(aggregate_fn)
                .and_then(stat_ref)
                .map(|value| cast(value, stat_dtype.clone()))
        };
        expr.map(|expr| expr.bind(self.stats.array.dtype()))
            .transpose()
    }
}

impl TableStatsBinder<'_> {
    fn stat_ref(&self, column: &FieldName, stat: Stat) -> Option<Expression> {
        let column_stats = self.stats.array.unmasked_field_by_name_opt(column)?;
        column_stats
            .dtype()
            .as_struct_fields_opt()?
            .field(stat.name())?;
        Some(get_item(stat.name(), get_item(column.clone(), root())))
    }
}

/// The top-level column `expr` reads, looking through casts.
fn top_level_column(expr: &BoundExpression) -> Option<&FieldName> {
    if expr.is::<Cast>() {
        return top_level_column(expr.child(0));
    }
    let name = expr.as_opt::<GetItem>()?;
    expr.child(0).is_root().then_some(name)
}
