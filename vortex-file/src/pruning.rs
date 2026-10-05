// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::aggregate_fn::AggregateFnRef;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::NullArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldPath;
use vortex_array::dtype::StructFields;
use vortex_array::expr::BoundExpression;
use vortex_array::expr::bound::lit;
use vortex_array::scalar::Scalar;
use vortex_array::scalar_fn::fns::get_item::GetItem;
use vortex_array::scalar_fn::fns::literal::Literal;
use vortex_array::scalar_fn::internal::row_count::substitute_row_count;
use vortex_array::stats::bind::StatBinder;
use vortex_array::stats::bind::bind_stats;
use vortex_error::VortexResult;
use vortex_session::VortexSession;

use crate::FileStatistics;

pub(crate) fn can_prune_file_stats(
    expr: &BoundExpression,
    row_count: u64,
    file_stats: &FileStatistics,
    struct_fields: &StructFields,
    session: &VortexSession,
) -> VortexResult<bool> {
    let Some(pruning_expr) = expr.falsify(session)? else {
        return Ok(false);
    };

    let binder = FileStatsBinder {
        file_stats,
        struct_fields,
    };
    let pruning_expr = bind_stats(pruning_expr, &binder)?;

    if let Some(result) = pruning_expr.as_opt::<Literal>() {
        return Ok(result.as_bool().value() == Some(true));
    }

    let pruning = NullArray::new(1).into_array().apply_bound(&pruning_expr)?;
    let row_count_replacement = ConstantArray::new(row_count, pruning.len()).into_array();
    let pruning = substitute_row_count(pruning, &row_count_replacement)?;

    let mut ctx = session.create_execution_ctx();
    let result = pruning
        .execute::<Canonical>(&mut ctx)?
        .into_bool()
        .into_array()
        .execute_scalar(0, &mut ctx)?;

    Ok(result.as_bool().value() == Some(true))
}

struct FileStatsBinder<'a> {
    file_stats: &'a FileStatistics,
    struct_fields: &'a StructFields,
}

impl StatBinder for FileStatsBinder<'_> {
    fn bind_aggregate(
        &self,
        input: &BoundExpression,
        aggregate_fn: &AggregateFnRef,
        stat_dtype: &DType,
    ) -> VortexResult<Option<BoundExpression>> {
        let Some(field_path) = direct_field_path(input) else {
            return Ok(None);
        };
        self.stat_ref(&field_path, aggregate_fn, stat_dtype)
    }
}

impl FileStatsBinder<'_> {
    fn stat_ref(
        &self,
        field_path: &FieldPath,
        aggregate: &AggregateFnRef,
        stat_dtype: &DType,
    ) -> VortexResult<Option<BoundExpression>> {
        // FileStats currently only holds top-level field statistics.
        if field_path.parts().len() != 1 {
            return Ok(None);
        }

        let Some(field_idx) = field_path.parts()[0]
            .as_name()
            .and_then(|name| self.struct_fields.find(name))
        else {
            return Ok(None);
        };
        let Some(field_stats) = self.file_stats.results().get(field_idx) else {
            return Ok(None);
        };
        let input_dtype = &self.file_stats.dtypes()[field_idx];
        let Some(result) = field_stats.get_result(aggregate).as_exact() else {
            return Ok(None);
        };
        if result.is_null() {
            return Ok(None);
        }
        let Some(return_dtype) = aggregate.return_dtype(input_dtype) else {
            return Ok(None);
        };
        if result.dtype().as_nullable() != return_dtype.as_nullable() {
            return Ok(None);
        }
        // Historical footer extrema use the input's nullability. Normalize the lookup value only.
        let result = if result.dtype() == &return_dtype {
            result
        } else {
            Scalar::try_new(return_dtype, result.into_value())?
        };
        let Some(partial) = aggregate.partial_from_result(input_dtype, &result)? else {
            return Ok(None);
        };
        if partial.is_null() || partial.dtype().as_nullable() != stat_dtype.as_nullable() {
            return Ok(None);
        }
        Ok(Some(lit(partial)))
    }
}

fn direct_field_path(expr: &BoundExpression) -> Option<FieldPath> {
    if expr.is_root() {
        return Some(FieldPath::root());
    }

    let field_name = expr.as_opt::<GetItem>()?;
    direct_field_path(expr.child(0)).map(|path| path.push(field_name.clone()))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use vortex_array::aggregate_fn::AggregateFnVTableExt;
    use vortex_array::aggregate_fn::NumericalAggregateOpts;
    use vortex_array::aggregate_fn::fns::count::Count;
    use vortex_array::aggregate_fn::fns::max::Max;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::expr::cast;
    use vortex_array::expr::col;
    use vortex_array::expr::gt;
    use vortex_array::expr::lit;
    use vortex_array::expr::stats::Precision;
    use vortex_array::stats::AggregateResults;

    use super::*;

    #[test]
    fn computed_inputs_do_not_bind_source_field_results() -> VortexResult<()> {
        let dtype = DType::from(PType::I32);
        let fields = StructFields::from_iter([("col", dtype.clone())]);
        let scope = DType::Struct(fields.clone(), Nullability::NonNullable);
        let aggregate = Count.bind(NumericalAggregateOpts::include_nans());
        let results =
            AggregateResults::try_new(&dtype, [(aggregate.clone(), Precision::exact(3u64))])?;
        let file_stats = FileStatistics::new(Arc::new([results]), Arc::new([dtype]));
        let binder = FileStatsBinder {
            file_stats: &file_stats,
            struct_fields: &fields,
        };
        let field = col("col").bind(&scope)?;
        let computed = cast(col("col"), DType::from(PType::I64)).bind(&scope)?;
        let stat_dtype = DType::Primitive(PType::U64, Nullability::Nullable);
        assert!(
            binder
                .bind_aggregate(&field, &aggregate, &stat_dtype)?
                .is_some()
        );
        assert!(
            binder
                .bind_aggregate(&computed, &aggregate, &stat_dtype)?
                .is_none()
        );
        Ok(())
    }

    #[test]
    fn builtin_safe_cast_keeps_file_extrema() -> VortexResult<()> {
        let dtype = DType::from(PType::I32);
        let fields = StructFields::from_iter([("col", dtype.clone())]);
        let scope = DType::Struct(fields.clone(), Nullability::NonNullable);
        let aggregate = Max.bind(NumericalAggregateOpts::skip_nans());
        let results = AggregateResults::try_new(
            &dtype,
            [(
                aggregate,
                Precision::exact(Scalar::primitive(3i32, Nullability::Nullable)),
            )],
        )?;
        let file_stats = FileStatistics::new(Arc::new([results]), Arc::new([dtype]));
        let predicate = gt(cast(col("col"), DType::from(PType::I64)), lit(10i64)).bind(&scope)?;
        assert!(can_prune_file_stats(
            &predicate,
            3,
            &file_stats,
            &fields,
            &vortex_array::array_session()
        )?);
        Ok(())
    }
}
